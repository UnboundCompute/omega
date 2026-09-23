"""The resident process — spec `agent/M1_SPEC.md` §Q12, §1.6; DL-016.

M1's build order lists the *parts* of the loop. This is the thing that runs
them: one process that opens the store, recovers whatever the last stop left in
flight, drains the queue on its own thread, and listens on the socket. Until it
existed every module in the milestone was correct and nothing ran.

**One process, two threads** (§Q12). Two processes would mean two openers of the
log and M0 makes that a hard failure by design — the singleton rule is physical,
not advisory. So the executor drain and the channel listener are threads inside
one process, and the socket sits between that process and its *clients*.

**`say` appends and then observes; it never runs a turn** (§1.6). That is the
whole discipline of this file. The only way to cause a turn is to append an
episode, so :meth:`Runtime.say` appends one and then watches the projection for
the terminal record the *drain thread* wrote. It would be one line shorter to
call ``run_turn`` here and return its result, and that one line is what would
make M5's clock a second engine wearing omega's name.

**What `say` returns is what the tray sees.** It reads
:func:`omega.projection.updates_since` — the same filter, the same cursor, the
same reader the socket pump uses. A CLI with its own private view of the log
would be a second projection, and the two would drift exactly as DL-006
describes; the first symptom would be a bug that reproduces on one surface only.

**Stopping is between turns, never inside one** (DL-016's restart test). The
drain checks for a stop between episodes and finishes the episode it is holding,
so a clean stop always leaves ``claimed == done`` and the next open reports a
clean startup. A stop that abandoned a claimed turn would manufacture the exact
shape the restart report exists to describe — and would do it on the path a
person takes every time they press Ctrl-C.

**The drain thread cannot die quietly.** Anything that escapes a turn — and
``run_turn`` already records every ``Exception`` as a failed turn, so what
escapes is a bug in the loop itself — is captured and re-raised at the next
:meth:`Runtime.say` or :meth:`Runtime.stop`. A background thread that ends in a
traceback nobody reads would leave the queue silently un-drained, which looks
from the outside exactly like an assistant that has stopped caring.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from omega import episodes, projection, provider
from omega.channel import DEFAULT_HOST, DEFAULT_PORT, Channel
from omega.executor import Executor, StartupReport
from omega.memory import MemoryStore, PathLike
from omega.queue import EventQueue
from omega.turn import ActResult, TurnContext, no_act_loop_yet

__all__ = [
    "DEFAULT_POLL",
    "DEFAULT_TURN_TIMEOUT",
    "DEFAULT_STOP_TIMEOUT",
    "NotRunning",
    "DrainFailed",
    "DrainStuck",
    "TurnTimeout",
    "Said",
    "Runtime",
]

#: How long the drain waits to be woken before looking anyway. It is a safety
#: net, not the mechanism: every append through the channel wakes the drain
#: immediately (``on_append``), and this is what catches an append made by some
#: other producer against the same queue. Small enough that such an event is not
#: visibly late, large enough that an idle omega is not a spinning CPU.
DEFAULT_POLL = 0.05

#: How long :meth:`Runtime.say` waits for a turn's terminal record. Generous,
#: because a turn is two model calls and M1 has no streaming: a caller that gave
#: up at five seconds would report a failure the log will contradict a moment
#: later. It is a *timeout*, not a cancellation — the turn keeps running and
#: still lands in the log, which is why giving up early is safe but misleading.
DEFAULT_TURN_TIMEOUT = 120.0

#: How long :meth:`Runtime.stop` waits for the in-flight turn to finish. It has
#: to exceed the slowest single turn or a clean stop would start abandoning
#: claimed turns, which is the one thing it exists to prevent.
DEFAULT_STOP_TIMEOUT = 150.0


class NotRunning(RuntimeError):
    """The runtime was used before ``start`` or after ``stop``.

    Refused rather than started implicitly. Opening the store is what takes M0's
    exclusive lock, so a lazily-started runtime would take that lock at an
    unpredictable moment and the singleton rule would stop being observable from
    the outside.
    """


class DrainFailed(RuntimeError):
    """Something escaped the drain thread and the loop is no longer running.

    ``__cause__`` carries the original. This is deliberately not survivable:
    ``run_turn`` already turns every ordinary failure — an outage, an unparseable
    verdict, a broken tool — into a recorded ``failed`` turn, so anything that
    gets this far is a defect in the loop, and continuing to drain past it would
    be guessing about an invariant we just watched break.
    """


class DrainStuck(RuntimeError):
    """The in-flight turn did not finish inside the stop timeout.

    The store is deliberately left **open** and the thread left alone. Closing
    the log under a thread that is still writing to it is how a clean shutdown
    becomes a damaged tail, and the turn may still complete and file its record.
    Calling :meth:`Runtime.stop` again waits again.
    """


class TurnTimeout(RuntimeError):
    """No terminal record for that turn arrived in time.

    The turn is **not** cancelled and nothing is rolled back — it is still
    claimed, still running, and will still write its record. This says only that
    the caller stopped watching, which is why it names the seq: the answer can be
    read out of the log afterwards.
    """

    def __init__(self, seq: int, waited: float) -> None:
        super().__init__(
            f"no terminal record for turn {seq} after {waited:.0f}s; the turn is "
            f"still running and will still be logged"
        )
        self.seq = seq


@dataclass(frozen=True)
class Said:
    """How one turn ended, as its *caller* sees it.

    Built from a :class:`omega.projection.Update` rather than from the episode,
    so the CLI is served by the same filter as the tray — see the module
    docstring. It carries both the wire ``state`` and the episode ``outcome``
    because they answer different questions: ``state`` is what a UI renders,
    ``outcome`` is what DL-011's metrics count.

    ``silent`` and ``failed`` are separate properties and neither is defined as
    the absence of the other. A silent turn is a **success** with no reply; a
    failed turn is a failure with an error. Collapsing them is the regression
    DL-011 calls the single most load-bearing consequence of the time-wake.
    """

    seq: int
    record_seq: int
    state: str
    outcome: Optional[str] = None
    reply: Optional[str] = None
    needs: Optional[str] = None
    error: Optional[str] = None

    @property
    def spoke(self) -> bool:
        return self.outcome == "spoke"

    @property
    def silent(self) -> bool:
        """A success. Never render this as a degraded ``spoke`` (DL-011)."""
        return self.outcome == "silent"

    @property
    def failed(self) -> bool:
        return self.state == projection.FAILED

    @property
    def blocked(self) -> bool:
        return self.state == projection.BLOCKED

    @classmethod
    def from_update(cls, for_seq: int, update: projection.Update) -> "Said":
        return cls(
            seq=for_seq,
            record_seq=update.seq,
            state=update.state,
            outcome=update.outcome,
            reply=update.reply,
            needs=update.needs,
            error=update.error,
        )


class Runtime:
    """The resident process, composed of the parts M1 already built.

    Owns exactly three things: the open store, the drain thread, and the
    listener. It adds no state of its own that matters — DL-016 fixes zero
    authoritative state in RAM, and everything here is either a thread, a cursor
    the log already holds, or a flag about this process's own lifetime.
    """

    def __init__(
        self,
        store_path: PathLike,
        *,
        complete: Optional[Callable[..., provider.Response]] = None,
        act: Callable[[TurnContext], ActResult] = no_act_loop_yet,
        listen: bool = True,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        poll: float = DEFAULT_POLL,
        turn_timeout: float = DEFAULT_TURN_TIMEOUT,
    ) -> None:
        if poll <= 0:
            raise ValueError(f"poll must be positive, got {poll}")
        self._store_path = Path(os.fspath(store_path))
        self._complete = complete
        self._act = act
        self._listen = listen
        self._host = host
        self._port = port
        self._poll = poll
        self._turn_timeout = turn_timeout

        self._store: Optional[MemoryStore] = None
        self._queue: Optional[EventQueue] = None
        self._executor: Optional[Executor] = None
        self._channel: Optional[Channel] = None
        self._thread: Optional[threading.Thread] = None
        self._report: Optional[StartupReport] = None

        self._stopping = threading.Event()
        self._woken = threading.Event()
        self._drain_error: Optional[BaseException] = None
        self._turns = 0
        self._started = False

    # --- lifecycle --------------------------------------------------------

    def start(self) -> StartupReport:
        """Open the store, recover, and begin draining. Returns the report.

        The order is the interesting part. ``recover`` runs **before** the drain
        thread exists, because draining while ``claimed > done`` would leave the
        interrupted turn with no record at all — the one shape the restart test
        cannot tell from a crash. ``Executor`` refuses it anyway (``NotRecovered``),
        so this ordering is belt and braces on the milestone's sharpest edge.

        The listener starts **last**, so nothing can arrive before there is a
        drain to take it.
        """
        if self._started:
            raise NotRunning("this runtime has already been started")
        self._started = True
        self._store = MemoryStore.open(self._store_path)
        try:
            self._queue = EventQueue(self._store)
            self._executor = Executor(
                self._queue, complete=self._complete, act=self._act
            )
            self._report = self._executor.recover()
            self._thread = threading.Thread(
                target=self._drain_loop, name="omega-executor", daemon=True
            )
            self._thread.start()
            if self._listen:
                self._channel = Channel(
                    self._queue,
                    host=self._host,
                    port=self._port,
                    poll=self._poll,
                    # The listener does not deliver events; it *appends* them and
                    # nudges. One notification path, and it is the same log
                    # everything else reads (§1.6).
                    on_append=self._on_append,
                )
                self._channel.start()
        except BaseException:
            # A half-started runtime still holds M0's exclusive lock, and the
            # next attempt would fail with AlreadyLocked naming nothing useful.
            self._stopping.set()
            self._unwind()
            raise
        return self._report

    def stop(self, *, timeout: float = DEFAULT_STOP_TIMEOUT) -> None:
        """Stop between turns, release everything, and surface what went wrong.

        Idempotent. The listener stops first so nothing new arrives, then the
        drain is given until ``timeout`` to finish the episode it is holding —
        **not** interrupted partway. That is DL-016's restart test taken
        seriously on the path a person actually uses: Ctrl-C is not a crash, so
        it must not leave a claimed turn without a record.

        Raises :class:`DrainStuck` without closing the store if the turn is
        still running when the timeout expires, and :class:`DrainFailed` if the
        drain thread ended in an exception — after the cleanup, so a failure to
        report cannot also be a failure to release.
        """
        self._stopping.set()
        self._woken.set()

        if self._channel is not None:
            self._channel.stop()
            self._channel = None

        thread, self._thread = self._thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)
            if thread.is_alive():
                self._thread = thread  # a later stop() waits again
                raise DrainStuck(
                    f"the turn in flight did not finish within {timeout:.0f}s; "
                    f"the store is still open and the turn may yet record itself"
                )

        self._unwind()

        if self._drain_error is not None:
            raise DrainFailed(
                f"the executor drain stopped on {type(self._drain_error).__name__}: "
                f"{self._drain_error}"
            ) from self._drain_error

    def _unwind(self) -> None:
        """Release the store. Never raises over an already-closed store, because
        it runs on the failure path of ``start`` as well as on ``stop``."""
        if self._store is not None:
            self._store.close()
            self._store = None
        self._queue = None
        self._executor = None

    def __enter__(self) -> "Runtime":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    # --- what it is made of -----------------------------------------------

    @property
    def report(self) -> StartupReport:
        """What the last stop left behind. Available after ``start``."""
        if self._report is None:
            raise NotRunning("the runtime has not been started")
        return self._report

    @property
    def queue(self) -> EventQueue:
        if self._queue is None:
            raise NotRunning("the runtime is not running")
        return self._queue

    @property
    def store_path(self) -> Path:
        return self._store_path

    @property
    def address(self) -> Optional[tuple[str, int]]:
        """Where the listener is bound, or ``None`` when it is disabled."""
        return None if self._channel is None else self._channel.address

    @property
    def turns(self) -> int:
        """Turns this process has run. Written only by the drain thread."""
        return self._turns

    @property
    def drain_error(self) -> Optional[BaseException]:
        """What killed the drain, or ``None`` while it is alive."""
        return self._drain_error

    @property
    def running(self) -> bool:
        return self._queue is not None and not self._stopping.is_set()

    # --- the drain thread -------------------------------------------------

    def wake(self) -> None:
        """Tell the drain there is something to look at.

        A hint, never a delivery: it carries no episode and the drain re-reads
        the log either way. That is what keeps a wake from a person, a wake from
        the socket and a wake from a future clock the *same* event as far as the
        loop is concerned (§1.6) — they differ only in what they appended.
        """
        self._woken.set()

    def _on_append(self, seq: int) -> None:
        self.wake()

    def _drain_loop(self) -> None:
        """Wake, drain what is waiting, sleep. Nothing escapes it unnoticed."""
        try:
            while not self._stopping.is_set():
                self._woken.wait(self._poll)
                # Clear *before* reading the log: an append that lands between
                # the clear and the read is seen by the read, and one that lands
                # after it re-sets the flag, so the next wait returns at once.
                # Clearing afterwards would drop exactly the wake in between.
                self._woken.clear()
                self._drain_pending()
        except BaseException as exc:  # noqa: BLE001 - see the module docstring
            self._drain_error = exc

    def _drain_pending(self) -> None:
        """One episode at a time, checking for a stop between each.

        ``Executor.drain`` would be one call, but it yields control only when the
        queue runs dry — a stop would then wait for an idle moment that a busy
        omega may not have. Stepping gives the same order and the same claim,
        with a stop point between every episode and none inside one.

        ``claimed < head`` is exactly "``pending()`` is not empty", said without
        constructing the generator.
        """
        queue, executor = self._queue, self._executor
        if queue is None or executor is None:  # pragma: no cover - stopped mid-flight
            return
        while not self._stopping.is_set() and queue.claimed() < queue.head():
            if executor.step() is not None:
                self._turns += 1

    # --- talking to it ----------------------------------------------------

    def say(
        self,
        text: str,
        *,
        channel: str = "cli",
        timeout: Optional[float] = None,
    ) -> Said:
        """Append one inbound episode and wait for the turn it causes.

        Two steps, and they are deliberately not one: the append is the
        *acknowledgement* and is durable when it returns (§1.1, tray requirement
        1), and the wait is an observation of what the drain thread did with it.
        Nothing here runs a turn (§1.6).
        """
        seq = self.append(episodes.inbound(text, channel=channel))
        return self.wait_for(seq, timeout=timeout)

    def append(self, payload: dict[str, Any], write_key: str = "") -> int:
        """Put an episode in the log and wake the drain. The only entry point.

        Public because M5's clock and any other producer must use *this* and not
        a private path — the moment there are two ways in, they drift.
        """
        if not self.running:
            raise NotRunning("the runtime is not running")
        self._raise_if_drain_died()
        seq = self.queue.append(payload, write_key)
        self.wake()
        return seq

    def wait_for(self, seq: int, *, timeout: Optional[float] = None) -> Said:
        """Watch the projection until the turn for ``seq`` records how it ended.

        Matching is on ``for_seq``, so a caller waiting on its own event is
        unaffected by the turns the drain runs before it — which is the ordinary
        case, not the exotic one: the queue is serial and single-consumer, so
        anything already waiting is handled first.

        Reads through :func:`omega.projection.updates_since`, the same filter the
        socket pump serves the tray from. A second reader over the same log would
        be a second projection to keep honest.
        """
        deadline = time.monotonic() + (
            self._turn_timeout if timeout is None else timeout
        )
        started = time.monotonic()
        cursor = seq
        while True:
            self._raise_if_drain_died()
            if not self.running:
                raise NotRunning("the runtime stopped before the turn was recorded")
            head = self.queue.head()
            if head > cursor:
                for update in projection.updates_since(self.queue, cursor):
                    if (
                        update.for_seq == seq
                        and update.kind in episodes.TERMINAL_KINDS
                    ):
                        return Said.from_update(seq, update)
                # Advance to the head *this pass* read, not to the last update
                # seen: an episode the filter withheld would otherwise be
                # re-read forever, and one appended mid-scan would be skipped.
                cursor = head
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TurnTimeout(seq, time.monotonic() - started)
            time.sleep(min(self._poll, remaining))

    def _raise_if_drain_died(self) -> None:
        if self._drain_error is not None:
            raise DrainFailed(
                f"the executor drain stopped on "
                f"{type(self._drain_error).__name__}: {self._drain_error}"
            ) from self._drain_error

    def __repr__(self) -> str:
        where = self._store_path
        if self._queue is None:
            return f"<Runtime {str(where)!r} stopped>"
        return f"<Runtime {str(where)!r} {self._queue!r} turns={self._turns}>"
