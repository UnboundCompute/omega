"""The single consumer — M1 step 4, spec `agent/M1_SPEC.md` §1.1, §1.2, §1.6.

One drain, serial, over one durable queue. It does three things:

**It advances past records it did not ask for.** ``turn.completed`` and
``tool.*`` land in the same log as the events, so they turn up in ``pending()``
too. Without the skip the executor answers its own output forever — §1.1 calls
this the one genuinely sharp edge in reusing the log as the queue.

**It claims before it acts.** The cursor moves *before* the turn runs, so the
loop is at-most-once. A crash mid-turn is recovered as *information* rather than
as a replay, because for a step that calls tools with side effects, re-delivery
is the standard way one action becomes two.

**It reports the interrupted turn instead of re-running it** (§1.2).
``claimed > done`` is the whole check — no scan, no scan window to get wrong.
The decision to pick the work back up is a judgement surfaced to the user, never
an automatic re-run.

**One entry point** (§1.6): nothing here exposes a way to cause a turn other
than appending an episode. Only one wake exists at M1, but the queue has to
already be the single entry point so M5's clock is a *producer* against the same
append and never a second path into the loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from omega import episodes, provider
from omega.memory import WriteKeyConflict
from omega.queue import EVENT_KINDS, EventQueue, Pending
from omega.turn import ActResult, TurnContext, TurnResult, no_act_loop_yet, run_turn

__all__ = [
    "INTERRUPTED_ERROR",
    "NotRecovered",
    "Interrupted",
    "StartupReport",
    "Executor",
]

#: What goes in the ``error`` field of the record that closes an interrupted
#: turn. A sentence rather than a code, because it is read by a human in a raw
#: log during exactly the phase this milestone exists to support — and it is
#: distinctive enough that it can never be confused with a real turn's failure.
INTERRUPTED_ERROR = "interrupted by a restart before the turn finished"


class NotRecovered(RuntimeError):
    """``drain`` was called before ``recover``.

    Fail-closed rather than convenient. Draining while ``claimed > done``
    would leave the interrupted turn with no record at all — the one shape the
    restart test cannot tell from a crash — and the report the user is owed
    would never be produced.
    """


@dataclass(frozen=True)
class Interrupted:
    """The turn that was in flight when the process died (§1.2).

    ``finished_before_crash`` is the honest half of the report. ``turn.completed``
    is appended *before* ``DONE`` advances, so crashing between them leaves
    ``claimed > done`` with a completed record already present. That is the safe
    direction of the error — it over-reports a cut-off turn rather than silently
    dropping one — and the field says which of the two actually happened instead
    of letting the report imply the worse one.
    """

    seq: int
    payload: dict[str, Any]
    record_seq: int
    finished_before_crash: bool

    @property
    def text(self) -> str:
        """What the user actually asked, for the question omega asks back."""
        if self.payload.get("kind") == episodes.MESSAGE_INBOUND:
            return str(self.payload.get("text", ""))
        return str(self.payload.get("summary", ""))

    @property
    def question(self) -> str:
        """§1.2 — omega *says so*. The decision to resume is the user's."""
        if self.finished_before_crash:
            return (
                f'"{self.text}" — that turn had finished; I was cut off before I '
                f"marked it done. Nothing was lost."
            )
        return (
            f'"{self.text}" — I was cut off partway through that. '
            f"Want me to pick it up?"
        )


@dataclass(frozen=True)
class StartupReport:
    """What the last stop left behind. Three-valued by construction: an
    interrupted turn, a half-skipped record, or a clean stop — and each is
    said, never inferred from an absence."""

    head: int
    claimed: int
    done: int
    interrupted: Optional[Interrupted] = None
    released_record_seq: Optional[int] = None
    checkpoints_rebuilt: bool = False

    @property
    def clean(self) -> bool:
        return self.interrupted is None and self.released_record_seq is None

    def lines(self) -> list[str]:
        """The report as omega would say it. Empty only when the stop was
        clean *and* nothing was rebuilt — an empty report is a claim, so it is
        only made when there is genuinely nothing to say."""
        out: list[str] = []
        if self.checkpoints_rebuilt:
            out.append(
                "my place in the queue was unreadable at startup, so I worked it "
                "out again from the log."
            )
        if self.interrupted is not None:
            out.append(self.interrupted.question)
        if self.released_record_seq is not None:
            out.append(
                f"a record at seq {self.released_record_seq} was half-filed when I "
                f"stopped; nothing was in flight."
            )
        return out


class Executor:
    """The one consumer. Claims, runs, records, releases — in that order."""

    __slots__ = ("_queue", "_complete", "_act", "_recovered")

    def __init__(
        self,
        queue: EventQueue,
        *,
        complete: Optional[Callable[..., provider.Response]] = None,
        act: Callable[[TurnContext], ActResult] = no_act_loop_yet,
    ) -> None:
        self._queue = queue
        self._complete = complete
        self._act = act
        self._recovered = False

    @property
    def queue(self) -> EventQueue:
        return self._queue

    @property
    def recovered(self) -> bool:
        return self._recovered

    # --- startup ----------------------------------------------------------

    def recover(self) -> StartupReport:
        """Close out whatever the last stop left in flight, and report it.

        Three cases, and each restores ``claimed == done`` so the queue is
        single-consumer again:

        * **nothing in flight** — a clean stop; the report says so;
        * **a half-skipped record** — the drain moved ``CLAIMED`` past a record
          and died before ``DONE`` followed. There is no turn and nothing to
          say, so ``DONE`` simply catches up;
        * **an interrupted turn** — a terminal record is appended under the
          turn's write key and ``DONE`` advances. The turn is **not** re-run.

        That last append is where M0's write-key dedup earns its place. If the
        turn had in fact completed — the crash landing between the record and
        the cursor — the log *rejects* the recovery record, and the rejection is
        how the executor learns which of the two happened. It is the invariant
        check doing real work rather than a feature nobody calls.

        A sidecar that could not be read is a harder case and is handled by
        rebuilding the cursors from the log: checkpoints are derived state
        (DL-021), the log is the source of truth (DL-017), and re-deriving is
        what the whole memory design does instead of migrating. The alternative
        — draining from zero — would re-run every turn omega has ever taken.
        """
        rebuilt = self._rebuild_cursors_if_lost()

        in_flight = self._queue.in_flight()
        if in_flight is None:
            self._recovered = True
            return StartupReport(
                head=self._queue.head(),
                claimed=self._queue.claimed(),
                done=self._queue.done(),
                checkpoints_rebuilt=rebuilt,
            )

        pending = self._queue.at(in_flight)
        if not pending.is_event:
            # A record the drain was skipping. Both cursors move for a skip, so
            # this is the window between them: there was never a turn here.
            self._queue.finish(in_flight)
            self._recovered = True
            return StartupReport(
                head=self._queue.head(),
                claimed=self._queue.claimed(),
                done=self._queue.done(),
                released_record_seq=in_flight,
                checkpoints_rebuilt=rebuilt,
            )

        record_seq, finished = self._close_out(pending)
        self._queue.finish(in_flight)
        self._recovered = True
        return StartupReport(
            head=self._queue.head(),
            claimed=self._queue.claimed(),
            done=self._queue.done(),
            interrupted=Interrupted(
                seq=in_flight,
                payload=pending.payload,
                record_seq=record_seq,
                finished_before_crash=finished,
            ),
            checkpoints_rebuilt=rebuilt,
        )

    def _close_out(self, pending: Pending) -> tuple[int, bool]:
        """Give an interrupted turn its one terminal record.

        Returns ``(record_seq, finished_before_crash)``. Never re-runs the turn
        and never calls a model: the whole point of at-most-once is that the
        work does not happen a second time on its own.
        """
        head_before = self._queue.head()
        key = episodes.turn_write_key(pending.seq)
        try:
            record_seq = self._queue.append(
                episodes.completed(
                    for_seq=pending.seq,
                    outcome="failed",
                    error=INTERRUPTED_ERROR,
                ),
                key,
            )
        except WriteKeyConflict:
            # The turn *had* finished; the crash landed between its record and
            # the cursor. The log refused a second record, which is the answer.
            return self._terminal_seq_for(pending.seq), True

        if record_seq <= head_before:
            # Dedup returned an existing, byte-identical record. Same answer.
            return record_seq, True
        return record_seq, False

    def _terminal_seq_for(self, for_seq: int) -> int:
        for episode in self._queue.store.episodes_since(for_seq):
            payload = episodes.decode(episode.payload)
            if episodes.is_terminal(payload) and payload.get("for_seq") == for_seq:
                return episode.seq
        raise RuntimeError(
            f"the log rejected a terminal record for turn {for_seq} but holds none"
        )

    def _rebuild_cursors_if_lost(self) -> bool:
        """Re-derive ``CLAIMED``/``DONE`` from the log when the sidecar was lost.

        A damaged sidecar reads as 0 everywhere (DL-021/Q9), which keeps the log
        openable — derived state must never make acknowledged episodes
        unreachable. But 0 also means *re-deliver everything*, and re-running
        finished turns is precisely what claim-before-act exists to prevent.

        So the cursors are rebuilt from the only durable evidence there is: an
        event whose terminal record is present is done. Both cursors are set to
        the end of the longest *prefix* of the log in which every event is
        resolved, which is conservative in the safe direction — a later event
        with no record is re-delivered and runs once, exactly as if it had just
        arrived.
        """
        store = self._queue.store
        if not store.checkpoints_reset:
            return False
        if self._queue.claimed() or self._queue.done():
            return False
        head = store.head()
        if head == 0:
            return False

        resolved: set[int] = set()
        kinds: dict[int, str] = {}
        for episode in store.episodes_since(0):
            payload = episodes.decode(episode.payload)
            kinds[episode.seq] = payload["kind"]
            if episodes.is_terminal(payload):
                resolved.add(int(payload["for_seq"]))

        settled = 0
        for seq in range(1, head + 1):
            if kinds[seq] in EVENT_KINDS and seq not in resolved:
                break
            settled = seq

        if settled:
            self._queue.claim(settled)
            self._queue.finish(settled)
        return True

    # --- draining ---------------------------------------------------------

    def step(self) -> Optional[TurnResult]:
        """Handle the next episode, if there is one.

        Returns the turn's result for an event, and ``None`` both when the
        queue is empty and when the next episode was a record that was skipped.
        Callers that need to tell those apart use :meth:`drain`, which counts
        them separately — a ``None`` meaning two different things is fine only
        when nothing branches on it.
        """
        if not self._recovered:
            raise NotRecovered("call recover() before draining; see §1.2")
        for pending in self._queue.pending():
            return self._handle(pending)
        return None

    def drain(self, *, max_turns: Optional[int] = None) -> list[TurnResult]:
        """Run every waiting episode to completion, in order.

        Loops until a pass finds nothing, rather than over a single snapshot:
        an event that arrives *during* a turn is not in the batch that was read
        before it, and a drain that stopped there would leave it sitting behind
        an idle executor until something else happened to wake it.
        """
        if not self._recovered:
            raise NotRecovered("call recover() before draining; see §1.2")

        results: list[TurnResult] = []
        while True:
            progressed = False
            for pending in self._queue.pending():
                progressed = True
                result = self._handle(pending)
                if result is not None:
                    results.append(result)
                    if max_turns is not None and len(results) >= max_turns:
                        return results
            if not progressed:
                return results

    def _handle(self, pending: Pending) -> Optional[TurnResult]:
        if not pending.is_event:
            self._queue.skip(pending.seq)
            return None

        # Claim, then act. Everything after this line is covered by the
        # startup report if the process dies (§1.2).
        self._queue.claim(pending.seq)
        result = run_turn(
            self._queue,
            pending,
            complete=self._complete,
            act=self._act,
        )
        self._queue.finish(pending.seq)
        return result

    def __repr__(self) -> str:
        return f"<Executor {self._queue!r} recovered={self._recovered}>"
