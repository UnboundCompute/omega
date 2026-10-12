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

import inspect
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from omega import (
    access,
    blobs,
    episodes,
    grants,
    habits,
    inbox as inboxes,
    learn,
    leave,
    lessons,
    listen,
    location,
    machine,
    mailcursor,
    mcp,
    notice,
    projects,
    promises,
    provider,
    tasks,
    tidy,
    tools,
    topics,
    transcripts,
    watcher,
)
from omega.derive import Learned, Moments, OpenWork, Readings
from omega.followups import Followups
from omega.grants import Grants
from omega.memory import WriteKeyConflict
from omega.queue import EVENT_KINDS, EventQueue, Pending
from omega.schedule import Schedule, Scheduler
from omega.turn import (
    NOT_RECALLED,
    ActResult,
    TurnContext,
    TurnResult,
    _transcript,
    no_act_loop_yet,
    run_turn,
    write_memory,
)

#: Messages that must arrive before another reflection pass runs (DL-054).
#:
#: A rate limit on a model call that nobody asked for, so it is set from cost
#: and from what a pass needs to see rather than from responsiveness: a pattern
#: is not visible in three messages, and there is no one waiting on the answer.
#: Twenty is roughly a session's worth of back-and-forth. It is a number that
#: wants measuring and has not been — the ledger says so rather than the comment
#: implying otherwise.
REFLECT_EVERY = 20

#: How much conversation one pass reads. Larger than ``REFLECT_EVERY`` on
#: purpose: the windows overlap, so a habit that shows up either side of a
#: boundary is still one pattern to one pass rather than two halves neither pass
#: can see. `_transcript` applies the same character budget recall uses, so this
#: is an upper bound on episodes and not on prompt size.
REFLECT_WINDOW = 40

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


def _takes_after(read: Optional[Callable[..., Sequence[str]]]) -> bool:
    """Does this sense accept the mail cursor as ``after=`` (DL-090)?"""
    if read is None:
        return False
    try:
        return "after" in inspect.signature(read).parameters
    except (TypeError, ValueError):
        return False


def _sense(
    read: Optional[Callable[[datetime], Sequence[str]]],
    now: datetime,
    name: str,
) -> Sequence[str]:
    """One outside sense's lines for a look, never raising (DL-080).

    The readers already turn their own failures into a "couldn't read" line;
    this catches what they did not expect, and still says so in words rather
    than dropping the block, because an absent block reads as "not connected"
    and a broken one must not look like that.
    """
    if read is None:
        return ()
    try:
        return tuple(read(now))
    except Exception as exc:  # noqa: BLE001 - a sense never kills the look
        return (f"couldn't read the {name}: {type(exc).__name__}",)


class Executor:
    """The one consumer. Claims, runs, records, releases — in that order."""

    __slots__ = (
        "_queue",
        "_complete",
        "_act",
        "_recovered",
        "_open",
        "_learned",
        "_grants",
        "_head_grants",
        "_followups",
        "_promises",
        "_promises_now",
        "_lessons",
        "_location",
        "_grants_now",
        "_topics",
        "_topics_now",
        "_tasks",
        "_tasks_now",
        "_watches",
        "_leaves",
        "_moments",
        "_standing",
        "_online",
        "_machine",
        "_calendar",
        "_inbox",
        "_inbox_after",
        "_inbox_names",
        "_readings",
        "_access",
        "_access_now",
        "_probe",
        "_probed",
    )

    def __init__(
        self,
        queue: EventQueue,
        *,
        complete: Optional[Callable[..., provider.Response]] = None,
        act: Callable[[TurnContext], ActResult] = no_act_loop_yet,
        online: Optional[Callable[[], bool]] = None,
        machine: Optional[Callable[[], Sequence[str]]] = None,
        calendar: Optional[Callable[[datetime], Sequence[str]]] = None,
        inbox: Optional[Callable[[datetime], Sequence[str]]] = None,
        probe: Optional[Callable[[str], Optional[bool]]] = None,
        inbox_names: Sequence[str] = (),
    ) -> None:
        self._queue = queue
        self._complete = complete
        self._act = act
        # DL-068's two senses for the unprompted look, both injected so a test
        # never opens a socket or runs pmset. ``None`` means "assume online"
        # and "no machine line" — the behaviour before DL-068.
        self._online = online
        self._machine = machine
        # DL-080's two outside senses, injected for the same reason: a test
        # never reaches a calendar feed or an IMAP server. Each takes the
        # look's clock and returns already-worded lines.
        self._calendar = calendar
        self._inbox = inbox
        # DL-090: an inbox sense that takes ``after`` reads from the look's
        # mail cursor, once an hour; one that takes only the clock is read at
        # every look, as before, and nothing is recorded of it.
        self._inbox_after = _takes_after(inbox)
        # DL-092: every mailbox the inbox sense reads, when there is more
        # than the default one; empty is the one mailbox, read as before.
        self._inbox_names = tuple(inbox_names)
        self._recovered = False
        self._open = OpenWork()
        self._learned = Learned()
        # Standing grants (DL-085), folded like the views beside it.
        self._grants = Grants()
        # The same fold kept at the log's head (DL-087 #8), for one question
        # only: which proposals no longer apply, to be written down as
        # withdrawn. `_grants` cannot answer it: it stops at the turn.
        self._head_grants = Grants()
        # Omega's own failures (DL-087 #2), folded the same way: a count that
        # survives a restart because the failures are in the log.
        self._followups = Followups()
        # Promises they made (DL-087 #1): one fold kept at the turn being
        # handled, as the views above are, and one at the log's head with a
        # head-kept grants fold beside it, for the pass after each turn's
        # reply, which must see what that turn (and its grant) just wrote.
        self._promises = promises.Promises()
        self._promises_now = promises.Promises()
        self._grants_now = Grants()
        # omega's own lessons and skills (DL-086), kept at the turn being
        # handled like the views above, and apart from `_learned`: what omega
        # learned about its own working is never a claim about the person.
        self._lessons = lessons.Lessons()
        # Where they are (DL-091), kept at the turn being handled: a fix that
        # arrives after the event is not where they were when they asked.
        self._location = location.Locations()
        # Topics they keep asking about and the ones kept up with (DL-087 #6):
        # one fold at the turn (what its brief lists), one at the head for the
        # pass after each reply and the daily look.
        self._topics = topics.Topics()
        self._topics_now = topics.Topics()
        # DL-095: one fold advanced through each handled event (the turn's own
        # view, for the brief and the step obligation) and one advanced to the
        # head on the idle pass (what is drivable right now).
        self._tasks = tasks.Tasks()
        self._tasks_now = tasks.Tasks()
        # Page watches (DL-096), kept at the log's head for the idle pass.
        self._watches = watcher.Watches()
        # The leave-now alarms (DL-093), folded at the log's head: what the
        # sense pass checks, pings and closes.
        self._leaves = leave.Leaves()
        # A third view rather than a field on `_learned` (DL-062): what is
        # true and what happened have different lifetimes, and keeping them
        # in one fold is what would let events leak into the claim set.
        self._moments = Moments()
        # What other machines reported about themselves (DL-072): the Mac's
        # disk and battery, and whether the person is at it. A fourth view for
        # the reason the third is one — a reading has a shelf life a claim
        # does not, and folding the two together is how a stale value gets
        # treated as a belief.
        self._readings = Readings()
        # Lost sign-ins (DL-087 #3): one fold kept at the turn being handled,
        # which decides whether a fire is paused and what a brief lists, and
        # one at the log's head for the idle pass that tells and probes.
        self._access = access.Access()
        self._access_now = access.Access()
        # Asks whether a lost origin works again: ``True``, ``False`` or
        # ``None``. Injected, so a test never logs in anywhere; ``None`` means
        # no probing, and recovery is then seen only in a read that works.
        self._probe = probe
        # When each origin was last probed, this process. Not in the log: a
        # restart is the usual fix, so the first idle pass after one probes.
        self._probed: dict[str, datetime] = {}
        # Read-only: `due` and `fire` belong to the clock thread and are
        # never called from here. A second fold of the same log rather than
        # a shared object, because the clock mutates its copy on its own
        # thread — and DL-036's whole claim is that this table is derived,
        # so two folds of one log agree by construction rather than by
        # locking. It exists so a teach drop can be told what is already
        # scheduled and name one to stop (DL-044 #6).
        self._standing = Scheduler(queue)

    @property
    def queue(self) -> EventQueue:
        return self._queue

    @property
    def recovered(self) -> bool:
        return self._recovered

    @property
    def open_work(self) -> OpenWork:
        """What omega is still waiting on a person for (DL-041).

        Owned here because the executor is the single consumer: one view
        advanced by the one thread that runs turns needs no lock, and any second
        owner would need one.

        **Not built by** :meth:`recover`, and that is a decision rather than an
        omission. Recovery would have to fold to the head of the log, and a view
        that has already reached the head cannot be pulled back to the episode a
        turn is actually handling — the fold only goes forwards. The first turn
        after a start therefore folds the log itself, which is the same work
        deferred by one event, and buys the property that matters: the prompt
        for an episode is a function of the log up to that episode, not of how
        far behind the drain happened to be. Anything wanting "open as of now"
        instead builds its own with :meth:`OpenWork.rebuild`, which is cheap
        (DL-041) and cannot be made stale by a reader that does not advance it.
        """
        return self._open

    @property
    def learned(self) -> Learned:
        """What omega has been taught (DL-042).

        Owned, advanced and bounded exactly like :attr:`open_work`, and kept as
        a second view rather than folded into the first because the two have
        different lifetimes: an obligation is discharged and gone, a claim
        stands until something supersedes it. Sharing one fold would mean every
        change to either one's rules had to be reasoned about against both.
        """
        return self._learned

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
                        # Returned *without* reflecting, on purpose. A caller
                        # that asked for a bounded number of turns has not been
                        # told the queue is empty, and reflecting over a window
                        # whose tail is still waiting would move the cursor past
                        # episodes no pass has read.
                        return results
            if not progressed:
                self.reflect()
                return results

    def _completer(self) -> Callable[..., provider.Response]:
        """The model seam, resolved the way a turn resolves it.

        ``run_turn`` has always fallen back to the process-wide provider when
        nothing was injected (``turn.py``: ``complete = complete or
        provider.complete``), and nothing in production injects one — the
        provider is installed globally at startup and the ``Executor`` is built
        with ``complete=None``. The two idle passes below read ``self._complete``
        directly and returned early when it was ``None``, so on the assembled
        process they were both unreachable: reflection and ingestion ran in
        tests, which inject a fake, and never once outside them. Resolving here
        means there is one answer to "which model does omega think with" and the
        idle passes get the same one the replies do.
        """
        return self._complete or provider.complete

    def reflect(self) -> bool:
        """Look back over recent conversation and keep what it showed (DL-054).

        Returns whether a pass ran. Called by :meth:`drain` at the moment a
        sweep of the queue finds nothing — the one moment omega is provably not
        mid-turn — so reflection costs no reply latency and adds no failure mode
        to the path that produces an answer. Public because the live evals drive
        it directly; nothing else should need to.

        **Never raises**, for :func:`omega.turn._teach`'s reason carried one step
        further. That one must not lose a composed reply to a second call going
        wrong; this one must not take down the drain thread, which would stop
        every future turn in the process to improve a memory. A failure becomes
        a ``reflection.done`` carrying the reason, which is the only place it can
        be seen later — there is no reply here to put a receipt in.

        **The cursor moves whether or not anything is filed**, including on
        failure. A pass that failed and then re-read the same window on the next
        idle tick would call the model again every few seconds for as long as
        the fault lasted, which turns a broken `learn` role into a bill. The
        window is bounded and conversation keeps arriving, so nothing is lost
        that the next pass will not see.
        """
        if not self._recovered:
            return False
        self._learned.advance(self._queue.store)
        if self._learned.since_reflection < REFLECT_EVERY:
            return False

        head = self._queue.head()
        # Reports skipped for recall's reason (DL-072), and because their
        # digests are already learned from by `work_reports` under their own
        # lens; reflecting on them again here would file the same habit twice.
        window = self._queue.recent(REFLECT_WINDOW, skipping=NOT_RECALLED)
        known = self._learned.claims()
        # Metered per pass, and the receipt says what it spent (DL-070).
        meter = provider.Meter(self._completer())
        try:
            found = learn.reflect(
                meter,
                transcript=_transcript(window),
                known=known,
            )
            filed = learn.file_claims(
                self._queue,
                found.claims,
                # There is no turn, so `for_seq` names the episode the pass read
                # up to. The field is required of every attached kind and this
                # is the honest value for it: the claim is *about* that stretch,
                # and pointing it at some earlier turn would invent a cause.
                for_seq=head,
                source_seq=head,
                explicit=False,
            )
            # Filed unconditionally, not behind a check on the lens. `reflect`
            # already returns nothing here for a lens that was not offered
            # moments (DL-062), so a second guard would be a second place for
            # the rule to drift, and this call is a no-op by construction.
            learn.file_moments(
                self._queue,
                found.moments,
                for_seq=head,
                source_seq=head,
            )
        except Exception as exc:  # noqa: BLE001 - see the docstring
            reason = str(exc) or type(exc).__name__
            self._queue.append(
                episodes.reflection_done(
                    through=head,
                    reason=reason,
                    usage=meter.usage(),
                )
            )
            return True
        self._queue.append(
            episodes.reflection_done(
                through=head,
                filed=len(filed),
                usage=meter.usage(),
            )
        )
        return True

    def tidy(self, *, now: Optional[str] = None) -> bool:
        """Keep omega's own memory clean (DL-087 #4). Returns whether a pass ran.

        Merges duplicate claims, marks old changeable-state ones, records
        contradictions for the person to settle, and reads their answers — all
        as appends, never edits (:mod:`omega.tidy`). Self-gating: at most daily,
        or sooner when the person has said something since a question went out.

        **Never raises**, for :meth:`reflect`'s reason. A pass that broke past
        :func:`tidy.run`'s own handling still leaves a receipt with the reason,
        and its cursors where they were.
        """
        if not self._recovered:
            return False
        self._learned.advance(self._queue.store)
        clock = now or episodes.now()
        if not tidy.due(self._learned, clock):
            return False
        try:
            tidy.run(self._queue, self._learned, self._completer(), now=clock)
        except Exception as exc:  # noqa: BLE001 - see the docstring
            self._queue.append(
                episodes.memory_tidied(
                    checked_through=self._learned.checked_through,
                    heard_through=self._learned.heard_through,
                    reason=str(exc) or type(exc).__name__,
                    at=clock,
                )
            )
        return True

    def ingest(self, *, root: Optional[Any] = None, now: Optional[Any] = None) -> int:
        """Read finished transcripts from other agents and keep what they showed.

        DL-057. Returns how many sessions were read. Called from the same idle
        moment as :meth:`reflect` and for the same two reasons: the queue being
        empty is the one point omega is provably not mid-turn, and a sense that
        cost reply latency would be a sense worth turning off.

        **One model call per session, and a hard cap on sessions per pass.**
        The digest is mechanical (:mod:`omega.transcripts`); the only model call
        is the same reflection pass DL-054 already runs, pointed at a different
        object through :data:`omega.learn.WORK`. That is the whole design: this
        path adds a *source*, not an engine, so everything already true of
        reflection — the refusal to invent, the cap of three, ``explicit=False``,
        the claim shape, the fold — is true here without being re-derived.

        **A receipt per session, filed or not.** Most sessions teach nothing,
        so "have I read this?" cannot be answered from the claims; it is
        answered from :attr:`Learned.ingested`, which is why the receipt is
        written even when the digest was empty or the pass filed none.

        **Never raises**, for :meth:`reflect`'s reason. A transcript is the
        first input to reach omega's learning path that omega did not produce
        and the person did not type, and the drain thread must not be something
        a malformed file on disk can stop.
        """
        if not self._recovered:
            return 0
        if self._queue.head() == 0:
            # An omega with an empty log has never been spoken to, and the first
            # thing in it should not be a belief about a person it has not met,
            # inferred from a file it found on disk. Also the honest answer to a
            # mechanical problem: every claim names the episode it came from,
            # and on an empty log there is no episode to name.
            return 0
        self._learned.advance(self._queue.store)
        try:
            sessions = transcripts.discover(
                root, now=now, seen=self._learned.ingested
            )
        except Exception:  # noqa: BLE001 - see the docstring
            # No receipt: nothing was read, so there is no session to name one
            # after, and a failure to *look* is not a failure to learn from any
            # particular thing.
            return 0

        read = 0
        for session in sessions[: transcripts.MAX_SESSIONS_PER_PASS]:
            self._ingest_one(session)
            read += 1
        return read

    def _ingest_one(self, session: transcripts.Session) -> None:
        # The head is taken before the digest, as it always was, so the claims
        # name the log as it stood when the session was read.
        head = self._queue.head()
        receipt = {
            "session": session.id,
            "source": session.source,
            "project": session.project,
        }
        try:
            digest = transcripts.digest(session)
        except Exception as exc:  # noqa: BLE001
            reason = str(exc) or type(exc).__name__
            self._learn_from_digest(
                None, make=episodes.transcript_ingested, receipt=receipt,
                lens=learn.WORK, reason=reason, cause=head,
            )
            return
        self._learn_from_digest(
            digest, make=episodes.transcript_ingested, receipt=receipt,
            lens=learn.WORK, cause=head,
        )
        self._note_project(
            digest, session=session.id, fallback=session.project,
            ended_at=session.modified.isoformat(),
        )

    def _note_project(
        self,
        digest: Optional[str],
        *,
        session: str,
        fallback: str,
        ended_at: str,
    ) -> None:
        """Where this session left its project, as one ``project.state`` (DL-087 #7).

        After the receipt, and only for a session that had a digest: a session
        too short to digest, or one whose digest could not be made, has
        nothing to read a state from, and its receipt already says so. One
        LEARN call, failing closed — an answer that is not the exact shape is
        a ``project.state`` with a ``reason`` and no state, never a guess.
        Separate from the reflection on purpose: the claims and the receipt
        are byte-for-byte what they were before this existed.

        **Never raises**, for :meth:`ingest`'s reason. Nothing is done to the
        repository the session was about; this only reads its digest.
        """
        if not digest:
            return
        try:
            name = projects.project_of(digest, fallback)
            meter = provider.Meter(self._completer())
            try:
                state = projects.extract(meter, digest)
            except Exception as exc:  # noqa: BLE001 - recorded, not raised
                self._queue.append(
                    episodes.project_state(
                        project=name, session=session, ended_at=ended_at,
                        reason=str(exc) or type(exc).__name__,
                        usage=meter.usage(),
                    )
                )
                return
            self._queue.append(
                episodes.project_state(
                    project=name, session=session, ended_at=ended_at,
                    usage=meter.usage(), **state,
                )
            )
        except Exception:  # noqa: BLE001 - see the docstring
            return

    def _learn_from_digest(
        self,
        digest: Optional[str],
        *,
        make: Callable[..., dict[str, Any]],
        receipt: dict[str, Any],
        lens: Any,
        reason: Optional[str] = None,
        cause: Optional[int] = None,
    ) -> None:
        """Reflect over one digest, file what it showed, write its receipt.

        The half of a transcript or a usage day that does not depend on where
        the digest was made (DL-072). A session read off this disk and a session
        the Mac read and sent arrive here as the same string, so they are
        reflected, filed and receipted by one body of code — which is what
        keeps ``Learned.ingested`` and ``Learned.digested`` one cursor whatever
        machine did the reading.

        ``digest`` is ``None`` (or empty) for a unit too short to show anything,
        which files a receipt with nothing filed; ``reason`` is a digest that
        could not be made at all. ``cause`` is the episode the claims name —
        the report they came from when there is one, else the head at the
        moment the unit was read, because there is no turn behind it.
        """
        # Advanced per unit, not once per pass: the second session of a pass
        # must see what the first one filed, or a habit visible in both gets
        # written twice instead of once and superseded.
        self._learned.advance(self._queue.store)
        head = cause if cause is not None else self._queue.head()
        if reason is not None:
            self._queue.append(make(**receipt, reason=reason))
            return
        if not digest:
            # Not a failure. A session too short to show a pattern, or a quiet
            # day, is the ordinary case, and calling it one would bury a real
            # fault in the count that exists to surface it.
            self._queue.append(make(**receipt, filed=0))
            return
        # Metered per pass, and the receipt says what it spent (DL-070).
        meter = provider.Meter(self._completer())
        try:
            found = learn.reflect(
                meter,
                transcript=digest,
                known=self._learned.claims(),
                observing=lens,
            )
            filed = learn.file_claims(
                self._queue,
                found.claims,
                # As in `reflect`: there is no turn, so the head at the moment
                # the unit was read (or the report it arrived in) is the honest
                # cause to point at.
                for_seq=head,
                source_seq=head,
                explicit=False,
            )
            # Filed unconditionally, not behind a check on the lens. `reflect`
            # already returns nothing here for a lens that was not offered
            # moments (DL-062), so a second guard would be a second place for
            # the rule to drift, and this call is a no-op by construction.
            learn.file_moments(
                self._queue,
                found.moments,
                for_seq=head,
                source_seq=head,
            )
        except Exception as exc:  # noqa: BLE001
            reason = str(exc) or type(exc).__name__
            self._queue.append(make(**receipt, reason=reason, usage=meter.usage()))
            return
        self._queue.append(make(**receipt, filed=len(filed), usage=meter.usage()))

    def hear(
        self, *, folder: Optional[Any] = None, now: Optional[Any] = None
    ) -> int:
        """Hear recordings dropped in the watched folder (DL-066).

        Named ``hear`` and not ``listen`` on purpose: :mod:`omega.listen` is the
        module this calls and ``listen=`` is already the flag that binds the
        HTTP channel, so a third meaning of the word inside one class would make
        every line of it ambiguous to read.

        Returns how many were heard. Structurally :meth:`ingest` over a third
        source, and deliberately so — the sense being added is a *source*, not
        an engine, so the receipt-as-cursor, the ``explicit=False``, the refusal
        to invent and the fold all arrive from the reflection path unchanged.

        **Two things genuinely differ from its two siblings.**

        One per pass, where a transcript pass takes three. Those digests are
        mechanical; this one shells out to whisper, and an hour of audio takes
        tens of minutes on this machine. The executor is a single consumer, so
        that is time omega cannot answer a message — bounded at one recording,
        unbounded at three. It is still a real stall, and the honest fix is a
        detached sub-loop re-entering as ``work.finished`` (DL-016 §1.5) rather
        than a smaller cap; that is not built, and this says so instead of
        implying the cost away.

        An environment check before the loop, not inside it. A receipt marks a
        recording heard permanently, so writing one because whisper is missing
        would burn every recording in the folder on a machine that had simply
        not installed it yet. :func:`omega.listen.not_ready` answers that once,
        and a pass that cannot run writes nothing at all — the same rule
        :meth:`ingest` applies when it fails to *look*.

        **Never raises**, for :meth:`reflect`'s reason: the drain thread must
        not be something a malformed file on disk can stop.
        """
        if not self._recovered:
            return 0
        if self._queue.head() == 0:
            # An omega with an empty log has never been spoken to, and the
            # first thing in it should not be a belief about a person it has
            # not met, inferred from a file it found on disk.
            return 0
        if listen.not_ready() is not None:
            return 0
        self._learned.advance(self._queue.store)
        try:
            waiting = listen.waiting(folder, now=now)
        except Exception:  # noqa: BLE001 - see the docstring
            # No receipt: nothing was heard, so there is no recording to name
            # one after, and a failure to look is not a failure to hear.
            return 0

        heard = 0
        for path in waiting[: listen.MAX_PER_PASS]:
            self._hear_one(path, folder=folder)
            heard += 1
        return heard

    def _hear_one(self, path: Any, *, folder: Optional[Any] = None) -> None:
        # Advanced per recording for `_learn_from_digest`'s reason: the second
        # recording of a pass must see what the first one filed.
        self._learned.advance(self._queue.store)
        # Taken before whisper runs, as it always was: a transcription takes
        # minutes, messages arrive meanwhile, and the claims should name the
        # log as it stood when the recording was picked up.
        head = self._queue.head()
        store = blobs.BlobStore.open(self._queue.store.root)

        try:
            audio = store.put(path)
        except Exception:  # noqa: BLE001
            # No receipt, and this is the one place that is forced rather than
            # chosen: the digest *is* the identity, so a failure to store the
            # bytes leaves nothing to write a receipt about.
            return

        receipt: dict[str, Any] = {
            "recording": audio.digest,
            "source": "folder",
            "title": path.name,
            "mime": audio.mime,
            "bytes": audio.bytes,
        }
        if audio.digest in self._learned.heard:
            # Already heard. No second receipt — the cursor is the answer, and
            # putting the bytes again was free because the store is content
            # addressed. Filed away so the queue stops offering it.
            listen.file_away(path, folder=folder)
            return

        try:
            said = listen.transcribe(path)
        except Exception as exc:  # noqa: BLE001
            # About this recording, so it earns a receipt: the environment was
            # checked once before the pass, which is what makes this failure
            # attributable to the file rather than to the machine.
            reason = str(exc) or type(exc).__name__
            self._queue.append(episodes.audio_captured(**receipt, reason=reason))
            listen.file_away(path, folder=folder)
            return
        receipt["duration"] = said.duration
        self._learn_from_heard(said.text, receipt=receipt, store=store, cause=head)
        listen.file_away(path, folder=folder)

    def _learn_from_heard(
        self,
        text: str,
        *,
        receipt: dict[str, Any],
        store: "blobs.BlobStore",
        cause: Optional[int] = None,
    ) -> None:
        """Keep, review and reflect over one heard transcript, then receipt it.

        The half of a recording that does not depend on where whisper ran
        (DL-072): the Mac transcribes and sends the text, a core on the Mac
        transcribes it itself, and both land here. ``receipt`` already names the
        recording and its duration; this adds the transcript and review digests
        and the outcome. ``cause`` is as in :meth:`_learn_from_digest`.
        """
        self._learned.advance(self._queue.store)
        head = cause if cause is not None else self._queue.head()
        title = str(receipt["title"])
        receipt["transcript"] = self._put_text(store, text, ".txt")

        # Metered per pass, and the receipt says what it spent (DL-070).
        meter = provider.Meter(self._completer())
        try:
            read = listen.review(text, complete=meter, title=title)
        except Exception as exc:  # noqa: BLE001
            # A partial, and the payload is built to say so: the transcript
            # digest stays on the receipt alongside the reason, so the work
            # that did land is not reported as lost.
            reason = str(exc) or type(exc).__name__
            self._queue.append(
                episodes.audio_captured(
                    **receipt,
                    reason=reason,
                    usage=meter.usage(),
                )
            )
            return
        rendered = read.render(title=title)
        if rendered:
            receipt["review"] = self._put_text(store, rendered, ".md")

        try:
            found = learn.reflect(
                meter,
                transcript=text[: listen.MAX_REVIEW_CHARS],
                known=self._learned.claims(),
                observing=learn.ROOM,
            )
            filed = learn.file_claims(
                self._queue,
                found.claims,
                # As in `_learn_from_digest`: there is no turn, so the head at
                # the moment the recording was heard is the honest cause.
                for_seq=head,
                source_seq=head,
                explicit=False,
            )
            learn.file_moments(
                self._queue, found.moments, for_seq=head, source_seq=head
            )
        except Exception as exc:  # noqa: BLE001
            reason = str(exc) or type(exc).__name__
            self._queue.append(
                episodes.audio_captured(
                    **receipt,
                    reason=reason,
                    usage=meter.usage(),
                )
            )
            return

        self._queue.append(
            episodes.audio_captured(
                **receipt,
                filed=len(filed),
                usage=meter.usage(),
            )
        )

    @staticmethod
    def _put_text(store: "blobs.BlobStore", text: str, suffix: str) -> str:
        """Store derived text as a blob and return its digest.

        Through a temp file because :meth:`BlobStore.put` takes a path: the
        store hashes a stream it opened itself, which is what lets it count
        ``bytes`` off the content rather than trusting a ``stat()``. Writing the
        text out to be read back is the price of not giving it a second entry
        point that could disagree.
        """
        with tempfile.TemporaryDirectory(prefix="omega-heard-") as tmp:
            scratch = Path(tmp) / f"text{suffix}"
            scratch.write_text(text, encoding="utf-8")
            return store.put(scratch).digest

    def digest_usage(
        self, *, path: Optional[Any] = None, now: Optional[Any] = None
    ) -> int:
        """Read finished days from the Mac's own usage record (DL-059).

        Returns how many days were digested. Structurally :meth:`ingest` with a
        different source, and deliberately so: the sense being added is a
        *source*, not an engine, so the cap, the ``explicit=False``, the refusal
        to invent, the receipt-as-cursor and the fold all come from the
        reflection path unchanged rather than being re-derived here.

        **The one thing that differs is where the bound lives.** A session is
        finished when its file stops changing, which :mod:`omega.transcripts`
        can see. A day is finished when it is over, which is a fact about the
        clock — so the eligibility rule is in :func:`omega.habits.discover` and
        this method holds no date logic at all.

        **Never raises**, for :meth:`ingest`'s reason and one more: this reads a
        database in an undocumented format belonging to the operating system,
        which an update is free to change under us. The drain thread must not be
        something a macOS release can stop.
        """
        if not self._recovered:
            return 0
        if self._queue.head() == 0:
            # As in `ingest`: the first thing in an empty log must not be a
            # belief about a person omega has not met, and there is no episode
            # for such a claim to name as its cause.
            return 0
        self._learned.advance(self._queue.store)
        try:
            days = habits.discover(path, now=now, seen=self._learned.digested)
        except Exception:  # noqa: BLE001 - see the docstring
            # No receipt, for the reason `discover` itself returns empty when
            # the database cannot be opened: a failure to *look* is not a
            # failure to learn from any particular day, and writing receipts for
            # days that were never read would mark them read forever.
            return 0

        read = 0
        for day in days[: habits.MAX_DAYS_PER_PASS]:
            self._digest_one_day(day)
            read += 1
        return read

    def _digest_one_day(self, day: habits.Day) -> None:
        # Taken before the digest, as in `_ingest_one`.
        head = self._queue.head()
        receipt = {"day": day.id, "source": day.source}
        try:
            digest = habits.digest(day)
        except Exception as exc:  # noqa: BLE001
            reason = str(exc) or type(exc).__name__
            self._learn_from_digest(
                None, make=episodes.usage_digested, receipt=receipt,
                lens=learn.DAY, reason=reason, cause=head,
            )
            return
        self._learn_from_digest(
            digest, make=episodes.usage_digested, receipt=receipt,
            lens=learn.DAY, cause=head,
        )

    def work_reports(self) -> int:
        """Learn from what a sense relay sent and the core has not yet worked.

        DL-072. Returns how many reports were worked. The Mac did the reading
        and the digest; this does the rest — the reflection, the filing and the
        receipt — through exactly the methods the local senses use, so a report
        ends in the same receipt a local read would have written and moves the
        same cursor. A report with a receipt is never worked again, and a unit
        the core already read off its own disk is never worked from a report.

        **The same per-pass caps as the local senses**, per sense: three
        sessions, two days, one recording. A relay that sent a backlog is worked
        off at the rate omega has always learned, not in one long stall.

        **Never raises**, for :meth:`ingest`'s reason, and one more: what is in
        a report was written by another machine, and the drain thread must not
        be something it can stop.
        """
        if not self._recovered:
            return 0
        try:
            self._learned.advance(self._queue.store)
            owed = self._learned.unworked()
        except Exception:  # noqa: BLE001 - see the docstring
            return 0
        caps = {
            episodes.REPORT_TRANSCRIPT: transcripts.MAX_SESSIONS_PER_PASS,
            episodes.REPORT_USAGE: habits.MAX_DAYS_PER_PASS,
            episodes.REPORT_RECORDING: listen.MAX_PER_PASS,
        }
        worked = 0
        for seq, source, unit in owed:
            if caps.get(source, 0) <= 0:
                continue
            caps[source] -= 1
            try:
                self._work_report(seq, self._queue.at(seq).payload)
            except Exception:  # noqa: BLE001 - see the docstring
                # The report was validated against the receipt it will become
                # when it was appended, so this is a bug rather than bad input.
                # It stays unworked and is tried again next pass.
                continue
            worked += 1
        return worked

    def _work_report(self, seq: int, report: dict[str, Any]) -> None:
        source, unit, meta = report["source"], report["unit"], dict(report["meta"])
        reason = report.get("reason")
        body = report["body"]
        if source == episodes.REPORT_TRANSCRIPT:
            self._learn_from_digest(
                body, make=episodes.transcript_ingested,
                receipt={"session": unit, **meta}, lens=learn.WORK,
                reason=reason, cause=seq,
            )
            if reason is None:
                # The report's arrival stands in for the session's end: the
                # relay sends a session once it has sat idle, and the digest
                # carries only its date.
                self._note_project(
                    body, session=unit, fallback=str(meta.get("project", "")),
                    ended_at=str(report.get("at", "")),
                )
        elif source == episodes.REPORT_USAGE:
            self._learn_from_digest(
                body, make=episodes.usage_digested,
                receipt={"day": unit, **meta}, lens=learn.DAY,
                reason=reason, cause=seq,
            )
        elif source == episodes.REPORT_RECORDING:
            receipt = {"recording": unit, **meta}
            if reason is not None:
                # Whisper failed on the Mac. The receipt says so, for the
                # reason a local transcription failure earns one: it is about
                # this recording, not about the machine.
                self._queue.append(episodes.audio_captured(**receipt, reason=reason))
                return
            store = blobs.BlobStore.open(self._queue.store.root)
            self._learn_from_heard(body, receipt=receipt, store=store, cause=seq)

    def notice(
        self,
        *,
        now: Optional[datetime] = None,
        look_every_seconds: int = notice.LOOK_EVERY_SECONDS,
    ) -> bool:
        """Wake condition (b): look at what is open, unasked (DL-011, DL-061).

        Returns whether an unprompted event was appended — **not** whether omega
        said anything. Those are different facts and the caller must not collapse
        them: a pass that looked and chose silence returns ``True`` here and is a
        success, which is DL-011's *"single most important consequence of the
        time-wake"* and the regression `runtime.Said` already refuses.

        This method **starts a turn and does not run one**. It appends one
        ordinary inbound on the ``self`` channel and returns; the drain picks it
        up like any other event, the judge decides, and the reply — if there is
        one — reaches the tray through the subscriber fan-out. There is no second
        engine here, on purpose (DL-011): initiative that grew its own loop would
        grow its own voice and its own idea of what is open.

        Structurally the sibling of :meth:`ingest` and :meth:`digest_usage` — a
        pass at the idle moment, self-gating underneath, never raising — with one
        difference worth naming: those two *write memory*, and this one *causes a
        turn*. So it is the only idle pass that can reach the person, and the two
        rate limits that decide whether it may are in :mod:`omega.notice`, in
        code, never in the judge.

        **Never raises**, for :meth:`digest_usage`'s reason. A sense that can kill
        the drain thread stops every future turn, and that is a worse outcome
        than a day with no nudge in it.
        """
        if not self._recovered:
            return False
        if self._queue.head() == 0:
            # As in `ingest` and `digest_usage`: the first thing in an empty log
            # must not be omega talking to itself about a person it has not met.
            return False

        moment = now or datetime.now(timezone.utc)
        store = self._queue.store
        try:
            st = notice.standing(store)
            if notice.owes_outage(st):
                # DL-084 #4: three failed looks in a row are an outage, told
                # once, naming the cause, before looking again. Not held while
                # offline: the tell needs no model, only the outbound path.
                self._queue.append(
                    episodes.inbound(
                        tools.outage_text(st.failed_looks, st.look_error or ""),
                        channel="schedule",
                        at=moment.isoformat(),
                    )
                )
                return True
            if self._lost_access(moment):
                return True
            if self._follow_up(moment):
                return True
            if not notice.may_look(
                st, now=moment, look_every_seconds=look_every_seconds
            ):
                return False
            if self._online is not None and not self._online():
                # DL-068 #5: a look with no network is a turn that fails on its
                # first model call. Skipped, not appended — the next idle
                # moment looks again, and nothing is lost by waiting.
                return False
            machine_lines: Sequence[str] = ()
            if self._machine is not None:
                try:
                    machine_lines = tuple(self._machine())
                except Exception:  # noqa: BLE001 - a sense never kills the look
                    machine_lines = ()
            machine_lines = self._machines(machine_lines, now=moment)
            lost = self._access_now.state().lost
            paused: list[str] = []
            if self._calendar is not None and access.GOOGLE in lost:
                calendar_lines: Sequence[str] = (access.paused_line(access.GOOGLE),)
                paused.append(access.GOOGLE)
            else:
                calendar_lines = _sense(self._calendar, moment, "calendar")
                self._refused_look(access.GOOGLE, calendar_lines, moment)
            mail_more: dict[str, str] = {}
            if self._inbox is not None and self._inbox_names and self._inbox_after:
                inbox_lines, mail_at, mail_more, held = self._look_mailboxes(moment, lost)
                paused += held
                for name in self._inbox_names:
                    origin = access.inbox_origin(name)
                    if origin not in held:
                        self._refused_look(origin, [
                            line for line in inbox_lines
                            if str(line).startswith(inboxes.tag(name, ""))
                        ], moment)
            elif self._inbox is not None and access.INBOX in lost:
                inbox_lines: Sequence[str] = (access.paused_line(access.INBOX),)
                mail_at: Optional[str] = None
                paused.append(access.INBOX)
            else:
                inbox_lines, mail_at = self._look_mail(moment)
                self._refused_look(access.INBOX, inbox_lines, moment)
            self._open.advance(store)
            self._learned.advance(store)
            self._moments.advance(store)
            text = notice.situation(
                now=moment,
                blocks=self._open.blocks(),
                claims=self._learned.claims(),
                schedules=self._standing_schedules(),
                # Bounded here rather than at the fold, so the horizon is read
                # against the clock this look is being made at and not against
                # wall-clock now. A backlog look would otherwise age out the
                # very moments it is supposed to be looking at.
                moments=self._moments.recent(now=moment.isoformat()),
                heard_at=st.heard_at,
                machine=machine_lines,
                said=st.said,
                flagged_since=st.flagged_since,
                machine_known=st.machine_known,
                calendar=calendar_lines,
                inbox=inbox_lines,
            )
            # Stamped with ``moment``, not left to default to wall-clock now.
            # ``notice.standing`` charges a spoken nudge to the day of the *look*
            # (notice.py), so the look's own stamp is the day budget's ledger
            # entry; letting it default would make the budget disagree with the
            # clock the decision was made against.
            look = self._queue.append(
                episodes.inbound(
                    text,
                    channel=notice.CHANNEL,
                    machine=list(machine_lines),
                    mail_at=mail_at,
                    at=moment.isoformat(),
                    mail_inboxes=mail_more or None,
                )
            )
            for origin in paused:
                self._queue.append(episodes.access_paused(
                    origin=origin, for_seq=look, at=moment.isoformat(),
                ))
        except Exception:  # noqa: BLE001 - see the docstring
            return False
        return True

    def _lost_access(self, moment: datetime) -> bool:
        """DL-087 #3: tell them, on the first refusal, which sign-in was lost
        and what fixes it; say so once when it works again; and check a lost
        one again at most hourly.

        Folded from the log, so a restart neither repeats a tell nor forgets
        one owed. One tell per pass, as :meth:`_follow_up`. With no refusal
        in the log the view is empty, nothing is probed, and this returns
        ``False`` having appended nothing.
        """
        self._access_now.advance(self._queue.store)
        state = self._access_now.state()
        if not state:
            return False
        at = moment.isoformat()
        if state.recovering:
            self._restore(state.recovering[0], at)
            return True
        if state.owed:
            self._lose(state.owed[0].origin, at)
            return True
        if self._probe is None or not state.lost:
            return False
        if self._online is not None and not self._online():
            return False
        every = timedelta(seconds=access.PROBE_EVERY_SECONDS)
        for origin in sorted(state.lost):
            last = self._probed.get(origin)
            if last is not None and moment - last < every:
                continue
            self._probed[origin] = moment
            try:
                works = self._probe(origin)
            except Exception:  # noqa: BLE001 - a probe never kills the pass
                works = None
            if works is True:
                self._restore(origin, at)
                return True
        return False

    def _lose(self, origin: str, at: str) -> None:
        """The one tell for a lost origin, then the record that it was told."""
        self._queue.append(episodes.inbound(
            tools.tell_instruction(access.lost_words(origin)),
            channel="schedule",
            at=at,
        ))
        self._queue.append(episodes.access_lost(
            origin=origin, fix=access.fix_for(origin), at=at,
        ))
        self._access_now.advance(self._queue.store)

    def _restore(self, origin: str, at: str) -> None:
        self._queue.append(episodes.access_restored(origin=origin, at=at))
        self._queue.append(episodes.inbound(
            tools.tell_instruction(access.restored_words(origin)),
            channel="schedule",
            at=at,
        ))
        self._probed.pop(origin, None)
        self._access_now.advance(self._queue.store)

    def _refused_look(self, origin: str, lines: Sequence[str], moment: datetime) -> None:
        """A look whose own read was refused tells them then and there, as a
        tool's refusal does on the next pass. Its lines already say so."""
        if not access.refused_in(origin, lines):
            return
        state = self._access_now.state()
        if origin in state.lost:
            return
        self._lose(origin, moment.isoformat())

    def _pausing(self, event: dict[str, Any]) -> Optional[str]:
        """The lost origin this fire would read from, or ``None`` to run it.

        Only a standing schedule is ever paused: a watch or a recurring job
        whose own fire was refused. Never a tell, a brief (which lists what is
        lost), a reminder or a research follow-up: those fire once, and a
        skipped one would be lost rather than paused.
        """
        sid = event.get("schedule_id")
        if event.get("channel") != "schedule" or not sid:
            return None
        if (
            tools.is_tell(event)
            or tools.is_brief(event)
            or tools.is_reminder(event)
            or tools.research_task(event) is not None
        ):
            return None
        return self._access.state().pausing(str(sid))

    def _look_mail(self, moment: datetime) -> tuple[Sequence[str], Optional[str]]:
        """The look's mail lines, and the stamp to record if it read (DL-090).

        At most once an hour (:data:`omega.notice.MAIL_EVERY_SECONDS`), and
        then only mail that arrived since the look last read, or the last hour
        when it never has. A look between reads says so in one line instead
        of a missing block, which would read as "not connected". The stamp is
        returned only for a reading that came back, so a failed read leaves
        the cursor where it was and the next look reads again.
        """
        if not self._inbox_after:
            return _sense(self._inbox, moment, "inbox"), None
        last = mailcursor.last_read(self._queue.store, mailcursor.LOOK)
        gap = timedelta(seconds=notice.MAIL_EVERY_SECONDS)
        if last is not None and moment - last < gap:
            when = last.astimezone().strftime("%H:%M")
            return (
                f"mail is read once an hour; last read at {when}, and nothing "
                f"that arrived since is lost",
            ), None
        after = last if last is not None else moment - timedelta(
            seconds=mailcursor.FIRST_WINDOW_SECONDS
        )
        lines = _sense(lambda now: self._inbox(now, after=after), moment, "inbox")
        read = bool(lines) and not str(lines[0]).startswith(inboxes.FAILED)
        return lines, (moment.isoformat() if read else None)

    def _look_mailboxes(
        self, moment: datetime, lost: Any
    ) -> tuple[Sequence[str], Optional[str], dict[str, str], list[str]]:
        """:meth:`_look_mail` for more than one mailbox (DL-092).

        Each mailbox keeps its own hour and its own cursor: one read within
        the hour says so in its own line, one whose origin is lost says it
        is paused, and the rest are read in one go, each from its own
        cursor. Returns the lines, the default mailbox's ``mail_at``, the
        other mailboxes' stamps, and the origins this look paused. A
        mailbox stamps only when its own reading came back.
        """
        store = self._queue.store
        gap = timedelta(seconds=notice.MAIL_EVERY_SECONDS)
        said: dict[str, list[str]] = {name: [] for name in self._inbox_names}
        afters: dict[str, datetime] = {}
        held: list[str] = []
        for name in self._inbox_names:
            origin = access.inbox_origin(name)
            if origin in lost:
                said[name].append(inboxes.tag(name, access.paused_line(origin)))
                held.append(origin)
                continue
            last = mailcursor.last_read(store, mailcursor.LOOK, mailbox=name)
            if last is not None and moment - last < gap:
                when = last.astimezone().strftime("%H:%M")
                said[name].append(inboxes.tag(
                    name,
                    f"mail is read once an hour; last read at {when}, and "
                    f"nothing that arrived since is lost",
                ))
                continue
            afters[name] = last if last is not None else moment - timedelta(
                seconds=mailcursor.FIRST_WINDOW_SECONDS
            )
        loose: list[str] = []
        read: list[str] = []
        if afters:
            got = _sense(lambda now: self._inbox(now, after=afters), moment, "inbox")
            seen = inboxes.outcomes(got)
            read = [name for name in afters if seen.get(name) is True]
            for line in got:
                owner = next((n for n in said if str(line).startswith(inboxes.tag(n, ""))), None)
                (said[owner] if owner is not None else loose).append(line)
        lines = [line for name in self._inbox_names for line in said[name]] + loose
        stamp = moment.isoformat()
        mail_at = stamp if inboxes.PERSONAL in read else None
        more = {name: stamp for name in read if name != inboxes.PERSONAL}
        return tuple(lines), mail_at, more, held

    def _follow_up(self, moment: datetime) -> bool:
        """DL-087 #2: tell them, once, about one thing that kept failing.

        Folded from the log, so it is the same answer after a restart. A
        subject that was told about and works again gets its
        ``failure.recovered`` here — a record, which the next brief turns
        into one line; it starts no turn. A subject owed its tell gets one
        tell inbound (delivered as written, no model call) and its
        ``failure.told``, one per pass, oldest run first. A watch under the
        ``tidy`` grant also gets a proposal to pause it, through the gate:
        the tell asks, and their yes is what runs it. Nothing is retried.

        With no failures in the log, the view is empty and this returns
        ``False`` having appended nothing.
        """
        store = self._queue.store
        self._followups.advance(store)
        state = self._followups.state()
        if not state:
            return False
        at = moment.isoformat()
        for e in state.recovering:
            self._queue.append(episodes.failure_recovered(
                subject=e.subject, kind=e.kind, count=e.count,
                last_error=e.last_error, at=at,
            ))
        owed = state.owed()
        if not owed:
            return False
        e = owed[0]
        proposal = self._pause_proposal(e) if e.watch else None
        seq = self._queue.append(episodes.inbound(
            tools.followup_text(e, propose=proposal is not None),
            channel="schedule",
            at=at,
        ))
        if proposal is not None:
            self._queue.append(episodes.grant_proposed(
                for_seq=seq, scope=grants.TIDY, proposes="action",
                actions=[proposal], at=at,
            ))
        self._queue.append(episodes.failure_told(
            subject=e.subject, kind=e.kind, count=e.count,
            last_error=e.last_error, at=at,
        ))
        return True

    def _pause_proposal(self, entry: Any) -> Optional[dict[str, Any]]:
        """Pausing a failing watch, as the ``tidy`` tool would propose it, or
        ``None`` when there is no tidy grant or no such watch running. The
        tool's own classifier builds it, so it carries the definition a yes
        must still find (and is refused exactly where a turn's would be)."""
        # A fresh fold, not `_grants`: that one is kept at the turn being
        # handled, and folding it ahead here would show a turn what came after
        # its event. Rare enough (a watch owed its tell) to pay for.
        state = Grants.rebuild(self._queue.store).state()
        if state.live(grants.TIDY) is None:
            return None
        box = tools.ToolBox(
            store_root=self._queue.store.root,
            running=tuple(self._standing_schedules()),
            grants=state,
        )
        try:
            d = box.classify(tools.TIDY, {
                "id": entry.ref,
                "change": "retire",
                "why": f"it has failed {entry.count} times in a row (last: {entry.last_error})",
            })
        except tools.ToolRejected:
            return None
        return {"tool": d.tool, "why": d.why, "args": d.args}

    def _machines(self, local: Sequence[str], *, now: datetime) -> Sequence[str]:
        """The local machine's lines plus every reported one's, labeled.

        With no reports — a core on the Mac and no relay — the local lines pass
        through unchanged, so that look reads exactly as it did before DL-072.
        With reports, each line names its machine, because a core on the VM
        describing "the disk" without saying whose would be describing the one
        the person does not use. A reported machine past
        :data:`omega.machine.STALE_SECONDS` renders as "couldn't determine".
        """
        try:
            self._readings.advance(self._queue.store)
            reported = self._readings.devices(episodes.REPORT_MACHINE)
        except Exception:  # noqa: BLE001 - a sense never kills the look
            reported = []
        if not reported:
            return local
        lines = [f"where omega runs: {line}" for line in local]
        for r in reported:
            lines.extend(machine.reported_lines(r.device, r.at, r.body, now=now))
        return tuple(lines)

    def _handle(self, pending: Pending) -> Optional[TurnResult]:
        if not pending.is_event:
            self._queue.skip(pending.seq)
            return None

        # Claim, then act. Everything after this line is covered by the
        # startup report if the process dies (§1.2).
        self._queue.claim(pending.seq)
        # Advanced *through this event*, and no further. Through, because an
        # answer discharges the question it answers and a turn that still listed
        # it as open would go on asking about something the person had just
        # settled — which is the one place this deliberately differs from
        # recall, which stops short of the triggering episode. No further,
        # because on a backlog the log's head is ahead of the turn: folding to
        # it would show this turn questions omega had not asked yet.
        self._open.advance(self._queue.store, upto=pending.seq)
        # Bounded identically, and for a sharper version of the same reason: a
        # claim folded past this episode would be applied to a turn that
        # happened before omega was taught it. An over-eager *question* looks
        # wrong; an over-eager *habit* looks like omega simply behaving that
        # way, which is the failure DL-042 pairs its violation metric against.
        self._learned.advance(self._queue.store, upto=pending.seq)
        # Through this event, as `_open` is: a brief must see what was done
        # before it, and an answer the proposal it answers.
        self._grants.advance(self._queue.store, upto=pending.seq)
        self._followups.advance(self._queue.store, upto=pending.seq)
        self._promises.advance(self._queue.store, upto=pending.seq)
        self._lessons.advance(self._queue.store, upto=pending.seq)
        self._access.advance(self._queue.store, upto=pending.seq)
        self._location.advance(self._queue.store, upto=pending.seq)
        self._topics.advance(self._queue.store, upto=pending.seq)
        self._tasks.advance(self._queue.store, upto=pending.seq)
        lost = self._pausing(pending.payload)
        if lost is not None:
            # DL-087 #3: what it reads from is lost and they have been told.
            # Skipped, not failed: written down, closed quiet, and never a
            # model call or a read the account would only refuse again.
            self._queue.append(episodes.access_paused(origin=lost, for_seq=pending.seq))
            record = write_memory(self._queue, seq=pending.seq, outcome="silent")
            self._queue.finish(pending.seq)
            return TurnResult(seq=pending.seq, outcome="silent", record_seq=record)
        # DL-095 gate-leak: the store head before the turn, so what the turn
        # wrote with this event's for_seq can be told from what stood before it.
        head_before = self._queue.store.head()
        result = run_turn(
            self._queue,
            pending,
            complete=self._complete,
            act=self._act,
            open_work=self._open.blocks(),
            learned=self._learned.matching(
                text=str(pending.payload.get("text", "")),
                channel=str(pending.payload.get("channel", "")),
                at=str(pending.payload.get("at", "")),
            ),
            # The whole set, not the matched subset: this one answers "what
            # does a newly taught claim replace" (DL-043 #6), and a claim is
            # contradicted by one it was never going to fire alongside.
            known=self._learned.claims(),
            # Folded to the head rather than to `pending.seq`, unlike the
            # two views above. Those answer "what was true when this
            # episode arrived"; this one answers "what is running now",
            # and cancelling a schedule created while this turn sat in a
            # backlog is correct — it is still firing.
            running=self._standing_schedules(),
            # A contradiction the person has not been asked about yet (DL-087
            # #4). The turn decides whether it may carry one; this only says
            # which are waiting, as of this episode.
            ask=self._learned.unasked(),
            grants=self._grants.state(),
            followups=self._followups.state(),
            promises=self._promises.state(),
            lessons=self._lessons.state(),
            access=self._access.state(),
            location=self._location.state(),
            topics=self._topics.state(),
            tasks=self._tasks.state(),
        )
        self._queue.finish(pending.seq)
        # After the reply is durable, and never able to fail or hold it.
        self._hold_promises(pending)
        self._notice_topics(pending)
        self._discharge_obligation(pending, result, head_before)
        self._withdraw()
        return result

    def _hold_promises(self, pending: Pending) -> None:
        """DL-087 #1: note the promises in their own message, and offer (or,
        with the gate dropped, set) a reminder for each that has a due date.

        Runs after the turn's terminal record, so the reply never waits on it
        and nothing here can change it. **Without a live ``promises`` grant it
        calls nothing and writes nothing** — the one exception is a reminder
        set under the grant earlier firing now, which closes its promise.

        The source is the inbound text of a prompted message and nothing else
        (:func:`omega.promises.theirs`); a code pre-filter gates the one LEARN
        call, which fails closed and is never retried. A promise already open
        in the same words is not noted twice. A due one becomes one tell and
        one ``promises`` proposal (the DL-085 gate: their yes is what sets
        it), or, with that gate dropped, the reminder itself, recorded as an
        action with its undo and listed in the next brief. A reminder further
        out than ``remind`` reaches is not offered: the promise stays open and
        the brief names it. Money is never acted on here: the only action is
        ``remind``.

        **Never raises.**
        """
        try:
            store = self._queue.store
            view = self._promises_now.advance(store).state()
            event = pending.payload
            fired = event.get("schedule_id")
            if fired and event.get("kind") == episodes.MESSAGE_INBOUND:
                kept = view.reminding(str(fired))
                if kept is not None:
                    self._queue.append(
                        episodes.promise_closed(promise=kept.seq, how="reminded")
                    )
            if not view.granted or not promises.theirs(event):
                return
            text = str(event["text"])
            if not promises.worth_reading(text):
                return
            at = str(event.get("at") or episodes.now())
            try:
                found = promises.extract(self._completer(), text, at)
            except Exception:  # noqa: BLE001 - fails closed: nothing is noted
                return
            noted: list[promises.Promise] = []
            seen: set[str] = set()
            for item in found:
                key = promises.normalized(item["text"])
                if not key or key in seen or view.same(item["text"]) is not None:
                    continue
                seen.add(key)
                payload = episodes.promise_noted(
                    text=item["text"], due=item["due"], source_seq=pending.seq
                )
                seq = self._queue.append(payload)
                noted.append(promises.Promise(
                    seq=seq, text=payload["text"], due=payload["due"],
                    source_seq=pending.seq, at=payload["at"],
                ))
            self._remind_of(pending.seq, noted)
        except Exception:  # noqa: BLE001 - see the docstring
            return

    def _remind_of(self, for_seq: int, noted: Sequence[promises.Promise]) -> None:
        """The reminders for the promises just noted from message ``for_seq``."""
        now = tools._now().replace(microsecond=0)
        reach = now + timedelta(minutes=tools.MAX_REMIND_MINUTES)
        due = [
            p for p in noted
            if p.due is not None and now < datetime.fromisoformat(p.due) <= reach
        ]
        if not due:
            return
        state = self._grants_now.advance(self._queue.store).state()
        if state.live(grants.PROMISES) is None:
            return
        actions = []
        for p in due:
            when = datetime.fromisoformat(p.due).astimezone()
            actions.append({
                "tool": tools.REMIND,
                "why": f"remind them about {p.text!r} at {tools._due_phrase(when, now)}",
                "args": {
                    "what": p.text, "due": p.due,
                    "id": promises.reminder_id(p.seq), "promise": p.seq,
                },
            })
        if state.gated(grants.PROMISES):
            # Asked, as every action under a grant is until they say stop:
            # one tell naming them all, and one proposal their yes runs.
            tell = self._queue.append(
                episodes.inbound(tools.promise_text(due), channel="schedule")
            )
            self._queue.append(episodes.grant_proposed(
                for_seq=tell, scope=grants.PROMISES, proposes="action", actions=actions,
            ))
            return
        box = tools.ToolBox(
            store_root=self._queue.store.root,
            append=self._queue.append,
            for_seq=for_seq,
            running=tuple(self._standing_schedules()),
            grants=state,
        )
        for action in actions:
            decision = tools.Decision(
                tool=tools.REMIND, tier=tools.LOCAL, why=action["why"], args=action["args"]
            )
            self._queue.append(episodes.tool_called(
                for_seq=for_seq, tool=decision.tool, args=decision.args
            ))
            try:
                result, ok = box.dispatch(decision), True
            except Exception as exc:  # noqa: BLE001 - recorded, as act records it
                result, ok = f"{decision.tool} raised {type(exc).__name__}: {exc}", False
            self._queue.append(episodes.tool_returned(
                for_seq=for_seq, tool=decision.tool, ok=ok,
                result=str(result) if ok else None, error=None if ok else str(result),
            ))
            if ok:
                self._queue.append(episodes.grant_acted(
                    for_seq=for_seq, scope=grants.PROMISES, tool=decision.tool,
                    why=decision.why, args=decision.args,
                    undo=grants.undo_for(decision.tool, decision.args), proposal=None,
                ))

    def _notice_topics(self, pending: Pending) -> None:
        """DL-087 #6: offer to keep up with a subject they keep asking about.

        After the turn's terminal record, as :meth:`_hold_promises`. **Without
        a live ``topics`` grant it calls nothing and writes nothing.** The
        counting is code (:meth:`omega.topics.State.crossing`): only on the
        message where a word first reaches three asking messages on two days
        within a fortnight is one LEARN call made, which names the subject and
        says whether it changes enough to follow; it fails closed and is never
        retried. A subject already offered, followed or dropped is never
        offered again. The offer is one tell and one ``topics`` proposal, or,
        with the gate dropped, the follow itself, recorded with its undo.

        **Never raises.**
        """
        try:
            store = self._queue.store
            view = self._topics_now.advance(store).state()
            if not view.granted or not promises.theirs(pending.payload):
                return
            crossed = view.crossing(pending.seq)
            if not crossed:
                return
            word, texts = crossed[0]
            try:
                named = topics.name_it(self._completer(), word, texts)
            except Exception:  # noqa: BLE001 - fails closed: nothing is offered
                return
            if named is None or not named.follow or view.known(named.name):
                return
            if len(view.followed) >= topics.MAX_FOLLOWED:
                return
            state = self._grants_now.advance(store).state()
            if state.live(grants.TOPICS) is None:
                return
            action = {
                "tool": tools.TOPICS,
                "why": f"keep up with {named.name}",
                "args": {"do": "follow", "name": named.name, "query": named.query, "word": word},
            }
            if state.gated(grants.TOPICS):
                tell = self._queue.append(
                    episodes.inbound(tools.topic_text(named.name), channel="schedule")
                )
                self._queue.append(episodes.grant_proposed(
                    for_seq=tell, scope=grants.TOPICS, proposes="action", actions=[action],
                ))
                return
            box = tools.ToolBox(
                store_root=store.root, append=self._queue.append,
                for_seq=pending.seq, grants=state, topics=view,
            )
            decision = tools.Decision(
                tool=tools.TOPICS, tier=tools.LOCAL, why=action["why"], args=action["args"]
            )
            self._queue.append(episodes.tool_called(
                for_seq=pending.seq, tool=decision.tool, args=decision.args
            ))
            try:
                result, ok = box.dispatch(decision), True
            except Exception as exc:  # noqa: BLE001 - recorded, as act records it
                result, ok = f"{decision.tool} raised {type(exc).__name__}: {exc}", False
            self._queue.append(episodes.tool_returned(
                for_seq=pending.seq, tool=decision.tool, ok=ok,
                result=str(result) if ok else None, error=None if ok else str(result),
            ))
            if ok:
                self._queue.append(episodes.grant_acted(
                    for_seq=pending.seq, scope=grants.TOPICS, tool=decision.tool,
                    why=decision.why, args=decision.args,
                    undo=grants.undo_for(decision.tool, decision.args), proposal=None,
                ))
        except Exception:  # noqa: BLE001 - see the docstring
            return

    def keep_up(self, *, now: Optional[str] = None) -> bool:
        """The daily look at each topic kept up with (DL-087 #6). Returns
        whether a pass ran.

        One topic per call, oldest followed first, and only one that has had
        no pass yet on this local day, while the day's token budget lasts
        (:data:`omega.topics.DAILY_TOKENS`). With nothing followed it is a
        fold and nothing else: no model call, no search, no record. Each pass
        writes one ``topic.update`` with only the items new to that topic and
        what it spent; the next brief lists them. A pass that failed is
        written down with why, and counts as that day's pass.

        **Never raises.**
        """
        if not self._recovered:
            return False
        try:
            view = self._topics_now.advance(self._queue.store).state()
            if not view.followed:
                return False
            clock = now or episodes.now()
            day = topics._local_day(clock)
            if not view.budget_left(day):
                return False
            due = [f for f in view.oldest() if not view.passed(f.key, day)]
            if not due:
                return False
            topic = due[0]
            meter = provider.Meter(self._completer())
            try:
                found = topics.research(meter, topic.name, topic.query, remote=mcp.remote())
            except Exception as exc:  # noqa: BLE001 - written down, never raised
                reason = (str(exc) or type(exc).__name__)[: episodes.MAX_FAILURE_ERROR]
                self._queue.append(episodes.topic_update(
                    name=topic.name, items=[], reason=reason, usage=meter.usage(), at=clock,
                ))
                return True
            self._queue.append(episodes.topic_update(
                name=topic.name, items=view.fresh(topic.key, found),
                usage=meter.usage(), at=clock,
            ))
            return True
        except Exception:  # noqa: BLE001 - see the docstring
            return False

    def _discharge_obligation(
        self, pending: Pending, result: Optional[TurnResult], head_before: int
    ) -> None:
        """DL-095/DL-040 gate-leak. A turn that carried a task-step obligation
        and ended in prose — no terminal ``step`` and no gated call it was
        blocked on — did not do the step. Record it ``task.dropped``, restate
        the step once, and on a second drop block it and tell the person once.

        Structural, read from the records only: it never inspects the turn's
        words, so it covers a bare schedule fire exactly when a scheduling path
        attached an obligation to it. After the terminal record, as
        :meth:`_hold_promises`. A no-op with the engine off. **Never raises.**
        """
        if not tools.tasks_on():
            return
        try:
            ob = tasks.obligation_of(pending.payload)
            if ob is None:
                return
            task, n = ob["task"], ob["n"]
            # Blocked on a gated/external "May I?" is work that owed a gated
            # action and asked for it — not a leak.
            if result is not None and result.outcome == "blocked":
                return
            # A terminal step, or a proposal/act under this event, discharges it.
            for episode in self._queue.store.episodes_since(head_before):
                payload = episodes.decode(episode.payload)
                if payload.get("for_seq") != pending.seq:
                    continue
                if payload.get("kind") in (
                    episodes.TASK_STEP, episodes.GRANT_PROPOSED, episodes.GRANT_ACTED
                ):
                    return
            # A drop. Record it, then restate once or block on the second.
            self._queue.append(episodes.task_dropped(for_seq=pending.seq, task=task, n=n))
            view = self._tasks.advance(self._queue.store).state()
            live = view.task(task)
            if live is None or n >= live.total:
                return
            clock = str(pending.payload.get("at")) or None
            step = live.plan.steps[n]
            if live.drops.get(n, 0) >= 2:
                why = "the step ended twice with nothing recorded about how it finished"
                self._queue.append(episodes.task_step(
                    for_seq=pending.seq, task=task, n=n, state="blocked",
                    why=why, at=clock,
                ))
                self._queue.append(episodes.inbound(
                    tools.task_blocked_text(live.plan.goal, n, why),
                    channel="schedule", at=clock,
                ))
            else:
                self._queue.append(episodes.inbound(
                    tools.task_step_text(
                        live.plan.goal, n, live.total, step["do"], step["done"],
                        retry=True,
                    ),
                    channel="task", obligation={"task": task, "n": n}, at=clock,
                ))
        except Exception:  # noqa: BLE001 - see the docstring
            return

    def _block_step(self, live: Any, n: int, clock: Optional[str], why: str) -> None:
        """Stop a task at step ``n`` on a cap and tell the person once (DL-095).
        The block has no turn of its own, so it is attributed to the task's own
        ``task.proposed`` seq."""
        self._queue.append(episodes.task_step(
            for_seq=live.plan.task, task=live.task, n=n, state="blocked",
            why=why, at=clock,
        ))
        self._queue.append(episodes.inbound(
            tools.task_blocked_text(live.plan.goal, n, why),
            channel="schedule", at=clock,
        ))

    def drive_tasks(self, *, now: Optional[str] = None) -> bool:
        """Drive one task step as its own unattended turn (DL-095). Returns
        whether a step was driven or a cap enforced.

        One step per call, oldest task first. A waiting step does not resume on
        its own; a step whose task has spent its token budget, or that would be
        this day's step past the per-day cap, is blocked and the person told
        once. Off by default and a fold otherwise — with the engine off, or no
        task drivable, it writes nothing. **Never raises.**
        """
        if not tools.tasks_on() or not self._recovered:
            return False
        try:
            view = self._tasks_now.advance(self._queue.store).state()
            clock = now or episodes.now()
            day = tasks._local_day(clock)
            # A running task that has spent its budget before its next step can
            # run is stopped first (``drivable`` already excludes it).
            for live in view.running():
                if live.current is None and live.cursor < live.total \
                        and live.spent >= live.plan.budget:
                    self._block_step(
                        live, live.cursor, clock,
                        f"the task's token budget of {live.plan.budget} is spent",
                    )
                    return True
            nxt = view.drivable()
            if nxt is None:
                return False
            task, n, step = nxt
            live = view.task(task)
            if view.steps_today(day) >= tasks.DAILY_STEPS:
                self._block_step(
                    live, n, clock,
                    f"the daily limit of {tasks.DAILY_STEPS} task steps is reached",
                )
                return True
            self._queue.append(episodes.inbound(
                tools.task_step_text(live.plan.goal, n, live.total, step["do"], step["done"]),
                channel="task", obligation={"task": task, "n": n}, at=clock,
            ))
            return True
        except Exception:  # noqa: BLE001 - see the docstring
            return False

    def watch_pages(self, *, now: Optional[str] = None) -> bool:
        """One due page-watch check (DL-096). Returns whether one ran.

        Expiry first: a watch past its ``until`` is stopped and they are told
        once. Then the watch most overdue is checked: a read and a comparison
        in code (:func:`omega.watcher.check`), with no model call unless it
        alerts and was set to ``summarize``. An alert is one tell. A watch
        whose MCP server lost its credentials waits until access is restored.
        With no watch standing it is a fold and nothing else.

        **Never raises.**
        """
        if not self._recovered:
            return False
        try:
            view = self._watches.advance(self._queue.store).state()
            if not view:
                return False
            clock = now or episodes.now()
            moment = datetime.fromisoformat(clock)
            for w in view.oldest():
                if watcher.expired(w, moment):
                    self._queue.append(episodes.watcher_stopped(id=w.id, reason="expired", at=clock))
                    self._tell(watcher.expiry_words(w), clock)
                    return True
            lost = self._access_now.advance(self._queue.store).state().lost
            due = [
                w for w in view.oldest()
                if watcher.due(w, moment)
                and access.mcp_origin(watcher.server_of(w.adapter) or "") not in lost
            ]
            if not due:
                return False
            w = min(due, key=lambda x: (x.checked_at or "", x.since, x.id))
            remote = mcp.remote() if w.adapter["kind"] == "mcp" else None
            done = watcher.check(w, remote=remote, at=clock, complete=self._complete)
            self._queue.append(done.checked)
            if done.alert is not None:
                # The page's words, so the tell names whose (DL-064).
                self._tell(done.alert["text"], clock, untrusted=watcher.source_of(w))
                self._queue.append(done.alert)
            elif not done.ok:
                failing = watcher.failing_alert(
                    self._watches.advance(self._queue.store).state().watches[w.id], clock
                )
                if failing is not None:
                    # Carries the read's error, which the server wrote.
                    self._tell(failing["text"], clock, untrusted=watcher.source_of(w))
                    self._queue.append(failing)
            return True
        except Exception:  # noqa: BLE001 - see the docstring
            return False

    def _tell(self, words: str, at: str, *, untrusted: Optional[str] = None) -> None:
        self._queue.append(episodes.inbound(
            tools.tell_instruction(words), channel="schedule", at=at,
            untrusted=untrusted,
        ))

    def leave_now(self, *, now: Optional[str] = None) -> bool:
        """The leave-now alarms (DL-093), on every sense pass. Returns whether
        anything was written.

        Code and nothing else: no model call. For each open alarm,
        :func:`omega.leave.due` says what is due and this does it: one Maps
        ``compute_routes`` call written down as ``leave.checked``, the ping
        and its ``leave.pinged`` when that check says it is time, the second
        ping when traffic got worse, the one estimate tell when traffic could
        not be read, and ``leave.done`` once ``arrive_by`` has passed. Each
        tell is appended before the record that it went, so a crash between
        the two repeats a ping rather than losing one. With no alarm set it is
        a fold and nothing else: no call, no record.

        **Never raises.**
        """
        if not self._recovered:
            return False
        try:
            view = self._leaves.advance(self._queue.store).state()
            alarms = view.open()
            if not alarms:
                return False
            moment = datetime.fromisoformat(now or episodes.now())
            if moment.tzinfo is None:
                moment = moment.replace(tzinfo=timezone.utc)
            wrote = False
            for alarm in alarms:
                try:
                    wrote = self._leave_one(alarm, moment) or wrote
                except Exception:  # noqa: BLE001 - one alarm never stops another
                    continue
            return wrote
        except Exception:  # noqa: BLE001 - see the docstring
            return False

    def _leave_one(self, alarm: "leave.Alarm", moment: datetime) -> bool:
        at = moment.isoformat()
        what = leave.due(alarm, moment)
        if what is None:
            return False
        if what == "done":
            self._queue.append(episodes.leave_done(id=alarm.id, at=at))
            self._leaves.advance(self._queue.store)
            return True
        if what == "ping":
            # The call cap is spent: the last drive time Maps gave stands.
            self._leave_tell(alarm, "live", alarm.estimate, leave.ping_text(
                alarm, alarm.estimate, moment), at)
            return True
        remote = mcp.remote()
        lost = access.mcp_origin(leave.SERVER) in self._access_now.advance(
            self._queue.store).state().lost
        if remote is None or not remote.knows(leave.FULL) or lost:
            # Not configured, or its key refused and not yet working again:
            # no call is spent, and they are told once, by the estimate.
            if alarm.pings:
                return False
            self._leave_tell(alarm, "estimate", alarm.estimate, leave.estimate_text(alarm), at)
            return True
        error: Optional[str] = None
        eta: Optional[int] = None
        try:
            decision = remote.classify(leave.FULL, leave.route_args(alarm))
            if decision is None or not decision.dispatches:
                raise tools.ToolError(f"the {leave.SERVER} MCP server would not run {leave.TOOL}")
            answer = remote.dispatch(decision)
            eta = leave.eta_minutes(str(getattr(answer, "log", answer)))
            if eta is None:
                error = f"no drive time in the {leave.SERVER} MCP server's answer"
        except Exception as exc:  # noqa: BLE001 - written down, then said
            # The adapter's own first line only (DL-064): past it an isError
            # answer carries the server's words, which no record keeps.
            error = ((str(exc) or type(exc).__name__).split("\n", 1)[0].strip()
                     or type(exc).__name__)[: episodes.MAX_FAILURE_ERROR]
        self._queue.append(episodes.leave_checked(id=alarm.id, eta_min=eta, error=error, at=at))
        alarm = self._leaves.advance(self._queue.store).state().alarms[alarm.id]
        then = leave.after_check(alarm, moment)
        if then == "ping":
            self._leave_tell(alarm, "live", eta, leave.ping_text(alarm, eta, moment), at)  # type: ignore[arg-type]
        elif then == "again":
            self._leave_tell(alarm, "again", eta, leave.ping_text(  # type: ignore[arg-type]
                alarm, eta, moment, again=True), at)  # type: ignore[arg-type]
        elif then == "estimate":
            self._leave_tell(alarm, "estimate", alarm.estimate, leave.estimate_text(alarm), at)
        return True

    def _leave_tell(self, alarm: "leave.Alarm", how: str, eta: int, text: str, at: str) -> None:
        self._queue.append(episodes.inbound(
            tools.tell_instruction(text), channel="schedule", at=at,
        ))
        self._queue.append(episodes.leave_pinged(id=alarm.id, how=how, eta_min=eta, at=at))
        self._leaves.advance(self._queue.store)

    def _withdraw(self) -> None:
        """Write down every proposal that stopped applying (DL-087 #8): its
        grant taken back, its gate put back, its schedule stopped or changed,
        its research finished. After the turn, because the turn is what most
        often makes one lapse (a revoke, a cancel). Writes nothing while no
        proposal is pending, so a log without them gains no records."""
        self._head_grants.advance(self._queue.store)
        grants.withdraw_lapsed(self._queue.append, self._head_grants.state())

    def _standing_schedules(self) -> list[Schedule]:
        self._standing.refresh()
        # The self-review (DL-086) is omega's own, not one of theirs: never
        # listed, unwatched or tidied, and never in a look or a teach.
        return [s for s in self._standing.schedules if s.id != lessons.REVIEW_ID]


    def __repr__(self) -> str:
        return f"<Executor {self._queue!r} recovered={self._recovered}>"
