"""The queue — M1 step 2, spec `agent/M1_SPEC.md` §1.1 and §1.2.

**The queue is the log, and the cursor is a checkpoint.** That result is
*derived*, not chosen: DL-016 fixes zero authoritative state in RAM, an
in-memory queue holds authoritative state (events accepted but not yet
processed), and the restart test says a ``kill -9`` loses only the turn that was
mid-flight. So the queue has to be durable, and the only durable thing there is
is the log::

    enqueue(event)   = store.append_episode(payload, write_key=...)
    pending()        = store.episodes_since(store.checkpoint(CLAIMED))
    claim(episode)   = store.set_checkpoint(CLAIMED, episode.seq)
    finish(episode)  = store.set_checkpoint(DONE,    episode.seq)

**Two cursors, not one** (§1.2). ``CLAIMED`` advances *before* the turn runs and
``DONE`` after its terminal record lands, so a crash is an integer comparison
rather than a search:

    claimed > done   ->  the episode at seq `claimed` was interrupted
    claimed == done  ->  nothing was in flight; a clean stop

Claim-before-act makes the loop **at-most-once**. The alternative — checkpoint
on completion — re-delivers the event after a crash and runs the turn again,
which for a step that calls tools with side effects is the standard way one
action becomes two.

No queue library, no second store, no new infrastructure. What this file adds on
top of M0 is the *discipline*: cursors only move forwards, the drain advances
past records it does not process, and the episode a cursor names is decoded
before anyone acts on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Optional

from omega import episodes
from omega.memory import Episode, MemoryStore

__all__ = [
    "CLAIMED",
    "DONE",
    "EVENT_KINDS",
    "RECORD_KINDS",
    "Pending",
    "CursorWentBackwards",
    "EventQueue",
]

#: The two named checkpoints (§1.1). M0 supports *named* checkpoints and
#: ``checkpoint_names()`` precisely so there can be several; this is the first
#: real use, and it is what turns M0's checkpoint API from speculative into
#: load-bearing.
CLAIMED = "executor.claimed"
DONE = "executor.done"

#: Kinds that *cause* a turn. Everything else in the log is a record the loop
#: wrote about a turn, and the drain must advance past those without processing
#: them — otherwise the executor treats its own output as a new event and loops
#: forever (§1.1, "the one genuinely sharp edge in reusing the log as the
#: queue").
#:
#: ``work.finished`` is an event and not a record: a detached sub-loop
#: re-enters as an *ordinary enqueued event* (§1.5, DL-016), which is exactly
#: what stops long work from becoming a second entry path into the loop.
EVENT_KINDS = frozenset({episodes.MESSAGE_INBOUND, episodes.WORK_FINISHED})

#: The complement. Derived rather than written out, so a new kind cannot be
#: added to the codec and silently land in neither set.
RECORD_KINDS = frozenset(episodes.KINDS) - EVENT_KINDS

# A kind that is in neither set would be skipped by the drain *and* not be a
# record -- i.e. an event that is never processed and never reported. Partition
# the space loudly at import time rather than discover it as a wedged queue.
assert EVENT_KINDS | RECORD_KINDS == frozenset(episodes.KINDS)
assert not (EVENT_KINDS & RECORD_KINDS)


class CursorWentBackwards(RuntimeError):
    """An attempt to move ``CLAIMED`` or ``DONE`` to a lower sequence.

    Refused rather than obeyed. A cursor moving backwards re-delivers episodes
    that were already claimed, which breaks at-most-once — the single property
    §1.2 gives up at-least-once to buy. M0 permits the move (a checkpoint is
    just a number there); the queue is where it becomes a bug.
    """


@dataclass(frozen=True)
class Pending:
    """One episode waiting at the cursor, already decoded.

    Decoded *here* so that no caller ever has to hold raw bytes to decide
    whether an episode is its business. ``payload`` is a validated episode dict,
    never ``None``: :func:`omega.episodes.decode` raises on anything it cannot
    vouch for, and a queue that answered ``None`` for an unreadable episode
    would make "nothing pending" and "something pending we cannot read" the same
    observation.
    """

    seq: int
    payload: dict[str, Any]
    write_key: str
    ts_micros: int

    @property
    def kind(self) -> str:
        return self.payload["kind"]

    @property
    def is_event(self) -> bool:
        """Does this episode cause a turn, or is it something the loop wrote?"""
        return self.kind in EVENT_KINDS


class EventQueue:
    """The single-consumer queue, expressed as two cursors over one log.

    Serial by construction: there is one drain, the log is totally ordered by
    ``seq``, and both cursors are monotonic. DL-016 predicted that at-most-once
    and ordering "fall out free" — they fall out of exactly those three facts,
    and nothing here adds a fourth.
    """

    __slots__ = ("_store",)

    def __init__(self, store: MemoryStore) -> None:
        self._store = store

    @property
    def store(self) -> MemoryStore:
        """The store underneath. Read-only: the queue owns the cursors, and a
        second writer moving them is the two-consumers bug the whole design is
        built to exclude."""
        return self._store

    # --- writing ----------------------------------------------------------

    def append(
        self,
        payload: dict[str, Any],
        write_key: str = "",
        ts_micros: Optional[int] = None,
    ) -> int:
        """§1.1's ``enqueue``. Encode, append, return the sequence number.

        **One verb for events and records alike**, because §1.1's whole result
        is that they go to the same place. An inbound message, a tool call and a
        ``turn.completed`` are all appends; what differs is only whether the
        drain processes what it finds (:data:`EVENT_KINDS`) or advances past it.

        The returned ``seq`` is the acknowledgement token the tray needs
        (§Q10, requirement 1) — it is durable before this function returns, so
        acknowledging on it is a promise the log has already kept.

        ``write_key`` is M0's dedup: re-appending the same key with the same
        payload returns the original seq, and with a *different* payload raises
        ``WriteKeyConflict``. The loop uses that as a live invariant check on
        its most dangerous bug — see :func:`omega.episodes.turn_write_key`.
        """
        return self._store.append_episode(
            episodes.encode(payload), write_key, ts_micros
        )

    # --- reading ----------------------------------------------------------

    def pending(self) -> Iterator[Pending]:
        """Everything after the ``CLAIMED`` cursor, in order, decoded.

        A generator on purpose: the drain claims and processes one episode at a
        time, and materialising the whole tail would mean holding in RAM a list
        the cursors already describe durably.

        Reading from a position *ahead* of the log raises ``CheckpointAhead``
        from the seam rather than returning empty. That propagates deliberately:
        a claimed cursor past the head means the log lost episodes the executor
        had already taken responsibility for, and reporting it as "nothing to
        do" is exactly the empty-passes-as-success failure `CLAUDE.md` bans.
        """
        for episode in self._store.episodes_since(self.claimed()):
            yield self._decode(episode)

    def at(self, seq: int) -> Pending:
        """The episode with ``seq``, decoded. Used by the startup report to say
        *what* was interrupted rather than only that something was."""
        if seq < 1:
            raise ValueError(f"seq must be positive, got {seq}")
        for episode in self._store.episodes_since(seq - 1):
            if episode.seq == seq:
                return self._decode(episode)
            break
        raise LookupError(f"no episode at seq {seq}")

    def recent(self, n: int, *, before: Optional[int] = None) -> list[Pending]:
        """The last ``n`` episodes, oldest first, optionally strictly before a
        seq. This is what ``recall`` reads (§2.2) — through ``episodes_since``,
        never through the diagnostics hatch.
        """
        if n < 0:
            raise ValueError(f"n must not be negative, got {n}")
        last = self._store.head() if before is None else min(before - 1, self._store.head())
        if n == 0 or last < 1:
            return []
        start = max(0, last - n)
        out = []
        for episode in self._store.episodes_since(start):
            if episode.seq > last:
                break
            out.append(self._decode(episode))
        return out

    def head(self) -> int:
        return self._store.head()

    # --- cursors ----------------------------------------------------------

    def claimed(self) -> int:
        return self._store.checkpoint(CLAIMED)

    def done(self) -> int:
        return self._store.checkpoint(DONE)

    def claim(self, seq: int) -> None:
        """Take responsibility for ``seq`` *before* doing anything about it.

        The whole of at-most-once is in the word "before". After this returns,
        a ``kill -9`` leaves ``claimed > done`` and the turn is reported at
        startup rather than re-run (§1.2).
        """
        self._advance(CLAIMED, seq)

    def finish(self, seq: int) -> None:
        """Release ``seq``. Called only *after* its terminal record is durable.

        The order matters and it is the safe direction of the error (§1.2):
        crashing between the record and this call over-reports a cut-off turn
        that had in fact finished, and the report is a question to the user, not
        an action. The reverse order would lose turns silently.
        """
        if seq > self.claimed():
            # DONE passing CLAIMED means something was released that was never
            # taken -- the cursor pair would then say "nothing in flight" while
            # a turn was running.
            raise CursorWentBackwards(
                f"cannot finish seq {seq} past claimed {self.claimed()}"
            )
        self._advance(DONE, seq)

    def skip(self, seq: int) -> None:
        """Advance both cursors past an episode the drain does not process.

        §1.1's three-line guard, spelled as one call so the two cursors cannot
        drift apart at a call site. Records the loop wrote — ``turn.completed``,
        ``tool.*`` — appear in ``pending()`` like anything else, and without
        this the executor would answer its own output forever.
        """
        self._advance(CLAIMED, seq)
        self._advance(DONE, seq)

    def in_flight(self) -> Optional[int]:
        """The seq of the interrupted turn, or ``None`` when nothing was in
        flight (§1.2). No scan, no scan window to get wrong, and it cannot be
        fooled by a later record landing between the two cursors."""
        claimed, done = self.claimed(), self.done()
        return claimed if claimed > done else None

    def _advance(self, name: str, seq: int) -> None:
        current = self._store.checkpoint(name)
        if seq < current:
            raise CursorWentBackwards(
                f"{name} is at {current}; refusing to move it back to {seq}"
            )
        if seq == current:
            return
        self._store.set_checkpoint(name, seq)

    @staticmethod
    def _decode(episode: Episode) -> Pending:
        return Pending(
            seq=episode.seq,
            payload=episodes.decode(episode.payload),
            write_key=episode.write_key,
            ts_micros=episode.ts_micros,
        )

    def __repr__(self) -> str:
        return (
            f"<EventQueue head={self.head()} "
            f"claimed={self.claimed()} done={self.done()}>"
        )
