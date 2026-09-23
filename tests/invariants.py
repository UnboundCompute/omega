"""M1's violation metric, as a checker — `agent/M1_SPEC.md` §"The M1 violation
metric".

The capability is *turns complete and land in the log*. The violation metric
that must not regress is:

- every event episode at seq <= ``DONE`` has **exactly one** terminal record
  naming it — ``turn.completed`` or ``turn.blocked``, never both and never two;
- ``claimed - done`` is never greater than 1 — more than one turn in flight
  means the queue stopped being single-consumer;
- an event at seq <= ``claimed`` with no terminal record is the interrupted turn
  and is the *only* one, so the startup report can name it.

**It is three-valued and it fails closed.** A log in which nothing was ever
processed reports ``undetermined``, not ``pass``. M0's own crash suite hit
exactly that bug — 2 of 20 trials killed the child during interpreter startup —
and a pass count would have read 20/20.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from omega import episodes
from omega.queue import EVENT_KINDS, EventQueue

PASS = "pass"
VIOLATED = "violated"
UNDETERMINED = "undetermined"


@dataclass(frozen=True)
class Check:
    verdict: str
    reasons: list[str] = field(default_factory=list)
    events: int = 0
    terminals: int = 0
    claimed: int = 0
    done: int = 0
    head: int = 0

    def __bool__(self) -> bool:
        """Only a real pass is truthy. ``undetermined`` is not a pass, and a
        checker that let it read as one would be the thing it exists to stop."""
        return self.verdict == PASS

    def __str__(self) -> str:
        return (
            f"{self.verdict} (head={self.head} claimed={self.claimed} "
            f"done={self.done} events={self.events} terminals={self.terminals})"
            + ("".join(f"\n  - {r}" for r in self.reasons) if self.reasons else "")
        )


def check(queue: EventQueue) -> Check:
    """Read the whole log and grade it against the violation metric."""
    head = queue.head()
    claimed = queue.claimed()
    done = queue.done()
    reasons: list[str] = []

    event_seqs: list[int] = []
    terminals_by_for_seq: dict[int, list[int]] = {}

    if head:
        for pending in queue.recent(head):
            if pending.kind in EVENT_KINDS:
                event_seqs.append(pending.seq)
            if episodes.is_terminal(pending.payload):
                terminals_by_for_seq.setdefault(
                    int(pending.payload["for_seq"]), []
                ).append(pending.seq)

    terminals = sum(len(v) for v in terminals_by_for_seq.values())

    # --- rule 2: at most one turn in flight -------------------------------
    if claimed - done > 1:
        reasons.append(
            f"claimed ({claimed}) is {claimed - done} ahead of done ({done}); "
            f"more than one turn was in flight"
        )
    if done > claimed:
        reasons.append(f"done ({done}) is ahead of claimed ({claimed})")

    # --- rule 1: exactly one terminal record per settled event ------------
    for seq in event_seqs:
        if seq > done:
            continue
        found = terminals_by_for_seq.get(seq, [])
        if len(found) != 1:
            reasons.append(
                f"event {seq} is at or below done ({done}) but has "
                f"{len(found)} terminal records {found}; expected exactly 1"
            )

    # --- rule 3: the unrecorded event is the interrupted one, and only it --
    unrecorded = [
        seq
        for seq in event_seqs
        if seq <= claimed and not terminals_by_for_seq.get(seq)
    ]
    if unrecorded and unrecorded != [claimed]:
        reasons.append(
            f"events {unrecorded} were claimed with no terminal record; only the "
            f"in-flight turn ({claimed}) may be in that state"
        )

    # A terminal record naming an episode that is not an event at all.
    for for_seq in terminals_by_for_seq:
        if for_seq not in event_seqs:
            reasons.append(
                f"a terminal record names seq {for_seq}, which is not an event"
            )

    if reasons:
        verdict = VIOLATED
    elif not event_seqs or done == 0:
        # Fail closed: nothing was processed, so nothing was proved.
        verdict = UNDETERMINED
        reasons.append(
            f"nothing to grade: {len(event_seqs)} events, done={done}. "
            f"This is not a pass."
        )
    else:
        verdict = PASS

    return Check(
        verdict=verdict,
        reasons=reasons,
        events=len(event_seqs),
        terminals=terminals,
        claimed=claimed,
        done=done,
        head=head,
    )
