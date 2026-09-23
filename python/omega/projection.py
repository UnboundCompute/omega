"""The outward projection of the episode stream — M1 step 5, spec §Q10.

The transport is **not a second protocol**. The tray's five durable work states
and M1's episode kinds are the same alphabet, because both came from the same
human test, so what goes out on the wire is a *filter* over what the loop
already wrote:

    understood         <- message.inbound            (a durable fact, not a guess)
    working            <- tool.called / tool.returned / work.finished
    blocked            <- turn.blocked
    verified complete  <- turn.completed{spoke|silent}
    failed             <- turn.completed{failed}

Two consequences worth naming, because they are the reason this is a projection
and not a status API:

**Progress describes real state.** Nothing here can say "working" unless a
``tool.called`` episode is in the log. A UI fed by this cannot invent a
reassuring status around an ``await``; it can only display what happened.

**Filtering is policy, and policy lives in Python** (DL-018). The tray has no
business seeing tool arguments or tool results: those are the parts of a turn
most likely to carry secrets and untrusted output (DL-014), and they are the
parts a UI has no use for. So the projection carries the *fact* of a tool call
and its name, never its payload. That rule is enforced by
``test_projection.py``, not by care.

DL-011 lands here too: ``silent`` and ``failed`` are different states on the
wire, and ``spoke`` and ``silent`` are the same *state* with different
``outcome`` and a null ``reply``. A model that chose not to speak and a turn
that broke must never render the same way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

from omega import episodes
from omega.queue import EventQueue, Pending

__all__ = [
    "VERSION",
    "UNDERSTOOD",
    "WORKING",
    "BLOCKED",
    "FAILED",
    "COMPLETE",
    "STATES",
    "Update",
    "project",
    "project_pending",
    "updates_since",
]

#: The wire version, versioned separately from the episode schema. They change
#: for different reasons — the storage shape is ours, the wire shape is shared
#: with a client we do not deploy — and tying them together would force a client
#: update for a storage-only change.
VERSION = 1

UNDERSTOOD = "understood"
WORKING = "working"
BLOCKED = "blocked"
FAILED = "failed"
COMPLETE = "complete"

#: The tray's five durable work states, and that is all of them. A sixth state
#: is a design change, not an implementation detail.
STATES = frozenset({UNDERSTOOD, WORKING, BLOCKED, FAILED, COMPLETE})


@dataclass(frozen=True)
class Update:
    """One line on the wire.

    ``seq`` is the episode that produced it, which doubles as the client's
    resume cursor: a client that reconnects saying ``since=seq`` gets exactly
    what it missed, because the log is the only place the stream lives.

    ``for_seq`` is the *turn* the update is about — for an inbound message that
    is its own seq, and for everything else it is the event that caused the
    turn. A client groups by ``for_seq`` and never has to keep a mapping table
    across a restart of either side.
    """

    seq: int
    state: str
    for_seq: int
    at: str
    kind: str
    text: Optional[str] = None
    reply: Optional[str] = None
    needs: Optional[str] = None
    error: Optional[str] = None
    tool: Optional[str] = None
    ok: Optional[bool] = None
    outcome: Optional[str] = None
    urgency: Optional[str] = None
    context: list[dict[str, str]] = field(default_factory=list)

    def wire(self) -> dict[str, Any]:
        """The JSON object a client receives. Optional fields are dropped
        rather than sent as nulls — except ``reply``, which is *meaningfully*
        null on a silent turn and would otherwise be indistinguishable from a
        turn that spoke nothing."""
        out: dict[str, Any] = {
            "v": VERSION,
            "op": "update",
            "seq": self.seq,
            "state": self.state,
            "for_seq": self.for_seq,
            "at": self.at,
            "kind": self.kind,
        }
        if self.outcome is not None:
            out["outcome"] = self.outcome
            out["reply"] = self.reply
        for name in ("text", "needs", "error", "tool", "urgency"):
            value = getattr(self, name)
            if value is not None:
                out[name] = value
        if self.ok is not None:
            out["ok"] = self.ok
        if self.context:
            out["context"] = self.context
        return out


#: Kinds that deliberately do not reach the wire.
#:
#: Named rather than left to fall off the end of :func:`project`, because the
#: difference between "we decided this is not a turn update" and "we forgot to
#: handle it" is invisible at the call site — both are ``None`` — and the second
#: one silently removes a kind from every UI.
#:
#: The schedule definitions are here because they are not about a turn: writing
#: down "brief me every morning" changes no work state, and DL-035 keeps the
#: *fire* visible instead. A fire is a `message.inbound`, so it projects as
#: ``understood`` like any other arriving message — which is what makes omega
#: acting on its own show up in the tray rather than happening invisibly.
#:
#: `claim.extracted` is here for a different reason, and it is not "learning is
#: invisible". DL-042 makes the *receipt* — what omega now believes and when it
#: will fire — part of the turn's reply, which already projects. Giving claims
#: their own wire update on top of that would be a second way for one turn to
#: reach the person, which `act.py` §1.6 refuses for the agent's output and
#: which would be no better arriving through the projection. One turn, one
#: outward voice; a turn that learns four things says so once.
NOT_PROJECTED = frozenset(
    {
        episodes.SCHEDULE_CREATED,
        episodes.SCHEDULE_CANCELLED,
        episodes.CLAIM_EXTRACTED,
        episodes.CLAIM_RETRACTED,
    }
)


def project(payload: dict[str, Any], seq: int) -> Optional[Update]:
    """One episode to one wire update, or ``None`` if policy says it stays in.

    ``None`` is a real answer and not an error: the filter exists precisely so
    that the outward stream is narrower than the log. What withholds today is
    :data:`NOT_PROJECTED`, and the caller must handle ``None`` regardless — a
    projection that could never withhold anything would not be a filter.
    """
    kind = payload.get("kind")
    at = str(payload.get("at", ""))

    if kind in NOT_PROJECTED:
        return None

    if kind == episodes.MESSAGE_INBOUND:
        return Update(
            seq=seq,
            state=UNDERSTOOD,
            for_seq=seq,
            at=at,
            kind=kind,
            text=str(payload.get("text", "")),
            urgency=str(payload.get("urgency", "normal")),
            context=_context(payload.get("context") or []),
        )

    if kind == episodes.TURN_BLOCKED:
        return Update(
            seq=seq,
            state=BLOCKED,
            for_seq=int(payload["for_seq"]),
            at=at,
            kind=kind,
            needs=str(payload.get("needs", "")),
        )

    if kind == episodes.TURN_COMPLETED:
        outcome = str(payload.get("outcome", ""))
        return Update(
            seq=seq,
            state=FAILED if outcome == "failed" else COMPLETE,
            for_seq=int(payload["for_seq"]),
            at=at,
            kind=kind,
            outcome=outcome,
            # Null on a silent turn, and it stays null: "chose not to speak" is
            # a success and must never arrive looking like an empty string.
            reply=payload.get("reply"),
            error=payload.get("error"),
        )

    if kind in (episodes.TOOL_CALLED, episodes.TOOL_RETURNED):
        # The name of the tool crosses; its arguments and its result do not.
        # Those are the untrusted, secret-bearing halves (DL-014) and a UI has
        # no use for them.
        return Update(
            seq=seq,
            state=WORKING,
            for_seq=int(payload["for_seq"]),
            at=at,
            kind=kind,
            tool=str(payload.get("tool", "")),
            ok=payload.get("ok") if kind == episodes.TOOL_RETURNED else None,
        )

    if kind == episodes.WORK_FINISHED:
        # Detached work re-entering (§1.5). It is still *work*, not an outcome:
        # the turn it wakes is what decides whether anything gets said.
        return Update(
            seq=seq,
            state=WORKING,
            for_seq=int(payload["for_seq"]),
            at=at,
            kind=kind,
            text=str(payload.get("summary", "")),
            ok=bool(payload.get("ok", True)),
        )

    return None


def project_pending(pending: Pending) -> Optional[Update]:
    return project(pending.payload, pending.seq)


def updates_since(queue: EventQueue, since: int) -> Iterator[Update]:
    """Every update after ``since``, oldest first.

    Reads through the queue, which reads through the memory seam. There is no
    projection cache and no in-RAM copy of the stream: a client's cursor is a
    number it sends, and the log answers it (DL-016 — zero authoritative state
    in RAM).
    """
    if since < 0:
        raise ValueError(f"since must be >= 0, got {since}")
    head = queue.head()
    if since > head:
        # Eager, like the seam's own ``episodes_since``: a cursor past the end
        # of the log is a client that has state we do not, and answering it
        # with an empty stream would hide that until something else broke.
        raise ValueError(f"since {since} is ahead of head {head}")
    if since == head:
        return
    # ``recent(n)`` means *the last n*, which is a window that moves when the
    # log grows. The executor drains on another thread, so between the head read
    # above and the one inside ``recent`` an append can land — and a relative
    # window would then slide forward by exactly that much and skip the oldest
    # episodes the caller asked for. Pinning the far end with ``before`` makes
    # the window absolute: seqs ``since+1 .. head``, whatever arrives meanwhile.
    for pending in queue.recent(head - since, before=head + 1):
        if pending.seq <= since:
            continue
        update = project(pending.payload, pending.seq)
        if update is not None:
            yield update


def _context(items: list[Any]) -> list[dict[str, str]]:
    """Ids and kinds only.

    §2.1 splits the responsibility explicitly — *the tray keeps the previews;
    the log keeps the identities* — so echoing titles back would be the wrong
    half of the split, and would put user-visible text on the wire for no
    reader. The id is what has to survive a ``kill -9`` on either side, and it
    does, because it lives in the payload.
    """
    out: list[dict[str, str]] = []
    for item in items:
        if isinstance(item, dict) and "id" in item:
            out.append({"id": str(item["id"]), "kind": str(item.get("kind", ""))})
    return out
