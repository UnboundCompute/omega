"""Where each unprompted mail read left off — the mail cursor. DL-090.

An hourly sweep of their inbox that re-reads the whole day every time hands
the turn the same mails every hour, and only a sentence in the watch's fire
text ("tell them only what is new since you last told them") stood between a
mail and a repeated alert. That sentence is a request to a model; this module
is the same rule in code.

**Unprompted reads only.** A turn their own message started reads mail exactly
as it always did ("did Kaushik write back?" searches days, not since the last
sweep). Every other turn — a watch firing, a brief, omega's own look — gets
back only mail that *arrived* (IMAP INTERNALDATE, set by the server, not the
sender's forgeable Date header) after the previous successful read made from
the same **origin**.

**Per origin, not one global cursor.** The origin of a turn is what caused it:
one schedule (a watch or a brief, by its id), omega's own look, or anything
else unprompted by its channel. A global cursor would let one path's read
swallow mail for another: the look reads a mail and stays quiet about it, and
a path that reads mail for a different purpose then never sees it. Each path
stops repeating *itself*. (The look does not stay quiet on the assumption
that some watch covers its mail: since DL-107 the look and thinking own the
mail reads, and a standing mail watch is offered for retirement.)

**Derived from the log, so it survives a restart, and it never loses mail.**
A read counts only when its ``tool.returned`` came back ok *and* its turn
finished without failing (spoke, silent or blocked). A failed read, a turn
that failed after reading, or a turn a crash cut short leaves the cursor where
it was, and the next read covers the gap. The stamp kept is the ``tool.called``
moment, taken before the server was contacted, so a mail that arrives during a
read is shown again rather than never.

**Coverage.** A read narrowed to one sender only covers that sender, and an
unread-only read does not cover mail they had already opened: a later read is
cut by an earlier one only if the earlier one would have shown it everything
this one asks for.

With no earlier read from this origin the window is the last
:data:`FIRST_WINDOW_SECONDS` — an hour, the sweep's own cadence.

**Per mailbox, too (DL-092).** With more than one mailbox each keeps its own
cursor under every origin: a read of the work mailbox never cuts the next
read of the personal one, and a mailbox whose read failed inside a read that
covered others keeps its own cursor where it was. Which mailboxes a read
asked for is in its ``tool.called`` arguments (``inbox``, a list), which
read in its ``tool.returned`` tagged lines; a look records its extra
mailboxes' stamps in ``mail_inboxes`` beside ``mail_at``. A read without
the argument, and every record written before it existed, is the default
mailbox's, so the default mailbox's cursor is read from the records it
always was.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Mapping, Optional

from omega import episodes
from omega.inbox import PERSONAL, outcomes

if TYPE_CHECKING:  # pragma: no cover - typing only
    from omega.memory import MemoryStore

__all__ = [
    "FIRST_WINDOW_SECONDS",
    "LOOK",
    "Read",
    "origin",
    "reads",
    "last_read",
    "since",
]

#: How far back the first unprompted read from an origin goes.
FIRST_WINDOW_SECONDS = 60 * 60

#: The origin of omega's own look and of the turns it raises.
LOOK = "self"

#: The tool whose reads move the cursor. Spelled here rather than imported
#: from :mod:`omega.tools`, which imports this module's callers.
_INBOX = "inbox"

#: The channels nobody typed on (:data:`omega.notice.UNPROMPTED_CHANNELS`).
_UNPROMPTED = frozenset({LOOK, "schedule"})


@dataclass(frozen=True)
class Read:
    """One successful unprompted read of the inbox."""

    origin: str
    at: datetime              # when the read began
    sender: Optional[str]     # the ``from`` it was narrowed to, or None for all
    unread_only: bool
    for_seq: int              # the turn it was made in
    mailbox: str = PERSONAL   # which mailbox it read (DL-092)


def origin(event: Mapping[str, Any]) -> Optional[str]:
    """What caused this turn, or ``None`` when their own message did."""
    kind = event.get("kind", episodes.MESSAGE_INBOUND)
    if kind != episodes.MESSAGE_INBOUND:
        return f"kind:{kind}"
    channel = str(event.get("channel", ""))
    if channel not in _UNPROMPTED:
        return None
    if channel == LOOK:
        return LOOK
    sid = event.get("schedule_id")
    return f"schedule:{sid}" if sid else f"channel:{channel}"


def _stamp(at: Any) -> Optional[datetime]:
    if not isinstance(at, str):
        return None
    try:
        parsed = datetime.fromisoformat(at)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def reads(store: "MemoryStore") -> list[Read]:
    """Every unprompted inbox read that counts, oldest first.

    A full scan, as :func:`omega.notice.standing` is and for its reason: a
    cursor resumed from a saved seq is the cheap version that fails open.
    """
    origins: dict[int, str] = {}
    called: dict[int, tuple[Optional[datetime], Mapping[str, Any]]] = {}
    found: list[Read] = []
    finished: set[int] = set()

    for episode in store.episodes_since(0):
        payload = episodes.decode(episode.payload)
        kind = payload.get("kind")
        if kind in (episodes.MESSAGE_INBOUND, episodes.WORK_FINISHED):
            where = origin(payload)
            if where is not None:
                origins[episode.seq] = where
                # The look's own read of the inbox, before its turn ran.
                at = _stamp(payload.get("mail_at"))
                if where == LOOK and at is not None:
                    found.append(Read(LOOK, at, None, False, episode.seq))
                more = payload.get("mail_inboxes")
                if where == LOOK and isinstance(more, dict):
                    for name, stamp in sorted(more.items()):
                        at = _stamp(stamp)
                        if isinstance(name, str) and name != PERSONAL and at is not None:
                            found.append(Read(LOOK, at, None, False, episode.seq, name))
        elif kind == episodes.TOOL_CALLED:
            if payload.get("tool") == _INBOX and payload.get("for_seq") in origins:
                args = payload.get("args")
                called[payload["for_seq"]] = (
                    _stamp(payload.get("at")), args if isinstance(args, dict) else {}
                )
        elif kind == episodes.TOOL_RETURNED:
            seq = payload.get("for_seq")
            if payload.get("tool") == _INBOX and seq in called:
                at, args = called.pop(seq)
                if payload.get("ok") is True and at is not None:
                    sender = args.get("from")
                    unread = args.get("unread_only")
                    for box in _covered(args, payload.get("result")):
                        found.append(Read(
                            origin=origins[seq],
                            at=at,
                            sender=sender if isinstance(sender, str) and sender else None,
                            unread_only=unread if isinstance(unread, bool) else True,
                            for_seq=seq,
                            mailbox=box,
                        ))
        elif kind == episodes.TURN_COMPLETED:
            if payload.get("outcome") != "failed":
                finished.add(payload.get("for_seq"))
        elif kind == episodes.TURN_BLOCKED:
            finished.add(payload.get("for_seq"))

    return [r for r in found if r.for_seq in finished]


def _covered(args: Mapping[str, Any], result: Any) -> list[str]:
    """The mailboxes one successful ``inbox`` read covered (DL-092).

    Without an ``inbox`` argument, the default mailbox, as every read before
    it. With one, each mailbox it asked for that its answer shows as read:
    one tagged as failed, or not answered at all, is not covered.
    """
    asked = args.get("inbox")
    if not isinstance(asked, list):
        return [PERSONAL]
    said = outcomes(str(result or "").split("\n"))
    return [b for b in asked if isinstance(b, str) and said.get(b) is True]


def _covers(earlier: Read, sender: Optional[str], unread_only: bool) -> bool:
    if earlier.sender is not None and (
        sender is None or earlier.sender.lower() != sender.lower()
    ):
        return False
    return unread_only or not earlier.unread_only


def last_read(
    store: "MemoryStore",
    where: str,
    *,
    sender: Optional[str] = None,
    unread_only: bool = False,
    exclude: Optional[int] = None,
    mailbox: str = PERSONAL,
) -> Optional[datetime]:
    """When ``where`` last read everything this read asks for, or ``None``.

    ``exclude`` is the current turn: two reads inside one turn see the same
    mail, so the second is never cut by the first. ``mailbox`` is whose
    cursor (DL-092): only that mailbox's reads count.
    """
    stamps = [
        r.at for r in reads(store)
        if r.origin == where and r.mailbox == mailbox and r.for_seq != exclude
        and _covers(r, sender, unread_only)
    ]
    return max(stamps) if stamps else None


def since(
    store: "MemoryStore",
    where: str,
    *,
    now: datetime,
    sender: Optional[str] = None,
    unread_only: bool = True,
    exclude: Optional[int] = None,
    mailbox: str = PERSONAL,
) -> datetime:
    """The moment after which mail is new to ``where`` in ``mailbox``: its
    last covering read, or :data:`FIRST_WINDOW_SECONDS` ago when it has none."""
    last = last_read(store, where, sender=sender, unread_only=unread_only,
                     exclude=exclude, mailbox=mailbox)
    return last if last is not None else now - timedelta(seconds=FIRST_WINDOW_SECONDS)
