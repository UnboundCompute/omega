"""Sign-ins and access omega has lost, and what fixes each (DL-087 #3).

An expired Google sign-in, an IMAP login the server refuses, or an MCP server
answering 401/403 used to read as one more failure: :mod:`omega.followups`
told them after three in a row, named no fix, and every job that read from
the account went on failing in the meantime. This is the same kind of fold
as that one, over the same log, for the one failure that needs the person.

**Explicit markers only.** An **origin** is lost when a read says in so many
words that it refused the credentials, and nothing else counts:

- ``inbox`` — the ``inbox`` or ``mail`` tool, or the look's inbox read, whose
  failure carries :data:`omega.inbox.IMAP_FIX` (written only when the server
  answered the LOGIN with AUTHENTICATIONFAILED or the like);
- ``google`` — the ``calendar`` tool or the look's calendar read carrying
  :data:`omega.gcal.SIGN_IN_AGAIN` (the refresh answered ``invalid_grant``);
- ``mcp:<server>`` — a call whose adapter error says the server refused the
  credentials (HTTP 401 or 403);
- ``inbox:<name>`` — an extra mailbox (DL-092), read from the same tools and
  the same look, whose tagged failure line carries that mailbox's own fix
  (:func:`omega.inbox.imap_fix`). The default mailbox stays ``inbox``. One
  answer can say one mailbox worked and another was refused; each is its own
  origin, so a refusal of one pauses only what reads that one.

Anything else is a plain failure and stays :mod:`omega.followups`'s.

**What the fold knows.** Which origins are lost (one ``access.lost`` each,
the one tell, never repeated while it stays lost); which refusals are still
owed that tell; which lost origins have worked since (owed an
``access.restored`` and its one short tell); and which schedules read from
which origin, learned from their own fires' refusals, so that while it stays
lost the executor skips them (``access.paused``) instead of running a fire
that can only fail again. A schedule is paused only after it has itself been
refused: one wasted fire, and never a guess about what a watch reads.

Nothing here retries, signs in, or writes anywhere. The fix it names is
omega's own sentence, so neither a log record nor a tell can carry a key.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, Optional, Sequence

from omega import episodes
from omega.calendar import NOT_CONFIGURED
from omega.gcal import SIGN_IN_AGAIN
from omega.inbox import IMAP_FIX, NOT_CONNECTED, PERSONAL, imap_fix, outcomes

if TYPE_CHECKING:  # pragma: no cover
    from omega.memory import MemoryStore

__all__ = [
    "INBOX",
    "INBOX_PREFIX",
    "GOOGLE",
    "PROBE_EVERY_SECONDS",
    "Lost",
    "State",
    "Access",
    "classify",
    "classify_all",
    "inbox_origin",
    "refused_in",
    "mcp_origin",
    "fix_for",
    "label_for",
    "lost_words",
    "restored_words",
    "paused_line",
    "probe",
]

INBOX = "inbox"
GOOGLE = "google"
_MCP = "mcp:"
#: An extra mailbox's origin is this and its name (DL-092).
INBOX_PREFIX = "inbox:"


def inbox_origin(name: str) -> str:
    """The origin of mailbox ``name``: ``inbox`` for the default, as always."""
    return INBOX if name == PERSONAL else f"{INBOX_PREFIX}{name}"

#: How often a lost origin is checked again, at most: hourly, the mail
#: cursor's cadence. Cheap (a login, a token refresh, a listing), but it is
#: still a request to an account that is refusing omega.
PROBE_EVERY_SECONDS = 60 * 60

# As omega.tools and omega.mcp; repeated, because mcp imports tools, which
# imports the world. tests/test_access.py pins them to the originals.
_INBOX_TOOLS = frozenset({"inbox", "mail"})
_MAIL_TOOL = "mail"
_CALENDAR_TOOL = "calendar"
_SEP = "__"
_MCP_REFUSED = "the {server} MCP server refused the credentials ("
_MCP_ANSWERED = "(Result from the {server} MCP server"
#: The server the leave-now alarm routes through (omega.leave.SERVER).
_LEAVE_SERVER = "maps"
#: How a Google failure inside a calendar answer that still read its feeds
#: begins (omega.calendar.lines).
_GOOGLE_PARTIAL = "couldn't read Google calendar:"


def mcp_origin(server: str) -> str:
    return f"{_MCP}{server}"


def fix_for(origin: str) -> str:
    """The one concrete thing that fixes ``origin``, in omega's words."""
    if origin == INBOX:
        return IMAP_FIX
    if origin.startswith(INBOX_PREFIX):
        return imap_fix(origin[len(INBOX_PREFIX):])
    if origin == GOOGLE:
        return SIGN_IN_AGAIN
    server = origin[len(_MCP):]
    return (
        f"the {server} MCP server refused its token: replace "
        f"OMEGA_MCP_{server.upper()}_TOKEN"
    )


def label_for(origin: str, environ: Optional[Mapping[str, str]] = None) -> str:
    if origin == INBOX:
        # Named only beside another mailbox (DL-092), so a machine with one
        # says what it always said.
        import os

        from omega.inbox import multi

        return "your personal inbox" if multi(os.environ if environ is None else environ) else "your inbox"
    if origin.startswith(INBOX_PREFIX):
        return f"your {origin[len(INBOX_PREFIX):]} inbox"
    if origin == GOOGLE:
        return "your Google calendar"
    return f"the {origin[len(_MCP):]} MCP server"


def lost_words(origin: str) -> str:
    """The one tell when ``origin`` is lost."""
    return (
        f"I can't get into {label_for(origin)}: {fix_for(origin)}. Until then "
        "I've paused what reads from it, and I'll say so here when it works again."
    )


def restored_words(origin: str) -> str:
    """The one short tell when it works again."""
    label = label_for(origin)
    return f"{label[0].upper()}{label[1:]} works again; what was paused is running again."


def paused_line(origin: str) -> str:
    """What a look says in place of a read it skipped."""
    return f"paused until it is fixed (they have been told): {fix_for(origin)}"


# --- the classifier -----------------------------------------------------------


def classify_all(payload: Mapping[str, Any]) -> list[tuple[str, bool]]:
    """What one ``tool.returned`` says about every origin it read (DL-092).

    Exactly ``[classify(payload)]`` (or nothing) unless the answer is an
    inbox tool's tagged with mailboxes: then one fact per mailbox it names,
    in name order. A listing's lines are all omega's own; an opened mail's
    are the sender's past its first line, so only that one is read.
    """
    if payload.get("kind") not in (episodes.TOOL_RETURNED, episodes.WATCHER_CHECKED,
                                   episodes.LEAVE_CHECKED):
        # A page watch's check (DL-096) counts as :func:`classify` says.
        return []
    tool = str(payload.get("tool", ""))
    if tool in _INBOX_TOOLS:
        ok = payload.get("ok") is True
        text = str((payload.get("result") if ok else payload.get("error")) or "")
        lines = text.split("\n")
        if ok and tool == _MAIL_TOOL:
            lines = lines[:1]
        said = outcomes(lines)
        if said:
            return [
                (inbox_origin(name), not worked)
                for name, worked in sorted(said.items())
                if worked is not None
            ]
    one = classify(payload)
    return [one] if one is not None else []


def classify(payload: Mapping[str, Any]) -> Optional[tuple[str, bool]]:
    """What one ``tool.returned`` says about an origin, or ``None``.

    ``(origin, True)`` is a refusal of the credentials; ``(origin, False)``
    is the origin working. Anything else (a timeout, a server error, a call
    the gate turned away, a tool that reads from no account) is ``None``.
    An answer tagged with mailboxes (DL-092) is :func:`classify_all`'s.
    """
    if payload.get("kind") == episodes.LEAVE_CHECKED:
        # DL-093: the leave-now alarm's own Maps call, written down as a
        # ``leave.checked`` rather than a ``tool.returned``, says the same.
        server = _LEAVE_SERVER
        if payload.get("eta_min") is not None:
            return mcp_origin(server), False
        if str(payload.get("error") or "").startswith(_MCP_REFUSED.format(server=server)):
            return mcp_origin(server), True
        return None
    # A page watch's check (DL-096) reads through the same MCP servers and
    # carries the same fields, so it counts as any MCP read does.
    if payload.get("kind") not in (episodes.TOOL_RETURNED, episodes.WATCHER_CHECKED):
        return None
    tool = str(payload.get("tool", ""))
    ok = payload.get("ok") is True
    text = str((payload.get("result") if ok else payload.get("error")) or "")
    if tool in _INBOX_TOOLS:
        if ok:
            # An inbox that is not set up says so as an answer: not a login.
            return None if text.startswith(NOT_CONNECTED) else (INBOX, False)
        return (INBOX, True) if IMAP_FIX in text else None
    if tool == _CALENDAR_TOOL:
        if SIGN_IN_AGAIN in text:
            return GOOGLE, True
        if ok and _GOOGLE_PARTIAL not in text and not text.startswith(NOT_CONFIGURED):
            return GOOGLE, False
        return None
    if _SEP in tool:
        server = tool.split(_SEP, 1)[0]
        if not server:
            return None
        if ok or text.startswith(_MCP_ANSWERED.format(server=server)):
            return mcp_origin(server), False
        if text.startswith(_MCP_REFUSED.format(server=server)):
            return mcp_origin(server), True
    return None


def refused_in(origin: str, lines: Sequence[str]) -> bool:
    """Does a look's block for ``origin`` carry a refusal? An extra
    mailbox's (DL-092) only in its own tagged failure line."""
    if origin.startswith(INBOX_PREFIX):
        name = origin[len(INBOX_PREFIX):]
        return outcomes(lines).get(name, True) is False
    mark = IMAP_FIX if origin == INBOX else SIGN_IN_AGAIN
    return any(mark in str(line) for line in lines)


# --- the view -----------------------------------------------------------------


@dataclass(frozen=True)
class Lost:
    """One origin whose access is lost, or owed the tell that says so."""

    origin: str
    since: str  # the ``access.lost`` (or, while owed, the refusal) ``at``

    @property
    def fix(self) -> str:
        return fix_for(self.origin)

    @property
    def label(self) -> str:
        return label_for(self.origin)


@dataclass(frozen=True)
class State:
    """Access as of one seq. Frozen, as :class:`omega.followups.State`."""

    through: int = 0
    lost: dict[str, Lost] = field(default_factory=dict)
    #: Refusals of an origin not yet lost: owed their one tell, oldest first.
    owed: tuple[Lost, ...] = ()
    #: Lost origins that worked since: owed ``access.restored``.
    recovering: tuple[str, ...] = ()
    #: Schedule id -> the origins its own fires were refused by.
    reads: dict[str, frozenset[str]] = field(default_factory=dict)

    def pausing(self, schedule_id: str) -> Optional[str]:
        """The lost origin a fire of this schedule would read from, if any."""
        hit = sorted(self.reads.get(schedule_id, frozenset()) & self.lost.keys())
        return hit[0] if hit else None

    def __bool__(self) -> bool:
        return bool(self.lost or self.owed or self.recovering)


class Access:
    """The fold. ``apply`` in seq order; :meth:`state` snapshots it."""

    __slots__ = ("_through", "_fires", "_lost", "_owed", "_recovering", "_reads")

    def __init__(self) -> None:
        self._through = 0
        #: Fire seq -> schedule id, for fires whose turn has not ended.
        self._fires: dict[int, str] = {}
        self._lost: dict[str, Lost] = {}
        self._owed: dict[str, Lost] = {}
        self._recovering: dict[str, None] = {}
        self._reads: dict[str, set[str]] = {}

    @property
    def through(self) -> int:
        return self._through

    def apply(self, seq: int, payload: dict[str, Any]) -> None:
        if seq <= self._through:
            raise ValueError(
                f"episodes must be folded in order: got seq {seq} after {self._through}"
            )
        kind = payload.get("kind")
        if kind == episodes.MESSAGE_INBOUND:
            sid = payload.get("schedule_id")
            if payload.get("channel") == "schedule" and sid:
                self._fires[seq] = str(sid)
        elif kind in (episodes.TURN_COMPLETED, episodes.TURN_BLOCKED):
            self._fires.pop(payload.get("for_seq"), None)  # type: ignore[arg-type]
        elif kind in (episodes.TOOL_RETURNED, episodes.WATCHER_CHECKED,
                      episodes.LEAVE_CHECKED):
            self._returned(payload)
        elif kind == episodes.ACCESS_LOST:
            origin = str(payload["origin"])
            self._owed.pop(origin, None)
            self._recovering.pop(origin, None)
            self._lost[origin] = Lost(origin, str(payload.get("at", "")))
        elif kind == episodes.ACCESS_RESTORED:
            origin = str(payload["origin"])
            self._lost.pop(origin, None)
            self._recovering.pop(origin, None)
            for origins in self._reads.values():
                origins.discard(origin)
        elif kind == episodes.SCHEDULE_CANCELLED:
            self._reads.pop(str(payload.get("id", "")), None)
        self._through = seq

    def _returned(self, payload: dict[str, Any]) -> None:
        for origin, refused in classify_all(payload):
            self._one(payload, origin, refused)

    def _one(self, payload: dict[str, Any], origin: str, refused: bool) -> None:
        if refused:
            sid = self._fires.get(payload.get("for_seq"))  # type: ignore[arg-type]
            if sid is not None:
                self._reads.setdefault(sid, set()).add(origin)
            if origin in self._lost:
                # Still lost, and they already know: never told twice.
                self._recovering.pop(origin, None)
            elif origin not in self._owed:
                self._owed[origin] = Lost(origin, str(payload.get("at", "")))
        else:
            # It worked. A refusal not yet told about is not told at all.
            self._owed.pop(origin, None)
            if origin in self._lost:
                self._recovering[origin] = None

    def state(self) -> State:
        return State(
            through=self._through,
            lost=dict(self._lost),
            owed=tuple(sorted(self._owed.values(), key=lambda e: (e.since, e.origin))),
            recovering=tuple(self._recovering),
            reads={sid: frozenset(o) for sid, o in self._reads.items() if o},
        )

    @classmethod
    def fold(cls, decoded: Iterable[tuple[int, dict[str, Any]]]) -> "Access":
        view = cls()
        for seq, payload in decoded:
            view.apply(seq, payload)
        return view

    def advance(self, store: "MemoryStore", *, upto: Optional[int] = None) -> "Access":
        """Fold what the log gained since :attr:`through`, to ``upto``."""
        for episode in store.episodes_since(self._through):
            if upto is not None and episode.seq > upto:
                break
            self.apply(episode.seq, episodes.decode(episode.payload))
        return self

    @classmethod
    def rebuild(cls, store: "MemoryStore") -> "Access":
        return cls().advance(store)


# --- the probe ------------------------------------------------------------------


def probe(origin: str, environ: Optional[Mapping[str, str]] = None) -> Optional[bool]:
    """Does ``origin`` work again? ``True``, ``False`` (still refused) or
    ``None`` (cannot tell). The production probe the entry point hands the
    executor: one login, one token refresh, or one listing, never a write and
    never a sign-in. Never raises."""
    import os

    env = os.environ if environ is None else environ
    try:
        if origin == INBOX:
            from omega import inbox

            return inbox.probe(env)
        if origin.startswith(INBOX_PREFIX):
            from omega import inbox

            return inbox.probe(env, name=origin[len(INBOX_PREFIX):])
        if origin == GOOGLE:
            from omega import gcal

            return gcal.probe(env)
        if origin.startswith(_MCP):
            from omega import mcp

            return mcp.probe(origin[len(_MCP):], env)
    except Exception:  # noqa: BLE001 - a probe never kills the look
        return None
    return None


#: The shape the executor takes a probe in.
Probe = Callable[[str], Optional[bool]]
