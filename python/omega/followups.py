"""What omega knows about its own failures, and owes the person about them
(DL-087 #2).

A fold over the episodic log, the way :mod:`omega.grants` and
:mod:`omega.finance` are: nothing here is stored beside the log, so a count
survives a restart because the failures it counts are in the log, and the
view is rebuilt from them. ``mcp.Remote.failures`` stays what it was, a cache
for the process that holds it.

One entry per **subject** that has gone wrong:

- a **schedule** — a watch, reminder or brief whose fire's turn ended
  ``failed`` (an exception, a watch that never looked, a provider error after
  the one retry). A fire that ends ``spoke`` or ``silent`` resets it; one that
  blocks on a person is neither.
- an **MCP server** — a call whose ``tool.returned`` carries the adapter's own
  failure (``the <server> MCP server ...``). A call that came back, even one
  the server answered with an error, resets it: the server worked. Refusals
  (money, writes, a call the gate turned away) are policy, never a failure.
- a **research task** (DL-085) — its follow-up fire failed, or ended with
  neither a ``report`` nor a next ``follow_up``. Reset by one that did either;
  closed by the task's ``work.finished``.

Each entry carries the consecutive count, the last error (redacted, capped),
when the run began, and whether omega has told them. Three in a row is owed
one tell; the tell is written down (``failure.told``) so it is never given
twice, and a told subject that works again is written down once more
(``failure.recovered``) so the next brief can say so in a line.

A refused sign-in (DL-087 #3, :mod:`omega.access`) is not counted here: it
is told on the first refusal, with its fix, by that module. A call refused
that way neither fails nor resets its MCP server, a fire that met one neither
fails nor resets its schedule, and a fire skipped while its access is lost
(``access.paused``) is neither too.

Nothing here retries anything. It only counts, and remembers what was said.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Iterable, Optional

from omega import access, episodes

if TYPE_CHECKING:  # pragma: no cover
    from omega.memory import MemoryStore

__all__ = [
    "FAIL_AFTER",
    "SCHEDULE",
    "MCP",
    "RESEARCH",
    "NO_REPORT",
    "Entry",
    "Briefing",
    "State",
    "Followups",
    "redact",
    "research_subject",
    "mcp_subject",
]

#: Consecutive failures before omega says something: one is a blip, two a bad
#: minute, three is something to tell them about. The same bar as the outage
#: tell's (:data:`omega.notice.LOOKS_BEFORE_TELLING`).
FAIL_AFTER = 3

SCHEDULE = "schedule"
MCP = "mcp"
RESEARCH = "research"
assert frozenset({SCHEDULE, MCP, RESEARCH}) == episodes.FAILURE_SUBJECTS

#: The error a research follow-up that neither reported nor carried on is
#: charged with: it ran, and the task went nowhere.
NO_REPORT = "the follow-up ended without a report or a next follow-up"

_SEP = "__"  # as omega.mcp.SEP; not imported, because mcp imports the world
_MCP_FAILED = "the {server} MCP server "
_MCP_ANSWERED = "(Result from the {server} MCP server"


def mcp_subject(server: str) -> str:
    return f"mcp:{server}"


def research_subject(task: int) -> str:
    return f"research:{task}"


# --- redaction ------------------------------------------------------------------

_URL_QUERY = re.compile(r"(https?://[^\s?#]+)[?#]\S*")
_BEARER = re.compile(r"(?i)\b(bearer|basic)\s+[^\s,;]+")
_NAMED_SECRET = re.compile(
    r"(?i)\b(bearer|token|api[_-]?key|key|secret|password|passwd|authorization)"
    r"(\s*[:=]\s*|\s+)([^\s,;]+)"
)
_PREFIXED_SECRET = re.compile(r"\b(?:sk|pk|rk|ghp|gho|ghs|ghu|github_pat|xox[abpr])[-_][A-Za-z0-9_\-]{8,}")
_LONG_RUN = re.compile(r"[A-Za-z0-9_\-+/=]{32,}")
_SPACE = re.compile(r"\s+")


def redact(error: Any, *, cap: int = episodes.MAX_FAILURE_ERROR) -> str:
    """An error as the log keeps it: one line, no secrets, at most ``cap``.

    What an error is made of here is a provider's or a server's own words, and
    those can carry a URL with a key in its query, a bearer header echoed
    back, or a token. Each is masked: queries dropped, anything after a named
    secret, anything with a known token prefix, and any long unbroken run.
    """
    text = _SPACE.sub(" ", str(error or "")).strip() or "an unknown error"
    text = _URL_QUERY.sub(r"\1", text)
    text = _BEARER.sub(lambda m: f"{m.group(1)} [redacted]", text)
    text = _NAMED_SECRET.sub(lambda m: f"{m.group(1)}{m.group(2)}[redacted]", text)
    text = _PREFIXED_SECRET.sub("[redacted]", text)
    text = _LONG_RUN.sub("[redacted]", text)
    if len(text) > cap:
        text = text[: cap - 1].rstrip() + "…"
    return text


# --- the view ------------------------------------------------------------------


@dataclass(frozen=True)
class Entry:
    """One subject's run of consecutive failures."""

    subject: str
    kind: str  # SCHEDULE, MCP or RESEARCH
    count: int
    last_error: str
    first_seen: str
    told: bool = False
    #: How the person would recognise it, read off the fire or the tool name.
    label: str = ""
    #: The schedule id behind a schedule subject (the subject itself), the
    #: server's name for an MCP one, the task's seq for a research one.
    ref: str = ""

    @property
    def watch(self) -> bool:
        """A standing watch: the only kind a ``tidy`` grant may pause."""
        return self.kind == SCHEDULE and self.label.startswith("the watch ")


@dataclass(frozen=True)
class Briefing:
    """What one brief owes them about omega's own failures."""

    #: Subjects they were told about that work again, since the brief before.
    recovered: tuple[Entry, ...] = ()
    #: Research tasks whose last follow-up failed, not yet shown at this count.
    research: tuple[Entry, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.recovered or self.research)


@dataclass(frozen=True)
class State:
    """The follow-ups as of one seq. Frozen, as :class:`omega.grants.State`."""

    through: int = 0
    entries: dict[str, Entry] = field(default_factory=dict)
    #: Told subjects that work again and have no ``failure.recovered`` yet.
    recovering: tuple[Entry, ...] = ()
    #: Brief seq -> what that brief owes them (only the latest is kept).
    briefed: dict[int, Briefing] = field(default_factory=dict)

    def owed(self) -> tuple[Entry, ...]:
        """Subjects owed their one tell, oldest run first. Research tasks
        are not among them: a failed task is told in the brief instead."""
        return tuple(
            sorted(
                (
                    e for e in self.entries.values()
                    if e.kind in (SCHEDULE, MCP) and e.count >= FAIL_AFTER and not e.told
                ),
                key=lambda e: (e.first_seen, e.subject),
            )
        )

    def briefing(self, seq: int) -> Briefing:
        return self.briefed.get(seq, Briefing())

    def __bool__(self) -> bool:
        return bool(self.entries or self.recovering or any(self.briefed.values()))


class Followups:
    """The fold. ``apply`` in seq order; :meth:`state` snapshots it."""

    __slots__ = (
        "_through", "_fires", "_entries", "_recovering", "_unbriefed",
        "_shown", "_briefed", "_refused",
    )

    def __init__(self) -> None:
        self._through = 0
        #: Fire seq -> (subject, kind, label, ref, at): turns not yet ended.
        self._fires: dict[int, tuple[str, str, str, str, str]] = {}
        self._entries: dict[str, Entry] = {}
        self._recovering: dict[str, Entry] = {}
        self._unbriefed: list[Entry] = []
        #: Research subject -> the count a brief last showed it at.
        self._shown: dict[str, int] = {}
        self._briefed: dict[int, Briefing] = {}
        #: Fire seqs whose turn met a refused sign-in (DL-087 #3).
        self._refused: set[int] = set()

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
            self._inbound(seq, payload)
        elif kind == episodes.TURN_COMPLETED:
            fire = self._fires.pop(payload.get("for_seq"), None)  # type: ignore[arg-type]
            if payload.get("for_seq") in self._refused:
                # A refused sign-in is told by omega.access, with its fix.
                self._refused.discard(payload.get("for_seq"))  # type: ignore[arg-type]
            elif fire is not None:
                subject, sub_kind, label, ref, at = fire
                tools = payload.get("tools") or []
                if payload.get("outcome") == "failed":
                    self._fail(subject, sub_kind, label, ref, payload.get("error"), at)
                elif sub_kind == RESEARCH and not (
                    "report" in tools or "follow_up" in tools
                ):
                    self._fail(subject, sub_kind, label, ref, NO_REPORT, at)
                else:
                    self._reset(subject)
        elif kind == episodes.TURN_BLOCKED:
            # Waiting on a person is neither a failure nor a fire that worked.
            self._fires.pop(payload.get("for_seq"), None)  # type: ignore[arg-type]
            self._refused.discard(payload.get("for_seq"))  # type: ignore[arg-type]
        elif kind == episodes.ACCESS_PAUSED:
            # Skipped because its access is lost: neither (DL-087 #3).
            self._fires.pop(payload.get("for_seq"), None)  # type: ignore[arg-type]
        elif kind == episodes.TOOL_RETURNED:
            self._returned(payload)
        elif kind == episodes.SCHEDULE_CANCELLED:
            # A schedule that is gone has nothing left to fail or recover.
            sid = str(payload.get("id", ""))
            self._entries.pop(sid, None)
            self._recovering.pop(sid, None)
        elif kind == episodes.WORK_FINISHED:
            # A research task that reported is finished; its run is over.
            subject = research_subject(int(payload["for_seq"]))
            self._entries.pop(subject, None)
            self._shown.pop(subject, None)
        elif kind == episodes.FAILURE_TOLD:
            e = self._entries.get(payload["subject"])
            if e is not None:
                self._entries[e.subject] = replace(e, told=True)
        elif kind == episodes.FAILURE_RECOVERED:
            e = self._recovering.pop(payload["subject"], None)
            if e is not None:
                self._unbriefed.append(e)
        self._through = seq

    def _inbound(self, seq: int, payload: dict[str, Any]) -> None:
        # Lazy: tools imports grants, which this module must not drag in.
        from omega.tools import is_brief, is_tell, research_task

        if payload.get("channel") != "schedule":
            return
        at = str(payload.get("at", ""))
        sid = payload.get("schedule_id")
        if sid and not is_tell(payload):
            task = research_task(payload)
            if task is not None:
                self._fires[seq] = (
                    research_subject(task), RESEARCH, _research_label(payload, task),
                    str(task), at,
                )
            elif not str(payload.get("text", "")).startswith(_RESEARCH_LEAD):
                self._fires[seq] = (str(sid), SCHEDULE, _schedule_label(payload), str(sid), at)
        if is_brief(payload):
            research = tuple(
                e for e in sorted(self._entries.values(), key=lambda e: e.subject)
                if e.kind == RESEARCH and self._shown.get(e.subject) != e.count
            )
            for e in research:
                self._shown[e.subject] = e.count
            self._briefed[seq] = Briefing(recovered=tuple(self._unbriefed), research=research)
            self._unbriefed = []

    def _returned(self, payload: dict[str, Any]) -> None:
        got = access.classify(payload)
        if got is not None and got[1]:
            # A refused sign-in: omega.access tells it, once, with the fix.
            if payload.get("for_seq") in self._fires:
                self._refused.add(payload["for_seq"])
            return
        tool = str(payload.get("tool", ""))
        if _SEP not in tool:
            return
        server = tool.split(_SEP, 1)[0]
        if not server:
            return
        error = str(payload.get("error") or "")
        subject = mcp_subject(server)
        if payload.get("ok") or error.startswith(_MCP_ANSWERED.format(server=server)):
            # The server answered. An answer that says the call failed is the
            # server working; the adapter keeps the same rule (Remote._note).
            self._reset(subject)
        elif error.startswith(_MCP_FAILED.format(server=server)):
            self._fail(
                subject, MCP, f"the {server} MCP server", server, error,
                str(payload.get("at", "")),
            )
        # Anything else is policy or a bug in omega's own code: a refusal is
        # the gate working, and it says nothing about the server.

    def _fail(
        self, subject: str, kind: str, label: str, ref: str, error: Any, at: str
    ) -> None:
        e = self._entries.get(subject)
        if e is None:
            self._recovering.pop(subject, None)
            self._entries[subject] = Entry(
                subject=subject, kind=kind, count=1, last_error=redact(error),
                first_seen=at, label=label, ref=ref,
            )
        else:
            self._entries[subject] = replace(
                e, count=e.count + 1, last_error=redact(error), label=label or e.label
            )

    def _reset(self, subject: str) -> None:
        e = self._entries.pop(subject, None)
        self._shown.pop(subject, None)
        if e is not None and e.told:
            self._recovering[subject] = e

    def state(self) -> State:
        return State(
            through=self._through,
            entries=dict(self._entries),
            recovering=tuple(self._recovering.values()),
            briefed=dict(list(self._briefed.items())[-1:]),
        )

    @classmethod
    def fold(cls, decoded: Iterable[tuple[int, dict[str, Any]]]) -> "Followups":
        view = cls()
        for seq, payload in decoded:
            view.apply(seq, payload)
        return view

    def advance(self, store: "MemoryStore", *, upto: Optional[int] = None) -> "Followups":
        """Fold what the log gained since :attr:`through`, to ``upto`` (as
        :meth:`omega.grants.Grants.advance`)."""
        for episode in store.episodes_since(self._through):
            if upto is not None and episode.seq > upto:
                break
            self.apply(episode.seq, episodes.decode(episode.payload))
        return self

    @classmethod
    def rebuild(cls, store: "MemoryStore") -> "Followups":
        return cls().advance(store)


# --- labels -----------------------------------------------------------------------

# As omega.tools; repeated rather than imported at module load, for the reason
# `_inbound` imports lazily. tests/test_followups.py pins them to the originals.
_WATCH_LEAD = "Standing watch you agreed to keep for them: "
_REMINDER_LEAD = "Reminder you promised them: "
_BRIEF_LEAD = "Daily brief you promised them: "
_RESEARCH_LEAD = "Research you are carrying for them: "
_REVIEW_LEAD = "Weekly self-review you hold for yourself: "  # omega.lessons


def _first(payload: dict[str, Any]) -> str:
    text = str(payload.get("text", ""))
    return text.splitlines()[0] if text else ""


def _schedule_label(payload: dict[str, Any]) -> str:
    first = _first(payload)
    if first.startswith(_WATCH_LEAD):
        return f"the watch you asked me to keep ({first[len(_WATCH_LEAD):][:80]})"
    if first.startswith(_REMINDER_LEAD):
        return f"your reminder ({first[len(_REMINDER_LEAD):][:80]})"
    if first.startswith(_BRIEF_LEAD):
        return "your daily brief"
    if first.startswith(_REVIEW_LEAD):
        return "my weekly self-review"
    return f"the scheduled job {payload.get('schedule_id')}"


def _research_label(payload: dict[str, Any], task: int) -> str:
    first = _first(payload)[len(_RESEARCH_LEAD):]
    marker = first.rfind(" (task ")
    what = first[:marker] if marker >= 0 else first
    return f"research task {task} ({what[:80]})"
