"""The person core (DL-098): thinking time, and picking the moment to speak.

Two things a person does that a loop on a timer does not.

**Thinking time.** Twice a day, at :data:`SLOTS` in their own zone (the
process zone, which is theirs: DL-078), omega looks over everything open at
once: what waits on them, the promises they made, what keeps failing, the
topics it keeps up with, the sign-ins it lost, research under way, and the
messages it drafted and held. It is the ordinary look (DL-076), one inbound
on the ``self`` channel carrying :data:`episodes.inbound`'s ``thinking``
field, judged by the same NUDGE/CHECK/SILENT judge; there is no second loop.
Most days the right answer is nothing, and the text says so. Its spend is
capped per day (:data:`THINKING_DAILY_TOKENS`) and its text by
:data:`MAX_THINKING_CHARS`.

**Picking the moment.** A non-urgent unprompted message (a look's nudge, the
thinking turn's message, a failure follow-up, the weekly self-review) is
never sent in quiet hours (:func:`quiet`), at most :data:`MAX_PER_DAY` are
sent a day, and outside a thinking slot only while they are here
(:func:`present`). Otherwise the turn's words are written down as a
``message.queued`` draft instead of spoken. A held draft goes out the next
time they are here (several as one message), or is shown to the next
thinking turn, which sends what still matters as one message, or nothing,
and drops the rest. Over the budget, the thinking turn's own message is held
too, and so carried to tomorrow.

What is exempt from the budget and the moment (DL-106, M1a): a reminder
they set (DL-075), a deadline warning (leave-now), a watch for a condition
they named, research follow-ups and reports they asked for, and omega telling
them it cannot do what it was asked (a lost sign-in, an outage). Exempt from
the budget is never exempt from the told-ledger (:mod:`omega.ledger`): each
of those names its keys, and one that would say again what it said, with
nothing new, ends quiet. Standing "anything new" watches no longer speak at
all (they are senses), and the brief is the morning thinking slot (DL-114),
so both count inside the day's two.

Everything is a fold over ``message.queued`` / ``message.delivered`` /
``message.dropped`` and the inbounds that carry ``queued`` or ``thinking``,
so the queue, the budget and the slots already served survive a restart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Iterable, Optional, Sequence

from omega import episodes, machine, notice
from omega.tools import BRIEF_LEAD
from omega.topics import spent

if TYPE_CHECKING:  # pragma: no cover
    from omega.memory import MemoryStore

__all__ = [
    "QUIET_FROM_HOUR",
    "QUIET_UNTIL_HOUR",
    "SLOTS",
    "MAX_PER_DAY",
    "PRESENT_SECONDS",
    "THINKING_DAILY_TOKENS",
    "MAX_THINKING_CHARS",
    "HOLD_QUIET",
    "HOLD_BUDGET",
    "HOLD_AWAY",
    "HOLD_CLOCK",
    "HOLD_SPOKE",
    "SPOKE_SECONDS",
    "SLOT_DEFER_SECONDS",
    "SLOT_LATE_SECONDS",
    "RECHECK_LEAD",
    "COMPOSE_LEAD",
    "KEYED_HEADING",
    "compose_text",
    "Draft",
    "State",
    "Outbox",
    "Gate",
    "quiet",
    "slot_due",
    "slot_start",
    "slots_missed",
    "conversing",
    "recheck_text",
    "carried_taint",
    "present",
    "hold",
    "thinking_left",
    "merged",
    "held_lines",
    "thinking_text",
    "settle",
    "dropped",
]

#: Never 23:00-08:00 local.
QUIET_FROM_HOUR = 23
QUIET_UNTIL_HOUR = 8
#: Thinking time, by name and local hour.
SLOTS = (("morning", 9), ("evening", 19))
#: Non-urgent unprompted messages a day ("once or twice a day is fine").
MAX_PER_DAY = 2
#: How recently a sign of them counts as "here": a presence report from the
#: Mac (DL-072/073), or a message of theirs on any channel.
PRESENT_SECONDS = 10 * 60
#: What the thinking turns may spend in a day, checked before each slot. A
#: slot with nothing left is not thought; the ordinary look runs instead.
THINKING_DAILY_TOKENS = 60_000
#: The thinking turn's own text is a prompt, so it is capped like one.
MAX_THINKING_CHARS = 4_000
#: No unprompted message goes out this soon after a message of theirs
#: (B4-lite): a held draft delivered 30 s after they spoke reads as not
#: listening. Also how recently they spoke for a slot to count them as
#: mid-conversation (C4).
SPOKE_SECONDS = 10 * 60
#: How long a thinking slot waits for them to finish a conversation (C4).
SLOT_DEFER_SECONDS = 15 * 60
#: A slot not started this long after its hour did not run (C4): it is
#: recorded as ``thinking.failed`` and the next slot is the retry. A starting
#: guess [A], wide enough for the deferral and a slow tick.
SLOT_LATE_SECONDS = 30 * 60

HOLD_QUIET = "quiet hours (23:00-08:00)"
HOLD_BUDGET = f"today already had its {MAX_PER_DAY} unprompted messages"
HOLD_AWAY = "they are not here; held for when they are, or the next thinking time"
HOLD_CLOCK = "couldn't tell the time"
HOLD_SPOKE = "they spoke in the last 10 minutes; held so it does not cut in"


def _local(now: datetime) -> datetime:
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return now.astimezone()


def _parse(at: Any) -> Optional[datetime]:
    try:
        parsed = datetime.fromisoformat(str(at))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


@dataclass(frozen=True)
class Draft:
    """One held message: its ``message.queued`` seq, words, maker and stamp."""

    seq: int
    text: str
    source: str
    at: str
    #: What untrusted content the turn that wrote it had read (DL-064).
    tainted: tuple[str, ...] = ()
    #: The told-ledger keys it speaks for (C1); empty when it named none.
    keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class State:
    through: int = 0
    #: Held drafts nothing has delivered or dropped yet, oldest first.
    drafts: tuple[Draft, ...] = ()
    #: Local day -> non-urgent unprompted messages delivered that day.
    delivered: dict[str, int] = field(default_factory=dict)
    #: Thinking slots already served, as ``"<date> <name>"``.
    thought: frozenset[str] = frozenset()
    #: Local day -> tokens the thinking turns spent.
    spending: dict[str, int] = field(default_factory=dict)
    #: The ``at`` of the newest message they sent, on any channel.
    heard_at: Optional[str] = None
    #: The ``at`` of the log's first episode: no slot before it was owed.
    began: Optional[str] = None
    #: Local days whose brief went out or was held as a draft (DL-114).
    briefed: frozenset[str] = frozenset()
    #: Local day -> the ``at`` of the last thinking turn that carried the brief.
    brief_tried: dict[str, str] = field(default_factory=dict)

    def delivered_on(self, day: str) -> int:
        return self.delivered.get(day, 0)


class Outbox:
    """The fold. Kept at the log's head by the executor, as its other views."""

    __slots__ = ("_through", "_drafts", "_taken", "_delivered", "_thought",
                 "_thinking", "_spending", "_heard_at", "_began", "_briefing",
                 "_briefed", "_brief_tried")

    def __init__(self) -> None:
        self._through = 0
        self._drafts: dict[int, Draft] = {}
        self._taken: set[int] = set()
        self._delivered: dict[str, int] = {}
        self._thought: set[str] = set()
        # thinking inbound seq -> its local day, for the spend.
        self._thinking: dict[int, str] = {}
        self._spending: dict[str, int] = {}
        self._heard_at: Optional[str] = None
        self._began: Optional[str] = None
        # brief-carrying thinking inbound seq -> its local day (gap 0a).
        self._briefing: dict[int, str] = {}
        self._briefed: set[str] = set()
        self._brief_tried: dict[str, str] = {}

    def apply(self, seq: int, payload: dict[str, Any]) -> None:
        self._through = seq
        kind = payload.get("kind")
        if self._began is None and isinstance(payload.get("at"), str):
            self._began = payload["at"]
        if kind == episodes.MESSAGE_INBOUND:
            channel = payload.get("channel")
            if channel not in notice.UNPROMPTED_CHANNELS:
                at = payload.get("at")
                if isinstance(at, str):
                    self._heard_at = at
            slot = payload.get("thinking")
            if channel == notice.CHANNEL and isinstance(slot, str):
                self._thought.add(slot)
                day = notice.local_date(str(payload.get("at", "")))
                if day is not None:
                    self._thinking[seq] = day
                    if str(payload.get("text", "")).startswith(BRIEF_LEAD):
                        self._briefing[seq] = day
                        self._brief_tried[day] = str(payload.get("at", ""))
            elif channel == "schedule" and isinstance(payload.get("queued"), list):
                # (Before B4-lite held drafts went out this way, verbatim; a
                # log written then still folds the same.)
                # A tell that carries drafts takes them when it is written:
                # it is delivered as written, so nothing else may send them.
                self._taken.update(payload["queued"])
        elif kind == episodes.MESSAGE_QUEUED:
            if payload.get("for_seq") in self._briefing:
                self._briefed.add(self._briefing[payload["for_seq"]])
            self._drafts[seq] = Draft(
                seq=seq, text=str(payload["text"]), source=str(payload["source"]),
                at=str(payload["at"]), tainted=tuple(payload.get("tainted") or ()),
                keys=tuple(payload.get("keys") or ()),
            )
        elif kind == episodes.MESSAGE_DELIVERED:
            if payload.get("for_seq") in self._briefing:
                self._briefed.add(self._briefing[payload["for_seq"]])
            self._taken.update(payload.get("queued") or [])
            day = notice.local_date(str(payload.get("at", "")))
            if day is not None:
                self._delivered[day] = self._delivered.get(day, 0) + 1
        elif kind == episodes.MESSAGE_DROPPED:
            self._taken.update(payload.get("queued") or [])
        elif kind == episodes.THINKING_FAILED:
            # Served, as far as the clock is concerned: the next slot retries.
            self._thought.add(str(payload["slot"]))
        elif kind == episodes.TURN_COMPLETED:
            day = self._thinking.pop(payload.get("for_seq"), None)
            if day is not None:
                self._spending[day] = self._spending.get(day, 0) + spent(payload.get("usage"))

    def state(self) -> State:
        drafts = tuple(
            d for s, d in sorted(self._drafts.items()) if s not in self._taken
        )
        return State(
            through=self._through,
            drafts=drafts,
            delivered=dict(self._delivered),
            thought=frozenset(self._thought),
            spending=dict(self._spending),
            heard_at=self._heard_at,
            began=self._began,
            briefed=frozenset(self._briefed),
            brief_tried=dict(self._brief_tried),
        )

    @classmethod
    def fold(cls, decoded: Iterable[tuple[int, dict[str, Any]]]) -> "Outbox":
        view = cls()
        for seq, payload in decoded:
            view.apply(seq, payload)
        return view

    def advance(self, store: "MemoryStore", *, upto: Optional[int] = None) -> "Outbox":
        for episode in store.episodes_since(self._through):
            if upto is not None and episode.seq > upto:
                break
            self.apply(episode.seq, episodes.decode(episode.payload))
        return self

    @classmethod
    def rebuild(cls, store: "MemoryStore") -> "Outbox":
        return cls().advance(store)


def quiet(now: datetime) -> bool:
    """Is it 23:00-08:00 where they are?"""
    hour = _local(now).hour
    return hour >= QUIET_FROM_HOUR or hour < QUIET_UNTIL_HOUR


def slot_start(slot: str) -> Optional[datetime]:
    """When the slot ``"<date> <name>"`` begins, on their clock."""
    day, _, name = slot.partition(" ")
    hour = dict(SLOTS).get(name)
    try:
        date = datetime.fromisoformat(day)
    except ValueError:
        return None
    if hour is None:
        return None
    return date.replace(hour=hour, minute=0, second=0, microsecond=0).astimezone()


def slot_due(state: State, now: datetime) -> Optional[str]:
    """The thinking slot owed now, as ``"<date> <name>"``, or ``None``.

    Its own clock (C4): a slot is owed from its hour until
    :data:`SLOT_LATE_SECONDS` after it, once. Past that it is not served
    late; :func:`slots_missed` names it, and it is recorded as failed.
    """
    if quiet(now):
        return None
    local = _local(now)
    for name, hour in SLOTS:
        slot = f"{local.date().isoformat()} {name}"
        start = slot_start(slot)
        if start is None or slot in state.thought:
            continue
        if timedelta(0) <= now - start < timedelta(seconds=SLOT_LATE_SECONDS):
            return slot
    return None


def slots_missed(state: State, now: datetime) -> list[str]:
    """Slots in the last day whose window closed unserved, oldest first.
    Only slots that began after the log did: a slot from before omega existed
    was never owed. A log whose start cannot be read is owed nothing."""
    since = _parse(state.began) if state.began else None
    if since is None:
        return []
    local = _local(now)
    out = []
    for back in (1, 0):
        day = (local - timedelta(days=back)).date().isoformat()
        for name, _hour in SLOTS:
            slot = f"{day} {name}"
            start = slot_start(slot)
            if start is None or slot in state.thought or start < since:
                continue
            if timedelta(seconds=SLOT_LATE_SECONDS) <= now - start < timedelta(days=1):
                out.append(slot)
    return out


def conversing(heard_at: Optional[str], *, now: datetime) -> bool:
    """Did they send a message in the last :data:`SPOKE_SECONDS`?"""
    heard = _parse(heard_at) if heard_at else None
    if heard is None:
        return False
    return 0 <= (now - heard).total_seconds() < SPOKE_SECONDS


def present(presence: Any, heard_at: Optional[str], *, now: datetime) -> bool:
    """Are they here? At the Mac by its presence report (DL-073), or they sent
    a message in the last :data:`PRESENT_SECONDS`. Not knowing is "no"."""
    if machine.at_the_mac(presence, now=now):
        return True
    heard = _parse(heard_at) if heard_at else None
    if heard is None:
        return False
    age = (now - heard).total_seconds()
    return 0 <= age < PRESENT_SECONDS


def hold(state: State, now: datetime, *, here: bool, slot: bool = False) -> Optional[str]:
    """Why a non-urgent unprompted message must wait, or ``None`` to send it.

    Quiet hours first, then the day's budget, then whether they spoke in the
    last :data:`SPOKE_SECONDS` (B4-lite: never cut in on them), then
    (outside a thinking slot, which is its own good time) whether they are
    here. A day that cannot be read has no budget left: this breaks closed,
    toward holding.
    """
    if quiet(now):
        return HOLD_QUIET
    day = notice.local_date(now.isoformat())
    if day is None:
        return HOLD_CLOCK
    if state.delivered_on(day) >= MAX_PER_DAY:
        return HOLD_BUDGET
    if conversing(state.heard_at, now=now):
        return HOLD_SPOKE
    if not slot and not here:
        return HOLD_AWAY
    return None


#: How long after a brief-carrying turn that did not get the brief out omega
#: waits before trying it again (gap 0a): the next hour, not the next tick.
BRIEF_RETRY_SECONDS = 60 * 60


def brief_owed(state: State, now: datetime) -> Optional[str]:
    """Today's brief, as the slot ``"<date> brief"``, when the morning slot
    that carries it has been served (run, or written down as failed) and
    neither delivered nor held the brief (gap 0a); ``None`` otherwise. Only
    while a brief stands, which the caller knows. Not in quiet hours, and not
    within :data:`BRIEF_RETRY_SECONDS` of the last try."""
    if quiet(now):
        return None
    day = notice.local_date(now.isoformat())
    if day is None or day in state.briefed or f"{day} morning" not in state.thought:
        return None
    tried = _parse(state.brief_tried.get(day)) if state.brief_tried.get(day) else None
    if tried is not None and (now - tried).total_seconds() < BRIEF_RETRY_SECONDS:
        return None
    return f"{day} brief"


def thinking_left(state: State, now: datetime) -> bool:
    day = notice.local_date(now.isoformat())
    return day is not None and state.spending.get(day, 0) < THINKING_DAILY_TOKENS


def merged(drafts: Sequence[Draft]) -> str:
    """Held drafts as one message: one as it is, several under one line."""
    texts = [d.text.strip() for d in drafts if d.text.strip()]
    if len(texts) == 1:
        return texts[0]
    return "A few things from while you were away:\n\n" + "\n\n".join(texts)


def _one_line(text: str, cap: int = 300) -> str:
    one = " ".join(text.split())
    return one if len(one) <= cap else one[: cap - 1].rstrip() + "…"


def held_lines(drafts: Sequence[Draft], *, now: datetime) -> list[str]:
    """``(ago): words`` for each held draft, for the look's situation."""
    lines = []
    for d in drafts:
        then = _parse(d.at)
        minutes = int((now - then).total_seconds() // 60) if then else 0
        ago = f"{minutes} minutes ago" if minutes < 90 else f"{minutes // 60} hours ago"
        lines.append(f"{ago} ({d.source}): {_one_line(d.text)}")
    return lines


def carried_taint(
    drafts: Sequence[Draft], *, said: Sequence[str] = ()
) -> Optional[str]:
    """The ``untrusted`` source of an event that shows these drafts, or
    ``None`` when every one was written clean (DL-064, DL-108). ``said`` is
    the taint of the replies the same event shows as already told them
    (:func:`omega.notice.said_taint`), which repeat mail the same way.

    A draft a look wrote after reading their mail may repeat the mail word
    for word, and the event that carries it back (a re-check, a look or a
    thinking turn listing what is held) has no inbox fence for
    :func:`omega.turn.read_untrusted` to find. Naming the drafts' sources on
    that event taints the turn it starts and every turn that recalls it, as
    a watched page's tell already does."""
    sources = list(dict.fromkeys([*(t for d in drafts for t in d.tainted), *said]))
    if not sources:
        return None
    joined = ", ".join(sources)
    cap = episodes.MAX_HOLD_REASON
    return joined if len(joined) <= cap else joined[: cap - 1].rstrip() + "…"


#: How the re-check of held drafts begins (B4-lite).
RECHECK_LEAD = "Held drafts, not sent yet."


def recheck_text(
    drafts: Sequence[Draft], *, said: Sequence[tuple[str, str]], now: datetime
) -> str:
    """The judge's re-check of held drafts before any of them goes (B4-lite).

    Held drafts used to go out verbatim the moment they were here, hours
    after a watch had told them the same thing. Now the drafts and what was
    already said unprompted go to the judge, which sends what is still new
    as one message, or nothing (and they are dropped)."""
    lines = [
        f"{RECHECK_LEAD} You wrote these earlier and held them for a better "
        "moment. They are here now. Send only what is still true and that you "
        "have not already told them (below), as one short message. If none of "
        "it is still worth their attention, say SILENT and the drafts are "
        "dropped.",
        "",
        "Held:",
        *(f"- {line}" for line in held_lines(drafts, now=now)),
        "",
        "What you last told them, unprompted:",
    ]
    told = [f"- {at[:16]}: {_one_line(text)}" for at, text in said]
    lines.extend(told or ["- nothing"])
    text = "\n".join(lines)
    if len(text) > MAX_THINKING_CHARS:
        text = text[: MAX_THINKING_CHARS - 1].rstrip() + "…"
    return text


def thinking_text(
    slot: str,
    *,
    situation: str,
    promises: Sequence[str] = (),
    failing: Sequence[str] = (),
    topics: Sequence[str] = (),
    lost: Sequence[str] = (),
    tasks: Sequence[str] = (),
    held: Sequence[str] = (),
    items: Sequence[str] = (),
    routine: Sequence[str] = (),
) -> str:
    """The thinking turn's text: the ordinary look's situation, under a
    header saying what this look is for, with what is open over all of it.
    ``items`` are the told-ledger's keyed items new since they were last told
    or weighed (C3), which this turn decides on. ``routine`` are their
    routine meetings still to come today (DL-110), which get no warning of
    their own and so are folded in here when they help."""
    name = slot.rsplit(" ", 1)[-1]
    lines = [
        f"Thinking time ({name}). Twice a day you go over everything open and "
        "decide whether any of it is worth one message to them now. Most days "
        "the right answer is nothing: say SILENT unless something needs them.",
        "If something does, write it as one short message. Anything already "
        "drafted below that still matters goes into it; what you leave out is "
        "dropped.",
    ]
    for heading, items in (
        (KEYED_HEADING, items),
        ("Promises they made that are still open (theirs to keep):", promises),
        ("Things of yours that keep failing:", failing),
        ("Topics you keep up with for them:", topics),
        ("Sign-ins you lost:", lost),
        ("Research under way for them:", tasks),
        ("Already drafted and held for a better moment:", held),
        ("Their routine meetings still to come today (no warning of their own; "
         "fold them in if it helps):", routine),
    ):
        if items:
            lines.append("")
            lines.append(heading)
            lines.extend(f"- {_one_line(i)}" for i in items)
    lines.append("")
    text = "\n".join(lines) + "\n" + situation
    if len(text) > MAX_THINKING_CHARS:
        text = text[: MAX_THINKING_CHARS - 1].rstrip() + "…"
    return text


#: How the keyed items are headed, in a thinking turn and a compose (C3).
KEYED_HEADING = (
    "New since you last told them, by key (for each: tell, ask, fold into the "
    "one message, or let go):"
)
#: How a compose begins (C3): the presence-moment composer.
COMPOSE_LEAD = "Compose."


def compose_text(items: Sequence[str], *, situation: str) -> str:
    """The composer's turn at a presence moment (C3): they are here, the day
    has room, and keyed items came in since they were last told. One short
    message, or nothing."""
    lines = [
        f"{COMPOSE_LEAD} They are here and you may send one message now. The "
        "items below came in since you last told them or last weighed them. "
        "Tell them what needs them, as one short message, most pressing first, "
        "and leave out what does not. Never repeat what an item says you "
        "already told them unless it changed. If none of it is worth their "
        "attention now, say SILENT.",
        "",
        KEYED_HEADING,
        *(f"- {_one_line(i)}" for i in items),
        "",
    ]
    text = "\n".join(lines) + "\n" + situation
    if len(text) > MAX_THINKING_CHARS:
        text = text[: MAX_THINKING_CHARS - 1].rstrip() + "…"
    return text


@dataclass(frozen=True)
class Gate:
    """What the outbox decided for one unprompted turn, before it ran.

    ``hold`` is why it must wait (``None``: speak); ``queued`` the held drafts
    the turn carries; ``source`` what is drafting; ``at`` the turn's moment,
    which is the day the budget charges.
    """

    hold: Optional[str]
    source: str
    at: str
    queued: tuple[int, ...] = ()
    #: The told-ledger keys the turn speaks for (C1), written on its draft or
    #: its delivery.
    keys: tuple[str, ...] = ()


def settle(
    queue: Any, seq: int, gate: Gate, text: str, *, tainted: Sequence[str] = ()
) -> bool:
    """Write down what happens to a turn that would speak; ``True`` if held.

    Held: its words become a draft, and the drafts it was shown are folded
    into that one. Spoken: one delivery, carrying those drafts. Written before
    the turn's terminal record, so a crash between the two over-counts the
    budget, which is the direction that sends less.
    """
    if gate.hold:
        if not text.strip():
            return True
        queue.append(episodes.message_queued(
            text=_clip(text), source=gate.source, why=gate.hold, for_seq=seq,
            tainted=tainted, at=gate.at, keys=list(gate.keys),
        ))
        if gate.queued:
            # Stamped when it happened, not with the turn's moment (C0).
            queue.append(episodes.message_dropped(
                queued=list(gate.queued), why="folded into a newer held draft",
                for_seq=seq,
            ))
        return True
    queue.append(episodes.message_delivered(
        for_seq=seq, queued=list(gate.queued), at=gate.at, keys=list(gate.keys),
    ))
    return False


#: Why a turn that was shown held drafts and stayed silent dropped them.
DROPPED_WHY = {
    "thinking": "the thinking turn chose not to send them",
    "recheck": "the re-check found nothing in them still worth sending",
}


def dropped(queue: Any, seq: int, gate: Gate) -> None:
    """A turn that was shown held drafts and said nothing drops them, stamped
    when it happened (C0): the turn's own moment hid how long it ran."""
    if gate.queued:
        queue.append(episodes.message_dropped(
            queued=list(gate.queued),
            why=DROPPED_WHY.get(gate.source, "the turn chose not to send them"),
            for_seq=seq,
        ))


def _clip(text: str) -> str:
    text = text.strip()
    if len(text) <= episodes.MAX_QUEUED_TEXT:
        return text
    return text[: episodes.MAX_QUEUED_TEXT - 1].rstrip() + "…"
