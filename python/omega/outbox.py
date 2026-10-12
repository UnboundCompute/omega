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

What is exempt never comes through here, and is byte-for-byte what it was:
things they asked for at a time (reminders, briefs, watches, research
follow-ups and reports), and omega telling them it cannot do what it was
asked (a lost sign-in, an outage, a promise it could not keep).

Everything is a fold over ``message.queued`` / ``message.delivered`` /
``message.dropped`` and the inbounds that carry ``queued`` or ``thinking``,
so the queue, the budget and the slots already served survive a restart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Iterable, Optional, Sequence

from omega import episodes, machine, notice
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
    "Draft",
    "State",
    "Outbox",
    "Gate",
    "quiet",
    "slot_due",
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

HOLD_QUIET = "quiet hours (23:00-08:00)"
HOLD_BUDGET = f"today already had its {MAX_PER_DAY} unprompted messages"
HOLD_AWAY = "they are not here; held for when they are, or the next thinking time"
HOLD_CLOCK = "couldn't tell the time"


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

    def delivered_on(self, day: str) -> int:
        return self.delivered.get(day, 0)


class Outbox:
    """The fold. Kept at the log's head by the executor, as its other views."""

    __slots__ = ("_through", "_drafts", "_taken", "_delivered", "_thought",
                 "_thinking", "_spending", "_heard_at")

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

    def apply(self, seq: int, payload: dict[str, Any]) -> None:
        self._through = seq
        kind = payload.get("kind")
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
            elif channel == "schedule" and isinstance(payload.get("queued"), list):
                # A tell that carries drafts takes them when it is written:
                # it is delivered as written, so nothing else may send them.
                self._taken.update(payload["queued"])
        elif kind == episodes.MESSAGE_QUEUED:
            self._drafts[seq] = Draft(
                seq=seq, text=str(payload["text"]), source=str(payload["source"]),
                at=str(payload["at"]),
            )
        elif kind == episodes.MESSAGE_DELIVERED:
            self._taken.update(payload.get("queued") or [])
            day = notice.local_date(str(payload.get("at", "")))
            if day is not None:
                self._delivered[day] = self._delivered.get(day, 0) + 1
        elif kind == episodes.MESSAGE_DROPPED:
            self._taken.update(payload.get("queued") or [])
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


def slot_due(state: State, now: datetime) -> Optional[str]:
    """The thinking slot owed now, as ``"<date> <name>"``, or ``None``.

    The latest slot today whose hour has come, if it has not been served. A
    slot missed (omega was down) is served late, the same day and outside
    quiet hours; one overtaken by the next slot is not served at all.
    """
    if quiet(now):
        return None
    local = _local(now)
    due = None
    for name, hour in SLOTS:
        if local.hour >= hour:
            due = f"{local.date().isoformat()} {name}"
    if due is None or due in state.thought:
        return None
    return due


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

    Quiet hours first, then the day's budget, then (outside a thinking slot,
    which is its own good time) whether they are here. A day that cannot be
    read has no budget left: this breaks closed, toward holding.
    """
    if quiet(now):
        return HOLD_QUIET
    day = notice.local_date(now.isoformat())
    if day is None:
        return HOLD_CLOCK
    if state.delivered_on(day) >= MAX_PER_DAY:
        return HOLD_BUDGET
    if not slot and not here:
        return HOLD_AWAY
    return None


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
) -> str:
    """The thinking turn's text: the ordinary look's situation, under a
    header saying what this look is for, with what is open over all of it."""
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
        ("Promises they made that are still open (theirs to keep):", promises),
        ("Things of yours that keep failing:", failing),
        ("Topics you keep up with for them:", topics),
        ("Sign-ins you lost:", lost),
        ("Research under way for them:", tasks),
        ("Already drafted and held for a better moment:", held),
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


def settle(queue: Any, seq: int, gate: Gate, text: str) -> bool:
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
            at=gate.at,
        ))
        if gate.queued:
            queue.append(episodes.message_dropped(
                queued=list(gate.queued), why="folded into a newer held draft",
                for_seq=seq, at=gate.at,
            ))
        return True
    queue.append(episodes.message_delivered(
        for_seq=seq, queued=list(gate.queued), at=gate.at,
    ))
    return False


def dropped(queue: Any, seq: int, gate: Gate) -> None:
    """A turn that was shown held drafts and said nothing drops them."""
    if gate.queued:
        queue.append(episodes.message_dropped(
            queued=list(gate.queued), why="the thinking turn chose not to send them",
            for_seq=seq, at=gate.at,
        ))


def _clip(text: str) -> str:
    text = text.strip()
    if len(text) <= episodes.MAX_QUEUED_TEXT:
        return text
    return text[: episodes.MAX_QUEUED_TEXT - 1].rstrip() + "…"
