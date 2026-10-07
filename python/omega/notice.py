"""Wake condition (b) — the pass omega runs when nobody asked. DL-011, DL-061, DL-068.

**This is not a second engine.** It appends one ordinary ``message.inbound`` on
the :data:`CHANNEL` channel and gets out of the way: the drain picks it up, the
ordinary turn runs, the judge decides, and a reply — if there is one — reaches
the tray because :meth:`omega.channel.Server` fans every new log update out to
its subscribers. DL-011 chose that shape deliberately, and the reason is worth
keeping in front of whoever reads this next: two engines would drift, *"the
unprompted path would grow its own memory view, its own voice, its own idea of
what's open, and initiative would quietly become a different agent wearing
omega's name."* So there is no new episode kind here, no new verdict, no second
prompt and no delivery path. A fire on the ``schedule`` channel already proved
the door works (`episodes.py:121`); this walks through the same one.

**The whole of this module is the look's rate limit and the situation text.**
Everything else already existed.

**DL-068 lifted the day budget.** DL-061 capped unprompted speech at one a
day on the argument that volume, not wrongness, is the failure mode. The
evidence came back the other way: sixty looks, zero spoken, and the person
asking for omega to say things "based on condition of system or something it
noticed" — not "x message per day". So speech is now bounded by *novelty*
rather than by a count: the situation shows the look what it already told
them (:attr:`Standing.said`), and the judge is told that repeating it is a
reason for silence. The day cap survives only as an opt-in argument to
:func:`may_look`. The risk this trades into is the firehose DL-061 feared, and
the ledger records it as disagree-and-log; the guard is novelty plus a look
gap that still only fires when the conversation has gone quiet.

**Undeterminable breaks closed**, as everywhere on this path. An unreadable
timestamp, a missing local date, a store that cannot be scanned back far enough
to prove today is still unspent — each one means *do not look*, never *look
anyway*. The capability this gates is interrupting a person, so the direction a
broken check falls is the whole of its safety.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Optional, Sequence

from omega import episodes

if TYPE_CHECKING:  # pragma: no cover - typing only
    from omega.derive import Claim, Moment, OpenBlock
    from omega.memory import MemoryStore
    from omega.schedule import Schedule

__all__ = [
    "CHANNEL",
    "LOOK_EVERY_SECONDS",
    "MAX_PER_DAY",
    "MAX_L1_CHARS",
    "SAID_KEPT",
    "SAID_WITHIN_SECONDS",
    "UNPROMPTED_CHANNELS",
    "Standing",
    "local_date",
    "standing",
    "may_look",
    "situation",
]

#: The channel an unprompted event arrives on. The *only* thing distinguishing
#: this from a typed message, exactly as ``schedule_id`` is the only thing
#: distinguishing a clock fire (`episodes.py:121`).
CHANNEL = "self"

#: The channels nobody typed on: omega's own look and the clock. What omega
#: said on these is what it said *unprompted*, which is what novelty is about.
UNPROMPTED_CHANNELS = frozenset({CHANNEL, "schedule"})

#: How often omega may *look*. Not how often it may speak. Halved from an hour
#: by DL-068, which also gave the look a machine sense that can change in it.
LOOK_EVERY_SECONDS = 30 * 60

#: No cap on unprompted speech per day (DL-068). ``None`` rather than a large
#: number, so that "unlimited" cannot be mistaken for a tuned value.
MAX_PER_DAY: Optional[int] = None

#: The novelty memory: the last few things omega said unprompted, from the
#: last day. Small, because it is a prompt, and a day, because "I told you
#: this yesterday" stops being a reason for silence after about that long.
SAID_KEPT = 5
SAID_WITHIN_SECONDS = 24 * 60 * 60

#: The situation text is a prompt, so it is capped like one. Small on purpose:
#: the turn's own recall already puts recent conversation and the matched claims
#: in front of the judge, so this adds only what a turn would not otherwise
#: show.
MAX_L1_CHARS = 2_000


def local_date(at: str) -> Optional[str]:
    """The local-clock date of an episode timestamp, or ``None`` if unreadable.

    ``None`` is a real answer and callers must break closed on it, for
    :func:`omega.derive.local_hour`'s reason: the rate limit is a statement
    about the calendar on the wall, and a day that cannot be determined must not
    read as a day with budget left in it.
    """
    try:
        parsed = datetime.fromisoformat(at)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        # What the log writes is UTC (`episodes.now`), so a naive stamp is read
        # as UTC and not as local — reading it as local would shift every day
        # boundary by the machine's offset.
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone().date().isoformat()


@dataclass(frozen=True)
class Standing:
    """What the unprompted pass has already done, folded out of the log.

    Kept as one value rather than three loose returns because the three are only
    meaningful together: "may omega look now" is a question about the last look
    *and* the days already spent *and* how far the fold actually reached.
    """

    #: The ``at`` of the newest unprompted event, or ``None`` if there is none.
    looked_at: Optional[str] = None

    #: The ``at`` of the newest inbound a *person* sent. Not the clock: a watch
    #: firing hourly is not them talking, and counting it both held every look
    #: back behind the watch and told the look "you last heard from them 30
    #: minutes ago" when they had been gone for hours (DL-076). Folded here
    #: rather than by a second scan, because "how long since I heard from
    #: them" is part of the same question.
    heard_at: Optional[str] = None

    #: Local dates on which an unprompted turn *spoke*. A day omega looked at
    #: and stayed silent is deliberately absent: silence spends no budget,
    #: which is what makes it a first-class outcome rather than a cheap failure.
    spoke_on: frozenset[str] = frozenset()

    #: The seq this fold reached. Reported for tests and diagnostics; nothing
    #: resumes from it, see :func:`standing`.
    through: int = 0

    #: Unprompted events whose turn has not completed yet. A pass that appended
    #: an event and then looked again before the turn ran would stack events on
    #: one another, so an outstanding one means *do not look*.
    outstanding: int = 0

    #: ``(at, reply)`` for the newest unprompted turns that spoke, oldest first,
    #: at most :data:`SAID_KEPT`. The novelty guard (DL-068): what the look is
    #: shown so that "already told them" is something it can see rather than
    #: something it is asked to remember.
    said: tuple[tuple[str, str], ...] = ()

    #: ``(key, at, line)`` for each machine line flagged at the newest look
    #: that carried readings: the look at which that flag *started*, and the
    #: line as it read then (DL-076). This is what lets a look tell a flag
    #: that is new from one it has been staring at all day, in code, instead
    #: of leaving the judge to diff two prompts it never sees side by side.
    flagged_since: tuple[tuple[str, str, str], ...] = ()

    #: Whether any look so far carried its readings. Without one there is no
    #: earlier look to compare with, so nothing is called new.
    machine_known: bool = False


def standing(store: "MemoryStore") -> Standing:
    """Fold the log for what the unprompted pass has already done.

    **Scans the whole log, and does not resume from a cursor.** That is a
    deliberate refusal of the obvious optimisation, because a forward fold from
    a saved seq would carry no :attr:`Standing.spoke_on` from before it — and a
    day-budget check that cannot see this morning's nudge concludes the day is
    unspent and nudges again. The cheap version fails *open*, on the one limit
    whose whole purpose is to fail closed. A full scan of an append-only log,
    at most hourly, is the honest price; revisit it when there is evidence the
    cost is real rather than imagined.
    """
    # seq -> the stamp of the look itself, not of whatever finished it.
    mine: dict[int, str] = {}
    # Every unprompted inbound (look or clock fire), for the novelty memory.
    unprompted: dict[int, str] = {}
    said: list[tuple[str, str]] = []
    completed: set[int] = set()
    looked_at: Optional[str] = None
    heard_at: Optional[str] = None
    spoke: set[str] = set()
    through = 0
    flagged: dict[str, tuple[str, str]] = {}
    machine_known = False

    for episode in store.episodes_since(0):
        through = episode.seq
        # ``episode.payload`` is the raw frame, not a mapping - decoding is the
        # only way to read it (see executor.py:338). An earlier version of this
        # fold treated it as a dict behind an ``isinstance`` guard, which turned
        # every episode into a silent skip: the fold ran, found nothing, and
        # ``may_look`` therefore said yes forever. Fail-closed guards belong on
        # values that might legitimately be absent, never on a decode.
        payload = episodes.decode(episode.payload)
        kind = payload.get("kind")
        if kind == episodes.MESSAGE_INBOUND:
            at = payload.get("at")
            if payload.get("channel") in UNPROMPTED_CHANNELS:
                unprompted[episode.seq] = at if isinstance(at, str) else ""
            if payload.get("channel") == CHANNEL:
                mine[episode.seq] = at if isinstance(at, str) else ""
                if isinstance(at, str):
                    looked_at = at
                    lines = payload.get("machine")
                    if isinstance(lines, list):
                        machine_known = True
                        flagged = _fold_flags(flagged, lines, at)
            elif (
                isinstance(at, str)
                and payload.get("channel") not in UNPROMPTED_CHANNELS
            ):
                heard_at = at
        elif kind == episodes.TURN_COMPLETED:
            for_seq = payload.get("for_seq")
            reply = payload.get("reply")
            if (
                for_seq in unprompted
                and payload.get("outcome") == "spoke"
                and isinstance(reply, str)
                and reply.strip()
            ):
                said.append((unprompted[for_seq], reply.strip()))
                del said[:-SAID_KEPT]
            if for_seq in mine:
                completed.add(for_seq)
                if payload.get("outcome") == "spoke":
                    # Read from the *look's* stamp, not the completion's: a turn
                    # raised at 23:59 that finishes at 00:01 spent the day it was
                    # raised for, and charging it to the next day would hand that
                    # day a second nudge it has not earned.
                    day = local_date(mine[for_seq])
                    if day is not None:
                        spoke.add(day)
        elif kind == episodes.TURN_BLOCKED:
            for_seq = payload.get("for_seq")
            if for_seq in mine:
                # Blocked is finished as far as this cursor is concerned: the
                # turn is over and waiting on a person, so it is not an event
                # still sitting unhandled in the queue.
                completed.add(for_seq)

    return Standing(
        looked_at=looked_at,
        heard_at=heard_at,
        spoke_on=frozenset(spoke),
        through=through,
        outstanding=len(set(mine) - completed),
        said=tuple(said),
        flagged_since=tuple((k, a, l) for k, (a, l) in flagged.items()),
        machine_known=machine_known,
    )


def machine_key(line: str) -> str:
    """What a machine line is *about*: ``"mac: Battery"`` for
    ``"mac: Battery: 20%, on battery - LOW"``. The value never contains
    ``": "``, so everything before the last one is the device and the reading."""
    return line.rsplit(": ", 1)[0]


def is_flagged(line: str) -> bool:
    """Did code flag this reading (``- LOW``, ``- the machine is saturated``)?
    :func:`omega.machine.describe` puts the flag after ``" - "`` in the value."""
    return " - " in line.rsplit(": ", 1)[-1]


def _fold_flags(
    flagged: dict[str, tuple[str, str]], lines: Sequence[object], at: str
) -> dict[str, tuple[str, str]]:
    """The flags after one look: kept with their start if still flagged, started
    now if newly flagged, dropped if the reading cleared or is gone."""
    out: dict[str, tuple[str, str]] = {}
    for line in lines:
        if isinstance(line, str) and is_flagged(line):
            key = machine_key(line)
            out[key] = flagged.get(key, (at, line))
    return out


def may_look(
    st: Standing,
    *,
    now: datetime,
    look_every_seconds: int = LOOK_EVERY_SECONDS,
    max_per_day: Optional[int] = MAX_PER_DAY,
) -> bool:
    """The look's limits, in the order that spends the least.

    The day check is off by default since DL-068 (``max_per_day=None``). When
    a caller passes a number it still runs first, because it is the one that
    can skip the look entirely.

    ``look_every_seconds`` is a gap against **two** stamps, the last look and
    the last thing *heard*, and the second one is not an extra nicety. DL-011
    words wake condition (b) as "time passed", and time passing means the
    conversation went quiet - not merely that the clock advanced since the last
    unprompted pass. Without the quiet half, the first thing omega does after
    being spoken to is look at what is open and consider interrupting, which
    re-judges the turn that just ran against a situation that turn already saw.
    It is also what makes idle omega idle: a person mid-conversation is the one
    moment a nudge is certain to be unwelcome, because they are already here.
    """
    if st.outstanding:
        return False

    if max_per_day is not None:
        today = local_date(now.isoformat())
        if today is None:
            return False  # undeterminable day, so no budget
        if len(st.spoke_on & {today}) >= max_per_day:
            return False

    gap = timedelta(seconds=look_every_seconds)
    for stamp in (st.looked_at, st.heard_at):
        if stamp is None:
            continue
        try:
            last = datetime.fromisoformat(stamp)
        except (TypeError, ValueError):
            return False  # unreadable stamp, so break closed
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        if (now - last) < gap:
            return False

    return True


def _clock(now: datetime) -> str:
    local = now.astimezone()
    return local.strftime("%A %d %B, %H:%M")


def _since(now: datetime, at: Optional[str]) -> Optional[str]:
    if not at:
        return None
    try:
        then = datetime.fromisoformat(at)
    except (TypeError, ValueError):
        return None
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)
    minutes = int((now - then).total_seconds() // 60)
    if minutes < 2:
        return "just now"
    if minutes < 90:
        return f"{minutes} minutes ago"
    hours = minutes // 60
    if hours < 36:
        return f"{hours} hours ago"
    return f"{hours // 24} days ago"


def situation(
    *,
    now: datetime,
    blocks: Sequence["OpenBlock"] = (),
    claims: Sequence["Claim"] = (),
    schedules: Sequence["Schedule"] = (),
    moments: Sequence["Moment"] = (),
    heard_at: Optional[str] = None,
    machine: Sequence[str] = (),
    said: Sequence[tuple[str, str]] = (),
    flagged_since: Sequence[tuple[str, str, str]] = (),
    machine_known: bool = False,
    max_chars: int = MAX_L1_CHARS,
) -> str:
    """The L1 text: what is open, as far as omega already knows it.

    **Only what omega already has** (DL-061, decided with the user): open
    obligations, what it has been taught, standing schedules, recent moments,
    and the clock. A
    live now-sense — frontmost app, window titles, the last few minutes — was
    the alternative and was rejected for v1 as the largest privacy surface omega
    would take on, needing tray work that is not ours.

    ``moments`` is the one part of this text that can differ between two looks
    an hour apart (DL-062), and that is the whole reason it is here. Everything
    else on this list is timeless or nearly so: the same claims, the same
    schedules, a clock that moved. Two unprompted passes over that produce two
    byte-identical digests and therefore the same verdict forever, which is how
    a loop that runs on time ends up never having anything to say. What happened
    is the only input that changes on its own.

    ``machine`` (DL-068) is the other input that changes on its own: disk,
    battery, load, already worded and flagged by :mod:`omega.machine`. ``said``
    is the novelty guard - what omega already told them unprompted in the last
    day - rendered so that repeating it is visibly repeating it.

    Recent conversation is deliberately **not** rendered here. The turn's own
    recall step already puts it in front of the judge, and writing it twice
    would spend the prompt twice to say one thing.

    Written as a description of a situation and never as a person speaking. The
    judge is told separately which channel this arrived on, but the text itself
    must not read like a request, or omega answers a question nobody asked.
    """
    lines = [
        "Nobody asked for this. Time passed and you are looking at what is open.",
        f"It is {_clock(now)}.",
    ]

    heard = _since(now, heard_at)
    if heard is None:
        lines.append("You have not heard from them yet.")
    else:
        lines.append(f"You last heard from them {heard}.")

    if machine:
        lines.append("")
        lines.append("Their machine right now:")
        since = {key: (at, then) for key, at, then in flagged_since}
        for line in machine:
            lines.append(f"- {line}{_flag_note(line, since, machine_known, now)}")

    if blocks:
        lines.append("")
        lines.append("Waiting on them:")
        for block in blocks:
            lines.append(f"- {getattr(block, 'needs', '')}")

    if schedules:
        lines.append("")
        lines.append("Standing schedules:")
        for sched in schedules:
            # The first line only: what the schedule is about. A watch's
            # instruction goes on to tell *its own* turn to answer "(nothing
            # new)" when it finds nothing, and pasting that into the look
            # handed the look the same way out (DL-076).
            instruction = str(getattr(sched, "instruction", "")).strip()
            lines.append(f"- {instruction.splitlines()[0] if instruction else ''}")

    if moments:
        lines.append("")
        # "Lately" and not "Recent events": the heading has to read as
        # unfinished business rather than as a log, or the judge treats it as
        # something already dealt with and there is nothing to say about it.
        lines.append("Lately:")
        for moment in moments:
            lines.append(f"- {getattr(moment, 'text', '')}")

    recent = []
    for at, text in said:
        ago = _since(now, at)
        if ago is None:
            continue
        try:
            then = datetime.fromisoformat(at)
        except (TypeError, ValueError):
            continue
        if then.tzinfo is None:
            then = then.replace(tzinfo=timezone.utc)
        if (now - then).total_seconds() <= SAID_WITHIN_SECONDS:
            recent.append((ago, text))
    if recent:
        lines.append("")
        lines.append(
            "You already told them, unprompted (do not repeat it unless it "
            "has got worse or more urgent since):"
        )
        for ago, text in recent:
            one = " ".join(text.split())
            if len(one) > 200:
                one = one[:199].rstrip() + "…"
            lines.append(f"- {ago}: {one}")

    lines.append("")
    lines.append(
        f"You have written down {len(claims)} things about them; the ones that "
        f"matched this moment are already in front of you, and `recall` will "
        f"show you the rest."
    )

    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"
    return text


def _flag_note(
    line: str,
    since: dict[str, tuple[str, str]],
    machine_known: bool,
    now: datetime,
) -> str:
    """How long a flagged reading has been flagged, or that it is new (DL-076).

    The judge sees one look at a time, so it cannot tell a battery that just
    went LOW from a disk that has been LOW for a week, and the first is a
    nudge while the second, already told, is not. Code can tell, so code says.
    An unflagged line gets nothing, and with no earlier look to compare with
    nothing is called new.
    """
    if not is_flagged(line):
        return ""
    started = since.get(machine_key(line))
    if started is None:
        return " (new since your last look)" if machine_known else ""
    at, then = started
    ago = _since(now, at)
    if ago is None:
        return ""
    was = then.rsplit(": ", 1)[-1].split(" - ")[0]
    return f" (first flagged {ago}, when it read {was})"
