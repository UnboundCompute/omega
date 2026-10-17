"""The calendar T-minus sense (C2, DL-110).

Code and nothing else, shaped like :mod:`omega.leave`: no model call decides
whether an event is worth a warning or when. :meth:`omega.executor.Executor.
calendar_tminus`, asked on every sense pass, reads the next hours of their
calendar and does what :func:`due` says is due, once per occurrence:

- **meeting**: a timed event that is not one of a series, warned
  :data:`MEETING_LEAD` before it starts, inside a :data:`LATE` window after
  that, so a pass that ran late still lands 25 to 35 minutes out.
- **flight**: an event that names a flight, warned :data:`FLIGHT_LEAD`
  before it leaves, with a leave-by from a labelled rough drive.
- **routine**: one instance of a series. Silent: the brief carries it
  (DL-114). Warned as a meeting only when it moved, when it clashes with
  another timed event, or when they asked to hear about routine ones too.
- All-day events are never warned, and nothing is ever said at or after an
  event's start: a warning that would arrive as "happening now" is dropped.

Each warning is a tell under a code key (C1), ``tminus:<the calendar key>``,
and the calendar key itself, so a restart or a second pass never tells it
twice and the brief knows it was told.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any, Optional, Sequence

from omega import ledger

__all__ = [
    "MEETING_LEAD",
    "FLIGHT_LEAD",
    "LATE",
    "FLIGHT_LATE",
    "AIRPORT_BEFORE",
    "ROUGH_DRIVE",
    "HORIZON",
    "READ_SECONDS",
    "ASK_KEY",
    "kind",
    "clash",
    "due",
    "key",
    "text",
    "routine_today",
    "ask_text",
]

MEETING_LEAD = timedelta(minutes=30)
FLIGHT_LEAD = timedelta(hours=3)
#: How long after its lead a warning may still go: a meeting's lands 30 to
#: 25 minutes out, a flight's 3 h to 2 h 45.
LATE = timedelta(minutes=5)
FLIGHT_LATE = timedelta(minutes=15)
#: The leave-by is a labelled guess, not a route: at the airport this long
#: before departure, after a drive this long. A live-traffic alarm is one
#: sentence away (DL-093).
AIRPORT_BEFORE = timedelta(minutes=90)
ROUGH_DRIVE = timedelta(minutes=45)
#: How far ahead each read looks, and how often the calendar is read; the
#: timing is checked against the last read on every pass.
HORIZON = timedelta(hours=12)
READ_SECONDS = 300
#: The told-ledger key of the one question about routine meetings (DL-110).
ASK_KEY = "ask:routine-warnings"

#: A flight names itself: "Flight to Pune (AI 101)" is how a mailed booking
#: lands on a calendar. A bare code is not trusted: "PR123" looks like one too.
_FLIGHT = re.compile(r"\bflight\b|\u2708", re.I)


def kind(o: Any) -> Optional[str]:
    """``"flight"``, ``"routine"``, ``"meeting"`` or ``None`` (all day)."""
    if o.all_day:
        return None
    if _FLIGHT.search(o.summary):
        return "flight"
    return "routine" if o.recurring else "meeting"


def clash(o: Any, others: Sequence[Any]) -> Optional[Any]:
    """Another timed event overlapping ``o``, if any."""
    for p in others:
        if p is o or p == o or p.all_day:
            continue
        if o.start < p.end and p.start < o.end:
            return p
    return None


def due(o: Any, now: datetime, others: Sequence[Any] = (), *,
        routine_warnings: bool = False) -> Optional[str]:
    """The warning due for ``o`` at ``now``: its :func:`kind`, or ``None``."""
    k = kind(o)
    if k is None:
        return None
    if k == "routine" and not (
        routine_warnings or o.moved_from is not None or clash(o, others) is not None
    ):
        return None
    lead, late = (FLIGHT_LEAD, FLIGHT_LATE) if k == "flight" else (MEETING_LEAD, LATE)
    opens = o.start - lead
    return k if opens <= now < min(opens + late, o.start) else None


def key(o: Any) -> str:
    return ledger.key("tminus", ledger.calendar_key(o))


def _hhmm(moment: datetime) -> str:
    return f"{moment.astimezone():%H:%M}"


def _quoted(o: Any) -> str:
    return '"' + o.summary.replace('"', "'") + '"'


def text(o: Any, now: datetime, others: Sequence[Any] = ()) -> str:
    """The warning, from the calendar's fields only."""
    minutes = max(1, round((o.start - now).total_seconds() / 60))
    where = f", at {o.location}" if o.location else ""
    if kind(o) == "flight":
        airport = o.start - AIRPORT_BEFORE
        leave_by = airport - ROUGH_DRIVE
        return (
            f"Your flight {_quoted(o)} leaves at {_hhmm(o.start)}, in about "
            f"{round(minutes / 60)} h{where}. To be at the airport by {_hhmm(airport)}, "
            f"leave by about {_hhmm(leave_by)}: a rough {int(ROUGH_DRIVE.total_seconds() // 60)}"
            f"-minute drive, not live traffic. Tell me \"I need to be at the airport by "
            f"{_hhmm(airport)}\" and I will watch the traffic."
        )
    head = f"{_quoted(o)} starts at {_hhmm(o.start)}, in {minutes} min{where}."
    if o.recurring and o.moved_from is not None:
        head += f" It moved from its usual {_hhmm(o.moved_from)}."
    other = clash(o, others) if o.recurring else None
    if other is not None:
        head += f" It clashes with {_quoted(other)} at {_hhmm(other.start)}."
    return head


def routine_today(occurrences: Sequence[Any], now: datetime) -> list[str]:
    """Their routine meetings still to come today, one line each, for the
    brief to fold in (DL-110, DL-114)."""
    today = now.astimezone().date()
    return [
        f"{_hhmm(o.start)} {o.summary}"
        for o in sorted(occurrences, key=lambda o: o.start)
        if kind(o) == "routine" and o.start >= now and o.start.astimezone().date() == today
    ]


def ask_text(o: Any) -> str:
    """The one question about routine meetings (DL-110)."""
    return (
        f"Your routine meetings, like {_quoted(o)}, get no warning from me unless "
        "one moves or clashes; they are in your morning brief. If you want a "
        "heads-up 30 minutes before them too, tell me \"warn me before routine "
        "meetings\"."
    )
