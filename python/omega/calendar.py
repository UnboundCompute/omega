"""Reading what is on their day — DL-080.

omega knew how the person works at a computer and nothing about the shape of
their day. This is the calendar half of fixing that: it reads Google
Calendar's *secret address in iCal format* — one or more https URLs in
``OMEGA_CALENDAR_ICS`` — and turns the next few hours into lines a look or a
tool can show.

**Why an ICS URL and not the Calendar API.** The secret address is read-only by
construction and needs no OAuth client, consent screen or refresh token. For a
sense that only ever *reads*, that is a great deal less surface for the same
answer.

**Why a parser of our own.** The repo has one dependency, and the RFC 5545
subset Google actually emits is small: VEVENTs with start, end, summary,
location and status, TZIDs, all-day dates, EXDATE, RECURRENCE-ID overrides, and
the common RRULE parts. That fits in one legible file. What does not fit is
said out loud rather than guessed: a recurring event whose rule uses a part
this reader does not implement (BYSETPOS, BYHOUR, an RDATE, ...) is reported as
*repeats, next time couldn't be determined*. A wrong time is worse than an
admitted gap, because a person acts on a time.

**What this does not read.** DESCRIPTION. Invite text is written by strangers,
can be long, and is exactly the injection surface a fetched web page is. Titles
and locations are read, but they are other people's text too, so they are
stripped of control characters and truncated before anything else sees them.

**The URL is a credential.** Anyone holding it can read the calendar, so it
never appears in an exception, a returned line, or a chained traceback: every
failure is re-raised as :class:`CalendarUnavailable` with a short reason built
here, and feeds are named by position ("feed 2 of 3"), never by address.

**Three-valued, always.** Not configured, couldn't read, and read-and-empty are
three different answers, and :func:`look_lines` keeps them apart. An empty
list is only ever "not configured" — it is never allowed to mean all-clear.
"""

from __future__ import annotations

import http.client
import os
import re
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime, time as dtime, timedelta, timezone, tzinfo
from typing import Callable, Iterator, Mapping, Optional
from zoneinfo import ZoneInfo

__all__ = [
    "ENV",
    "LOOK_HOURS",
    "CACHE_SECONDS",
    "MAX_STEPS",
    "CalendarUnavailable",
    "Occurrence",
    "Window",
    "Calendar",
    "feeds",
    "get",
    "parse",
    "occurrences",
    "read_window",
    "lines",
    "look_lines",
    "tool_text",
]

#: Where the secret addresses come from. One or more https URLs, separated by
#: whitespace and/or commas — a person with a work and a personal calendar has
#: two.
ENV = "OMEGA_CALENDAR_ICS"

#: How far ahead a look reads. Twelve hours covers "the rest of the working
#: day" from any morning look without pulling tomorrow into every evening one.
LOOK_HOURS = 12

#: How long fetched text is reused. A look runs often; the calendar changes
#: rarely, and Google itself serves this feed from a cache that lags by minutes.
CACHE_SECONDS = 300

#: Ceiling on recurrence periods walked for one event in one query. A daily
#: event since 1990 is ~13,000 periods even without skipping ahead; the cap is
#: there for the pathological rule, so that one feed can never stall a look.
MAX_STEPS = 50_000

#: Text limits for other people's words.
MAX_SUMMARY = 120
MAX_LOCATION = 60

#: The message a person sees when nothing is connected, with the path to the
#: setting, because "set OMEGA_CALENDAR_ICS" alone does not say where the URL is.
NOT_CONFIGURED = (
    "no calendar is connected: set OMEGA_CALENDAR_ICS on the host "
    "(Google Calendar → Settings → your calendar → Integrate calendar → "
    "Secret address in iCal format)"
)


class CalendarUnavailable(Exception):
    """A calendar could not be read. The message is a short human reason and
    never contains the feed's URL."""


@dataclass(frozen=True)
class Occurrence:
    """One concrete instance of an event, in aware time."""

    start: datetime
    end: datetime
    all_day: bool
    summary: str
    location: str
    #: True when the event named a time zone this host could not resolve, so
    #: its wall-clock times were read in the process zone instead.
    approximate: bool = False


@dataclass(frozen=True)
class Window:
    """What a read of one span of time found, across every feed."""

    occurrences: tuple[Occurrence, ...]
    #: Summaries of recurring events whose rule this reader cannot expand.
    unreadable: tuple[str, ...] = ()
    #: Per-feed failures, named by position, never by URL.
    errors: tuple[str, ...] = ()


# --- the process zone ----------------------------------------------------------


class _Local(tzinfo):
    """The process's own zone, asked of the C library at each use.

    Not a snapshot like ``datetime.now().astimezone().tzinfo`` (a fixed offset,
    wrong on the far side of a DST change) and not a guess at an IANA name:
    ``time.localtime`` is the same source every other ``astimezone()`` in omega
    uses, and it honours a ``TZ`` change after ``time.tzset()``.
    """

    def _offset(self, dt: datetime) -> time.struct_time:
        try:
            stamp = time.mktime(
                (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, 0, 0, -1)
            )
            return time.localtime(stamp)
        except (OverflowError, ValueError, OSError):
            return time.localtime(0)

    def utcoffset(self, dt: Optional[datetime]) -> timedelta:
        if dt is None:
            return timedelta(seconds=-time.timezone)
        return timedelta(seconds=self._offset(dt).tm_gmtoff)

    def dst(self, dt: Optional[datetime]) -> timedelta:
        if dt is None:
            return timedelta(0)
        return timedelta(hours=1) if self._offset(dt).tm_isdst > 0 else timedelta(0)

    def tzname(self, dt: Optional[datetime]) -> str:
        return time.localtime().tm_zone if dt is None else self._offset(dt).tm_zone

    def fromutc(self, dt: datetime) -> datetime:
        stamp = (dt.replace(tzinfo=None) - datetime(1970, 1, 1)).total_seconds()
        lt = time.localtime(stamp)
        return datetime(*lt[:6], dt.microsecond, tzinfo=self)

    def __repr__(self) -> str:
        return "<process local zone>"


_LOCAL = _Local()


# --- reading the text ------------------------------------------------------------


def _unfold(text: str) -> list[str]:
    """Logical content lines: a physical line starting with a space or tab
    continues the one before it (RFC 5545 §3.1)."""
    out: list[str] = []
    for raw in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if raw[:1] in (" ", "\t") and out:
            out[-1] += raw[1:]
        elif raw:
            out.append(raw)
    return out


def _split_unquoted(text: str, sep: str) -> list[str]:
    """Split on ``sep`` everywhere except inside double quotes."""
    parts, buf, quoted = [], [], False
    for ch in text:
        if ch == '"':
            quoted = not quoted
        if ch == sep and not quoted:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    parts.append("".join(buf))
    return parts


def _property(line: str) -> Optional[tuple[str, dict[str, str], str]]:
    """``NAME;PARAM=val;PARAM="quoted:val":VALUE`` → (NAME, params, VALUE).

    The value starts at the first colon *outside* quotes, because a quoted
    parameter (a TZID, an ALTREP URL) may itself contain colons.
    """
    quoted = False
    for i, ch in enumerate(line):
        if ch == '"':
            quoted = not quoted
        elif ch == ":" and not quoted:
            head, value = line[:i], line[i + 1 :]
            break
    else:
        return None
    name, *raw_params = _split_unquoted(head, ";")
    params: dict[str, str] = {}
    for p in raw_params:
        key, _, val = p.partition("=")
        params[key.strip().upper()] = val.strip().strip('"')
    return name.strip().upper(), params, value


_ESCAPE = re.compile(r"\\([\\;,nN])")


def _unescape(value: str) -> str:
    """TEXT value escapes: ``\\n``, ``\\,``, ``\\;``, ``\\\\``."""
    return _ESCAPE.sub(lambda m: "\n" if m.group(1) in "nN" else m.group(1), value)


def _clean(text: str, limit: int) -> str:
    """Other people's text, made safe to show: control and format characters
    dropped (no terminal escapes, no bidi overrides), whitespace collapsed,
    truncated."""
    kept = "".join(
        " " if ch.isspace() else ch
        for ch in text
        if ch.isspace() or unicodedata.category(ch) not in ("Cc", "Cf")
    )
    kept = " ".join(kept.split())
    return kept if len(kept) <= limit else kept[: limit - 1].rstrip() + "…"


# --- dates and times ---------------------------------------------------------------


@dataclass(frozen=True)
class _When:
    """One DTSTART-shaped value: a naive wall-clock time and the zone it is in.

    All-day values carry midnight and the process zone, so every value can go
    through the same arithmetic; ``all_day`` says how to show it.
    """

    wall: datetime
    tz: tzinfo
    all_day: bool
    approximate: bool = False

    @property
    def aware(self) -> datetime:
        return self.wall.replace(tzinfo=self.tz)


def _zone(tzid: str) -> tuple[tzinfo, bool]:
    """A TZID → (zone, approximate). Unknown names fall back to the process
    zone and say so, rather than failing the whole event."""
    name = tzid.strip().lstrip("/")
    try:
        return ZoneInfo(name), False
    except (ValueError, OSError, KeyError):
        return _LOCAL, True


def _when_one(value: str, params: Mapping[str, str]) -> _When:
    """Parse one DATE or DATE-TIME value. Raises ValueError on garbage."""
    value = value.strip()
    if params.get("VALUE", "").upper() == "DATE" or len(value) == 8:
        d = datetime.strptime(value, "%Y%m%d")
        return _When(d, _LOCAL, True)
    utc = value.endswith(("Z", "z"))
    wall = datetime.strptime(value.rstrip("Zz")[:15], "%Y%m%dT%H%M%S")
    if utc:
        return _When(wall, timezone.utc, False)
    if "TZID" in params:
        tz, approximate = _zone(params["TZID"])
        return _When(wall, tz, False, approximate)
    return _When(wall, _LOCAL, False)  # floating: their wall clock, wherever they are


def _whens(value: str, params: Mapping[str, str]) -> list[_When]:
    """A comma list of values (EXDATE allows several per property)."""
    out = []
    for part in value.split(","):
        if part.strip():
            try:
                out.append(_when_one(part, params))
            except ValueError:
                continue
    return out


_DURATION = re.compile(
    r"^([+-])?P(?:(\d+)W)?(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?$"
)


def _duration(value: str) -> Optional[timedelta]:
    m = _DURATION.match(value.strip().upper())
    if not m:
        return None
    sign, w, d, h, mi, s = m.groups()
    span = timedelta(
        weeks=int(w or 0), days=int(d or 0), hours=int(h or 0),
        minutes=int(mi or 0), seconds=int(s or 0),
    )
    return timedelta(0) if sign == "-" else span


class _Marks:
    """A set of instances named by EXDATE or RECURRENCE-ID.

    Kept two ways because the values may be dated or timed independently of
    the series they mark: timed values match by instant, and every value also
    matches by its wall date, which is how a dated mark names an instance.
    """

    def __init__(self) -> None:
        self.instants: set[datetime] = set()
        self.dates: set[date] = set()

    def add(self, w: _When) -> None:
        if w.all_day:
            self.dates.add(w.wall.date())
        else:
            self.instants.add(w.aware.astimezone(timezone.utc))

    def has(self, start: _When) -> bool:
        if start.all_day:
            return start.wall.date() in self.dates or any(
                i.astimezone(start.tz).date() == start.wall.date() for i in self.instants
            )
        return (
            start.aware.astimezone(timezone.utc) in self.instants
            or start.wall.date() in self.dates
        )


# --- events -------------------------------------------------------------------------


@dataclass
class _Event:
    uid: str
    summary: str
    location: str
    start: _When
    length: timedelta  # in wall-clock terms of start's zone
    cancelled: bool
    rrules: list[str] = field(default_factory=list)
    has_rdate: bool = False
    exdates: _Marks = field(default_factory=_Marks)
    recurrence_id: Optional[_When] = None


@dataclass
class Calendar:
    """A parsed feed. Opaque to callers: pass it to :func:`occurrences`."""

    masters: list[_Event]
    #: uid → the instances an override replaces (moved or cancelled).
    overridden: dict[str, _Marks]
    #: Non-cancelled overrides, each shown as a single event of its own.
    moved: list[_Event]


def _event(props: dict[str, list[tuple[dict[str, str], str]]]) -> Optional[_Event]:
    """Build one event from its properties, or None if it has no usable start."""

    def first(name: str) -> Optional[tuple[dict[str, str], str]]:
        return props.get(name, [None])[0]

    dtstart = first("DTSTART")
    if dtstart is None:
        return None
    try:
        start = _when_one(dtstart[1], dtstart[0])
    except ValueError:
        return None

    length: Optional[timedelta] = None
    dtend = first("DTEND")
    if dtend is not None:
        try:
            end = _when_one(dtend[1], dtend[0])
        except ValueError:
            end = None
        if end is not None and end.all_day == start.all_day:
            if start.all_day:
                length = end.wall - start.wall
            else:  # read the end on the start's wall clock, so a weekly series keeps it
                length = end.aware.astimezone(start.tz).replace(tzinfo=None) - start.wall
    elif first("DURATION") is not None:
        length = _duration(first("DURATION")[1])
    if start.all_day:
        if length is None or length <= timedelta(0):
            length = timedelta(days=1)
        length = timedelta(days=max(1, length.days))
    elif length is None or length < timedelta(0):
        length = timedelta(0)

    summary = first("SUMMARY")
    location = first("LOCATION")
    status = first("STATUS")
    ev = _Event(
        uid=(first("UID") or ({}, ""))[1].strip(),
        summary=_clean(_unescape(summary[1]), MAX_SUMMARY) if summary else "",
        location=_clean(_unescape(location[1]), MAX_LOCATION) if location else "",
        start=start,
        length=length,
        cancelled=bool(status) and status[1].strip().upper() == "CANCELLED",
        rrules=[v for _, v in props.get("RRULE", [])],
        has_rdate="RDATE" in props,
    )
    for params, value in props.get("EXDATE", []):
        for w in _whens(value, params):
            ev.exdates.add(w)
    rid = first("RECURRENCE-ID")
    if rid is not None:
        found = _whens(rid[1], rid[0])
        ev.recurrence_id = found[0] if found else None
    return ev


def parse(text: str) -> Calendar:
    """Parse ICS text. Never raises on bad content: what can't be read is skipped.

    Only VEVENTs inside a VCALENDAR are read, and only their own properties —
    a VALARM nested inside an event has its own SUMMARY and DESCRIPTION, and
    those must not leak into the event they belong to.
    """
    stack: list[str] = []
    props: dict[str, list[tuple[dict[str, str], str]]] = {}
    events: list[_Event] = []
    for line in _unfold(text):
        prop = _property(line)
        if prop is None:
            continue
        name, params, value = prop
        if name == "BEGIN":
            stack.append(value.strip().upper())
            if stack[-1] == "VEVENT":
                props = {}
            continue
        if name == "END":
            ending = value.strip().upper()
            if ending not in stack:
                continue  # a stray END closes nothing
            while stack.pop() != ending:
                pass
            if ending == "VEVENT" and "VCALENDAR" in stack:
                ev = _event(props)
                if ev is not None:
                    events.append(ev)
            continue
        if stack and stack[-1] == "VEVENT" and "VCALENDAR" in stack:
            if name != "DESCRIPTION":  # never read: strangers' text
                props.setdefault(name, []).append((params, value))

    masters = [e for e in events if e.recurrence_id is None]
    by_uid = {e.uid: e for e in masters if e.uid}
    overridden: dict[str, _Marks] = {}
    moved: list[_Event] = []
    for e in events:
        if e.recurrence_id is None:
            continue
        overridden.setdefault(e.uid, _Marks()).add(e.recurrence_id)
        if e.cancelled:
            continue
        master = by_uid.get(e.uid)
        if master is not None:  # an override that only moved it keeps the title
            e.summary = e.summary or master.summary
            e.location = e.location or master.location
        e.rrules = []  # an override is one instance, never a series of its own
        e.has_rdate = False
        moved.append(e)
    return Calendar(masters=masters, overridden=overridden, moved=moved)


# --- recurrence rules ---------------------------------------------------------------

_DAYS = {"MO": 0, "TU": 1, "WE": 2, "TH": 3, "FR": 4, "SA": 5, "SU": 6}
_BYDAY = re.compile(r"^([+-]?\d{1,2})?(MO|TU|WE|TH|FR|SA|SU)$")
_SUPPORTED = {"FREQ", "INTERVAL", "COUNT", "UNTIL", "BYDAY", "BYMONTHDAY", "BYMONTH", "WKST"}


class _Unreadable(Exception):
    """A rule this reader will not guess at."""


class _Done(Exception):
    """A series has run out: past its COUNT or its UNTIL."""


@dataclass(frozen=True)
class _Rule:
    freq: str
    interval: int
    count: Optional[int]
    until: Optional[_When]
    byday: tuple[tuple[int, int], ...]  # (ordinal or 0, weekday)
    bymonthday: tuple[int, ...]
    bymonth: tuple[int, ...]
    wkst: int


def _parts(text: str) -> dict[str, str]:
    out = {}
    for piece in text.strip().split(";"):
        if piece:
            key, _, val = piece.partition("=")
            out[key.strip().upper()] = val.strip().upper()
    return out


def _until(parts: Mapping[str, str]) -> Optional[_When]:
    """UNTIL, if present and readable — needed even for rules that aren't, to
    know whether an unreadable series has already ended."""
    if "UNTIL" not in parts:
        return None
    return _when_one(parts["UNTIL"], {})


def _ints(text: str, lo: int, hi: int, *, signed: bool = False) -> tuple[int, ...]:
    out = []
    for piece in text.split(","):
        n = int(piece)
        if not (lo <= abs(n) <= hi) or (n < 0 and not signed):
            raise _Unreadable(f"out of range: {n}")
        out.append(n)
    return tuple(out)


def _rule(text: str) -> _Rule:
    """Read one RRULE, or raise :class:`_Unreadable` for anything outside the
    subset — an unknown part is a rule we would otherwise expand wrongly."""
    parts = _parts(text)
    try:
        if set(parts) - _SUPPORTED:
            raise _Unreadable("unsupported part")
        freq = parts.get("FREQ", "")
        if freq not in ("DAILY", "WEEKLY", "MONTHLY", "YEARLY"):
            raise _Unreadable("unsupported frequency")
        interval = int(parts.get("INTERVAL", "1"))
        count = int(parts["COUNT"]) if "COUNT" in parts else None
        if interval < 1 or (count is not None and count < 1):
            raise _Unreadable("bad interval or count")
        byday: list[tuple[int, int]] = []
        for piece in filter(None, parts.get("BYDAY", "").split(",")):
            m = _BYDAY.match(piece)
            if not m:
                raise _Unreadable("bad BYDAY")
            ordinal = int(m.group(1) or 0)
            if ordinal and not (1 <= abs(ordinal) <= 5):
                raise _Unreadable("bad BYDAY ordinal")
            byday.append((ordinal, _DAYS[m.group(2)]))
        bymonthday = _ints(parts["BYMONTHDAY"], 1, 31, signed=True) if "BYMONTHDAY" in parts else ()
        bymonth = _ints(parts["BYMONTH"], 1, 12) if "BYMONTH" in parts else ()
        wkst = _DAYS.get(parts.get("WKST", "MO"))
        if wkst is None:
            raise _Unreadable("bad WKST")
        until = _until(parts)
    except ValueError as exc:
        raise _Unreadable(str(exc)) from None
    ordinals = any(o for o, _ in byday)
    # Combinations whose meaning is year-relative or contradictory: not guessed.
    if freq in ("DAILY", "WEEKLY") and ordinals:
        raise _Unreadable("ordinal BYDAY outside a month")
    if freq == "WEEKLY" and bymonthday:
        raise _Unreadable("BYMONTHDAY in a weekly rule")
    if freq == "YEARLY" and byday and not bymonth:
        raise _Unreadable("BYDAY across a whole year")
    return _Rule(freq, interval, count, until, tuple(byday), bymonthday, bymonth, wkst)


def _last_day(year: int, month: int) -> int:
    nxt = date(year + month // 12, month % 12 + 1, 1)
    return (nxt - timedelta(days=1)).day


def _month_days(rule: _Rule, year: int, month: int, start: datetime) -> list[date]:
    """The days in one month a MONTHLY rule (or a YEARLY one, per BYMONTH) hits.

    A day the month doesn't have — the 31st in June, a 5th Tuesday — is simply
    not there: RFC 5545 skips it, and so does this.
    """
    last = _last_day(year, month)

    def matches_byday(d: date) -> bool:
        for ordinal, wd in rule.byday:
            if d.weekday() != wd:
                continue
            if ordinal == 0:
                return True
            nth = (d.day - 1) // 7 + 1 if ordinal > 0 else -((last - d.day) // 7 + 1)
            if nth == ordinal:
                return True
        return False

    if rule.bymonthday:
        days = {n if n > 0 else last + 1 + n for n in rule.bymonthday}
        hits = [date(year, month, n) for n in sorted(days) if 1 <= n <= last]
        return [d for d in hits if matches_byday(d)] if rule.byday else hits
    if rule.byday:
        return [d for n in range(1, last + 1) if matches_byday(d := date(year, month, n))]
    return [date(year, month, start.day)] if start.day <= last else []


def _periods(rule: _Rule, start: datetime, skip: Optional[date]) -> Iterator[list[date]]:
    """Candidate days, one period (day / week / month / year) at a time, in order.

    ``skip`` jumps whole periods forward to near the window, which is what
    keeps a daily-since-1990 event cheap. Only used when there is no COUNT,
    because a count has to be counted from the first instance.
    """
    first, n = start.date(), rule.interval
    if rule.freq == "DAILY":
        k = max(0, (skip - first).days // n) if skip else 0
        while True:
            yield [first + timedelta(days=k * n)]
            k += 1
    elif rule.freq == "WEEKLY":
        week0 = first - timedelta(days=(first.weekday() - rule.wkst) % 7)
        offsets = sorted({(wd - rule.wkst) % 7 for _, wd in rule.byday}) or [
            (first.weekday() - rule.wkst) % 7
        ]
        k = max(0, (skip - week0).days // 7 // n) if skip else 0
        while True:
            base = week0 + timedelta(weeks=k * n)
            yield [base + timedelta(days=o) for o in offsets]
            k += 1
    elif rule.freq == "MONTHLY":
        m0 = first.year * 12 + first.month - 1
        k = max(0, ((skip.year * 12 + skip.month - 1) - m0) // n) if skip else 0
        while True:
            y, m = divmod(m0 + k * n, 12)
            yield _month_days(rule, y, m + 1, start)
            k += 1
    else:  # YEARLY
        k = max(0, (skip.year - first.year) // n) if skip else 0
        months = rule.bymonth or (
            tuple(range(1, 13)) if rule.bymonthday else (first.month,)
        )
        while True:
            y = first.year + k * n
            yield sorted(d for m in months for d in _month_days(rule, y, m, start))
            k += 1


def _keep(rule: _Rule, d: date) -> bool:
    """BYxxx parts that *filter* rather than expand at this frequency."""
    if rule.bymonth and rule.freq != "YEARLY" and d.month not in rule.bymonth:
        return False
    if rule.freq == "DAILY":
        if rule.byday and d.weekday() not in {wd for _, wd in rule.byday}:
            return False
        if rule.bymonthday:
            last = _last_day(d.year, d.month)
            if d.day not in {n if n > 0 else last + 1 + n for n in rule.bymonthday}:
                return False
    return True


def _past_until(rule: _Rule, w: _When) -> bool:
    if rule.until is None:
        return False
    if rule.until.all_day:
        return w.wall.date() > rule.until.wall.date()
    return w.aware > rule.until.aware


def _starts(ev: _Event, rule: _Rule, win_start: datetime, win_end: datetime) -> Iterator[_When]:
    """Instance starts of a series that could overlap the window, in order.

    Expanded on the event's own wall clock — a 09:00 New York standup is at
    09:00 New York on both sides of a DST change — and only then made aware.
    Raises :class:`_Unreadable` if the step cap runs out first.
    """
    s = ev.start
    skip: Optional[date] = None
    if rule.count is None:
        near = win_start.astimezone(s.tz).replace(tzinfo=None) - ev.length
        skip = near.date() - timedelta(days=1)
        if skip <= s.wall.date():
            skip = None
    seen = 0

    def candidate(wall: datetime) -> Optional[_When]:
        nonlocal seen
        w = _When(wall, s.tz, s.all_day, s.approximate)
        seen += 1
        if (rule.count is not None and seen > rule.count) or _past_until(rule, w):
            raise _Done
        return w

    try:
        if skip is None:  # DTSTART is always the first instance (RFC 5545 §3.8.5.3)
            yield candidate(s.wall)
        for steps, period in enumerate(_periods(rule, s.wall, skip)):
            if steps >= MAX_STEPS:
                raise _Unreadable("too many steps")
            for d in period:
                wall = datetime.combine(d, s.wall.time())
                if wall <= s.wall or not _keep(rule, d):
                    continue
                w = candidate(wall)
                if w.aware >= win_end:
                    return
                yield w
    except _Done:
        return
    except (OverflowError, ValueError):  # walked off the end of the calendar
        return


def _occurrence(ev: _Event, w: _When) -> Occurrence:
    return Occurrence(
        start=w.aware,
        end=(w.wall + ev.length).replace(tzinfo=w.tz),
        all_day=ev.start.all_day,
        summary=ev.summary or "(no title)",
        location=ev.location,
        approximate=ev.start.approximate,
    )


def _overlaps(o: Occurrence, start: datetime, end: datetime) -> bool:
    if o.start >= end:
        return False
    if o.end == o.start:  # a zero-length event is a point
        return o.start >= start
    return o.end > start


def _could_overlap(ev: _Event, rrules: list[str], win_start: datetime, win_end: datetime) -> bool:
    """Whether an unreadable series might have an instance in the window — it
    starts before the window ends and is not ended by an UNTIL we *can* read."""
    if ev.start.aware >= win_end:
        return False
    for text in rrules:
        try:
            until = _until(_parts(text))
        except ValueError:
            return True
        if until is None:
            return True
        last = until.aware if not until.all_day else until.aware + timedelta(days=1)
        if last + ev.length >= win_start:
            return True
    return not rrules  # RDATE alone: no end we can read


def occurrences(cal: Calendar, start: datetime, end: datetime) -> Window:
    """Every occurrence overlapping ``[start, end)``, sorted by start then title."""
    found: list[Occurrence] = []
    unreadable: list[str] = []
    for ev in cal.masters:
        if ev.cancelled:
            continue
        if not ev.rrules and not ev.has_rdate:
            o = _occurrence(ev, ev.start)
            if _overlaps(o, start, end):
                found.append(o)
            continue
        try:
            if ev.has_rdate or len(ev.rrules) != 1:
                raise _Unreadable("RDATE or several RRULEs")
            rule = _rule(ev.rrules[0])
            marks = cal.overridden.get(ev.uid)
            instances = []
            for w in _starts(ev, rule, start, end):
                if ev.exdates.has(w) or (marks is not None and marks.has(w)):
                    continue
                o = _occurrence(ev, w)
                if _overlaps(o, start, end):
                    instances.append(o)
            found.extend(instances)
        except _Unreadable:
            if _could_overlap(ev, ev.rrules, start, end):
                unreadable.append(ev.summary or "(no title)")
    for ev in cal.moved:
        o = _occurrence(ev, ev.start)
        if _overlaps(o, start, end):
            found.append(o)
    found.sort(key=lambda o: (o.start, o.summary))
    return Window(tuple(found), tuple(sorted(set(unreadable))), ())


# --- fetching ---------------------------------------------------------------------------


def feeds(environ: Mapping[str, str] = os.environ) -> list[str]:
    """The configured feed URLs, in order, without duplicates."""
    out: list[str] = []
    for url in re.split(r"[\s,]+", environ.get(ENV, "")):
        if url and url not in out:
            out.append(url)
    return out


class _HttpsOnlyRedirects(urllib.request.HTTPRedirectHandler):
    """Follow a redirect only to another https address: a downgrade would send
    the secret path in the clear."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        if urllib.parse.urlsplit(newurl).scheme.lower() != "https":
            raise CalendarUnavailable("redirected away from https")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _scrub(text: str, url: str) -> str:
    """Belt and braces: remove the URL (and its host) from a reason."""
    host = urllib.parse.urlsplit(url).netloc
    for secret in (url, host):
        if secret:
            text = text.replace(secret, "…")
    return text


def get(url: str, *, timeout: float = 10.0, max_bytes: int = 5_000_000) -> str:
    """Fetch one feed's text. https only, size-capped.

    Every failure becomes :class:`CalendarUnavailable` with a reason built
    here — ``from None`` so not even the chained traceback carries the URL.
    """
    try:
        scheme = urllib.parse.urlsplit(url).scheme.lower()
    except ValueError:
        scheme = ""
    if scheme != "https":
        raise CalendarUnavailable("only https calendar addresses are read")
    reason: str
    try:
        opener = urllib.request.build_opener(_HttpsOnlyRedirects)
        request = urllib.request.Request(url, headers={"User-Agent": "omega-calendar"})
        with opener.open(request, timeout=timeout) as resp:
            body = resp.read(max_bytes + 1)
    except CalendarUnavailable as exc:
        reason = str(exc)
    except urllib.error.HTTPError as exc:
        reason = f"HTTP {exc.code}"
    except urllib.error.URLError as exc:
        why = exc.reason
        if isinstance(why, TimeoutError):
            reason = "timed out"
        elif isinstance(why, BaseException):
            reason = f"couldn't connect ({type(why).__name__})"
        else:
            reason = f"couldn't connect ({_scrub(str(why), url)[:80]})"
    except TimeoutError:
        reason = "timed out"
    except (OSError, http.client.HTTPException, ValueError) as exc:
        reason = f"connection failed ({type(exc).__name__})"
    else:
        if len(body) > max_bytes:
            raise CalendarUnavailable(f"calendar is larger than {max_bytes // 1_000_000} MB")
        return body.decode("utf-8", errors="replace")
    raise CalendarUnavailable(_scrub(reason, url)) from None


_CACHE: dict[str, tuple[float, str]] = {}


def _fetch(url: str, get: Callable[[str], str]) -> str:
    """Fetched text, reused for :data:`CACHE_SECONDS`. Monotonic, so a clock
    change can't make a stale copy look fresh. Failures are never cached."""
    hit = _CACHE.get(url)
    now = time.monotonic()
    if hit is not None and now - hit[0] < CACHE_SECONDS:
        return hit[1]
    text = get(url)
    _CACHE[url] = (now, text)
    return text


def read_window(
    start: datetime,
    end: datetime,
    *,
    environ: Mapping[str, str] = os.environ,
    get: Callable[[str], str] = get,
) -> Window:
    """Read every feed and merge what overlaps ``[start, end)``.

    A feed that fails is recorded in ``errors`` by position; if *every* feed
    fails there is nothing to show, so that raises.
    """
    urls = feeds(environ)
    found: dict[tuple, Occurrence] = {}
    unreadable: set[str] = set()
    errors: list[str] = []
    for i, url in enumerate(urls, 1):
        label = f"feed {i} of {len(urls)}"
        try:
            window = occurrences(parse(_fetch(url, get)), start, end)
        except CalendarUnavailable as exc:
            errors.append(f"{label}: {_scrub(str(exc), url)}")
            continue
        except Exception:  # a parser bug must not take the look down with it
            errors.append(f"{label}: couldn't be parsed")
            continue
        for o in window.occurrences:  # the same invite on two calendars shows once
            found.setdefault((o.start, o.end, o.all_day, o.summary), o)
        unreadable.update(window.unreadable)
    if urls and len(errors) == len(urls):
        if len(urls) == 1:
            raise CalendarUnavailable(errors[0].split(": ", 1)[1])
        raise CalendarUnavailable("; ".join(errors))
    ordered = sorted(found.values(), key=lambda o: (o.start, o.summary))
    return Window(tuple(ordered), tuple(sorted(unreadable)), tuple(errors))


# --- words ---------------------------------------------------------------------------


def _date(d: date) -> str:
    """``Sat 10 Oct`` — no year: a calendar look is about the next few days."""
    return f"{d:%a} {d.day} {d:%b}"


def _day(d: date, today: date) -> str:
    if d == today:
        return "today"
    if d == today + timedelta(days=1):
        return "tomorrow"
    return _date(d)


def _line(o: Occurrence, now: datetime) -> str:
    today = now.date()
    start, end = o.start.astimezone(), o.end.astimezone()
    what = o.summary + (f" ({o.location})" if o.location else "")
    if o.all_day:
        first = max(start.date(), today) if end > now else start.date()
        last = (end - timedelta(seconds=1)).date()
        span = "all day" if last <= first else f"all day until {_day(last, today)}"
        text = f"{_day(first, today)}, {span}: {what}"
    elif start <= now < end:
        until = f"{end:%H:%M}" if end.date() == today else f"{_day(end.date(), today)} {end:%H:%M}"
        text = f"now, until {until}: {what}"
    else:
        when = f"{_day(start.date(), today)} {start:%H:%M}"
        if end > start:
            same_day = end.date() == start.date() or end - start <= timedelta(hours=24)
            when += f"-{end:%H:%M}" if same_day else f"-{_day(end.date(), today)} {end:%H:%M}"
        text = f"{when} {what}"
    return text + (" (time zone not read)" if o.approximate else "")


def lines(window: Window, now: datetime) -> list[str]:
    """Human lines, in the process zone, one per occurrence, then the gaps."""
    local = now.astimezone()
    out = [_line(o, local) for o in window.occurrences]
    out += [f"repeats, next time couldn't be determined: {s}" for s in window.unreadable]
    out += [f"couldn't read {e}" for e in window.errors]
    return out


def _render(window: Window, now: datetime, empty: str) -> list[str]:
    """``lines`` with the "nothing" sentence when — and only when — the read
    really found nothing. An unreadable series is not nothing."""
    if not window.occurrences and not window.unreadable:
        return [empty] + lines(window, now)
    return lines(window, now)


def look_lines(
    now: Optional[datetime] = None,
    *,
    environ: Mapping[str, str] = os.environ,
    get: Callable[[str], str] = get,
    hours: int = LOOK_HOURS,
) -> list[str]:
    """The look's calendar block. ``[]`` means *not configured* and nothing
    else; a failure and an empty calendar each say so in words."""
    if not feeds(environ):
        return []
    now = (now or datetime.now()).astimezone()
    try:
        window = read_window(now, now + timedelta(hours=hours), environ=environ, get=get)
    except CalendarUnavailable as exc:
        return [f"couldn't read the calendar: {exc}"]
    return _render(window, now, f"nothing on the calendar in the next {hours} hours")


def tool_text(
    days: int,
    now: Optional[datetime] = None,
    *,
    environ: Mapping[str, str] = os.environ,
    get: Callable[[str], str] = get,
) -> str:
    """The ``calendar`` tool's answer: local midnight today through the end of
    the day ``days`` from now (0 = today, 1 = today and tomorrow).

    Raises :class:`CalendarUnavailable` when every feed fails, so the tool
    layer reports a failure as a failure.
    """
    if not feeds(environ):
        return NOT_CONFIGURED
    now = (now or datetime.now()).astimezone()
    first = now.date()
    last = first + timedelta(days=max(0, int(days)))
    start = datetime.combine(first, dtime(), _LOCAL)
    end = datetime.combine(last + timedelta(days=1), dtime(), _LOCAL)
    window = read_window(start, end, environ=environ, get=get)
    empty = f"nothing on the calendar from {_date(first)} to {_date(last)}"
    return "\n".join(_render(window, now, empty))
