"""Reading what the Mac already wrote down — DL-059.

The second sense, and the one that is about the person rather than about their
work. :mod:`omega.transcripts` learns how they work from agent transcripts;
this learns *when* they work, and around what, from the usage record macOS has
been keeping on its own since long before omega existed.

**Nothing here observes anything.** That is the whole reason this file reads a
database instead of sampling the frontmost app: ``knowledgeC.db`` is a finished
artifact on disk, so a completed day has exactly the property that makes a
finished session safe to learn from, and the DL-057 machine applies unchanged —
discover finished units, render a mechanical digest, one model call, claims with
``explicit=False``, a receipt per unit that *is* the cursor. The alternative was
for omega to become a recorder: a sampler thread, in-memory state lost on every
restart, and no history at all before the day it shipped. That is building a
worse copy of a record the operating system is already keeping.

**The unit is a finished day**, for the reason a session must be idle: today is
a window that is still growing, and re-digesting it would file the same claim
twice. :func:`discover` only offers days strictly before the current local date.

**What this reads, and what it must not.** Bundle identifiers and durations.
``knowledgeC`` also carries window titles and document names, and those are
somebody else's content in exactly the way a tool result is — a window title is
attacker-controlled the moment a web page sets one. Restricting the query to the
app-usage stream's bundle id and its two timestamps is the same structural
defence the transcript digest uses: not reading the untrusted bytes is cheaper
and stronger than reading them carefully.

**Read from a copy.** The live database is open and journalling under a process
we do not control. Snapshotting it costs tens of milliseconds and removes the
whole class of "the file moved under us", and it means omega never holds a
handle on a file the system is writing to.

**The schema is undocumented and has moved between macOS releases**, so this is
the first part of omega an OS update can break. It fails to a recorded reason
rather than an exception, which is the most a reader of somebody else's private
format can honestly promise.
"""

from __future__ import annotations

import shutil
import sqlite3
import tempfile
from dataclasses import dataclass
from datetime import date as _date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

__all__ = [
    "Day",
    "SOURCE",
    "DEFAULT_PATH",
    "STREAM",
    "APPLE_EPOCH",
    "LOOKBACK_DAYS",
    "MAX_DAYS_PER_PASS",
    "MAX_DIGEST_CHARS",
    "MAX_APPS",
    "MIN_ACTIVE_SECONDS",
    "MIN_APPS",
    "default_path",
    "discover",
    "digest",
]

#: Which record these came from. On the receipt for the same reason the
#: transcript reader puts one there: a second source added later must not make
#: the first one's receipts ambiguous.
SOURCE = "knowledgec"

#: Where macOS keeps it. Under Full Disk Access — without it every read of this
#: path fails with "authorization denied", which :func:`discover` treats as
#: *there is nothing to look at yet* rather than as a fault.
DEFAULT_PATH = Path("Library") / "Application Support" / "Knowledge" / "knowledgeC.db"

#: The one stream this reads. ``ZOBJECT`` carries dozens — notifications, media
#: playback, Safari history, device state — and most of them are either free
#: text or things the person did not consent to being read by a process that
#: talks. App usage is the narrowest stream that answers "when do they work".
STREAM = "/app/usage"

#: Cocoa's epoch (2001-01-01T00:00:00Z) as a Unix timestamp. Every date in this
#: database is seconds since then.
APPLE_EPOCH = 978_307_200

#: How far back a cold start will go. Thirty days is enough for a weekly shape
#: to be visible and short enough that the backlog drains in a quarter of an
#: hour at the cap below. Older days are not lost so much as never offered:
#: habits from two months ago are a worse guess at today than last week's.
LOOKBACK_DAYS = 30

#: Days one pass will digest. The cost bound, exactly as for sessions: each day
#: is one model call. Undigested days do not expire, so a low cap delays
#: learning rather than losing it.
MAX_DAYS_PER_PASS = 2

#: Ceiling on a rendered digest. Small, because a day renders to a table rather
#: than to prose and a day that needs six thousand characters is a day whose
#: aggregation has gone wrong.
MAX_DIGEST_CHARS = 6_000

#: How many apps a digest names. The tail of this distribution is background
#: agents and one-second activations, which are noise about the machine rather
#: than evidence about the person.
MAX_APPS = 20

#: Below this much recorded activity a day is not evidence of anything. A day
#: with nine minutes of app use is a day the laptop was shut, and asking a model
#: what it shows about someone's habits is asking it to invent — which is the
#: firehose DL-054 was written against, arriving through a new door.
MIN_ACTIVE_SECONDS = 45 * 60

#: And below this many distinct apps. Guards the other shape of an empty day:
#: four hours of one screensaver is a long time and no information.
MIN_APPS = 3


def default_path() -> Path:
    """Where the usage record lives for this user."""
    return Path.home() / DEFAULT_PATH


@dataclass(frozen=True)
class Day:
    """One finished day, and where to read it from."""

    date: _date
    path: Path

    @property
    def id(self) -> str:
        """What the receipt records. A local calendar date, because that is the
        unit a person means by "yesterday" and the unit the day boundaries were
        computed in."""
        return self.date.isoformat()

    @property
    def source(self) -> str:
        return SOURCE


def _readable(path: Path) -> bool:
    """Whether the database can be opened at all.

    A plain read of the first bytes rather than a SQLite connection, because the
    thing being tested is the *permission* and SQLite would additionally want to
    build a shared-memory file beside a database it is opening read-only.

    Checked here, up front, rather than per day, and this is not a tidiness
    choice. Without Full Disk Access every day would digest to a failure, each
    failure would write a receipt, and every receipt is permanent — so the whole
    lookback would be marked read, with nothing read, and granting the
    permission afterwards would recover none of it. A failure to *look* must not
    be recorded as having looked.
    """
    try:
        with path.open("rb") as fh:
            return fh.read(16).startswith(b"SQLite format 3")
    except OSError:
        return False


def discover(
    path: Optional[Path] = None,
    *,
    now: Optional[datetime] = None,
    lookback_days: int = LOOKBACK_DAYS,
    seen: Iterable[str] = (),
) -> list[Day]:
    """Finished days not yet digested, oldest first.

    Oldest first for :func:`omega.transcripts.discover`'s reason: learning in
    the order the days happened is the only order in which a later day can
    supersede an earlier claim rather than contradict it.

    Only days strictly before today, because today is still being written.

    ``seen`` is the set of day ids that already have a receipt. Passed in rather
    than read here because the answer lives in omega's log and this module
    deliberately knows nothing about the log (DL-036).

    Returns ``[]`` — not an error — when the database cannot be read. That is
    the ordinary state on a machine where the permission has not been granted,
    and on one where the file does not exist at all.
    """
    path = path or default_path()
    if not _readable(path):
        return []
    moment = now or datetime.now().astimezone()
    if moment.tzinfo is not None:
        moment = moment.astimezone()
    today = moment.date()
    already = set(seen)
    out: list[Day] = []
    for back in range(lookback_days, 0, -1):
        day = today - timedelta(days=back)
        if day.isoformat() in already:
            continue
        out.append(Day(date=day, path=path))
    return out


def _bounds(day: _date) -> tuple[float, float]:
    """The day's half-open range, in Cocoa seconds.

    Local midnight to local midnight: a naive ``datetime``'s ``timestamp()``
    reads it in the machine's zone, which is the definition of "that day" a
    person means and the one a daylight-saving boundary keeps correct — the day
    that gains or loses an hour really is that long.
    """
    start = datetime(day.year, day.month, day.day).timestamp()
    tomorrow = day + timedelta(days=1)
    end = datetime(tomorrow.year, tomorrow.month, tomorrow.day).timestamp()
    return start - APPLE_EPOCH, end - APPLE_EPOCH


def _rows(day: Day) -> list[tuple[str, float, float]]:
    """App usage intervals for one day: ``(bundle id, start, end)`` in Unix
    seconds, read from a snapshot of the database.

    The query names its three columns explicitly and filters to one stream.
    ``SELECT *`` here would pull window titles and document names into memory
    on their way to a model, which is the thing the module docstring says this
    does not do — so the column list is a security boundary, not a style.
    """
    lo, hi = _bounds(day.date)
    with tempfile.TemporaryDirectory(prefix="omega-usage-") as tmp:
        copy = Path(tmp) / "snapshot.db"
        shutil.copyfile(day.path, copy)
        # The sidecars, when they exist: a WAL database whose log is left behind
        # reads as of the last checkpoint, which can be hours stale. Missing
        # ones are the normal case for a database that was closed cleanly.
        for suffix in ("-wal", "-shm"):
            sidecar = day.path.with_name(day.path.name + suffix)
            try:
                shutil.copyfile(sidecar, copy.with_name(copy.name + suffix))
            except OSError:
                pass
        conn = sqlite3.connect(copy)
        try:
            cursor = conn.execute(
                "SELECT ZVALUESTRING, ZSTARTDATE, ZENDDATE FROM ZOBJECT "
                "WHERE ZSTREAMNAME = ? AND ZSTARTDATE >= ? AND ZSTARTDATE < ? "
                "ORDER BY ZSTARTDATE",
                (STREAM, lo, hi),
            )
            out = []
            for bundle, start, end in cursor:
                if not isinstance(bundle, str) or not bundle.strip():
                    continue
                if not isinstance(start, (int, float)):
                    continue
                if not isinstance(end, (int, float)) or end < start:
                    continue
                out.append(
                    (
                        bundle.strip(),
                        float(start) + APPLE_EPOCH,
                        float(end) + APPLE_EPOCH,
                    )
                )
            return out
        finally:
            conn.close()


def _union_seconds(intervals: list[tuple[float, float]]) -> float:
    """Wall-clock time covered by a set of possibly-overlapping intervals.

    Summing them instead would report twenty-six hours in a day, because macOS
    records an app as in use while another is in focus. An inflated total is not
    a rounding error here — it is the number the model is asked to reason about,
    and "they worked 26 hours" is a false premise that will produce a confident
    false claim.
    """
    if not intervals:
        return 0.0
    ordered = sorted(intervals)
    total = 0.0
    start, end = ordered[0]
    for lo, hi in ordered[1:]:
        if lo > end:
            total += end - start
            start, end = lo, hi
        elif hi > end:
            end = hi
    return total + (end - start)


def _clock(seconds: float) -> str:
    minutes = int(seconds // 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m"


def digest(day: Day, *, max_chars: int = MAX_DIGEST_CHARS) -> Optional[str]:
    """Render one day as what it says about when and how the person worked, or
    ``None`` if it says nothing.

    ``None`` rather than an empty digest is *fail closed on empty*, and it is
    load-bearing twice over: a quiet day must not reach a model that was asked
    to find a pattern in it, and a day the machine was off must not cost a model
    call to conclude nothing.
    """
    rows = _rows(day)
    if not rows:
        return None

    per_app: dict[str, list[tuple[float, float]]] = {}
    for bundle, start, end in rows:
        per_app.setdefault(bundle, []).append((start, end))

    totals = {
        bundle: _union_seconds(spans) for bundle, spans in per_app.items()
    }
    active = _union_seconds([(s, e) for _, s, e in rows])
    if active < MIN_ACTIVE_SECONDS or len(totals) < MIN_APPS:
        return None

    first = datetime.fromtimestamp(min(s for _, s, _ in rows))
    last = datetime.fromtimestamp(max(e for _, _, e in rows))

    # Which app held the most time in each hour. The shape of a day is in its
    # sequence, not only its totals — "editor all morning, browser all evening"
    # and the reverse have identical totals and mean different things.
    by_hour: dict[int, dict[str, float]] = {}
    for bundle, start, end in rows:
        cursor = start
        while cursor < end:
            moment = datetime.fromtimestamp(cursor)
            edge = moment.replace(minute=0, second=0, microsecond=0) + timedelta(
                hours=1
            )
            stop = min(end, edge.timestamp())
            hour = by_hour.setdefault(moment.hour, {})
            hour[bundle] = hour.get(bundle, 0.0) + (stop - cursor)
            cursor = stop

    ranked = sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))[:MAX_APPS]
    lines = [
        f"Date: {day.date.isoformat()} ({day.date.strftime('%A')})",
        f"Awake at the machine from {first:%H:%M} to {last:%H:%M}.",
        f"Recorded app use: {_clock(active)} across {len(totals)} apps.",
        "",
        "Where the time went:",
    ]
    for bundle, seconds in ranked:
        lines.append(f"  - {bundle} — {_clock(seconds)}")
    if len(totals) > MAX_APPS:
        lines.append(f"  … {len(totals) - MAX_APPS} more apps, briefly …")
    lines += ["", "Busiest app by hour:"]
    for hour in sorted(by_hour):
        bundle, seconds = max(
            by_hour[hour].items(), key=lambda kv: (kv[1], kv[0])
        )
        lines.append(f"  {hour:02d}:00  {bundle} ({_clock(seconds)})")

    rendered = "\n".join(lines)
    if len(rendered) > max_chars:
        rendered = rendered[:max_chars].rstrip() + "\n  … digest truncated …"
    return rendered
