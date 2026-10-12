"""The leave-now alarm (DL-093).

They say "I need to be at the airport by 18:00"; the ``leave_by`` tool writes
one ``leave.set``. From then on this is code and nothing else: no model call
per check. :meth:`omega.executor.Executor.leave_now`, asked on every sense
pass, folds the alarms out of the log and does what :func:`due` says is due:

- **check**: one Maps ``compute_routes`` call (driving) through the person's
  own MCP server, written down as ``leave.checked``. The first is at
  ``arrive_by - (ROUGH_MINUTES + LEAD)``, or at once when that has passed.
  After a check, the next one is at the start of the final
  :data:`WINDOW` before the leave time, and every :data:`EVERY` inside it,
  never later than the leave time itself; after the ping, every
  :data:`EVERY` until ``arrive_by``. Never more than :data:`MAX_CALLS` per
  alarm, because each one is billed.
- **ping**: when a check finds the leave time (``arrive_by - eta -
  BUFFER``) reached, one tell: "Leave now for <dest>: <eta> min by car with
  traffic, arrive ~<t>." and a directions link. Once more if a later check
  finds the drive :data:`WORSE_MINUTES` or more longer than the ping said.
- **estimate**: Maps not configured, not reachable, refusing its key, or
  answering nothing readable: one tell saying traffic could not be read and
  when to leave by the last estimate, labelled as such. Never a silent no-ping.
- **done**: ``arrive_by`` passed.

All state is the fold, so an alarm survives a restart. A refused key is
recognised by :func:`omega.access.classify` from ``leave.checked`` the way a
refused ``tool.returned`` is, so the lost-access tell, the hourly probe and
the restored tell all apply to ``mcp:maps``; while it stays lost no call is
spent.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Optional
from urllib.parse import quote

from omega import episodes

if TYPE_CHECKING:  # pragma: no cover
    from omega.memory import MemoryStore

__all__ = [
    "SERVER",
    "TOOL",
    "FULL",
    "BUFFER",
    "ROUGH_MINUTES",
    "LEAD",
    "WINDOW",
    "EVERY",
    "WORSE_MINUTES",
    "MAX_CALLS",
    "Check",
    "Ping",
    "Alarm",
    "State",
    "Leaves",
    "fold",
    "next_check",
    "due",
    "after_check",
    "route_args",
    "eta_minutes",
    "directions_url",
    "ping_text",
    "estimate_text",
    "listing",
]

#: The Maps MCP server and its routes tool, as ``OMEGA_MCP_SERVERS`` names it.
SERVER = "maps"
TOOL = "compute_routes"
FULL = f"{SERVER}__{TOOL}"  # omega.mcp.SEP

#: Slack between arriving at the door and arriving on time.
BUFFER = timedelta(minutes=10)
#: The drive assumed before Maps has said anything. A labelled guess: it only
#: decides when the first check runs, and that check replaces it.
ROUGH_MINUTES = 60
#: How long before the rough leave time the first check runs.
LEAD = timedelta(minutes=60)
#: The final stretch before the leave time that is checked every EVERY.
WINDOW = timedelta(minutes=45)
EVERY = timedelta(minutes=10)
#: A drive this much longer than the ping said is worth a second ping.
WORSE_MINUTES = 10
#: Maps calls per alarm, at most: each one is billed.
MAX_CALLS = 12


def _stamp(raw: Any) -> datetime:
    moment = datetime.fromisoformat(str(raw))
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


# --- the view -----------------------------------------------------------------


@dataclass(frozen=True)
class Check:
    at: datetime
    eta_min: Optional[int]
    error: Optional[str] = None


@dataclass(frozen=True)
class Ping:
    at: datetime
    how: str
    eta_min: int


@dataclass(frozen=True)
class Alarm:
    """One ``leave.set`` and everything since."""

    id: str
    seq: int
    destination: str
    arrive_by: datetime
    origin: str
    origin_via: str
    set_at: datetime
    checks: tuple[Check, ...] = ()
    pings: tuple[Ping, ...] = ()
    cancelled: bool = False
    done: bool = False

    @property
    def open(self) -> bool:
        return not (self.cancelled or self.done)

    @property
    def calls(self) -> int:
        return len(self.checks)

    @property
    def last_eta(self) -> Optional[Check]:
        """The newest check that read a drive time, or ``None``."""
        return next((c for c in reversed(self.checks) if c.eta_min is not None), None)

    @property
    def estimate(self) -> int:
        """The drive as best known: the last one read, else the rough guess."""
        last = self.last_eta
        return last.eta_min if last is not None else ROUGH_MINUTES  # type: ignore[return-value]

    def ping(self, how: str) -> Optional[Ping]:
        return next((p for p in self.pings if p.how == how), None)

    def leave_at(self, eta_min: int) -> datetime:
        return self.arrive_by - timedelta(minutes=eta_min) - BUFFER


@dataclass(frozen=True)
class State:
    through: int = 0
    alarms: Mapping[str, Alarm] = field(default_factory=dict)

    def open(self) -> list[Alarm]:
        """The alarms still standing, soonest arrive-by first."""
        return sorted((a for a in self.alarms.values() if a.open),
                      key=lambda a: (a.arrive_by, a.seq))

    def __bool__(self) -> bool:
        return bool(self.alarms)


class Leaves:
    """The fold. ``apply`` in seq order; :meth:`state` snapshots it."""

    __slots__ = ("_through", "_alarms")

    def __init__(self) -> None:
        self._through = 0
        self._alarms: dict[str, Alarm] = {}

    @property
    def through(self) -> int:
        return self._through

    def apply(self, seq: int, payload: dict[str, Any]) -> None:
        if seq <= self._through:
            raise ValueError(
                f"episodes must be folded in order: got seq {seq} after {self._through}"
            )
        self._through = seq
        kind = payload.get("kind")
        if kind not in episodes.LEAVE_KINDS:
            return
        aid = str(payload["id"])
        at = _stamp(payload["at"])
        if kind == episodes.LEAVE_SET:
            self._alarms[aid] = Alarm(
                id=aid, seq=seq, destination=payload["destination"],
                arrive_by=_stamp(payload["arrive_by"]), origin=payload["origin"],
                origin_via=payload["origin_via"], set_at=at,
            )
            return
        alarm = self._alarms.get(aid)
        if alarm is None:
            return
        if kind == episodes.LEAVE_CHECKED:
            alarm = replace(alarm, checks=alarm.checks + (
                Check(at=at, eta_min=payload["eta_min"], error=payload.get("error")),))
        elif kind == episodes.LEAVE_PINGED:
            alarm = replace(alarm, pings=alarm.pings + (
                Ping(at=at, how=payload["how"], eta_min=payload["eta_min"]),))
        elif kind == episodes.LEAVE_CANCELLED:
            alarm = replace(alarm, cancelled=True)
        else:  # LEAVE_DONE
            alarm = replace(alarm, done=True)
        self._alarms[aid] = alarm

    def state(self) -> State:
        return State(through=self._through, alarms=dict(self._alarms))

    def advance(self, store: "MemoryStore", *, upto: Optional[int] = None) -> "Leaves":
        for episode in store.episodes_since(self._through):
            if upto is not None and episode.seq > upto:
                break
            self.apply(episode.seq, episodes.decode(episode.payload))
        return self

    @classmethod
    def fold(cls, decoded: Iterable[tuple[int, dict[str, Any]]]) -> "Leaves":
        view = cls()
        for seq, payload in decoded:
            view.apply(seq, payload)
        return view


def fold(store: "MemoryStore") -> State:
    """The whole log, folded."""
    return Leaves().advance(store).state()


# --- the timing: pure, so the tests pin every minute of it -----------------------


def next_check(alarm: Alarm) -> Optional[datetime]:
    """When the next Maps call is due, or ``None`` when no more will be made.

    ``None`` once the call cap is spent, once the second ping has gone, and
    for any moment at or past ``arrive_by``.
    """
    if not alarm.open or alarm.calls >= MAX_CALLS or alarm.ping("again") is not None:
        return None
    if not alarm.checks:
        when = alarm.arrive_by - timedelta(minutes=ROUGH_MINUTES) - LEAD
    else:
        last = alarm.checks[-1].at
        if alarm.ping("live") is not None:
            when = last + EVERY
        else:
            leave = alarm.leave_at(alarm.estimate)
            if leave - last > WINDOW:
                when = leave - WINDOW
            else:
                when = min(last + EVERY, max(leave, last))
                if when <= last:
                    when = last + EVERY
    return when if when < alarm.arrive_by else None


def due(alarm: Alarm, now: datetime) -> Optional[str]:
    """What is due for ``alarm`` at ``now``: ``"done"``, ``"check"``,
    ``"ping"`` (the leave time reached with no call left to make, so the last
    drive time read stands) or ``None``."""
    if not alarm.open:
        return None
    if now >= alarm.arrive_by:
        return "done"
    when = next_check(alarm)
    if when is not None and now >= when:
        return "check"
    if when is None and alarm.ping("live") is None and alarm.last_eta is not None \
            and now >= alarm.leave_at(alarm.estimate):
        return "ping"
    return None


def after_check(alarm: Alarm, now: datetime) -> Optional[str]:
    """What the check just folded into ``alarm`` calls for: ``"ping"``,
    ``"again"``, ``"estimate"`` (traffic could not be read and no ping of any
    kind has gone yet) or ``None``."""
    last = alarm.checks[-1] if alarm.checks else None
    if last is None:
        return None
    if last.eta_min is None:
        return "estimate" if not alarm.pings else None
    first = alarm.ping("live")
    if first is None:
        return "ping" if now >= alarm.leave_at(last.eta_min) else None
    if alarm.ping("again") is None and last.eta_min - first.eta_min >= WORSE_MINUTES:
        return "again"
    return None


# --- Maps: the call and its answer --------------------------------------------------


_COORDS = re.compile(r"^\s*(-?\d{1,3}(?:\.\d+)?)\s*,\s*(-?\d{1,3}(?:\.\d+)?)\s*$")


def _waypoint(place: str) -> dict[str, Any]:
    found = _COORDS.match(place)
    if found:
        lat, lon = float(found.group(1)), float(found.group(2))
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            return {"latLng": {"latitude": lat, "longitude": lon}}
    return {"address": place}


def route_args(alarm: Alarm) -> dict[str, Any]:
    """``compute_routes``'s arguments: driving, origin to destination."""
    return {"origin": _waypoint(alarm.origin), "destination": _waypoint(alarm.destination),
            "travelMode": "DRIVE"}


_SECONDS = re.compile(r'"duration"\s*:\s*(?:"(\d+(?:\.\d+)?)s"|\{\s*"seconds"\s*:\s*"?(\d+))')


def eta_minutes(text: str) -> Optional[int]:
    """The first route's drive in whole minutes (rounded up) from a
    ``compute_routes`` answer, or ``None`` when it carries none.

    Reads ``routes[0].duration`` (the traffic-aware one; ``staticDuration``
    is the drive without traffic and is never read) and nothing the server
    says around it.
    """
    brace = text.find("{")
    if brace >= 0:
        try:
            body, _ = json.JSONDecoder().raw_decode(text[brace:])
        except ValueError:
            body = None
        routes = body.get("routes") if isinstance(body, dict) else None
        if isinstance(routes, list):
            if not routes or not isinstance(routes[0], dict):
                return None
            raw = routes[0].get("duration")
            if isinstance(raw, str) and raw.endswith("s"):
                try:
                    return _minutes(float(raw[:-1]))
                except ValueError:
                    return None
            if isinstance(raw, dict) and "seconds" in raw:
                try:
                    return _minutes(float(raw["seconds"]))
                except (TypeError, ValueError):
                    return None
            return None
    found = _SECONDS.search(text)
    if found is None:
        return None
    return _minutes(float(found.group(1) or found.group(2)))


def _minutes(seconds: float) -> Optional[int]:
    if not math.isfinite(seconds) or seconds < 0:
        return None
    minutes = math.ceil(seconds / 60)
    return minutes if minutes <= episodes.MAX_LEAVE_ETA else None


# --- words ----------------------------------------------------------------------------


def _hhmm(moment: datetime) -> str:
    return f"{moment.astimezone():%H:%M}"


def directions_url(alarm: Alarm) -> str:
    """A Google Maps directions link, driving, origin to destination."""
    return (
        "https://www.google.com/maps/dir/?api=1"
        f"&origin={quote(alarm.origin, safe=',')}"
        f"&destination={quote(alarm.destination, safe=',')}"
        "&travelmode=driving"
    )


def ping_text(alarm: Alarm, eta_min: int, now: datetime, *, again: bool = False) -> str:
    """The ping: go now, how long, when they get there, and the link."""
    head = "Traffic got worse. " if again else ""
    arrive = now + timedelta(minutes=eta_min)
    return (
        f"{head}Leave now for {alarm.destination}: {eta_min} min by car with traffic, "
        f"arrive ~{_hhmm(arrive)}.\n{directions_url(alarm)}"
    )


def estimate_text(alarm: Alarm) -> str:
    """The one message when traffic cannot be read, labelled an estimate."""
    last = alarm.last_eta
    minutes = alarm.estimate
    basis = (
        f"the {minutes}-minute drive Maps gave at {_hhmm(last.at)}"
        if last is not None else f"a rough {minutes}-minute guess"
    )
    return (
        f"Can't read traffic for {alarm.destination}; leave by "
        f"{_hhmm(alarm.leave_at(minutes))}. That is an estimate without live "
        f"traffic, from {basis}.\n{directions_url(alarm)}"
    )


def listing(state: State, now: datetime) -> str:
    """``leave_by list``: each standing alarm and what is known of it."""
    found = state.open()
    if not found:
        return "no leave-now alarms set"
    lines = []
    for a in found:
        line = f"{a.id}: be at {a.destination} by {_hhmm(a.arrive_by)}, from {a.origin}"
        last = a.last_eta
        if last is not None:
            line += f"; {last.eta_min} min by car as of {_hhmm(last.at)}"
        if a.ping("live") is not None:
            line += "; already told them to leave"
        lines.append(line)
    return "\n".join(lines)
