"""Where the person is (DL-091).

A fold over the episodic log, the way :mod:`omega.followups` and
:mod:`omega.promises` are: nothing is stored beside the log, so the current
location and the saved places survive a restart because the records they are
made of are in the log.

Three records feed it (:mod:`omega.episodes`):

- ``location.set`` is one fix: what they said ("I'm at the office"), what
  their phone's Shortcut posted, or a saved place they named. History is kept;
  the **current** location is simply the latest fix, and it is **fresh** for
  :data:`FRESH_SECONDS`. After that it is only ever their *last known*
  location, worded as such, never as where they are now.
- ``place.saved`` names a place ("home"), and ``place.forgotten`` drops it.

Where it may go is narrow on purpose. A prompted turn (their own message) sees
one context line and the ``location`` tool; nothing unprompted does, so a
brief, a watch, a look or a tell never mentions it. It is never recalled as
conversation, never learned as a claim, never projected to a surface, and
omega sends it nowhere: the only way it leaves is as an argument the model
chooses to pass a tool (a maps search, say) for the question they asked.

The phone's Shortcut reaches it through :class:`Listener`: a stdlib HTTP
listener on loopback, started only when :data:`ENV_TOKEN` is set, that takes
``POST /location`` with a bearer token and hands one ``location.set`` to the
core's own append, so the log keeps its one writer.
"""

from __future__ import annotations

import hmac
import json
import math
import re
import socket
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, Optional

from omega import episodes

if TYPE_CHECKING:  # pragma: no cover
    from omega.memory import MemoryStore

__all__ = [
    "FRESH_SECONDS",
    "ENV_TOKEN",
    "ENV_PORT",
    "DEFAULT_PORT",
    "MAX_BODY",
    "MIN_TOKEN",
    "RATE_LIMIT",
    "RATE_WINDOW",
    "Fix",
    "Place",
    "State",
    "Locations",
    "fold",
    "describe",
    "context_line",
    "place_name",
    "parse_when",
    "Settings",
    "settings",
    "BadReport",
    "parse_report",
    "Listener",
]

#: How long a fix counts as where they are: three hours. After that it is
#: their last known location, and is worded so.
FRESH_SECONDS = 3 * 3600

#: The Shortcut's shared secret. Unset, the listener does not start at all.
ENV_TOKEN = "OMEGA_LOCATION_TOKEN"
#: The loopback port it listens on; ``tailscale serve`` fronts it.
ENV_PORT = "OMEGA_LOCATION_PORT"
#: One above the channel's 7717.
DEFAULT_PORT = 7718
#: A fix is a few numbers and two short strings; anything bigger is not one.
MAX_BODY = 4096
#: A token shorter than this is a password someone could guess, and the
#: listener refuses to start on it rather than guard a door with it.
MIN_TOKEN = 16
#: Requests per window, counted whatever their outcome: a phone posts a fix a
#: few times an hour, so a dozen a minute is already a script.
RATE_LIMIT = 12
RATE_WINDOW = 60.0

#: How a fix's source reads in a sentence.
_VIA = {"said": "what they told you", "shortcut": "their phone", "saved": "a place they saved"}
_HISTORY_LINES = 50


# --- the view -------------------------------------------------------------------


@dataclass(frozen=True)
class Fix:
    """One ``location.set``."""

    seq: int
    source: str
    at: datetime
    label: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    accuracy_m: Optional[float] = None
    city: Optional[str] = None
    place: Optional[str] = None


@dataclass(frozen=True)
class Place:
    """One saved place, by its name."""

    name: str
    seq: int
    label: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None


@dataclass(frozen=True)
class State:
    """The fold as of :attr:`through`. Empty is no record at all."""

    through: int = 0
    fixes: tuple[Fix, ...] = ()
    places: Mapping[str, Place] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.fixes or self.places)

    @property
    def current(self) -> Optional[Fix]:
        """The latest fix in the log, or ``None``."""
        return self.fixes[-1] if self.fixes else None

    def at_or_before(self, moment: datetime) -> Optional[Fix]:
        """The newest fix recorded at or before ``moment`` (ties to the later
        seq), or ``None``."""
        best: Optional[Fix] = None
        for fix in self.fixes:
            if fix.at <= moment and (best is None or fix.at >= best.at):
                best = fix
        return best

    def since(self, moment: datetime) -> list[Fix]:
        """Fixes recorded at or after ``moment``, newest first."""
        return sorted((f for f in self.fixes if f.at >= moment),
                      key=lambda f: (f.at, f.seq), reverse=True)


def _stamp(raw: Any) -> Optional[datetime]:
    try:
        moment = datetime.fromisoformat(str(raw))
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


class Locations:
    """The fold. ``apply`` in seq order; :meth:`state` snapshots it."""

    __slots__ = ("_through", "_fixes", "_places")

    def __init__(self) -> None:
        self._through = 0
        self._fixes: list[Fix] = []
        self._places: dict[str, Place] = {}

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
        if kind == episodes.LOCATION_SET:
            at = _stamp(payload.get("at"))
            if at is None:  # pragma: no cover - the validator requires a zone
                return
            self._fixes.append(Fix(
                seq=seq, source=payload["source"], at=at, label=payload.get("label"),
                lat=payload.get("lat"), lon=payload.get("lon"),
                accuracy_m=payload.get("accuracy_m"), city=payload.get("city"),
                place=payload.get("place"),
            ))
        elif kind == episodes.PLACE_SAVED:
            name = payload["name"]
            self._places[name] = Place(
                name=name, seq=seq, label=payload.get("label"),
                lat=payload.get("lat"), lon=payload.get("lon"),
            )
        elif kind == episodes.PLACE_FORGOTTEN:
            self._places.pop(payload["name"], None)

    def state(self) -> State:
        return State(through=self._through, fixes=tuple(self._fixes),
                     places=dict(self._places))

    def advance(self, store: "MemoryStore", *, upto: Optional[int] = None) -> "Locations":
        """Fold what the log gained since :attr:`through`, to ``upto``."""
        for episode in store.episodes_since(self._through):
            if upto is not None and episode.seq > upto:
                break
            self.apply(episode.seq, episodes.decode(episode.payload))
        return self

    @classmethod
    def fold(cls, decoded: Iterable[tuple[int, dict[str, Any]]]) -> "Locations":
        view = cls()
        for seq, payload in decoded:
            view.apply(seq, payload)
        return view


def fold(store: "MemoryStore") -> State:
    """The whole log, folded."""
    return Locations().advance(store).state()


# --- words ----------------------------------------------------------------------


def _coords(lat: Optional[float], lon: Optional[float], accuracy: Optional[float] = None) -> str:
    if lat is None or lon is None:
        return ""
    text = f"{lat:.5f},{lon:.5f}"
    if accuracy is not None:
        text += f" ±{accuracy:.0f} m"
    return text


def describe(fix: Any) -> str:
    """A fix (or a :class:`Place`) as words a maps tool can take: its name,
    label or city, with the coordinates after it when there are any."""
    place = getattr(fix, "place", None) or (fix.name if isinstance(fix, Place) else None)
    label = fix.label
    city = getattr(fix, "city", None)
    if place and label and label.casefold() != place:
        name = f"{place} ({label})"
    else:
        name = place or label or ""
    if city and city.casefold() not in name.casefold():
        name = f"{name}, {city}" if name else city
    coords = _coords(fix.lat, fix.lon, getattr(fix, "accuracy_m", None))
    if name and coords:
        return f"{name} [{coords}]"
    return name or coords


def _local(moment: datetime) -> datetime:
    return moment.astimezone()


def _clock(moment: datetime, now: datetime) -> str:
    """HH:MM, with the day when it is not today's."""
    local, today = _local(moment), _local(now)
    if local.date() == today.date():
        return f"{local:%H:%M}"
    return f"{local:%H:%M} on {local:%a %d %b}"


def _age(delta: timedelta) -> str:
    minutes = max(0, int(delta.total_seconds() // 60))
    if minutes < 60:
        return f"{minutes} minute{'s' if minutes != 1 else ''}"
    hours = minutes // 60
    if hours < 48:
        return f"{hours} hour{'s' if hours != 1 else ''}"
    days = hours // 24
    return f"{days} day{'s' if days != 1 else ''}"


def fresh(fix: Fix, now: datetime) -> bool:
    """Is ``fix`` still where they are, as of ``now``? A fix stamped a little
    after ``now`` (a clock a second ahead) counts as fresh."""
    return (now - fix.at).total_seconds() <= FRESH_SECONDS


def context_line(state: Optional[State], at: Any) -> str:
    """The one line a prompted turn gets, or ``""`` when there is no fix.

    Fresh: ``Their location: <where> (as of HH:MM, via <source>)``. Stale:
    their *last known* location and when, stated as not current.
    """
    fix = state.current if state else None
    if fix is None:
        return ""
    now = _stamp(at) or datetime.now(timezone.utc)
    where, when, via = describe(fix), _clock(fix.at, now), _VIA[fix.source]
    if fresh(fix, now):
        return f"Their location: {where} (as of {when}, via {via}).\n"
    return (
        f"Their location now is unknown: last known {where} at {when}, via {via}, "
        f"{_age(now - fix.at)} ago. Do not treat it as where they are now; ask "
        f"if it matters.\n"
    )


_SPACES = re.compile(r"\s+")


def place_name(text: str) -> str:
    """A saved place's name as the log keeps it: trimmed, spaced once,
    case-folded, without trailing punctuation."""
    return _SPACES.sub(" ", text).strip().strip(".,;:!?").strip().casefold()


_COORDS = re.compile(r"^\s*(-?\d{1,3}(?:\.\d+)?)\s*,\s*(-?\d{1,3}(?:\.\d+)?)\s*$")


def coordinates(text: str) -> Optional[tuple[float, float]]:
    """``"lat,lon"`` in range, or ``None``."""
    found = _COORDS.match(text)
    if not found:
        return None
    lat, lon = float(found.group(1)), float(found.group(2))
    if -90 <= lat <= 90 and -180 <= lon <= 180:
        return lat, lon
    return None


_AGO = re.compile(r"^(\d{1,4})\s*(m|min|mins|minute|minutes|h|hr|hrs|hour|hours|d|day|days)\s+ago$")
_HHMM = re.compile(r"^(\d{1,2})[:.](\d{2})$")


def parse_when(text: str, now: datetime) -> Optional[datetime]:
    """A past moment from words, in their zone, or ``None`` when it cannot be
    read. Takes an ISO date or datetime (a bare date means the end of that
    day), ``HH:MM`` (today, or yesterday when that is still ahead),
    ``yesterday`` with or without ``HH:MM``, and ``N minutes|hours|days ago``.
    """
    words = _SPACES.sub(" ", text).strip().casefold()
    local = _local(now)
    if not words:
        return None
    ago = _AGO.match(words)
    if ago:
        n, unit = int(ago.group(1)), ago.group(2)[0]
        step = {"m": timedelta(minutes=n), "h": timedelta(hours=n), "d": timedelta(days=n)}[unit]
        return now - step
    if words == "yesterday" or words.startswith("yesterday "):
        rest = words[len("yesterday"):].strip().removeprefix("at ").strip()
        day = local.date() - timedelta(days=1)
        if not rest:
            return datetime.combine(day, datetime.max.time().replace(microsecond=0), local.tzinfo)
        hhmm = _HHMM.match(rest)
        if not hhmm or int(hhmm.group(1)) > 23 or int(hhmm.group(2)) > 59:
            return None
        return datetime.combine(day, datetime.min.time(), local.tzinfo).replace(
            hour=int(hhmm.group(1)), minute=int(hhmm.group(2)))
    hhmm = _HHMM.match(words.removeprefix("at ").strip())
    if hhmm:
        hour, minute = int(hhmm.group(1)), int(hhmm.group(2))
        if hour > 23 or minute > 59:
            return None
        moment = local.replace(hour=hour, minute=minute, second=0, microsecond=0)
        return moment - timedelta(days=1) if moment > local else moment
    try:
        day = date.fromisoformat(words)
    except ValueError:
        day = None
    if day is not None:
        return datetime.combine(day, datetime.max.time().replace(microsecond=0), local.tzinfo)
    try:
        moment = datetime.fromisoformat(text.strip())
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=local.tzinfo)


# --- what the tool answers --------------------------------------------------------


#: What ``where`` adds when there is no fix or only a stale one: how they
#: can give one, and that a place recalled from earlier conversation is
#: only ever where they were as of when they said it.
_UNKNOWN_HELP = (
    "They can say \"I'm at <place>\", save a place (\"home is <address>\"), or "
    "set up the iPhone Shortcut. A place recalled from earlier conversation "
    "is not where they are now: give it only with its date (\"as of ...\")."
)


def where_text(state: State, now: datetime, when: Optional[datetime] = None) -> str:
    """``where``: now, or as of ``when``."""
    if when is None:
        fix = state.current
        if fix is None:
            return f"no location recorded: they have not said where they are. {_UNKNOWN_HELP}"
        line = f"{describe(fix)}, as of {_clock(fix.at, now)} via {_VIA[fix.source]}"
        if fresh(fix, now):
            return f"their location: {line}"
        return (f"not known now; last known {line}, {_age(now - fix.at)} ago "
                f"(older than {FRESH_SECONDS // 3600} hours, so not where they are now). "
                f"{_UNKNOWN_HELP}")
    fix = state.at_or_before(when)
    if fix is None:
        return "no location recorded at or before then"
    gap = when - fix.at
    note = "" if gap < timedelta(minutes=1) else f", {_age(gap)} before then"
    return (f"at {_clock(when, now)} the latest location recorded was "
            f"{describe(fix)}, as of {_clock(fix.at, now)} via {_VIA[fix.source]}{note}")


def places_text(state: State) -> str:
    if not state.places:
        return "no saved places"
    return "\n".join(f"{p.name}: {describe(Place(name='', seq=p.seq, label=p.label, lat=p.lat, lon=p.lon)) or p.name}"
                     for p in sorted(state.places.values(), key=lambda p: p.name))


def history_text(state: State, now: datetime, days: int) -> str:
    found = state.since(now - timedelta(days=days))
    if not found:
        return f"no location recorded in the last {days} day{'s' if days != 1 else ''}"
    lines = [f"{_clock(f.at, now)}: {describe(f)} (via {_VIA[f.source]})"
             for f in found[:_HISTORY_LINES]]
    if len(found) > _HISTORY_LINES:
        lines.append(f"+{len(found) - _HISTORY_LINES} older")
    return "\n".join(lines)


# --- the Shortcut's door ------------------------------------------------------------


@dataclass(frozen=True)
class Settings:
    """What the environment asked for. ``token`` empty: no listener."""

    token: str = ""
    port: int = DEFAULT_PORT
    #: Why it will not start although a token is set, or ``""``.
    problem: str = ""

    @property
    def wanted(self) -> bool:
        return bool(self.token)

    @property
    def startable(self) -> bool:
        return bool(self.token) and not self.problem


def settings(environ: Mapping[str, str]) -> Settings:
    """Read :data:`ENV_TOKEN` and :data:`ENV_PORT`. The token is never put in
    a message, only its absence or its length."""
    token = (environ.get(ENV_TOKEN) or "").strip()
    if not token:
        return Settings()
    raw = (environ.get(ENV_PORT) or "").strip()
    port = DEFAULT_PORT
    if raw:
        if not raw.isdigit() or not 1 <= int(raw) <= 65535:
            return Settings(token=token, port=0, problem=f"{ENV_PORT} is not a port")
        port = int(raw)
    if len(token) < MIN_TOKEN:
        return Settings(token=token, port=port,
                        problem=f"{ENV_TOKEN} is shorter than {MIN_TOKEN} characters")
    return Settings(token=token, port=port)


class BadReport(ValueError):
    """A body that is not a fix. The message names the field, never a value."""


_REPORT_KEYS = frozenset({"lat", "lon", "accuracy", "label", "city"})


def _number(body: Mapping[str, Any], key: str) -> Optional[float]:
    value = body.get(key)
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise BadReport(f"{key} must be a number")
    if isinstance(value, str):
        try:
            value = float(value.strip())
        except ValueError:
            raise BadReport(f"{key} must be a number") from None
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise BadReport(f"{key} must be a number")
    return float(value)


def _string(body: Mapping[str, Any], key: str, cap: int) -> Optional[str]:
    value = body.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise BadReport(f"{key} must be a string")
    value = _SPACES.sub(" ", value).strip()
    if not value:
        return None
    if len(value) > cap:
        raise BadReport(f"{key} is longer than {cap} characters")
    return value


def parse_report(raw: bytes, *, at: Optional[str] = None) -> dict[str, Any]:
    """A Shortcut's body as a ``location.set`` episode, or :class:`BadReport`.

    ``{lat, lon, accuracy?, label?, city?}``; lat and lon together or not at
    all, and a city alone is a rough fix. Numbers may come as strings, since a
    Shortcut's JSON builder types them loosely.
    """
    try:
        body = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise BadReport("the body is not JSON") from None
    if not isinstance(body, dict):
        raise BadReport("the body must be a JSON object")
    unknown = set(body) - _REPORT_KEYS
    if unknown:
        raise BadReport(f"unknown fields; allowed are {sorted(_REPORT_KEYS)}")
    lat, lon = _number(body, "lat"), _number(body, "lon")
    if (lat is None) != (lon is None):
        raise BadReport("lat and lon come together")
    if lat is not None and not -90 <= lat <= 90:
        raise BadReport("lat must be within ±90")
    if lon is not None and not -180 <= lon <= 180:
        raise BadReport("lon must be within ±180")
    accuracy = _number(body, "accuracy")
    if accuracy is not None and not 0 <= accuracy <= episodes.MAX_LOCATION_ACCURACY:
        raise BadReport(f"accuracy must be 0 to {episodes.MAX_LOCATION_ACCURACY} metres")
    label = _string(body, "label", episodes.MAX_LOCATION_LABEL)
    city = _string(body, "city", episodes.MAX_LOCATION_CITY)
    if lat is None and city is None:
        raise BadReport("a fix needs lat and lon, or a city")
    try:
        return episodes.location_set(source="shortcut", label=label, lat=lat, lon=lon,
                                     accuracy_m=accuracy, city=city, at=at)
    except episodes.BadPayload:  # pragma: no cover - checked above, field by field
        raise BadReport("the fix is not valid") from None


class _Server(HTTPServer):
    # One request at a time, and a slow one cannot hold the door: the
    # handler's socket times out.
    allow_reuse_address = True
    timeout = 5

    def __init__(self, address: tuple[str, int], listener: "Listener") -> None:
        self.listener = listener
        super().__init__(address, _Handler)


class _Handler(BaseHTTPRequestHandler):
    server: _Server
    timeout = 5
    server_version = "omega"
    sys_version = ""

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        # Nothing is logged: not the request line, not a header, not a body.
        return

    def _reply(self, code: int, body: dict[str, Any], *, auth: bool = False) -> None:
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        if auth:
            self.send_header("WWW-Authenticate", 'Bearer realm="omega"')
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(data)
        self.close_connection = True

    def _refuse_method(self) -> None:
        if not self.server.listener.admit():
            return self._reply(429, {"error": "too many requests"})
        self._reply(405 if self.path.split("?")[0] == "/location" else 404,
                    {"error": "only POST /location"})

    do_GET = do_PUT = do_DELETE = do_PATCH = do_HEAD = do_OPTIONS = _refuse_method

    def do_POST(self) -> None:  # noqa: N802
        listener = self.server.listener
        if not listener.admit():
            return self._reply(429, {"error": "too many requests"})
        if self.path.split("?")[0] != "/location":
            return self._reply(404, {"error": "only POST /location"})
        if not listener.authorized(self.headers.get("Authorization", "")):
            return self._reply(401, {"error": "unauthorized"}, auth=True)
        length = self.headers.get("Content-Length")
        if length is None or not length.strip().isdigit():
            return self._reply(411, {"error": "Content-Length is required"})
        if int(length) > MAX_BODY:
            return self._reply(413, {"error": f"the body is over {MAX_BODY} bytes"})
        try:
            raw = self.rfile.read(int(length))
        except (OSError, socket.timeout):
            return
        try:
            payload = parse_report(raw)
        except BadReport as exc:
            return self._reply(400, {"error": str(exc)})
        try:
            listener.append(payload)
        except Exception:  # the core is stopping, or its log refused it
            return self._reply(503, {"error": "omega could not record it"})
        self._reply(200, {"ok": True})


class Listener:
    """The Shortcut's door: ``POST /location`` on loopback, behind a bearer
    token, appending through ``append`` (the core's own single writer).

    Never started without a token: :func:`settings` decides, and the runtime
    does not build one otherwise.
    """

    def __init__(
        self,
        token: str,
        append: Callable[[dict[str, Any]], Any],
        *,
        host: str = "127.0.0.1",
        port: int = DEFAULT_PORT,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not token:
            raise ValueError(f"no {ENV_TOKEN}: the location listener does not start")
        if host not in ("127.0.0.1", "::1", "localhost"):
            raise ValueError("the location listener binds loopback only")
        self._expected = f"Bearer {token}".encode()
        self.append = append
        self._host, self._port = host, port
        self._clock = clock
        self._hits: deque[float] = deque()
        self._lock = threading.Lock()
        self._server: Optional[_Server] = None
        self._thread: Optional[threading.Thread] = None

    def authorized(self, header: str) -> bool:
        # Constant time over the whole header value, so neither the scheme nor
        # a prefix of the token can be found by timing.
        return hmac.compare_digest(header.strip().encode(), self._expected)

    def admit(self) -> bool:
        """Count one request; ``False`` once the window holds
        :data:`RATE_LIMIT`."""
        with self._lock:
            now = self._clock()
            while self._hits and now - self._hits[0] >= RATE_WINDOW:
                self._hits.popleft()
            if len(self._hits) >= RATE_LIMIT:
                return False
            self._hits.append(now)
            return True

    @property
    def address(self) -> Optional[tuple[str, int]]:
        if self._server is None:
            return None
        host, port = self._server.server_address[:2]
        return str(host), int(port)

    def start(self) -> "Listener":
        self._server = _Server((self._host, self._port), self)
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="omega-location", daemon=True
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        server, self._server = self._server, None
        if server is not None:
            server.shutdown()
            server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
