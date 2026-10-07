"""The machine sense — what state this Mac is in, read mechanically. DL-068 #4.

The look (DL-061) was starved: every hour it saw the same clock and the same
claims, and silence was the right answer to an input that never changed. The
person asked for omega to speak "based on the condition of the system", and
this is that condition — the few numbers a helper sitting beside them would
notice without being asked: the disk filling, the battery running down, the
machine pinned.

**Mechanical, no model, no content.** Disk, battery and load are read from
the OS, never an app, a window or a file. That is the line DL-061 drew when it
rejected a live now-sense as the largest privacy surface omega could take on,
and this stays on the near side of it.

**Never raises.** Every reading is optional; one that cannot be taken is left
out rather than guessed, and a look over a machine it could not read is still a
look over everything else.
"""

from __future__ import annotations

import os
import re
import shutil
import socket
import subprocess
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from typing import Any, Callable, Optional

__all__ = [
    "Reading",
    "LOW_DISK_FRACTION",
    "LOW_DISK_BYTES",
    "LOW_BATTERY_PERCENT",
    "read",
    "parse_battery",
    "describe",
    "online",
    "STALE_SECONDS",
    "HEARTBEAT_SECONDS",
    "BATTERY_STEP",
    "AWAY_SECONDS",
    "PRESENCE_EVERY_SECONDS",
    "PRESENCE_STALE_SECONDS",
    "as_body",
    "from_body",
    "changed",
    "reported_lines",
    "parse_idle",
    "idle_seconds",
    "is_away",
    "MAC",
    "at_the_mac",
]

#: Below either of these the disk is called low. Both, because a fraction alone
#: calls a 4 TB disk with 300 GB free "low", and bytes alone never notice a small
#: disk that is nearly full.
LOW_DISK_FRACTION = 0.10
LOW_DISK_BYTES = 20 * 1024**3

#: A battery at or under this, and not charging, is called low.
LOW_BATTERY_PERCENT = 20

_GB = 1024**3


@dataclass(frozen=True)
class Reading:
    """One look at the machine. ``None`` means "could not read", never zero."""

    disk_free: Optional[int] = None
    disk_total: Optional[int] = None
    battery_percent: Optional[int] = None
    charging: Optional[bool] = None
    on_battery: Optional[bool] = None
    load: Optional[float] = None
    cpus: Optional[int] = None


def parse_battery(text: str) -> tuple[Optional[int], Optional[bool], Optional[bool]]:
    """``pmset -g batt`` as (percent, charging, on_battery). A desktop with no
    battery reports none, and that is ``(None, None, None)``, not an error."""
    on_battery: Optional[bool] = None
    if "Battery Power" in text:
        on_battery = True
    elif "AC Power" in text:
        on_battery = False
    match = re.search(r"(\d{1,3})%;\s*([a-zA-Z ]+?);", text)
    if match is None:
        return None, None, on_battery
    state = match.group(2).strip().lower()
    charging = state in ("charging", "charged", "finishing charge")
    return int(match.group(1)), charging, on_battery


def _battery() -> tuple[Optional[int], Optional[bool], Optional[bool]]:
    try:
        out = subprocess.run(
            ["pmset", "-g", "batt"], capture_output=True, text=True, timeout=3
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None, None, None
    return parse_battery(out)


def read(
    *,
    root: str = "/",
    battery: Callable[[], tuple[Optional[int], Optional[bool], Optional[bool]]] = _battery,
) -> Reading:
    """Take one reading. ``battery`` is injectable so a test never runs pmset."""
    free = total = None
    try:
        usage = shutil.disk_usage(root)
        free, total = usage.free, usage.total
    except OSError:
        pass
    try:
        percent, charging, on_battery = battery()
    except Exception:  # noqa: BLE001 - see the module docstring
        percent = charging = on_battery = None
    try:
        load: Optional[float] = os.getloadavg()[0]
    except OSError:
        load = None
    return Reading(
        disk_free=free,
        disk_total=total,
        battery_percent=percent,
        charging=charging,
        on_battery=on_battery,
        load=load,
        cpus=os.cpu_count(),
    )


def describe(reading: Reading) -> list[str]:
    """Plain lines for the look, with a ``LOW`` flag past a threshold.

    The flag is decided here, in code, not left to the model to infer from a
    number: "7 GB free" means nothing to a judge that does not know the disk is
    239 GB, and a threshold is exactly the kind of rule that belongs in code.
    """
    lines: list[str] = []
    if reading.disk_free is not None and reading.disk_total:
        free_gb = reading.disk_free / _GB
        pct = 100 * reading.disk_free / reading.disk_total
        low = (
            reading.disk_free < LOW_DISK_BYTES
            or reading.disk_free / reading.disk_total < LOW_DISK_FRACTION
        )
        lines.append(
            f"Disk: {free_gb:.0f} GB free of {reading.disk_total / _GB:.0f} GB "
            f"({pct:.0f}%)" + (" - LOW" if low else "")
        )
    if reading.battery_percent is not None:
        if reading.charging:
            state = "charging"
        elif reading.on_battery:
            state = "on battery"
        else:
            state = "plugged in"
        low = (
            reading.battery_percent <= LOW_BATTERY_PERCENT
            and not reading.charging
            and reading.on_battery is not False
        )
        lines.append(
            f"Battery: {reading.battery_percent}%, {state}" + (" - LOW" if low else "")
        )
    if reading.load is not None and reading.cpus:
        busy = reading.load > reading.cpus
        lines.append(
            f"Load: {reading.load:.1f} across {reading.cpus} cores"
            + (" - the machine is saturated" if busy else "")
        )
    return lines


def online(host: str = "api.openai.com", port: int = 443, timeout: float = 2.0) -> bool:
    """Is there a route to the provider right now? DL-068 #5.

    A TCP connect, nothing sent. Asked before a look or a fire, because a turn
    started with no network is a turn that fails on its first call — 25 of the
    first 60 looks did exactly that while the Mac sat in DarkWake — and a
    skipped look costs nothing where a failed turn costs a row in the log.
    """
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


# --- readings that travel (DL-072, DL-073 §3) --------------------------------
# Once the core left the Mac (DL-069) the reading above described the VM, which
# is not the machine the person sits at. The Mac's relay now takes the same
# reading and sends it, and the core shows both. Everything below is about a
# reading that was taken *somewhere else, some time ago* — which is why each
# piece of it is about age.

#: Past this a reported reading is not shown as its last value (DL-072). A
#: heartbeat comes every 30 minutes, so 90 is three missed heartbeats: long
#: enough that one dropped connection is not "couldn't determine", short enough
#: that a Mac asleep since lunch is not described as it was at lunch.
STALE_SECONDS = 90 * 60

#: How often an unchanged reading is resent anyway, so silence from the relay
#: can be told apart from a machine whose condition simply has not moved.
HEARTBEAT_SECONDS = 30 * 60

#: A battery that moved this many points since the last report is a change.
BATTERY_STEP = 10

#: Idle at or past this, the person is away from the Mac (DL-073 §3).
AWAY_SECONDS = 10 * 60

#: How often a presence report is resent while the person is active. Shorter
#: than :data:`PRESENCE_STALE_SECONDS` on purpose, so a person working steadily
#: is never read as "unknown" between two reports.
PRESENCE_EVERY_SECONDS = 5 * 60

#: Past this a presence report says nothing, and nothing means "not at the Mac".
PRESENCE_STALE_SECONDS = 10 * 60


def as_body(reading: Reading) -> dict[str, Any]:
    """A reading as the body of a ``report`` — its fields, by their names."""
    return asdict(reading)


def from_body(body: dict[str, Any]) -> Reading:
    """The reverse, ignoring any field this version does not know."""
    known = {f.name for f in fields(Reading)}
    return Reading(**{k: v for k, v in body.items() if k in known})


def _disk_low(r: Reading) -> Optional[bool]:
    if r.disk_free is None or not r.disk_total:
        return None
    return r.disk_free < LOW_DISK_BYTES or r.disk_free / r.disk_total < LOW_DISK_FRACTION


def _battery_low(r: Reading) -> Optional[bool]:
    if r.battery_percent is None:
        return None
    return (
        r.battery_percent <= LOW_BATTERY_PERCENT
        and not r.charging
        and r.on_battery is not False
    )


def changed(previous: Optional[Reading], current: Reading) -> bool:
    """Is ``current`` worth sending, given the last one sent? (DL-072)

    A low flag flipping, the battery moving by :data:`BATTERY_STEP`, or the
    power source changing. Not the disk's exact byte count or the load, which
    move on every reading and would turn "on change" into "always".
    """
    if previous is None:
        return True
    if _disk_low(previous) != _disk_low(current):
        return True
    if _battery_low(previous) != _battery_low(current):
        return True
    if previous.on_battery != current.on_battery:
        return True
    if (previous.battery_percent is None) != (current.battery_percent is None):
        return True
    if (
        previous.battery_percent is not None
        and current.battery_percent is not None
        and abs(previous.battery_percent - current.battery_percent) >= BATTERY_STEP
    ):
        return True
    return False


def _age_seconds(at: str, now: datetime) -> Optional[float]:
    """Seconds since ``at``, or ``None`` when ``at`` cannot be read as a time.

    A reading stamped in the future (a clock ahead of the core's) counts as
    fresh rather than as negative age; it is still the newest thing known.
    """
    try:
        stamp = datetime.fromisoformat(at)
    except (TypeError, ValueError):
        return None
    if stamp.tzinfo is None:
        return None
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return max(0.0, (now - stamp).total_seconds())


def _ago(seconds: float) -> str:
    minutes = int(seconds // 60)
    if minutes < 120:
        return f"{minutes} min ago"
    return f"{minutes // 60} h ago"


def reported_lines(device: str, at: str, body: dict[str, Any], *, now: datetime) -> list[str]:
    """One reported device's lines for the look, labeled with the device.

    **Fails closed on age** (DL-072). Past :data:`STALE_SECONDS`, or with a
    stamp that cannot be read, the device renders as "couldn't determine" and
    never as its last value — a battery at 80% three hours ago is not a battery
    at 80%, and a look that said so would be a confident sentence about a
    machine nobody has heard from.
    """
    age = _age_seconds(at, now)
    if age is None:
        return [f"{device}: couldn't determine (its last reading has no readable time)"]
    if age > STALE_SECONDS:
        return [f"{device}: couldn't determine (last reading {_ago(age)}, too old to trust)"]
    lines = describe(from_body(body))
    if not lines:
        return [f"{device}: couldn't determine (its last reading held nothing)"]
    return [f"{device}: {line}" for line in lines]


def parse_idle(text: str) -> Optional[float]:
    """``ioreg -c IOHIDSystem`` output as seconds since the last input.

    ``HIDIdleTime`` is nanoseconds since the last keyboard, mouse or trackpad
    event. It is the one number read: no app, no window, no content — the near
    side of the line DL-061 drew, which DL-073 §3 keeps.
    """
    match = re.search(r'"HIDIdleTime"\s*=\s*(\d+)', text)
    if match is None:
        return None
    return int(match.group(1)) / 1e9


def idle_seconds() -> Optional[float]:
    """Seconds since the person last touched this Mac, or ``None`` if unreadable.

    Needs no permission prompt. ``None`` sends nothing, and nothing goes stale,
    and stale reads as "not at the Mac" — so a failure here routes toward the
    phone, which is the direction DL-073 chose for unknown.
    """
    try:
        out = subprocess.run(
            ["ioreg", "-c", "IOHIDSystem", "-d", "4"],
            capture_output=True,
            text=True,
            timeout=3,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_idle(out)


def is_away(idle: float) -> bool:
    return idle >= AWAY_SECONDS


#: The device name the Mac relay reports under, and the one "at the Mac" reads.
#: Here rather than in the relay so the core can ask about presence without
#: importing the relay (which imports the channel).
MAC = "mac"


def at_the_mac(presence: Optional[Any], *, now: datetime) -> bool:
    """Is the person at the Mac right now? (DL-073 §3)

    ``presence`` is the latest presence report — anything with ``at`` and
    ``body`` (:class:`omega.derive.Reported`) — or ``None`` if none ever came.
    True **only** when that report is under :data:`PRESENCE_STALE_SECONDS` old
    *and* says idle under :data:`AWAY_SECONDS`. Missing, stale, unreadable or
    away are all ``False``.

    **It fails toward the phone.** Every way of not knowing answers "not at the
    Mac", because the cost of being wrong is lopsided: a message sent to the
    phone while the person was at the Mac buzzes twice, and one held for a Mac
    nobody is sitting at is never seen.
    """
    if presence is None:
        return False
    age = _age_seconds(getattr(presence, "at", None), now)
    if age is None or age >= PRESENCE_STALE_SECONDS:
        return False
    body = getattr(presence, "body", None)
    idle = body.get("idle") if isinstance(body, dict) else None
    if isinstance(idle, bool) or not isinstance(idle, (int, float)) or idle < 0:
        return False
    return not is_away(idle)
