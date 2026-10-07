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
from dataclasses import dataclass
from typing import Callable, Optional

__all__ = [
    "Reading",
    "LOW_DISK_FRACTION",
    "LOW_DISK_BYTES",
    "LOW_BATTERY_PERCENT",
    "read",
    "parse_battery",
    "describe",
    "online",
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
