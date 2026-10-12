"""Shared fixtures for the M0 suite.

Note what is *not* here: nothing imports ``omega._log``. The seam is the whole
interface the suite is allowed to use, and if the seam were not sufficient for
its own tests it would not be sufficient (spec case 24).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Iterator

import pytest

# Let the suite import its own helpers (rawlog, crash_child) by plain name.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from omega.memory import EPISODES_FILENAME, MemoryStore  # noqa: E402


@pytest.fixture
def store_dir(tmp_path: Path) -> Path:
    """A fresh directory for one store."""
    d = tmp_path / "memory"
    d.mkdir()
    return d


@pytest.fixture
def log_path(store_dir: Path) -> Path:
    """The log file inside ``store_dir`` — for the byte-surgery cases."""
    return store_dir / EPISODES_FILENAME


@pytest.fixture
def store(store_dir: Path):
    """An open store, closed at teardown so the lock never leaks between tests."""
    s = MemoryStore.open(store_dir)
    try:
        yield s
    finally:
        s.close()


#: The zone a case pins when what it asserts is a rendered local date.
PINNED_ZONE = "Asia/Kolkata"


@pytest.fixture
def pinned_zone() -> Iterator[str]:
    """Run one case in :data:`PINNED_ZONE`, whatever zone the suite runs in.

    For a case whose expected text is a local rendering of a pinned instant
    ("since Sun 11 Oct"). Every ``astimezone()`` in omega reads the process
    zone, so setting ``TZ`` and calling ``tzset`` is the whole seam; the
    previous zone is put back afterwards, so the CI leg that runs the suite
    under UTC+14 still runs every other case there.
    """
    before = os.environ.get("TZ")
    os.environ["TZ"] = PINNED_ZONE
    time.tzset()
    try:
        yield PINNED_ZONE
    finally:
        if before is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = before
        time.tzset()


def seed(path: Path, n: int, key_prefix: str = "") -> list[bytes]:
    """Append ``n`` episodes and return their payloads, closing the store.

    Returns the payloads so a caller can assert exact contents rather than
    counts — a length check alone would pass on the wrong bytes.
    """
    payloads = [f"episode-{i}".encode() * (i % 3 + 1) for i in range(1, n + 1)]
    with MemoryStore.open(path) as s:
        for i, payload in enumerate(payloads, start=1):
            key = f"{key_prefix}{i}" if key_prefix else ""
            assert s.append_episode(payload, key, 1_000_000 + i) == i
    return payloads


# --- a far "now", for the CI leg that hunts date-dependent tests ---------------
#
# CI runs the suite again, in a parallel job, under ``TZ=Pacific/Kiritimati``
# (UTC+14) with ``OMEGA_TEST_NOW`` set to an instant years away. A test that
# passes only because of today's date, or the runner's zone, then fails on the
# push that adds it rather than on some later day (main's reds of 10-07 to
# 10-12).
#
# Known limit: only module-level names are swapped, once per case, so a
# function-local ``from datetime import datetime`` reads the real clock.
#
# Pure Python and test-only. Every ``datetime`` / ``date`` name bound in an
# ``omega`` or suite module is swapped for a subclass whose ``now``, ``today``
# and ``utcnow`` read the real clock plus an offset, and the offset is reset at
# the start of every case, so each case starts at exactly ``OMEGA_TEST_NOW``
# and time then passes at its real rate. Construction, parsing and
# ``isinstance`` behave exactly as with the real classes, whose instances are
# all anything ever gets. ``time.time`` and ``time.monotonic`` are left alone:
# they measure intervals, and a file's mtime is on the real clock.
#
# Unset - the default, and the normal run - nothing here does anything.

import datetime as _datetime_module  # noqa: E402

_REAL_DATETIME = _datetime_module.datetime
_REAL_DATE = _datetime_module.date
_UTC = _datetime_module.timezone.utc
_TESTS_DIR = str(Path(__file__).resolve().parent)
_SHIFT = [_datetime_module.timedelta(0)]


def _far_target() -> "_datetime_module.datetime | None":
    raw = os.environ.get("OMEGA_TEST_NOW", "").strip()
    if not raw:
        return None
    target = _REAL_DATETIME.fromisoformat(raw)
    return target if target.tzinfo is not None else target.astimezone()


_TARGET = _far_target()


class _DatetimeMeta(type):
    def __instancecheck__(cls, obj: object) -> bool:
        return isinstance(obj, _REAL_DATETIME)

    def __subclasscheck__(cls, sub: type) -> bool:
        return issubclass(sub, _REAL_DATETIME)


class _DateMeta(type):
    def __instancecheck__(cls, obj: object) -> bool:
        return isinstance(obj, _REAL_DATE)

    def __subclasscheck__(cls, sub: type) -> bool:
        return issubclass(sub, _REAL_DATE)


class _ShiftedDatetime(_REAL_DATETIME, metaclass=_DatetimeMeta):
    """``datetime`` with a shifted ``now``. Instances are always the real class."""

    def __new__(cls, *args, **kwargs):  # type: ignore[no-untyped-def]
        return _REAL_DATETIME(*args, **kwargs)

    @classmethod
    def now(cls, tz=None):  # type: ignore[no-untyped-def, override]
        return _REAL_DATETIME.now(tz) + _SHIFT[0]

    @classmethod
    def today(cls):  # type: ignore[no-untyped-def, override]
        return _REAL_DATETIME.now() + _SHIFT[0]

    @classmethod
    def utcnow(cls):  # type: ignore[no-untyped-def, override]
        return _REAL_DATETIME.now(_UTC).replace(tzinfo=None) + _SHIFT[0]


class _ShiftedDate(_REAL_DATE, metaclass=_DateMeta):
    """``date`` with a shifted ``today``. Instances are always the real class."""

    def __new__(cls, *args, **kwargs):  # type: ignore[no-untyped-def]
        return _REAL_DATE(*args, **kwargs)

    @classmethod
    def today(cls):  # type: ignore[no-untyped-def, override]
        return (_REAL_DATETIME.now() + _SHIFT[0]).date()


def _ours(name: str, module: object) -> bool:
    if module is sys.modules[__name__]:
        return False  # this file holds the real classes the shift reads
    if name == "omega" or name.startswith("omega."):
        return True
    return str(getattr(module, "__file__", "") or "").startswith(_TESTS_DIR)


@pytest.fixture(autouse=True)
def _far_now(monkeypatch: pytest.MonkeyPatch) -> None:
    if _TARGET is None:
        return
    _SHIFT[0] = _TARGET - _REAL_DATETIME.now(_UTC)
    for name, module in list(sys.modules.items()):
        if module is None or not _ours(name, module):
            continue
        for attr, value in list(vars(module).items()):
            if value is _REAL_DATETIME:
                monkeypatch.setattr(module, attr, _ShiftedDatetime)
            elif value is _REAL_DATE:
                monkeypatch.setattr(module, attr, _ShiftedDate)
