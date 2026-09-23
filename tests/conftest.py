"""Shared fixtures for the M0 suite.

Note what is *not* here: nothing imports ``omega._log``. The seam is the whole
interface the suite is allowed to use, and if the seam were not sufficient for
its own tests it would not be sufficient (spec case 24).
"""

from __future__ import annotations

import sys
from pathlib import Path

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
