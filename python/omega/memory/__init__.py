"""THE SEAM — the only module in the tree permitted to import ``omega._log``.

DL-018: Rust owns structure, Python owns meaning. Everything below this line
speaks **episodes**; everything above it (the rest of ``omega``) must never
learn that frames, offsets, CRCs or file headers exist. That rule is enforced
by a test (spec case 24), not by discipline.

This is a seam, not a layer: it translates vocabulary and nothing else. No
caching, no policy, no retries, no cleverness. The one thing it decides is
where the log file lives when it is handed a directory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Union

from omega import _log

__all__ = [
    "Episode",
    "MemoryStore",
    # Limits and names that callers legitimately need to reason about episodes.
    "EPISODES_FILENAME",
    "MAX_BODY",
    "MAX_KEY",
    "HEADER_LEN",
    "VERSION",
    # Errors, re-exported so nothing else ever has to reach for omega._log.
    "OmegaLogError",
    "AlreadyLocked",
    "CheckpointAhead",
    "CorruptFrame",
    "LogClosed",
    "NotAnOmegaLog",
    "SequenceBreak",
    "TooLarge",
    "UnsupportedVersion",
    "WriteKeyConflict",
]

PathLike = Union[str, "os.PathLike[str]"]

# --- constants ------------------------------------------------------------
EPISODES_FILENAME: str = _log.EPISODES_FILENAME
MAX_BODY: int = _log.MAX_BODY
MAX_KEY: int = _log.MAX_KEY
HEADER_LEN: int = _log.HEADER_LEN
VERSION: int = _log.VERSION

# --- errors ---------------------------------------------------------------
OmegaLogError = _log.OmegaLogError
AlreadyLocked = _log.AlreadyLocked
CheckpointAhead = _log.CheckpointAhead
CorruptFrame = _log.CorruptFrame
LogClosed = _log.LogClosed
NotAnOmegaLog = _log.NotAnOmegaLog
SequenceBreak = _log.SequenceBreak
TooLarge = _log.TooLarge
UnsupportedVersion = _log.UnsupportedVersion
WriteKeyConflict = _log.WriteKeyConflict


@dataclass(frozen=True)
class Episode:
    """One episode as it came back out of the log.

    ``payload`` is opaque bytes: the log never parsed it and neither does this.
    """

    seq: int
    ts_micros: int
    write_key: str
    payload: bytes


class _Diagnostics:
    """Deliberately-named escape hatch for the things that are *not* episodes.

    The offset index, the file size and the truncation count are frame-level
    facts. They exist here only so the M0 suite can assert the spec's
    index-is-a-cache cases (34, 35) without importing ``omega._log`` — which
    would defeat the seam it is testing. Nothing in ``omega`` outside the tests
    should touch this.
    """

    __slots__ = ("_log",)

    def __init__(self, raw: "_log.Log") -> None:
        self._log = raw

    @property
    def path(self) -> Path:
        return Path(self._log.path)

    @property
    def recovered_bytes(self) -> int:
        """Bytes discarded by torn-tail truncation when this handle opened."""
        return self._log.recovered_bytes

    def size_bytes(self) -> int:
        return self._log.size_bytes()

    def offsets(self) -> list[int]:
        """``offsets()[n - 1]`` is the file offset of the episode with seq n."""
        return self._log.offsets()

    def rebuild_indexes(self) -> None:
        """Drop both in-memory caches and rebuild them from the file."""
        self._log.rebuild_indexes()


class MemoryStore:
    """Episodes go in durably and come back out in order. That is all of M0."""

    __slots__ = ("_log", "_diagnostics")

    def __init__(self, raw: "_log.Log") -> None:
        self._log = raw
        self._diagnostics = _Diagnostics(raw)

    # --- opening ----------------------------------------------------------
    @classmethod
    def open(cls, path: PathLike) -> "MemoryStore":
        """Open (creating if needed) the store at ``path``.

        ``path`` may be the log **file** or a **directory**; an existing
        directory gets ``EPISODES_FILENAME`` joined onto it, so callers never
        have to know the log's filename.
        """
        p = Path(os.fspath(path))
        if p.is_dir():
            p = p / EPISODES_FILENAME
        return cls(_log.Log(p))

    # --- writing ----------------------------------------------------------
    def append_episode(
        self,
        payload: bytes,
        write_key: str = "",
        ts_micros: Optional[int] = None,
    ) -> int:
        """Append one episode, fsync, and return its sequence number.

        An empty ``write_key`` means "no key" and is never deduplicated. A
        non-empty key that is already stored returns the existing seq when the
        payload is identical and raises ``WriteKeyConflict`` when it is not.
        """
        return self._log.append(payload, write_key, ts_micros)

    # --- reading ----------------------------------------------------------
    def head(self) -> int:
        """Seq of the last episode, or 0 when the store is empty."""
        return self._log.head()

    def episodes_since(self, seq: int) -> Iterator[Episode]:
        """Episodes with sequence > ``seq``, in order.

        ``seq > head()`` raises ``CheckpointAhead`` here and now — never an
        empty iterator. The raise is eager on purpose: a check that passes on
        empty is not a check.
        """
        records = self._log.episodes_since(seq)
        return (Episode(*record) for record in records)

    # --- checkpoints ------------------------------------------------------
    def checkpoint(self, name: str) -> int:
        """An unset checkpoint reads as 0, never None."""
        return self._log.checkpoint(name)

    def set_checkpoint(self, name: str, seq: int) -> None:
        self._log.set_checkpoint(name, seq)

    def checkpoint_names(self) -> list[str]:
        return self._log.checkpoint_names()

    # --- lifetime ---------------------------------------------------------
    @property
    def closed(self) -> bool:
        return self._log.closed

    @property
    def diagnostics(self) -> _Diagnostics:
        return self._diagnostics

    def close(self) -> None:
        """Release the lock and the file handle. Idempotent."""
        self._log.close()

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.close()
        return False

    def __repr__(self) -> str:
        if self.closed:
            return f"<MemoryStore {str(self._diagnostics.path)!r} closed>"
        return f"<MemoryStore {str(self._diagnostics.path)!r} head={self.head()}>"
