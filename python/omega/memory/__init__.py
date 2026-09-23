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
    # Durability, made observable. Part of the hatch, not the interface.
    "file_syncs",
    "dir_syncs",
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

# The widths the log is written in: sequence numbers are u64, timestamps i64.
_MAX_SEQ = (1 << 64) - 1
_MIN_TS, _MAX_TS = -(1 << 63), (1 << 63) - 1


def _in_range(value: object, low: int, high: int, argument: str, what: str) -> None:
    """Refuse an out-of-range integer here, where the argument still has a name.

    An ``int`` Python is happy with may still not fit the width the log is
    written in, and the conversion at the boundary then raises
    ``OverflowError`` — "can't convert negative int to unsigned" — which is
    outside the declared taxonomy and names neither the argument nor the rule
    it broke. ``seq=-1`` is not an arithmetic accident; it is an argument that
    cannot mean anything, so it is a ``ValueError`` (the declared mapping for
    an invalid argument) that says which argument and what was allowed.

    Only the *range* is checked. A non-integer keeps falling through to the
    boundary, which already raises a clear ``TypeError`` naming the argument,
    and duplicating that here would be a second opinion, not a check.
    """
    if isinstance(value, int) and not low <= value <= high:
        raise ValueError(f"{argument} must be {what} in {low}..{high}, not {value}")

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


# --- durability, made observable ------------------------------------------
# Spec case 44 / audit finding 4: the entire suite — 115 tests including 40
# `kill -9` trials — passed with every durability call deleted, 94x faster.
# `kill -9` cannot test `fsync`, because the kernel completes an in-flight
# write and the page cache outlives the process. A rule no test can fail is not
# a rule, so the rule is held by observing the *call*.
#
# These two are part of the diagnostics hatch, not the episode interface: they
# exist for the M0 suite and case 24 fails if production code reaches for them.


def file_syncs() -> int:
    """File ``fsync``s completed **on the calling thread** since it started.

    Read it before an operation and after it; the difference is how many times
    that operation actually reached the disk. Thread-local and monotonic on
    purpose — a process-wide count under a parallel runner is either flaky or
    satisfied by some *other* thread's append, and the latter is exactly the
    mutant this exists to catch. Never read it from a worker thread.
    """
    return _log.file_syncs()


def dir_syncs() -> int:
    """Directory ``fsync``s completed on the calling thread since it started."""
    return _log.dir_syncs()


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
    would defeat the seam it is testing.

    This is the one hole in the seam, so it is the one most worth enforcing:
    case 24 fails if anything under ``python/`` other than this module reaches
    for ``.diagnostics``. The tests are the only permitted caller, and that is
    a build failure rather than a request.
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

    @property
    def repaired_lengths(self) -> int:
        """Damaged frame length fields recovery rewrote on this open.

        A frame whose length field is wrong but whose body still checksums and
        carries the expected sequence number is whole: only those four bytes
        are damaged, and the body itself says what they should have been. So
        recovery writes them back rather than refusing to open (which loses
        nothing but strands everything) or truncating (which destroys an
        episode that is provably intact). Non-zero means the file was damaged
        and is now correct — survivable, but worth knowing happened.
        """
        return self._log.repaired_lengths

    @property
    def discarded_tail_path(self) -> Optional[Path]:
        """Where a truncated tail's bytes were kept on this open, or ``None``.

        Recovery truncates a tail that holds no frame verifying end to end,
        because nothing in it was ever acknowledged. That judgement is about
        what the bytes *are*, not about what put them there, so the bytes
        themselves are copied beside the log rather than destroyed. ``None``
        means either nothing was truncated or what was truncated was only
        zeros, which is a crash artifact with nothing in it to read.
        """
        kept = self._log.discarded_tail_path
        return None if kept is None else Path(kept)

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
    @staticmethod
    def log_path_for(path: PathLike) -> Path:
        """Where ``MemoryStore.open(path)`` puts the log. A pure function.

        The rule, spec case 46: **the path decides, never the disk.**

            ``path`` ends in ``EPISODES_FILENAME``  ->  that is the log file
            anything else                           ->  ``path`` is the store
                                                        directory and the log is
                                                        ``path/EPISODES_FILENAME``

        It reads only the string a caller handed in, so the same argument names
        the same file forever — before the store exists, after it exists, and
        after it has been deleted. The two spellings converge: ``open(d)`` and
        ``open(d / EPISODES_FILENAME)`` are the same store, not two.

        This replaces an ``is_dir()`` test, which was order-dependent (audit
        finding 6). Under it the same argument meant ``p`` when ``p`` did not
        exist and ``p/episodes.log`` when someone had made the directory first
        — and the file form then permanently blocked the directory form, because
        a path already occupied by a file can never be made into one.
        """
        p = Path(os.fspath(path))
        if p.name == EPISODES_FILENAME:
            return p
        return p / EPISODES_FILENAME

    @classmethod
    def open(cls, path: PathLike) -> "MemoryStore":
        """Open (creating if needed) the store at ``path``.

        ``path`` may be the log **file** or the store **directory**; see
        :meth:`log_path_for` for the rule, which depends only on ``path``
        itself.

        When ``path`` is the **directory** form it is created if it is missing,
        so the log lands in the same place whether or not ``path`` already
        existed — that is what case 46 asserts. Exactly the one directory the
        caller named is created; a missing *parent* above it is still an error
        rather than a guess, and the **file** form creates nothing at all,
        because there the caller named a file and not the directory holding it
        (case 23).

        A **broken symlink** in the directory position raises ``ValueError``
        naming the path and its target, and creates nothing. ``exists()``
        follows symlinks, so a dangling link reads as "not there" while the
        name is in fact taken — the create behind it used to fail with a bare
        ``FileExistsError`` that named neither the link nor the reason.
        Resolving the link instead would put the store somewhere the caller
        never named, and silently repairing it would destroy someone's
        intent, so this fails closed and says what is wrong.
        """
        log_path = cls.log_path_for(path)
        store_dir = Path(os.fspath(path))
        if log_path != store_dir and not store_dir.exists():
            if store_dir.is_symlink():
                raise ValueError(
                    f"store path {str(store_dir)!r} is a symlink to "
                    f"{os.readlink(store_dir)!r}, which does not exist; "
                    "point it at a directory or remove it"
                )
            store_dir.mkdir(parents=False, exist_ok=True)
        return cls(_log.Log(log_path))

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

        A ``ts_micros`` outside the i64 the log stores is a ``ValueError``.
        """
        if ts_micros is not None:
            _in_range(ts_micros, _MIN_TS, _MAX_TS, "ts_micros", "a microsecond time")
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

        A ``seq`` that is not a sequence number at all — negative, or wider
        than the u64 they are written in — is a ``ValueError`` naming the
        argument, not an ``OverflowError`` from the boundary.
        """
        _in_range(seq, 0, _MAX_SEQ, "seq", "a sequence number")
        records = self._log.episodes_since(seq)
        return (Episode(*record) for record in records)

    # --- checkpoints ------------------------------------------------------
    def checkpoint(self, name: str) -> int:
        """An unset checkpoint reads as 0, never None."""
        return self._log.checkpoint(name)

    def set_checkpoint(self, name: str, seq: int) -> None:
        """Record that ``name`` has consumed up to ``seq``.

        ``seq > head()`` raises ``CheckpointAhead``; a ``seq`` that is not a
        sequence number at all — negative, or wider than u64 — is a
        ``ValueError`` naming the argument, on the same rule as
        :meth:`episodes_since`.
        """
        _in_range(seq, 0, _MAX_SEQ, "seq", "a sequence number")
        self._log.set_checkpoint(name, seq)

    def checkpoint_names(self) -> list[str]:
        return self._log.checkpoint_names()

    @property
    def checkpoints_reset(self) -> bool:
        """True when this open found an unreadable checkpoint sidecar.

        Spec case 42. Checkpoints are derived state, so a damaged sidecar must
        never be the reason acknowledged episodes become unreachable: it is set
        aside and every checkpoint reads 0, exactly as a *missing* sidecar
        always has. That is survivable but not silent — consumers will replay,
        and they are entitled to know why. So it is reported here rather than
        raised.
        """
        return self._log.checkpoints_reset

    @property
    def damaged_checkpoints_path(self) -> Optional[Path]:
        """Where an unreadable sidecar was moved on this open, or ``None``.

        The evidence is preserved, never deleted. ``None`` while
        :attr:`checkpoints_reset` is true means the move itself failed (a
        read-only directory, say) — the log still opened, which is the whole
        point of the rule.
        """
        moved = self._log.damaged_checkpoints_path
        return None if moved is None else Path(moved)

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
