"""The content-addressed blob store beside the log — DL-027.

A context item names what was attached; it never carries it. ``{id, kind,
title}`` had nowhere to put a screenshot's pixels, so the tray could stage a
capture and omega could only learn that something called *"Area capture"*
existed. This is where those bytes go, and the episode carries a *reference* —
``{blob, mime, bytes}`` — instead of the content.

**Beside the log, never inside it.** DL-017 makes memory a graph *derived from*
the append-only log: a schema change is a re-derive, not a migration, so every
byte an episode carries is re-read on every rebuild for the life of the store.
Base64ing a 240 KB screenshot into an episode buys a simpler protocol once and
pays for it on every derivation forever. The log's value is that it is cheap to
scan, and attachments are the one payload that would destroy that.

**The reference is the content.** A blob is stored under the digest of its own
bytes. A bare path would have been cheaper to build and would have broken the
property the log exists for: a file omega does not control can move, be edited
or be deleted, so replaying the same log later could yield different content, or
none, with nothing saying so. Content addressing makes the reference immutable
by construction — a digest either resolves to exactly the bytes that were
attached or does not resolve at all, and the failure is loud.

**Not part of the log, and it must stay that way.** This module imports nothing
from ``omega.memory`` and the seam knows nothing about it. They are siblings
under one store directory: the log owns episodes, this owns bytes, neither can
damage the other's files, and a future change to either is not a change to both.

Deduplication is not a feature in here; it is what content addressing *is*. The
same screenshot attached twice is one file because it has one name. There is no
index — the filename is the key and the filesystem is the lookup.

**Deliberately absent** (DL-027 defers both, and neither blocks the seam):
garbage collection of unreferenced blobs, which needs the graph to know what is
referenced, and any size cap. What is *not* deferred is refusing a path that is
not a regular file — without that, ``attach /dev/zero`` copies until the disk
fills, which is the one unbounded case a missing size cap actually opens.
"""

from __future__ import annotations

import hashlib
import mimetypes
import os
import re
import stat
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Union

__all__ = [
    "BLOBS_DIRNAME",
    "DIGEST_ALGORITHM",
    "DIGEST_PREFIX",
    "DIGEST_RE",
    "DEFAULT_MIME",
    "FANOUT",
    "BlobError",
    "BadDigest",
    "NotARegularFile",
    "IngestFailed",
    "BlobRef",
    "BlobStore",
    "is_digest",
    "mime_for",
]

PathLike = Union[str, "os.PathLike[str]"]

#: The directory the blobs live in, under the store directory the log is in.
BLOBS_DIRNAME = "blobs"

DIGEST_ALGORITHM = "sha256"

#: The digest is *prefixed with its algorithm on the wire and in the episode*.
#: A bare hex string would be a guess about which function produced it, and the
#: day a second one exists every stored reference would be ambiguous — in a log
#: that is append-only, so unfixable. The prefix costs seven bytes.
DIGEST_PREFIX = f"{DIGEST_ALGORITHM}:"

#: What a reference must look like anywhere it appears. Lowercase hex only: the
#: same bytes must have exactly one spelling, or ``has()`` and the filesystem
#: would disagree on a case-insensitive one.
DIGEST_RE = re.compile(r"\Asha256:[0-9a-f]{64}\Z")

#: How many hex characters name the fanout directory. Two gives 256 of them,
#: which keeps any one directory small enough that listing it stays cheap. It is
#: layout, not identity — the digest is the whole name, split for the disk's
#: sake, which is why nothing outside this module ever sees the split.
FANOUT = 2

#: What an extension nobody recognises means. Not a guess and not an error:
#: omega can still store, reference and hand back bytes it cannot name.
DEFAULT_MIME = "application/octet-stream"

#: Streaming granularity. Never the whole file — a blob store whose ingest peaks
#: at the file's size is one screenshot away from being the reason omega died.
CHUNK = 1 << 20

#: Prefix for the in-progress file. It cannot be a digest and so can never be
#: mistaken for a blob, which is what makes an interrupted ``put`` invisible
#: rather than half-present.
_TMP_PREFIX = ".ingest-"

#: ``O_NONBLOCK`` is the load-bearing flag. Opening a FIFO for reading *blocks
#: until a writer appears*, so a check that opened first and asked what it had
#: second would hang forever on exactly the path it exists to refuse. With it,
#: the open returns immediately and ``fstat`` on the descriptor gets to answer.
#: On a regular file it means nothing at all.
_OPEN_FLAGS = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)


class BlobError(RuntimeError):
    """The blob store could not do what it was asked."""


class BadDigest(BlobError):
    """A string was used as a digest and is not one.

    Refused rather than normalised. A digest that has been cleaned up on the way
    in is no longer a claim about the bytes — it is a claim about the bytes
    *after we changed the question*, and content addressing is worth exactly as
    much as that claim.
    """


class NotARegularFile(BlobError):
    """The path does not name a regular file that exists.

    One error for missing, directory, FIFO, socket and device on purpose: the
    caller's next move is the same in every case, and the message says which it
    was. The check is what bounds ingest without a size cap — a character device
    has no end, so streaming one is not a large copy but an unending one.
    """


class IngestFailed(BlobError):
    """The bytes could not be read in, or could not be made durable.

    Distinct from :class:`NotARegularFile` because it says nothing is wrong with
    what the caller asked for — the source was a real file and the store still
    could not take it. Retrying is reasonable here and pointless there.
    """


def is_digest(value: object) -> bool:
    """Is ``value`` a well-formed reference? Total, and never raises.

    Used by :mod:`omega.episodes` to validate a context item's ``blob`` without
    the payload codec growing its own second spelling of the format.
    """
    return isinstance(value, str) and DIGEST_RE.match(value) is not None


def mime_for(path: PathLike) -> str:
    """The media type implied by the filename, or :data:`DEFAULT_MIME`.

    Guessed from the *name*, never sniffed from the content. Sniffing would make
    the stored mime depend on how good this build's detector is, and the episode
    is permanent — so it records what the sender's filename claimed, which is a
    fact about the attachment rather than an inference that can go stale.
    """
    guessed, _encoding = mimetypes.guess_type(os.fspath(path))
    return guessed or DEFAULT_MIME


@dataclass(frozen=True)
class BlobRef:
    """What an episode carries in place of the bytes.

    ``bytes`` is counted off the stream that was hashed, not read from
    ``stat()``. The two can differ if the file changes mid-ingest, and only one
    of them describes the content the digest names.
    """

    digest: str
    mime: str
    bytes: int


class BlobStore:
    """Bytes go in and are named by themselves. That is all of it."""

    __slots__ = ("_root",)

    def __init__(self, root: Path) -> None:
        self._root = root

    # --- opening ----------------------------------------------------------

    @staticmethod
    def root_for(store_dir: PathLike) -> Path:
        """Where the blobs live for a given store. A pure function.

        Reads only the string it was handed, on the same rule as
        ``MemoryStore.log_path_for``: the path decides, never the disk, so the
        same argument names the same directory before, during and after the
        store exists.
        """
        return Path(os.fspath(store_dir)) / BLOBS_DIRNAME

    @classmethod
    def open(cls, store_dir: PathLike) -> "BlobStore":
        """Open (creating if needed) the blob store beside ``store_dir``.

        Parents are created, which is the one place this deliberately differs
        from the log's opener. The log refuses a missing parent because the
        caller named a store that is not there; here the parent *is* that store
        directory, and whether it already exists depends only on which of the
        two opens ran first. Making the order matter would be a bug that appears
        in production and not in a test, or the reverse.
        """
        root = cls.root_for(store_dir)
        try:
            root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise IngestFailed(
                f"cannot open the blob store at {str(root)!r}: {exc}"
            ) from exc
        return cls(root)

    @property
    def root(self) -> Path:
        return self._root

    # --- writing ----------------------------------------------------------

    def put(self, path: PathLike) -> BlobRef:
        """Store the contents of ``path`` and return its reference.

        Streamed, so ingest costs a chunk of memory and not a file's worth, and
        hashed *while* streaming, so the bytes are read exactly once — reading
        twice would also mean the digest and the stored copy could describe
        different content if the source changed in between.

        Storing content that is already stored is a **no-op that succeeds** and
        returns the same reference. That is not a cache and there is nothing to
        invalidate: the file is named by its content, so "already there" and
        "identical" are the same statement.
        """
        source = Path(os.fspath(path))
        mime = mime_for(source)
        tmp = self._root / f"{_TMP_PREFIX}{uuid.uuid4().hex}"

        fd = self._open_regular(source)
        try:
            digest, size = self._drain_to(fd, tmp, source)
            final = self.path_for(digest)
            # The digest is not known until the last byte has been read, so the
            # temp file cannot be written into its final fanout directory. It is
            # written into the blob root instead — same store, same filesystem,
            # so the rename below is still the atomic single-step publish.
            if not final.exists():
                try:
                    self._install(tmp, final)
                except OSError as exc:
                    raise IngestFailed(
                        f"cannot store {str(source)!r} as {digest}: {exc}"
                    ) from exc
        finally:
            _discard(tmp)

        return BlobRef(digest=digest, mime=mime, bytes=size)

    def _open_regular(self, source: Path) -> int:
        """A descriptor on ``source``, or :class:`NotARegularFile`.

        The kind is decided from the **open descriptor**, not from a ``stat`` of
        the path. A stat-then-open would leave a window in which the thing
        checked and the thing read are not the same object, and the one case
        that matters — a device or FIFO swapped in — is the case the check
        exists for. Here there is no window: what was opened is what is asked
        about, and if it is not a regular file it is closed unread.
        """
        try:
            fd = os.open(source, _OPEN_FLAGS)
        except OSError as exc:
            raise NotARegularFile(
                f"cannot attach {str(source)!r}: {exc.strerror or exc}"
            ) from exc
        try:
            mode = os.fstat(fd).st_mode
        except OSError as exc:  # pragma: no cover - fstat on a live fd
            os.close(fd)
            raise IngestFailed(f"cannot attach {str(source)!r}: {exc}") from exc
        if not stat.S_ISREG(mode):
            os.close(fd)
            raise NotARegularFile(
                f"cannot attach {str(source)!r}: it is a {_kind_of(mode)}, "
                f"not a regular file"
            )
        return fd

    def _drain_to(self, fd: int, tmp: Path, source: Path) -> tuple[str, int]:
        """Copy the descriptor into ``tmp``, hashing as it goes. Returns both.

        ``tmp`` is opened ``x`` so a name collision is a failure rather than a
        silent overwrite — the name is random, so a collision means something is
        wrong with an assumption, and that is worth hearing about.
        """
        hasher = hashlib.sha256()
        size = 0
        try:
            with os.fdopen(fd, "rb") as src, open(tmp, "xb") as dst:
                while True:
                    chunk = src.read(CHUNK)
                    if not chunk:
                        break
                    hasher.update(chunk)
                    size += len(chunk)
                    dst.write(chunk)
                dst.flush()
                # Data first, then the name (in _install). The reverse order can
                # publish a complete name over incomplete content.
                os.fsync(dst.fileno())
        except OSError as exc:
            raise IngestFailed(f"cannot read {str(source)!r}: {exc}") from exc
        return DIGEST_PREFIX + hasher.hexdigest(), size

    def _install(self, tmp: Path, final: Path) -> None:
        """Publish the finished temp file under its digest.

        write-temp, fsync, rename, fsync-dir — the order the log's checkpoint
        sidecar uses (``src/checkpoint.rs::persist``), for the same reasons.
        ``os.replace`` is the only step that makes a whole file appear under a
        name at once, so nothing under a digest is ever partial. The directory
        fsync is what makes the *name* durable: without it a crash can lose the
        rename even though the data it published was already on the disk, and
        the blob would then be missing while the episode referencing it is not.
        """
        parent = final.parent
        fresh = not parent.exists()
        parent.mkdir(parents=True, exist_ok=True)
        if fresh:
            # The fanout directory is itself a new entry in the blob root, and
            # an entry nobody synced can vanish with everything under it.
            _sync_dir(self._root)
        os.replace(tmp, final)
        _sync_dir(parent)

    # --- reading ----------------------------------------------------------

    def path_for(self, digest: str) -> Path:
        """Where ``digest`` is stored, whether or not it is stored yet.

        Pure, like the log's ``log_path_for``: it reads the digest and nothing
        else. No extension — the media type lives in the episode, where it is a
        fact about the attachment rather than a property of a filename that
        anything on the box could rename.
        """
        if not is_digest(digest):
            raise BadDigest(
                f"{digest!r} is not a blob reference; expected "
                f"{DIGEST_PREFIX!r} followed by 64 lowercase hex digits"
            )
        hex_digits = digest[len(DIGEST_PREFIX):]
        return self._root / hex_digits[:FANOUT] / hex_digits[FANOUT:]

    def has(self, digest: str) -> bool:
        """Is this reference resolvable *right now*?

        ``is_file`` rather than ``exists``: a directory sitting where a blob
        belongs is not a blob, and answering yes for it would turn a storage
        fault into a read failure somewhere far away.
        """
        return self.path_for(digest).is_file()

    def __repr__(self) -> str:
        return f"<BlobStore {str(self._root)!r}>"


# --- filesystem helpers -----------------------------------------------------


def _sync_dir(path: Path) -> None:
    """``fsync`` a directory, so a newly created or renamed entry in it lasts.

    The same call ``src/lib.rs::sync_dir`` makes on the log's side. Unlike the
    log's, this one is not counted: the log counts its syncs because M0 has a
    spec case asserting they happen (a rule no test can fail is not a rule), and
    that hatch belongs to the log's own diagnostics, not to a sibling store.
    """
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _discard(tmp: Path) -> None:
    """Remove an in-progress file. Never raises — it runs on the failure path.

    After a successful ``os.replace`` there is nothing here to remove, and that
    is the ordinary case rather than an error.
    """
    try:
        tmp.unlink()
    except OSError:
        pass


def _kind_of(mode: int) -> str:
    """What the caller actually pointed at, in words, for the refusal message."""
    for test, name in (
        (stat.S_ISDIR, "directory"),
        (stat.S_ISFIFO, "FIFO"),
        (stat.S_ISSOCK, "socket"),
        (stat.S_ISCHR, "character device"),
        (stat.S_ISBLK, "block device"),
    ):
        if test(mode):
            return name
    return "special file"  # pragma: no cover - the list above is exhaustive
