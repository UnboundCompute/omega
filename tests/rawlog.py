"""Byte-level surgery on a log file, for the torn / corrupt cases.

The yellow and red cases are only reachable by deliberately mutating the file,
so this module restates the on-disk format **from M0_SPEC.md** rather than
importing it. That is on purpose twice over: the seam must not leak frame
vocabulary, and a helper that read its layout from the implementation would
make the format tests tautological. (``omega._log`` does export ``PREFIX_LEN``,
but only the seam may import it — spec case 24 — and the seam deliberately does
not re-export frame vocabulary. So the number is spelled out here, once, beside
the layout it belongs to.)

    Header, 32 bytes:
      magic     8 bytes   b"OMEGALOG"
      version   u32       = 2
      reserved  20 bytes  zero

    Record frame, repeated:
      body_len  u32       18..=MAX_BODY
      len_crc   u32       CRC-32/ISO-HDLC over the FOUR ENCODED BYTES of body_len
      crc32     u32       CRC-32/ISO-HDLC over the body bytes
      body:
        seq        u64
        ts_micros  i64
        key_len    u16
        key        key_len bytes, UTF-8
        payload    the rest

``len_crc`` is version 2's whole reason for existing. ``crc32`` covers the body,
so checking it requires already knowing how long the body is — which left
``body_len`` as the one field nothing could vouch for, and a flipped bit in it
read as a torn tail and silently truncated the file. ``len_crc`` breaks that
circularity, so this module must be able to write a frame whose ``len_crc`` is
right *or* deliberately wrong: cases 40 and 41 are exactly the two sides of that
check.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

MAGIC = b"OMEGALOG"
VERSION = 2
HEADER_LEN = 32
#: body_len + len_crc + crc32. Was 8 in version 1; the length's own checksum
#: is the four bytes that were added.
PREFIX_LEN = 12
#: Offsets inside the prefix, so no test has to count bytes by hand.
BODY_LEN_OFFSET = 0
LEN_CRC_OFFSET = 4
BODY_CRC_OFFSET = 8
FIXED_BODY_LEN = 18  # seq + ts_micros + key_len
MAX_BODY = 64 * 1024 * 1024
MAX_KEY = 512


def crc32(data: bytes) -> int:
    """CRC-32/ISO-HDLC — the same polynomial and convention as crc32fast."""
    return zlib.crc32(data) & 0xFFFFFFFF


def len_crc32(body_len: int) -> int:
    """The checksum a given ``body_len`` must carry.

    Over the four *encoded* bytes, so it can be checked against the raw prefix
    without decoding anything else.
    """
    return crc32(struct.pack("<I", body_len & 0xFFFFFFFF))


@dataclass(frozen=True)
class Frame:
    """One parsed frame, with where it sits in the file."""

    index: int  # 0-based position in the file
    offset: int  # file offset of body_len
    body_len: int
    len_crc: int
    crc: int
    body: bytes

    @property
    def end(self) -> int:
        return self.offset + PREFIX_LEN + self.body_len

    @property
    def seq(self) -> int:
        return struct.unpack_from("<Q", self.body, 0)[0]

    @property
    def len_crc_is_intact(self) -> bool:
        return self.len_crc == len_crc32(self.body_len)


def encode_frame(
    seq: int,
    ts_micros: int,
    key: str,
    payload: bytes,
    len_crc: Optional[int] = None,
    body_crc: Optional[int] = None,
) -> bytes:
    """One complete frame.

    ``len_crc`` and ``body_crc`` default to the correct values. Pass either
    explicitly to forge a frame whose checksum is deliberately wrong — case 41
    needs a length field that fails its own checksum without any other damage.
    """
    key_bytes = key.encode("utf-8")
    body = struct.pack("<QqH", seq, ts_micros, len(key_bytes)) + key_bytes + payload
    body_len = len(body)
    return (
        struct.pack(
            "<III",
            body_len,
            len_crc32(body_len) if len_crc is None else len_crc,
            crc32(body) if body_crc is None else body_crc,
        )
        + body
    )


def header_bytes() -> bytes:
    return MAGIC + struct.pack("<I", VERSION) + b"\x00" * 20


def read(path: Path) -> bytes:
    return Path(path).read_bytes()


def write(path: Path, data: bytes) -> None:
    Path(path).write_bytes(data)


def patch(path: Path, offset: int, data: bytes) -> None:
    """Overwrite ``len(data)`` bytes at ``offset``, leaving the length alone."""
    with open(path, "r+b") as fh:
        fh.seek(offset)
        fh.write(data)
        fh.flush()


def append_bytes(path: Path, data: bytes) -> None:
    with open(path, "ab") as fh:
        fh.write(data)
        fh.flush()


def truncate(path: Path, size: int) -> None:
    with open(path, "r+b") as fh:
        fh.truncate(size)


def size(path: Path) -> int:
    return Path(path).stat().st_size


def scan(path: Path) -> list[Frame]:
    """An independent full scan: every frame that parses cleanly, in order.

    Stops at the first thing that is not a well-formed frame, which is exactly
    what a fresh scan of a recovered file should find and nothing more. The
    length's own checksum is checked *before* the length is used for anything,
    which is the ordering version 2 exists to impose.
    """
    data = read(path)
    frames: list[Frame] = []
    offset = HEADER_LEN
    index = 0
    while offset + PREFIX_LEN <= len(data):
        body_len, len_crc, crc = struct.unpack_from("<III", data, offset)
        if len_crc != len_crc32(body_len):
            break
        if body_len < FIXED_BODY_LEN or body_len > MAX_BODY:
            break
        end = offset + PREFIX_LEN + body_len
        if end > len(data):
            break
        body = data[offset + PREFIX_LEN : end]
        if crc32(body) != crc:
            break
        frames.append(Frame(index, offset, body_len, len_crc, crc, body))
        offset = end
        index += 1
    return frames


def frame_offsets(path: Path) -> list[int]:
    """The offsets a fresh scan finds — what the log's index must agree with."""
    return [f.offset for f in scan(path)]


def corrupt_crc(path: Path, index: int) -> None:
    """Flip a bit in frame ``index``'s **body** CRC, leaving everything else."""
    frame = scan(path)[index]
    bad = (frame.crc ^ 0x00000001) & 0xFFFFFFFF
    patch(path, frame.offset + BODY_CRC_OFFSET, struct.pack("<I", bad))


def corrupt_len_crc(path: Path, index: int) -> None:
    """Flip a bit in frame ``index``'s **length** CRC, leaving the length alone.

    The length field itself is untouched and still names a real frame, so the
    only thing wrong with the file is that the length can no longer vouch for
    itself. Case 41: nothing the length says may be acted on.
    """
    frame = scan(path)[index]
    bad = (frame.len_crc ^ 0x00000001) & 0xFFFFFFFF
    patch(path, frame.offset + LEN_CRC_OFFSET, struct.pack("<I", bad))


def set_body_len(path: Path, index: int, value: int, fix_len_crc: bool = True) -> None:
    """Rewrite frame ``index``'s ``body_len`` **and, by default, its len_crc**.

    Fixing the length's checksum is what makes the damaged length *believable*,
    which is the only way to reach the branches that judge the length on its
    own merits — the implausible-value branch (case 15) and the
    plausible-but-too-large branch (case 40). Leaving the checksum stale
    (``fix_len_crc=False``) instead exercises case 41's branch, where the length
    fails its own checksum and nothing it says may be used at all.
    """
    frame = scan(path)[index]
    encoded = struct.pack("<I", value & 0xFFFFFFFF)
    patch(path, frame.offset + BODY_LEN_OFFSET, encoded)
    if fix_len_crc:
        patch(path, frame.offset + LEN_CRC_OFFSET, struct.pack("<I", len_crc32(value)))


def set_seq(path: Path, index: int, value: int) -> None:
    """Rewrite a frame's seq **and fix its body CRC**.

    Without the CRC fix the scan would trip on the checksum first and report
    ``CorruptFrame``, so the sequence-break case would never be exercised. The
    length field is untouched, so its own checksum still holds.
    """
    frame = scan(path)[index]
    body = bytearray(frame.body)
    struct.pack_into("<Q", body, 0, value)
    body = bytes(body)
    patch(path, frame.offset + BODY_CRC_OFFSET, struct.pack("<I", crc32(body)))
    patch(path, frame.offset + PREFIX_LEN, body)
