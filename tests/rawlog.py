"""Byte-level surgery on a log file, for the torn / corrupt cases.

The yellow and red cases are only reachable by deliberately mutating the file,
so this module restates the on-disk format **from M0_SPEC.md** rather than
importing it. That is on purpose twice over: the seam must not leak frame
vocabulary, and a helper that read its layout from the implementation would
make the format tests tautological.

    Header, 32 bytes:
      magic     8 bytes   b"OMEGALOG"
      version   u32       = 1
      reserved  20 bytes  zero

    Record frame, repeated:
      body_len  u32       18..=MAX_BODY
      crc32     u32       CRC-32/ISO-HDLC over the body bytes
      body:
        seq        u64
        ts_micros  i64
        key_len    u16
        key        key_len bytes, UTF-8
        payload    the rest
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

MAGIC = b"OMEGALOG"
VERSION = 1
HEADER_LEN = 32
PREFIX_LEN = 8  # body_len + crc32
FIXED_BODY_LEN = 18  # seq + ts_micros + key_len
MAX_BODY = 64 * 1024 * 1024
MAX_KEY = 512


def crc32(data: bytes) -> int:
    """CRC-32/ISO-HDLC — the same polynomial and convention as crc32fast."""
    return zlib.crc32(data) & 0xFFFFFFFF


@dataclass(frozen=True)
class Frame:
    """One parsed frame, with where it sits in the file."""

    index: int  # 0-based position in the file
    offset: int  # file offset of body_len
    body_len: int
    crc: int
    body: bytes

    @property
    def end(self) -> int:
        return self.offset + PREFIX_LEN + self.body_len

    @property
    def seq(self) -> int:
        return struct.unpack_from("<Q", self.body, 0)[0]


def encode_frame(seq: int, ts_micros: int, key: str, payload: bytes) -> bytes:
    key_bytes = key.encode("utf-8")
    body = struct.pack("<QqH", seq, ts_micros, len(key_bytes)) + key_bytes + payload
    return struct.pack("<II", len(body), crc32(body)) + body


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
    what a fresh scan of a recovered file should find and nothing more.
    """
    data = read(path)
    frames: list[Frame] = []
    offset = HEADER_LEN
    index = 0
    while offset + PREFIX_LEN <= len(data):
        body_len, crc = struct.unpack_from("<II", data, offset)
        if body_len < FIXED_BODY_LEN or body_len > MAX_BODY:
            break
        end = offset + PREFIX_LEN + body_len
        if end > len(data):
            break
        body = data[offset + PREFIX_LEN : end]
        if crc32(body) != crc:
            break
        frames.append(Frame(index, offset, body_len, crc, body))
        offset = end
        index += 1
    return frames


def frame_offsets(path: Path) -> list[int]:
    """The offsets a fresh scan finds — what the log's index must agree with."""
    return [f.offset for f in scan(path)]


def corrupt_crc(path: Path, index: int) -> None:
    """Flip a bit in frame ``index``'s CRC field, leaving everything else."""
    frame = scan(path)[index]
    bad = (frame.crc ^ 0x00000001) & 0xFFFFFFFF
    patch(path, frame.offset + 4, struct.pack("<I", bad))


def set_body_len(path: Path, index: int, value: int) -> None:
    frame = scan(path)[index]
    patch(path, frame.offset, struct.pack("<I", value))


def set_seq(path: Path, index: int, value: int) -> None:
    """Rewrite a frame's seq **and fix its CRC**.

    Without the CRC fix the scan would trip on the checksum first and report
    ``CorruptFrame``, so the sequence-break case would never be exercised.
    """
    frame = scan(path)[index]
    body = bytearray(frame.body)
    struct.pack_into("<Q", body, 0, value)
    body = bytes(body)
    patch(path, frame.offset + 4, struct.pack("<I", crc32(body)))
    patch(path, frame.offset + PREFIX_LEN, body)
