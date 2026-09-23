"""Yellow — torn and zero-filled tails. M0_SPEC.md cases 25-30, 37-39.

These paths are **only** reachable by deliberately mutating the file: killing a
process does not tear a write (see case 32's docstring). So every case here does
its damage with plain file I/O and then asks the log to recover from it.

The rule being tested, in one line: *incomplete at EOF is a torn tail and gets
truncated; complete but invalid with data after it is corruption and refuses to
open.*
"""

from __future__ import annotations

from pathlib import Path

import pytest
import rawlog

from omega.memory import CorruptFrame, MemoryStore
from conftest import seed


def _assert_recovers_to(store_dir: Path, payloads: list[bytes]) -> None:
    """Reopen and assert the exact surviving episodes, not just a count."""
    with MemoryStore.open(store_dir) as s:
        assert s.head() == len(payloads)
        episodes = list(s.episodes_since(0))
    assert len(episodes) == len(payloads)
    assert [e.seq for e in episodes] == list(range(1, len(payloads) + 1))
    assert [e.payload for e in episodes] == payloads


# --- 25-27: torn tails ----------------------------------------------------


def test_case_25_torn_tail_mid_payload(store_dir: Path, log_path: Path) -> None:
    """Case 25 — a tail torn in the middle of a payload truncates to the last
    good seq; nothing before it is lost."""
    payloads = seed(store_dir, 5)
    last = rawlog.scan(log_path)[-1]
    # Cut halfway through the final frame's body.
    rawlog.truncate(log_path, last.offset + rawlog.PREFIX_LEN + last.body_len // 2)

    with MemoryStore.open(store_dir) as s:
        assert s.head() == 4
        assert s.diagnostics.recovered_bytes > 0
        got = [e.payload for e in s.episodes_since(0)]
    assert got == payloads[:4]
    # The truncation is durable: the file now ends at the last good frame.
    assert rawlog.size(log_path) == last.offset


def test_case_26_torn_tail_at_a_frame_boundary(store_dir: Path, log_path: Path) -> None:
    """Case 26 — the length prefix landed but the body did not."""
    payloads = seed(store_dir, 4)
    last = rawlog.scan(log_path)[-1]
    rawlog.truncate(log_path, last.offset + rawlog.PREFIX_LEN)

    _assert_recovers_to(store_dir, payloads[:3])
    assert rawlog.size(log_path) == last.offset


def test_case_26_torn_tail_one_byte_into_the_body(
    store_dir: Path, log_path: Path
) -> None:
    """Case 26 — and one byte into the body, the other side of the boundary."""
    payloads = seed(store_dir, 4)
    last = rawlog.scan(log_path)[-1]
    rawlog.truncate(log_path, last.offset + rawlog.PREFIX_LEN + 1)

    _assert_recovers_to(store_dir, payloads[:3])


@pytest.mark.parametrize("n_bytes", [1, 2, 7, rawlog.PREFIX_LEN - 1])
def test_case_27_torn_tail_shorter_than_a_frame_prefix(
    store_dir: Path, log_path: Path, n_bytes: int
) -> None:
    """Case 27 — a partial tail shorter than the frame prefix is a torn tail.

    The prefix became 12 bytes in version 2 (``body_len``, ``len_crc``,
    ``crc32``), and this case kept testing 1, 2 and 7 while its prose still
    said 8. It therefore stopped at the *old* boundary and never reached the
    last length that is short — ``PREFIX_LEN - 1``, the one byte away from a
    readable length field. It is named here rather than written as a literal
    so the parametrization cannot go stale behind the format again.

    Non-zero bytes, so this is not the zero-fill path (case 39 is).
    """
    payloads = seed(store_dir, 3)
    clean_size = rawlog.size(log_path)
    rawlog.append_bytes(log_path, bytes([0xAB]) * n_bytes)

    _assert_recovers_to(store_dir, payloads)
    assert rawlog.size(log_path) == clean_size


# --- 28: torn header ------------------------------------------------------


@pytest.mark.parametrize("size", [0, 1, 8, 16, 31])
def test_case_28_torn_header(store_dir: Path, log_path: Path, size: int) -> None:
    """Case 28 — a file of 0 or 1..31 bytes is a torn header: recover as an
    empty log, no error. No record can exist yet, so nothing can be lost."""
    with MemoryStore.open(store_dir):
        pass  # creates the header
    rawlog.truncate(log_path, size)
    assert rawlog.size(log_path) == size

    with MemoryStore.open(store_dir) as s:
        assert s.head() == 0
        assert list(s.episodes_since(0)) == []
        assert s.append_episode(b"after the torn header", "", 1) == 1
        assert [e.payload for e in s.episodes_since(0)] == [b"after the torn header"]

    assert rawlog.size(log_path) > rawlog.HEADER_LEN
    assert rawlog.read(log_path)[:8] == rawlog.MAGIC


def test_case_28_missing_file_is_an_empty_log(store_dir: Path, log_path: Path) -> None:
    """Case 28 — and a missing file is created as an empty log, not an error."""
    assert not log_path.exists()
    with MemoryStore.open(store_dir) as s:
        assert s.head() == 0
        assert list(s.episodes_since(0)) == []
    assert log_path.exists()
    assert rawlog.size(log_path) == rawlog.HEADER_LEN


# --- 29-30: after recovery ------------------------------------------------


def test_case_29_append_after_recovery_continues_the_sequence(
    store_dir: Path, log_path: Path
) -> None:
    """Case 29 — after a torn-tail recovery the next append is head+1, no gap.

    This is the one that silently breaks if recovery forgets to reset the
    writer: a resumed log that skips a sequence number is a log whose contiguity
    guarantee is gone.
    """
    payloads = seed(store_dir, 5)
    last = rawlog.scan(log_path)[-1]
    rawlog.truncate(log_path, last.offset + rawlog.PREFIX_LEN + 3)

    with MemoryStore.open(store_dir) as s:
        assert s.head() == 4
        assert s.append_episode(b"resumed-5", "", 5) == 5
        assert s.append_episode(b"resumed-6", "", 6) == 6
        episodes = list(s.episodes_since(0))

    assert [e.seq for e in episodes] == [1, 2, 3, 4, 5, 6]
    assert [e.payload for e in episodes] == payloads[:4] + [b"resumed-5", b"resumed-6"]

    # And the resumed records survive their own reopen.
    _assert_recovers_to(store_dir, payloads[:4] + [b"resumed-5", b"resumed-6"])


def test_case_30_recovery_is_idempotent(store_dir: Path, log_path: Path) -> None:
    """Case 30 — recover, "crash", recover again → byte-identical, same state."""
    payloads = seed(store_dir, 6)
    last = rawlog.scan(log_path)[-1]
    rawlog.truncate(log_path, last.offset + rawlog.PREFIX_LEN + 2)

    with MemoryStore.open(store_dir) as s:
        first_head = s.head()
        first_offsets = s.diagnostics.offsets()
        first_episodes = list(s.episodes_since(0))
        assert s.diagnostics.recovered_bytes > 0
    after_first = rawlog.read(log_path)

    for attempt in range(3):
        with MemoryStore.open(store_dir) as s:
            assert s.head() == first_head, f"attempt {attempt}"
            assert s.diagnostics.offsets() == first_offsets
            assert list(s.episodes_since(0)) == first_episodes
            # Nothing left to truncate the second time round.
            assert s.diagnostics.recovered_bytes == 0
        assert rawlog.read(log_path) == after_first, f"attempt {attempt}"

    assert first_head == 5
    assert [e.payload for e in first_episodes] == payloads[:5]


# --- 37-39: zero-filled tails ---------------------------------------------


@pytest.mark.parametrize("zeros", [rawlog.PREFIX_LEN, 18, 64, 4096, 65536])
def test_case_37_zero_filled_tail(store_dir: Path, log_path: Path, zeros: int) -> None:
    """Case 37 — a run of NULs to EOF is a crash artifact, not damage.

    The shortest run here is ``PREFIX_LEN``: the first length at which the
    zeros are a readable (and implausible) ``body_len`` rather than a run too
    short to hold one, which is case 39's path. It used to be the literal 8,
    which was that boundary in version 1 and is on case 39's side now.

    Real power loss or a kernel panic can persist a write's *size extension*
    without its data pages, so the tail reads back as zeros. The log must open,
    head() must be N, all N episodes must be readable, and the zeros must be
    truncated away. The spec originally got this wrong and the log refused to
    open, making every acknowledged episode unreachable.
    """
    payloads = seed(store_dir, 5)
    clean_size = rawlog.size(log_path)
    rawlog.append_bytes(log_path, b"\x00" * zeros)
    assert rawlog.size(log_path) == clean_size + zeros

    with MemoryStore.open(store_dir) as s:
        assert s.head() == 5
        episodes = list(s.episodes_since(0))
        assert s.diagnostics.recovered_bytes == zeros
    assert len(episodes) == 5
    assert [e.payload for e in episodes] == payloads
    assert rawlog.size(log_path) == clean_size, "the zeros must be truncated"


def test_case_37_zero_filled_tail_on_an_empty_log(
    store_dir: Path, log_path: Path
) -> None:
    """Case 37 — a zero tail on a log that only ever had a header also opens."""
    with MemoryStore.open(store_dir):
        pass
    rawlog.append_bytes(log_path, b"\x00" * 1024)

    with MemoryStore.open(store_dir) as s:
        assert s.head() == 0
        assert list(s.episodes_since(0)) == []
        assert s.append_episode(b"after the zeros", "", 1) == 1
        assert [e.payload for e in s.episodes_since(0)] == [b"after the zeros"]


def test_case_37_and_29_append_after_a_zero_tail_continues_the_sequence(
    store_dir: Path, log_path: Path
) -> None:
    """Cases 37 + 29 — after the zeros are truncated, the next append is head+1."""
    payloads = seed(store_dir, 4)
    rawlog.append_bytes(log_path, b"\x00" * 2048)

    with MemoryStore.open(store_dir) as s:
        assert s.head() == 4
        assert s.append_episode(b"resumed-5", "", 5) == 5
    _assert_recovers_to(store_dir, payloads + [b"resumed-5"])


def test_case_38_zeros_with_real_frames_after_them_is_corruption(
    store_dir: Path, log_path: Path
) -> None:
    """Case 38 — zeros only mean "crash artifact" when they run to EOF.

    Zeros in the middle, with real frame bytes after them, are damage: data
    following proves that region was durable once, and truncating there would
    silently destroy the episodes beyond it.
    """
    seed(store_dir, 3)
    rawlog.append_bytes(log_path, b"\x00" * 64)
    rawlog.append_bytes(log_path, rawlog.encode_frame(4, 4_000, "k4", b"after-zeros"))
    size_before = rawlog.size(log_path)

    with pytest.raises(CorruptFrame):
        MemoryStore.open(store_dir)

    assert rawlog.size(log_path) == size_before, "a failed open must not truncate"


@pytest.mark.parametrize("zeros", [8, rawlog.PREFIX_LEN])
def test_case_38_a_single_zero_word_followed_by_data(
    store_dir: Path, log_path: Path, zeros: int
) -> None:
    """Case 38 — even a short run of zeros followed by real data is corruption.

    Both sides of the prefix boundary: a run too short to be a ``body_len`` at
    all, and a whole prefix of zeros. Neither is a crash artifact, because
    something durable follows them.
    """
    seed(store_dir, 3)
    rawlog.append_bytes(log_path, b"\x00" * zeros)
    rawlog.append_bytes(log_path, rawlog.encode_frame(4, 4_000, "", b"tail"))

    with pytest.raises(CorruptFrame):
        MemoryStore.open(store_dir)


@pytest.mark.parametrize("zeros", [1, 2, 7, rawlog.PREFIX_LEN - 1])
def test_case_39_short_zero_run_is_a_torn_tail(
    store_dir: Path, log_path: Path, zeros: int
) -> None:
    """Case 39 — a single NUL, or a run shorter than a frame prefix, recovers
    as a torn tail with no loss. ``PREFIX_LEN - 1`` is the last such length and
    is named, not spelled 7, so it follows the format rather than trailing it.
    """
    payloads = seed(store_dir, 4)
    clean_size = rawlog.size(log_path)
    rawlog.append_bytes(log_path, b"\x00" * zeros)

    with MemoryStore.open(store_dir) as s:
        assert s.head() == 4
        episodes = list(s.episodes_since(0))
        assert s.diagnostics.recovered_bytes == zeros
    assert [e.payload for e in episodes] == payloads
    assert rawlog.size(log_path) == clean_size
