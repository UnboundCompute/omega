"""Red — it must fail, loudly and without damage. M0_SPEC.md cases 13-23.

Every assertion here names the *specific* exception type. A bare ``Exception``
would pass on the wrong failure and is therefore not a check.
"""

from __future__ import annotations

import os
import struct
import subprocess
import sys
from pathlib import Path

import pytest
import rawlog

from omega.memory import (
    AlreadyLocked,
    CheckpointAhead,
    CorruptFrame,
    MAX_BODY,
    MAX_KEY,
    MemoryStore,
    NotAnOmegaLog,
    SequenceBreak,
    TooLarge,
    UnsupportedVersion,
    WriteKeyConflict,
)
from conftest import seed


def test_case_13_middle_crc_is_corruption_not_a_torn_tail(
    store_dir: Path, log_path: Path
) -> None:
    """Case 13 — CRC broken in a middle frame → CorruptFrame, file untouched.

    Data after the frame proves it was fully durable, so truncating would
    silently destroy real episodes. The file must come back byte-for-byte the
    size it was, and repairing the CRC must bring all five episodes back —
    which is the actual proof that nothing was discarded.
    """
    payloads = seed(store_dir, 5)
    before = rawlog.read(log_path)

    rawlog.corrupt_crc(log_path, 2)  # the middle frame of five
    broken = rawlog.read(log_path)

    with pytest.raises(CorruptFrame):
        MemoryStore.open(store_dir)

    assert rawlog.read(log_path) == broken, "a failed open must not truncate"
    assert len(broken) == len(before)

    # Repair and confirm the later episodes were never thrown away.
    rawlog.write(log_path, before)
    with MemoryStore.open(store_dir) as s:
        episodes = list(s.episodes_since(0))
    assert len(episodes) == 5
    assert [e.payload for e in episodes] == payloads


def test_case_14_bad_magic(store_dir: Path, log_path: Path) -> None:
    """Case 14 — bad magic → NotAnOmegaLog, and the file is not rewritten."""
    seed(store_dir, 3)
    before = rawlog.read(log_path)

    rawlog.patch(log_path, 0, b"NOTALOG!")
    with pytest.raises(NotAnOmegaLog):
        MemoryStore.open(store_dir)
    assert len(rawlog.read(log_path)) == len(before)

    rawlog.write(log_path, before)
    with MemoryStore.open(store_dir) as s:
        assert s.head() == 3


def test_case_14_unknown_version(store_dir: Path, log_path: Path) -> None:
    """Case 14 — an unknown format version → UnsupportedVersion, refuse to open."""
    seed(store_dir, 3)
    before = rawlog.read(log_path)

    rawlog.patch(log_path, 8, struct.pack("<I", 99))
    with pytest.raises(UnsupportedVersion) as exc:
        MemoryStore.open(store_dir)
    assert "99" in str(exc.value)
    assert len(rawlog.read(log_path)) == len(before)


def test_case_15_body_len_over_max(store_dir: Path, log_path: Path) -> None:
    """Case 15 — body_len > MAX_BODY → CorruptFrame. Not a length we ever wrote."""
    seed(store_dir, 3)
    rawlog.set_body_len(log_path, 1, MAX_BODY + 1)
    with pytest.raises(CorruptFrame):
        MemoryStore.open(store_dir)


def test_case_15_body_len_below_the_floor(store_dir: Path, log_path: Path) -> None:
    """Case 15 — and below the 18-byte floor, with data after it, is also corrupt."""
    seed(store_dir, 3)
    rawlog.set_body_len(log_path, 1, 7)
    with pytest.raises(CorruptFrame):
        MemoryStore.open(store_dir)


def test_case_16_sequence_break_in_a_middle_frame(
    store_dir: Path, log_path: Path
) -> None:
    """Case 16 — a hand-broken seq in a middle frame → SequenceBreak.

    The CRC is recomputed over the edited body on purpose: otherwise the scan
    would trip on the checksum first and this case would never run.
    """
    seed(store_dir, 5)
    rawlog.set_seq(log_path, 2, 99)  # frame 2 should carry seq 3

    with pytest.raises(SequenceBreak) as exc:
        MemoryStore.open(store_dir)
    message = str(exc.value)
    assert "3" in message and "99" in message


def test_case_17_payload_over_max_body(store: MemoryStore, log_path: Path) -> None:
    """Case 17 — an oversized payload fails, leaves the file byte-identical, and
    the log stays openable and usable."""
    store.append_episode(b"first", "k1", 1)
    before = rawlog.read(log_path)

    with pytest.raises(TooLarge):
        store.append_episode(b"\x00" * (MAX_BODY + 1), "too-big", 2)

    assert rawlog.read(log_path) == before
    assert store.head() == 1
    # Still usable afterwards: the rejection did not poison the handle.
    assert store.append_episode(b"second", "k2", 3) == 2
    assert [e.payload for e in store.episodes_since(0)] == [b"first", b"second"]


def test_case_17_rejected_append_leaves_the_log_openable(
    store_dir: Path, log_path: Path
) -> None:
    """Case 17 — and the file still opens cleanly in a fresh handle."""
    with MemoryStore.open(store_dir) as s:
        s.append_episode(b"first", "k1", 1)
        with pytest.raises(TooLarge):
            s.append_episode(b"\x00" * (MAX_BODY + 1), "too-big", 2)
    before = rawlog.read(log_path)

    with MemoryStore.open(store_dir) as s:
        assert s.head() == 1
        assert [e.payload for e in s.episodes_since(0)] == [b"first"]
    assert rawlog.read(log_path) == before


def test_case_18_key_over_max_key(store: MemoryStore, log_path: Path) -> None:
    """Case 18 — a key over MAX_KEY fails and the file is byte-identical."""
    store.append_episode(b"first", "k1", 1)
    before = rawlog.read(log_path)

    with pytest.raises(TooLarge):
        store.append_episode(b"payload", "K" * (MAX_KEY + 1), 2)

    assert rawlog.read(log_path) == before
    assert store.head() == 1
    # Exactly MAX_KEY is fine — the boundary is inclusive.
    assert store.append_episode(b"payload", "K" * MAX_KEY, 3) == 2


def test_case_19_write_key_conflict(store: MemoryStore, log_path: Path) -> None:
    """Case 19 — same key, different payload → WriteKeyConflict, nothing written."""
    assert store.append_episode(b"original", "entity:alice", 1) == 1
    before = rawlog.read(log_path)

    with pytest.raises(WriteKeyConflict) as exc:
        store.append_episode(b"changed", "entity:alice", 2)
    assert "entity:alice" in str(exc.value)

    assert rawlog.read(log_path) == before
    assert store.head() == 1
    episodes = list(store.episodes_since(0))
    assert len(episodes) == 1
    assert episodes[0].payload == b"original"


def test_case_20_episodes_since_ahead_of_head(store: MemoryStore) -> None:
    """Case 20 — since(head()+1) raises CheckpointAhead, not an empty iterator.

    Asserted eagerly: the call itself must raise. If the seam wrapped this in a
    lazy generator the error would only appear on first ``next()``, and a
    caller doing ``if not list(...)`` would read a consumer bug as "no news".
    """
    for _ in range(3):
        store.append_episode(b"e")
    assert store.head() == 3

    with pytest.raises(CheckpointAhead):
        store.episodes_since(4)  # not list(...): the call itself must raise
    with pytest.raises(CheckpointAhead):
        store.episodes_since(1_000_000)

    # And it is genuinely distinct from the legitimate empty case.
    assert list(store.episodes_since(3)) == []


def test_case_20_ahead_on_an_empty_log(store: MemoryStore) -> None:
    """Case 20 — on an empty log, since(1) is ahead; since(0) is simply empty."""
    assert store.head() == 0
    assert list(store.episodes_since(0)) == []
    with pytest.raises(CheckpointAhead):
        store.episodes_since(1)


def test_case_21_set_checkpoint_ahead_of_head(store: MemoryStore) -> None:
    """Case 21 — set_checkpoint(name, head()+1) is an error."""
    for _ in range(3):
        store.append_episode(b"e")

    with pytest.raises(CheckpointAhead):
        store.set_checkpoint("graph", 4)
    assert store.checkpoint("graph") == 0

    store.set_checkpoint("graph", 3)  # exactly head is fine
    assert store.checkpoint("graph") == 3


def test_case_22_second_process_is_locked_out(store_dir: Path) -> None:
    """Case 22 — a second *process* opening the same log → AlreadyLocked.

    The singleton rule (DL-016) made physical. Run in a real subprocess because
    that is the case the rule exists for.
    """
    script = (
        "import sys\n"
        "from omega.memory import AlreadyLocked, MemoryStore\n"
        "try:\n"
        "    MemoryStore.open(sys.argv[1])\n"
        "except AlreadyLocked:\n"
        "    print('LOCKED')\n"
        "else:\n"
        "    print('OPENED')\n"
    )
    with MemoryStore.open(store_dir) as holder:
        holder.append_episode(b"held", "", 1)
        result = subprocess.run(
            [sys.executable, "-c", script, str(store_dir)],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parents[1]),
            timeout=60,
        )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "LOCKED", result.stdout + result.stderr

    # Once the holder is gone the lock is released and the log opens normally.
    result = subprocess.run(
        [sys.executable, "-c", script, str(store_dir)],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
        timeout=60,
    )
    assert result.stdout.strip() == "OPENED", result.stdout + result.stderr


def test_case_22_second_handle_in_one_process_is_locked_out(store_dir: Path) -> None:
    """Case 22 — and a second handle inside one process is locked out too."""
    with MemoryStore.open(store_dir) as first:
        first.append_episode(b"held", "", 1)
        with pytest.raises(AlreadyLocked):
            MemoryStore.open(store_dir)
    second = MemoryStore.open(store_dir)  # released by the context manager
    try:
        assert second.head() == 1
    finally:
        second.close()


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_case_23_unwritable_directory(tmp_path: Path) -> None:
    """Case 23 — an unwritable path is a clear OSError, never a silent no-op."""
    locked_dir = tmp_path / "readonly"
    locked_dir.mkdir()
    os.chmod(locked_dir, 0o555)
    try:
        with pytest.raises(OSError) as exc:
            MemoryStore.open(locked_dir)
        assert str(exc.value), "the error must say something"
        # Nothing was created: no silent no-op leaving an empty store behind.
        assert list(locked_dir.iterdir()) == []
    finally:
        os.chmod(locked_dir, 0o755)


def test_case_23_missing_parent_directory(tmp_path: Path) -> None:
    """Case 23 — and a path whose parent does not exist errors rather than guessing."""
    missing = tmp_path / "no-such-dir" / "episodes.log"
    with pytest.raises(OSError) as exc:
        MemoryStore.open(missing)
    assert str(exc.value)
    assert not missing.exists()
    assert not missing.parent.exists()
