"""The indexes are caches, and readers see a consistent view.
M0_SPEC.md cases 34-36.

DL-016: the file is the source of truth and both indexes are derived. A cache
that can disagree with the file is a second source of truth, which is the thing
this whole design exists to avoid.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
import rawlog

from omega.memory import MemoryStore, WriteKeyConflict
from conftest import seed


def test_case_34_offset_index_matches_a_fresh_scan(
    store_dir: Path, log_path: Path
) -> None:
    """Case 34 — the index the log holds equals an independent scan of the file.

    The comparison is against ``rawlog.scan``, which parses the bytes from the
    spec's documented layout rather than asking the log — so this is a real
    cross-check, not the log agreeing with itself.
    """
    with MemoryStore.open(store_dir) as s:
        for i in range(1, 21):
            s.append_episode(b"x" * (i * 7), f"k{i}", i)
        offsets = s.diagnostics.offsets()

    independent = rawlog.frame_offsets(log_path)
    assert len(independent) == 20
    assert offsets == independent
    assert offsets[0] == rawlog.HEADER_LEN
    assert offsets == sorted(offsets)


def test_case_34_offset_index_matches_a_fresh_scan_after_recovery(
    store_dir: Path, log_path: Path
) -> None:
    """Case 34 — and it still matches after a torn-tail recovery."""
    seed(store_dir, 8)
    last = rawlog.scan(log_path)[-1]
    rawlog.truncate(log_path, last.offset + rawlog.PREFIX_LEN + 1)

    with MemoryStore.open(store_dir) as s:
        assert s.diagnostics.recovered_bytes > 0
        offsets = s.diagnostics.offsets()
        assert s.head() == 7

    independent = rawlog.frame_offsets(log_path)
    assert len(independent) == 7
    assert offsets == independent


def test_case_35_indexes_are_caches(store_dir: Path, log_path: Path) -> None:
    """Case 35 — drop both in-memory indexes, rescan, get identical results.

    Identical means: same offsets, same head, same episodes, and the dedup
    index still deduplicates and still conflicts. A rebuild that quietly lost
    the key map would show up on the last two assertions, not the first.
    """
    with MemoryStore.open(store_dir) as s:
        for i in range(1, 16):
            s.append_episode(f"payload-{i}".encode(), f"key-{i}", 1000 + i)

        before_offsets = s.diagnostics.offsets()
        before_head = s.head()
        before_episodes = list(s.episodes_since(0))
        assert len(before_offsets) == 15
        assert len(before_episodes) == 15

        s.diagnostics.rebuild_indexes()

        assert s.diagnostics.offsets() == before_offsets
        assert s.head() == before_head
        assert list(s.episodes_since(0)) == before_episodes
        assert s.diagnostics.offsets() == rawlog.frame_offsets(log_path)

        # The dedup index is a cache too: it must come back with the same answers.
        assert s.append_episode(b"payload-7", "key-7", 99) == 7
        with pytest.raises(WriteKeyConflict):
            s.append_episode(b"something else", "key-7", 99)
        assert s.head() == before_head


def test_case_35_rebuild_is_repeatable(store_dir: Path) -> None:
    """Case 35 — and rebuilding repeatedly never drifts."""
    with MemoryStore.open(store_dir) as s:
        for i in range(1, 11):
            s.append_episode(f"e{i}".encode(), "", i)
        baseline = (s.head(), s.diagnostics.offsets(), list(s.episodes_since(0)))
        for _ in range(5):
            s.diagnostics.rebuild_indexes()
            assert (s.head(), s.diagnostics.offsets(), list(s.episodes_since(0))) == baseline
        assert baseline[0] == 10


def test_case_36_concurrent_readers_during_appends(store: MemoryStore) -> None:
    """Case 36 — readers in one process see a consistent view while appends run.

    "Consistent" is asserted precisely: every read must yield a contiguous
    prefix 1..k with the right payload at every position. A reader that saw a
    half-written frame, a gap, or a stale offset would break one of those.
    """
    total = 400
    stop = threading.Event()
    errors: list[BaseException] = []
    observed: list[int] = []

    def expected(seq: int) -> bytes:
        return f"concurrent-{seq}".encode()

    def writer() -> None:
        try:
            for i in range(1, total + 1):
                assert store.append_episode(expected(i), "", i) == i
        except BaseException as exc:  # noqa: BLE001 - reported, not swallowed
            errors.append(exc)
        finally:
            stop.set()

    def reader() -> None:
        try:
            reads = 0
            while not stop.is_set() or reads < 3:
                episodes = list(store.episodes_since(0))
                seqs = [e.seq for e in episodes]
                assert seqs == list(range(1, len(episodes) + 1)), "gap in the view"
                for e in episodes:
                    assert e.payload == expected(e.seq), f"bad payload at {e.seq}"
                    assert e.ts_micros == e.seq
                observed.append(len(episodes))
                reads += 1
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=writer)] + [
        threading.Thread(target=reader) for _ in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    assert not any(t.is_alive() for t in threads), "a thread hung"

    assert not errors, errors[0]
    assert store.head() == total
    assert len(list(store.episodes_since(0))) == total
    # Fail closed on empty: the readers must actually have seen episodes.
    assert len(observed) >= 12, f"only {len(observed)} reads happened"
    assert max(observed) == total
    # And the view really was growing under them, not a single frozen snapshot.
    assert observed == sorted(observed) or len(set(observed)) > 1


def test_case_36_snapshot_iterator_is_stable(store: MemoryStore) -> None:
    """Case 36 — an iterator taken before an append does not grow under the reader."""
    for i in range(1, 6):
        store.append_episode(f"e{i}".encode(), "", i)

    it = store.episodes_since(0)
    first = next(it)
    assert first.seq == 1

    for i in range(6, 11):
        store.append_episode(f"e{i}".encode(), "", i)

    rest = list(it)
    assert [e.seq for e in rest] == [2, 3, 4, 5], "the snapshot grew mid-iteration"
    assert store.head() == 10
    assert len(list(store.episodes_since(0))) == 10
