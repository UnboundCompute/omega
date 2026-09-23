"""Green — it does what it should. M0_SPEC.md cases 1-12."""

from __future__ import annotations

from pathlib import Path

import pytest

from omega.memory import (
    Episode,
    MemoryStore,
    WriteKeyConflict,
)


def test_case_01_round_trip_one_episode(store: MemoryStore) -> None:
    """Case 1 — append one episode, read it back byte-identical, ts and key too."""
    payload = b"\x00the first episode\xff"
    seq = store.append_episode(payload, "key-one", 1_700_000_000_123_456)
    assert seq == 1

    episodes = list(store.episodes_since(0))
    assert len(episodes) == 1
    got = episodes[0]
    assert isinstance(got, Episode)
    assert got.seq == 1
    assert got.ts_micros == 1_700_000_000_123_456
    assert got.write_key == "key-one"
    assert got.payload == payload


def test_case_02_sequences_are_contiguous(store: MemoryStore) -> None:
    """Case 2 — append N; sequences are exactly 1..N with no gaps."""
    n = 50
    seqs = [store.append_episode(f"e{i}".encode()) for i in range(1, n + 1)]
    assert seqs == list(range(1, n + 1))
    assert [e.seq for e in store.episodes_since(0)] == list(range(1, n + 1))


def test_case_03_episodes_since(store: MemoryStore) -> None:
    """Case 3 — since(0) yields all; since(k) yields exactly k+1..N."""
    n = 10
    for i in range(1, n + 1):
        store.append_episode(f"e{i}".encode(), "", 9_000 + i)

    assert [e.seq for e in store.episodes_since(0)] == list(range(1, n + 1))
    for k in range(0, n + 1):
        got = list(store.episodes_since(k))
        assert [e.seq for e in got] == list(range(k + 1, n + 1)), f"since({k})"
        assert [e.payload for e in got] == [
            f"e{i}".encode() for i in range(k + 1, n + 1)
        ]
    assert list(store.episodes_since(n)) == []  # exactly at head: empty, no error


def test_case_04_head(store: MemoryStore) -> None:
    """Case 4 — head() is 0 on a fresh log, then N."""
    assert store.head() == 0
    for i in range(1, 8):
        store.append_episode(b"x")
        assert store.head() == i
    assert store.head() == 7


def test_case_05_close_and_reopen(store_dir: Path) -> None:
    """Case 5 — close and reopen: same episodes, same order, same head()."""
    payloads = [b"alpha", b"beta", b"gamma"]
    with MemoryStore.open(store_dir) as s:
        for i, p in enumerate(payloads, start=1):
            s.append_episode(p, f"k{i}", 100 + i)
        before = list(s.episodes_since(0))
        assert s.head() == 3

    with MemoryStore.open(store_dir) as s:
        after = list(s.episodes_since(0))
        assert s.head() == 3

    assert len(after) == 3
    assert after == before
    assert [e.payload for e in after] == payloads
    assert [e.write_key for e in after] == ["k1", "k2", "k3"]
    assert [e.ts_micros for e in after] == [101, 102, 103]


def test_case_06_same_key_same_payload_dedupes(store: MemoryStore) -> None:
    """Case 6 — same write-key, identical payload → one record, same seq twice."""
    first = store.append_episode(b"identical", "dedup-key", 11)
    second = store.append_episode(b"identical", "dedup-key", 22)
    assert first == second == 1
    assert store.head() == 1

    episodes = list(store.episodes_since(0))
    assert len(episodes) == 1
    # The stored timestamp is the first write's: nothing was written the second time.
    assert episodes[0].ts_micros == 11


def test_case_07_two_keys_two_records(store: MemoryStore) -> None:
    """Case 7 — two different write-keys → two records, different seqs."""
    a = store.append_episode(b"one", "key-a")
    b = store.append_episode(b"two", "key-b")
    assert (a, b) == (1, 2)
    episodes = list(store.episodes_since(0))
    assert [e.write_key for e in episodes] == ["key-a", "key-b"]
    assert [e.payload for e in episodes] == [b"one", b"two"]


def test_case_08_empty_key_is_not_a_key(store: MemoryStore) -> None:
    """Case 8 — two appends with an empty key → two records."""
    a = store.append_episode(b"same bytes", "")
    b = store.append_episode(b"same bytes", "")
    assert (a, b) == (1, 2)
    episodes = list(store.episodes_since(0))
    assert len(episodes) == 2
    assert [e.write_key for e in episodes] == ["", ""]
    assert [e.payload for e in episodes] == [b"same bytes", b"same bytes"]


def test_case_09_checkpoint_round_trip(store: MemoryStore) -> None:
    """Case 9 — checkpoint round-trips; an unset checkpoint reads 0."""
    assert store.checkpoint("graph") == 0
    assert store.checkpoint("never-set-at-all") == 0

    for _ in range(5):
        store.append_episode(b"e")

    store.set_checkpoint("graph", 3)
    assert store.checkpoint("graph") == 3
    assert store.checkpoint("user-model") == 0

    store.set_checkpoint("user-model", 5)
    assert store.checkpoint("graph") == 3
    assert store.checkpoint("user-model") == 5
    assert sorted(store.checkpoint_names()) == ["graph", "user-model"]


def test_case_09_checkpoint_survives_reopen(store_dir: Path) -> None:
    """Case 9 — and it round-trips across a close/reopen, which is its whole job."""
    with MemoryStore.open(store_dir) as s:
        for _ in range(4):
            s.append_episode(b"e")
        s.set_checkpoint("graph", 2)

    with MemoryStore.open(store_dir) as s:
        assert s.checkpoint("graph") == 2
        assert s.checkpoint_names() == ["graph"]


def test_case_09_zero_filled_checkpoint_sidecar_does_not_block_the_log(
    store_dir: Path, log_path: Path
) -> None:
    """Case 9, extended — the same crash that zero-fills the log's tail can
    zero-fill the sidecar's, and the sidecar loads during open.

    Checkpoints are derived state (spec, "Checkpoints"). Derived state must
    never be able to make acknowledged episodes unreachable, which is the
    violation metric. Not one of the 39 numbered cases; added because the
    sidecar sits on the open path.
    """
    with MemoryStore.open(store_dir) as s:
        for i in range(1, 4):
            s.append_episode(f"e{i}".encode(), "", i)
        s.set_checkpoint("graph", 2)

    sidecar = log_path.with_name(log_path.name + ".checkpoints")
    assert sidecar.is_file()
    sidecar.write_bytes(sidecar.read_bytes() + b"\x00" * 64)

    with MemoryStore.open(store_dir) as s:
        assert s.head() == 3
        assert [e.payload for e in s.episodes_since(0)] == [b"e1", b"e2", b"e3"]
        assert s.checkpoint("graph") == 2


def test_case_09_missing_checkpoint_sidecar_is_an_empty_set(
    store_dir: Path, log_path: Path
) -> None:
    """Case 9, extended — a sidecar that is gone reads as "nothing consumed",
    not as an error. Losing a checkpoint costs replay; losing the log costs
    everything."""
    with MemoryStore.open(store_dir) as s:
        for i in range(1, 4):
            s.append_episode(f"e{i}".encode(), "", i)
        s.set_checkpoint("graph", 2)

    log_path.with_name(log_path.name + ".checkpoints").unlink()

    with MemoryStore.open(store_dir) as s:
        assert s.head() == 3
        assert [e.payload for e in s.episodes_since(0)] == [b"e1", b"e2", b"e3"]
        assert s.checkpoint("graph") == 0
        assert s.checkpoint_names() == []


@pytest.mark.parametrize(
    "name,payload",
    [
        ("empty", b""),
        ("embedded-nuls", b"before\x00\x00\x00after"),
        ("invalid-utf8", b"\xff\xfe\xfd\x80\x81 not utf-8 \xc3\x28"),
        ("all-bytes", bytes(range(256))),
        ("ten-mib", b"\xa5" * (10 * 1024 * 1024)),
    ],
    # Explicit ids: without them pytest inlines the payload bytes into the test
    # id, which turns a 10 MiB case into a 40 MB report.
    ids=["empty", "embedded-nuls", "invalid-utf8", "all-bytes", "ten-mib"],
)
def test_case_10_payload_edges_round_trip(
    store: MemoryStore, name: str, payload: bytes
) -> None:
    """Case 10 — empty, embedded NULs, invalid UTF-8 and 10 MiB all round-trip."""
    seq = store.append_episode(payload, name, 7)
    episodes = list(store.episodes_since(0))
    assert len(episodes) == 1
    assert episodes[0].seq == seq
    assert episodes[0].payload == payload
    assert len(episodes[0].payload) == len(payload)
    assert episodes[0].write_key == name


def test_case_10_payload_edges_survive_reopen(store_dir: Path) -> None:
    """Case 10 — and they survive a reopen, which is where a bad length shows up."""
    payloads = [b"", b"a\x00b", b"\xff\xfe\x80", bytes(range(256))]
    with MemoryStore.open(store_dir) as s:
        for i, p in enumerate(payloads, start=1):
            s.append_episode(p, f"edge{i}", i)
    with MemoryStore.open(store_dir) as s:
        got = [e.payload for e in s.episodes_since(0)]
    assert got == payloads


def test_case_11_ten_thousand_appends(store: MemoryStore) -> None:
    """Case 11 — 10,000 appends: all readable, contiguous, correct order."""
    n = 10_000
    for i in range(1, n + 1):
        assert store.append_episode(str(i).encode(), "", i) == i
    assert store.head() == n

    episodes = list(store.episodes_since(0))
    assert len(episodes) == n
    assert [e.seq for e in episodes] == list(range(1, n + 1))
    assert [e.payload for e in episodes] == [str(i).encode() for i in range(1, n + 1)]
    assert [e.ts_micros for e in episodes] == list(range(1, n + 1))


def test_case_12_dedup_index_survives_restart(store_dir: Path) -> None:
    """Case 12 — the same write-key after reopening still deduplicates."""
    with MemoryStore.open(store_dir) as s:
        assert s.append_episode(b"extracted once", "entity:alice", 1) == 1
        assert s.append_episode(b"another", "entity:bob", 2) == 2

    with MemoryStore.open(store_dir) as s:
        # Identical payload → the original seq, nothing written.
        assert s.append_episode(b"extracted once", "entity:alice", 999) == 1
        assert s.head() == 2
        # And the conflict rule survives the restart too.
        with pytest.raises(WriteKeyConflict):
            s.append_episode(b"different now", "entity:alice", 999)
        assert s.head() == 2
        episodes = list(s.episodes_since(0))
        assert len(episodes) == 2
        assert [e.write_key for e in episodes] == ["entity:alice", "entity:bob"]
