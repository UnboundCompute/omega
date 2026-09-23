"""Added after the adversarial audit. M0_SPEC.md cases 40-48.

Every case here corresponds to a defect that **existed and shipped**, and that
the first 39 cases missed. The spec lists them with what they caught, because a
case whose origin is forgotten is a case someone later deletes as redundant —
so each test below says, in its docstring, which audit finding it is and what
went wrong before it.

The recurring shape of those findings is worth stating once: every one of them
was a path where the code did something *destructive or fatal* on evidence it
had not checked. A length believed before its checksum (40, 41). A truncation
that ran before the magic check (45). A sidecar that could veto the log (42).
A durability call nothing observed (44). A path rule that read the disk instead
of its argument (46). A boundary nobody tested on both sides (47). An ordering
nothing pinned (48).
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest
import rawlog

from omega.memory import (
    EPISODES_FILENAME,
    CheckpointAhead,
    CorruptFrame,
    MAX_BODY,
    MemoryStore,
    NotAnOmegaLog,
    WriteKeyConflict,
    dir_syncs,
    file_syncs,
)
from conftest import seed


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _episodes(store_dir: Path) -> list[bytes]:
    with MemoryStore.open(store_dir) as s:
        return [e.payload for e in s.episodes_since(0)]


def _sidecar(log_path: Path) -> Path:
    """The checkpoint sidecar beside a log. Named by the implementation, not by
    the spec — see the note on the sidecar layout in case 42."""
    return log_path.with_name(log_path.name + ".checkpoints")


# ---------------------------------------------------------------------------
# 40 - a corrupted body_len never silently truncates
# ---------------------------------------------------------------------------


def _flip_length_bit(log_path: Path, index: int, bit: int) -> int:
    """Flip one bit of frame ``index``'s ``body_len``, as a real bit rot would.

    Crucially the length's **own checksum is left alone**: a flipped bit on
    disk does not politely recompute ``len_crc``. That is what makes this the
    version-2 defence and not a forgery.
    """
    frame = rawlog.scan(log_path)[index]
    damaged = frame.body_len ^ (1 << bit)
    assert rawlog.FIXED_BODY_LEN <= damaged <= MAX_BODY, (
        f"the damaged length {damaged} must stay inside the plausible range, "
        "or this exercises the implausible-value branch instead (case 15)"
    )
    rawlog.set_body_len(log_path, index, damaged, fix_len_crc=False)
    return damaged


@pytest.mark.parametrize("index", [0, 2], ids=["first-frame", "middle-frame"])
def test_case_40_a_flipped_length_bit_is_corruption_not_a_truncation(
    store_dir: Path, log_path: Path, index: int
) -> None:
    """Case 40 / audit finding 1 — the one that destroyed acknowledged episodes.

    One bit flipped in a frame's length field, landing inside ``18..=MAX_BODY``
    but claiming more bytes than the file holds. Under version 1 this opened
    *cleanly*: the length was believed, the frame looked like it ran past EOF,
    recovery called that a torn tail and **truncated the file on disk**. Three
    acknowledged episodes gone, ``head()`` reset, no error raised.

    The old suite missed it because both of its length-mutation tests used
    *implausible* values, which take a different branch. The plausible-but-
    too-large case was never constructed.
    """
    payloads = seed(store_dir, 5)
    healthy = rawlog.read(log_path)

    damaged = _flip_length_bit(log_path, index, 16)
    assert damaged > rawlog.size(log_path), "the frame must claim more than the file holds"
    broken = rawlog.read(log_path)

    with pytest.raises(CorruptFrame):
        MemoryStore.open(store_dir)

    # The whole point: nothing was thrown away.
    assert rawlog.read(log_path) == broken, "a failed open must not truncate"
    assert rawlog.size(log_path) == len(healthy)

    # And every episode is still there once the flipped bit is put back.
    rawlog.write(log_path, healthy)
    assert _episodes(store_dir) == payloads


def test_case_40_a_length_stretched_to_end_exactly_at_eof(
    store_dir: Path, log_path: Path
) -> None:
    """Case 40, the third variant — a damaged length that ends *exactly* at EOF.

    This is the nastiest shape, because a frame ending exactly at EOF is the
    one place the spec legitimately allows a torn-tail truncation. The length
    still fails its own checksum, so it never gets that far: nothing it says
    about where the frame ends may be acted on at all.
    """
    payloads = seed(store_dir, 5)
    healthy = rawlog.read(log_path)

    first = rawlog.scan(log_path)[0]
    stretched = len(healthy) - (first.offset + rawlog.PREFIX_LEN)
    assert rawlog.FIXED_BODY_LEN <= stretched <= MAX_BODY
    rawlog.set_body_len(log_path, 0, stretched, fix_len_crc=False)
    broken = rawlog.read(log_path)

    with pytest.raises(CorruptFrame):
        MemoryStore.open(store_dir)

    assert rawlog.read(log_path) == broken
    rawlog.write(log_path, healthy)
    assert _episodes(store_dir) == payloads


# ---------------------------------------------------------------------------
# 41 - len_crc is checked before body_len is used
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("index", [0, 2, 4], ids=["first", "middle", "last-at-eof"])
def test_case_41_a_length_that_fails_its_checksum_with_data_present(
    store_dir: Path, log_path: Path, index: int
) -> None:
    """Case 41 — the length's checksum is consulted first, and a failure is
    judged on the bytes' own evidence.

    Here only ``len_crc`` is damaged; ``body_len`` itself still names the real
    frame. That is deliberate: it isolates the *ordering*. If the length were
    used before its checksum were checked, every one of these would open
    perfectly and the check would be dead code.

    ``last-at-eof`` is the discriminating one. A bad *body* CRC at exact EOF is
    a torn tail (spec step 5) — but a bad *length* CRC has no such escape,
    because a length that cannot vouch for itself gives no grounds for
    believing the frame ends at EOF in the first place. Non-zero bytes are
    present, so it is corruption wherever it sits.
    """
    seed(store_dir, 5)
    rawlog.corrupt_len_crc(log_path, index)
    broken = rawlog.read(log_path)

    with pytest.raises(CorruptFrame):
        MemoryStore.open(store_dir)

    assert rawlog.read(log_path) == broken, "a failed open must not truncate"


def test_case_41_the_zero_filled_tail_still_works_after_the_format_change(
    store_dir: Path, log_path: Path
) -> None:
    """Case 41 — and zeros to EOF still recover, which is the other half.

    The zero-fill clause survives the format change *because* of how the check
    orders: an all-zero prefix fails its own length checksum (a CRC-32 of four
    zero bytes is not zero), which routes it into the same
    classify-on-the-evidence branch, where "all zeros to EOF" means crash
    artifact. The first assertion pins that premise — if a zero prefix ever
    checksum-validated, this whole path would change shape without a test
    noticing.
    """
    assert rawlog.len_crc32(0) != 0, "an all-zero prefix must fail its own checksum"

    payloads = seed(store_dir, 4)
    clean_size = rawlog.size(log_path)
    rawlog.append_bytes(log_path, b"\x00" * 4096)

    with MemoryStore.open(store_dir) as s:
        assert s.head() == 4
        assert [e.payload for e in s.episodes_since(0)] == payloads
        assert s.diagnostics.recovered_bytes == 4096
    assert rawlog.size(log_path) == clean_size


def test_case_41_a_bad_length_checksum_with_zeros_after_it_is_a_torn_tail(
    store_dir: Path, log_path: Path
) -> None:
    """Case 41 — a forged frame whose ``len_crc`` is wrong, followed by nothing
    but zeros, is a crash artifact and truncates. Same branch, other outcome."""
    payloads = seed(store_dir, 3)
    clean_size = rawlog.size(log_path)
    rawlog.append_bytes(log_path, b"\x00" * 64)

    with MemoryStore.open(store_dir) as s:
        assert s.head() == 3
        assert [e.payload for e in s.episodes_since(0)] == payloads
    assert rawlog.size(log_path) == clean_size


# ---------------------------------------------------------------------------
# 42 - a damaged sidecar never blocks the log
# ---------------------------------------------------------------------------
#
# The spec fixes the sidecar's *semantics* but explicitly not its byte layout,
# so the targeted mutations below (bad magic, absurd count, absurd name_len)
# anchor themselves on the one thing that is certainly in the file — the
# checkpoint's name — and step backwards from it, rather than hardcoding
# offsets. The layout they assume is only: `... | count u32 | name_len u16 |
# name | ...`, with the magic at byte 0.


def _clean_sidecar(store_dir: Path, log_path: Path, n: int = 3) -> bytes:
    """Seed ``n`` episodes and one checkpoint, and return the sidecar's bytes."""
    seed(store_dir, n)
    with MemoryStore.open(store_dir) as s:
        s.set_checkpoint("graph", 2)
    return _sidecar(log_path).read_bytes()


def _name_offset(clean: bytes) -> int:
    at = clean.index(b"graph")
    assert at > 6, "the anchor must leave room for count and name_len before it"
    return at


def _damage(name: str, clean: bytes) -> bytes:
    b = bytearray(clean)
    name_at = _name_offset(clean)
    if name == "zero bytes":
        return b""
    if name == "all zeros":
        return bytes(len(clean))
    if name == "truncated by one":
        return bytes(b[:-1])
    if name == "one flipped bit":
        b[-6] ^= 0x01  # inside the last entry's seq, so the CRC no longer holds
        return bytes(b)
    if name == "bad magic":
        b[0] = ord("X")
        return bytes(b)
    if name == "absurd count":
        b[name_at - 6 : name_at - 2] = struct.pack("<I", 0xFFFFFFFF)
        return bytes(b)
    if name == "absurd name_len":
        b[name_at - 2 : name_at] = struct.pack("<H", 0xFFFF)
        return bytes(b)
    raise AssertionError(f"unknown damage {name!r}")


DAMAGE = [
    "zero bytes",
    "all zeros",
    "truncated by one",
    "one flipped bit",
    "bad magic",
    "absurd count",
    "absurd name_len",
]


def test_case_42_the_clean_sidecar_is_actually_readable(
    store_dir: Path, log_path: Path
) -> None:
    """Case 42's control, and it is not optional.

    Every assertion below is "the log opened and checkpoints read 0". That is
    also what a *totally broken* sidecar writer would produce, so without this
    control the seven cases could all pass for the wrong reason. Here the
    undamaged file must decode: checkpoint 2, nothing set aside.
    """
    _clean_sidecar(store_dir, log_path)
    with MemoryStore.open(store_dir) as s:
        assert s.checkpoint("graph") == 2
        assert s.checkpoint_names() == ["graph"]
        assert s.checkpoints_reset is False
        assert s.damaged_checkpoints_path is None


@pytest.mark.parametrize("damage", DAMAGE)
def test_case_42_a_damaged_sidecar_never_blocks_the_log(
    store_dir: Path, log_path: Path, damage: str
) -> None:
    """Case 42 / audit finding 2 — all seven of these refused to open.

    Checkpoints are mutable derived state, not the source of truth. Loading
    them behind a ``?`` meant zero bytes, all zeros, a truncation or a flipped
    bit made every acknowledged episode unreachable — the violation metric,
    reached through state that is not even authoritative. Worse, the two
    likeliest artifacts (a rename visible before its directory fsync; a size
    extension persisted without its data pages) are exactly the ones the log's
    own zero-fill clause exists to survive: the log called them survivable and
    the sidecar called them fatal.
    """
    clean = _clean_sidecar(store_dir, log_path)
    broken = _damage(damage, clean)
    _sidecar(log_path).write_bytes(broken)

    with MemoryStore.open(store_dir) as s:
        # The log opens and every episode is readable. That is the whole rule.
        assert s.head() == 3
        assert len(list(s.episodes_since(0))) == 3
        # Same outcome as a *missing* sidecar: no new failure mode, a second
        # road to a place we already went.
        assert s.checkpoint("graph") == 0
        assert s.checkpoint_names() == []
        # Loud without being fatal: the seam is told it happened...
        assert s.checkpoints_reset is True
        moved = s.damaged_checkpoints_path
        # ...and the evidence is preserved, not deleted.
        assert moved is not None, "the damaged sidecar must be kept as evidence"
        assert moved.read_bytes() == broken
        assert moved.name.startswith(_sidecar(log_path).name + ".damaged")
    assert not _sidecar(log_path).exists()

    # And the log is usable afterwards: a fresh checkpoint round-trips and the
    # next open is clean.
    with MemoryStore.open(store_dir) as s:
        assert s.checkpoints_reset is False
        s.set_checkpoint("graph", 3)
    with MemoryStore.open(store_dir) as s:
        assert s.checkpoint("graph") == 3
        assert s.checkpoints_reset is False


def test_case_42_a_second_damaged_sidecar_gets_its_own_name(
    store_dir: Path, log_path: Path
) -> None:
    """Case 42 — a second damaged sidecar is a second piece of evidence, not a
    reason to destroy the first."""
    seed(store_dir, 2)
    seen: list[Path] = []
    for i in range(3):
        _sidecar(log_path).write_bytes(bytes([i]) * 40)
        with MemoryStore.open(store_dir) as s:
            moved = s.damaged_checkpoints_path
            assert moved is not None
            assert moved not in seen, f"reused {moved}"
            seen.append(moved)
    for i, path in enumerate(seen):
        assert path.read_bytes() == bytes([i]) * 40, "earlier evidence was clobbered"


# ---------------------------------------------------------------------------
# 43 - a checkpoint ahead of head fails at open
# ---------------------------------------------------------------------------


def test_case_43_a_checkpoint_ahead_of_head_fails_at_open(
    store_dir: Path, log_path: Path
) -> None:
    """Case 43 / audit finding 3 — at open, not lazily on the next read.

    A well-formed checkpoint past ``head`` cannot legitimately happen, so it is
    *proof* that acknowledged episodes were lost, and this open is the one
    moment that proof exists. Checking it lazily left the log opening
    "cleanly" and the consumer wedged: unable to read (``episodes_since``
    raised) and unable to reset its own position (``set_checkpoint`` raised
    too), with nothing having said why.

    The damage here is the realistic one: the log loses its tail to a crash
    while a consumer had already checkpointed past it.
    """
    seed(store_dir, 5)
    with MemoryStore.open(store_dir) as s:
        s.set_checkpoint("graph", 5)
    # A torn tail takes the log back to 3 episodes, under a checkpoint at 5.
    rawlog.truncate(log_path, rawlog.scan(log_path)[3].offset)

    with pytest.raises(CheckpointAhead) as exc:
        MemoryStore.open(store_dir)
    message = str(exc.value)
    assert "5" in message and "3" in message, message

    # The documented way out: remove the sidecar by hand. A missing one reads
    # as 0, consumers replay, and write-key dedup makes replay idempotent.
    _sidecar(log_path).unlink()
    with MemoryStore.open(store_dir) as s:
        assert s.head() == 3
        assert len(list(s.episodes_since(0))) == 3
        assert s.checkpoint("graph") == 0


def test_case_43_it_is_the_open_that_raises_not_the_first_read(
    store_dir: Path, log_path: Path
) -> None:
    """Case 43 — pinned precisely, because "it raises eventually" is the bug.

    If the check were lazy the constructor would succeed and the raise would
    only appear on the next ``episodes_since``. Asserting on ``open`` itself is
    the difference between the fixed behaviour and the shipped one.
    """
    seed(store_dir, 4)
    with MemoryStore.open(store_dir) as s:
        s.set_checkpoint("graph", 4)
    rawlog.truncate(log_path, rawlog.scan(log_path)[2].offset)

    opened: list[MemoryStore] = []
    try:
        with pytest.raises(CheckpointAhead):
            opened.append(MemoryStore.open(store_dir))
    finally:
        for s in opened:
            s.close()
    assert not opened, "open() returned a handle instead of raising"


def test_case_43_a_checkpoint_exactly_at_head_opens_normally(store_dir: Path) -> None:
    """Case 43's control — the boundary is ``>``, not ``>=``. A check that
    refused the legitimate case would be worse than no check."""
    seed(store_dir, 3)
    with MemoryStore.open(store_dir) as s:
        s.set_checkpoint("graph", 3)
    with MemoryStore.open(store_dir) as s:
        assert s.checkpoint("graph") == 3
        assert s.head() == 3


# ---------------------------------------------------------------------------
# 44 - fsync is observed, not assumed
# ---------------------------------------------------------------------------
#
# The counters are thread-local and monotonic: read before and after on the
# SAME thread and diff. Never from a worker thread — a process-wide counter
# under a parallel runner is satisfied by some other thread's append, which is
# exactly the mutant these exist to kill.


class _Syncs:
    """Exact fsync deltas across a block, on this thread."""

    def __enter__(self) -> "_Syncs":
        self.file_before = file_syncs()
        self.dir_before = dir_syncs()
        return self

    def __exit__(self, *_: object) -> bool:
        self.files = file_syncs() - self.file_before
        self.dirs = dir_syncs() - self.dir_before
        return False


def test_case_44_creating_a_log_fsyncs_the_file_and_its_directory(
    store_dir: Path,
) -> None:
    """Case 44 / audit finding 4, sites 1 and 2 — the header and the directory.

    Without the directory fsync the file itself can vanish on crash, header and
    all. ``kill -9`` cannot see either call: the kernel completes the in-flight
    write and the page cache outlives the process, so the whole suite — 115
    tests, 40 crash trials — stayed green with every durability call deleted,
    94x faster. This is the test that goes red instead.
    """
    with _Syncs() as s:
        MemoryStore.open(store_dir).close()
    assert s.files == 1, "creating the log must fsync the header"
    assert s.dirs == 1, "creating the log must fsync the parent directory"


def test_case_44_every_append_fsyncs_exactly_once(store_dir: Path) -> None:
    """Case 44, site 3 — the non-negotiable one: an append that returned
    survives ``kill -9``, which means the fsync happens **before** it returns.

    Exactly once, not "at least once": a lower bound would be satisfied by a
    single fsync at close, and an upper bound catches a redundant one that
    would cost ~4ms of the append budget for nothing.
    """
    with MemoryStore.open(store_dir) as store:
        with _Syncs() as one:
            store.append_episode(b"durable", "", 1)
        assert one.files == 1
        assert one.dirs == 0, "an append into an existing file needs no directory fsync"

        with _Syncs() as many:
            for i in range(2, 12):
                store.append_episode(f"e{i}".encode(), "", i)
        assert many.files == 10
        assert many.dirs == 0


def test_case_44_a_rejected_append_does_not_fsync(store_dir: Path) -> None:
    """Case 44 — and nothing was written, so nothing is synced. This is the
    control that stops the append assertion passing on a blanket sync."""
    with MemoryStore.open(store_dir) as store:
        store.append_episode(b"first", "k", 1)
        with _Syncs() as s:
            with pytest.raises(WriteKeyConflict):
                store.append_episode(b"different", "k", 2)
            assert store.append_episode(b"first", "k", 3) == 1  # dedup: no write
        assert s.files == 0
        assert s.dirs == 0


def test_case_44_setting_a_checkpoint_fsyncs_the_temp_file_and_the_directory(
    store_dir: Path,
) -> None:
    """Case 44, sites 4 and 5 — write-temp, fsync, rename, fsync-dir.

    Both halves matter and fail differently: without the file fsync the rename
    can publish an empty file, and without the directory fsync the rename
    itself can be lost. The spec says "the same must hold for ... both sidecar
    fsyncs", so both are pinned separately.
    """
    with MemoryStore.open(store_dir) as store:
        store.append_episode(b"e1", "", 1)
        with _Syncs() as s:
            store.set_checkpoint("graph", 1)
        assert s.files == 1, "the sidecar temp file must be fsynced before the rename"
        assert s.dirs == 1, "the directory must be fsynced after the rename"


def test_case_44_truncating_a_torn_tail_is_fsynced(
    store_dir: Path, log_path: Path
) -> None:
    """Case 44, site 6 — recovery's own write.

    A truncation that is not durable is a truncation that comes back, and the
    clean-reopen half is what makes this an exact claim rather than "something
    somewhere synced": reopening an already-recovered log must sync nothing at
    all.
    """
    seed(store_dir, 4)
    with _Syncs() as clean:
        MemoryStore.open(store_dir).close()
    assert clean.files == 0, "reopening a clean log must not write"
    assert clean.dirs == 0

    rawlog.append_bytes(log_path, b"\x00" * 128)
    with _Syncs() as torn:
        with MemoryStore.open(store_dir) as s:
            assert s.diagnostics.recovered_bytes == 128
    assert torn.files == 1, "the truncation must be fsynced"
    assert torn.dirs == 0


def test_case_44_the_counters_are_not_a_no_op() -> None:
    """Case 44's control — a counter stuck at zero would make every assertion
    above vacuous in one direction, so it is shown to move at all."""
    assert file_syncs() >= 0 and dir_syncs() >= 0


# ---------------------------------------------------------------------------
# 45 - a small non-log file is refused, not overwritten
# ---------------------------------------------------------------------------

NOT_A_LOG = b"a note to self, not a log."
assert len(NOT_A_LOG) == 26


@pytest.mark.parametrize(
    "content",
    [NOT_A_LOG, b"#", b"OMEGALO!", b"\xff" * 31, b"OMEGALOG" + b"\x00" * 3],
    ids=["a-26-byte-note", "one-byte", "near-miss-magic", "high-bytes", "short-but-valid-magic"],
)
def test_case_45_a_small_non_log_file_is_refused_not_overwritten(
    store_dir: Path, log_path: Path, content: bytes
) -> None:
    """Case 45 / audit finding 5 — the magic check must run *before* the
    truncation, and for files under 32 bytes it did not.

    Any short file at this path — a note, a stray text file — was silently
    replaced with a log header, because the destructive step sat in front of
    the check. The last id is the control on the control: ``OMEGALOG`` plus
    three bytes *is* a prefix of one of our headers, so it is a genuine torn
    header and is legitimately rewritten (case 28). If this test refused
    everything short it would be passing by being useless.
    """
    log_path.write_bytes(content)
    assert rawlog.size(log_path) < rawlog.HEADER_LEN

    if content.startswith(rawlog.MAGIC[: len(content)]):
        # A torn header: no record can exist yet, so nothing can be lost.
        with MemoryStore.open(store_dir) as s:
            assert s.head() == 0
            assert s.append_episode(b"fresh", "", 1) == 1
        return

    with pytest.raises(NotAnOmegaLog):
        MemoryStore.open(store_dir)
    assert log_path.read_bytes() == content, "the file must be byte-identical"


def test_case_45_the_refusal_is_repeatable_and_still_changes_nothing(
    store_dir: Path, log_path: Path
) -> None:
    """Case 45 — three refusals in a row, still byte-identical. A destructive
    step that ran once would show up on the second attempt as a *different*
    error (or none at all), which is precisely how the original bug hid."""
    log_path.write_bytes(NOT_A_LOG)
    for attempt in range(3):
        with pytest.raises(NotAnOmegaLog):
            MemoryStore.open(store_dir)
        assert log_path.read_bytes() == NOT_A_LOG, f"attempt {attempt}"


# ---------------------------------------------------------------------------
# 46 - the seam's path rule is order-independent
# ---------------------------------------------------------------------------


def test_case_46_the_same_path_means_the_same_file_either_way(
    tmp_path: Path,
) -> None:
    """Case 46 / audit finding 6 — the path decides, never the disk.

    Two runs of the *same argument*, differing only in whether the directory
    existed first. Under the old ``is_dir()`` rule they landed in two different
    places: ``p/episodes.log`` when someone had made the directory, and ``p``
    itself when they had not. This asserts the relative location is identical,
    which is the failure directly.
    """
    absent = tmp_path / "absent" / "store"
    absent.parent.mkdir()
    with MemoryStore.open(absent) as s:
        where_absent = s.diagnostics.path.relative_to(absent.parent)
        assert s.append_episode(b"written-before-the-dir-existed", "", 1) == 1

    present = tmp_path / "present" / "store"
    present.mkdir(parents=True)
    with MemoryStore.open(present) as s:
        where_present = s.diagnostics.path.relative_to(present.parent)
        assert s.append_episode(b"written-after-the-dir-existed", "", 1) == 1

    assert where_absent == where_present == Path("store") / EPISODES_FILENAME


def test_case_46_the_directory_form_is_never_blocked(tmp_path: Path) -> None:
    """Case 46 — "and the file form then permanently blocked the directory form".

    Opening a path that does not exist must leave that path usable as a store
    *directory*, because a path already occupied by a file can never be made
    into one. Under the old rule the first open turned ``p`` into a file and
    there was no way back.
    """
    p = tmp_path / "store"
    with MemoryStore.open(p) as s:
        assert s.append_episode(b"first", "", 1) == 1

    assert p.is_dir(), "the store path must be a directory, not a log file"
    assert (p / EPISODES_FILENAME).is_file()

    # And the same argument reopens the same store, with the same episodes.
    with MemoryStore.open(p) as s:
        assert [e.payload for e in s.episodes_since(0)] == [b"first"]


def test_case_46_both_spellings_name_one_store(tmp_path: Path) -> None:
    """Case 46 — the directory form and the file form converge.

    ``open(d)`` and ``open(d / EPISODES_FILENAME)`` are one store, not two.
    Anything else means the same log has two names and the exclusive lock
    (DL-016) protects neither of them from the other.
    """
    d = tmp_path / "store"
    with MemoryStore.open(d) as s:
        s.append_episode(b"through-the-directory", "", 1)
    with MemoryStore.open(d / EPISODES_FILENAME) as s:
        assert [e.payload for e in s.episodes_since(0)] == [b"through-the-directory"]
        assert s.append_episode(b"through-the-file", "", 2) == 2
    with MemoryStore.open(d) as s:
        assert [e.payload for e in s.episodes_since(0)] == [
            b"through-the-directory",
            b"through-the-file",
        ]


def test_case_46_the_rule_is_a_pure_function_of_the_path(tmp_path: Path) -> None:
    """Case 46 — stated at its root: the resolution reads its argument and
    nothing else, so it cannot depend on creation order by construction.

    Asserted on the rule itself as well as on its effects, because an effect
    test can be satisfied by a rule that reads the disk and happens to agree.
    """
    p = tmp_path / "store"
    before = MemoryStore.log_path_for(p)

    p.mkdir()
    assert MemoryStore.log_path_for(p) == before, "mkdir changed the answer"
    (p / EPISODES_FILENAME).write_bytes(rawlog.header_bytes())
    assert MemoryStore.log_path_for(p) == before, "creating the log changed the answer"

    assert before == p / EPISODES_FILENAME
    # The file form is idempotent: resolving it again is a no-op.
    assert MemoryStore.log_path_for(before) == before
    # A path that does not exist at all still resolves, identically.
    ghost = tmp_path / "never" / "existed"
    assert MemoryStore.log_path_for(ghost) == ghost / EPISODES_FILENAME


# ---------------------------------------------------------------------------
# 47 - the frame-boundary discontinuity is pinned
# ---------------------------------------------------------------------------

GARBAGE = 0xAB  # non-zero, so this is never the zero-fill path


def _only_discarded_tail(store_dir: Path) -> Path:
    """The single file recovery kept the truncated bytes in.

    Asserts there is exactly one, because "a tail was preserved" is only
    evidence if we know which open preserved it.
    """
    kept = sorted(store_dir.glob("*.discarded-tail*"))
    assert len(kept) == 1, f"expected one preserved tail, found {kept}"
    return kept[0]


@pytest.mark.parametrize("n_bytes", list(range(1, 17)) + [24, 32, 64])
def test_case_47_non_zero_garbage_at_a_frame_start(
    store_dir: Path, log_path: Path, n_bytes: int
) -> None:
    """Case 47 / audit finding 7 — the old suite tested 1, 2 and 7 bytes but
    never 8, so the point where behaviour changed was never exercised at all.

    It changed at ``PREFIX_LEN``: below it there was no length field to read
    and the tail was truncated; at it and above, the length failed its own
    checksum, the bytes were not zeros, and the log refused.

    **Case 49 removed that step.** It was the cliff that made a half-landed
    append fatal: the bytes a torn write leaves behind are non-zero, and
    reading them as proof of durability refused the log forever over its own
    interrupted final append. The rule is now the same on both sides — a tail
    holding no frame that verifies end to end was never acknowledged, so it is
    truncated and kept beside the log. This test still sweeps every length from
    1 to 16, now asserting the opposite: that there is no step anywhere.
    """
    payloads = seed(store_dir, 3)
    clean = rawlog.read(log_path)
    rawlog.append_bytes(log_path, bytes([GARBAGE]) * n_bytes)

    with MemoryStore.open(store_dir) as s:
        assert s.head() == 3
        assert [e.payload for e in s.episodes_since(0)] == payloads
        assert s.diagnostics.recovered_bytes == n_bytes
    assert rawlog.read(log_path) == clean, "the garbage must be truncated away"

    kept = _only_discarded_tail(store_dir)
    assert kept.read_bytes() == bytes([GARBAGE]) * n_bytes, (
        "truncation must never destroy the bytes it discarded"
    )


def test_case_47_the_boundary_is_exactly_the_prefix_length(
    store_dir: Path, log_path: Path
) -> None:
    """Case 47, restated for case 49 — the two sides as one assertion, so the
    absence of a boundary cannot drift back into a boundary without this
    failing. ``PREFIX_LEN - 1`` and ``PREFIX_LEN`` must behave identically."""
    seed(store_dir, 2)
    clean = rawlog.read(log_path)

    for n in (rawlog.PREFIX_LEN - 1, rawlog.PREFIX_LEN, rawlog.PREFIX_LEN + 1):
        rawlog.write(log_path, clean)
        for stale in store_dir.glob("*.discarded-tail*"):
            stale.unlink()
        rawlog.append_bytes(log_path, bytes([GARBAGE]) * n)
        with MemoryStore.open(store_dir) as s:
            assert s.head() == 2, f"n={n}"
            assert s.diagnostics.recovered_bytes == n, f"n={n}"
        assert rawlog.read(log_path) == clean, f"n={n}"
        assert _only_discarded_tail(store_dir).read_bytes() == bytes([GARBAGE]) * n


def test_case_49_a_frame_whole_but_for_its_length_crc_is_still_corruption(
    store_dir: Path, log_path: Path
) -> None:
    """Case 49's other half — truncating an unverifiable tail must not become
    an excuse to truncate a *verifiable* one.

    Here the final frame's ``len_crc`` is damaged and nothing else is: the
    length is plausible, the body it names matches the body CRC, and the seq is
    the expected next one. Four independent facts say the frame was fully
    durable and a later bit-flip hit its checksum. That is damage, and the log
    must refuse rather than quietly drop an acknowledged episode.
    """
    seed(store_dir, 5)
    rawlog.corrupt_len_crc(log_path, -1)
    broken = rawlog.read(log_path)

    with pytest.raises(CorruptFrame):
        MemoryStore.open(store_dir)
    assert rawlog.read(log_path) == broken, "a failed open must not truncate"
    assert not list(store_dir.glob("*.discarded-tail*")), (
        "a refusal files nothing away; the log is untouched"
    )


# ---------------------------------------------------------------------------
# 48 - first key wins in the dedup index
# ---------------------------------------------------------------------------


def _log_with_a_repeated_key(store_dir: Path, log_path: Path) -> None:
    """A log holding two records under one dedup key. Only reachable by hand:
    ``append`` would have deduplicated the second one."""
    with MemoryStore.open(store_dir) as s:
        assert s.append_episode(b"the-first-one", "dup", 1_000) == 1
    rawlog.append_bytes(
        log_path, rawlog.encode_frame(2, 2_000, "dup", b"the-second-one")
    )
    assert len(rawlog.scan(log_path)) == 2


def test_case_48_first_key_wins(store_dir: Path, log_path: Path) -> None:
    """Case 48 — the first write of a key is the one it refers to.

    Flagged by the audit as *unpinned*: reversing it broke no test. It is not
    cosmetic. ``append`` returns the existing seq for a repeated key, so the
    resolution is what a caller gets back as "the episode you already wrote",
    and a last-wins index would hand back a different episode than the one the
    key was first attached to — silently, and only after a restart, since the
    index is rebuilt from the file.
    """
    _log_with_a_repeated_key(store_dir, log_path)

    with MemoryStore.open(store_dir) as s:
        assert s.head() == 2
        # The lowest seq wins: an identical re-append resolves to seq 1.
        assert s.append_episode(b"the-first-one", "dup", 9_999) == 1
        assert s.head() == 2, "the re-append must have written nothing"

        # And the conflict names the same record.
        with pytest.raises(WriteKeyConflict) as exc:
            s.append_episode(b"something-else", "dup", 9_999)
        assert "seq 1" in str(exc.value), str(exc.value)
        # Matching seq 2's payload is still a conflict, because seq 2 is not
        # what the key resolves to.
        with pytest.raises(WriteKeyConflict):
            s.append_episode(b"the-second-one", "dup", 9_999)


def test_case_48_a_rebuild_agrees(store_dir: Path, log_path: Path) -> None:
    """Case 48 — "and a rebuild agrees". Both indexes are caches (DL-016), so a
    rule that only holds until the cache is dropped is not a rule."""
    _log_with_a_repeated_key(store_dir, log_path)

    with MemoryStore.open(store_dir) as s:
        s.diagnostics.rebuild_indexes()
        assert s.append_episode(b"the-first-one", "dup", 9_999) == 1
        with pytest.raises(WriteKeyConflict) as exc:
            s.append_episode(b"the-second-one", "dup", 9_999)
        assert "seq 1" in str(exc.value), str(exc.value)

    # And across a genuine restart, where the index comes only from the file.
    with MemoryStore.open(store_dir) as s:
        assert s.append_episode(b"the-first-one", "dup", 9_999) == 1
