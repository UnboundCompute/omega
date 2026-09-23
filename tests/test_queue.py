"""The queue is the log — M1_SPEC.md §1.1, §1.2.

Green, red and yellow. The load-bearing claims under test are that the cursors
are durable (so a restart loses nothing but the in-flight turn), that they only
move forwards (so at-most-once holds), and that the drain can tell an event it
must process from a record it must advance past (so the executor does not answer
its own output forever).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omega import episodes
from omega.memory import (
    CheckpointAhead,
    MemoryStore,
    WriteKeyConflict,
)
from omega.queue import (
    CLAIMED,
    DONE,
    EVENT_KINDS,
    RECORD_KINDS,
    CursorWentBackwards,
    EventQueue,
)


def inbound(text: str = "hello") -> dict:
    return episodes.inbound(text, channel="tray", at="2026-09-23T00:00:00+00:00")


def completed(for_seq: int) -> dict:
    return episodes.completed(
        for_seq=for_seq,
        outcome="spoke",
        reply="hi",
        at="2026-09-23T00:00:01+00:00",
    )


@pytest.fixture
def q(store: MemoryStore) -> EventQueue:
    return EventQueue(store)


# --- green ------------------------------------------------------------------


def test_the_kinds_partition_and_are_not_empty() -> None:
    """§1.1 — every kind is either an event or a record, and both sets have
    members. A partition assertion over two empty sets would pass on empty."""
    assert EVENT_KINDS
    assert RECORD_KINDS
    assert EVENT_KINDS | RECORD_KINDS == frozenset(episodes.KINDS)
    assert not (EVENT_KINDS & RECORD_KINDS)

    # Named individually, because the *membership* is the design, not the
    # partition: work.finished is an event (a detached sub-loop re-entering,
    # §1.5) and every turn record is not.
    assert episodes.MESSAGE_INBOUND in EVENT_KINDS
    assert episodes.WORK_FINISHED in EVENT_KINDS
    for record in (
        episodes.TURN_COMPLETED,
        episodes.TURN_BLOCKED,
        episodes.TOOL_CALLED,
        episodes.TOOL_RETURNED,
    ):
        assert record in RECORD_KINDS


def test_append_returns_a_durable_seq_and_pending_decodes_it(q: EventQueue) -> None:
    seq = q.append(inbound("first"))
    assert seq == 1

    waiting = list(q.pending())
    assert len(waiting) == 1
    assert waiting[0].seq == 1
    assert waiting[0].kind == episodes.MESSAGE_INBOUND
    assert waiting[0].payload["text"] == "first"
    assert waiting[0].is_event is True


def test_claim_then_finish_moves_both_cursors_in_order(q: EventQueue) -> None:
    seq = q.append(inbound())
    assert q.claimed() == 0 and q.done() == 0
    assert q.in_flight() is None

    q.claim(seq)
    assert q.claimed() == seq
    assert q.done() == 0
    # Claim-before-act: between these two lines a kill -9 is a reported
    # interruption, not a replay.
    assert q.in_flight() == seq

    q.finish(seq)
    assert q.done() == seq
    assert q.in_flight() is None
    assert list(q.pending()) == []


def test_skip_advances_past_a_record_the_drain_does_not_process(q: EventQueue) -> None:
    """§1.1 — without this the executor answers its own output forever."""
    event = q.append(inbound())
    q.claim(event)
    record = q.append(completed(event), episodes.turn_write_key(event))
    q.finish(event)

    waiting = list(q.pending())
    assert [p.seq for p in waiting] == [record]
    assert waiting[0].is_event is False

    q.skip(record)
    assert q.claimed() == record and q.done() == record
    assert list(q.pending()) == []


def test_pending_is_empty_on_an_empty_queue_and_says_so(q: EventQueue) -> None:
    """An empty queue is not an error and not a failure — it is the resting
    state of a resident process."""
    assert q.head() == 0
    assert list(q.pending()) == []
    assert q.in_flight() is None
    assert q.claimed() == 0 and q.done() == 0


def test_pending_yields_in_sequence_order_from_the_cursor(q: EventQueue) -> None:
    seqs = [q.append(inbound(f"m{i}")) for i in range(5)]
    assert seqs == [1, 2, 3, 4, 5]

    q.claim(2)
    q.finish(2)
    assert [p.seq for p in q.pending()] == [3, 4, 5]
    assert [p.payload["text"] for p in q.pending()] == ["m2", "m3", "m4"]


def test_recent_returns_the_last_n_oldest_first(q: EventQueue) -> None:
    for i in range(6):
        q.append(inbound(f"m{i}"))

    assert [p.payload["text"] for p in q.recent(3)] == ["m3", "m4", "m5"]
    assert [p.payload["text"] for p in q.recent(100)] == [f"m{i}" for i in range(6)]
    assert q.recent(0) == []


def test_recent_before_excludes_the_episode_it_is_given(q: EventQueue) -> None:
    """Recall is history *plus* the new event, never the new event twice."""
    for i in range(5):
        q.append(inbound(f"m{i}"))

    assert [p.seq for p in q.recent(10, before=3)] == [1, 2]
    assert q.recent(10, before=1) == []


def test_at_returns_the_named_episode_decoded(q: EventQueue) -> None:
    q.append(inbound("one"))
    seq = q.append(inbound("two"))
    q.append(inbound("three"))

    found = q.at(seq)
    assert found.seq == seq
    assert found.payload["text"] == "two"


def test_the_returned_seq_is_the_acknowledgement_token(store_dir: Path) -> None:
    """§Q10 requirement 1 — acknowledged once, and the ack is durable before
    ``append`` returns. Asserted by reopening rather than by trusting the call."""
    with MemoryStore.open(store_dir) as s:
        seq = EventQueue(s).append(inbound("durable"))

    with MemoryStore.open(store_dir) as s:
        again = EventQueue(s)
        assert again.head() == seq
        assert again.at(seq).payload["text"] == "durable"


# --- red --------------------------------------------------------------------


def test_a_cursor_may_not_move_backwards(q: EventQueue) -> None:
    """At-most-once dies the moment a cursor can rewind, so the queue refuses —
    M0 permits it, because there a checkpoint is just a number."""
    for _ in range(3):
        q.append(inbound())
    q.claim(3)
    q.finish(3)

    with pytest.raises(CursorWentBackwards):
        q.claim(2)
    with pytest.raises(CursorWentBackwards):
        q.finish(1)
    with pytest.raises(CursorWentBackwards):
        q.skip(2)

    assert q.claimed() == 3 and q.done() == 3


def test_done_may_not_pass_claimed(q: EventQueue) -> None:
    """DONE ahead of CLAIMED means a turn was released that was never taken —
    the cursor pair would then report "nothing in flight" during a live turn."""
    q.append(inbound())
    q.append(inbound())
    q.claim(1)

    with pytest.raises(CursorWentBackwards):
        q.finish(2)
    assert q.done() == 0


def test_claiming_past_the_head_is_an_error_not_a_shrug(q: EventQueue) -> None:
    """M0's CheckpointAhead is a liveness property of the loop now (§1.1), so
    it must surface rather than be swallowed into "nothing to do"."""
    q.append(inbound())
    with pytest.raises(CheckpointAhead):
        q.claim(2)
    assert q.claimed() == 0


def test_an_invalid_payload_never_reaches_the_log(q: EventQueue) -> None:
    with pytest.raises(episodes.BadPayload):
        q.append({"v": 1, "kind": "not.a.kind"})
    assert q.head() == 0
    assert list(q.pending()) == []


def test_at_refuses_a_seq_that_is_not_an_episode(q: EventQueue) -> None:
    q.append(inbound())
    with pytest.raises(ValueError):
        q.at(0)
    with pytest.raises(LookupError):
        q.at(2)


def test_a_double_delivered_event_becomes_one_episode(q: EventQueue) -> None:
    """The same event offered twice under one write key is one queue entry —
    M0's dedup, used as the loop's protection against a retrying client."""
    payload = inbound("say it once")
    first = q.append(payload, "tray:msg-7")
    second = q.append(payload, "tray:msg-7")

    assert first == second == 1
    assert q.head() == 1
    assert [p.seq for p in q.pending()] == [1]


def test_the_same_key_with_different_content_is_loud(q: EventQueue) -> None:
    q.append(inbound("original"), "tray:msg-7")
    with pytest.raises(WriteKeyConflict):
        q.append(inbound("different"), "tray:msg-7")
    assert q.head() == 1


def test_a_terminal_record_cannot_be_written_twice_for_one_turn(q: EventQueue) -> None:
    """§2.1 — the write key makes the double-write a *log* error, not a
    convention someone has to remember."""
    event = q.append(inbound())
    key = episodes.turn_write_key(event)
    q.append(completed(event), key)

    with pytest.raises(WriteKeyConflict):
        q.append(
            episodes.completed(
                for_seq=event,
                outcome="silent",
                at="2026-09-23T00:00:09+00:00",
            ),
            key,
        )
    assert q.head() == 2


# --- yellow -----------------------------------------------------------------


def test_the_cursors_survive_a_close_and_reopen(store_dir: Path) -> None:
    """The queue is durable because the cursors are — this is the whole reason
    the log is the queue rather than a list in RAM (DL-016)."""
    with MemoryStore.open(store_dir) as s:
        q = EventQueue(s)
        for _ in range(3):
            q.append(inbound())
        q.claim(2)
        q.finish(1)

    with MemoryStore.open(store_dir) as s:
        q = EventQueue(s)
        assert q.claimed() == 2
        assert q.done() == 1
        assert q.in_flight() == 2
        assert [p.seq for p in q.pending()] == [3]
        assert sorted(s.checkpoint_names()) == sorted([CLAIMED, DONE])


def test_a_reset_checkpoint_sidecar_redelivers_rather_than_bricking(
    store_dir: Path, log_path: Path
) -> None:
    """The honest limit of at-most-once, stated rather than assumed.

    DL-021/Q9: a damaged sidecar is set aside and every checkpoint reads 0, so
    the log still opens — derived state never holds acknowledged episodes
    hostage. The cost is that the queue re-delivers, and the store *says so*
    through ``checkpoints_reset`` rather than leaving it to be discovered.
    """
    with MemoryStore.open(store_dir) as s:
        q = EventQueue(s)
        for _ in range(3):
            q.append(inbound())
        q.claim(3)
        q.finish(3)

    sidecar = Path(str(log_path) + ".checkpoints")
    assert sidecar.exists(), "the sidecar should exist after setting a checkpoint"
    sidecar.write_bytes(b"not a sidecar")

    with MemoryStore.open(store_dir) as s:
        assert s.checkpoints_reset is True
        q = EventQueue(s)
        assert q.claimed() == 0 and q.done() == 0
        assert [p.seq for p in q.pending()] == [1, 2, 3]


def test_claiming_the_same_seq_twice_is_a_no_op_not_a_rewind(q: EventQueue) -> None:
    """Idempotent forwards, refused backwards: a restart that re-claims the
    episode it already holds must not look like a cursor bug."""
    q.append(inbound())
    q.claim(1)
    q.claim(1)
    assert q.claimed() == 1


def test_an_event_appended_during_a_claimed_turn_waits_its_turn(q: EventQueue) -> None:
    """Single consumer: arriving mid-turn changes nothing about ordering."""
    first = q.append(inbound("first"))
    q.claim(first)

    later = q.append(inbound("arrived mid-turn"))
    assert q.in_flight() == first
    # The new event is behind the cursor's claim, not ahead of it.
    assert [p.seq for p in q.pending()] == [later]

    record = q.append(completed(first), episodes.turn_write_key(first))
    q.finish(first)
    assert q.in_flight() is None

    # The mid-turn event is still ahead of the turn's own record, because the
    # log orders by arrival and nothing jumps the queue.
    waiting = list(q.pending())
    assert [p.seq for p in waiting] == [later, record]
    assert [p.kind for p in waiting] == [
        episodes.MESSAGE_INBOUND,
        episodes.TURN_COMPLETED,
    ]
    assert [p.is_event for p in waiting] == [True, False]
