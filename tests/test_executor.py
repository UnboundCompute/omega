"""The drain and the startup report — M1_SPEC.md §1.1, §1.2, §1.6.

In-process cases. The real ``kill -9`` lives in ``test_restart.py``; these fix
the behaviour that one then has to survive.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import invariants
from omega import episodes, provider
from omega.executor import INTERRUPTED_ERROR, Executor, NotRecovered
from omega.memory import MemoryStore
from omega.queue import EventQueue
from omega.turn import ActResult, TurnContext, run_turn

AT = "2026-09-23T12:00:00+00:00"


@pytest.fixture
def q(store: MemoryStore) -> EventQueue:
    return EventQueue(store)


def speaking() -> provider.FakeProvider:
    return provider.FakeProvider(
        {
            provider.JUDGE: lambda role, messages: "SPEAK",
            provider.ACT: lambda role, messages: "ok",
        }
    )


def silent() -> provider.FakeProvider:
    return provider.FakeProvider({provider.JUDGE: lambda role, messages: "SILENT"})


def send(q: EventQueue, text: str) -> int:
    return q.append(episodes.inbound(text, channel="tray", at=AT))


def terminals(q: EventQueue) -> list[dict]:
    return [p.payload for p in q.recent(q.head()) if episodes.is_terminal(p.payload)]


# --- green: draining --------------------------------------------------------


def test_an_empty_queue_drains_to_nothing_and_touches_nothing(q: EventQueue) -> None:
    ex = Executor(q, complete=speaking().complete)
    assert ex.recover().clean is True
    assert ex.drain() == []
    assert q.head() == 0 and q.claimed() == 0 and q.done() == 0
    assert ex.step() is None


def test_the_drain_runs_events_in_order_and_advances_both_cursors(
    q: EventQueue,
) -> None:
    for i in range(3):
        send(q, f"m{i}")
    fp = speaking()
    ex = Executor(q, complete=fp.complete)
    ex.recover()

    results = ex.drain()

    assert [r.seq for r in results] == [1, 2, 3]
    assert all(r.outcome == "spoke" for r in results)
    assert q.claimed() == q.done() == q.head()
    assert len(terminals(q)) == 3
    assert invariants.check(q), str(invariants.check(q))


def test_the_drain_advances_past_its_own_records_without_looping(
    q: EventQueue,
) -> None:
    """§1.1's sharp edge. The turn's own ``turn.completed`` lands in the same
    log, so a drain that did not skip records would answer its own output
    forever — this test would hang rather than fail."""
    send(q, "hello")
    ex = Executor(q, complete=speaking().complete)
    ex.recover()

    first = ex.drain()
    assert len(first) == 1
    # The record written by that turn is now the only thing behind the cursor
    # on the next pass, and it produces no turn.
    assert ex.drain() == []
    assert q.claimed() == q.done() == q.head()


def test_a_log_of_records_only_produces_no_turns_and_still_drains(
    q: EventQueue,
) -> None:
    """The pathological queue: nothing in it is an event. It must terminate."""
    seq = send(q, "hello")
    q.append(episodes.tool_called(for_seq=seq, tool="shell", args={}, at=AT))
    q.append(
        episodes.tool_returned(for_seq=seq, tool="shell", ok=True, result="ok", at=AT)
    )
    q.claim(seq)
    q.append(episodes.completed(for_seq=seq, outcome="silent", at=AT),
             episodes.turn_write_key(seq))
    q.finish(seq)

    ex = Executor(q, complete=speaking().complete)
    ex.recover()
    assert ex.drain() == []
    assert q.claimed() == q.done() == q.head()


def test_step_handles_one_episode_at_a_time(q: EventQueue) -> None:
    send(q, "one")
    send(q, "two")
    ex = Executor(q, complete=speaking().complete)
    ex.recover()

    first = ex.step()
    assert first is not None and first.seq == 1
    assert q.done() == 1

    # Both events were already waiting, so the second one is next in line —
    # the records those turns wrote sit behind them.
    second = ex.step()
    assert second is not None and second.seq == 2
    assert q.done() == 2

    # Now the two records. Each is skipped, not run, and step() says so by
    # returning no result while still moving the cursor.
    assert ex.step() is None
    assert q.done() == 3
    assert ex.step() is None
    assert q.done() == 4
    assert ex.step() is None
    assert q.done() == q.head() == 4


def test_silence_drains_exactly_like_speech(q: EventQueue) -> None:
    """DL-011 — a silent turn advances DONE like a speaking one. If it did not,
    one quiet event would stall the queue forever."""
    send(q, "fyi")
    ex = Executor(q, complete=silent().complete)
    ex.recover()

    results = ex.drain()
    assert [r.outcome for r in results] == ["silent"]
    assert q.claimed() == q.done() == q.head()
    assert terminals(q)[0]["outcome"] == "silent"
    assert invariants.check(q)


def test_a_failed_turn_drains_too_and_leaves_the_queue_moving(q: EventQueue) -> None:
    """A provider outage must not wedge the single consumer."""
    send(q, "first")
    send(q, "second")
    ex = Executor(q, complete=provider.FakeProvider({}).complete)
    ex.recover()

    results = ex.drain()
    assert [r.outcome for r in results] == ["failed", "failed"]
    assert q.claimed() == q.done() == q.head()
    assert invariants.check(q)


def test_an_event_arriving_mid_turn_is_drained_in_the_same_pass(
    q: EventQueue,
) -> None:
    """The listener appends while the executor is inside a turn. It must not
    sit behind an idle drain waiting for something else to wake it."""
    send(q, "first")
    arrived: list[int] = []

    def judge_then_interrupt(role: str, messages: list) -> str:
        if role == provider.JUDGE and not arrived:
            arrived.append(send(q, "arrived mid-turn"))
        return "SPEAK" if role == provider.JUDGE else "ok"

    fp = provider.FakeProvider(
        {provider.JUDGE: judge_then_interrupt, provider.ACT: judge_then_interrupt}
    )
    ex = Executor(q, complete=fp.complete)
    ex.recover()

    results = ex.drain()

    assert len(arrived) == 1
    assert [r.seq for r in results] == [1, arrived[0]]
    assert q.claimed() == q.done() == q.head()
    assert invariants.check(q)


def test_max_turns_stops_early_and_leaves_the_rest_pending(q: EventQueue) -> None:
    for i in range(3):
        send(q, f"m{i}")
    ex = Executor(q, complete=speaking().complete)
    ex.recover()

    results = ex.drain(max_turns=2)
    assert len(results) == 2
    assert q.claimed() == q.done()
    assert [p.seq for p in q.pending()], "the third event must still be waiting"
    assert invariants.check(q)


def test_the_act_step_given_to_the_executor_is_the_one_the_turn_runs(
    q: EventQueue,
) -> None:
    send(q, "do it")
    seen: list[int] = []

    def act(ctx: TurnContext) -> ActResult:
        seen.append(ctx.seq)
        return ActResult(tools=("shell",))

    fp = provider.FakeProvider(
        {provider.JUDGE: "ACT", provider.ACT: lambda role, messages: "done"}
    )
    ex = Executor(q, complete=fp.complete, act=act)
    ex.recover()

    results = ex.drain()
    assert seen == [1]
    assert results[0].tools == ("shell",)


# --- red: draining before recovery -----------------------------------------


def test_draining_before_recovering_is_refused(q: EventQueue) -> None:
    """Fail closed. Draining while a turn is in flight would leave it with no
    record at all — the one shape the restart report cannot distinguish from a
    crash."""
    send(q, "hello")
    ex = Executor(q, complete=speaking().complete)

    with pytest.raises(NotRecovered):
        ex.drain()
    with pytest.raises(NotRecovered):
        ex.step()
    assert q.claimed() == 0


# --- the startup report -----------------------------------------------------


def test_a_clean_stop_reports_clean_and_says_nothing(q: EventQueue) -> None:
    send(q, "hello")
    ex = Executor(q, complete=speaking().complete)
    ex.recover()
    ex.drain()

    again = Executor(q, complete=speaking().complete)
    report = again.recover()
    assert report.clean is True
    assert report.interrupted is None
    assert report.lines() == []


def test_an_interrupted_turn_is_recorded_once_and_never_re_run(
    q: EventQueue,
) -> None:
    """§1.2 — the heart of the restart test, in process.

    The turn is claimed and then the process 'dies'. On restart the executor
    closes it out with one terminal record, advances DONE, and **does not call
    the model again**: the decision to pick the work up is the user's.
    """
    seq = send(q, "book the flight")
    q.claim(seq)  # claimed, then killed: nothing else ran

    fp = speaking()
    ex = Executor(q, complete=fp.complete)
    report = ex.recover()

    assert report.interrupted is not None
    assert report.interrupted.seq == seq
    assert report.interrupted.finished_before_crash is False
    assert "book the flight" in report.interrupted.question
    assert "pick it up" in report.interrupted.question
    assert report.interrupted.question in report.lines()

    assert fp.calls == [], "recovery must not re-run the turn"
    records = terminals(q)
    assert len(records) == 1
    assert records[0]["for_seq"] == seq
    assert records[0]["outcome"] == "failed"
    assert records[0]["error"] == INTERRUPTED_ERROR
    assert q.claimed() == q.done() == seq
    assert invariants.check(q), str(invariants.check(q))


def test_a_turn_that_finished_before_the_crash_is_reported_as_such(
    q: EventQueue,
) -> None:
    """The safe direction of the error (§1.2). ``turn.completed`` is appended
    *before* DONE advances, so a crash between them over-reports. The write key
    is what lets recovery tell the two apart: the log refuses the second
    record, and the refusal is the answer."""
    seq = send(q, "what's open?")
    q.claim(seq)
    result = run_turn(q, q.at(seq), complete=speaking().complete, at=AT)
    # ... and the crash lands here, before finish().

    fp = speaking()
    ex = Executor(q, complete=fp.complete)
    report = ex.recover()

    assert report.interrupted is not None
    assert report.interrupted.finished_before_crash is True
    assert report.interrupted.record_seq == result.record_seq
    assert "Nothing was lost" in report.interrupted.question
    assert fp.calls == []

    records = terminals(q)
    assert len(records) == 1, "recovery must not add a second terminal record"
    assert records[0]["outcome"] == "spoke"
    assert records[0]["reply"] == "ok"
    assert q.claimed() == q.done() == seq
    assert invariants.check(q)


def test_a_half_skipped_record_is_released_without_inventing_a_turn(
    q: EventQueue,
) -> None:
    """skip() moves both cursors; this is the window between them. There was
    never a turn here, so there is nothing to record and nothing to ask."""
    seq = send(q, "hello")
    q.claim(seq)
    q.append(episodes.completed(for_seq=seq, outcome="silent", at=AT),
             episodes.turn_write_key(seq))
    q.finish(seq)
    record_seq = q.head()
    q.claim(record_seq)  # CLAIMED moved, then the crash, before DONE followed

    ex = Executor(q, complete=speaking().complete)
    report = ex.recover()

    assert report.interrupted is None
    assert report.released_record_seq == record_seq
    assert report.clean is False
    assert q.claimed() == q.done() == record_seq
    assert len(terminals(q)) == 1


def test_recovery_then_drain_processes_only_what_was_never_run(
    q: EventQueue,
) -> None:
    interrupted = send(q, "the cut-off one")
    q.claim(interrupted)
    later = send(q, "the one that arrived after")

    fp = speaking()
    ex = Executor(q, complete=fp.complete)
    ex.recover()
    results = ex.drain()

    assert [r.seq for r in results] == [later]
    assert len(fp.calls_for(provider.JUDGE)) == 1
    records = {t["for_seq"]: t for t in terminals(q)}
    assert records[interrupted]["error"] == INTERRUPTED_ERROR
    assert records[later]["outcome"] == "spoke"
    assert invariants.check(q)


def test_recover_is_safe_to_call_twice(q: EventQueue) -> None:
    seq = send(q, "hello")
    q.claim(seq)
    ex = Executor(q, complete=speaking().complete)

    first = ex.recover()
    second = ex.recover()

    assert first.interrupted is not None
    assert second.interrupted is None and second.clean is True
    assert len(terminals(q)) == 1


# --- yellow: the sidecar we are allowed to lose -----------------------------


def test_lost_cursors_are_rebuilt_from_the_log_rather_than_replayed(
    store_dir: Path, log_path: Path
) -> None:
    """DL-021/Q9 says a damaged sidecar must not brick the log, so every
    checkpoint reads 0. Read naively that means *re-deliver everything*, which
    would re-run turns that already completed. The cursors are derived state and
    the log is the source of truth (DL-017), so they are re-derived instead.
    """
    with MemoryStore.open(store_dir) as s:
        q = EventQueue(s)
        ex = Executor(q, complete=speaking().complete)
        ex.recover()
        send(q, "first")
        send(q, "second")
        ex.drain()
        settled = q.done()
        assert settled == q.head()

    Path(str(log_path) + ".checkpoints").write_bytes(b"garbage")

    with MemoryStore.open(store_dir) as s:
        q = EventQueue(s)
        assert q.claimed() == 0 and q.done() == 0, "the sidecar should have reset"
        fp = speaking()
        ex = Executor(q, complete=fp.complete)
        send(q, "third, never handled")
        report = ex.recover()

        assert report.checkpoints_rebuilt is True
        assert report.lines(), "a rebuilt cursor is worth saying out loud"
        assert q.done() == settled

        results = ex.drain()
        assert len(results) == 1, "only the unhandled event may run"
        assert len(fp.calls_for(provider.JUDGE)) == 1
        assert invariants.check(q)
        assert len(terminals(q)) == 3


def test_the_violation_checker_is_not_a_no_op(q: EventQueue) -> None:
    """Its control. Every other case in this file asserts the checker passes,
    which is worth nothing unless the checker can fail — and unless an empty
    log reports *undetermined* rather than pass."""
    empty = invariants.check(q)
    assert empty.verdict == invariants.UNDETERMINED
    assert not empty, "undetermined must never read as a pass"

    # Two terminal records for one turn: the double-write the write key exists
    # to stop, forced past it with a second key.
    seq = send(q, "hello")
    q.claim(seq)
    q.append(episodes.completed(for_seq=seq, outcome="silent", at=AT),
             episodes.turn_write_key(seq))
    q.append(
        episodes.completed(for_seq=seq, outcome="spoke", reply="twice", at=AT),
        "some-other-key",
    )
    q.finish(seq)

    bad = invariants.check(q)
    assert bad.verdict == invariants.VIOLATED
    assert any("terminal records" in r for r in bad.reasons)


def test_the_violation_checker_catches_two_turns_in_flight(q: EventQueue) -> None:
    send(q, "one")
    send(q, "two")
    send(q, "three")
    q.claim(3)
    q.finish(1)

    bad = invariants.check(q)
    assert bad.verdict == invariants.VIOLATED
    assert any("in flight" in r for r in bad.reasons)
