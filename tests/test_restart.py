"""kill -9 through the turn loop — M1_SPEC.md §1.2, §1.1.

M0 proved the *log* survives a kill. This proves the *loop* does: after a kill
at any moment, the next start reconciles the queue and says what was lost.

The violation metric is the one in `invariants.py` — every settled event has
exactly one terminal record, at most one turn was ever in flight, and the only
unrecorded claimed event is the interrupted one. It is three-valued, so a trial
in which nothing was processed reports *couldn't determine* and is not counted
as a pass.
"""

from __future__ import annotations

import os
import random
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

import invariants
from omega import episodes, provider
from omega.executor import INTERRUPTED_ERROR, Executor
from omega.memory import MemoryStore
from omega.queue import EventQueue

REPO_ROOT = Path(__file__).resolve().parents[1]
CHILD = Path(__file__).resolve().parent / "loop_child.py"

#: Reliability is pass^k. Passing once is not passing.
TRIALS = 20

APPEND_RE = re.compile(r"^APPEND (\d+)$")
CLAIMED_RE = re.compile(r"^CLAIMED (\d+) (\d+)$")
TURN_RE = re.compile(r"^TURN (\d+) (\w+)$")

#: What each churn trial actually concluded. Asserted at the end, from the
#: trials that ran — never from the constant, which would pass on empty.
CHURN: list[dict] = []


def _spawn(*args: str) -> subprocess.Popen:
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    return subprocess.Popen(
        [sys.executable, str(CHILD), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        cwd=str(REPO_ROOT),
        env=env,
    )


def _kill9(proc: subprocess.Popen) -> int:
    os.kill(proc.pid, signal.SIGKILL)
    proc.wait(timeout=30)
    return proc.returncode


def _fake() -> provider.FakeProvider:
    """The restarted process must not need a model to recover. If recovery ever
    calls one, this raises rather than quietly re-running the turn."""
    def refuse(role: str, messages: list) -> str:
        raise AssertionError(f"recovery called the model for role {role!r}")

    return provider.FakeProvider({provider.JUDGE: refuse, provider.ACT: refuse})


@pytest.mark.parametrize("trial", range(TRIALS))
def test_kill9_mid_turn_is_recovered_as_information_not_replayed(
    store_dir: Path, trial: int
) -> None:
    """The deterministic case: killed with the turn claimed and no record.

    The child parks inside the judge step and prints its cursors first, so the
    kill is *known* to land after CLAIMED moved and before any record was
    appended — the exact window §1.2 is about. On restart the executor must:
    file one terminal record for that turn, advance DONE, ask the user whether
    to pick the work up, and **not** run the turn again.
    """
    proc = _spawn("block", str(store_dir))
    try:
        appended = APPEND_RE.match(proc.stdout.readline().strip())
        assert appended, f"child never enqueued: {proc.stderr.read()!r}"
        seq = int(appended.group(1))

        claimed_line = CLAIMED_RE.match(proc.stdout.readline().strip())
        assert claimed_line, f"child never reached the turn: {proc.stderr.read()!r}"
        assert (int(claimed_line.group(1)), int(claimed_line.group(2))) == (seq, 0)

        assert _kill9(proc) == -signal.SIGKILL
    finally:
        if proc.poll() is None:
            _kill9(proc)
        proc.stdout.close()
        proc.stderr.close()

    # --- the state the dead process left behind --------------------------
    with MemoryStore.open(store_dir) as store:
        queue = EventQueue(store)
        assert queue.claimed() == seq and queue.done() == 0
        assert queue.in_flight() == seq
        before = invariants.check(queue)
        assert before.verdict == invariants.UNDETERMINED, (
            "an in-flight turn with nothing settled is not yet gradeable; "
            f"got {before}"
        )
        assert not before, "undetermined must never read as a pass"

    # --- the restart ------------------------------------------------------
    with MemoryStore.open(store_dir) as store:
        queue = EventQueue(store)
        executor = Executor(queue, complete=_fake().complete)
        report = executor.recover()

        assert report.clean is False
        assert report.interrupted is not None
        assert report.interrupted.seq == seq
        assert report.interrupted.finished_before_crash is False
        assert "book the flight to Lisbon" in report.interrupted.question
        assert "pick it up" in report.interrupted.question

        records = [
            p.payload
            for p in queue.recent(queue.head())
            if episodes.is_terminal(p.payload)
        ]
        assert len(records) == 1, f"expected one terminal record, got {records}"
        assert records[0]["for_seq"] == seq
        assert records[0]["outcome"] == "failed"
        assert records[0]["error"] == INTERRUPTED_ERROR
        assert records[0]["reply"] is None, "a cut-off turn did not speak"

        after = invariants.check(queue)
        assert after.verdict == invariants.PASS, str(after)
        assert queue.claimed() == queue.done() == seq

        # The queue moves again, and the dead turn is not re-delivered.
        assert executor.drain() == []
        assert list(queue.pending()) == []


def test_a_restart_never_re_runs_the_work(store_dir: Path) -> None:
    """The same window, with the side effect made visible.

    at-most-once is the whole reason the interrupted turn is reported rather
    than replayed: for a turn that had already sent an email, re-delivery is how
    one action becomes two. This asserts the model is not called a second time
    for that event even after a full drain.
    """
    proc = _spawn("block", str(store_dir))
    try:
        seq = int(APPEND_RE.match(proc.stdout.readline().strip()).group(1))
        assert CLAIMED_RE.match(proc.stdout.readline().strip())
        _kill9(proc)
    finally:
        if proc.poll() is None:
            _kill9(proc)
        proc.stdout.close()
        proc.stderr.close()

    calls: list[str] = []

    def count(role: str, messages: list) -> str:
        calls.append(role)
        return "SPEAK" if role == provider.JUDGE else "ok"

    fake = provider.FakeProvider({provider.JUDGE: count, provider.ACT: count})
    with MemoryStore.open(store_dir) as store:
        queue = EventQueue(store)
        executor = Executor(queue, complete=fake.complete)
        executor.recover()
        results = executor.drain()

    assert calls == [], f"the interrupted turn was re-run: {calls}"
    assert results == []
    assert seq >= 1


@pytest.mark.parametrize("trial", range(TRIALS))
def test_kill9_at_a_random_moment_leaves_a_gradeable_queue(
    store_dir: Path, trial: int
) -> None:
    """The randomized case: the kill lands wherever it lands.

    Between the first append and the last DONE there are five distinct windows
    — mid-append, after claim, after the record but before DONE, between turns,
    and idle — and a fixed delay would only ever probe one of them. Each trial
    kills at a random offset and then grades the reopened queue against the
    violation metric.
    """
    proc = _spawn("churn", str(store_dir), str(1000 + trial))
    try:
        first = proc.stdout.readline().strip()
        assert APPEND_RE.match(first), (
            f"child never got going: {first!r} {proc.stderr.read()!r}"
        )
        time.sleep(random.uniform(0.005, 0.120))
        assert proc.poll() is None, f"child died early: {proc.stderr.read()}"
        assert _kill9(proc) == -signal.SIGKILL
        stdout = first + "\n" + proc.stdout.read()
        stderr = proc.stderr.read()
    finally:
        if proc.poll() is None:
            _kill9(proc)
        proc.stdout.close()
        proc.stderr.close()

    acked_turns = [
        (int(m.group(1)), m.group(2))
        for m in (TURN_RE.match(line.strip()) for line in stdout.splitlines())
        if m
    ]
    acked_appends = [
        int(m.group(1))
        for m in (APPEND_RE.match(line.strip()) for line in stdout.splitlines())
        if m
    ]
    assert acked_appends, f"nothing was acknowledged: {stderr!r}"

    with MemoryStore.open(store_dir) as store:
        queue = EventQueue(store)

        # Every acknowledged append is in the log: M0's promise, still holding
        # now that the loop is the thing doing the appending.
        assert queue.head() >= max(acked_appends)

        # Every acknowledged turn is settled and has its record.
        for seq, outcome in acked_turns:
            assert queue.done() >= seq, (
                f"turn {seq} was acknowledged as finished but done={queue.done()}"
            )
        recorded = {
            int(p.payload["for_seq"]): p.payload
            for p in queue.recent(queue.head())
            if episodes.is_terminal(p.payload)
        }
        for seq, outcome in acked_turns:
            assert recorded[seq]["outcome"] == outcome

        executor = Executor(queue, complete=_fake().complete)
        report = executor.recover()
        graded = invariants.check(queue)

        assert graded.verdict != invariants.VIOLATED, str(graded)
        assert queue.claimed() == queue.done()

    CHURN.append(
        {
            "trial": trial,
            "verdict": graded.verdict,
            "turns": len(acked_turns),
            "events": graded.events,
            "settled": graded.done,
            "interrupted": report.interrupted is not None,
            "released": report.released_record_seq is not None,
        }
    )


def test_the_randomized_trials_actually_determined_something() -> None:
    """The control for the case above. A crash suite that quietly killed the
    child during interpreter start-up every time would otherwise report 20/20
    while proving nothing — M0's own suite hit exactly that, 2 trials in 20.

    So: count the trials that *ran*, require that most of them reached a real
    verdict, and require that the restart window was genuinely hit — if no trial
    ever landed mid-turn, the recovery path was never under test and this file
    is measuring the idle case twenty times.
    """
    assert len(CHURN) == TRIALS, f"only {len(CHURN)}/{TRIALS} trials ran"

    determined = [t for t in CHURN if t["verdict"] == invariants.PASS]
    undetermined = [t for t in CHURN if t["verdict"] == invariants.UNDETERMINED]
    assert not [t for t in CHURN if t["verdict"] == invariants.VIOLATED]
    assert len(determined) + len(undetermined) == TRIALS

    assert len(determined) >= TRIALS * 3 // 4, (
        f"only {len(determined)}/{TRIALS} trials graded pass; "
        f"{len(undetermined)} could not be determined: {undetermined}"
    )
    # A pass must have had something to grade: at least one event in the log
    # and a DONE cursor that moved. Anything else is the checker passing on
    # empty, which is the failure mode this whole file is guarding against.
    for trial in determined:
        assert trial["events"] >= 1 and trial["settled"] >= 1, trial

    # And across the set, turns really did run to completion under the kill
    # timings — not merely appends. The bar is deliberately low because the
    # kill offset is random by design; a zero here would mean every trial died
    # before its first turn ever finished.
    assert sum(t["turns"] for t in CHURN) >= TRIALS // 4, (
        f"only {sum(t['turns'] for t in CHURN)} turns were acknowledged across "
        f"{TRIALS} trials: {CHURN}"
    )
    mid_turn = [t for t in CHURN if t["interrupted"] or t["released"]]
    assert mid_turn, (
        "no trial was killed inside a turn, so recovery was never exercised; "
        f"trials: {CHURN}"
    )
