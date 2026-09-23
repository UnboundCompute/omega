"""kill -9. M0_SPEC.md cases 31, 32, 33.

The violation metric: *after any crash, at any point, the log reopens and every
episode that was acknowledged is present.* Not "usually" — reliability is
pass^k, so every case here runs 20 times (case 33).
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

from omega.memory import MemoryStore

REPO_ROOT = Path(__file__).resolve().parents[1]
CHILD = Path(__file__).resolve().parent / "crash_child.py"

#: Spec case 33. Passing once is not passing.
TRIALS = 20

ACK_RE = re.compile(r"^ACK (\d+) ([0-9a-f]*)$")

#: Filled in by the trials below; case 33 asserts against it afterwards.
TRIALS_RUN: dict[str, list[int]] = {"31": [], "32": []}


def _spawn(mode: str, directory: Path) -> subprocess.Popen:
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    return subprocess.Popen(
        [sys.executable, str(CHILD), mode, str(directory)],
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


def _parse_acks(text: str) -> list[tuple[int, bytes]]:
    """Only whole, well-formed ack lines count. A line the parent never fully
    read is not an acknowledgement and must not be treated as one."""
    acks: list[tuple[int, bytes]] = []
    for line in text.splitlines():
        m = ACK_RE.match(line.strip())
        if m:
            acks.append((int(m.group(1)), bytes.fromhex(m.group(2))))
    return acks


@pytest.mark.parametrize("trial", range(TRIALS))
def test_case_31_kill9_right_after_append_returns(store_dir: Path, trial: int) -> None:
    """Case 31 — kill -9 immediately after append returns; the episode is there.

    The child acknowledges the append on stdout only after ``append_episode``
    has returned, and the parent kills it only after *reading* that ack. So the
    kill is strictly after the fsync the append promised.
    """
    proc = _spawn("once", store_dir)
    try:
        line = proc.stdout.readline()
        acks = _parse_acks(line)
        assert len(acks) == 1, f"child did not acknowledge: {line!r} {proc.stderr}"
        seq, payload = acks[0]
        assert seq == 1
        assert _kill9(proc) == -signal.SIGKILL
    finally:
        if proc.poll() is None:
            _kill9(proc)
        proc.stdout.close()
        proc.stderr.close()

    with MemoryStore.open(store_dir) as s:
        assert s.head() == seq
        episodes = list(s.episodes_since(0))
    assert len(episodes) == 1
    assert episodes[0].seq == seq
    assert episodes[0].payload == payload
    assert episodes[0].payload == b"the-one-episode-that-must-survive"

    TRIALS_RUN["31"].append(trial)


@pytest.mark.parametrize("trial", range(TRIALS))
def test_case_32_kill9_during_an_append(store_dir: Path, trial: int) -> None:
    """Case 32 — kill -9 while appends are in flight: all-or-nothing, reopens clean.

    **Read this honestly and do not let it be mistaken for evidence it does not
    provide.** Killing a process does NOT tear a write: the kernel completes an
    in-flight ``pwrite`` even though the process is gone. What this case proves
    is that *acknowledged appends survive* and that the log always reopens. It
    proves NOTHING about torn-tail recovery — a green run here is not coverage
    of cases 25-28, which are only reachable by mutating the file deliberately
    (test_yellow_tails.py). A test that passes without exercising what it claims
    is an eval bug.
    """
    proc = _spawn("loop", store_dir)
    try:
        # Wait for the child to be genuinely mid-flight before killing it:
        # block until the first ack, then let it run a random extra slice. A
        # fixed sleep from spawn would sometimes land during interpreter
        # start-up, killing a process that had not appended anything — which
        # would make the case pass while exercising nothing.
        first = proc.stdout.readline()
        assert _parse_acks(first), f"child never got going: {first!r} {proc.stderr.read()!r}"
        time.sleep(random.uniform(0.002, 0.060))
        assert proc.poll() is None, f"child died early: {proc.stderr.read()}"
        assert _kill9(proc) == -signal.SIGKILL
        stdout = first + proc.stdout.read()
        stderr = proc.stderr.read()
    finally:
        if proc.poll() is None:
            _kill9(proc)
        proc.stdout.close()
        proc.stderr.close()

    acks = _parse_acks(stdout)
    assert acks, f"the child acknowledged nothing in the window: {stderr!r}"
    assert [seq for seq, _ in acks] == list(range(1, len(acks) + 1))
    last_acked = acks[-1][0]

    # The log always reopens. A raise here fails the case.
    with MemoryStore.open(store_dir) as s:
        head = s.head()
        episodes = list(s.episodes_since(0))

    # Every acknowledged episode is present, byte-identical.
    assert head >= last_acked, f"lost an acknowledged append: head={head} acked={last_acked}"
    # At most one more: the append that was in flight when the kill landed.
    assert head <= last_acked + 1, f"head={head} ran ahead of acks={last_acked}"

    assert len(episodes) == head
    assert [e.seq for e in episodes] == list(range(1, head + 1))
    by_seq = {e.seq: e.payload for e in episodes}
    for seq, payload in acks:
        assert by_seq[seq] == payload, f"seq {seq} came back as {by_seq[seq]!r}"

    # Nothing partial: the unacknowledged tail is either a whole episode or absent.
    if head == last_acked + 1:
        assert by_seq[head] == f"crash-loop-{head}".encode()

    TRIALS_RUN["32"].append(trial)


def test_case_33_both_crash_cases_ran_twenty_times() -> None:
    """Case 33 — cases 31 and 32 each ran 20 times. Passing once is not passing.

    This asserts the trials that *actually executed and passed*, not the
    parametrisation: a count taken from the constant would pass on empty.
    """
    assert TRIALS == 20
    assert sorted(TRIALS_RUN["31"]) == list(range(20)), (
        f"case 31 completed {len(TRIALS_RUN['31'])}/20 trials"
    )
    assert sorted(TRIALS_RUN["32"]) == list(range(20)), (
        f"case 32 completed {len(TRIALS_RUN['32'])}/20 trials"
    )
