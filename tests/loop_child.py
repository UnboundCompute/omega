"""The child process for the turn-loop ``kill -9`` cases — M1_SPEC.md §1.2.

Run as:

    python loop_child.py block <dir>          enqueue one event, then hang
                                              *inside* the turn, claimed
    python loop_child.py churn <dir> <seed> <point> <n>
                                              enqueue and drain until the n-th
                                              time the loop reaches <point>,
                                              then park there

Every interesting line goes to stdout and is flushed before the next thing
happens, so a line the parent has actually read is a promise about what had
already occurred when the kill landed:

    APPEND <seq>            an event is durably in the queue
    CLAIMED <claimed> <done>  the turn is in flight and the model is "thinking"
    TURN <seq> <outcome>    the turn finished *and* DONE advanced past it
    PARKED <point> <n>      the loop is held at its n-th <point>; kill it now

The churn mode's kill point is *injected*, not timed. The parent used to sleep
a random 5-120 ms and kill wherever the child had got to, which under a loaded
machine meant the child had often not reached a turn at all. Now the child
parks itself at a named point and says so, and the parent kills it there:

    append    just before an inbound append (the n-th after the first)
    claimed   just after a turn's claim, before the model is asked
    think     inside the judge call, the turn in flight
    record    the terminal record durable, DONE not yet advanced
    skip      a record being skipped, CLAIMED moved past it and DONE not yet
    between   after a turn is settled and acknowledged, before the next append

No network and no API key: the provider is a fake whose judge callable is also
where the child parks so the kill is guaranteed to land mid-turn.
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

from omega import episodes, provider
from omega.executor import Executor
from omega.memory import MemoryStore
from omega.queue import CLAIMED, DONE, EventQueue


def say(line: str) -> None:
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


#: Where the churn mode can be told to park, in the order a turn reaches them.
POINTS = ("append", "claimed", "think", "record", "skip", "between")


def park(line: str) -> None:
    """Say where the loop is held, then hold it there until the kill lands."""
    say(line)
    while True:
        time.sleep(3600)


def main() -> int:
    mode = sys.argv[1]
    directory = Path(sys.argv[2])
    store = MemoryStore.open(directory)
    queue = EventQueue(store)

    if mode == "block":
        def judge(role: str, messages: list) -> str:
            if role == provider.JUDGE:
                say(f"CLAIMED {queue.claimed()} {queue.done()}")
                while True:  # the parent kills us here; that is the case
                    time.sleep(3600)
            return "ok"

        fake = provider.FakeProvider({provider.JUDGE: judge, provider.ACT: judge})
        executor = Executor(queue, complete=fake.complete)
        executor.recover()
        seq = queue.append(
            episodes.inbound("book the flight to Lisbon", channel="tray")
        )
        say(f"APPEND {seq}")
        executor.drain()
        raise SystemExit("the blocking turn returned, which it must not")

    if mode == "churn":
        rng = random.Random(int(sys.argv[3]))
        point, nth = sys.argv[4], int(sys.argv[5])
        if point not in POINTS:
            raise SystemExit(f"unknown kill point {point!r}")
        seen = {p: 0 for p in POINTS}

        def reach(here: str) -> None:
            seen[here] += 1
            if here == point and seen[here] == nth:
                park(f"PARKED {point} {nth}")

        def think(role: str, messages: list) -> str:
            if role == provider.JUDGE:
                reach("think")
                return rng.choice(["SPEAK", "SPEAK", "SILENT"])
            return "acknowledged"

        class Parking(EventQueue):
            def claim(self, seq: int) -> None:
                super().claim(seq)
                reach("claimed")

            def finish(self, seq: int) -> None:
                reach("record")
                super().finish(seq)

            def skip(self, seq: int) -> None:
                # EventQueue.skip, with the park in the one window it has.
                self._advance(CLAIMED, seq)
                reach("skip")
                self._advance(DONE, seq)

        queue = Parking(store)
        fake = provider.FakeProvider({provider.JUDGE: think, provider.ACT: think})
        executor = Executor(queue, complete=fake.complete)
        executor.recover()
        i = 0
        while True:
            i += 1
            if i > 1:
                reach("append")
            seq = queue.append(
                episodes.inbound(f"churn message {i}", channel="tray")
            )
            say(f"APPEND {seq}")
            for result in executor.drain():
                say(f"TURN {result.seq} {result.outcome}")
                reach("between")

    raise SystemExit(f"unknown mode {mode!r}")


if __name__ == "__main__":
    sys.exit(main())
