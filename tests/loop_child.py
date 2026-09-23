"""The child process for the turn-loop ``kill -9`` cases — M1_SPEC.md §1.2.

Run as:

    python loop_child.py block <dir>          enqueue one event, then hang
                                              *inside* the turn, claimed
    python loop_child.py churn <dir> <seed>   enqueue and drain forever

Every interesting line goes to stdout and is flushed before the next thing
happens, so a line the parent has actually read is a promise about what had
already occurred when the kill landed:

    APPEND <seq>            an event is durably in the queue
    CLAIMED <claimed> <done>  the turn is in flight and the model is "thinking"
    TURN <seq> <outcome>    the turn finished *and* DONE advanced past it

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
from omega.queue import EventQueue


def say(line: str) -> None:
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


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

        def think(role: str, messages: list) -> str:
            # Stands in for model latency. It is what makes the kill land
            # inside a turn often enough for the case to be exercising the
            # recovery path rather than the idle one.
            time.sleep(rng.uniform(0.001, 0.010))
            if role == provider.JUDGE:
                return rng.choice(["SPEAK", "SPEAK", "SILENT"])
            return "acknowledged"

        fake = provider.FakeProvider({provider.JUDGE: think, provider.ACT: think})
        executor = Executor(queue, complete=fake.complete)
        executor.recover()
        i = 0
        while True:
            i += 1
            seq = queue.append(
                episodes.inbound(f"churn message {i}", channel="tray")
            )
            say(f"APPEND {seq}")
            for result in executor.drain():
                say(f"TURN {result.seq} {result.outcome}")

    raise SystemExit(f"unknown mode {mode!r}")


if __name__ == "__main__":
    sys.exit(main())
