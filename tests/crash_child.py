"""The child process for the ``kill -9`` cases (spec 31, 32, 33).

Run as:

    python crash_child.py once  <dir>     append one episode, ack it, then hang
    python crash_child.py loop  <dir>     append forever, acking each one

An "ack" is one line on stdout: ``ACK <seq> <payload-hex>``, flushed before the
next thing happens. The parent treats a line it has actually read as the log's
promise that the episode is durable, then kills -9 and checks the promise held.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from omega.memory import MemoryStore


def ack(seq: int, payload: bytes) -> None:
    sys.stdout.write(f"ACK {seq} {payload.hex()}\n")
    sys.stdout.flush()


def main() -> int:
    mode = sys.argv[1]
    directory = Path(sys.argv[2])
    store = MemoryStore.open(directory)

    if mode == "once":
        payload = b"the-one-episode-that-must-survive"
        seq = store.append_episode(payload, "", 4_242_424_242)
        ack(seq, payload)
        # Do not close: the parent kills us here, which is the point.
        while True:
            time.sleep(3600)

    if mode == "loop":
        i = 0
        while True:
            i += 1
            payload = f"crash-loop-{i}".encode()
            seq = store.append_episode(payload, "", 5_000_000 + i)
            ack(seq, payload)

    raise SystemExit(f"unknown mode {mode!r}")


if __name__ == "__main__":
    sys.exit(main())
