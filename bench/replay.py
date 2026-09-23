"""How fast does the log replay? — the measurement DL-041 rests on.

DL-041 decides that memory's derived view is rebuilt from the log at every boot
and never stored. That is only the right call while replay is cheap, so the
number is not a sentence in a ledger: it is this script, kept beside the code it
describes so it can be re-run when the log format or the derivation changes.

It is deliberately **not** a test and lives outside ``testpaths`` — it takes
tens of seconds, and a suite that sometimes takes a minute is a suite people
stop running. Run it by hand:

    .venv/bin/python -u bench/replay.py

The threshold DL-041 states: if replay at a realistic log size passes about a
second, the decision gets revisited — and the remedy is a deletable snapshot
*cache* validated against the head seq, never a second source of truth.
"""

import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, "python")

from omega import episodes, memory  # noqa: E402

#: Sizes to report. Replay is linear, so two points are enough to show it is
#: linear and to extrapolate; the cost of a third is all in the append side.
SIZES = (10_000, 50_000)


def build(store: memory.MemoryStore, n: int) -> None:
    """Append ``n`` episodes shaped like real traffic.

    A chat turn is an inbound *and* a completion, so they are written in pairs.
    The text is the length a person actually types rather than a short token,
    because decode cost scales with payload size and a benchmark on toy strings
    would report a number the real system never sees.
    """
    for i in range(n // 2):
        store.append_episode(
            episodes.encode(
                episodes.inbound(
                    f"message number {i} with a realistic amount of text in it, "
                    f"the kind a person actually types when they are thinking "
                    f"out loud about something they care about",
                    channel="tray",
                )
            )
        )
        store.append_episode(
            episodes.encode(
                episodes.completed(
                    for_seq=store.head(),
                    outcome="spoke",
                    reply=f"reply number {i}, roughly the length of a real answer "
                    f"to a real question, which is what makes the decode cost "
                    f"representative rather than optimistic",
                )
            )
        )


def replay(store: memory.MemoryStore) -> tuple[int, dict[str, int]]:
    """One full derive pass: every episode, decoded.

    Decoding is inside the timed region on purpose. A replay that only walks
    frames is not a replay — the derived view is built out of payload *fields*,
    so a number that excluded decode would be measuring the wrong half.
    """
    seen = 0
    kinds: dict[str, int] = {}
    for episode in store.episodes_since(0):
        payload = episodes.decode(episode.payload)
        kinds[payload["kind"]] = kinds.get(payload["kind"], 0) + 1
        seen += 1
    return seen, kinds


def main() -> int:
    for n in SIZES:
        root = Path(tempfile.mkdtemp(prefix="omega-bench-"))
        try:
            store = memory.MemoryStore.open(root)
            started = time.perf_counter()
            build(store, n)
            append_s = time.perf_counter() - started
            store.close()

            # Reopened, so this is a cold boot rather than a warm cache —
            # which is the only situation the decision is actually about.
            store = memory.MemoryStore.open(root)
            started = time.perf_counter()
            seen, kinds = replay(store)
            replay_s = time.perf_counter() - started
            megabytes = sum(
                f.stat().st_size for f in root.rglob("*") if f.is_file()
            ) / 1e6
            store.close()

            print(
                f"n={seen:>7,}  replay={replay_s:6.3f}s  "
                f"({seen / replay_s:>10,.0f} ep/s)  log={megabytes:6.1f}MB  "
                f"append={append_s:6.2f}s ({append_s / seen * 1e3:.1f}ms each)  "
                f"{kinds}"
            )
        finally:
            shutil.rmtree(root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
