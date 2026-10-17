"""What the outbox and the slot clock did lately — the recent view (C0, C4).

Read-only; run it on a **copy** of the log, as ``bench/metrics.py``:

    .venv/bin/python -u bench/recent.py /tmp/omega-copy [--last 40]

One line per record, newest last: held drafts, their delivery or drop (with
why and which drafts), thinking turns started, slots written down as failed,
re-checks of held drafts, and tidy offers. A drop shows its own stamp, so how
long the turn that dropped it ran is visible.
"""

import argparse
import sys
from typing import Any

sys.path.insert(0, "python")

from omega import episodes, grants, memory, notice  # noqa: E402


def _one(text: Any, cap: int = 100) -> str:
    one = " ".join(str(text).split())
    return one if len(one) <= cap else one[: cap - 1] + "…"


def describe(seq: int, p: dict[str, Any]) -> str:
    """One record as a line, or ``""`` when it is not one this view shows."""
    kind = p.get("kind")
    at = str(p.get("at", ""))[:19]
    if kind == episodes.MESSAGE_QUEUED:
        return f"{seq} {at} held ({p.get('source')}; {p.get('why')}): {_one(p.get('text'))}"
    if kind == episodes.MESSAGE_DELIVERED:
        return f"{seq} {at} delivered by {p.get('for_seq')}, carrying {p.get('queued')}"
    if kind == episodes.MESSAGE_DROPPED:
        return f"{seq} {at} dropped {p.get('queued')} by {p.get('for_seq')}: {p.get('why')}"
    if kind == episodes.THINKING_FAILED:
        return f"{seq} {at} thinking failed, {p.get('slot')}: {p.get('why')}"
    if kind == episodes.MESSAGE_INBOUND and p.get("channel") == notice.CHANNEL:
        if p.get("thinking"):
            return f"{seq} {at} thinking started, {p.get('thinking')}, shown {p.get('queued')}"
        if isinstance(p.get("queued"), list):
            return f"{seq} {at} re-check of held {p.get('queued')}"
        return ""
    if kind == episodes.GRANT_PROPOSED and p.get("scope") == grants.TIDY:
        acts = [a.get("why") for a in p.get("actions") or ()]
        return f"{seq} {at} tidy offered: {_one('; '.join(map(str, acts)))}"
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("store", help="a COPY of the omega store directory")
    parser.add_argument("--last", type=int, default=40)
    args = parser.parse_args()
    store = memory.MemoryStore.open(args.store)
    lines = [
        line for line in (
            describe(e.seq, episodes.decode(e.payload)) for e in store.episodes_since(0)
        ) if line
    ]
    for line in lines[-args.last:]:
        print(line)
    if not lines:
        print("(nothing of these kinds in this log)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
