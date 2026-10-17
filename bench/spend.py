"""What omega spent, per day and per kind of turn, and how it thought (C4, DL-111).

Read-only; run it on a **copy** of the log, as ``bench/metrics.py``:

    .venv/bin/python -u bench/spend.py /tmp/omega-copy [--days 14]

Tokens come only from ``turn.completed.usage``; a turn's kind and day from
the inbound it answered: ``thinking`` (a slot), ``notice`` (any other unprompted
notice), ``schedule`` (a reminder or job firing) or ``prompted`` (their own
message). After the table, the journal's decision mix (with the let-go
share) and how many cited seqs were dropped as fabricated.
"""

import argparse
import sys
from typing import Any

sys.path.insert(0, "python")

from omega import episodes, memory, notice  # noqa: E402
from omega.topics import spent  # noqa: E402

KINDS = ("thinking", "notice", "schedule", "prompted")


def kind_of(p: dict[str, Any]) -> str:
    channel = p.get("channel")
    if channel == notice.CHANNEL:
        return "thinking" if p.get("thinking") else "notice"
    if channel == "schedule":
        return "schedule"
    return "prompted"


def fold(decoded) -> dict[str, Any]:
    """``decoded`` is ``(seq, payload)`` pairs; returns the per-day table,
    the journal's decision mix and its fabricated-citation count."""
    asked: dict[int, tuple[str, str]] = {}
    days: dict[str, dict[str, int]] = {}
    mix: dict[str, int] = {}
    fabricated = 0
    entries = 0
    for seq, p in decoded:
        kind = p.get("kind")
        if kind == episodes.MESSAGE_INBOUND:
            # Charged to the day of the inbound, as the outbox's own budget
            # charges it: a turn that ran past midnight is that day's.
            day = notice.local_date(str(p.get("at", "")))
            if day is not None:
                asked[seq] = (kind_of(p), day)
        elif kind == episodes.TURN_COMPLETED:
            found = asked.pop(p.get("for_seq"), None)
            if found is None:
                continue
            turn, day = found
            row = days.setdefault(day, dict.fromkeys(KINDS, 0))
            row[turn] += spent(p.get("usage"))
        elif kind == episodes.JOURNAL_WRITTEN:
            entries += 1
            fabricated += int(p.get("fabricated") or 0)
            for d in p.get("decisions") or ():
                mix[d["decision"]] = mix.get(d["decision"], 0) + 1
    return {"days": days, "mix": mix, "fabricated": fabricated, "entries": entries}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("store", help="a COPY of the omega store directory")
    parser.add_argument("--days", type=int, default=14)
    args = parser.parse_args()
    store = memory.MemoryStore.open(args.store)
    found = fold((e.seq, episodes.decode(e.payload)) for e in store.episodes_since(0))
    print("day         " + "".join(f"{k:>10}" for k in KINDS) + "     total")
    for day in sorted(found["days"])[-args.days:]:
        row = found["days"][day]
        print(f"{day}  " + "".join(f"{row[k]:>10}" for k in KINDS)
              + f"{sum(row.values()):>10}")
    if not found["days"]:
        print("(no metered turns in this log)")
    mix = found["mix"]
    total = sum(mix.values())
    print(f"\njournal: {found['entries']} entries, {total} decisions")
    for choice in sorted(episodes.JOURNAL_DECISIONS):
        n = mix.get(choice, 0)
        share = f"{100 * n / total:.0f}%" if total else "-"
        print(f"  {choice:<7} {n:>5}  {share}")
    print(f"fabricated citations dropped: {found['fabricated']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
