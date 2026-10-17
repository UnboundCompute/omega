"""The M0' live invariants, read off a log — so a gate is a number, not a hunch.

Read-only: it decodes episodes and prints counts; it never appends. Run it on
a **copy** of the log (opening a store may take its lock):

    cp -R ~/.omega/store /tmp/omega-copy
    .venv/bin/python -u bench/metrics.py /tmp/omega-copy [--since 2026-10-17]

Every line is ``name: hits/denominator`` so an empty denominator shows as
``0/0`` — a gate that passes on nothing is not a pass (CLAUDE.md, fail closed
on empty). ``--since`` (a local date) bounds every count to the days after a
deploy.

What each line measures (AGENT.md, M0' gated invariants):

- ``at fires on time``: fires of cron schedules within 2 min of their minute.
- ``thinking on its slot``: thinking turns started within 15 min of 09:00 or
  19:00, out of slots started plus slots written down as failed.
- ``claims anchored on omega``: new inferred claims a conversation reflection
  pass anchored on a fire or a look (should be 0), out of inferred claims
  filed. A transcript, usage or audio claim anchored on a look's record is not
  counted: those passes anchor on the head because no turn is behind them
  (``tidy.conversation_claims``), the same test tidy retracts by.
- ``reflections without a person``: passes with no new person message since
  the previous pass (should be 0), out of passes.
- ``verbatim held-draft tells``: schedule tells carrying held drafts (should
  be 0), out of deliveries of held drafts.
- ``delivered within 10 min of them``: unprompted deliveries less than 10 min
  after a message of theirs (should be 0), out of deliveries.
"""

import argparse
import sys
from datetime import datetime, timezone
from typing import Any, Optional

sys.path.insert(0, "python")

from omega import episodes, memory, notice, outbox, promises, tidy  # noqa: E402


def _at(payload: dict[str, Any]) -> Optional[datetime]:
    try:
        at = datetime.fromisoformat(str(payload.get("at")))
    except ValueError:
        return None
    return at if at.tzinfo else at.replace(tzinfo=timezone.utc)


def _line(name: str, hits: int, of: int, note: str = "") -> str:
    return f"{name}: {hits}/{of}" + (f"  ({note})" if note else "")


def measure(store: memory.MemoryStore, since: Optional[str]) -> list[str]:
    log = [(e.seq, episodes.decode(e.payload)) for e in store.episodes_since(0)]
    by_seq = dict(log)

    def counted(p: dict[str, Any]) -> bool:
        day = notice.local_date(str(p.get("at", "")))
        return since is None or (day is not None and day >= since)

    crons: dict[str, str] = {}
    fires = on_time = 0
    started = slot_on_time = failed = 0
    inferred = anchored = 0
    passes = empty_passes = 0
    person_since_pass = False
    held_tells = held_deliveries = 0
    deliveries = early = 0
    heard: Optional[datetime] = None

    reflected = tidy.conversation_claims(log)

    for seq, p in log:
        kind = p.get("kind")
        if kind == episodes.SCHEDULE_CREATED:
            if p.get("cron"):
                crons[str(p["id"])] = str(p["cron"])
            else:
                crons.pop(str(p["id"]), None)
            continue
        if kind == episodes.MESSAGE_INBOUND:
            if promises.theirs(p):
                heard = _at(p)
                person_since_pass = True
            if not counted(p):
                continue
            sid = p.get("schedule_id")
            at = _at(p)
            if p.get("channel") == "schedule" and sid in crons and at is not None:
                minute, hour = crons[sid].split()[:2]
                if minute.isdigit() and hour.isdigit():
                    local = at.astimezone()
                    want = local.replace(hour=int(hour), minute=int(minute), second=0, microsecond=0)
                    fires += 1
                    on_time += abs((local - want).total_seconds()) <= 120
            if p.get("channel") == "schedule" and p.get("queued"):
                held_tells += 1
            slot = p.get("thinking")
            if p.get("channel") == notice.CHANNEL and isinstance(slot, str) and at is not None:
                started += 1
                start = outbox.slot_start(slot)
                slot_on_time += start is not None and abs((at - start).total_seconds()) <= 15 * 60
        elif not counted(p):
            continue
        elif kind == episodes.THINKING_FAILED:
            failed += 1
        elif kind == episodes.CLAIM_EXTRACTED and p.get("explicit") is False:
            inferred += 1
            anchored += seq in reflected and tidy.fire_anchor(
                by_seq, p.get("for_seq"), p.get("source_seq")
            ) is not None
        elif kind == episodes.REFLECTION_DONE:
            passes += 1
            empty_passes += not person_since_pass
            person_since_pass = False
        elif kind == episodes.MESSAGE_DELIVERED:
            deliveries += 1
            held_deliveries += bool(p.get("queued"))
            at = _at(p)
            if heard is not None and at is not None:
                early += 0 <= (at - heard).total_seconds() < outbox.SPOKE_SECONDS
    return [
        _line("at fires on time", on_time, fires, "within 2 min"),
        _line("thinking on its slot", slot_on_time, started + failed,
              f"{failed} written down as failed"),
        _line("claims anchored on omega", anchored, inferred, "want 0"),
        _line("reflections without a person", empty_passes, passes, "want 0"),
        _line("verbatim held-draft tells", held_tells, held_deliveries, "want 0"),
        _line("delivered within 10 min of them", early, deliveries, "want 0"),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("store", help="a COPY of the omega store directory")
    parser.add_argument("--since", help="local date, YYYY-MM-DD", default=None)
    args = parser.parse_args()
    if args.since is not None:
        datetime.strptime(args.since, "%Y-%m-%d")
    store = memory.MemoryStore.open(args.store)
    for line in measure(store, args.since):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
