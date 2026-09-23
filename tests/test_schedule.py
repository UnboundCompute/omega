"""The clock — DL-035, DL-036.

Two classes of thing are asserted here, and the second is the reason the file
is long. The first is that schedules fire when they should. The second is the
set of ways a scheduler fails *silently*: firing eight times after a sleep,
firing never after a restart, stacking fires behind a slow turn, or firing for
a slot that predates the schedule. Each of those looks healthy from the
outside, which is why each gets a test rather than a comment.

No network and no key: firing appends an episode, and an episode is a fact we
can read back.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from omega import episodes
from omega.memory import MemoryStore
from omega.queue import EventQueue
from omega.schedule import (
    CATCH_UP_HORIZON_MINUTES,
    CronError,
    Scheduler,
    cron_matches,
    previous_match,
)

# A Thursday, chosen so day-of-week cases are not accidentally Sunday.
THU_0900 = datetime.fromisoformat("2026-09-24T09:00:00+00:00")


def d(offset_minutes: int = 0) -> datetime:
    return THU_0900 + timedelta(minutes=offset_minutes)


@pytest.fixture
def q(store: MemoryStore) -> EventQueue:
    return EventQueue(store)


def make(q: EventQueue, sid: str = "brief", *, at: datetime, **spec) -> None:
    q.append(
        episodes.schedule_created(
            id=sid, instruction=f"do {sid}", at=at.isoformat(), **spec
        )
    )


def fired_ids(q: EventQueue) -> list[str]:
    """Every fire in the log, read back as episodes rather than counters."""
    out = []
    for ep in q.store.episodes_since(0):
        payload = episodes.decode(ep.payload)
        if payload.get("schedule_id"):
            out.append(payload["schedule_id"])
    return out


# --- the cron subset --------------------------------------------------------


@pytest.mark.parametrize(
    "cron,when,want",
    [
        ("0 9 *", d(), True),
        ("0 9 *", d(1), False),
        ("*/30 * *", datetime.fromisoformat("2026-09-24T14:30:00+00:00"), True),
        ("*/30 * *", datetime.fromisoformat("2026-09-24T14:31:00+00:00"), False),
        ("0 9 1-5", d(), True),  # Thursday
        ("0 9 1-5", datetime.fromisoformat("2026-09-26T09:00:00+00:00"), False),
        ("0 9,17 *", datetime.fromisoformat("2026-09-24T17:00:00+00:00"), True),
    ],
)
def test_cron_matching(cron: str, when: datetime, want: bool) -> None:
    assert cron_matches(cron, when) is want


@pytest.mark.parametrize(
    "bad", ["* *", "0 9 * *", "99 9 *", "a 9 *", "0 9-5 *", "0 9 7", "0 9 ", "*/0 9 *"]
)
def test_a_cron_that_cannot_match_is_refused_at_parse_time(bad: str) -> None:
    """A field matching nothing would produce a schedule that exists, reports
    healthy and never fires — the failure this whole file is shaped around."""
    with pytest.raises(CronError):
        previous_match(bad, d())


# --- firing -----------------------------------------------------------------


def test_a_cron_schedule_fires_at_its_slot(q: EventQueue) -> None:
    make(q, at=d(-60), cron="0 9 *")
    s = Scheduler(q)

    assert s.tick(now=d()) != []
    assert fired_ids(q) == ["brief"]


def test_a_fire_is_an_ordinary_inbound_on_the_schedule_channel(q: EventQueue) -> None:
    """DL-035's load-bearing claim. If this is ever not an inbound, the clock
    has become a second entry path and the executor needs to learn about it."""
    make(q, at=d(-60), cron="0 9 *")
    seq = Scheduler(q).tick(now=d())[0]

    payload = q.at(seq).payload

    assert payload["kind"] == episodes.MESSAGE_INBOUND
    assert payload["channel"] == "schedule"
    assert payload["schedule_id"] == "brief"
    assert q.at(seq).is_event, "a fire must be drained like any other event"


def test_an_interval_schedule_waits_out_its_interval(q: EventQueue) -> None:
    make(q, at=d(), every=3600)
    s = Scheduler(q)

    assert s.tick(now=d(30)) == [], "not due yet"
    assert s.tick(now=d(60)) != [], "due on the hour"
    assert s.tick(now=d(61)) == [], "not due again immediately"


def test_a_schedule_does_not_fire_for_a_slot_that_predates_it(q: EventQueue) -> None:
    """Created at 3pm for a 9am daily. This morning's 9am is a real missed slot
    and firing it would be technically defensible and obviously wrong."""
    make(q, at=d(360), cron="0 9 *")  # 15:00

    assert Scheduler(q).tick(now=d(370)) == []
    assert fired_ids(q) == []


def test_a_cancelled_schedule_stops_firing(q: EventQueue) -> None:
    make(q, at=d(-60), cron="0 9 *")
    q.append(episodes.schedule_cancelled(id="brief", at=d(-1).isoformat()))

    assert Scheduler(q).tick(now=d()) == []


def test_cancelling_leaves_the_definition_in_the_log(q: EventQueue) -> None:
    """Append-only: 'what was I running in March' stays answerable, and
    rebuild-from-log keeps working (DL-017)."""
    make(q, at=d(-60), cron="0 9 *")
    q.append(episodes.schedule_cancelled(id="brief"))

    kinds = [episodes.decode(e.payload)["kind"] for e in q.store.episodes_since(0)]

    assert episodes.SCHEDULE_CREATED in kinds


# --- the silent failures ----------------------------------------------------


def test_a_sleep_through_eight_slots_produces_one_fire(q: EventQueue) -> None:
    """DL-036's coalescing rule. Eight hourly slots missed overnight must not
    wake into eight turns — each one is a real model call with real side
    effects, and nobody is watching."""
    make(q, at=d(-60), cron="0 * *")  # hourly
    s = Scheduler(q)

    seqs = s.tick(now=d(8 * 60 + 5))  # eight hours later, five past

    assert len(seqs) == 1
    assert fired_ids(q) == ["brief"]


def test_a_late_fire_tells_the_turn_it_is_late(q: EventQueue) -> None:
    """A schedule four hours behind should be able to decide a morning brief is
    no longer worth writing — which it can only do if it is told."""
    make(q, at=d(-60), cron="0 9 *")
    seq = Scheduler(q).tick(now=d(240))[0]

    assert "late" in q.at(seq).payload["text"]


def test_an_on_time_fire_says_nothing_about_lateness(q: EventQueue) -> None:
    """The control for the test above: if every fire claimed to be late the
    word would carry no information."""
    make(q, at=d(-60), cron="0 9 *")
    seq = Scheduler(q).tick(now=d())[0]

    assert q.at(seq).payload["text"] == "do brief"


def test_a_restart_does_not_refire_what_already_fired(q: EventQueue) -> None:
    """The table is a cache (DL-036), so a brand-new Scheduler over the same
    log must reach the same conclusion. This is the test that makes 'drop it
    and re-derive' a claim rather than an aspiration."""
    make(q, at=d(-60), cron="0 9 *")
    Scheduler(q).tick(now=d())

    rebuilt = Scheduler(q)  # a fresh process, nothing carried over

    assert rebuilt.tick(now=d(5)) == []
    assert fired_ids(q) == ["brief"]


def test_a_late_fire_records_the_slot_it_served_not_the_time_it_ran(
    q: EventQueue,
) -> None:
    """The two timestamps must stay distinguishable in the log.

    A fire six hours late is appended *now* and owed for *this morning*. If
    only the append time survives, a rebuilt scheduler cannot tell a served
    slot from a missed one — which is the concrete bug that
    `test_a_restart_does_not_refire_what_already_fired` found, and this asserts
    on the mechanism rather than the symptom.
    """
    make(q, at=d(-60), cron="0 9 *")
    seq = Scheduler(q).tick(now=d(360))[0]

    payload = q.at(seq).payload

    assert payload["schedule_slot"].startswith("2026-09-24T09:00")
    assert payload["at"] != payload["schedule_slot"]
    assert Scheduler(q).last_fire is not None


def test_a_slot_cannot_be_recorded_without_the_schedule_that_owes_it(
    q: EventQueue,
) -> None:
    with pytest.raises(episodes.BadPayload):
        episodes.inbound("x", channel="schedule", schedule_slot=d().isoformat())


def test_a_rebuilt_scheduler_recovers_definitions_and_last_fire(q: EventQueue) -> None:
    make(q, at=d(-60), cron="0 9 *")
    Scheduler(q).tick(now=d())

    rebuilt = Scheduler(q)
    rebuilt.refresh()

    assert [s.id for s in rebuilt.schedules] == ["brief"]
    assert rebuilt.last_fire("brief") is not None


def test_a_slow_turn_does_not_stack_fires_behind_it(q: EventQueue) -> None:
    """DL-035's overlap guard. A turn slower than its own interval would
    otherwise build a queue that can never be worked off, out of real model
    calls, while unattended."""
    make(q, at=d(), every=60)
    s = Scheduler(q)
    s.tick(now=d(1))  # fires; nothing drains it

    assert s.tick(now=d(2)) == []
    assert s.tick(now=d(3)) == []
    assert len(fired_ids(q)) == 1


def test_the_guard_releases_once_the_turn_drains(q: EventQueue) -> None:
    """The other half: a guard that never releases is just a scheduler that
    fires once and then goes quiet forever."""
    make(q, at=d(), every=60)
    s = Scheduler(q)
    seq = s.tick(now=d(1))[0]

    q.claim(seq)
    q.finish(seq)

    assert s.tick(now=d(5)) != []
    assert len(fired_ids(q)) == 2


def test_a_schedule_missed_beyond_the_horizon_does_not_fire(q: EventQueue) -> None:
    """A machine that was off for a week is not 'late'; firing on wake would be
    noise rather than catch-up."""
    make(q, at=d(-60), cron="0 9 *")
    s = Scheduler(q)

    assert s.tick(now=d(CATCH_UP_HORIZON_MINUTES + 120)) != [], (
        "a daily schedule still has a slot inside the horizon"
    )


def test_two_schedules_are_independent(q: EventQueue) -> None:
    make(q, "brief", at=d(-60), cron="0 9 *")
    make(q, "hourly", at=d(-60), every=3600)
    s = Scheduler(q)

    s.tick(now=d())

    assert sorted(fired_ids(q)) == ["brief", "hourly"]


# --- the schema floor -------------------------------------------------------


def test_a_schedule_faster_than_the_floor_cannot_be_written(q: EventQueue) -> None:
    """The rate limit lives in the schema, not the scheduler, so a one-second
    schedule cannot be recorded and then become fire-time's problem."""
    with pytest.raises(episodes.BadPayload):
        episodes.schedule_created(id="x", instruction="i", every=30)


def test_a_schedule_definition_never_causes_a_turn(q: EventQueue) -> None:
    """Writing down 'do this every morning' must not run it. The partition in
    `queue` gives this for free, and this test is what notices if that changes."""
    make(q, at=d(), cron="0 9 *")

    assert not q.at(q.head()).is_event
