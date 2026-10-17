"""C2, DL-110 — the calendar T-minus sense, routine events, timely delivery,
and the in-channel steer.

Every clock is pinned, every provider scripted, every calendar synthetic: no
real events, names or mail.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pytest

from omega import episodes, grants, ledger, outbox, projection, provider, tminus, tools
from omega.act import act_loop
from omega.calendar import Occurrence
from omega.executor import Executor
from omega.memory import MemoryStore
from omega.queue import EventQueue
from omega.schedule import Scheduler


def local(day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 10, day, hour, minute).astimezone()


def event(summary: str, start: datetime, minutes: int = 30, *, uid: str = "",
          recurring: bool = False, moved_from: datetime | None = None,
          all_day: bool = False, location: str = "") -> Occurrence:
    return Occurrence(
        start=start, end=start + timedelta(minutes=minutes), all_day=all_day,
        summary=summary, location=location, uid=uid or f"{summary}@example.test",
        recurring=recurring, moved_from=moved_from,
    )


@pytest.fixture
def q(store: MemoryStore) -> EventQueue:
    return EventQueue(store)


def payloads(q: EventQueue) -> list[dict]:
    return [episodes.decode(e.payload) for e in q.store.episodes_since(0)]


def kinds(q: EventQueue, kind: str) -> list[dict]:
    return [p for p in payloads(q) if p.get("kind") == kind]


def warnings(q: EventQueue) -> list[dict]:
    return [p for p in kinds(q, episodes.MESSAGE_INBOUND)
            if any(k.startswith("tminus:") for k in p.get("keys") or ())]


def asked(q: EventQueue) -> list[dict]:
    return [p for p in kinds(q, episodes.MESSAGE_INBOUND)
            if tminus.ASK_KEY in (p.get("keys") or ())]


def executor(q: EventQueue, found: list[Occurrence], *, complete=None, act=None,
             pace: bool = True) -> Executor:
    kw: dict[str, Any] = {"act": act} if act is not None else {}
    ex = Executor(
        q, complete=(complete or provider.FakeProvider({})).complete, pace=pace,
        events=lambda start, end: [o for o in found if start <= o.end and o.start < end],
        **kw,
    )
    ex.recover()
    return ex


def sweep(ex: Executor, start: datetime, end: datetime) -> None:
    """One sense pass a minute, as the runtime makes them."""
    t = start
    while t <= end:
        ex.calendar_tminus(now=t.isoformat())
        t += timedelta(minutes=1)


def here(q: EventQueue, when: datetime) -> None:
    q.append(episodes.sense_reported(
        source=episodes.REPORT_PRESENCE, unit=f"mac@{when.timestamp()}",
        device="mac", body={"idle": 5.0}, at=when.isoformat(),
    ))


# --- meetings and flights -----------------------------------------------------


def test_each_meeting_is_warned_once_25_to_35_minutes_before_it_starts(q: EventQueue) -> None:
    starts = [local(14, 10), local(14, 13, 15), local(14, 16, 40)]
    found = [event(f"Synthetic review {i}", s) for i, s in enumerate(starts)]
    ex = executor(q, found)
    sweep(ex, local(14, 8), local(14, 18))
    told = warnings(q)
    assert len(told) == 3
    for tell, start in zip(told, starts):
        lead = start - datetime.fromisoformat(tell["at"])
        assert timedelta(minutes=25) <= lead <= timedelta(minutes=35)
        assert tell["urgency"] == "timely"
        assert ledger.calendar_key(found[starts.index(start)]) in tell["keys"]
    # Delivered verbatim, timely, with no model call and no budget charge.
    ex.drain()
    done = [p for p in kinds(q, episodes.TURN_COMPLETED) if p["outcome"] == "spoke"]
    assert len(done) == 3 and all(p["urgency"] == "timely" for p in done)
    assert "starts at" in done[0]["reply"] and kinds(q, episodes.MESSAGE_DELIVERED) == []


def test_nothing_is_said_at_or_after_a_start(q: EventQueue) -> None:
    """A meeting first seen ten minutes out, one under way and one over: no
    "happening now" message for any of them."""
    found = [
        event("Synthetic late add", local(14, 10, 10)),
        event("Synthetic in progress", local(14, 9, 30), minutes=90),
        event("Synthetic done", local(14, 8)),
        # The control: the sense is live over the same sweep.
        event("Synthetic control", local(14, 11, 30)),
    ]
    ex = executor(q, found)
    sweep(ex, local(14, 10), local(14, 12))
    [tell] = warnings(q)
    assert "Synthetic control" in tools.told(tell)
    assert tminus.due(found[0], local(14, 10, 10)) is None


def test_a_restart_does_not_warn_twice(q: EventQueue) -> None:
    found = [event("Synthetic sync", local(14, 10))]
    sweep(executor(q, found), local(14, 9, 30), local(14, 9, 31))
    sweep(executor(q, found), local(14, 9, 32), local(14, 9, 40))
    assert len(warnings(q)) == 1


def test_a_flight_is_warned_three_hours_out_with_a_leave_by(q: EventQueue) -> None:
    flight = event("Flight to Example City (ZZ 123)", local(14, 18), minutes=120)
    ex = executor(q, [flight])
    sweep(ex, local(14, 12), local(14, 18, 30))
    [tell] = warnings(q)
    lead = flight.start - datetime.fromisoformat(tell["at"])
    assert timedelta(hours=2, minutes=45) < lead <= timedelta(hours=3)
    text = tools.told(tell)
    assert "airport by 16:30" in text and "leave by about 15:45" in text
    assert "not live traffic" in text and tell["urgency"] == "timely"


def test_an_all_day_event_is_never_warned(q: EventQueue) -> None:
    ex = executor(q, [event("Synthetic offsite", local(14, 0), minutes=1440, all_day=True),
                      event("Synthetic control", local(14, 1))])
    sweep(ex, local(13, 22), local(14, 2))
    [tell] = warnings(q)
    assert "Synthetic control" in tools.told(tell)


def test_without_pace_or_events_nothing_is_read(q: EventQueue) -> None:
    calls: list[Any] = []
    ex = Executor(q, pace=False, events=lambda a, b: calls.append(a) or [])
    ex.recover()
    assert ex.calendar_tminus(now=local(14, 9).isoformat()) is False and calls == []


# --- routine events -----------------------------------------------------------


def standup(day: int, **kw: Any) -> Occurrence:
    return event("Synthetic standup", kw.pop("start", local(day, 10)), minutes=15,
                 uid="standup@example.test", recurring=True, **kw)


def test_a_routine_meeting_is_silent_and_asked_about_once(q: EventQueue) -> None:
    found = [standup(14), standup(15)]
    here(q, local(14, 9))
    sweep(executor(q, found), local(14, 9), local(14, 10, 30))
    here(q, local(15, 9))
    sweep(executor(q, found), local(15, 9), local(15, 10, 30))
    assert warnings(q) == []
    # The control: the same sweep warns an unchanged one asked about.
    assert tminus.due(found[0], local(14, 9, 30), found, routine_warnings=True) == "routine"
    [ask] = asked(q)
    assert ask["queued"] == [] and "warn me before routine meetings" in tools.told(ask)


def test_a_routine_meeting_that_moved_is_warned(q: EventQueue) -> None:
    moved = standup(14, start=local(14, 11), moved_from=local(14, 10))
    sweep(executor(q, [moved]), local(14, 10), local(14, 11))
    [tell] = warnings(q)
    assert "moved from its usual 10:00" in tools.told(tell)


def test_a_routine_meeting_that_clashes_is_warned(q: EventQueue) -> None:
    found = [standup(14), event("Synthetic vendor call", local(14, 10), minutes=60)]
    sweep(executor(q, found), local(14, 9), local(14, 10, 30))
    told = [tools.told(t) for t in warnings(q)]
    assert len(told) == 2
    assert any("clashes with \"Synthetic vendor call\"" in t for t in told)


def test_asked_for_routine_warnings_they_get_them(q: EventQueue) -> None:
    calls: list[str] = []
    fp = provider.FakeProvider({provider.JUDGE: lambda r, m: calls.append("j") or "SPEAK",
                                provider.ACT: lambda r, m: calls.append("a") or "ok"})
    q.append(episodes.inbound("warn me before routine meetings", channel="tray",
                              at=local(14, 8).isoformat()))
    ex = executor(q, [standup(14)], complete=fp)
    ex.drain()
    done = kinds(q, episodes.TURN_COMPLETED)[-1]
    assert done["outcome"] == "spoke" and "routine meetings too" in done["reply"]
    assert calls == []
    sweep(ex, local(14, 9), local(14, 10, 30))
    assert len(warnings(q)) == 1 and asked(q) == []


def test_the_brief_folds_in_routine_meetings_still_to_come() -> None:
    now = local(14, 9, 5)
    lines = tminus.routine_today(
        [standup(14), standup(15), event("Synthetic one-off", local(14, 11)),
         standup(14, start=local(14, 8))], now)
    assert lines == ["10:00 Synthetic standup"]
    text = outbox.thinking_text("2026-10-14 morning", situation="", routine=lines)
    assert "routine meetings still to come today" in text and "10:00 Synthetic standup" in text


# --- urgency on the wire ------------------------------------------------------


def test_a_timely_reply_carries_its_urgency_to_the_stream() -> None:
    record = episodes.completed(for_seq=3, outcome="spoke", reply="x", urgency="timely")
    update = projection.project(record, 4)
    assert update is not None and update.urgency == "timely"
    plain = projection.project(episodes.completed(for_seq=3, outcome="spoke", reply="x"), 5)
    assert plain is not None and plain.urgency is None
    with pytest.raises(episodes.BadPayload):
        episodes.completed(for_seq=3, outcome="spoke", reply="x", urgency="now!")


# --- DL-075: reminders still always deliver -----------------------------------


def test_a_reminder_still_speaks_beside_a_warning_in_quiet_hours(q: EventQueue) -> None:
    found = [event("Synthetic early call", local(14, 23, 30))]
    fp = provider.FakeProvider({provider.JUDGE: ["SPEAK"],
                                provider.ACT: ["Synthetic: take the synthetic pills now."]})
    ex = executor(q, found, complete=fp)
    q.append(episodes.inbound(
        tools.remind_instruction("take the synthetic pills"), channel="schedule",
        schedule_id="r1-abcd", at=local(14, 23, 0).isoformat(),
    ))
    sweep(ex, local(14, 22, 55), local(14, 23, 5))
    ex.drain()
    spoke = [p["reply"] for p in kinds(q, episodes.TURN_COMPLETED) if p["outcome"] == "spoke"]
    assert any("synthetic pills" in r for r in spoke)
    assert any("Synthetic early call" in r for r in spoke)


# --- the in-channel steer -----------------------------------------------------


def watch(q: EventQueue, what: str, sid: str) -> None:
    q.append(episodes.schedule_created(
        id=sid, instruction=tools.watch_instruction(what), every=3600,
        at=local(14, 8).isoformat(),
    ))


def give(q: EventQueue, *, drop: bool = False) -> None:
    seq = q.append(episodes.inbound("you have the tidy grant", channel="tray",
                                    at=local(14, 8).isoformat()))
    q.append(episodes.grant_given(for_seq=seq, scope=grants.TIDY))
    if drop:
        p = q.append(episodes.grant_proposed(
            for_seq=seq, scope=grants.TIDY, proposes="drop_gate", actions=[]))
        q.append(episodes.grant_answered(
            for_seq=seq, scope=grants.TIDY, proposal=p, proposed_for=seq, yes=True))
    q.claim(seq)
    q.finish(seq)


def running(q: EventQueue) -> set[str]:
    s = Scheduler(q)
    s.refresh()
    return {x.id for x in s.schedules}


def test_a_steer_proposes_retiring_the_watch_it_names_and_a_yes_does(q: EventQueue) -> None:
    watch(q, "their mail for LinkedIn job alerts", "w1-abcd")
    watch(q, "their mail for Acme invoices", "w2-abcd")
    give(q)
    q.append(episodes.inbound("stop the LinkedIn ones", channel="tray",
                              at=local(14, 9).isoformat()))
    fp = provider.FakeProvider({provider.JUDGE: ["SPEAK"], provider.ACT: ["Stopped it."]})
    ex = executor(q, [], complete=fp, act=act_loop)
    ex.drain()
    [proposal] = kinds(q, episodes.GRANT_PROPOSED)
    [action] = proposal["actions"]
    assert action["tool"] == tools.TIDY and action["args"]["id"] == "w1-abcd"
    assert action["args"]["change"] == "retire"
    reply = kinds(q, episodes.TURN_COMPLETED)[-1]
    assert reply["reply"].startswith("Should I retire") and "yes p" in reply["reply"]
    assert fp.calls == [] and running(q) == {"w1-abcd", "w2-abcd"}
    pid = reply["proposals"][0]["id"]
    q.append(episodes.inbound(f"yes {pid}", channel="tray", at=local(14, 9, 1).isoformat()))
    ex.drain()
    assert running(q) == {"w2-abcd"}


def test_with_the_gate_dropped_a_steer_retires_it_with_its_undo(q: EventQueue) -> None:
    watch(q, "their mail for LinkedIn job alerts", "w1-abcd")
    watch(q, "their mail for Acme invoices", "w2-abcd")
    give(q, drop=True)
    q.append(episodes.inbound("Stop the linkedin alerts please.", channel="tray",
                              at=local(14, 9).isoformat()))
    ex = executor(q, [])
    ex.drain()
    assert running(q) == {"w2-abcd"}
    [acted] = kinds(q, episodes.GRANT_ACTED)
    assert acted["args"]["id"] == "w1-abcd" and acted["undo"]
    assert kinds(q, episodes.TURN_COMPLETED)[-1]["reply"].startswith("Done: retire")


@pytest.mark.parametrize("text", [
    "stop working on the report",  # not a steer
    "stop the Zzyzx ones",  # names no watch
])
def test_what_is_not_a_steer_runs_the_ordinary_turn(q: EventQueue, text: str) -> None:
    watch(q, "their mail for LinkedIn job alerts", "w1-abcd")
    give(q, drop=True)
    q.append(episodes.inbound(text, channel="tray", at=local(14, 9).isoformat()))
    fp = provider.FakeProvider({provider.JUDGE: ["SPEAK"], provider.ACT: ["Sure."]})
    executor(q, [], complete=fp, act=act_loop).drain()
    assert running(q) == {"w1-abcd"} and fp.calls
    assert kinds(q, episodes.GRANT_ACTED) == []


def test_without_the_tidy_grant_a_steer_is_the_ordinary_turn(q: EventQueue) -> None:
    watch(q, "their mail for LinkedIn job alerts", "w1-abcd")
    q.append(episodes.inbound("stop the LinkedIn ones", channel="tray",
                              at=local(14, 9).isoformat()))
    fp = provider.FakeProvider({provider.JUDGE: ["SPEAK"], provider.ACT: ["Sure."]})
    executor(q, [], complete=fp, act=act_loop).drain()
    assert fp.calls and kinds(q, episodes.GRANT_PROPOSED) == []
    assert running(q) == {"w1-abcd"}
