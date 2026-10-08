"""DL-068 — omega speaks when something calls for it, not on a budget.

Watches, reminders, the silence sentinel on unprompted channels, the machine
sense, and the offline skip. **Nothing here opens a socket, runs pmset or reads
a key**: every sense is injected, and the real ``machine.read`` is exercised
only with an injected battery reader.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from omega import episodes, machine, notice, provider, tools
from omega.executor import Executor
from omega.memory import MemoryStore
from omega.queue import EventQueue
from omega.schedule import Scheduler
from omega.tools import LOCAL, ToolBox, ToolError, ToolRejected
from omega.turn import ActResult, TurnContext, run_turn

AT = "2026-10-07T12:00:00+00:00"
T0 = datetime.fromisoformat(AT)


@pytest.fixture
def q(store: MemoryStore) -> EventQueue:
    return EventQueue(store)


def _box(q: EventQueue, tmp_path: Path, running=()) -> ToolBox:
    root = tmp_path / "box"
    root.mkdir(exist_ok=True)
    return ToolBox(store_root=root, append=q.append, for_seq=7, running=running)


def _payloads(q: EventQueue) -> list[dict]:
    return [episodes.decode(e.payload) for e in q.store.episodes_since(0)]


# --- watch -------------------------------------------------------------------


def test_a_watch_files_an_ordinary_schedule_with_the_watch_framing(
    q: EventQueue, tmp_path: Path
) -> None:
    box = _box(q, tmp_path)
    decision = box.classify("watch", {"what": "Bigg Boss 20 news", "every_minutes": 60})
    assert decision.tier == LOCAL
    out = box.dispatch(decision)

    [created] = [p for p in _payloads(q) if p["kind"] == episodes.SCHEDULE_CREATED]
    assert created["every"] == 3600
    assert created["cron"] is None
    assert "once" not in created, "a watch is standing, not one-shot"
    assert "Bigg Boss 20 news" in created["instruction"]
    assert tools.NOTHING_NEW in created["instruction"]
    assert created["id"] in out


@pytest.mark.parametrize(
    "args",
    [
        {"what": "x", "every_minutes": tools.MIN_WATCH_MINUTES - 1},
        {"what": "x", "every_minutes": True},
        {"what": "x", "every_minutes": "60"},
        {"what": "   ", "every_minutes": 60},
    ],
)
def test_a_watch_is_refused_below_its_floor_or_without_a_subject(
    q: EventQueue, tmp_path: Path, args: dict
) -> None:
    with pytest.raises(ToolRejected):
        _box(q, tmp_path).classify("watch", args)
    assert q.head() == 0


def test_a_box_with_no_log_cannot_set_a_watch(tmp_path: Path) -> None:
    box = ToolBox(store_root=tmp_path)
    decision = box.classify("watch", {"what": "x", "every_minutes": 30})
    with pytest.raises(ToolError):
        box.dispatch(decision)


# --- remind ------------------------------------------------------------------


def test_a_reminder_is_one_shot_and_has_no_way_to_stay_quiet(
    q: EventQueue, tmp_path: Path
) -> None:
    """The live failure: "remind me in 2 min about testing albert" was
    answered with "set a timer yourself"."""
    box = _box(q, tmp_path)
    box.dispatch(box.classify("remind", {"what": "testing albert", "in_minutes": 2}))

    [created] = [p for p in _payloads(q) if p["kind"] == episodes.SCHEDULE_CREATED]
    assert created["every"] == 120
    assert created["once"] is True
    assert "testing albert" in created["instruction"]
    assert tools.NOTHING_NEW not in created["instruction"]


def test_a_reminder_that_needs_a_lookup_may_act_rather_than_only_speak() -> None:
    """The live failure after DL-068 shipped: "remind me in 2 min ... the
    current weather" fired on time, but the judge was told to SPEAK on every
    reminder, so the turn had no tools and said "I need to check live
    weather" instead of checking. A reminder must be able to reach ACT."""
    from omega import turn

    rule = turn._JUDGE_SYSTEM.split("reminder firing")[1]
    assert "ACT" in rule
    assert "never SILENT" in rule
    assert "looked up" in tools.remind_instruction("the weather")


def test_a_reminder_rounds_up_and_is_bounded(q: EventQueue, tmp_path: Path) -> None:
    box = _box(q, tmp_path)
    assert box.classify("remind", {"what": "x", "in_minutes": 0.3}).args["in_minutes"] == 1
    assert box.classify("remind", {"what": "x", "in_minutes": 1.2}).args["in_minutes"] == 2
    with pytest.raises(ToolRejected):
        box.classify("remind", {"what": "x", "in_minutes": tools.MAX_REMIND_MINUTES + 1})
    with pytest.raises(ToolRejected):
        box.classify("remind", {"what": "x", "in_minutes": False})


def test_once_needs_every_and_is_only_ever_true() -> None:
    with pytest.raises(episodes.BadPayload):
        episodes.schedule_created(id="r", instruction="x", cron="0 9 *", once=True)
    bad = episodes.schedule_created(id="r", instruction="x", every=60)
    bad["once"] = False
    with pytest.raises(episodes.BadPayload):
        episodes._validate(bad)


def test_a_reminder_fires_exactly_once_even_across_a_restart(q: EventQueue) -> None:
    q.append(
        episodes.schedule_created(
            id="r1", instruction="remind", every=120, once=True, at=AT
        )
    )
    clock = Scheduler(q)
    assert clock.tick(now=T0 + timedelta(minutes=1)) == []
    fired = clock.tick(now=T0 + timedelta(minutes=2))
    assert len(fired) == 1
    # Not drained: the in-flight guard alone would also hold this tick back,
    # which is why the assertions that matter are `schedules` and the reborn
    # fold below, where no in-flight state exists.
    assert clock.tick(now=T0 + timedelta(minutes=10)) == []
    assert clock.schedules == []

    # A fresh fold of the same log — what a restart builds — agrees.
    reborn = Scheduler(q)
    assert reborn.tick(now=T0 + timedelta(hours=5)) == []
    assert reborn.schedules == []


def test_a_standing_schedule_is_not_retired_by_its_fire(q: EventQueue) -> None:
    """The control for the test above: without ``once`` nothing changes."""
    q.append(episodes.schedule_created(id="w1", instruction="look", every=60, at=AT))
    clock = Scheduler(q)
    assert len(clock.tick(now=T0 + timedelta(minutes=2))) == 1
    reborn = Scheduler(q)
    reborn.refresh()
    assert [s.id for s in reborn.schedules] == ["w1"]


# --- unwatch -----------------------------------------------------------------


def test_unwatch_stops_only_what_is_running(q: EventQueue, tmp_path: Path) -> None:
    q.append(episodes.schedule_created(id="w1", instruction="look", every=900, at=AT))
    clock = Scheduler(q)
    clock.refresh()
    running = tuple(clock.schedules)
    box = _box(q, tmp_path, running=running)

    with pytest.raises(ToolError, match="w1"):
        box.dispatch(box.classify("unwatch", {"id": "nope"}))
    box.dispatch(box.classify("unwatch", {"id": "w1"}))

    assert _payloads(q)[-1]["kind"] == episodes.SCHEDULE_CANCELLED
    after = Scheduler(q)
    after.refresh()
    assert after.schedules == []


# --- schedules (DL-077) --------------------------------------------------------


def _running(q: EventQueue) -> tuple:
    clock = Scheduler(q)
    clock.refresh()
    return tuple(clock.schedules)


def test_schedules_lists_what_is_running_by_id_and_in_words(
    q: EventQueue, tmp_path: Path
) -> None:
    now = datetime.now(timezone.utc)
    q.append(episodes.schedule_created(
        id="w1", instruction=tools.watch_instruction("Bigg Boss news"), every=3600, at=AT
    ))
    q.append(episodes.schedule_created(
        id="r1", instruction=tools.remind_instruction("call mum"), every=600,
        once=True, at=now.isoformat(),
    ))
    q.append(episodes.schedule_created(id="s1", instruction="water the plants", cron="0 9 *", at=AT))
    q.append(episodes.schedule_created(id="gone", instruction="old", every=900, at=AT))
    q.append(episodes.schedule_cancelled(id="gone", at=AT))
    box = _box(q, tmp_path, running=_running(q))
    before = len(_payloads(q))

    decision = box.classify("schedules", {})
    assert decision.tier == tools.EXPLORATION
    out = box.dispatch(decision)

    lines = dict(line.split(" — ", 1) for line in out.splitlines())
    assert set(lines) == {"w1", "r1", "s1"}, "a cancelled schedule is not listed"
    assert lines["w1"] == "watch, every 60 minutes: Bigg Boss news"
    assert lines["r1"].startswith("reminder, once, due in about 1") and lines["r1"].endswith(": call mum")
    assert lines["s1"].startswith("schedule, at 9:00") and lines["s1"].endswith(": water the plants")
    assert len(_payloads(q)) == before, "listing writes nothing"


def test_schedules_says_so_when_nothing_is_running(q: EventQueue, tmp_path: Path) -> None:
    box = _box(q, tmp_path)
    out = box.dispatch(box.classify("schedules", {}))
    assert out.startswith("nothing is scheduled")
    assert _payloads(q) == []


def test_unwatch_receipt_names_what_stopped_not_the_id(q: EventQueue, tmp_path: Path) -> None:
    q.append(episodes.schedule_created(
        id="w1", instruction=tools.watch_instruction("flight prices"), every=900, at=AT
    ))
    q.append(episodes.schedule_created(id="w2", instruction="other", every=900, at=AT))
    box = _box(q, tmp_path, running=_running(q))
    out = box.dispatch(box.classify("unwatch", {"id": "w1"}))
    assert out == "stopped watch, every 15 minutes: flight prices"
    cancelled = [p for p in _payloads(q) if p["kind"] == episodes.SCHEDULE_CANCELLED]
    assert [p["id"] for p in cancelled] == ["w1"], "only the named one is cancelled"


# --- the silence sentinel ----------------------------------------------------


def _put(q: EventQueue, channel: str):
    extra = {"schedule_id": "w1"} if channel == "schedule" else {}
    seq = q.append(episodes.inbound("look", channel=channel, at=AT, **extra))
    q.claim(seq)
    return q.at(seq)


def _act_says(text: str):
    def act(ctx: TurnContext) -> ActResult:
        return ActResult(tools=("fetch",), stop_reason="done", text=text)

    return act


@pytest.mark.parametrize("channel", [notice.CHANNEL, "schedule"])
@pytest.mark.parametrize("said", ["(nothing new)", "`(nothing new)`", "Nothing new."])
def test_an_unprompted_check_that_found_nothing_is_silent(
    q: EventQueue, channel: str, said: str
) -> None:
    fp = provider.FakeProvider({provider.JUDGE: "ACT", provider.ACT: ["unused"]})
    result = run_turn(q, _put(q, channel), complete=fp.complete, act=_act_says(said), at=AT)
    assert result.outcome == "silent"


def test_the_sentinel_means_nothing_to_a_person_who_asked(q: EventQueue) -> None:
    """Silence is never the answer to someone who wrote (DL-011); on a typed
    channel the sentinel is just text, and it is spoken."""
    fp = provider.FakeProvider({provider.JUDGE: "ACT", provider.ACT: ["unused"]})
    result = run_turn(
        q, _put(q, "tray"), complete=fp.complete, act=_act_says("(nothing new)"), at=AT
    )
    assert result.outcome == "spoke"


def test_a_real_finding_on_an_unprompted_channel_is_spoken(q: EventQueue) -> None:
    fp = provider.FakeProvider({provider.JUDGE: "ACT", provider.ACT: ["unused"]})
    result = run_turn(
        q,
        _put(q, "schedule"),
        complete=fp.complete,
        act=_act_says("Nothing new on the show, but your disk is nearly full."),
        at=AT,
    )
    assert result.outcome == "spoke"


# --- the machine sense -------------------------------------------------------


def test_battery_parsing_reads_what_pmset_prints() -> None:
    on_battery = (
        "Now drawing from 'Battery Power'\n"
        " -InternalBattery-0 (id=23068771)\t77%; discharging; 6:50 remaining present: true"
    )
    assert machine.parse_battery(on_battery) == (77, False, True)
    charging = (
        "Now drawing from 'AC Power'\n"
        " -InternalBattery-0 (id=1)\t41%; charging; 1:10 remaining present: true"
    )
    assert machine.parse_battery(charging) == (41, True, False)
    desktop = "Now drawing from 'AC Power'\n"
    assert machine.parse_battery(desktop) == (None, None, False)
    assert machine.parse_battery("") == (None, None, None)


def test_describe_flags_low_disk_by_either_threshold() -> None:
    gb = 1024**3
    nearly_full = machine.Reading(disk_free=7 * gb, disk_total=228 * gb)
    assert machine.describe(nearly_full)[0].endswith("LOW")
    roomy = machine.Reading(disk_free=400 * gb, disk_total=1000 * gb)
    assert not machine.describe(roomy)[0].endswith("LOW")
    small_but_full = machine.Reading(disk_free=25 * gb, disk_total=500 * gb)
    assert machine.describe(small_but_full)[0].endswith("LOW")
    # 15% free clears the fraction; 15 GB is still under the byte floor.
    big_enough_fraction = machine.Reading(disk_free=15 * gb, disk_total=100 * gb)
    assert machine.describe(big_enough_fraction)[0].endswith("LOW")


def test_describe_flags_a_low_battery_only_when_it_is_draining() -> None:
    low = machine.Reading(battery_percent=12, charging=False, on_battery=True)
    assert machine.describe(low) == ["Battery: 12%, on battery - LOW"]
    plugged = machine.Reading(battery_percent=12, charging=True, on_battery=False)
    assert "LOW" not in machine.describe(plugged)[0]


def test_an_unreadable_machine_is_left_out_not_guessed(tmp_path: Path) -> None:
    def broken():
        raise RuntimeError("pmset exploded")

    reading = machine.read(root=str(tmp_path / "missing"), battery=broken)
    assert reading.disk_free is None and reading.battery_percent is None
    assert all(not line.startswith(("Disk", "Battery")) for line in machine.describe(reading))
    assert machine.describe(machine.Reading()) == []


# --- the look: machine line, novelty, offline --------------------------------


NOW = datetime(2026, 10, 7, 15, 0, tzinfo=timezone.utc)


def _quiet_log(q: EventQueue) -> None:
    q.append(
        episodes.inbound(
            "hello", channel="tray", at=(NOW - timedelta(hours=2)).isoformat()
        )
    )


def _silent() -> provider.FakeProvider:
    return provider.FakeProvider({provider.JUDGE: "SILENT"})


def test_the_look_carries_the_machine_line(q: EventQueue) -> None:
    _quiet_log(q)
    ex = Executor(
        q,
        complete=_silent().complete,
        online=lambda: True,
        machine=lambda: ["Disk: 7 GB free of 228 GB (3%) - LOW"],
    )
    ex.recover()
    ex.drain()
    assert ex.notice(now=NOW) is True
    look = _payloads(q)[-1]
    assert look["channel"] == notice.CHANNEL
    assert "Their machine right now:" in look["text"]
    assert "7 GB free of 228 GB (3%) - LOW" in look["text"]


def test_a_machine_sense_that_raises_does_not_stop_the_look(q: EventQueue) -> None:
    def broken():
        raise OSError("no")

    _quiet_log(q)
    ex = Executor(q, complete=_silent().complete, machine=broken)
    ex.recover()
    ex.drain()
    assert ex.notice(now=NOW) is True
    assert "Their machine" not in _payloads(q)[-1]["text"]


def test_no_network_means_no_look_and_nothing_written(q: EventQueue) -> None:
    """25 of the first 60 looks failed in DarkWake. Skipped now, not failed."""
    _quiet_log(q)
    ex = Executor(q, complete=_silent().complete, online=lambda: False)
    ex.recover()
    ex.drain()
    head = q.head()
    assert ex.notice(now=NOW) is False
    assert q.head() == head


def test_what_was_already_said_is_shown_and_ages_out_after_a_day() -> None:
    said = (
        ((NOW - timedelta(hours=3)).isoformat(), "Your disk is nearly full."),
        ((NOW - timedelta(days=2)).isoformat(), "An old thing."),
    )
    text = notice.situation(now=NOW, said=said)
    assert "You already told them" in text
    assert "Your disk is nearly full." in text
    assert "An old thing." not in text
    assert "You already told them" not in notice.situation(now=NOW)


def test_said_counts_clock_fires_and_looks_but_not_typed_turns(q: EventQueue) -> None:
    typed = q.append(episodes.inbound("hi", channel="tray", at=AT))
    q.append(episodes.completed(for_seq=typed, outcome="spoke", reply="hello!", at=AT))
    fire = q.append(
        episodes.inbound("watch", channel="schedule", schedule_id="w1", at=AT)
    )
    q.append(
        episodes.completed(for_seq=fire, outcome="spoke", reply="News: X.", at=AT)
    )
    look = q.append(episodes.inbound("look", channel=notice.CHANNEL, at=AT))
    q.append(episodes.completed(for_seq=look, outcome="silent", at=AT))

    assert [text for _, text in notice.standing(q.store).said] == ["News: X."]


def test_auditing_the_disk_needs_no_go(q: EventQueue, tmp_path: Path) -> None:
    """The live failure: omega's machine sense flagged the disk LOW, and
    "where is the space going?" blocked on a go for `df` and `du` — two
    programs with no flag that writes. The settings-changing neighbours stay
    asked-about."""
    box = _box(q, tmp_path)
    for argv in (["df", "-h"], ["du", "-x", "-h", "-d", "1", str(tmp_path)]):
        decision = box.classify("run_code", {"argv": argv, "cwd": str(tmp_path)})
        assert decision.tier == tools.EXPLORATION, argv
    for argv in (["sysctl", "-w", "x=1"], ["pmset", "-a", "sleep", "0"]):
        decision = box.classify("run_code", {"argv": argv, "cwd": str(tmp_path)})
        assert decision.tier == tools.EXTERNAL, argv


# --- a reminder coming due always reaches them (DL-075) ----------------------


def _reminder(q: EventQueue, what: str = "take a walk", *, channel: str = "schedule"):
    extra = {"schedule_id": "r9-0001"} if channel == "schedule" else {}
    text = tools.remind_instruction(what)
    seq = q.append(episodes.inbound(text, channel=channel, at=AT, **extra))
    q.claim(seq)
    return q.at(seq)


def test_a_reminder_the_judge_would_silence_is_still_spoken(q: EventQueue) -> None:
    """The live failure: a 6pm walk reminder fired, the judge said SILENT, and
    nobody heard it. Code overrides the verdict; the prompt only asked."""
    fp = provider.FakeProvider(
        {provider.JUDGE: "SILENT", provider.ACT: "Time for your walk."}
    )
    result = run_turn(q, _reminder(q), complete=fp.complete, at=AT)
    assert result.outcome == "spoke"
    assert result.reply == "Time for your walk."
    assert result.verdict.raw == "SILENT", "what the judge said is kept"


def test_a_late_reminder_is_still_a_reminder(q: EventQueue) -> None:
    text = tools.remind_instruction("take a walk") + "\n\n(Scheduled for 18:00, running 9 minutes late.)"
    assert tools.is_reminder({"channel": "schedule", "text": text})


def test_a_reminder_cannot_answer_nothing_new(q: EventQueue) -> None:
    fp = provider.FakeProvider({provider.JUDGE: "ACT", provider.ACT: ["unused"]})
    result = run_turn(
        q, _reminder(q), complete=fp.complete, act=_act_says("(nothing new)"), at=AT
    )
    assert result.outcome == "spoke"


def test_a_watch_may_still_stay_silent(q: EventQueue) -> None:
    fp = provider.FakeProvider({provider.JUDGE: "SILENT", provider.ACT: ["unused"]})
    result = run_turn(q, _put(q, "schedule"), complete=fp.complete, at=AT)
    assert result.outcome == "silent"


def test_only_the_clock_can_fire_a_reminder() -> None:
    """Typing the reminder's words is a message, not a promise coming due."""
    text = tools.remind_instruction("take a walk")
    assert tools.is_reminder({"channel": "schedule", "text": text})
    assert not tools.is_reminder({"channel": "tray", "text": text})
    assert not tools.is_reminder({"channel": "schedule", "text": tools.watch_instruction("news")})
