"""C3, DL-106/114/120 — one composer over the told-ledger, in the executor.

Every clock is pinned, every provider scripted, every fixture synthetic: no
real mail, names or subjects.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from omega import episodes, ledger, notice, outbox, provider, tools
from omega.executor import (
    AUTHORED_REMINDER,
    BRIEF_FOLDED,
    CANCELLED_IN_FLIGHT,
    Executor,
)
from omega.memory import MemoryStore
from omega.queue import EventQueue
from omega.turn import REPEATED, SENSED

WORDS = "Synthetic: the demo build failed again on main."


def local(day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 10, day, hour, minute).astimezone()


@pytest.fixture
def q(store: MemoryStore) -> EventQueue:
    return EventQueue(store)


def payloads(q: EventQueue) -> list[dict]:
    return [episodes.decode(e.payload) for e in q.store.episodes_since(0)]


def kinds(q: EventQueue, kind: str) -> list[dict]:
    return [p for p in payloads(q) if p.get("kind") == kind]


def last_turn(q: EventQueue) -> dict:
    return kinds(q, episodes.TURN_COMPLETED)[-1]


def fake(words: str = WORDS, verdict: str = "NUDGE worth saying",
         calls: list | None = None) -> provider.FakeProvider:
    def judge(role, messages):
        if calls is not None:
            calls.append("judge")
        return verdict

    def act(role, messages):
        if calls is not None:
            calls.append("act")
        return words

    return provider.FakeProvider({provider.JUDGE: judge, provider.ACT: act})


def executor(q: EventQueue, complete=None, *, pace: bool = True) -> Executor:
    ex = Executor(q, complete=(complete or fake()).complete, pace=pace)
    ex.recover()
    return ex


def noted(q: EventQueue, when: datetime, key: str = "ci:demo/build",
          what: str = "Synthetic CI: build failed on main", **kw) -> int:
    return q.append(episodes.observation_recorded(
        key=key, source=kw.pop("source", "ci"), what=what,
        ref=kw.pop("ref", f"{key}@{when.isoformat()}"), at=when.isoformat(), **kw,
    ))


def fire(q: EventQueue, text: str, sid: str, when: datetime) -> int:
    return q.append(episodes.inbound(
        text, channel="schedule", schedule_id=sid, at=when.isoformat(),
    ))


def here(q: EventQueue, when: datetime) -> None:
    q.append(episodes.sense_reported(
        source=episodes.REPORT_PRESENCE, unit=f"mac@{when.timestamp()}",
        device="mac", body={"idle": 5.0}, at=when.isoformat(),
    ))


# --- thinking slots and the brief (DL-114) ----------------------------------


def test_the_thinking_slot_lists_what_is_waiting_under_its_keys(q: EventQueue) -> None:
    q.append(episodes.inbound("hello", channel="tray", at=local(14, 6).isoformat()))
    noted(q, local(14, 8, 30))
    ex = executor(q)
    ex.drain()
    assert ex.notice(now=local(14, 9, 5)) is True
    slot = kinds(q, episodes.MESSAGE_INBOUND)[-1]
    assert slot["thinking"] == "2026-10-14 morning"
    assert slot["keys"] == ["ci:demo/build"]
    assert outbox.KEYED_HEADING in slot["text"] and "[ci:demo/build]" in slot["text"]
    # Without a brief standing, the morning slot is not a brief.
    assert not tools.is_brief(slot)
    ex.drain()
    assert last_turn(q)["outcome"] == "spoke" and last_turn(q)["keys"] == ["ci:demo/build"]
    # Told, with nothing new: the evening slot does not list it again.
    ex.notice(now=local(14, 19, 5))
    assert "keys" not in kinds(q, episodes.MESSAGE_INBOUND)[-1]


def test_with_a_brief_standing_the_morning_slot_is_the_brief(q: EventQueue) -> None:
    q.append(episodes.inbound("hello", channel="tray", at=local(14, 6).isoformat()))
    q.append(episodes.schedule_created(
        id="b1-abcd", instruction=tools.brief_instruction(), cron="0 8 *",
    ))
    ex = executor(q)
    ex.drain()
    ex.notice(now=local(14, 9, 5))
    slot = kinds(q, episodes.MESSAGE_INBOUND)[-1]
    assert slot["thinking"] == "2026-10-14 morning"
    assert slot["text"].startswith(tools.BRIEF_LEAD) and tools.is_brief(slot)
    # The evening slot is not a brief.
    ex.drain()
    ex.notice(now=local(14, 19, 5))
    assert not tools.is_brief(kinds(q, episodes.MESSAGE_INBOUND)[-1])


def test_the_briefs_own_fire_is_folded_into_the_slot_without_a_model_call(
    q: EventQueue,
) -> None:
    calls: list[str] = []
    ex = executor(q, fake(calls=calls))
    fire(q, tools.brief_instruction(), "b1-abcd", local(14, 8))
    ex.drain()
    assert last_turn(q)["outcome"] == "silent" and last_turn(q)["judged"] == BRIEF_FOLDED
    assert calls == []


def test_without_pace_the_brief_fire_still_speaks(q: EventQueue) -> None:
    ex = executor(q, pace=False)
    fire(q, tools.brief_instruction(), "b1-abcd", local(14, 8))
    ex.drain()
    assert last_turn(q)["outcome"] == "spoke"


# --- reminders: DL-075 kept, DL-120, in-flight cancel ------------------------


def test_a_reminder_they_set_speaks_in_quiet_hours_and_names_what_it_covers(
    q: EventQueue,
) -> None:
    set_at = local(14, 20)
    q.append(episodes.inbound("remind me", channel="tray", at=set_at.isoformat()))
    q.append(episodes.schedule_created(
        id="r1-abcd", instruction=tools.remind_instruction("check in for flight ZZ123"),
        every=3 * 3600, at=set_at.isoformat(), once=True,
    ))
    noted(q, local(14, 21), key="travel:zz123", source="travel",
          what="Synthetic: flight ZZ123 boarding pass ready")
    ex = executor(q)
    ex.drain()
    fire(q, tools.remind_instruction("check in for flight ZZ123"), "r1-abcd", local(14, 23, 0))
    ex.drain()
    done = last_turn(q)
    assert done["outcome"] == "spoke"
    assert done["keys"] == ["reminder:r1-abcd", "travel:zz123"]
    # Its keys are told: the composer does not tell them the flight again.
    state = ledger.Ledger.rebuild(q.store).state()
    assert not [i.key for i in state.waiting(local(14, 23, 5))]


def test_a_reminder_they_set_is_never_silenced_as_a_repeat(q: EventQueue) -> None:
    ex = executor(q)
    for day in (14, 15):
        fire(q, tools.remind_instruction("water the plants"), "r1-abcd", local(day, 23, 30))
        ex.drain()
        assert last_turn(q)["outcome"] == "spoke"


def test_a_reminder_omega_set_itself_becomes_a_follow_up_for_the_composer(
    q: EventQueue,
) -> None:
    calls: list[str] = []
    look = q.append(episodes.inbound(
        "a synthetic look", channel=notice.CHANNEL, at=local(14, 9).isoformat(),
    ))
    q.append(episodes.completed(for_seq=look, outcome="silent", at=local(14, 9).isoformat()))
    sid = f"r{look}-abcd"
    q.append(episodes.schedule_created(
        id=sid, instruction=tools.remind_instruction("ask how the demo went"),
        every=3600, at=local(14, 9).isoformat(), once=True,
    ))
    ex = executor(q, fake(calls=calls))
    ex.drain()
    fired = fire(q, tools.remind_instruction("ask how the demo went"), sid, local(14, 10))
    ex.drain()
    assert last_turn(q)["judged"] == AUTHORED_REMINDER and calls == []
    [obs] = kinds(q, episodes.OBSERVATION_RECORDED)
    assert obs["key"] == f"followup:{sid}" and obs["source"] == "followup"
    assert obs["what"] == "ask how the demo went" and obs["evidence_seq"] == fired
    assert "untrusted" not in obs
    # And the composer has it.
    state = ledger.Ledger.rebuild(q.store).state()
    assert [i.key for i in state.waiting(local(14, 10, 5))] == [f"followup:{sid}"]


def test_a_follow_up_omega_set_off_mail_keeps_the_mails_taint(q: EventQueue) -> None:
    look = q.append(episodes.inbound(
        "a synthetic look", channel=notice.CHANNEL, untrusted=tools.INBOX_SOURCE,
        at=local(14, 9).isoformat(),
    ))
    q.append(episodes.completed(
        for_seq=look, outcome="silent", tainted=[tools.INBOX_SOURCE],
        at=local(14, 9).isoformat(),
    ))
    sid = f"r{look}-abcd"
    q.append(episodes.schedule_created(
        id=sid, instruction=tools.remind_instruction("synthetic sender's ask"),
        every=3600, at=local(14, 9).isoformat(), once=True,
    ))
    ex = executor(q)
    ex.drain()
    fire(q, tools.remind_instruction("synthetic sender's ask"), sid, local(14, 10))
    ex.drain()
    [obs] = kinds(q, episodes.OBSERVATION_RECORDED)
    assert obs["untrusted"] == tools.INBOX_SOURCE


def test_a_fire_cancelled_while_it_waited_is_closed_quiet(q: EventQueue) -> None:
    for pace in (True, False):
        calls: list[str] = []
        ex = executor(q, fake(calls=calls), pace=pace)
        sid = f"r1-{'on' if pace else 'off'}"
        fire(q, tools.remind_instruction("call the office"), sid, local(14, 12))
        q.append(episodes.schedule_cancelled(id=sid, at=local(14, 12, 1).isoformat()))
        ex.drain()
        assert last_turn(q)["judged"] == CANCELLED_IN_FLIGHT and calls == []


def test_a_fire_cancelled_then_set_again_still_runs(q: EventQueue) -> None:
    ex = executor(q)
    fire(q, tools.remind_instruction("call the office"), "pr7", local(14, 12))
    q.append(episodes.schedule_cancelled(id="pr7", at=local(14, 12, 1).isoformat()))
    q.append(episodes.schedule_created(
        id="pr7", instruction=tools.remind_instruction("call the office"),
        every=600, at=local(14, 12, 2).isoformat(), once=True,
    ))
    ex.drain()
    assert last_turn(q)["outcome"] == "spoke"


def test_cancelling_a_reminder_closes_its_key(q: EventQueue) -> None:
    q.append(episodes.schedule_created(
        id="r1-abcd", instruction=tools.remind_instruction("call the office"),
        every=600, at=local(14, 12).isoformat(), once=True,
    ))
    assert not ledger.Ledger.rebuild(q.store).state().get("reminder:r1-abcd").closed
    q.append(episodes.schedule_cancelled(id="r1-abcd", at=local(14, 12, 1).isoformat()))
    assert ledger.Ledger.rebuild(q.store).state().get("reminder:r1-abcd").closed


# --- watches: DL-106 ----------------------------------------------------------


def test_a_standing_watch_is_a_sense_and_writes_what_it_found(q: EventQueue) -> None:
    ex = executor(q)
    fired = fire(q, tools.watch_instruction("the synthetic price page"), "w1-abcd", local(14, 12))
    ex.drain()
    done = last_turn(q)
    assert done["outcome"] == "silent" and done["judged"] == SENSED
    [obs] = kinds(q, episodes.OBSERVATION_RECORDED)
    assert obs["key"] == "watch:w1-abcd" and obs["what"] == WORDS
    assert obs["evidence_seq"] == fired
    assert not kinds(q, episodes.MESSAGE_QUEUED)


def test_a_condition_watch_speaks_once_then_not_again_without_news(q: EventQueue) -> None:
    ex = executor(q)
    text = tools.watch_instruction("tell me when the synthetic sender replies")
    assert tools.is_condition_watch({"text": text, "channel": "schedule"})
    fire(q, text, "w2-abcd", local(14, 12))
    ex.drain()
    assert last_turn(q)["outcome"] == "spoke" and last_turn(q)["keys"] == ["watch:w2-abcd"]
    fire(q, text, "w2-abcd", local(14, 13))
    ex.drain()
    assert last_turn(q)["outcome"] == "silent" and last_turn(q)["judged"] == REPEATED
    # Different words are news: it speaks.
    ex2 = executor(q, fake("Synthetic: they replied a second time."))
    fire(q, text, "w2-abcd", local(14, 14))
    ex2.drain()
    assert last_turn(q)["outcome"] == "spoke"


def test_without_pace_a_standing_watch_speaks_as_before(q: EventQueue) -> None:
    ex = executor(q, pace=False)
    fire(q, tools.watch_instruction("the synthetic price page"), "w1-abcd", local(14, 12))
    ex.drain()
    assert last_turn(q)["outcome"] == "spoke"
    assert not kinds(q, episodes.OBSERVATION_RECORDED)


# --- re-check drops, the machine as a sense ----------------------------------


def test_a_held_draft_whose_key_was_told_since_is_dropped_with_why(q: EventQueue) -> None:
    q.append(episodes.inbound("hello", channel="tray", at=local(14, 6).isoformat()))
    noted(q, local(14, 11))
    draft = q.append(episodes.message_queued(
        text=WORDS, source="compose", why=outbox.HOLD_AWAY, keys=["ci:demo/build"],
        at=local(14, 11, 5).isoformat(),
    ))
    # A reminder told them about it after the draft was written.
    tell = q.append(episodes.inbound("a tell", channel="schedule", at=local(14, 11, 10).isoformat()))
    q.append(episodes.completed(
        for_seq=tell, outcome="spoke", reply="told", keys=["ci:demo/build"],
        at=local(14, 11, 10).isoformat(),
    ))
    for name, hour in outbox.SLOTS:
        q.append(episodes.inbound(
            "earlier", channel=notice.CHANNEL, thinking=f"2026-10-14 {name}",
            at=local(14, 0, 1).isoformat(),
        ))
    ex = executor(q, fake(verdict="SILENT nothing"))
    ex.drain()
    here(q, local(14, 13))
    ex.notice(now=local(14, 13, 1))
    [drop] = kinds(q, episodes.MESSAGE_DROPPED)
    assert drop["queued"] == [draft]
    assert drop["why"] == "they were told about it after it was drafted"
    assert outbox.Outbox.rebuild(q.store).state().drafts == ()
    # Nothing left to re-check: no recheck turn was queued for it.
    assert not [p for p in kinds(q, episodes.MESSAGE_INBOUND) if p.get("queued")]


def test_a_newly_flagged_machine_reading_is_written_down_once(q: EventQueue) -> None:
    q.append(episodes.inbound("hello", channel="tray", at=local(14, 6).isoformat()))
    for name, hour in outbox.SLOTS:
        q.append(episodes.inbound(
            "earlier", channel=notice.CHANNEL, thinking=f"2026-10-14 {name}",
            at=local(14, 0, 1).isoformat(),
        ))
    ex = Executor(q, complete=fake(verdict="SILENT nothing").complete, pace=True,
                  machine=lambda: ["battery: 12% - LOW"])
    ex.recover()
    ex.drain()
    ex.notice(now=local(14, 12))
    ex.drain()
    ex.notice(now=local(14, 13))
    ex.drain()
    obs = kinds(q, episodes.OBSERVATION_RECORDED)
    assert [o["key"] for o in obs] == ["machine:battery"]
    assert obs[0]["source"] == "machine" and "LOW" in obs[0]["what"]
    assert "untrusted" not in obs[0]
