"""DL-098 — the person core: thinking time, and picking the moment to speak.

Every clock is pinned, in the machine's own zone (quiet hours and the slots
are statements about the clock on their wall), and every provider is
scripted: no socket, no key. The queue and the budget are folds, so most cases
end by rebuilding the executor over the same store and asking again.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pytest

from omega import (
    episodes, followups, lessons, notice, outbox, projection, provider, thinking, tools,
)
from omega.executor import Executor
from omega.turn import SENSE_LOOK
from omega.memory import MemoryStore
from omega.queue import EventQueue


def local(day: int, hour: int, minute: int = 0) -> datetime:
    """A wall-clock moment where omega runs, with that moment's own offset."""
    return datetime(2026, 10, day, hour, minute).astimezone()


NUDGE = "Your battery is at 20% and unplugged - worth plugging in."


@pytest.fixture
def q(store: MemoryStore) -> EventQueue:
    return EventQueue(store)


def payloads(q: EventQueue) -> list[dict]:
    return [episodes.decode(e.payload) for e in q.store.episodes_since(0)]


def kinds(q: EventQueue, kind: str) -> list[dict]:
    return [p for p in payloads(q) if p.get("kind") == kind]


def last_turn(q: EventQueue) -> dict:
    return kinds(q, episodes.TURN_COMPLETED)[-1]


def fake(verdict: str = "NUDGE the battery is low", words: str = NUDGE,
         seen: list | None = None) -> provider.FakeProvider:
    def judge(role, messages):
        if seen is not None:
            seen.append(str(messages))
        return verdict

    def act(role, messages):
        if thinking.SYSTEM[:40] in str(messages[0]):
            # A thinking slot (C4) is the act role alone: it sees what the
            # judge saw and stays silent where the judge would have.
            if seen is not None:
                seen.append(str(messages))
            if verdict.upper().startswith("SILENT"):
                return verdict
        return words

    return provider.FakeProvider({provider.JUDGE: judge, provider.ACT: act})


def executor(q: EventQueue, complete: Any = None, *, pace: bool = True) -> Executor:
    ex = Executor(q, complete=(complete or fake()).complete, pace=pace)
    ex.recover()
    return ex


def said(q: EventQueue, text: str, when: datetime, channel: str = "tray") -> int:
    return q.append(episodes.inbound(text, channel=channel, at=when.isoformat()))


def at_the_mac(q: EventQueue, when: datetime, idle: float = 5.0) -> int:
    return q.append(episodes.sense_reported(
        source=episodes.REPORT_PRESENCE, unit=f"mac@{when.timestamp()}",
        device="mac", body={"idle": idle}, at=when.isoformat(),
    ))


def thought(q: EventQueue, when: datetime) -> None:
    """Both of that day's thinking slots already thought, long ago, so a look
    that day is an ordinary look."""
    day = when.date().isoformat()
    for name, hour in outbox.SLOTS:
        early = when.replace(hour=0, minute=1).isoformat()
        q.append(episodes.inbound(
            "an earlier thinking turn", channel=notice.CHANNEL,
            thinking=f"{day} {name}", at=early,
        ))
    executor(q, fake("SILENT nothing")).drain()


def start(q: EventQueue, when: datetime) -> Executor:
    """A log with one old exchange in it, the person long gone, and the day's
    thinking already done."""
    said(q, "hello", when - timedelta(hours=3))
    thought(q, when)
    ex = executor(q)
    ex.drain()
    return ex


KEY = "ci:demo/build"


def noted(q: EventQueue, when: datetime, key: str = KEY,
          what: str = "Synthetic CI: build failed on main") -> int:
    """Something a sense wrote down for the composer (C3): under pace a look
    only speaks when it has a keyed item to compose from."""
    return q.append(episodes.observation_recorded(
        key=key, source="ci", what=what, ref=f"{key}@{when.isoformat()}",
        at=when.isoformat(),
    ))


def deliveries(q: EventQueue) -> list[dict]:
    return kinds(q, episodes.MESSAGE_DELIVERED)


# --- the pure rules ----------------------------------------------------------


def test_quiet_hours_are_eleven_at_night_to_eight_in_the_morning() -> None:
    assert outbox.quiet(local(14, 23, 0))
    assert outbox.quiet(local(14, 2, 0))
    assert outbox.quiet(local(14, 7, 59))
    assert not outbox.quiet(local(14, 8, 0))
    assert not outbox.quiet(local(14, 22, 59))


def test_a_slot_is_owed_from_its_hour_for_its_window_only() -> None:
    empty = outbox.State()
    assert outbox.slot_due(empty, local(14, 8, 59)) is None
    assert outbox.slot_due(empty, local(14, 9, 0)) == "2026-10-14 morning"
    assert outbox.slot_due(empty, local(14, 9, 29)) == "2026-10-14 morning"
    # C4: past its window a slot is not served late; it is missed.
    assert outbox.slot_due(empty, local(14, 9, 30)) is None
    assert outbox.slot_due(empty, local(14, 18, 59)) is None
    assert outbox.slot_due(empty, local(14, 19, 0)) == "2026-10-14 evening"
    assert outbox.slot_due(empty, local(14, 23, 30)) is None  # quiet
    done = outbox.State(thought=frozenset({"2026-10-14 morning"}))
    assert outbox.slot_due(done, local(14, 10, 0)) is None
    assert outbox.slot_due(done, local(14, 19, 5)) == "2026-10-14 evening"


def test_hold_is_quiet_then_budget_then_presence() -> None:
    now = local(14, 12, 0)
    day = notice.local_date(now.isoformat())
    empty = outbox.State()
    spent = outbox.State(delivered={day: outbox.MAX_PER_DAY})
    assert outbox.hold(empty, local(14, 23, 30), here=True) == outbox.HOLD_QUIET
    assert outbox.hold(spent, now, here=True) == outbox.HOLD_BUDGET
    assert outbox.hold(spent, now, here=True, slot=True) == outbox.HOLD_BUDGET
    assert outbox.hold(empty, now, here=False) == outbox.HOLD_AWAY
    assert outbox.hold(empty, now, here=False, slot=True) is None
    assert outbox.hold(empty, now, here=True) is None


def test_present_is_the_mac_or_a_recent_message_and_unknown_is_away() -> None:
    now = local(14, 12, 0)
    assert not outbox.present(None, None, now=now)
    assert outbox.present(None, (now - timedelta(minutes=3)).isoformat(), now=now)
    assert not outbox.present(None, (now - timedelta(minutes=11)).isoformat(), now=now)
    assert not outbox.present(None, "not a time", now=now)


# --- holding a look -----------------------------------------------------------


def test_a_plain_look_is_a_sense_and_never_speaks(q: EventQueue) -> None:
    ex = start(q, local(14, 12))
    at_the_mac(q, local(14, 11, 58))
    assert ex.notice(now=local(14, 12)) is True
    look = kinds(q, episodes.MESSAGE_INBOUND)[-1]
    assert "keys" not in look and not look["text"].startswith(outbox.COMPOSE_LEAD)
    ex.drain()
    done = last_turn(q)
    assert done["outcome"] == "silent" and done["judged"] == SENSE_LOOK
    assert not kinds(q, episodes.MESSAGE_QUEUED) and not deliveries(q)


def test_a_keyed_item_waits_while_they_are_away_then_is_composed(q: EventQueue) -> None:
    ex = start(q, local(14, 12))
    noted(q, local(14, 11, 50))
    assert ex.notice(now=local(14, 12)) is True
    look = kinds(q, episodes.MESSAGE_INBOUND)[-1]
    # C3: the composer speaks at a presence moment; away, the look stays a sense.
    assert "keys" not in look
    ex.drain()
    assert last_turn(q)["judged"] == SENSE_LOOK
    assert not kinds(q, episodes.MESSAGE_QUEUED) and not deliveries(q)
    at_the_mac(q, local(14, 13, 0))
    ex.notice(now=local(14, 13, 1))
    assert kinds(q, episodes.MESSAGE_INBOUND)[-1]["keys"] == [KEY]
    ex.drain()
    assert last_turn(q)["outcome"] == "spoke" and len(deliveries(q)) == 1


def held(q: EventQueue, when: datetime, key: str = KEY) -> int:
    """A composed draft already held for them (a slot over budget, say)."""
    noted(q, when - timedelta(minutes=5), key=key)
    return q.append(episodes.message_queued(
        text=NUDGE, source="compose", why=outbox.HOLD_AWAY, keys=[key],
        at=when.isoformat(),
    ))


def test_a_look_while_they_are_here_speaks_and_is_counted(q: EventQueue) -> None:
    ex = start(q, local(14, 12))
    noted(q, local(14, 11, 50))
    at_the_mac(q, local(14, 11, 58))
    ex.notice(now=local(14, 12))
    look = kinds(q, episodes.MESSAGE_INBOUND)[-1]
    assert look["text"].startswith(outbox.COMPOSE_LEAD) and look["keys"] == [KEY]
    assert "Synthetic CI: build failed on main" in look["text"]
    ex.drain()
    assert last_turn(q)["outcome"] == "spoke" and last_turn(q)["keys"] == [KEY]
    assert last_turn(q)["reply"] == NUDGE
    [d] = deliveries(q)
    assert d["queued"] == [] and d["for_seq"] == last_turn(q)["for_seq"]
    assert d["keys"] == [KEY]
    # Told: nothing new on it, so the next look has nothing to compose.
    ex.notice(now=local(14, 12, 30))
    assert "keys" not in kinds(q, episodes.MESSAGE_INBOUND)[-1]


def test_a_held_draft_is_rechecked_once_when_they_come_back(q: EventQueue) -> None:
    ex = start(q, local(14, 12))
    held(q, local(14, 12))
    [draft_seq] = [e.seq for e in q.store.episodes_since(0)
                   if episodes.decode(e.payload)["kind"] == episodes.MESSAGE_QUEUED]
    at_the_mac(q, local(14, 14, 0))
    assert ex.notice(now=local(14, 14, 1)) is True
    ex.drain()
    # B4-lite: not a verbatim tell. The judge re-checks it against `said`.
    recheck = kinds(q, episodes.MESSAGE_INBOUND)[-1]
    assert recheck["channel"] == notice.CHANNEL and recheck["queued"] == [draft_seq]
    assert recheck["text"].startswith(outbox.RECHECK_LEAD)
    assert "thinking" not in recheck
    # Written clean, the draft leaves its re-check untainted (DL-108).
    assert "untrusted" not in recheck
    assert not [p for p in kinds(q, episodes.MESSAGE_INBOUND)
                if p["channel"] == "schedule" and p.get("queued")]
    assert last_turn(q)["outcome"] == "spoke" and last_turn(q)["reply"] == NUDGE
    assert deliveries(q)[-1]["queued"] == [draft_seq]
    # Taken: a second pass, and a restarted executor, send nothing more.
    before = q.head()
    ex.notice(now=local(14, 14, 2))
    assert outbox.Outbox.rebuild(q.store).state().drafts == ()
    assert not any(
        p.get("queued") for p in payloads(q)[before:]
        if p["kind"] == episodes.MESSAGE_INBOUND
    )


def test_a_recheck_of_a_draft_written_off_mail_names_that_taint(q: EventQueue) -> None:
    start(q, local(14, 12))
    q.append(episodes.message_queued(
        text="A synthetic mail looked urgent.", source="look",
        why=outbox.HOLD_AWAY, tainted=[tools.INBOX_SOURCE],
        at=local(14, 12).isoformat(),
    ))
    at_the_mac(q, local(14, 14, 0))
    ex = executor(q)
    assert ex.notice(now=local(14, 14, 1)) is True
    recheck = kinds(q, episodes.MESSAGE_INBOUND)[-1]
    assert recheck["text"].startswith(outbox.RECHECK_LEAD)
    assert recheck["untrusted"] == tools.INBOX_SOURCE
    ex.drain()
    assert last_turn(q)["tainted"] == [tools.INBOX_SOURCE]


def test_a_recheck_showing_a_reply_drawn_from_mail_names_that_taint(q: EventQueue) -> None:
    start(q, local(14, 12))
    # Yesterday evening a look that read their mail told them something; the
    # re-check lists it under "what you last told them", outside any fence.
    look = q.append(episodes.inbound(
        "a synthetic look", channel=notice.CHANNEL, at=local(13, 20).isoformat(),
    ))
    q.append(episodes.completed(
        for_seq=look, outcome="spoke", reply="A synthetic sender wants an answer.",
        tainted=[tools.INBOX_SOURCE], at=local(13, 20).isoformat(),
    ))
    q.append(episodes.message_queued(
        text=NUDGE, source="look", why=outbox.HOLD_AWAY, at=local(14, 12).isoformat(),
    ))
    at_the_mac(q, local(14, 14, 0))
    ex = executor(q)
    assert ex.notice(now=local(14, 14, 1)) is True
    recheck = kinds(q, episodes.MESSAGE_INBOUND)[-1]
    assert recheck["text"].startswith(outbox.RECHECK_LEAD)
    assert "A synthetic sender wants an answer." in recheck["text"]
    assert recheck["untrusted"] == tools.INBOX_SOURCE


def test_a_recheck_that_stays_silent_drops_the_drafts_stamped_now(q: EventQueue) -> None:
    ex = start(q, local(14, 12))
    held(q, local(14, 12))
    at_the_mac(q, local(14, 14, 0))
    ex = executor(q, fake("SILENT already told them"))
    assert ex.notice(now=local(14, 14, 1)) is True
    ex.drain()
    [drop] = kinds(q, episodes.MESSAGE_DROPPED)
    assert drop["why"] == outbox.DROPPED_WHY["recheck"]
    # Stamped when it happened (C0), not with the recheck's moment.
    assert drop["at"] != local(14, 14, 1).isoformat()
    assert outbox.Outbox.rebuild(q.store).state().drafts == ()
    assert not deliveries(q)


def test_nothing_held_goes_out_within_ten_minutes_of_them_speaking(q: EventQueue) -> None:
    ex = start(q, local(14, 12))
    held(q, local(14, 12))
    said(q, "back now", local(14, 14, 0))
    at_the_mac(q, local(14, 14, 0))
    before = q.head()
    assert ex.notice(now=local(14, 14, 9)) is False
    assert not any(p.get("queued") is not None for p in payloads(q)[before:])
    box = outbox.Outbox.rebuild(q.store).state()
    assert outbox.hold(box, local(14, 14, 9), here=True) == outbox.HOLD_SPOKE
    assert outbox.hold(box, local(14, 14, 10), here=True) is None
    assert outbox.hold(box, local(14, 14, 9), here=True, slot=True) == outbox.HOLD_SPOKE


def test_several_held_drafts_go_out_as_one_message() -> None:
    a = outbox.Draft(seq=5, text="The disk is nearly full.", source="look", at="x")
    b = outbox.Draft(seq=9, text="The flight watch keeps failing.", source="followup", at="x")
    assert outbox.merged([a]) == "The disk is nearly full."
    both = outbox.merged([a, b])
    assert both.count("\n\n") == 2 and "disk" in both and "flight" in both


def test_nothing_is_delivered_in_quiet_hours_even_with_them_here(q: EventQueue) -> None:
    ex = start(q, local(14, 12))
    held(q, local(14, 12))
    at_the_mac(q, local(14, 23, 30))
    before = q.head()
    ex.notice(now=local(14, 23, 31))
    assert not any(
        p.get("queued") for p in payloads(q)[before:]
    ), "a held draft went out at 23:31"
    assert len(outbox.Outbox.rebuild(q.store).state().drafts) == 1


def test_the_day_budget_survives_a_restart(q: EventQueue) -> None:
    ex = start(q, local(14, 10))
    for hour in (10, 12):
        noted(q, local(14, hour, 0), key=f"ci:demo/build-{hour}")
        at_the_mac(q, local(14, hour, 0))
        ex.notice(now=local(14, hour, 1))
        ex.drain()
    assert len(deliveries(q)) == 2
    # A new process over the same log: the third is held, here or not.
    ex2 = executor(q)
    noted(q, local(14, 15, 0), key="ci:demo/build-15")
    at_the_mac(q, local(14, 15, 0))
    ex2.notice(now=local(14, 15, 1))
    ex2.drain()
    assert last_turn(q)["outcome"] == "silent"
    # Over budget nothing is composed: the item waits in the ledger instead.
    assert "keys" not in kinds(q, episodes.MESSAGE_INBOUND)[-1]
    assert len(deliveries(q)) == 2
    # Tomorrow has its own budget, and the waiting item is told then.
    at_the_mac(q, local(15, 12, 0))
    thought(q, local(15, 12))
    ex2.notice(now=local(15, 12, 1))
    ex2.drain()
    assert len(deliveries(q)) == 3
    assert deliveries(q)[-1]["keys"] == ["ci:demo/build-15"]


# --- thinking time --------------------------------------------------------------


def test_thinking_time_is_a_look_over_everything_open(q: EventQueue) -> None:
    seen: list[str] = []
    said(q, "hello", local(14, 6))
    ex = executor(q, fake("SILENT nothing needs them", seen=seen))
    ex.drain()
    q.append(episodes.message_queued(
        text="The disk is nearly full.", source="look", why=outbox.HOLD_AWAY,
        at=local(14, 2).isoformat(),
    ))
    draft = q.head()
    assert ex.notice(now=local(14, 9, 5)) is True
    look = kinds(q, episodes.MESSAGE_INBOUND)[-1]
    assert look["channel"] == notice.CHANNEL
    assert look["thinking"] == "2026-10-14 morning" and look["queued"] == [draft]
    assert look["text"].startswith("Thinking time (morning).")
    assert "Most days the right answer is nothing" in look["text"]
    assert "The disk is nearly full." in look["text"]
    assert len(look["text"]) <= outbox.MAX_THINKING_CHARS
    seen.clear()
    ex.drain()
    # C4: the slot is its own thinking pass (no look judge), and journalled.
    assert thinking.SYSTEM[:40] in seen[0]
    assert "Do they need a nudge right now?" not in seen[0]
    assert last_turn(q)["outcome"] == "silent"
    [entry] = kinds(q, episodes.JOURNAL_WRITTEN)
    assert entry["slot"] == "2026-10-14 morning" and entry["outcome"] == "silent"
    [drop] = kinds(q, episodes.MESSAGE_DROPPED)
    assert drop["queued"] == [draft]
    assert outbox.Outbox.rebuild(q.store).state().drafts == ()


def test_a_thinking_turn_speaks_at_its_slot_without_them_here(q: EventQueue) -> None:
    said(q, "hello", local(14, 6))
    ex = executor(q)
    ex.drain()
    ex.notice(now=local(14, 19, 2))
    ex.drain()
    assert last_turn(q)["outcome"] == "spoke"
    assert deliveries(q)[-1]["for_seq"] == last_turn(q)["for_seq"]


def test_a_slot_is_thought_once_even_across_a_restart(q: EventQueue) -> None:
    said(q, "hello", local(14, 6))
    ex = executor(q, fake("SILENT nothing"))
    ex.drain()
    ex.notice(now=local(14, 9, 5))
    ex.drain()
    ex2 = executor(q, fake("SILENT nothing"))
    ex2.notice(now=local(14, 9, 40))
    ex2.drain()
    looks = [p for p in kinds(q, episodes.MESSAGE_INBOUND) if p["channel"] == notice.CHANNEL]
    assert [p.get("thinking") for p in looks] == ["2026-10-14 morning", None]


def test_over_budget_the_thinking_message_is_carried_to_tomorrow(q: EventQueue) -> None:
    said(q, "hello", local(14, 6))
    ex = executor(q)
    ex.drain()
    day = local(14, 12).isoformat()
    for n in range(outbox.MAX_PER_DAY):
        q.append(episodes.message_delivered(for_seq=1, at=day))
    # They answered, so M3' backoff on unanswered messages does not apply.
    said(q, "thanks", local(14, 13))
    ex.drain()
    q.append(episodes.message_queued(
        text="An older draft.", source="look", why=outbox.HOLD_AWAY, at=day,
    ))
    older = q.head()
    ex.notice(now=local(14, 19, 2))
    ex.drain()
    assert last_turn(q)["outcome"] == "silent"
    held = kinds(q, episodes.MESSAGE_QUEUED)[-1]
    assert held["source"] == "thinking" and held["why"] == outbox.HOLD_BUDGET
    assert kinds(q, episodes.MESSAGE_DROPPED)[-1]["queued"] == [older]
    # Tomorrow morning's thinking turn is shown it, and sends it.
    ex.notice(now=local(15, 9, 3))
    look = kinds(q, episodes.MESSAGE_INBOUND)[-1]
    assert look["thinking"] == "2026-10-15 morning"
    assert NUDGE in look["text"]
    ex.drain()
    assert last_turn(q)["outcome"] == "spoke"
    assert deliveries(q)[-1]["queued"] == [look["queued"][0]]


def test_thinking_over_its_daily_spend_falls_back_to_the_look(q: EventQueue) -> None:
    said(q, "hello", local(14, 6))
    ex = executor(q, fake("SILENT nothing"))
    ex.drain()
    ex.notice(now=local(14, 9, 5))
    ex.drain()
    # The morning turn cost more than the day allows (written as its record).
    seq = [p for p in kinds(q, episodes.MESSAGE_INBOUND) if p.get("thinking")]
    assert seq
    state = outbox.Outbox.rebuild(q.store).state()
    assert state.spending  # metered, or counted as unmetered
    q2_spent = outbox.State(
        thought=state.thought,
        spending={notice.local_date(local(14, 19).isoformat()): outbox.THINKING_DAILY_TOKENS},
    )
    assert not outbox.thinking_left(q2_spent, local(14, 19, 2))
    assert outbox.thinking_left(outbox.State(), local(14, 19, 2))


# --- the slot clock (C4) -----------------------------------------------------


def failures(q: EventQueue) -> list[dict]:
    return kinds(q, episodes.THINKING_FAILED)


def test_a_slot_runs_on_its_own_clock_not_the_look_gap(q: EventQueue) -> None:
    said(q, "hello", local(14, 6))
    ex = executor(q, fake("SILENT nothing"))
    ex.drain()
    # An ordinary look a minute before the slot: the look gap would hold
    # the next look for half an hour, and used to hold the slot with it.
    ex.notice(now=local(14, 8, 59))
    ex.drain()
    assert ex.notice(now=local(14, 9, 0)) is True
    ex.drain()
    looks = [p for p in kinds(q, episodes.MESSAGE_INBOUND) if p["channel"] == notice.CHANNEL]
    assert [p.get("thinking") for p in looks] == [None, "2026-10-14 morning"]


def test_a_slot_that_never_ran_is_written_down_and_not_served_late(q: EventQueue) -> None:
    said(q, "hello", local(14, 6))
    ex = executor(q, fake("SILENT nothing"))
    ex.drain()
    ex.notice(now=local(14, 11, 0))
    ex.drain()
    assert [(f["slot"], f["why"]) for f in failures(q)] == [("2026-10-14 morning", "no entry")]
    looks = [p for p in kinds(q, episodes.MESSAGE_INBOUND) if p["channel"] == notice.CHANNEL]
    assert all(p.get("thinking") is None for p in looks)
    # Once: a second pass and a restart write nothing more for it.
    again = executor(q, fake("SILENT nothing"))
    again.notice(now=local(14, 12, 0))
    again.drain()
    assert len(failures(q)) == 1
    ex.drain()
    # The next boundary is the retry.
    ex.notice(now=local(14, 19, 1))
    assert kinds(q, episodes.MESSAGE_INBOUND)[-1].get("thinking") == "2026-10-14 evening"


def test_no_slot_before_the_log_began_is_owed(q: EventQueue) -> None:
    said(q, "hello", local(14, 12))
    ex = executor(q, fake("SILENT nothing"))
    ex.drain()
    ex.notice(now=local(14, 13, 0))
    assert failures(q) == []


def test_a_slot_over_its_spend_is_written_down_and_the_look_runs(
    q: EventQueue, monkeypatch: pytest.MonkeyPatch
) -> None:
    said(q, "hello", local(14, 6))
    ex = executor(q, fake("SILENT nothing"))
    ex.drain()
    monkeypatch.setattr(outbox, "THINKING_DAILY_TOKENS", 0)
    assert ex.notice(now=local(14, 9, 2)) is True
    assert [(f["slot"], f["why"]) for f in failures(q)] == [("2026-10-14 morning", "spend exceeded")]
    look = kinds(q, episodes.MESSAGE_INBOUND)[-1]
    assert look["channel"] == notice.CHANNEL and look.get("thinking") is None


def test_a_slot_waits_for_a_conversation_then_runs(q: EventQueue) -> None:
    said(q, "hello", local(14, 6))
    ex = executor(q, fake("SILENT nothing"))
    ex.drain()
    said(q, "quick question", local(14, 9, 0))
    ex.drain()
    before = q.head()
    assert ex.notice(now=local(14, 9, 3)) is False
    assert payloads(q)[before:] == []
    # Fifteen minutes on, it runs, mid-conversation or not.
    said(q, "and another", local(14, 9, 12))
    ex.drain()
    assert ex.notice(now=local(14, 9, 15)) is True
    assert kinds(q, episodes.MESSAGE_INBOUND)[-1].get("thinking") == "2026-10-14 morning"
    assert failures(q) == []


def test_a_slot_put_off_until_its_window_closed_says_so(q: EventQueue) -> None:
    said(q, "hello", local(14, 6))
    ex = Executor(q, complete=fake("SILENT nothing").complete, pace=True,
                  online=lambda: False)
    ex.recover()
    ex.drain()
    assert ex.notice(now=local(14, 9, 1)) is False
    ex.notice(now=local(14, 9, 40))
    assert [(f["slot"], f["why"]) for f in failures(q)] == [
        ("2026-10-14 morning", "deferral expired")
    ]


# --- follow-ups -----------------------------------------------------------------


def _owe_followup(q: EventQueue, when: datetime) -> None:
    sid = "w1-abcd"
    q.append(episodes.schedule_created(
        id=sid, instruction=tools.watch_instruction("the flight prices"), every=900,
        at=when.isoformat(),
    ))
    for _ in range(followups.FAIL_AFTER):
        seq = q.append(episodes.inbound(
            tools.watch_instruction("the flight prices"), channel="schedule",
            schedule_id=sid, at=when.isoformat(),
        ))
        q.append(episodes.completed(
            for_seq=seq, outcome="failed", error="RuntimeError: down", at=when.isoformat(),
        ))


def test_a_follow_up_while_they_are_away_is_held(q: EventQueue) -> None:
    said(q, "hello", local(14, 6))
    _owe_followup(q, local(14, 11))
    ex = executor(q)
    ex.drain()
    before = len(kinds(q, episodes.MESSAGE_INBOUND))
    assert ex.notice(now=local(14, 12)) is True
    assert len(kinds(q, episodes.MESSAGE_INBOUND)) == before  # no tell
    [draft] = kinds(q, episodes.MESSAGE_QUEUED)
    assert draft["source"] == "followup" and "has failed 3 times" in draft["text"]
    assert kinds(q, episodes.FAILURE_TOLD)  # owed no longer: it is held


def test_a_follow_up_with_them_here_is_told_as_before_and_counted(q: EventQueue) -> None:
    said(q, "hello", local(14, 6))
    _owe_followup(q, local(14, 11))
    ex = executor(q)
    ex.drain()
    at_the_mac(q, local(14, 11, 59))
    ex.notice(now=local(14, 12))
    tell = kinds(q, episodes.MESSAGE_INBOUND)[-1]
    assert tell["queued"] == []
    assert tools.is_tell(tell) and "has failed 3 times" in tell["text"]
    ex.drain()
    assert last_turn(q)["outcome"] == "spoke"
    assert deliveries(q)[-1]["for_seq"] == last_turn(q)["for_seq"]


# --- the self-review ----------------------------------------------------------


def test_the_self_review_is_gated_like_a_look(q: EventQueue) -> None:
    ex = executor(q)
    event = episodes.inbound(
        f"{lessons.REVIEW_LEAD} x", channel="schedule", schedule_id=lessons.REVIEW_ID,
        at=local(18, 20, 0).isoformat(),
    )
    gate = ex._gate(event)
    assert gate is not None and gate.source == "review" and gate.hold == outbox.HOLD_AWAY


# --- what is exempt, and equivalence ----------------------------------------------


def test_exempt_events_get_no_gate(q: EventQueue) -> None:
    ex = executor(q)
    at = local(14, 23, 30).isoformat()  # quiet hours: a gate would hold them
    for event in (
        episodes.inbound(tools.remind_instruction("call the office"),
                         channel="schedule", schedule_id="r1-abcd", at=at),
        episodes.inbound(tools.tell_instruction("I lost the Google sign-in."),
                         channel="schedule", at=at),
        episodes.inbound(tools.watch_instruction("the flight prices"),
                         channel="schedule", schedule_id="w1-abcd", at=at),
        episodes.inbound("hi", channel="tray", at=at),
    ):
        assert ex._gate(event) is None, event["text"]


def _strip(ps: list[dict]) -> list[dict]:
    """The terminal records' wall-clock stamp (`_handle` passes none) aside,
    and the told-ledger's keys (C3), which pace adds and nothing else reads
    as behaviour: what is said, when, and to whom must be the same."""
    out = []
    for p in ps:
        p = dict(p)
        p.pop("keys", None)
        if p["kind"] == episodes.TURN_COMPLETED:
            p.pop("at")
        out.append(p)
    return out


def _keys(store_dir) -> set[str]:
    with MemoryStore.open(store_dir) as s:
        return {k for e in s.episodes_since(0) for k in episodes.decode(e.payload).get("keys") or ()}


def _exempt_run(store_dir, pace: bool) -> list[dict]:
    with MemoryStore.open(store_dir) as s:
        q = EventQueue(s)
        night = local(14, 23, 30).isoformat()
        said(q, "hello", local(14, 6))
        q.append(episodes.inbound(
            tools.remind_instruction("call the office"), channel="schedule",
            schedule_id="r1-abcd",
            at=night,
        ))
        q.append(episodes.inbound(
            tools.tell_instruction("I lost the Google sign-in."), channel="schedule",
            at=night,
        ))
        ex = Executor(q, complete=fake().complete, pace=pace)
        ex.recover()
        ex.drain()
        return _strip(payloads(q))


def test_exempt_paths_are_byte_identical_with_the_outbox_on(tmp_path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    off = _exempt_run(tmp_path / "a", pace=False)
    on = _exempt_run(tmp_path / "b", pace=True)
    assert on == off
    # The keys are there under pace: the reminder named its own (DL-075 path).
    assert _keys(tmp_path / "b") >= {"reminder:r1-abcd"}
    assert not [p for p in on if p["kind"] in episodes.OUTBOX_KINDS]
    assert [p["outcome"] for p in on if p["kind"] == episodes.TURN_COMPLETED] == [
        "spoke", "spoke", "spoke",
    ]


def test_off_writes_nothing_new_and_a_look_reads_as_before(q: EventQueue) -> None:
    ex = executor(q, pace=False)
    said(q, "hello", local(14, 6))
    ex.drain()
    ex.notice(now=local(14, 9, 5))  # a slot, were it on
    ex.drain()
    look = kinds(q, episodes.MESSAGE_INBOUND)[-1]
    assert "queued" not in look and "thinking" not in look
    assert not look["text"].startswith("Thinking time")
    assert last_turn(q)["outcome"] == "spoke"
    assert not [p for p in payloads(q) if p["kind"] in episodes.OUTBOX_KINDS]


def test_a_look_with_nothing_held_reads_the_same_on_and_off(tmp_path) -> None:
    texts = []
    for name, pace in (("a", False), ("b", True)):
        (tmp_path / name).mkdir()
        with MemoryStore.open(tmp_path / name) as s:
            q = EventQueue(s)
            said(q, "hello", local(14, 6))
            q.append(episodes.inbound(
                "an earlier look", channel=notice.CHANNEL, thinking="2026-10-14 morning",
                at=local(14, 9).isoformat(),
            ))
            q.append(episodes.completed(for_seq=2, outcome="silent", at=local(14, 9).isoformat()))
            ex = Executor(q, complete=fake().complete, pace=pace)
            ex.recover()
            ex.notice(now=local(14, 12))
            texts.append(kinds(q, episodes.MESSAGE_INBOUND)[-1])
    assert texts[0] == texts[1]
    assert notice.situation(now=local(14, 12)) == notice.situation(now=local(14, 12), held=())


def test_outbox_records_round_trip_and_reject_bad_shapes() -> None:
    at = local(14, 12).isoformat()
    for p in (
        episodes.message_queued(text="x", source="look", why="w", at=at),
        episodes.message_queued(text="x", source="followup", why="w", for_seq=3, at=at),
        episodes.message_delivered(for_seq=3, queued=[1, 2], at=at),
        episodes.message_dropped(queued=[1], why="w", at=at),
    ):
        assert episodes.decode(episodes.encode(p)) == p
    with pytest.raises(episodes.BadPayload):
        episodes.message_queued(text="x", source="nobody", why="w", at=at)
    with pytest.raises(episodes.BadPayload):
        episodes.message_queued(text="  ", source="look", why="w", at=at)
    with pytest.raises(episodes.BadPayload):
        episodes.message_delivered(for_seq=3, queued=[0], at=at)
    with pytest.raises(episodes.BadPayload):
        episodes.inbound("x", channel="schedule", queued=[True], at=at)


def _leave_run(store_dir, pace: bool) -> list[dict]:
    """A leave-now alarm due at 01:00 with no Maps: its labelled estimate ping
    lands in quiet hours (DL-093), and its tell's turn is drained."""
    arrive = local(15, 1, 0)
    with MemoryStore.open(store_dir) as s:
        q = EventQueue(s)
        said(q, "hello", local(14, 20))
        q.append(episodes.leave_set(
            id="l1", destination="the airport", arrive_by=arrive.isoformat(),
            origin="the office", origin_via="said", for_seq=1,
            at=local(14, 20).isoformat(),
        ))
        ex = Executor(q, complete=fake().complete, pace=pace)
        ex.recover()
        t = local(14, 23, 0)
        while t <= arrive:
            ex.leave_now(now=t.isoformat())
            ex.drain()
            t += timedelta(minutes=5)
        return _strip(payloads(q))


def test_a_leave_ping_in_quiet_hours_is_identical_with_the_outbox_on(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.delenv("OMEGA_MCP_SERVERS", raising=False)
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    off = _leave_run(tmp_path / "a", pace=False)
    on = _leave_run(tmp_path / "b", pace=True)
    assert "leave:l1" in _keys(tmp_path / "b") and not _keys(tmp_path / "a")
    [ping] = [p for p in on if p["kind"] == episodes.LEAVE_PINGED]
    assert outbox.quiet(datetime.fromisoformat(ping["at"]))
    assert on == off
    assert not [p for p in on if p["kind"] in episodes.OUTBOX_KINDS]
    assert [p["outcome"] for p in on if p["kind"] == episodes.TURN_COMPLETED][-1] == "spoke"


def test_backoff_holds_unanswered_messages_but_never_the_brief() -> None:
    # M3': two unanswered unprompted messages an hour ago hold the next one,
    # but the morning brief they asked for still goes (DL-080).
    sent = local(15, 8)
    box = outbox.State(unanswered=outbox.BACKOFF_AFTER, last_sent_at=sent.isoformat())
    now = local(15, 9, 3)
    assert outbox.hold(box, now, here=True, slot=True) == outbox.HOLD_BACKOFF
    assert outbox.hold(box, now, here=True, slot=True, brief=True) is None
    assert outbox.hold(box, sent + timedelta(days=1, minutes=1), here=True, slot=True) is None
