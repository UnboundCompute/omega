"""DL-089 — a watch always looks, and "nothing new" counts only after a real look.

Two live failures: calendar and inbox watches judged a bare SILENT in a second
or two with no tool call, so they never opened what they watched; and a self
look judged CHECK that ran no tool, answered "(nothing new)", and was accepted
as an all-clear. **No socket, no key**: every provider and act step here is a
fake.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from omega import episodes, notice, provider, tools
from omega.memory import MemoryStore
from omega.queue import EventQueue
from omega.turn import (
    ACT_THEN_SPEAK,
    UNLOOKED,
    ActResult,
    TurnContext,
    run_turn,
)

AT = "2026-10-11T12:00:00+00:00"
T0 = datetime.fromisoformat(AT)


@pytest.fixture
def q(store: MemoryStore) -> EventQueue:
    return EventQueue(store)


def _put(q: EventQueue, text: str, channel: str, at: str = AT):
    extra = {"schedule_id": "w1"} if channel == "schedule" else {}
    seq = q.append(episodes.inbound(text, channel=channel, at=at, **extra))
    q.claim(seq)
    return q.at(seq)


def _watch(q: EventQueue):
    return _put(q, tools.watch_instruction("their calendar"), "schedule")


def _act(text: str, used: tuple[str, ...], ran: list[int] | None = None):
    def act(ctx: TurnContext) -> ActResult:
        if ran is not None:
            ran.append(ctx.seq)
        return ActResult(tools=used, stop_reason="done", text=text)

    return act


def _fake(judge: str) -> provider.FakeProvider:
    return provider.FakeProvider({provider.JUDGE: judge, provider.ACT: "composed"})


# --- is_watch -----------------------------------------------------------------


def test_a_watch_is_read_off_the_fire_like_a_reminder() -> None:
    text = tools.watch_instruction("their inbox")
    assert tools.is_watch({"channel": "schedule", "text": text})
    assert not tools.is_watch({"channel": "tray", "text": text}), "typed, not fired"
    assert not tools.is_watch({"channel": "schedule", "text": tools.remind_instruction("x")})
    assert not tools.is_watch({"channel": "schedule", "text": tools.brief_instruction()})
    assert "is_watch" in tools.__all__


# --- #1: any verdict on a watch fire becomes ACT ------------------------------


@pytest.mark.parametrize("judged", ["SILENT", "SPEAK", "NOTHING to see"])
def test_a_watch_acts_whatever_the_judge_said(q: EventQueue, judged: str) -> None:
    ran: list[int] = []
    result = run_turn(
        q,
        _watch(q),
        complete=_fake(judged).complete,
        act=_act("A new meeting at 3pm.", ("calendar",), ran),
        at=AT,
    )
    assert ran, "the watch looked"
    assert result.verdict.choice == ACT_THEN_SPEAK
    assert result.verdict.raw == judged, "what the judge said is kept"
    assert result.outcome == "spoke" and result.reply == "A new meeting at 3pm."


# --- #2/#3: quiet needs a look ------------------------------------------------


def test_a_watch_that_looked_and_found_nothing_stays_silent(q: EventQueue) -> None:
    """A 15-minute watch must not ping every 15 minutes (DL-089 #3)."""
    result = run_turn(
        q, _watch(q), complete=_fake("SILENT").complete,
        act=_act("(nothing new)", ("calendar",)), at=AT,
    )
    assert result.outcome == "silent"
    assert result.tools == ("calendar",)


@pytest.mark.parametrize("channel", ["watch", notice.CHANNEL])
def test_nothing_new_without_looking_is_a_failed_turn(q: EventQueue, channel: str) -> None:
    p = _watch(q) if channel == "watch" else _put(q, "what is open", notice.CHANNEL)
    result = run_turn(
        q, p, complete=_fake("CHECK").complete, act=_act("(nothing new)", ()), at=AT
    )
    assert result.outcome == "failed"
    assert UNLOOKED in (result.error or "")
    record = q.at(q.head()).payload
    assert record["kind"] == episodes.TURN_COMPLETED
    assert record["outcome"] == "failed" and UNLOOKED in record["error"]


def test_nothing_new_from_the_reply_model_without_looking_fails_too(q: EventQueue) -> None:
    """The other way to reach the sentinel: act composed nothing and the reply
    model said it. Still no look, so still no all-clear."""
    p = _put(q, "what is open", notice.CHANNEL)
    fp = provider.FakeProvider({provider.JUDGE: "NUDGE", provider.ACT: "(nothing new)"})
    result = run_turn(q, p, complete=fp.complete, at=AT)
    assert result.outcome == "failed" and UNLOOKED in (result.error or "")


def test_a_self_look_that_looked_may_still_end_silent(q: EventQueue) -> None:
    p = _put(q, "what is open", notice.CHANNEL)
    result = run_turn(
        q, p, complete=_fake("CHECK").complete, act=_act("(nothing new)", ("inbox",)), at=AT
    )
    assert result.outcome == "silent"


def test_a_bare_silent_look_is_unchanged(q: EventQueue) -> None:
    """Out of scope (DL-089 "not done"): a look the judge left alone stays a
    silent success with no act step."""
    ran: list[int] = []
    p = _put(q, "what is open", notice.CHANNEL)
    result = run_turn(q, p, complete=_fake("SILENT").complete, act=_act("x", (), ran), at=AT)
    assert result.outcome == "silent" and ran == []


def test_no_look_failures_feed_the_tell_once_count(q: EventQueue) -> None:
    """DL-084 #4 counts failed looks; a look that said nothing new without
    looking is one, so a persistent no-look is told, once."""
    when = datetime(2026, 9, 27, 12, 0, tzinfo=timezone.utc)
    for h in range(3):
        stamp = (when + timedelta(hours=h)).isoformat()
        run_turn(
            q,
            _put(q, "what is open", notice.CHANNEL, at=stamp),
            complete=_fake("CHECK").complete,
            act=_act("(nothing new)", ()),
            at=stamp,
        )
    st = notice.standing(q.store)
    assert st.failed_looks == 3 and UNLOOKED in (st.look_error or "")
    assert notice.owes_outage(st) is True


# --- unchanged ----------------------------------------------------------------


def test_a_reminder_still_speaks_and_never_acts_on_a_silent_verdict(q: EventQueue) -> None:
    ran: list[int] = []
    p = _put(q, tools.remind_instruction("take a walk"), "schedule")
    fp = provider.FakeProvider({provider.JUDGE: "SILENT", provider.ACT: "Time for your walk."})
    result = run_turn(q, p, complete=fp.complete, act=_act("x", (), ran), at=AT)
    assert result.outcome == "spoke" and result.reply == "Time for your walk."
    assert ran == []


def test_a_reminder_or_brief_with_no_tools_still_speaks_the_sentinel(q: EventQueue) -> None:
    for text in (tools.remind_instruction("call the bank"), tools.brief_instruction()):
        result = run_turn(
            q, _put(q, text, "schedule"), complete=_fake("ACT").complete,
            act=_act("(nothing new)", ()), at=AT,
        )
        assert result.outcome == "spoke", text


def test_their_own_message_is_unchanged(q: EventQueue) -> None:
    result = run_turn(
        q, _put(q, "anything new?", "tray"), complete=_fake("ACT").complete,
        act=_act("(nothing new)", ()), at=AT,
    )
    assert result.outcome == "spoke" and result.reply == "(nothing new)"


def test_a_tell_is_unchanged(q: EventQueue) -> None:
    fp = provider.FakeProvider({})
    p = _put(q, tools.tell_instruction("my last 3 looks failed"), "schedule")
    result = run_turn(q, p, complete=fp.complete, at=AT)
    assert result.outcome == "spoke" and fp.calls == []
