"""DL-121: close green CI, urgency classes, a journal-only review, and an act
that looked.

Every clock is pinned, every provider scripted, every fixture synthetic: no
real mail, names or subjects.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import pytest

from omega import cirun, episodes, ledger, outbox, provider, thinking, tools
from omega.executor import Executor
from omega.memory import MemoryStore
from omega.queue import EventQueue

CI_KEY = "ci:demo/build"
CI_WHAT = "CI failed: build on demo-org/demo-app"


def local(day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 10, day, hour, minute).astimezone()


@pytest.fixture
def q(store: MemoryStore) -> EventQueue:
    return EventQueue(store)


def kinds(q: EventQueue, kind: str) -> list[dict]:
    return [p for p in (episodes.decode(e.payload) for e in q.store.episodes_since(0))
            if p.get("kind") == kind]


def last_turn(q: EventQueue) -> dict:
    return kinds(q, episodes.TURN_COMPLETED)[-1]


def noted(q: EventQueue, when: datetime, key: str, what: str, source: str, **kw: Any) -> int:
    return q.append(episodes.observation_recorded(
        key=key, source=source, what=what, ref=f"{key}@{when.isoformat()}",
        at=when.isoformat(), **kw,
    ))


def answer(**fields: Any) -> str:
    return json.dumps({"changed": "", "questions": [], "insights": [], "decisions": [],
                       "offers": [], "message": None, "silence_why": "", **fields})


def scripted(think: Any) -> provider.FakeProvider:
    script = list(think) if isinstance(think, list) else None

    def acting(role, messages):
        if thinking.SYSTEM[:40] in str(messages[0]):
            return script.pop(0) if script is not None else think
        return "Synthetic: noted."

    return provider.FakeProvider({provider.JUDGE: lambda r, m: "SILENT nothing",
                                  provider.ACT: acting})


def executor(q: EventQueue, fp: provider.FakeProvider, **kw: Any) -> Executor:
    ex = Executor(q, complete=fp.complete, pace=True, **kw)
    ex.recover()
    return ex


def hello(q: EventQueue) -> None:
    q.append(episodes.inbound("hello", channel="tray", at=local(14, 6).isoformat()))


# --- A1: a CI key closes when its latest run is green ---------------------------


def test_a_green_latest_run_closes_the_ci_key(q: EventQueue) -> None:
    hello(q)
    noted(q, local(14, 8, 0), CI_KEY, CI_WHAT, "ci")
    asked: list[str] = []

    def runs(item):
        asked.append(item.key)
        return cirun.Run(conclusion="success", at=local(14, 8, 40))

    ex = executor(q, scripted(answer(silence_why="quiet")), ci_runs=runs)
    ex.drain()
    ex.notice(now=local(14, 9, 5))
    assert asked == [CI_KEY]
    item = ledger.Ledger.rebuild(q.store).state().items[CI_KEY]
    assert item.closed and item.what.startswith("CI green again: build")
    assert CI_KEY not in (kinds(q, episodes.MESSAGE_INBOUND)[-1].get("keys") or [])


def test_a_green_run_older_than_the_failure_does_not_close_it(q: EventQueue) -> None:
    hello(q)
    noted(q, local(14, 8, 0), CI_KEY, CI_WHAT, "ci")
    ex = executor(q, scripted(answer(silence_why="quiet")),
                  ci_runs=lambda item: cirun.Run("success", at=local(14, 7, 0)))
    ex.drain()
    ex.notice(now=local(14, 9, 5))
    assert not ledger.Ledger.rebuild(q.store).state().items[CI_KEY].closed


def test_a_run_that_cannot_be_read_leaves_it_open_and_says_so(q: EventQueue) -> None:
    hello(q)
    noted(q, local(14, 8, 0), CI_KEY, CI_WHAT, "ci")
    ex = executor(q, scripted(answer(silence_why="quiet")), ci_runs=lambda item: None)
    ex.drain()
    ex.notice(now=local(14, 9, 5))
    assert not ledger.Ledger.rebuild(q.store).state().items[CI_KEY].closed
    slot = kinds(q, episodes.MESSAGE_INBOUND)[-1]
    assert f"[{CI_KEY}] couldn't check its latest run" in slot["text"]


def test_the_github_reader_fails_closed() -> None:
    class Item:
        what = "Synthetic: not a CI line"
        url = None

    assert cirun.latest(Item(), env={}) is None


# --- A3: urgency classes -----------------------------------------------------------


def test_urgency_classes() -> None:
    assert ledger.urgency_of("Security alert: new sign-in", "no-reply@accounts.google.com") == "urgent"
    assert ledger.urgency_of("Job alert: 30 new roles for you", "alerts@jobs.example") == "noise"
    assert ledger.urgency_of("Weekly newsletter", "news@example.org") == "noise"
    assert ledger.urgency_of("Lunch tomorrow?", "pat@example.org", person=True) == "urgent"
    assert ledger.urgency_of("Your order shipped", "orders@shop.example") == "normal"


def _at_the_cap(q: EventQueue) -> None:
    for _ in range(outbox.MAX_PER_DAY):
        q.append(episodes.message_delivered(for_seq=1, at=local(14, 12).isoformat()))
    # They answered, so the backoff on unanswered messages does not apply.
    q.append(episodes.inbound("thanks", channel="tray", at=local(14, 13).isoformat()))


def test_at_the_cap_a_security_alert_goes_and_a_job_alert_never_composes(q: EventQueue) -> None:
    hello(q)
    sec, job = "mail:sec-1", "mail:job-1"
    words = "Synthetic: a new sign-in to your account. Was it you?"
    ex = executor(q, scripted(answer(
        decisions=[{"key": sec, "decision": "ask", "why": "a sign-in they should confirm"}],
        message=words,
    )))
    ex.drain()
    _at_the_cap(q)
    ex.drain()
    noted(q, local(14, 18, 30), job, "Job alert: 30 new roles", "mail", urgency="noise")
    noted(q, local(14, 18, 31), sec, "Security alert: new sign-in", "mail", urgency="urgent")
    ex.notice(now=local(14, 19, 2))
    slot = kinds(q, episodes.MESSAGE_INBOUND)[-1]
    assert slot["keys"] == [sec], "noise is never composed on its own"
    ex.drain()
    done = last_turn(q)
    assert done["outcome"] == "spoke" and done["keys"] == [sec]
    assert words in done["reply"]


def test_at_the_cap_a_message_naming_no_urgent_key_is_held(q: EventQueue) -> None:
    hello(q)
    sec = "mail:sec-1"
    ex = executor(q, scripted(answer(
        decisions=[{"key": sec, "decision": "fold", "why": "later"}],
        message="Synthetic: nothing pressing.",
    )))
    ex.drain()
    _at_the_cap(q)
    ex.drain()
    noted(q, local(14, 18, 31), sec, "Security alert: new sign-in", "mail", urgency="urgent")
    ex.notice(now=local(14, 19, 2))
    ex.drain()
    assert last_turn(q)["outcome"] == "silent"
    assert kinds(q, episodes.MESSAGE_QUEUED)[-1]["why"] == outbox.HOLD_BUDGET


def test_a_reminder_still_delivers_at_the_cap(q: EventQueue) -> None:
    hello(q)
    q.append(episodes.schedule_created(
        id="r1-abcd", instruction=tools.remind_instruction("water the plants"),
        every=3 * 3600, at=local(14, 20).isoformat(), once=True,
    ))
    ex = executor(q, scripted(answer(silence_why="quiet")))
    ex.drain()
    _at_the_cap(q)
    ex.drain()
    q.append(episodes.inbound(tools.remind_instruction("water the plants"), channel="schedule",
                              schedule_id="r1-abcd", at=local(14, 23, 30).isoformat()))
    ex.drain()
    assert last_turn(q)["outcome"] == "spoke"


# --- A4: the self-review is journal-only -------------------------------------------
# (exercised end to end in test_lessons: a report lands in the journal and is
# never queued.)


def test_the_review_is_never_gated_or_queued(q: EventQueue) -> None:
    from omega import lessons

    ex = executor(q, scripted(answer()))
    review = episodes.inbound(lessons.instruction(), channel="schedule",
                              schedule_id=lessons.REVIEW_ID, at=local(14, 8).isoformat())
    assert ex._gate(review) is None


# --- 8: an act with no lookup is folded --------------------------------------------


def test_an_act_with_no_lookup_is_folded_and_the_journal_says_why(q: EventQueue) -> None:
    hello(q)
    noted(q, local(14, 8, 30), CI_KEY, "Synthetic CI: build failed on main", "ci")
    ex = executor(q, scripted(answer(
        decisions=[{"key": CI_KEY, "decision": "act", "why": "it recurs"}],
        message="Synthetic: the build failed again.",
    )))
    ex.drain()
    ex.notice(now=local(14, 9, 5))
    ex.drain()
    [entry] = kinds(q, episodes.JOURNAL_WRITTEN)
    [d] = entry["decisions"]
    assert d["decision"] == "fold" and d["why"].startswith(thinking.NO_LOOKUP)
    assert "it recurs" in d["why"]
    assert entry["outcome"] == "silent" and entry["why"].startswith(thinking.NO_LOOKUP)
    assert last_turn(q)["outcome"] == "silent"


def test_an_act_that_looked_stays_an_act(q: EventQueue) -> None:
    hello(q)
    noted(q, local(14, 8, 30), CI_KEY, "Synthetic CI: build failed on main", "ci")
    look = provider.Response(
        text="", model="fake", finish_reason="tool_calls", prompt_tokens=10,
        completion_tokens=5,
        tool_calls=(provider.ToolCall(id="c1", name=tools.LOOK_BACK, arguments={}),),
    )
    ex = executor(q, scripted([look, answer(
        decisions=[{"key": CI_KEY, "decision": "act", "why": "it recurs"}],
        message="Synthetic: the build failed again, same step as before.",
    )]))
    ex.drain()
    ex.notice(now=local(14, 9, 5))
    ex.drain()
    [entry] = kinds(q, episodes.JOURNAL_WRITTEN)
    assert entry["decisions"][0]["decision"] == "act"
    assert last_turn(q)["outcome"] == "spoke"
