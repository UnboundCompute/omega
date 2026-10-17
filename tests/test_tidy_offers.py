"""C2 part 1 and B3 — tidy changes omega offers on its own, as proposals.

A standing "anything new" watch on mail or the calendar repeats the look,
which reads both already; a watch that runs every N minutes but names a
time of day drifts from it. Each is *offered* once, under the tidy grant,
and nothing changes until they say yes (DL-085). Read off the running
watches, so no schedule id is named in code. Every name and subject here is
synthetic.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from omega import episodes, provider, tools
from omega.act import act_loop
from omega.derive import OpenWork
from omega.executor import Executor
from omega.followups import Followups
from omega.grants import Grants
from omega.memory import MemoryStore
from omega.queue import EventQueue
from omega.schedule import Schedule, Scheduler
from omega.turn import run_turn

NOW = datetime(2026, 10, 11, 12, 0, tzinfo=timezone.utc)
AT = NOW.isoformat()


@pytest.fixture
def q(store: MemoryStore) -> EventQueue:
    return EventQueue(store)


def payloads(q: EventQueue) -> list[dict]:
    return [episodes.decode(e.payload) for e in q.store.episodes_since(0)]


def kinds(q: EventQueue, kind: str) -> list[dict]:
    return [p for p in payloads(q) if p.get("kind") == kind]


def say(q: EventQueue, text: str, channel: str = "tray") -> int:
    return q.append(episodes.inbound(text, channel=channel, at=AT))


def watch(q: EventQueue, what: str, sid: str = "w1-abcd", every: int = 3600) -> None:
    q.append(episodes.schedule_created(
        id=sid, instruction=tools.watch_instruction(what), every=every, at=AT
    ))


def give(q: EventQueue) -> None:
    seq = say(q, "you have the tidy grant")
    q.append(episodes.grant_given(for_seq=seq, scope="tidy", at=AT))


def executor(q: EventQueue, *, mail: bool = True) -> Executor:
    ex = Executor(
        q, complete=provider.FakeProvider({}).complete, act=act_loop,
        inbox=(lambda now: ()) if mail else None,
    )
    ex.recover()
    return ex


def running(q: EventQueue):
    s = Scheduler(q)
    s.refresh()
    return s.schedules


def deliver(q: EventQueue):
    [seq] = [
        e.seq for e in q.store.episodes_since(0)
        if tools.is_tell(episodes.decode(e.payload))
    ][-1:]
    q.claim(seq)
    fp = provider.FakeProvider({})
    result = run_turn(q, q.at(seq), complete=fp.complete, act=act_loop, at=AT)
    q.finish(seq)
    return result, fp


def answer(q: EventQueue, text: str, act: list[Any]):
    seq = say(q, text)
    sched = Scheduler(q)
    sched.refresh()
    q.claim(seq)
    fp = provider.FakeProvider({provider.JUDGE: ["SPEAK"], provider.ACT: list(act)})
    result = run_turn(
        q, q.at(seq), complete=fp.complete, act=act_loop,
        open_work=OpenWork().advance(q.store, upto=seq).blocks(),
        running=sched.schedules,
        grants=Grants().advance(q.store, upto=seq).state(),
        followups=Followups().advance(q.store, upto=seq).state(),
        at=AT,
    )
    q.finish(seq)
    return result


def _s(what: str, every: int = 3600, **kw: Any) -> Schedule:
    return Schedule(
        id=kw.pop("id", "w1-abcd"), instruction=tools.watch_instruction(what),
        created_at=NOW, every=every, **kw,
    )


# --- what is offered ----------------------------------------------------------


def test_a_standing_mail_watch_is_offered_for_retirement() -> None:
    [(args, ask)] = tools.tidy_offers([_s("my inbox for anything new")], mail=True, calendar=False)
    assert args["change"] == "retire" and args["id"] == "w1-abcd"
    assert ask.endswith(tools.TIDY_STOP_ASK)


def test_a_summary_watch_is_framed_as_their_own_ask() -> None:
    [(_, ask)] = tools.tidy_offers([_s("a Gmail summary of new mail")], mail=True, calendar=False)
    assert ask.startswith(
        "You asked for a Gmail summary every hour; I'd fold it into the morning instead"
    )


def test_a_condition_watch_is_never_offered() -> None:
    offers = tools.tidy_offers(
        [_s("my email for when Zed Example replies"), _s("mail from Quinn about the lease", id="w2")],
        mail=True, calendar=True,
    )
    assert offers == []


def test_without_the_sense_a_mail_watch_is_not_offered() -> None:
    """Only the look reading their mail makes the watch a repeat of it."""
    assert tools.tidy_offers([_s("my inbox for anything new")], mail=False, calendar=False) == []


def test_a_watch_naming_a_clock_time_is_offered_that_time() -> None:
    [(args, ask)] = tools.tidy_offers(
        [_s("the build status at 10am", every=86400)], mail=False, calendar=False
    )
    assert args == {"id": "w1-abcd", "change": "at", "at": "10:00", "why": "they asked for it at 10:00"}
    assert ask.endswith(tools.TIDY_PIN_ASK)


def test_cron_and_reminders_and_unclear_times_are_left_alone() -> None:
    offers = tools.tidy_offers(
        [
            _s("the build status at 10am", cron="0 10 *"),
            _s("the build status at 10am", id="w2", once=True),
            _s("the build at 9am or 5pm", id="w3"),
        ],
        mail=False, calendar=False,
    )
    assert offers == []


@pytest.mark.parametrize("text,want", [
    ("at 10am", "10:00"), ("9:30 pm", "21:30"), ("at 18:05", "18:05"),
    ("noon-ish", None), ("13pm", None), ("10am and 10:00", "10:00"),
])
def test_clock_time_in(text: str, want: Any) -> None:
    assert tools.clock_time_in(text) == want


# --- the executor asks once, and a yes runs it --------------------------------


def test_a_mail_watch_is_proposed_once_and_a_yes_retires_it(q: EventQueue) -> None:
    watch(q, "my inbox for anything new")
    give(q)
    assert executor(q).notice(now=NOW) is True
    [proposal] = kinds(q, episodes.GRANT_PROPOSED)
    [action] = proposal["actions"]
    assert action["tool"] == "tidy" and action["args"]["change"] == "retire"
    assert action["args"]["was"]["id"] == "w1-abcd"

    result, fp = deliver(q)
    assert result.outcome == "spoke" and fp.calls == []
    assert "w1-abcd" in {s.id for s in running(q)}, "nothing changed on its own"

    # Asked once ever: a restarted executor reads it back off the log.
    executor(q).notice(now=NOW)
    assert len(kinds(q, episodes.GRANT_PROPOSED)) == 1

    assert answer(q, "yes", ["Stopped it."]).outcome == "spoke"
    assert "w1-abcd" not in {s.id for s in running(q)}


def test_a_yes_pins_a_drifting_watch_to_its_time(q: EventQueue) -> None:
    watch(q, "the build status at 10am", every=86400)
    give(q)
    assert executor(q, mail=False).notice(now=NOW) is True
    [proposal] = kinds(q, episodes.GRANT_PROPOSED)
    assert proposal["actions"][0]["args"]["cron"] == "0 10 *"
    deliver(q)

    answer(q, "yes", ["Moved it."])
    [s] = [s for s in running(q) if s.id == "w1-abcd"]
    assert s.cron == "0 10 *" and s.every is None


def test_without_the_tidy_grant_nothing_is_offered(q: EventQueue) -> None:
    watch(q, "my inbox for anything new")
    executor(q).notice(now=NOW)
    assert kinds(q, episodes.GRANT_PROPOSED) == []
