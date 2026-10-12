"""Approvals that do not stall (DL-087 #8).

A gated proposal is filed and the turn ends having spoken; several can wait
at once; each is answered by its id, or by a bare yes only while it is the one
open question and the newest; the brief lists every one still waiting; one
that no longer applies is withdrawn, in the log, and told once.

Fakes only: every model answer is scripted on `provider.FakeProvider`, the
store is a temp directory, and nothing here reaches a network or a key.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from omega import discord as dc
from omega import episodes, grants, projection, provider, tools
from omega.act import act_loop
from omega.derive import OpenWork
from omega.executor import Executor
from omega.grants import Grants
from omega.memory import MemoryStore
from omega.provider import ToolCall
from omega.queue import EventQueue
from omega.schedule import Scheduler
from omega.tools import ToolBox
from omega.turn import NOT_RECALLED, run_turn

AT = "2026-10-11T09:00:00+00:00"
WEEK_ON = "2026-10-19T09:00:00+00:00"


@pytest.fixture(autouse=True)
def clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every episode built without an ``at`` is stamped :data:`AT`.

    A proposal filed by the act loop takes its ``at`` from
    :func:`episodes.now`, and the brief says ``since <that day>`` and marks
    it old a week on. Left on the wall clock, both expectations held only
    while today was the day the fixtures were written.
    """
    monkeypatch.setattr(episodes, "now", lambda: AT)


@pytest.fixture
def q(store: MemoryStore) -> EventQueue:
    return EventQueue(store)


def call(name: str, **arguments: Any) -> ToolCall:
    return ToolCall(id="c1", name=name, arguments=dict(arguments))


def say(q: EventQueue, text: str, channel: str = "tray", at: str = AT, **extra: Any) -> int:
    return q.append(episodes.inbound(text, channel=channel, at=at, **extra))


def turn(q: EventQueue, seq: int, *, judge: Any = "ACT", act: list[Any]):
    """One turn as the executor runs it: the views folded through this event,
    then the withdrawals the executor writes after it."""
    sched = Scheduler(q)
    sched.refresh()
    q.claim(seq)
    fp = provider.FakeProvider({provider.JUDGE: [judge], provider.ACT: list(act)})
    result = run_turn(
        q, q.at(seq), complete=fp.complete, act=act_loop,
        open_work=OpenWork().advance(q.store, upto=seq).blocks(),
        running=sched.schedules,
        grants=Grants().advance(q.store, upto=seq).state(),
        at=AT,
    )
    q.finish(seq)
    grants.withdraw_lapsed(q.append, state(q))
    return result, fp


def state(q: EventQueue) -> grants.State:
    return Grants.rebuild(q.store).state()


def kinds(q: EventQueue, kind: str) -> list[dict]:
    return [p.payload for p in q.recent(1000) if p.payload.get("kind") == kind]


def running_ids(q: EventQueue) -> set[str]:
    s = Scheduler(q)
    s.refresh()
    return {x.id for x in s.schedules}


def watch(q: EventQueue, sid: str, what: str) -> None:
    q.append(episodes.schedule_created(
        id=sid, instruction=tools.watch_instruction(what), every=900, at=AT
    ))


def give(q: EventQueue, scope: str = "tidy") -> int:
    seq = say(q, f"you have the {scope} grant")
    q.append(episodes.grant_given(for_seq=seq, scope=scope, at=AT))
    q.append(episodes.completed(for_seq=seq, outcome="spoke", reply="ok", at=AT))
    return seq


def propose(q: EventQueue, sid: str, at: str = AT) -> tuple[Any, int]:
    """A prompted turn proposes retiring ``sid``; returns (result, proposal id)."""
    seq = say(q, f"tidy {sid}", at=at)
    result, _ = turn(q, seq, act=[call("tidy", id=sid, change="retire", why="it is stale")])
    [p] = [p for p in kinds(q, episodes.GRANT_PROPOSED) if p["for_seq"] == seq]
    pseq = next(x.seq for x in q.recent(1000) if x.payload == p)
    return result, pseq


def brief(q: EventQueue, at: str = AT) -> int:
    return say(q, tools.brief_instruction(), channel="schedule", schedule_id="b1-0001", at=at)


@pytest.fixture
def two(q: EventQueue) -> tuple[int, int]:
    """Two watches, the tidy grant, and two proposals pending at once."""
    watch(q, "w1-aaaa", "the calendar")
    watch(q, "w2-bbbb", "the inbox")
    give(q)
    _, a = propose(q, "w1-aaaa")
    _, b = propose(q, "w2-bbbb")
    return a, b


# --- filed, not blocked -------------------------------------------------------


def test_a_proposal_is_filed_and_the_turn_speaks_with_its_id(q: EventQueue) -> None:
    watch(q, "w1-aaaa", "the calendar")
    give(q)
    result, pid = propose(q, "w1-aaaa")
    assert result.outcome == "spoke" and result.needs is None
    assert f'Reply "yes p{pid}" or "no p{pid}".' in result.reply
    assert kinds(q, episodes.TURN_BLOCKED) == []
    assert not OpenWork().advance(q.store).blocks(), "nothing is left blocked"
    assert "w1-aaaa" in running_ids(q), "nothing ran on its own"


def test_several_proposals_wait_at_once(q: EventQueue, two: tuple[int, int]) -> None:
    a, b = two
    assert [p.seq for p in state(q).pending] == [a, b]


# --- answers ------------------------------------------------------------------


def test_one_answer_can_name_several_ids(q: EventQueue, two: tuple[int, int]) -> None:
    a, b = two
    result, _ = turn(q, say(q, f"yes p{a} p{b}"), judge="SPEAK", act=["Both retired."])
    assert result.outcome == "spoke"
    assert running_ids(q) == set()
    assert sorted(x["proposal"] for x in kinds(q, episodes.GRANT_ANSWERED)) == [a, b]
    assert sorted(x["args"]["id"] for x in kinds(q, episodes.GRANT_ACTED)) == ["w1-aaaa", "w2-bbbb"]
    assert state(q).pending == ()


def test_a_no_by_id_runs_nothing_and_leaves_the_other(q: EventQueue, two: tuple[int, int]) -> None:
    a, b = two
    turn(q, say(q, f"no p{a}"), judge="SPEAK", act=["Left it."])
    assert running_ids(q) == {"w1-aaaa", "w2-bbbb"}
    assert kinds(q, episodes.GRANT_ACTED) == []
    assert [p.seq for p in state(q).pending] == [b]


def test_a_bare_yes_answers_nothing_while_two_are_pending(q: EventQueue, two: tuple[int, int]) -> None:
    turn(q, say(q, "yes"), judge="SPEAK", act=["Which one?"])
    assert kinds(q, episodes.GRANT_ANSWERED) == []
    assert running_ids(q) == {"w1-aaaa", "w2-bbbb"}


def test_a_bare_yes_answers_the_only_one_while_it_is_the_newest_question(q: EventQueue) -> None:
    watch(q, "w1-aaaa", "the calendar")
    give(q)
    _, pid = propose(q, "w1-aaaa")
    turn(q, say(q, "yes"), judge="SPEAK", act=["Retired."])
    assert [x["proposal"] for x in kinds(q, episodes.GRANT_ANSWERED)] == [pid]
    assert "w1-aaaa" not in running_ids(q)


def test_a_bare_yes_after_omega_said_something_else_is_not_approval(q: EventQueue) -> None:
    watch(q, "w1-aaaa", "the calendar")
    give(q)
    propose(q, "w1-aaaa")
    # Something newer was said since the question: a plain yes now could be
    # an answer to that, so it is not read as one to the proposal.
    turn(q, say(q, "what's the weather"), judge="SPEAK", act=["Sunny. Want an umbrella reminder?"])
    turn(q, say(q, "yes"), judge="SPEAK", act=["ok"])
    assert kinds(q, episodes.GRANT_ANSWERED) == []
    assert "w1-aaaa" in running_ids(q)
    # By id it still answers, from anywhere.
    [p] = state(q).pending
    turn(q, say(q, f"yes {p.pid}", channel="discord"), judge="SPEAK", act=["Retired."])
    assert "w1-aaaa" not in running_ids(q)


def test_a_yes_to_a_newer_may_i_is_never_approval(q: EventQueue) -> None:
    watch(q, "w1-aaaa", "the calendar")
    give(q)
    propose(q, "w1-aaaa")
    blocked, _ = turn(q, say(q, "clean my downloads"), act=[
        call("run_code", argv=["rm", "-rf", "/tmp/x"], cwd="/tmp")])
    assert blocked.outcome == "blocked" and "May I?" in blocked.needs
    turn(q, say(q, "yes"), judge="SPEAK", act=["ok"])
    assert kinds(q, episodes.GRANT_ANSWERED) == []
    assert "w1-aaaa" in running_ids(q)


def test_a_late_answer_to_an_answered_proposal_is_closed_and_runs_nothing(q: EventQueue) -> None:
    watch(q, "w1-aaaa", "the calendar")
    give(q)
    _, pid = propose(q, "w1-aaaa")
    turn(q, say(q, f"no p{pid}"), judge="SPEAK", act=["Left it."])
    result, fp = turn(q, say(q, f"yes p{pid}"), judge="SPEAK", act=["should not be asked"])
    assert result.outcome == "spoke"
    assert result.reply == f"p{pid}: that one's closed — you already said no to it."
    assert fp.calls_for(provider.ACT) == [], "decided in code, no model asked"
    assert len(kinds(q, episodes.GRANT_ANSWERED)) == 1
    assert "w1-aaaa" in running_ids(q)


# --- withdrawn, not dropped ---------------------------------------------------


def test_revoking_the_grant_withdraws_its_proposals_in_the_log(q: EventQueue, two: tuple[int, int]) -> None:
    a, b = two
    seq = say(q, "stop tidying")
    turn(q, seq, act=[call("grant", scope="tidy", change="revoke"), "Done."])
    withdrawn = kinds(q, episodes.GRANT_WITHDRAWN)
    assert [w["proposal"] for w in withdrawn] == [a, b]
    assert {w["reason"] for w in withdrawn} == {"you took back the tidy grant"}
    assert state(q).pending == ()
    # Written once: a second pass over the head finds nothing more to write.
    assert grants.withdraw_lapsed(q.append, state(q)) == []

    result, fp = turn(q, say(q, f"yes p{a}"), judge="SPEAK", act=["nope"])
    assert result.reply == f"p{a}: that one's closed — you took back the tidy grant."
    assert fp.calls_for(provider.ACT) == []
    assert running_ids(q) == {"w1-aaaa", "w2-bbbb"}
    assert kinds(q, episodes.GRANT_ANSWERED) == []


def test_a_cancelled_or_changed_schedule_withdraws_its_proposal(q: EventQueue, two: tuple[int, int]) -> None:
    a, b = two
    q.append(episodes.schedule_cancelled(id="w1-aaaa", at=AT))
    watch(q, "w2-bbbb", "the inbox, hourly")
    reasons = {w["proposal"]: w["reason"] for w in (
        episodes.decode(e.payload) for e in q.store.episodes_since(0)
    ) if w.get("kind") == episodes.GRANT_WITHDRAWN}
    assert reasons == {}, "nothing is written until the executor looks"
    grants.withdraw_lapsed(q.append, state(q))
    reasons = {w["proposal"]: w["reason"] for w in kinds(q, episodes.GRANT_WITHDRAWN)}
    assert reasons == {a: "w1-aaaa is no longer running", b: "w2-bbbb changed since I asked"}


def test_a_lapsed_proposal_never_runs_even_before_it_is_written_down(q: EventQueue) -> None:
    watch(q, "w1-aaaa", "the calendar")
    give(q)
    _, pid = propose(q, "w1-aaaa")
    seq = say(q, "actually")
    q.append(episodes.grant_revoked(for_seq=seq, scope="tidy", at=AT))
    q.append(episodes.completed(for_seq=seq, outcome="spoke", reply="ok", at=AT))
    # No withdraw_lapsed between: the fold alone closes it.
    sched = Scheduler(q)
    sched.refresh()
    ask = say(q, f"yes p{pid}")
    q.claim(ask)
    fp = provider.FakeProvider({provider.JUDGE: ["SPEAK"], provider.ACT: ["x"]})
    result = run_turn(
        q, q.at(ask), complete=fp.complete, act=act_loop, running=sched.schedules,
        grants=Grants().advance(q.store, upto=ask).state(), at=AT,
    )
    assert "that one's closed" in result.reply
    assert "w1-aaaa" in running_ids(q)


def test_the_executor_writes_withdrawals_after_a_turn(q: EventQueue) -> None:
    watch(q, "w1-aaaa", "the calendar")
    give(q)
    _, pid = propose(q, "w1-aaaa")
    seq = say(q, "stop tidying for me")
    fp = provider.FakeProvider({
        provider.JUDGE: ["ACT"],
        provider.ACT: [call("grant", scope="tidy", change="revoke"), "Done."],
    })
    ex = Executor(q, complete=fp.complete, act=act_loop)
    ex.recover()
    ex.drain()
    [w] = kinds(q, episodes.GRANT_WITHDRAWN)
    assert w["proposal"] == pid and w["reason"] == "you took back the tidy grant"
    assert q.at(seq) is not None


def test_a_finished_research_task_withdraws_its_follow_up(q: EventQueue) -> None:
    give(q, "research")
    ask = say(q, "research e-readers for me")
    turn(q, ask, act=[call("follow_up", what="new e-readers", in_minutes=60)])
    [p] = state(q).pending
    q.append(episodes.work_finished(for_seq=ask, summary="Get the Kobo.", at=AT))
    grants.withdraw_lapsed(q.append, state(q))
    [w] = kinds(q, episodes.GRANT_WITHDRAWN)
    assert w == {**w, "proposal": p.seq, "reason": f"research task {ask} is finished"}


def test_dl085_void_on_revoke_is_the_same_withdrawal(q: EventQueue) -> None:
    """The DL-085 void (a yes after the grant is gone runs nothing) and DL-087
    #8's withdrawal are one path: the fold lapses the proposal on the revoke,
    and the answer is told that it closed, with why."""
    watch(q, "w1-aaaa", "the calendar")
    give(q)
    _, pid = propose(q, "w1-aaaa")
    q.append(episodes.grant_revoked(for_seq=say(q, "stop tidying"), scope="tidy", at=AT))
    s = state(q)
    assert s.pending == () and s.closed[pid][1] == "you took back the tidy grant"


# --- never dropped: the brief ------------------------------------------------


def test_the_brief_lists_every_pending_proposal(q: EventQueue, two: tuple[int, int]) -> None:
    a, b = two
    result, _ = turn(q, brief(q), act=["Your day is clear."])
    assert result.outcome == "spoke"
    assert result.reply.startswith("Your day is clear.\n\nWaiting on you:\n")
    assert (
        f"- p{a} — retire watch, every 15 minutes: the calendar (w1-aaaa), "
        f'because it is stale, since Sun 11 Oct. Reply "yes p{a}" or "no p{a}".'
    ) in result.reply.splitlines()
    for pid in (a, b):
        assert f'- p{pid} — ' in result.reply
        assert f'Reply "yes p{pid}" or "no p{pid}".' in result.reply
    assert "(old)" not in result.reply


def test_after_a_week_a_proposal_is_marked_old_and_still_listed(q: EventQueue, two: tuple[int, int]) -> None:
    a, b = two
    result, _ = turn(q, brief(q, at=WEEK_ON), act=["Your day is clear."])
    for pid in (a, b):
        line = next(x for x in result.reply.splitlines() if x.startswith(f"- p{pid} — "))
        assert "(old)" in line
    assert [p.seq for p in state(q).pending] == [a, b]


def test_a_withdrawal_is_told_once_in_the_next_brief(q: EventQueue, two: tuple[int, int]) -> None:
    a, b = two
    q.append(episodes.schedule_cancelled(id="w1-aaaa", at=AT))
    grants.withdraw_lapsed(q.append, state(q))
    first, _ = turn(q, brief(q), act=["Your day."])
    assert f"No longer asking p{a} (" in first.reply
    assert first.reply.rstrip().endswith("w1-aaaa is no longer running.")
    assert f"- p{b} — " in first.reply
    second, _ = turn(q, brief(q), act=["Your day."])
    assert "No longer asking" not in second.reply
    assert f"- p{b} — " in second.reply


def test_the_brief_still_lists_them_when_it_stops_on_a_may_i(q: EventQueue, two: tuple[int, int]) -> None:
    a, _ = two
    result, _ = turn(q, brief(q), act=[call("run_code", argv=["rm", "x"], cwd="/tmp")])
    assert result.outcome == "blocked" and "May I?" in result.needs
    assert f"- p{a} — " in result.needs


# --- the pending tool ---------------------------------------------------------


def _box(q: EventQueue, *, at: str = AT) -> ToolBox:
    return ToolBox(store_root=q.store.root, grants=state(q), at=at, prompted=True)


def test_pending_is_hidden_while_nothing_waits(q: EventQueue) -> None:
    assert tools.PENDING in tools.hidden(_box(q))
    give(q)
    assert tools.PENDING in tools.hidden(_box(q))


def test_pending_lists_open_proposals_read_only(q: EventQueue, two: tuple[int, int]) -> None:
    a, b = two
    box = _box(q, at=WEEK_ON)
    assert tools.PENDING not in tools.hidden(box)
    assert grants.REVERSIBILITY[tools.PENDING] == grants.READ
    decision = box.classify(tools.PENDING, {})
    assert decision.tier == tools.EXPLORATION
    out = str(box.dispatch(decision))
    lines = out.splitlines()
    assert [x.split(":")[0] for x in lines] == [f"p{a}", f"p{b}"]
    assert all("(old)" in x and "Reply \"yes p" in x for x in lines)


def test_pending_says_so_when_offered_with_nothing_left(q: EventQueue) -> None:
    box = ToolBox(store_root=q.store.root, grants=None, at=AT)
    assert str(box.dispatch(box.classify(tools.PENDING, {}))) == "nothing is waiting on their answer"


def test_a_turn_with_a_proposal_offers_pending(q: EventQueue, two: tuple[int, int]) -> None:
    _, fp = turn(q, say(q, "what's up"), act=["Not much."])
    offered = {t["function"]["name"] for t in fp.offers_for(provider.ACT)[0] if "function" in t}
    assert tools.PENDING in offered


# --- equivalence --------------------------------------------------------------


def test_with_no_proposals_a_brief_reads_as_before(q: EventQueue) -> None:
    give(q)
    result, fp = turn(q, brief(q), act=["Your day is clear."])
    assert result.reply == "Your day is clear."
    offered = {t["function"]["name"] for t in fp.offers_for(provider.ACT)[0] if "function" in t}
    assert tools.PENDING not in offered
    prompt = str(fp.calls_for(provider.ACT)[0])
    assert "added after your brief in code" not in prompt
    assert kinds(q, episodes.GRANT_WITHDRAWN) == []


def test_with_no_proposals_a_yes_is_the_models_to_answer(q: EventQueue) -> None:
    result, fp = turn(q, say(q, "yes"), judge="SPEAK", act=["Great."])
    assert result.reply == "Great."
    assert len(fp.calls_for(provider.ACT)) == 1


def test_withdrawn_is_neither_recalled_nor_projected() -> None:
    assert episodes.GRANT_WITHDRAWN in NOT_RECALLED
    assert episodes.GRANT_WITHDRAWN in projection.NOT_PROJECTED
    assert episodes.GRANT_WITHDRAWN not in episodes.GRANT_KINDS


def test_withdrawn_is_validated() -> None:
    with pytest.raises(episodes.BadPayload):
        episodes.grant_withdrawn(proposal=3, reason="")
    with pytest.raises(episodes.BadPayload):
        episodes.grant_withdrawn(proposal=3, reason="x" * (episodes.MAX_WITHDRAWN_REASON + 1))
    with pytest.raises(episodes.BadPayload):
        episodes.grant_withdrawn(proposal=-1, reason="gone")
    ok = episodes.grant_withdrawn(proposal=3, reason="gone", at=AT)
    assert episodes.decode(episodes.encode(ok)) == ok


# --- money and outward calls, unchanged --------------------------------------


def test_money_never_runs_from_a_yes_by_id(q: EventQueue) -> None:
    give(q)
    seq = say(q, "hm")
    pseq = q.append(episodes.grant_proposed(
        for_seq=seq, scope="tidy", proposes="action",
        actions=[{"tool": "pay", "why": "pay the bill", "args": {}}], at=AT))
    turn(q, say(q, f"yes p{pseq}"), judge="SPEAK", act=["ok"])
    assert all(p["tool"] != "pay" for p in kinds(q, episodes.TOOL_CALLED))
    assert kinds(q, episodes.GRANT_ACTED) == []


def test_a_yes_runs_exactly_the_proposed_call_and_an_outward_one_still_asks(
    q: EventQueue, two: tuple[int, int]
) -> None:
    a, _ = two
    result, _ = turn(q, say(q, f"yes p{a}"), act=[
        call("run_code", argv=["rm", "-rf", "/tmp/x"], cwd="/tmp")])
    assert result.outcome == "blocked" and "May I?" in result.needs
    assert [x["tool"] for x in kinds(q, episodes.TOOL_CALLED)] == ["tidy"]
    assert running_ids(q) == {"w2-bbbb"}


# --- the follow-up tell ---------------------------------------------------------


def test_a_follow_up_tell_carries_its_proposal_id(q: EventQueue) -> None:
    watch(q, "w1-aaaa", "the calendar")
    give(q)
    tell = q.append(episodes.inbound(
        tools.tell_instruction(f"{tools.FOLLOWUP_LEAD}the calendar has failed 3 times. "
                               f"{tools.FOLLOWUP_ASK}"), channel="schedule", at=AT))
    pseq = q.append(episodes.grant_proposed(
        for_seq=tell, scope="tidy", proposes="action",
        actions=[{"tool": "tidy", "why": "pause it", "args": {"id": "w1-aaaa", "change": "retire"}}],
        at=AT))
    q.claim(tell)
    fp = provider.FakeProvider({})
    result = run_turn(q, q.at(tell), complete=fp.complete, act=act_loop, at=AT)
    q.finish(tell)
    assert result.outcome == "spoke" and fp.calls == []
    assert result.reply.endswith(
        f'Should I pause it until it\'s fixed? Reply "yes p{pseq}" or "no p{pseq}".'
    )


# --- Discord reactions ----------------------------------------------------------


ASKED = 'Should I retire w1? Reply "yes p12" or "no p12".'


@pytest.mark.parametrize("emoji, word", [("✅", "yes p12"), ("❌", "no p12")])
def test_a_check_or_cross_from_the_owner_on_omegas_proposal_answers_it(emoji: str, word: str) -> None:
    assert dc.reaction_answer(
        emoji=emoji, user_id=1, owner_id=1, author_id=9, self_id=9,
        in_guild=False, content=ASKED,
    ) == word


@pytest.mark.parametrize("change", [
    {"user_id": 2},  # not the owner
    {"author_id": 1},  # not omega's message
    {"in_guild": True},
    {"emoji": "\U0001f44d"},
    {"content": "Sunny today."},
    {"content": ASKED + '\nShould I retire w2? Reply "yes p13" or "no p13".'},
])
def test_any_other_reaction_says_nothing(change: dict) -> None:
    kw = dict(emoji="✅", user_id=1, owner_id=1, author_id=9, self_id=9,
              in_guild=False, content=ASKED)
    kw.update(change)
    assert dc.reaction_answer(**kw) is None


def test_a_reaction_is_filed_as_their_answer_once() -> None:
    sent = []

    class FakeInbound:
        def say(self, message_id: Any, text: str):
            sent.append(dc.say_request(message_id, text))
            return {"op": "ack", "seq": len(sent), "duplicate": len(sent) > 1}

    dm = dc.Dm(FakeInbound(), hearing=None, log=lambda _: None)  # type: ignore[arg-type]
    asyncio.run(dm.react(555, "yes p12"))
    asyncio.run(dm.react(555, "yes p12"))
    assert [r["text"] for r in sent] == ["yes p12", "yes p12"]
    assert sent[0]["id"] == sent[1]["id"] == f"{dc.CHANNEL}:react:555:yes"


# --- promises (DL-087 #1) ride the same pending queue -------------------------


def _promise_tell(q: EventQueue, monkeypatch: pytest.MonkeyPatch, text: str) -> tuple[Any, Any, int]:
    from tests import test_promises as tp

    monkeypatch.setattr(tools, "_now", lambda: tp.NOW.astimezone())
    tp.give(q)
    script = tp.Script(tp.answer((text, tp.TUESDAY)))
    tp.say(q, f"I'll {text} Tuesday")
    ex = tp.executor(q, script)
    tp.turn(ex, q)
    [proposal] = tp.seqs(q, episodes.GRANT_PROPOSED)
    return tp, ex, proposal


def test_a_promise_tell_is_spoken_with_its_id(q: EventQueue, monkeypatch: pytest.MonkeyPatch) -> None:
    tp, ex, proposal = _promise_tell(q, monkeypatch, "send the deck")
    result = tp.turn(ex, q)
    pid = grants.pid(proposal)
    assert result.outcome == "spoke"
    assert result.reply.endswith(f"Should I remind you then? {grants.how_to_answer(pid)}")
    assert "Reply yes or no." not in result.reply
    [still] = Grants.rebuild(q.store).state().pending
    assert still.seq == proposal and still.scope == grants.PROMISES
    tp.say(q, f"yes {pid}")
    tp.turn(ex, q)
    [noted] = tp.seqs(q, episodes.PROMISE_NOTED)
    assert f"pr{noted}" in tp.running(q)


def test_closing_the_promise_withdraws_its_proposal(q: EventQueue, monkeypatch: pytest.MonkeyPatch) -> None:
    tp, ex, proposal = _promise_tell(q, monkeypatch, "send the deck")
    tp.turn(ex, q)
    [noted] = tp.seqs(q, episodes.PROMISE_NOTED)
    q.append(episodes.promise_closed(promise=noted, how="done"))
    state = Grants.rebuild(q.store).state()
    assert state.pending == ()
    assert state.closed[proposal][1] == f"you kept promise {noted}"
    tp.say(q, f"yes {grants.pid(proposal)}")
    result = tp.turn(ex, q)
    assert "closed" in result.reply and f"you kept promise {noted}" in result.reply
    assert tp.running(q) == {} and tp.kinds(q, episodes.GRANT_ACTED) == []


def test_a_closed_promise_drops_out_of_a_shared_proposal() -> None:
    def remind(promise: int) -> dict[str, Any]:
        return {"tool": "remind", "why": "w", "args": {"what": "x", "due": "d",
                                                      "id": f"pr{promise}", "promise": promise}}

    g = Grants()
    g.apply(1, episodes.grant_given(for_seq=1, scope=grants.PROMISES))
    g.apply(2, episodes.grant_proposed(
        for_seq=2, scope=grants.PROMISES, proposes="action", actions=[remind(7), remind(8)],
    ))
    g.apply(3, episodes.promise_closed(promise=7, how="dropped"))
    [p] = g.state().pending
    assert [a["args"]["promise"] for a in p.actions] == [8]
    g.apply(4, episodes.promise_closed(promise=8, how="dropped"))
    state = g.state()
    assert state.pending == () and state.closed[2][1] == "you dropped promise 8"


def test_revoking_promises_withdraws_its_proposal(q: EventQueue, monkeypatch: pytest.MonkeyPatch) -> None:
    tp, ex, proposal = _promise_tell(q, monkeypatch, "send the deck")
    tp.turn(ex, q)
    q.append(episodes.grant_revoked(for_seq=proposal, scope=grants.PROMISES))
    state = Grants.rebuild(q.store).state()
    assert state.pending == ()
    assert state.closed[proposal][1] == "you took back the promises grant"
