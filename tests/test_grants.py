"""Standing grants, behind a gate that drops only with their yes (DL-085).

Fakes only: every model answer is scripted on `provider.FakeProvider`, the
store is a temp directory, and nothing here reaches a network or a key.
"""

from __future__ import annotations

from typing import Any

import pytest

from omega import episodes, grants, provider, tools
from omega.act import act_loop
from omega.derive import OpenWork
from omega.grants import Grants
from omega.memory import MemoryStore
from omega.provider import ToolCall
from omega.queue import EventQueue
from omega.schedule import Scheduler
from omega.tools import ToolBox, ToolRejected
from omega.turn import run_turn

AT = "2026-10-11T09:00:00+00:00"


@pytest.fixture
def q(store: MemoryStore) -> EventQueue:
    return EventQueue(store)


def call(name: str, call_id: str = "c1", **arguments: Any) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=dict(arguments))


def say(q: EventQueue, text: str, channel: str = "tray", **extra: Any) -> int:
    return q.append(episodes.inbound(text, channel=channel, at=AT, **extra))


def turn(q: EventQueue, seq: int, *, judge: Any = "ACT", act: list[Any]):
    """One turn exactly as the executor runs it: the views folded through
    this event, schedules as they stand now, the real act loop."""
    sched = Scheduler(q)
    sched.refresh()
    state = Grants().advance(q.store, upto=seq).state()
    open_work = OpenWork().advance(q.store, upto=seq).blocks()
    q.claim(seq)
    fp = provider.FakeProvider({provider.JUDGE: [judge], provider.ACT: list(act)})
    result = run_turn(
        q, q.at(seq), complete=fp.complete, act=act_loop, open_work=open_work,
        running=sched.schedules, grants=state, at=AT,
    )
    q.finish(seq)
    return result, fp


def kinds(q: EventQueue, kind: str) -> list[dict]:
    return [p.payload for p in q.recent(500) if p.payload.get("kind") == kind]


def state(q: EventQueue) -> grants.State:
    return Grants.rebuild(q.store).state()


def running_ids(q: EventQueue) -> set[str]:
    s = Scheduler(q)
    s.refresh()
    return {x.id for x in s.schedules}


def watch(q: EventQueue, sid: str = "w1-abcd", what: str = "the calendar", every: int = 900) -> None:
    q.append(episodes.schedule_created(
        id=sid, instruction=tools.watch_instruction(what), every=every, at=AT
    ))


def give(q: EventQueue, scope: str) -> int:
    """A grant given through the real path: their message, the grant tool."""
    words = {"tidy": "you can tidy my watches", "research": "research things for me"}[scope]
    seq = say(q, words)
    result, _ = turn(q, seq, act=[call("grant", scope=scope, change="give"), "Done."])
    assert result.outcome == "spoke", result
    return seq


def propose_tidy(q: EventQueue, sid: str = "w1-abcd") -> tuple[int, int]:
    """A prompted turn asks to retire ``sid``; returns (turn seq, proposal seq)."""
    seq = say(q, "please tidy up")
    result, _ = turn(q, seq, act=[call("tidy", id=sid, change="retire", why="it is stale")])
    # DL-087 #8: a proposal is filed and the turn speaks; it no longer blocks.
    assert result.outcome == "spoke", result
    [proposal] = [p for p in kinds(q, episodes.GRANT_PROPOSED) if p["for_seq"] == seq]
    pseq = next(x.seq for x in q.recent(500) if x.payload is not None and x.payload == proposal)
    assert f"p{pseq}" in result.reply
    return seq, pseq


# --- the episodes and the fold -------------------------------------------------


def test_grants_are_derived_from_the_log_and_revocation_removes_them() -> None:
    view = Grants.fold([
        (3, episodes.grant_given(for_seq=3, scope="tidy", at=AT)),
        (5, episodes.grant_given(for_seq=5, scope="research", at=AT)),
        (8, episodes.grant_revoked(for_seq=8, scope="tidy", at=AT)),
    ])
    s = view.state()
    assert set(s.grants) == {"research"}
    assert s.grants["research"].for_seq == 5 and s.grants["research"].gated
    assert view.through == 8
    # Frozen snapshot: what a turn read cannot move under it.
    view.apply(9, episodes.grant_revoked(for_seq=9, scope="research", at=AT))
    assert set(s.grants) == {"research"} and not view.state().grants


def test_grant_episodes_are_validated() -> None:
    with pytest.raises(episodes.BadPayload):
        episodes.grant_given(for_seq=3, scope="calendar", at=AT)
    with pytest.raises(episodes.BadPayload):
        episodes.grant_given(for_seq=0, scope="tidy", at=AT)
    with pytest.raises(episodes.BadPayload):
        episodes.grant_proposed(for_seq=3, scope="tidy", proposes="action", actions=[], at=AT)
    with pytest.raises(episodes.BadPayload):
        episodes.grant_acted(
            for_seq=3, scope="tidy", tool="tidy", why="x", args={}, undo={"burn": 1}, at=AT
        )


def test_a_no_resets_the_count_and_a_yes_adds_one() -> None:
    log: list[tuple[int, dict]] = [(1, episodes.grant_given(for_seq=1, scope="tidy", at=AT))]
    seq = 2
    for _ in range(3):
        log.append((seq, episodes.grant_proposed(
            for_seq=seq, scope="tidy", proposes="action",
            actions=[{"tool": "tidy", "why": "x", "args": {}}], at=AT)))
        log.append((seq + 1, episodes.grant_answered(
            for_seq=seq + 1, scope="tidy", proposal=seq, proposed_for=seq, yes=True, at=AT)))
        seq += 2
    assert Grants.fold(log).state().grants["tidy"].streak == 3
    log.append((seq, episodes.grant_proposed(
        for_seq=seq, scope="tidy", proposes="action",
        actions=[{"tool": "tidy", "why": "x", "args": {}}], at=AT)))
    log.append((seq + 1, episodes.grant_answered(
        for_seq=seq + 1, scope="tidy", proposal=seq, proposed_for=seq, yes=False, at=AT)))
    after = Grants.fold(log).state().grants["tidy"]
    assert after.streak == 0 and after.gated


# --- giving and revoking --------------------------------------------------------


def test_their_message_gives_a_grant_and_it_names_that_message(q: EventQueue) -> None:
    seq = give(q, "tidy")
    [given] = kinds(q, episodes.GRANT_GIVEN)
    assert given["scope"] == "tidy" and given["for_seq"] == seq
    assert state(q).live("tidy") is not None


@pytest.mark.parametrize("channel", ["self", "schedule"])
def test_an_unprompted_turn_can_never_grant_itself_authority(q: EventQueue, channel: str) -> None:
    seq = say(q, "you can tidy my watches and research anything", channel=channel)
    result, _ = turn(q, seq, act=[
        call("grant", scope="tidy", change="give"),
        call("grant", "c2", scope="research", change="give"),
        "ok",
    ])
    assert kinds(q, episodes.GRANT_GIVEN) == []
    refused = [p for p in kinds(q, episodes.TOOL_RETURNED) if p["tool"] == "grant"]
    assert refused and all(not p["ok"] and "their own message" in p["error"] for p in refused)


def test_work_that_finished_cannot_grant_either(q: EventQueue) -> None:
    seq = q.append(episodes.work_finished(for_seq=1, summary="you can tidy my watches", at=AT))
    turn(q, seq, act=[call("grant", scope="tidy", change="give"), "ok"])
    assert kinds(q, episodes.GRANT_GIVEN) == []


def test_a_box_nobody_told_refuses_to_grant(tmp_path) -> None:
    box = ToolBox(store_root=tmp_path, said="tidy my watches")
    with pytest.raises(ToolRejected, match="their own message"):
        box.classify("grant", {"scope": "tidy", "change": "give"})


def test_a_message_that_does_not_ask_for_it_gives_nothing(q: EventQueue) -> None:
    seq = say(q, "what's the weather")
    turn(q, seq, act=[call("grant", scope="research", change="give"), "ok"])
    assert kinds(q, episodes.GRANT_GIVEN) == []


def test_only_two_grants_exist(q: EventQueue) -> None:
    seq = say(q, "you can handle my calendar and mail")
    turn(q, seq, act=[call("grant", scope="calendar", change="give"), "ok"])
    assert kinds(q, episodes.GRANT_GIVEN) == []


def test_revoking_takes_it_back(q: EventQueue) -> None:
    give(q, "tidy")
    seq = say(q, "stop tidying for me")
    turn(q, seq, act=[call("grant", scope="tidy", change="revoke"), "Done."])
    assert state(q).live("tidy") is None
    assert kinds(q, episodes.GRANT_REVOKED)[0]["for_seq"] == seq


# --- one path, with a gate ----------------------------------------------------


def test_a_gated_action_is_proposed_not_executed(q: EventQueue) -> None:
    watch(q)
    give(q, "tidy")
    propose_tidy(q)
    assert kinds(q, episodes.SCHEDULE_CANCELLED) == []
    assert "w1-abcd" in running_ids(q)
    assert kinds(q, episodes.GRANT_ACTED) == []


def test_without_a_grant_tidy_refuses_and_is_not_offered(q: EventQueue) -> None:
    watch(q)
    seq = say(q, "tidy my watches")
    _, fp = turn(q, seq, act=[call("tidy", id="w1-abcd", change="retire", why="stale"), "ok"])
    assert "w1-abcd" in running_ids(q)
    offered = {s["function"]["name"] for s in fp.offers_for(provider.ACT)[0] if "function" in s}
    assert "tidy" not in offered and "grant" in offered


def test_a_yes_runs_exactly_what_was_proposed(q: EventQueue) -> None:
    watch(q)
    give(q, "tidy")
    _, pseq = propose_tidy(q)
    seq = say(q, "yes")
    # The judge says SPEAK; an answer still reaches the act step.
    result, _ = turn(q, seq, judge="SPEAK", act=["Retired it."])
    assert result.outcome == "spoke"
    assert "w1-abcd" not in running_ids(q)
    [answered] = kinds(q, episodes.GRANT_ANSWERED)
    assert answered["yes"] is True and answered["proposal"] == pseq
    [acted] = kinds(q, episodes.GRANT_ACTED)
    assert acted["gated"] is True and acted["proposal"] == pseq
    assert acted["undo"]["restore"]["id"] == "w1-abcd"
    assert state(q).grants["tidy"].streak == 1
    # The question it asked is settled.
    assert OpenWork.rebuild(q.store).blocks() == []


def test_a_no_runs_nothing_and_resets_the_count(q: EventQueue) -> None:
    watch(q)
    watch(q, "w1-ef01", "the news")
    give(q, "tidy")
    propose_tidy(q)
    turn(q, say(q, "yes"), act=["Done."])
    assert state(q).grants["tidy"].streak == 1
    propose_tidy(q, "w1-ef01")
    turn(q, say(q, "no thanks"), act=["Leaving it."])
    assert "w1-ef01" in running_ids(q)
    assert state(q).grants["tidy"].streak == 0
    assert len(kinds(q, episodes.GRANT_ACTED)) == 1


def test_a_yes_to_some_other_question_is_not_read_as_approval(q: EventQueue) -> None:
    watch(q)
    give(q, "tidy")
    propose_tidy(q)
    # A newer, unrelated question: an external run_code asks "May I?".
    other = say(q, "clean my downloads folder")
    blocked, _ = turn(q, other, act=[call("run_code", argv=["rm", "-rf", "/tmp/x"], cwd="/tmp")])
    assert blocked.outcome == "blocked" and "May I?" in blocked.needs
    turn(q, say(q, "yes"), judge="SPEAK", act=["ok"])
    assert kinds(q, episodes.GRANT_ANSWERED) == []
    assert "w1-abcd" in running_ids(q)


def test_a_late_answer_by_id_resolves_from_another_channel(q: EventQueue) -> None:
    watch(q)
    give(q, "tidy")
    _, pseq = propose_tidy(q)
    other = say(q, "clean my downloads folder")
    turn(q, other, act=[call("run_code", argv=["rm", "x"], cwd="/tmp")])
    turn(q, say(q, f"yes p{pseq}", channel="discord"), act=["Retired."])
    assert "w1-abcd" not in running_ids(q)
    assert kinds(q, episodes.GRANT_ANSWERED)[0]["proposal"] == pseq


def test_ten_approvals_propose_dropping_the_gate_and_do_not_drop_it(q: EventQueue) -> None:
    for i in range(grants.GATE_DROP_AFTER):
        watch(q, f"w1-{i:04x}", f"thing {i}")
    give(q, "tidy")
    for i in range(grants.GATE_DROP_AFTER):
        propose_tidy(q, f"w1-{i:04x}")
        result, _ = turn(q, say(q, "yes"), act=["Done."])
        if i < grants.GATE_DROP_AFTER - 1:
            assert result.outcome == "spoke"
    # DL-087 #8: asked as part of the reply, not by blocking the turn.
    assert result.outcome == "spoke"
    assert "10 approvals in a row" in result.reply
    drops = [p for p in kinds(q, episodes.GRANT_PROPOSED) if p["proposes"] == "drop_gate"]
    assert len(drops) == 1
    s = state(q)
    assert s.gated("tidy"), "the gate drops only on their yes, never on a count"
    assert [p.proposes for p in s.pending] == ["drop_gate"]


def _ten_in_a_row(q: EventQueue) -> int:
    for i in range(grants.GATE_DROP_AFTER + 1):
        watch(q, f"w1-{i:04x}", f"thing {i}")
    give(q, "tidy")
    for i in range(grants.GATE_DROP_AFTER):
        propose_tidy(q, f"w1-{i:04x}")
        turn(q, say(q, "yes"), act=["Done."])
    [drop] = [x for x in q.recent(500)
              if x.payload.get("kind") == episodes.GRANT_PROPOSED
              and x.payload["proposes"] == "drop_gate"]
    return drop.seq


def test_the_gate_drops_only_on_their_yes(q: EventQueue) -> None:
    drop = _ten_in_a_row(q)
    turn(q, say(q, f"yes p{drop}"), act=["I'll stop asking."])
    s = state(q)
    assert not s.gated("tidy") and s.grants["tidy"].streak == 0
    [yes] = [a for a in kinds(q, episodes.GRANT_ANSWERED) if a["proposal"] == drop]
    assert yes["yes"] is True


def test_a_no_to_dropping_the_gate_keeps_it_and_resets(q: EventQueue) -> None:
    drop = _ten_in_a_row(q)
    turn(q, say(q, "no"), act=["I'll keep asking."])
    s = state(q)
    assert s.gated("tidy") and s.grants["tidy"].streak == 0 and not s.pending


def test_nothing_can_drop_the_gate_by_itself(tmp_path) -> None:
    s = Grants.fold([(1, episodes.grant_given(for_seq=1, scope="tidy", at=AT))]).state()
    box = ToolBox(store_root=tmp_path, prompted=True, said="tidy", grants=s)
    with pytest.raises(ToolRejected):
        box.classify("grant", {"scope": "tidy", "change": "drop_gate"})


def test_one_sentence_puts_the_gate_back(q: EventQueue) -> None:
    drop = _ten_in_a_row(q)
    turn(q, say(q, f"yes p{drop}"), act=["ok"])
    seq = say(q, "ask me first again about tidy")
    turn(q, seq, act=[call("grant", scope="tidy", change="restore_gate"), "Will do."])
    s = state(q)
    assert s.gated("tidy") and s.grants["tidy"].streak == 0
    assert kinds(q, episodes.GRANT_RESTORED)[0]["for_seq"] == seq


# --- with the gate dropped: done, reported, undoable ----------------------------


def test_the_brief_lists_what_was_done_and_how_to_undo_it(q: EventQueue) -> None:
    drop = _ten_in_a_row(q)
    turn(q, say(q, f"yes p{drop}"), act=["ok"])
    last = f"w1-{grants.GATE_DROP_AFTER:04x}"
    seq = say(q, "tidy up again")
    result, _ = turn(q, seq, act=[call("tidy", id=last, change="retire", why="duplicate"), "Retired."])
    assert result.outcome == "spoke"
    assert last not in running_ids(q)
    [acted] = [a for a in kinds(q, episodes.GRANT_ACTED) if not a["gated"]]
    acted_seq = next(x.seq for x in q.recent(500) if x.payload == acted)

    brief = say(q, tools.brief_instruction(), channel="schedule", schedule_id="b1-0001")
    _, fp = turn(q, brief, act=["Your day..."])
    prompt = str(fp.calls_for(provider.ACT)[0][-1]["content"])
    assert "Done without asking since the last brief" in prompt
    assert f"undo(action={acted_seq})" in prompt and last in prompt

    # And the undo is real: the retired watch comes back, as it was.
    turn(q, say(q, "undo that"), act=[call("undo", action=acted_seq), "It's back."])
    assert last in running_ids(q)

    # The next brief lists nothing new.
    brief2 = say(q, tools.brief_instruction(), channel="schedule", schedule_id="b1-0001")
    _, fp2 = turn(q, brief2, act=["Your day..."])
    assert "Done without asking" not in str(fp2.calls_for(provider.ACT)[0][-1]["content"])


def test_a_widened_watch_is_restored_by_undo(q: EventQueue) -> None:
    drop = _ten_in_a_row(q)
    turn(q, say(q, f"yes p{drop}"), act=["ok"])
    last = f"w1-{grants.GATE_DROP_AFTER:04x}"
    turn(q, say(q, "tidy"), act=[
        call("tidy", id=last, change="every", every_minutes=180, why="your looks see the calendar"),
        "Widened."])
    s = Scheduler(q); s.refresh()
    assert {x.id: x.every for x in s.schedules}[last] == 180 * 60
    [acted] = [x for x in q.recent(500) if x.payload.get("kind") == episodes.GRANT_ACTED
               and not x.payload["gated"]]
    turn(q, say(q, "undo it"), act=[call("undo", action=acted.seq), "Back."])
    s = Scheduler(q); s.refresh()
    assert {x.id: x.every for x in s.schedules}[last] == 900


# --- tiers: money and the irreversible ----------------------------------------------


def _everything_granted(tmp_path) -> ToolBox:
    s = Grants.fold([
        (1, episodes.grant_given(for_seq=1, scope="tidy", at=AT)),
        (2, episodes.grant_given(for_seq=2, scope="research", at=AT)),
    ]).state()
    return ToolBox(store_root=tmp_path, prompted=True, said="research and tidy", grants=s, for_seq=3)


@pytest.mark.parametrize(
    "name",
    # The last two only the MCP adapter's broader check catches (DL-088).
    ["pay", "send_money", "stripe_create_payment", "upi-transfer", "buy", "place_order", "book_ride"],
)
def test_money_is_refused_whatever_is_granted(tmp_path, name: str) -> None:
    box = _everything_granted(tmp_path)
    with pytest.raises(ToolRejected, match="DL-088"):
        box.classify(name, {"amount": 5})
    assert grants.tier_of(name, tools.LOCAL) == grants.MONEY
    assert not grants.is_money("finance")


def test_money_never_runs_from_a_yes_either(q: EventQueue) -> None:
    give(q, "tidy")
    # A proposal naming a money tool, as if written by something else: a yes
    # to it still runs nothing.
    seq = say(q, "hm")
    q.append(episodes.grant_proposed(
        for_seq=seq, scope="tidy", proposes="action",
        actions=[{"tool": "pay", "why": "pay the bill", "args": {}}], at=AT))
    q.append(episodes.blocked(for_seq=seq, needs="Should I pay the bill?", at=AT))
    turn(q, say(q, "yes"), act=["ok"])
    assert all(p["tool"] != "pay" for p in kinds(q, episodes.TOOL_CALLED))
    assert kinds(q, episodes.GRANT_ACTED) == []


def test_irreversible_actions_are_always_confirmed_even_with_the_gate_dropped(q: EventQueue) -> None:
    drop = _ten_in_a_row(q)
    turn(q, say(q, f"yes p{drop}"), act=["ok"])
    seq = say(q, "tidy my downloads")
    result, _ = turn(q, seq, act=[call("run_code", argv=["rm", "-rf", "/tmp/x"], cwd="/tmp")])
    assert result.outcome == "blocked" and "May I?" in result.needs
    assert not [p for p in kinds(q, episodes.TOOL_CALLED) if p["tool"] == "run_code"]
    assert grants.tier_of("run_code", tools.EXTERNAL) == grants.IRREVERSIBLE
    assert grants.tier_of("write_file", tools.EXTERNAL) == grants.IRREVERSIBLE


def test_every_tool_has_a_reversibility_tier() -> None:
    assert tools.TOOL_NAMES | {tools.WEB_SEARCH} <= set(grants.REVERSIBILITY)
    # Nothing a grant covers is irreversible or touches money.
    for tool in grants.GATED_TOOLS:
        assert grants.REVERSIBILITY[tool] == grants.OWN and not grants.is_money(tool)
    # Calendar and mail are reads only; no tool writes to them.
    assert grants.REVERSIBILITY["calendar"] == grants.REVERSIBILITY["mail"] == grants.READ


# --- research, carried to the end -------------------------------------------------


def test_research_is_carried_to_a_report(q: EventQueue) -> None:
    give(q, "research")
    ask = say(q, "research the best e-reader for me, the new ones come out tomorrow")
    result, _ = turn(q, ask, act=[call("follow_up", what="new e-readers", in_minutes=60)])
    # DL-087 #8: filed and asked; the turn speaks rather than blocks.
    assert result.outcome == "spoke" and "Should I" in result.reply
    turn(q, say(q, "yes"), act=["I'll look again in an hour."])
    [sched] = [s for s in kinds(q, episodes.SCHEDULE_CREATED) if s["id"].startswith("f")]
    assert tools.research_task({"channel": "schedule", "text": sched["instruction"]}) == ask
    assert state(q).tasks == {ask: "new e-readers"}

    # The follow-up fires; the fire reports, and the report is its own event.
    fire = say(q, sched["instruction"], channel="schedule", schedule_id=sched["id"])
    result, _ = turn(q, fire, judge="SILENT", act=[call("report", summary="Get the Kobo."), "(nothing new)"])
    assert result.outcome == "silent"  # the report's own turn tells them
    [done] = kinds(q, episodes.WORK_FINISHED)
    assert done["for_seq"] == ask and done["summary"] == "Get the Kobo."
    assert state(q).tasks == {}
    finished = next(
        x.seq for x in q.recent(500) if x.payload.get("kind") == episodes.WORK_FINISHED
    )
    told, _ = turn(q, finished, judge="SILENT", act=["Research done: get the Kobo."])
    assert told.outcome == "spoke" and "Kobo" in told.reply


def test_report_is_refused_in_a_turn_they_started(tmp_path) -> None:
    box = _everything_granted(tmp_path)
    with pytest.raises(ToolRejected):
        box.classify("report", {"summary": "x"})


def test_revoking_research_stops_its_follow_ups_and_voids_proposals(q: EventQueue) -> None:
    give(q, "research")
    ask = say(q, "research flights to Goa")
    turn(q, ask, act=[call("follow_up", what="fares", in_minutes=120)])
    turn(q, say(q, "yes"), act=["ok"])
    turn(q, say(q, "research hotels too"), act=[call("follow_up", what="hotels", in_minutes=60)])
    assert len(state(q).pending) == 1
    turn(q, say(q, "stop the research"), act=[call("grant", scope="research", change="revoke"), "Stopped."])
    assert not [i for i in running_ids(q) if i.startswith("f")]
    assert state(q).pending == ()
    turn(q, say(q, "yes"), judge="SPEAK", act=["ok"])
    assert len([s for s in kinds(q, episodes.SCHEDULE_CREATED) if s["id"].startswith("f")]) == 1


# --- no grant, no change ----------------------------------------------------------


def test_with_no_grants_the_prompt_has_no_grants_section(q: EventQueue) -> None:
    seq = say(q, "hello")
    _, fp = turn(q, seq, act=["hi"])
    assert "Standing grants" not in str(fp.calls_for(provider.ACT)[0][-1]["content"])
