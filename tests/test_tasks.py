"""DL-095 — the task engine: a plan approved once, then run step by step.

The person's mode is *approve the plan, then run it*. Omega drafts a plan,
files it for approval through the ordinary DL-087 #8 queue (a ``tasks``
proposal), and on their yes the runtime drives each step as its own
unattended turn, one at a time, resumable after a restart because the whole
of a task's state is a fold over its ``task.*`` records.

Every step ends in an explicit terminal state. A step turn that carries the
obligation and ends in prose with no terminal state and no gated call is the
DL-040 gate-leak: dropped, restated once, blocked on the second drop.

**No socket, no key**: every provider is scripted and every clock is pinned.
The engine is off by default; with ``OMEGA_TASKS`` unset nothing it adds is
offered, dispatched, driven or written — the equivalence the flag guards.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

import pytest

from omega import discord, episodes, grants, provider, tasks, tools
from omega import turn as turn_mod
from omega.act import act_loop
from omega.executor import Executor
from omega.memory import MemoryStore
from omega.provider import ToolCall
from omega.queue import EventQueue
from omega.tasks import Tasks

NOW = datetime(2026, 10, 11, 12, 0, tzinfo=timezone.utc)
AT = NOW.isoformat()


@pytest.fixture(autouse=True)
def pinned(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tools, "_now", lambda: NOW.astimezone())
    monkeypatch.delenv("OMEGA_MCP_SERVERS", raising=False)
    monkeypatch.delenv("OMEGA_WEB_SEARCH", raising=False)


@pytest.fixture
def on(monkeypatch: pytest.MonkeyPatch) -> None:
    """Turn the engine on for this test."""
    monkeypatch.setenv("OMEGA_TASKS", "1")


@pytest.fixture
def q(store: MemoryStore) -> EventQueue:
    return EventQueue(store)


# --- helpers ------------------------------------------------------------------


def records(q: EventQueue) -> list[tuple[int, dict]]:
    return [(e.seq, episodes.decode(e.payload)) for e in q.store.episodes_since(0)]


def payloads(q: EventQueue) -> list[dict]:
    return [p for _, p in records(q)]


def kinds(q: EventQueue, kind: str) -> list[dict]:
    return [p for p in payloads(q) if p.get("kind") == kind]


def seq_of(q: EventQueue, kind: str) -> int:
    """The seq of the (last) record of ``kind``."""
    return [s for s, p in records(q) if p.get("kind") == kind][-1]


def say(q: EventQueue, text: str, *, channel: str = "tray", at: str = AT, **extra: Any) -> int:
    return q.append(episodes.inbound(text, channel=channel, at=at, **extra))


STEPS = (
    {"do": "draft the note", "done": "the draft record exists"},
    {"do": "send the note", "done": "the send is logged"},
)


def propose_args(goal: str = "write and send a note", budget: int = 200_000) -> dict:
    return {"do": "propose", "goal": goal, "steps": [dict(s) for s in STEPS], "budget": budget}


class Script:
    """A scripted provider. ``judge`` says what each turn should do; ``act``
    is a flat list of answers (a :class:`ToolCall`, a list of them, or a
    final string) popped in order."""

    def __init__(self, *, judge: Any = "act", act: Optional[list[Any]] = None) -> None:
        self._judge = judge
        self.acts = list(act or [])
        self.fp = provider.FakeProvider({
            provider.JUDGE: (judge if callable(judge) else (lambda r, m: judge)),
            provider.ACT: self._act,
            provider.LEARN: lambda r, m: "{}",
        })

    def _act(self, role: str, messages: list[dict]) -> Any:
        if not self.acts:
            return "Done."
        return self.acts.pop(0)

    def act_offers(self) -> list[set[str]]:
        """The tool names offered on each ACT call, in order."""
        out = []
        for offer in self.fp.offers_for(provider.ACT):
            out.append({s["function"]["name"] for s in (offer or []) if "function" in s})
        return out


def executor(q: EventQueue, script: Script) -> Executor:
    ex = Executor(q, complete=script.fp.complete, act=act_loop)
    ex.recover()
    return ex


def drain(ex: Executor, q: EventQueue) -> None:
    while next(q.pending(), None) is not None:
        ex.step()


def call(tool: str, **arguments: Any) -> ToolCall:
    return ToolCall(id="c1", name=tool, arguments=dict(arguments))


def usage_row(total: int) -> list[dict]:
    return [{"role": "act", "model": "m", "calls": 1, "input": total,
             "cached": 0, "output": 0, "reasoning": 0, "unmetered": 0}]


def fold(*episodes_in: dict) -> tasks.State:
    view = Tasks()
    for i, p in enumerate(episodes_in, start=1):
        view.apply(i, p)
    return view.state()


# --- the fold -----------------------------------------------------------------


def test_a_started_plan_is_running_and_its_cursor_starts_at_zero() -> None:
    task = 1
    state = fold(
        episodes.task_proposed(for_seq=task, goal="g", steps=list(STEPS), at=AT),
        episodes.task_started(for_seq=2, task=task, at=AT),
    )
    live = state.task(task)
    assert live is not None and live.running and not live.completed
    assert live.cursor == 0 and live.drivable
    assert [t.task for t in state.running()] == [task]


def test_a_done_step_advances_the_cursor_and_the_last_completes_the_task() -> None:
    state = fold(
        episodes.task_proposed(for_seq=1, goal="g", steps=list(STEPS), at=AT),
        episodes.task_started(for_seq=2, task=1, at=AT),
        episodes.task_step(for_seq=3, task=1, n=0, state="done", evidence="r#3", at=AT),
    )
    live = state.task(1)
    assert live.cursor == 1 and not live.completed and live.drivable
    state2 = fold(
        episodes.task_proposed(for_seq=1, goal="g", steps=list(STEPS), at=AT),
        episodes.task_started(for_seq=2, task=1, at=AT),
        episodes.task_step(for_seq=3, task=1, n=0, state="done", evidence="r#3", at=AT),
        episodes.task_step(for_seq=4, task=1, n=1, state="done", evidence="r#4", at=AT),
    )
    done = state2.task(1)
    assert done.completed and not done.running and not done.drivable


def test_spend_accrues_from_the_obligated_turns_usage() -> None:
    ob = episodes.inbound("x", channel="task", at=AT, obligation={"task": 1, "n": 0})
    state = fold(
        episodes.task_proposed(for_seq=1, goal="g", steps=list(STEPS), budget=1000, at=AT),
        episodes.task_started(for_seq=2, task=1, at=AT),
        ob,  # seq 3
        episodes.completed(for_seq=3, outcome="spoke", reply="hi", usage=usage_row(400), at=AT),
    )
    assert state.task(1).spent == 400


def test_a_step_in_flight_is_outstanding_and_not_drivable_until_it_lands() -> None:
    ob = episodes.inbound("x", channel="task", at=AT, obligation={"task": 1, "n": 0})
    state = fold(
        episodes.task_proposed(for_seq=1, goal="g", steps=list(STEPS), at=AT),
        episodes.task_started(for_seq=2, task=1, at=AT),
        ob,  # seq 3: driven, no terminal yet
    )
    live = state.task(1)
    assert live.outstanding == frozenset({0})
    assert not live.drivable, "a step already driven is not driven again"
    assert state.drivable() is None


def test_a_dropped_step_is_not_re_driven_fresh() -> None:
    ob = episodes.inbound("x", channel="task", at=AT, obligation={"task": 1, "n": 0})
    state = fold(
        episodes.task_proposed(for_seq=1, goal="g", steps=list(STEPS), at=AT),
        episodes.task_started(for_seq=2, task=1, at=AT),
        ob,  # seq 3
        episodes.task_dropped(for_seq=3, task=1, n=0, at=AT),
    )
    live = state.task(1)
    assert live.drops == {0: 1} and not live.drivable


def test_obligation_inbounds_count_against_the_day() -> None:
    ob = episodes.inbound("x", channel="task", at=AT, obligation={"task": 1, "n": 0})
    state = fold(
        episodes.task_proposed(for_seq=1, goal="g", steps=list(STEPS), at=AT),
        episodes.task_started(for_seq=2, task=1, at=AT),
        ob,
    )
    day = tasks._local_day(AT)
    assert state.steps_today(day) == 1


def test_a_cancelled_task_stops_being_driven_or_run() -> None:
    state = fold(
        episodes.task_proposed(for_seq=1, goal="g", steps=list(STEPS), at=AT),
        episodes.task_started(for_seq=2, task=1, at=AT),
        episodes.task_cancelled(task=1, at=AT),
    )
    live = state.task(1)
    assert live.cancelled and not live.running and not live.drivable


def test_obligation_of_reads_only_a_well_formed_marker() -> None:
    assert tasks.obligation_of({"obligation": {"task": 3, "n": 1}}) == {"task": 3, "n": 1}
    assert tasks.obligation_of({"obligation": {"task": True, "n": 1}}) is None
    assert tasks.obligation_of({}) is None


# --- step validation (the terminal-state discipline) --------------------------


def box_for(q: EventQueue, *, task_run: Optional[dict] = None, tasks_state: Any = None) -> tools.ToolBox:
    return tools.ToolBox(
        store_root=q.store.root, append=q.append, for_seq=1, prompted=True, at=AT,
        tasks=tasks_state, task_run=task_run,
    )


def test_a_done_step_must_carry_evidence(q: EventQueue, on: None) -> None:
    box = box_for(q, task_run={"task": 1, "n": 0})
    with pytest.raises(tools.ToolRejected):
        box.classify(tools.STEP, {"state": "done"})
    d = box.classify(tools.STEP, {"state": "done", "evidence": "r#7"})
    assert d.args["evidence"] == "r#7"


def test_a_waiting_step_names_what_it_waits_on_and_until(q: EventQueue, on: None) -> None:
    box = box_for(q, task_run={"task": 1, "n": 0})
    with pytest.raises(tools.ToolRejected):
        box.classify(tools.STEP, {"state": "waiting", "until": "noon"})  # no `on`
    with pytest.raises(tools.ToolRejected):
        box.classify(tools.STEP, {"state": "waiting", "on": "whim", "until": "noon"})
    d = box.classify(tools.STEP, {"state": "waiting", "on": "tap", "until": "they pay"})
    assert d.args["on"] == "tap" and d.args["until"] == "they pay"


def test_blocked_and_failed_need_a_why(q: EventQueue, on: None) -> None:
    box = box_for(q, task_run={"task": 1, "n": 0})
    for state in ("blocked", "failed"):
        with pytest.raises(tools.ToolRejected):
            box.classify(tools.STEP, {"state": state})
        d = box.classify(tools.STEP, {"state": state, "why": "the door is shut"})
        assert d.args["why"] == "the door is shut"


def test_step_runs_only_inside_a_step_turn(q: EventQueue, on: None) -> None:
    box = box_for(q, task_run=None)
    with pytest.raises(tools.ToolRejected):
        box.classify(tools.STEP, {"state": "done", "evidence": "r#1"})


# --- propose validation -------------------------------------------------------


def test_a_plan_of_more_than_the_cap_is_rejected(q: EventQueue, on: None) -> None:
    box = box_for(q, tasks_state=tasks.State())
    many = [{"do": f"s{i}", "done": f"d{i}"} for i in range(episodes.MAX_TASK_STEPS + 1)]
    with pytest.raises(tools.ToolRejected):
        box.classify(tools.TASK, {"do": "propose", "goal": "g", "steps": many})


def test_a_step_missing_do_or_done_is_rejected(q: EventQueue, on: None) -> None:
    box = box_for(q, tasks_state=tasks.State())
    with pytest.raises(tools.ToolRejected):
        box.classify(tools.TASK, {"do": "propose", "goal": "g", "steps": [{"do": "x"}]})


def test_a_bad_budget_is_rejected_and_the_default_fills_in(q: EventQueue, on: None) -> None:
    box = box_for(q, tasks_state=tasks.State())
    with pytest.raises(tools.ToolRejected):
        box.classify(tools.TASK, {"do": "propose", "goal": "g", "steps": list(STEPS), "budget": 0})
    d = box.classify(tools.TASK, {"do": "propose", "goal": "g", "steps": list(STEPS)})
    assert d.args["budget"] == episodes.DEFAULT_TASK_BUDGET


# --- equivalence: with the engine off, nothing changes ------------------------


def test_with_the_engine_off_task_and_step_are_not_offered(q: EventQueue) -> None:
    box = box_for(q, task_run={"task": 1, "n": 0}, tasks_state=tasks.State())
    offered = {s["function"]["name"] for s in box.schemas()}
    assert tools.TASK not in offered and tools.STEP not in offered


def test_with_the_engine_off_task_and_step_do_not_classify(q: EventQueue) -> None:
    box = box_for(q, task_run={"task": 1, "n": 0}, tasks_state=tasks.State())
    with pytest.raises(tools.ToolRejected):
        box.classify(tools.TASK, propose_args())
    with pytest.raises(tools.ToolRejected):
        box.classify(tools.STEP, {"state": "done", "evidence": "r#1"})


def test_with_the_engine_off_drive_tasks_does_nothing(q: EventQueue) -> None:
    # A started task sits in the log; with the flag off the runtime ignores it.
    task = say(q, "x")
    q.append(episodes.task_proposed(for_seq=task, goal="g", steps=list(STEPS), at=AT))
    tseq = seq_of(q, episodes.TASK_PROPOSED)
    q.append(episodes.task_started(for_seq=task, task=tseq, at=AT))
    q.claim(task)
    q.finish(task)
    script = Script()
    ex = executor(q, script)
    before = q.store.head()
    assert ex.drive_tasks(now=AT) is False
    assert q.store.head() == before, "nothing is written with the engine off"


def test_with_the_engine_on_the_schemas_gain_task_and_step(q: EventQueue, on: None) -> None:
    propose = box_for(q, tasks_state=tasks.State())  # prompted, no task_run
    offered = {s["function"]["name"] for s in propose.schemas()}
    assert tools.TASK in offered and tools.STEP not in offered
    # a step turn is unprompted: build it that way.
    step_box = tools.ToolBox(
        store_root=q.store.root, append=q.append, for_seq=1, prompted=False, at=AT,
        tasks=tasks.State(), task_run={"task": 1, "n": 0},
    )
    names = {s["function"]["name"] for s in step_box.schemas()}
    assert tools.STEP in names and tools.TASK not in names


# --- the whole path: propose, approve, drive, step ----------------------------


def test_propose_files_a_tasks_proposal_through_the_eight_queue(q: EventQueue, on: None) -> None:
    seq = say(q, "please write and send a note for me")
    script = Script(act=[call(tools.TASK, **propose_args()), "Here is the plan."])
    ex = executor(q, script)
    drain(ex, q)
    proposed = kinds(q, episodes.TASK_PROPOSED)
    assert len(proposed) == 1 and proposed[0]["goal"] == "write and send a note"
    gp = kinds(q, episodes.GRANT_PROPOSED)[-1]
    assert gp["proposes"] == "task" and gp["scope"] == grants.TASKS
    # the proposal names the task.proposed seq, and no task has started yet.
    assert gp["task"] == seq_of(q, episodes.TASK_PROPOSED)
    assert kinds(q, episodes.TASK_STARTED) == []
    # the propose turn offered `task`, never `step`.
    assert any(tools.TASK in names for names in script.act_offers())
    assert all(tools.STEP not in names for names in script.act_offers())


def test_a_reaction_on_the_proposal_reads_as_yes_pn(q: EventQueue, on: None) -> None:
    say(q, "please write and send a note")
    script = Script(act=[call(tools.TASK, **propose_args()), "Plan filed."])
    ex = executor(q, script)
    drain(ex, q)
    pid = grants.pid(seq_of(q, episodes.GRANT_PROPOSED))
    content = f'I want to run the plan. {grants.how_to_answer(pid)}'
    answer = discord.reaction_answer(
        emoji="✅", user_id=1, owner_id=1, author_id=9, self_id=9,
        in_guild=False, content=content,
    )
    assert answer == f"yes {pid}"


def test_a_yes_starts_the_task(q: EventQueue, on: None) -> None:
    say(q, "please write and send a note")
    script = Script(act=[call(tools.TASK, **propose_args()), "Plan filed."])
    ex = executor(q, script)
    drain(ex, q)
    pid = grants.pid(seq_of(q, episodes.GRANT_PROPOSED))
    say(q, f'yes {pid}')
    script.acts.append("Great, I'll run it now.")
    drain(ex, q)
    started = kinds(q, episodes.TASK_STARTED)
    assert len(started) == 1 and started[0]["task"] == seq_of(q, episodes.TASK_PROPOSED)


def test_a_no_leaves_the_plan_unstarted(q: EventQueue, on: None) -> None:
    say(q, "please write and send a note")
    script = Script(act=[call(tools.TASK, **propose_args()), "Plan filed."])
    ex = executor(q, script)
    drain(ex, q)
    pid = grants.pid(seq_of(q, episodes.GRANT_PROPOSED))
    say(q, f'no {pid}')
    script.acts.append("Understood, I won't run it.")
    drain(ex, q)
    assert kinds(q, episodes.TASK_STARTED) == []


def start_a_task(q: EventQueue, script: Script, budget: int = 200_000) -> tuple[Executor, int]:
    """Propose and approve a two-step task; return the executor and the task seq."""
    say(q, "please write and send a note")
    script.acts = [call(tools.TASK, **propose_args(budget=budget)), "Plan filed."]
    ex = executor(q, script)
    drain(ex, q)
    tseq = seq_of(q, episodes.TASK_PROPOSED)
    pid = grants.pid(seq_of(q, episodes.GRANT_PROPOSED))
    say(q, f'yes {pid}')
    script.acts.append("Running now.")
    drain(ex, q)
    return ex, tseq


def test_drive_tasks_drives_one_step_as_an_obligated_turn(q: EventQueue, on: None) -> None:
    script = Script()
    ex, tseq = start_a_task(q, script)
    assert ex.drive_tasks(now=AT) is True
    driven = [p for p in kinds(q, episodes.MESSAGE_INBOUND)
              if tasks.obligation_of(p) == {"task": tseq, "n": 0}]
    assert len(driven) == 1 and driven[0]["channel"] == "task"
    # a second call while it is in flight drives nothing.
    assert ex.drive_tasks(now=AT) is False


def test_a_step_turn_still_acts_when_the_judge_says_silent(q: EventQueue, on: None) -> None:
    # The judge says nothing to the step turn; the obligation forces it to act
    # all the same. (Propose and approve go the ordinary way.)
    judge = lambda r, m: "silent" if tools.TASK_LEAD in str(m) else "act"  # noqa: E731
    script = Script(judge=judge)
    ex, tseq = start_a_task(q, script)
    ex.drive_tasks(now=AT)
    # the silent judge would stay quiet, but the obligation forces the turn to
    # act: the model gets to call `step`, and it does.
    script.acts = [call(tools.STEP, state="done", evidence="r#draft"), "Step done."]
    drain(ex, q)
    steps = kinds(q, episodes.TASK_STEP)
    assert len(steps) == 1 and steps[0]["state"] == "done" and steps[0]["n"] == 0
    # the step turn offered `step` and never `task` on the same turn.
    step_offers = [names for names in script.act_offers() if tools.STEP in names]
    assert step_offers and all(tools.TASK not in names for names in step_offers)


def test_a_done_step_advances_the_run_to_the_next_step(q: EventQueue, on: None) -> None:
    script = Script()
    ex, tseq = start_a_task(q, script)
    ex.drive_tasks(now=AT)
    script.acts = [call(tools.STEP, state="done", evidence="r#draft"), "Drafted."]
    drain(ex, q)
    view = Tasks.rebuild(q.store).state()
    assert view.task(tseq).cursor == 1
    # the next drive reaches step 1.
    assert ex.drive_tasks(now=AT) is True
    driven = [p for p in kinds(q, episodes.MESSAGE_INBOUND)
              if tasks.obligation_of(p) == {"task": tseq, "n": 1}]
    assert len(driven) == 1


def test_the_run_resumes_from_the_log_after_a_restart(q: EventQueue, on: None) -> None:
    script = Script()
    ex, tseq = start_a_task(q, script)
    ex.drive_tasks(now=AT)
    script.acts = [call(tools.STEP, state="done", evidence="r#draft"), "Drafted."]
    drain(ex, q)
    # a fresh executor, as after a restart: state is the fold, nothing beside it.
    ex2 = executor(q, script)
    assert ex2.drive_tasks(now=AT) is True
    driven = [p for p in kinds(q, episodes.MESSAGE_INBOUND)
              if tasks.obligation_of(p) == {"task": tseq, "n": 1}]
    assert len(driven) == 1


def test_a_step_waiting_on_tap_pauses_the_run_with_no_payment_tool(q: EventQueue, on: None) -> None:
    script = Script()
    ex, tseq = start_a_task(q, script)
    ex.drive_tasks(now=AT)
    script.acts = [
        call(tools.STEP, state="waiting", on="tap", until="they tap the pay link"),
        "I've sent you a link to pay.",
    ]
    drain(ex, q)
    view = Tasks.rebuild(q.store).state()
    live = view.task(tseq)
    assert live.waiting and live.running and not live.drivable
    # driving does not resume a waiting step on its own.
    assert ex.drive_tasks(now=AT) is False
    # no payment tool was ever offered on any ACT turn.
    assert all("pay" not in n and "checkout" not in n
               for names in script.act_offers() for n in names)


# --- caps ---------------------------------------------------------------------


def seed_started(q: EventQueue, *, budget: int) -> int:
    """A started two-step task in the log; return its seq."""
    origin = say(q, "x")
    q.append(episodes.task_proposed(for_seq=origin, goal="g", steps=list(STEPS),
                                    budget=budget, at=AT))
    tseq = seq_of(q, episodes.TASK_PROPOSED)
    q.append(episodes.task_started(for_seq=origin, task=tseq, at=AT))
    q.claim(origin)
    q.finish(origin)
    return tseq


def test_a_spent_budget_blocks_the_next_step_and_tells_once(q: EventQueue, on: None) -> None:
    tseq = seed_started(q, budget=100)
    # step 0 done; an obligated turn that spent more than the budget.
    ob = q.append(episodes.inbound("x", channel="task", at=AT,
                                   obligation={"task": tseq, "n": 0}))
    q.append(episodes.task_step(for_seq=ob, task=tseq, n=0, state="done", evidence="r#1", at=AT))
    q.append(episodes.completed(for_seq=ob, outcome="spoke", reply="ok",
                                usage=usage_row(500), at=AT))
    script = Script()
    ex = executor(q, script)
    assert ex.drive_tasks(now=AT) is True
    blocked = [p for p in kinds(q, episodes.TASK_STEP) if p["state"] == "blocked"]
    assert len(blocked) == 1 and blocked[0]["n"] == 1
    tells = [p for p in kinds(q, episodes.MESSAGE_INBOUND)
             if p.get("channel") == "schedule" and "had to stop" in p.get("text", "")]
    assert len(tells) == 1


def test_the_daily_cap_blocks_the_next_step_and_tells_once(
    q: EventQueue, on: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tasks, "DAILY_STEPS", 1)
    tseq = seed_started(q, budget=200_000)
    # one step driven today (counts against the day) and done: the next is over.
    q.append(episodes.inbound("x", channel="task", at=AT, obligation={"task": tseq, "n": 0}))
    q.append(episodes.task_step(for_seq=seq_of(q, episodes.MESSAGE_INBOUND),
                                task=tseq, n=0, state="done", evidence="r#1", at=AT))
    script = Script()
    ex = executor(q, script)
    assert ex.drive_tasks(now=AT) is True
    blocked = [p for p in kinds(q, episodes.TASK_STEP) if p["state"] == "blocked"]
    assert len(blocked) == 1 and blocked[0]["n"] == 1
    tells = [p for p in kinds(q, episodes.MESSAGE_INBOUND)
             if p.get("channel") == "schedule" and "had to stop" in p.get("text", "")]
    assert len(tells) == 1


# --- the brief ----------------------------------------------------------------


def test_a_running_task_shows_in_the_brief(q: EventQueue, on: None) -> None:
    state = fold(
        episodes.task_proposed(for_seq=1, goal="water the plants", steps=list(STEPS), at=AT),
        episodes.task_started(for_seq=2, task=1, at=AT),
        episodes.task_step(for_seq=3, task=1, n=0, state="done", evidence="r#3", at=AT),
    )
    ctx = turn_mod.TurnContext(
        queue=None, event=episodes.inbound(tools.brief_instruction(), channel="schedule", at=AT),
        seq=99, recalled=(), complete=(lambda *a, **k: None), tasks=state,
    )
    tail = turn_mod._running_tasks(ctx)
    assert turn_mod.TASKS_RUNNING in tail
    assert "water the plants: on step 2 of 2" in tail


def test_a_waiting_task_shows_as_waiting_in_the_brief(q: EventQueue, on: None) -> None:
    state = fold(
        episodes.task_proposed(for_seq=1, goal="book a table", steps=list(STEPS), at=AT),
        episodes.task_started(for_seq=2, task=1, at=AT),
        episodes.task_step(for_seq=3, task=1, n=0, state="waiting", on="tap",
                           until="they pay", at=AT),
    )
    ctx = turn_mod.TurnContext(
        queue=None, event=episodes.inbound(tools.brief_instruction(), channel="schedule", at=AT),
        seq=99, recalled=(), complete=(lambda *a, **k: None), tasks=state,
    )
    assert "book a table: waiting" in turn_mod._running_tasks(ctx)


# --- the gate-leak fix (DL-040 remainder) -------------------------------------


def test_a_step_that_ends_in_prose_is_dropped_restated_then_blocked(q: EventQueue, on: None) -> None:
    # The model never calls `step`: both the drive and the one retry end in
    # prose. The executor records each drop and, on the second, blocks and
    # tells the person once.
    script = Script(act=["I had a look at it."] * 4)
    ex, tseq = start_a_task(q, script)
    ex.drive_tasks(now=AT)
    drain(ex, q)  # drives the retry inbound the drop appended, which drops again
    drops = [p for p in kinds(q, episodes.TASK_DROPPED) if p["task"] == tseq]
    assert len(drops) == 2
    retried = [p for p in kinds(q, episodes.MESSAGE_INBOUND)
               if tasks.obligation_of(p) == {"task": tseq, "n": 0}]
    assert len(retried) == 2, "the first drive plus one restated retry"
    assert any("the one retry" in p.get("text", "") for p in retried)
    blocked = [p for p in kinds(q, episodes.TASK_STEP)
               if p["state"] == "blocked" and p["n"] == 0]
    assert len(blocked) == 1
    tells = [p for p in kinds(q, episodes.MESSAGE_INBOUND)
             if p.get("channel") == "schedule" and "had to stop" in p.get("text", "")]
    assert len(tells) == 1


def test_a_retry_that_records_a_terminal_state_discharges_the_obligation(q: EventQueue, on: None) -> None:
    script = Script()
    ex, tseq = start_a_task(q, script)
    ex.drive_tasks(now=AT)
    # first attempt ends in prose -> dropped and restated.
    script.acts = ["I looked at it."]
    while next(q.pending(), None) is not None:
        ex.step()
        if kinds(q, episodes.TASK_DROPPED):
            break
    assert len(kinds(q, episodes.TASK_DROPPED)) == 1
    # the retry now calls `step`: done, no block.
    script.acts = [call(tools.STEP, state="done", evidence="r#draft"), "Done this time."]
    drain(ex, q)
    assert len(kinds(q, episodes.TASK_DROPPED)) == 1, "no second drop"
    done = [p for p in kinds(q, episodes.TASK_STEP) if p["state"] == "done"]
    assert len(done) == 1 and done[0]["n"] == 0
    blocked = [p for p in kinds(q, episodes.TASK_STEP) if p["state"] == "blocked"]
    assert blocked == []


def test_an_unattended_prose_turn_without_an_obligation_drops_nothing(q: EventQueue, on: None) -> None:
    # A plain clock-origin fire with no obligation: the gate-leak does not fire.
    script = Script()
    ex = executor(q, script)
    say(q, "good morning", channel="schedule", schedule_id="s1")
    script.acts = ["Morning!"]
    drain(ex, q)
    assert kinds(q, episodes.TASK_DROPPED) == []


def test_tasks_is_not_a_grant_they_can_give(q: EventQueue) -> None:
    # `tasks` only names the queue a plan's approval is filed under; it is never
    # a standing grant, so the `grant` tool's offer is exactly what it was.
    grant = next(s for s in tools.schemas() if s["function"]["name"] == "grant")
    assert grants.TASKS not in grant["function"]["parameters"]["properties"]["scope"]["enum"]
    box = box_for(q)
    with pytest.raises(tools.ToolRejected):
        box.classify("grant", {"change": "give", "scope": grants.TASKS})
