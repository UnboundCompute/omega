"""The `act` sub-loop — M1_SPEC.md §1.5, §2.3, §2.4; DL-014, DL-028.

Everything here runs on `provider.FakeProvider`: no network, no key, no spend.
The three properties worth stating up front, because they are what the file is
built around:

* an **external** call ends the turn as a question and **dispatches nothing**;
* §1.5's *is this moving?* has two answers and both of them **stop** — a
  repeated call and an exhausted pass cap — and neither is an error;
* a tool failure is **recorded and visible to the next pass, and never
  retried** (§2.4).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omega import episodes, provider
from omega.act import DONE_ASKING, EXHAUSTED, STALLED, act_loop
from omega.memory import MemoryStore
from omega.provider import ToolCall
from omega.queue import EventQueue
from omega.tools import ToolBox
from omega.turn import TurnContext

AT = "2026-09-23T12:00:00+00:00"


@pytest.fixture
def q(store: MemoryStore) -> EventQueue:
    return EventQueue(store)


@pytest.fixture
def box(tmp_path: Path) -> ToolBox:
    room = tmp_path / "store"
    room.mkdir()
    return ToolBox(store_root=room)


def ctx_for(q: EventQueue, fp: provider.FakeProvider) -> TurnContext:
    seq = q.append(episodes.inbound("do the thing", channel="tray", at=AT))
    q.claim(seq)
    return TurnContext(
        seq=seq,
        event=q.at(seq).payload,
        recalled=[],
        queue=q,
        complete=fp.complete,
    )


def acting(*answers: object) -> provider.FakeProvider:
    """A provider scripted for the `act` role only — the sub-loop is the only
    caller under test here."""
    return provider.FakeProvider({provider.ACT: list(answers)})


def call(name: str, call_id: str = "c1", **arguments: object) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=dict(arguments))


def tool_episodes(q: EventQueue, seq: int) -> list[dict]:
    return [
        p.payload
        for p in q.recent(50)
        if p.payload.get("kind", "").startswith("tool.")
        and p.payload.get("for_seq") == seq
    ]


# --- done: the model stops asking -------------------------------------------


def test_a_pass_with_no_tool_calls_is_done(q: EventQueue, box: ToolBox) -> None:
    """§1.5's 'are we done, verified?' — the model ceasing to ask is the signal,
    and it is the only one."""
    fp = acting("nothing to do here")
    result = act_loop(ctx_for(q, fp), box=box)
    assert result.stop_reason == DONE_ASKING
    assert result.tools == ()
    assert result.blocked_on is None
    assert result.error is None


def test_a_read_runs_and_the_loop_ends_when_the_model_stops(
    q: EventQueue, box: ToolBox, tmp_path: Path
) -> None:
    note = tmp_path / "note.md"
    note.write_text("the milk", encoding="utf-8")
    fp = acting(call("read_file", path=str(note)), "")
    c = ctx_for(q, fp)

    result = act_loop(c, box=box)

    assert result.tools == ("read_file",)
    assert result.stop_reason == DONE_ASKING
    # The result reached the next pass as a tool message, which is the whole
    # point of a sub-loop rather than a sub-agent: one context, one transcript.
    second = fp.calls_for(provider.ACT)[1]
    assert any(m.get("role") == "tool" and "the milk" in m["content"] for m in second)


def test_tools_are_offered_to_act_and_to_nobody_else(q: EventQueue, box: ToolBox) -> None:
    fp = acting("")
    act_loop(ctx_for(q, fp), box=box)
    offered = fp.offers_for(provider.ACT)[0]
    assert offered is not None
    assert {s["function"]["name"] for s in offered} == {
        "read_file",
        "write_file",
        "run_code",
        "fetch",
        "recall",
    }


# --- the log ----------------------------------------------------------------


def test_called_then_returned_land_in_order_with_the_right_for_seq(
    q: EventQueue, box: ToolBox, tmp_path: Path
) -> None:
    note = tmp_path / "note.md"
    note.write_text("contents", encoding="utf-8")
    fp = acting(call("read_file", path=str(note)), "")
    c = ctx_for(q, fp)

    act_loop(c, box=box)

    logged = tool_episodes(q, c.seq)
    assert [p["kind"] for p in logged] == [episodes.TOOL_CALLED, episodes.TOOL_RETURNED]
    assert logged[0]["tool"] == "read_file"
    assert logged[1]["ok"] is True
    assert logged[1]["result"] == "contents"
    assert all(p["for_seq"] == c.seq for p in logged)


def test_the_log_records_what_came_back_not_that_it_succeeded(
    q: EventQueue, box: ToolBox, tmp_path: Path
) -> None:
    """Grade the world, not the words: a tool reporting ok is not the job being
    done, so what lands is the actual text."""
    fp = acting(call("run_code", argv=["echo", "moved 3 files"], cwd=str(tmp_path)), "")
    c = ctx_for(q, fp)
    act_loop(c, box=box)
    returned = tool_episodes(q, c.seq)[1]
    assert "moved 3 files" in returned["result"]
    assert "exit status 0" in returned["result"]


# --- external ends the turn and dispatches nothing --------------------------


def test_an_external_call_blocks_the_turn(q: EventQueue, box: ToolBox) -> None:
    fp = acting(call("fetch", url="https://example.com/x"))
    c = ctx_for(q, fp)

    result = act_loop(c, box=box)

    assert result.blocked_on is not None
    assert "https://example.com/x" in result.blocked_on
    assert result.error is None  # asking is not failing (§2.1)
    assert tool_episodes(q, c.seq) == []


def test_an_external_call_in_a_pass_stops_the_whole_pass(
    q: EventQueue, box: ToolBox, tmp_path: Path
) -> None:
    """Not 'block that one and run the rest'. The calls in one pass are the
    model's plan, and running half a plan produces side effects for a plan
    nobody approved.
    """
    victim = tmp_path / "outside.txt"
    fp = acting(
        [
            call("read_file", call_id="a", path=str(tmp_path)),
            call("write_file", call_id="b", path=str(victim), content="x"),
        ]
    )
    c = ctx_for(q, fp)

    result = act_loop(c, box=box)

    assert result.blocked_on is not None
    assert not victim.exists()
    assert tool_episodes(q, c.seq) == []
    assert result.tools == ()


def test_the_question_names_every_external_call_not_just_the_first(
    q: EventQueue, box: ToolBox, tmp_path: Path
) -> None:
    fp = acting(
        [
            call("fetch", call_id="a", url="https://example.com/one"),
            call("run_code", call_id="b", argv=["rm", "-rf", "x"], cwd=str(tmp_path)),
        ]
    )
    result = act_loop(ctx_for(q, fp), box=box)
    assert "https://example.com/one" in (result.blocked_on or "")
    assert "rm" in (result.blocked_on or "")


def test_a_write_inside_the_store_runs_without_asking(
    q: EventQueue, box: ToolBox
) -> None:
    target = box.store_root / "notes" / "today.md"
    fp = acting(call("write_file", path=str(target), content="ok"), "")
    result = act_loop(ctx_for(q, fp), box=box)
    assert result.blocked_on is None
    assert target.read_text(encoding="utf-8") == "ok"
    assert result.tools == ("write_file",)


# --- is this moving? --------------------------------------------------------


def test_repeating_a_call_verbatim_stops_the_loop(
    q: EventQueue, box: ToolBox, tmp_path: Path
) -> None:
    note = tmp_path / "note.md"
    note.write_text("same", encoding="utf-8")
    fp = acting(
        call("read_file", call_id="a", path=str(note)),
        call("read_file", call_id="b", path=str(note)),
        "should never be reached",
    )
    c = ctx_for(q, fp)

    result = act_loop(c, box=box)

    assert result.stop_reason.startswith(STALLED)
    assert result.error is None  # stopping is not failing
    assert len(fp.calls_for(provider.ACT)) == 2
    # The first call did happen and stays in the log; only the repeat stopped.
    assert len(tool_episodes(q, c.seq)) == 2


def test_key_order_cannot_disguise_a_repeat(
    q: EventQueue, box: ToolBox, tmp_path: Path
) -> None:
    fp = acting(
        ToolCall(id="a", name="run_code", arguments={"argv": ["ls"], "cwd": str(tmp_path)}),
        ToolCall(id="b", name="run_code", arguments={"cwd": str(tmp_path), "argv": ["ls"]}),
    )
    result = act_loop(ctx_for(q, fp), box=box)
    assert result.stop_reason.startswith(STALLED)


def test_a_different_argument_is_progress(
    q: EventQueue, box: ToolBox, tmp_path: Path
) -> None:
    """The control for stall detection: if this failed, the loop would be
    refusing to do any work twice rather than refusing to repeat itself."""
    one = tmp_path / "one.md"
    two = tmp_path / "two.md"
    one.write_text("1", encoding="utf-8")
    two.write_text("2", encoding="utf-8")
    fp = acting(
        call("read_file", call_id="a", path=str(one)),
        call("read_file", call_id="b", path=str(two)),
        "",
    )
    result = act_loop(ctx_for(q, fp), box=box)
    assert result.stop_reason == DONE_ASKING
    assert result.tools == ("read_file", "read_file")


def test_the_pass_cap_stops_the_loop_and_is_not_an_error(
    q: EventQueue, box: ToolBox, tmp_path: Path
) -> None:
    fp = acting(*[call("read_file", call_id=f"c{i}", path=f"{tmp_path}/f{i}") for i in range(9)])
    c = ctx_for(q, fp)

    result = act_loop(c, box=box, max_passes=3)

    assert result.stop_reason.startswith(EXHAUSTED)
    assert result.error is None
    assert result.blocked_on is None
    assert len(fp.calls_for(provider.ACT)) == 3


def test_the_pass_cap_defaults_to_the_spec_number(q: EventQueue, box: ToolBox) -> None:
    from omega.turn import MAX_ACT_PASSES

    fp = acting(*[call("read_file", call_id=f"c{i}", path=f"/nope/f{i}") for i in range(50)])
    act_loop(ctx_for(q, fp), box=box)
    assert len(fp.calls_for(provider.ACT)) == MAX_ACT_PASSES


# --- failures are surfaced and never retried --------------------------------


def test_a_failing_tool_is_recorded_and_not_retried(
    q: EventQueue, box: ToolBox, tmp_path: Path
) -> None:
    """§2.4. The loop does not retry; the *next pass* gets to see the error and
    decide. A silent retry is how one action becomes two."""
    missing = tmp_path / "not-here.md"
    fp = acting(call("read_file", path=str(missing)), "")
    c = ctx_for(q, fp)

    result = act_loop(c, box=box)

    logged = tool_episodes(q, c.seq)
    assert [p["kind"] for p in logged] == [episodes.TOOL_CALLED, episodes.TOOL_RETURNED]
    assert logged[1]["ok"] is False
    assert logged[1]["error"]
    assert logged[1]["result"] is None
    # One attempt, not two.
    assert sum(1 for p in logged if p["kind"] == episodes.TOOL_CALLED) == 1
    # A failed tool is not work done.
    assert result.tools == ()
    # And the next pass saw it.
    second = fp.calls_for(provider.ACT)[1]
    assert any(m.get("role") == "tool" and "does not exist" in m["content"] for m in second)


def test_an_unknown_tool_is_a_failed_call_not_a_failed_turn(
    q: EventQueue, box: ToolBox
) -> None:
    fp = acting(call("delete_everything"), "")
    c = ctx_for(q, fp)

    result = act_loop(c, box=box)

    assert result.error is None
    assert result.stop_reason == DONE_ASKING
    returned = tool_episodes(q, c.seq)[1]
    assert returned["ok"] is False
    assert "refused" in returned["error"]


def test_unreadable_arguments_come_back_to_the_model_as_a_failure(
    q: EventQueue, box: ToolBox, tmp_path: Path
) -> None:
    fp = acting(call("run_code", argv="ls -l", cwd=str(tmp_path)), "")
    c = ctx_for(q, fp)

    act_loop(c, box=box)

    second = fp.calls_for(provider.ACT)[1]
    assert any(m.get("role") == "tool" and "shell" in m["content"] for m in second)


def test_a_tool_that_raises_something_unexpected_is_still_a_tool_failure(
    q: EventQueue, box: ToolBox, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bug inside a tool must not take the turn down with it — it is recorded
    as what it is and the next pass can react."""
    from omega import tools as tools_mod

    def boom(*args, **kwargs):
        raise ZeroDivisionError("a bug, not a refusal")

    monkeypatch.setattr(tools_mod, "read_file", boom)
    fp = acting(call("read_file", path=str(tmp_path / "x")), "")
    c = ctx_for(q, fp)

    result = act_loop(c, box=box)

    assert result.error is None
    returned = tool_episodes(q, c.seq)[1]
    assert returned["ok"] is False
    assert "ZeroDivisionError" in returned["error"]


# --- the transcript ---------------------------------------------------------


def test_the_assistant_turn_carrying_the_calls_is_kept(
    q: EventQueue, box: ToolBox, tmp_path: Path
) -> None:
    """A tool result with no assistant turn to answer is a malformed
    conversation; the loop appends both, in order."""
    note = tmp_path / "n.md"
    note.write_text("x", encoding="utf-8")
    fp = acting(call("read_file", call_id="abc", path=str(note)), "")

    act_loop(ctx_for(q, fp), box=box)

    second = fp.calls_for(provider.ACT)[1]
    assistant = [m for m in second if m.get("role") == "assistant"]
    assert assistant and assistant[0]["tool_calls"][0]["id"] == "abc"
    result_msg = [m for m in second if m.get("role") == "tool"][0]
    assert result_msg["tool_call_id"] == "abc"


# --- wiring -----------------------------------------------------------------


def test_the_assembled_runtime_acts_for_real_by_default() -> None:
    """The step-7 wiring, asserted where it can be read: an assembled omega
    runs the sub-loop. `Executor` keeps the empty slot as *its* default, so a
    turn engine built directly in a test still does nothing it was not handed.
    """
    import inspect

    from omega.executor import Executor
    from omega.runtime import Runtime
    from omega.turn import no_act_loop_yet

    assert inspect.signature(Runtime.__init__).parameters["act"].default is act_loop
    assert (
        inspect.signature(Executor.__init__).parameters["act"].default
        is no_act_loop_yet
    )


def test_reading_back_after_a_write_is_verification_not_a_stall(
    q: EventQueue, box: ToolBox, tmp_path: Path
) -> None:
    """§1.5's two questions must not cancel each other out.

    *Are we done, verified?* is answered by reading the world back — "a tool
    reporting success is not the answer; the resulting state is". *Is this
    moving?* is answered by refusing to repeat a call. Read, write, read-back
    is the first question's own shape and the second question's definition of
    a repeat, so without a rule saying which wins, the guard would make the
    loop structurally unable to verify anything it did.

    The rule: a successful state change makes every earlier observation stale.
    """
    note = tmp_path / "store" / "note.md"
    note.write_text("before", encoding="utf-8")
    fp = acting(
        call("read_file", call_id="a", path=str(note)),
        call("write_file", call_id="b", path=str(note), content="after"),
        call("read_file", call_id="c", path=str(note)),
        "",
    )

    result = act_loop(ctx_for(q, fp), box=box)

    assert result.stop_reason == DONE_ASKING, result.stop_reason
    assert result.tools == ("read_file", "write_file", "read_file")
    assert note.read_text(encoding="utf-8") == "after"


def test_only_a_write_clears_the_set_not_any_other_call(
    q: EventQueue, box: ToolBox, tmp_path: Path
) -> None:
    """The control for the rule above. An intervening *read* changes nothing,
    so the repeat after it is still the same call against the same world and
    still a stall — otherwise "clears on progress" would quietly become
    "clears on any activity", which catches nothing."""
    one = tmp_path / "store" / "one.md"
    two = tmp_path / "store" / "two.md"
    one.write_text("1", encoding="utf-8")
    two.write_text("2", encoding="utf-8")
    fp = acting(
        call("read_file", call_id="a", path=str(one)),
        call("read_file", call_id="b", path=str(two)),
        call("read_file", call_id="c", path=str(one)),
        "",
    )

    result = act_loop(ctx_for(q, fp), box=box)

    assert result.stop_reason.startswith(STALLED), result.stop_reason


# --- recall reaches the act loop without a store (DL-060) -------------------


def _known() -> list:
    from omega.derive import Claim

    return [
        Claim(
            seq=61,
            text="The user's dog is named Shiro and is a Shih Tzu.",
            trigger={"any": ["dog", "Shiro"]},
            situation="noticed while working",
            source_seq=1,
            explicit=False,
        )
    ]


def test_the_box_the_act_loop_builds_carries_what_omega_knows(q: EventQueue) -> None:
    """The DL-058 lesson applied to DL-060: build the subject the way the
    process builds it.

    Every other recall test hands the box its claims directly, and all of them
    would stay green if `act_loop` built a box with none — which is exactly the
    live failure, a tool that works in a test and answers "I have nothing" in
    the tray. So this one passes **no box at all** and lets the loop construct
    it from the turn, which is the only path production ever takes.
    """
    fp = acting(call("recall", about="dog"), "")
    seq = q.append(episodes.inbound("what do u remember about me?", channel="tray", at=AT))
    q.claim(seq)
    ctx = TurnContext(
        seq=seq,
        event=q.at(seq).payload,
        recalled=[],
        queue=q,
        complete=fp.complete,
        # Not in `learned`: no trigger fires on this question. That is the bug
        # DL-060 exists for, and putting the claim in `learned` here would test
        # a turn that never happens.
        learned=(),
        known=_known(),
    )

    result = act_loop(ctx)

    assert result.tools == ("recall",)
    returned = [p for p in tool_episodes(q, seq) if p["kind"] == episodes.TOOL_RETURNED]
    assert returned and returned[0]["ok"] is True
    # The model saw the claim. Without the wiring it would have seen "you have
    # not written anything down" and said so, with every test above still green.
    second = fp.calls_for(provider.ACT)[1]
    assert any(m.get("role") == "tool" and "Shiro" in m["content"] for m in second)
