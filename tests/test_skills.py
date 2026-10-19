"""DL-086 — skills omega saves for itself, and why they never widen authority.

A skill is a starting plan in the act prompt and nothing else: every call it
suggests is classified, gated and refused exactly as it would be without it.
**No socket, no key**: every provider is scripted.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from omega import episodes, lessons, tools
from omega.memory import MemoryStore
from omega.queue import EventQueue

from tests.test_lessons import (  # noqa: F401 - the q fixture is shared
    AT, SKILL, act_turn, answer, call, failed_turn, kinds, offered, q, review,
    save_skill, say, seqs, state,
)


def test_a_cited_skill_is_saved_and_reported(q: EventQueue) -> None:
    skill = save_skill(q)
    [saved] = kinds(q, episodes.SKILL_SAVED)
    assert saved["name"] == "weekly digest"
    assert saved["steps"] == [
        {"tool": "calendar", "args_that_mattered": "the next 7 days"},
        {"tool": "inbox", "args_that_mattered": '{"days": 7}'},
    ]
    [s] = state(q).skills
    assert s.seq == skill
    # Journal-only (DL-121 A4): the report is the review's journal entry.
    reply = kinds(q, episodes.JOURNAL_WRITTEN)[-1]["changed"]
    assert reply.startswith("Self-review: learned 0, retracted 0, saved 1 skill.")
    assert "weekly digest" in reply


@pytest.mark.parametrize("tool", ["no_such_tool", "pay", "stripe_create_payment", "github__create_issue"])
def test_a_skill_must_name_tools_omega_has_and_never_money(q: EventQueue, tool: str) -> None:
    cite = say(q, "thanks, perfect")
    review(q, answer(skills=[{**SKILL, "steps": [{"tool": tool, "args_that_mattered": ""}], "cites": [cite]}]))
    assert kinds(q, episodes.SKILL_SAVED) == []


def test_an_uncited_skill_is_dropped(q: EventQueue) -> None:
    say(q, "thanks, perfect")
    review(q, answer(skills=[{**SKILL, "cites": [424242]}]))
    assert kinds(q, episodes.SKILL_SAVED) == []


def test_the_matching_skill_is_a_starting_plan_in_the_act_prompt(q: EventQueue) -> None:
    save_skill(q)
    _, fp = act_turn(q, "can I get my weekly digest please", ["Here it is."])
    act = str(fp.calls_for("act")[0])
    assert "a starting plan, not a script; deviate if it fits better" in act
    assert "1. calendar - the next 7 days" in act and "2. inbox" in act
    assert "three bullets, no greetings" in act
    _, fp = act_turn(q, "what is the weather", ["Sunny."])
    assert "starting plan" not in str(fp.calls_for("act")[0])


def test_two_corrections_retire_a_skill(q: EventQueue) -> None:
    skill = save_skill(q)
    c1 = say(q, "no, that's wrong, I wanted mail too")
    r1, _ = review(q, answer(corrected=[{"skill": skill, "cites": [c1]}]))
    assert state(q).skill(skill).corrections == 1
    assert r1.reply is None
    assert "corrected" in kinds(q, episodes.JOURNAL_WRITTEN)[-1]["changed"]
    c2 = say(q, "no, I meant the whole month")
    r2, _ = review(q, answer(corrected=[{"skill": skill, "cites": [c2]}]))
    assert state(q).skill(skill) is None
    [retired] = kinds(q, episodes.SKILL_RETIRED)
    assert retired["skill"] == skill
    assert r2.reply is None
    assert "retired skill" in kinds(q, episodes.JOURNAL_WRITTEN)[-1]["changed"]


def test_saving_the_same_name_again_replaces_the_older_way(q: EventQueue) -> None:
    save_skill(q)
    newer = save_skill(q, {**SKILL, "prefs": "one paragraph"})
    [s] = state(q).skills
    assert s.seq == newer and s.prefs == "one paragraph"


# --- a skill never widens authority ------------------------------------------------


def _forged(q: EventQueue, tool: str, when: str = "do the thing") -> None:
    """A skill in the log naming ``tool``, as if a review had saved it. The
    review never would for a money tool; the gate must hold regardless."""
    q.append(episodes.skill_saved(
        name=f"use {tool}", when=when,
        steps=[{"tool": tool, "args_that_mattered": "whatever"}],
        prefs="", cites=[1], at=AT,
    ))


def _outcome(q: EventQueue, calls: list[Any], with_skill: bool, tool: str):
    if with_skill:
        _forged(q, tool)
    result, _ = act_turn(q, "please do the thing", [*calls, "Done."])
    returned = [p for p in kinds(q, episodes.TOOL_RETURNED) if p["for_seq"] == result.seq]
    return result.outcome, result.needs, [(p["tool"], p["ok"], p.get("error")) for p in returned]


@pytest.mark.parametrize("tool, args", [
    # EXTERNAL: a write outside the store blocks the turn to ask.
    (tools.WRITE_FILE, {"path": "/etc/omega-test", "content": "x"}),
    # Gated: tidy needs the tidy grant.
    (tools.TIDY, {"id": "w1", "change": "pause", "why": "skill said so"}),
    # Money: refused under any grant, ever.
    ("stripe_create_payment", {"amount": 100}),
])
def test_a_skill_never_unlocks_a_gated_external_or_money_tool(
    store_dir, tmp_path, tool: str, args: dict
) -> None:
    seen = []
    for with_skill in (False, True):
        d = tmp_path / f"s{int(with_skill)}"
        d.mkdir()
        with MemoryStore.open(d) as store:
            q = EventQueue(store)
            seen.append(_outcome(q, [call(tool, **args)], with_skill, tool))
    assert seen[0] == seen[1], "the same call ends the same way with the skill as without"
    outcome, needs, returned = seen[1]
    assert outcome == "blocked" or all(ok is False for _, ok, _ in returned)
    assert not any(ok for _, ok, _ in returned)


def test_skill_tools_are_never_offered_because_of_a_skill(q: EventQueue) -> None:
    _, plain = act_turn(q, "please do the thing", ["Done."])
    _forged(q, "stripe_create_payment")
    _, skilled = act_turn(q, "please do the thing", ["Done."])
    extra = set(offered(skilled)) - set(offered(plain))
    assert extra == {tools.FORGET}, "only forget, because a skill now stands"
