"""C0 — omega's own words are never read back as the person's.

Before C0 a fire rendered as ``you:`` in recall, the reflection pass read
every channel and anchored on the log's head, and the trigger counted fires
and looks as conversation. So omega's own watch rules came back as things it
had "learned" about the person, three times over in three phrasings. The
pairs here are the capability (a real message still teaches) and the
violation (a fire never does). Every message is synthetic.
"""

from __future__ import annotations

import json

import pytest

from omega import derive, episodes, executor, provider, tidy, turn
from omega.executor import Executor
from omega.memory import MemoryStore
from omega.queue import EventQueue

from tests.teaching import _a_claim, _answer, _claim_obj, _log, _Teaching

AT = "2026-09-23T12:00:00+00:00"


@pytest.fixture
def q(store: MemoryStore) -> EventQueue:
    return EventQueue(store)


def _fire(q: EventQueue, text: str = "Watch: the build page for anything new") -> int:
    return q.append(episodes.inbound(text, channel="schedule", schedule_id="w1-abcd", at=AT))


def _say(q: EventQueue, text: str) -> int:
    return q.append(episodes.inbound(text, channel="tray", at=AT))


def _run(q: EventQueue, teaching: _Teaching) -> Executor:
    ex = Executor(q, complete=teaching.complete)
    ex.recover()
    ex.drain()
    return ex


def _learn_prompts(teaching: _Teaching) -> list[str]:
    return [
        "\n".join(json.dumps(m) for m in messages)
        for role, messages in teaching.calls if role == provider.LEARN
    ]


def _kinds(q: EventQueue, kind: str) -> list[tuple[int, dict]]:
    return [(s, p) for s, p in _log(q) if p.get("kind") == kind]


# --- a fire is labelled in recall, and only there ------------------------------


def test_a_recalled_fire_says_it_is_not_a_person() -> None:
    fire = episodes.inbound("Watch: the build page", channel="schedule", at=AT)
    assert turn._render_payload(fire, recalled=True).startswith(turn.FIRED_LEAD)


def test_a_fire_shown_to_its_own_judge_is_unchanged() -> None:
    fire = episodes.inbound("Watch: the build page", channel="schedule", at=AT)
    assert turn._render_payload(fire) == "you: Watch: the build page"


# --- the trigger and the window are the person's ------------------------------


def test_fires_alone_never_trigger_a_reflection(q: EventQueue) -> None:
    for _ in range(executor.REFLECT_EVERY + 2):
        _fire(q)
    teaching = _Teaching(learn_answer=_answer(_a_claim()), verdict="SILENT")
    _run(q, teaching)
    assert _learn_prompts(teaching) == []
    assert _kinds(q, episodes.REFLECTION_DONE) == []


def test_the_pass_reads_their_messages_and_not_the_fires(q: EventQueue) -> None:
    for i in range(executor.REFLECT_EVERY):
        _say(q, f"note number {i} from them")
        _fire(q, "Watch: the zebra-quartz page")
    teaching = _Teaching(learn_answer=_answer(_a_claim("they write in the evening")), verdict="SILENT")
    _run(q, teaching)

    [prompt] = _learn_prompts(teaching)
    assert "note number 0 from them" in prompt
    assert "zebra-quartz" not in prompt, "a fire was read as their conversation"
    [(_, claim)] = _kinds(q, episodes.CLAIM_EXTRACTED)
    by_seq = dict(_log(q))
    assert by_seq[claim["for_seq"]]["channel"] == "tray", (
        "a claim from the conversation is anchored on something they said"
    )


def test_a_rule_about_omega_itself_becomes_a_lesson(q: EventQueue) -> None:
    for i in range(executor.REFLECT_EVERY):
        _say(q, f"note number {i} from them")
    teaching = _Teaching(
        learn_answer=_answer(
            _a_claim("check the build page every hour", about="you"),
            _a_claim("they like short replies", about="them"),
        ),
        verdict="SILENT",
    )
    _run(q, teaching)

    claims = [p["text"] for _, p in _kinds(q, episodes.CLAIM_EXTRACTED)]
    lessons = [p["text"] for _, p in _kinds(q, episodes.LESSON_LEARNED)]
    assert claims == ["they like short replies"]
    assert lessons == ["check the build page every hour"]


def test_a_paraphrase_of_a_known_claim_is_not_filed_again(q: EventQueue) -> None:
    for i in range(executor.REFLECT_EVERY):
        _say(q, f"note number {i} from them")
    teaching = _Teaching(
        learn_answer=_answer(
            _a_claim("they prefer short status updates"),
            _a_claim("prefers status updates kept short"),
        ),
        verdict="SILENT",
    )
    _run(q, teaching)
    assert [p["text"] for _, p in _kinds(q, episodes.CLAIM_EXTRACTED)] == [
        "they prefer short status updates"
    ]


def test_novel_keeps_a_refinement_of_what_it_supersedes() -> None:
    known = [_claim_obj(5, "they prefer short status updates")]
    same = {"text": "prefers short status updates", "supersedes": None}
    refined = {"text": "prefers short status updates on weekdays", "supersedes": 5}
    assert tidy.novel([same], known) == []
    assert tidy.novel([refined], known) == [refined]
    assert tidy.similarity("tea please", "coffee now") == 0.0


# --- what was misfiled before C0 is retracted structurally --------------------


def test_a_claim_anchored_on_a_fire_is_retracted_and_a_person_one_stays(q: EventQueue) -> None:
    fire = _fire(q)
    q.append(episodes.completed(for_seq=fire, outcome="silent", at=AT))
    bad = q.append(episodes.claim_extracted(
        for_seq=fire, text="checks the build page hourly", source_seq=fire,
        situation="reflecting", explicit=False, at=AT,
    ))
    said = _say(q, "I am off on Fridays")
    good = q.append(episodes.claim_extracted(
        for_seq=said, text="is off on Fridays", source_seq=said,
        situation="reflecting", explicit=False, at=AT,
    ))
    taught_on_a_fire = q.append(episodes.claim_extracted(
        for_seq=fire, text="likes tea", source_seq=fire,
        situation="taught", explicit=True, at=AT,
    ))

    learned = derive.Learned.rebuild(q.store)
    found = tidy.misattributed(q.store, learned.claims())
    assert [(c.seq, f) for c, f in found] == [(bad, fire)]

    tidy.run(q, learned, provider.FakeProvider({}).complete, now=AT)
    assert [p["claim_seq"] for _, p in _kinds(q, episodes.CLAIM_RETRACTED)] == [bad]
    live = {c.seq for c in derive.Learned.rebuild(q.store).claims()}
    assert bad not in live and {good, taught_on_a_fire} <= live
