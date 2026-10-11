"""Omega keeping its own memory clean — DL-087 #4.

The pair `CLAUDE.md` asks for:

    capability  duplicates are merged, old changeable state is marked with its
                age, and a contradiction is asked once and settled by the answer
    violation   no claim ever leaves recall unless a ``claim.merged`` folded it
                or a ``claim.resolved`` dropped it

`_only_named_left` is the violation guard, and every case that runs a pass ends
on it. The equivalence cases are the other half of "never patch a derived
view": with no hygiene episodes in the log, every surface renders exactly as it
did before this existed, and with them it differs only where a record says so.
"""

from __future__ import annotations

import json

import pytest

from omega import derive, episodes, learn, provider, tidy, tools, turn
from omega.executor import Executor
from omega.memory import MemoryStore
from omega.queue import EventQueue

from tests.teaching import _log

JAN = "2026-01-01T09:00:00+00:00"
MAR = "2026-03-01T09:00:00+00:00"
JUN = "2026-06-01T09:00:00+00:00"
AUG = "2026-08-20T09:00:00+00:00"
NOW = "2026-09-01T09:00:00+00:00"
LATER = "2026-09-03T09:00:00+00:00"

HYGIENE = {
    episodes.CLAIM_MERGED,
    episodes.CLAIM_STALE,
    episodes.CLAIM_CONTRADICTION,
    episodes.CLAIM_ASKED,
    episodes.CLAIM_RESOLVED,
    episodes.MEMORY_TIDIED,
}


@pytest.fixture
def q(store: MemoryStore) -> EventQueue:
    return EventQueue(store)


class _Model:
    """A fake provider for all three roles that counts `learn` calls and
    answers them from a script — a list consumed in order, so a case that
    makes a call it did not plan for fails rather than reusing an answer."""

    def __init__(self, learn=(), *, verdict="SPEAK", reply="Morning."):
        self.learn = list(learn)
        self.learn_prompts: list[str] = []
        self._fake = provider.FakeProvider(
            {
                provider.JUDGE: lambda role, messages: verdict,
                provider.ACT: lambda role, messages: reply,
                provider.LEARN: self._learn,
            }
        )

    def _learn(self, role, messages):
        self.learn_prompts.append(json.dumps(messages))
        if not self.learn:
            raise AssertionError("an unscripted learn call")
        return self.learn.pop(0)

    @property
    def complete(self):
        return self._fake.complete


def _claim(q: EventQueue, text: str, *, at: str, explicit: bool = True, trigger=None) -> int:
    said = q.append(episodes.inbound(f"note: {text}", channel="tray", at=at))
    return q.append(
        episodes.claim_extracted(
            for_seq=said,
            text=text,
            source_seq=said,
            situation="taught in the tray",
            explicit=explicit,
            trigger=trigger,
            at=at,
        )
    )


def _executor(q: EventQueue, model: _Model) -> Executor:
    ex = Executor(q, complete=model.complete)
    ex.recover()
    ex.drain()
    return ex


def _kinds(q: EventQueue, kind: str) -> list[dict]:
    return [p for _, p in _log(q) if p.get("kind") == kind]


def _live(q: EventQueue) -> list[derive.Claim]:
    return derive.Learned.rebuild(q.store).claims()


def _only_named_left(before: list[derive.Claim], q: EventQueue) -> None:
    """The violation metric: a claim leaves the set only by a merge or an answer."""
    after = {c.seq for c in _live(q)}
    named = {s for p in _kinds(q, episodes.CLAIM_MERGED) for s in p["folded"]}
    named |= {p["dropped"] for p in _kinds(q, episodes.CLAIM_RESOLVED)}
    gone = {c.seq for c in before} - after
    assert gone <= named, f"claims {sorted(gone - named)} left with no record naming them"


# --- duplicates -------------------------------------------------------------


def test_duplicates_are_merged_and_the_originals_stay(q: EventQueue) -> None:
    said = _claim(q, "Keep status updates short.", at=JAN)
    again = _claim(q, "keep  the status updates SHORT", at=AUG, explicit=False)
    other = _claim(q, "Prefers tea over coffee", at=JAN)
    before = _live(q)
    ex = _executor(q, _Model())

    assert ex.tidy(now=NOW) is True

    merged = _kinds(q, episodes.CLAIM_MERGED)
    # The explicit one survives the newer inferred one: a merge never replaces
    # what the person said with what omega guessed.
    assert merged == [
        {"v": 1, "kind": episodes.CLAIM_MERGED, "kept": said, "folded": [again], "at": NOW}
    ]
    assert [c.seq for c in _live(q)] == [said, other]
    # Both originals are still in the log, word for word.
    texts = [p["text"] for p in _kinds(q, episodes.CLAIM_EXTRACTED)]
    assert "keep  the status updates SHORT" in texts and "Keep status updates short." in texts
    # The survivor was said again in August, so that is its last-said time.
    learned = derive.Learned.rebuild(q.store)
    assert learned.raw(said).at == AUG
    _only_named_left(before, q)


def test_the_same_words_under_a_different_trigger_are_not_merged(q: EventQueue) -> None:
    _claim(q, "keep it short", at=JAN)
    _claim(q, "keep it short", at=JAN, trigger={"hours": [6, 10]})
    ex = _executor(q, _Model())
    ex.tidy(now=NOW)
    assert _kinds(q, episodes.CLAIM_MERGED) == []
    assert len(_live(q)) == 2


def test_a_rerun_changes_nothing(q: EventQueue) -> None:
    _claim(q, "Keep status updates short", at=JAN)
    _claim(q, "keep status updates short", at=MAR)
    _claim(q, "They work at Acme", at=JAN)
    _claim(q, "They live in Berlin", at=JAN)
    _claim(q, "They live in Lisbon", at=AUG)
    model = _Model(['{"contradicts": [1]}'])
    ex = _executor(q, model)
    ex.tidy(now=NOW)
    first = _live(q)
    notes = [c.note(NOW) for c in first]
    head = q.head()

    # Inside the day the gate is shut, and nothing is written at all.
    assert ex.tidy(now=NOW) is False
    assert q.head() == head

    # A day later it runs again and finds nothing left to do: no claim record,
    # no model call, the same view.
    assert ex.tidy(now=LATER) is True
    new = [p for seq, p in _log(q) if seq > head]
    assert [p["kind"] for p in new] == [episodes.MEMORY_TIDIED]
    assert new[0]["merged"] == new[0]["staled"] == new[0]["contradictions"] == 0
    assert model.learn == [] and len(model.learn_prompts) == 1
    again = _live(q)
    assert again == first and [c.note(NOW) for c in again] == notes


def test_nothing_to_tidy_writes_nothing(q: EventQueue) -> None:
    q.append(episodes.inbound("hello", channel="tray", at=JAN))
    ex = _executor(q, _Model())
    head = q.head()
    assert ex.tidy(now=NOW) is False
    assert q.head() == head


# --- staleness --------------------------------------------------------------


def test_old_changeable_state_is_marked_and_recalled_with_its_age(q: EventQueue) -> None:
    job = _claim(q, "They work at Acme as a designer", at=JAN)
    before = _live(q)
    ex = _executor(q, _Model())
    ex.tidy(now=NOW)

    assert _kinds(q, episodes.CLAIM_STALE) == [
        {
            "v": 1,
            "kind": episodes.CLAIM_STALE,
            "claim_seq": job,
            "category": "job",
            "basis_at": JAN,
            "age_days": 243,
            "at": NOW,
        }
    ]
    claims = _live(q)
    # Marked, not hidden: still in the set, still recalled, now with its age.
    assert [c.seq for c in claims] == [job]
    shown = tools.recall(claims, at=NOW)
    assert "- They work at Acme as a designer (always) [last said 2026-01-01, 8 months ago; may be out of date]" in shown
    section = turn._learned_section(claims, at=NOW)
    assert "- They work at Acme as a designer [last said 2026-01-01, 8 months ago" in section
    _only_named_left(before, q)


def test_recent_changeable_state_is_not_marked(q: EventQueue) -> None:
    _claim(q, "They plan to visit Rome", at=AUG)
    ex = _executor(q, _Model())
    ex.tidy(now=NOW)
    assert _kinds(q, episodes.CLAIM_STALE) == []


def test_durable_facts_are_never_aged(q: EventQueue) -> None:
    _claim(q, "They prefer short replies with no preamble", at="2020-01-01T09:00:00+00:00")
    _claim(q, "Uses British spelling", at="2019-05-01T09:00:00+00:00")
    ex = _executor(q, _Model())
    ex.tidy(now=NOW)
    assert _kinds(q, episodes.CLAIM_STALE) == []
    assert all(c.note(NOW) == "" for c in _live(q))


def test_saying_it_again_clears_the_mark(q: EventQueue) -> None:
    job = _claim(q, "They work at Acme", at=JAN)
    ex = _executor(q, _Model())
    ex.tidy(now=NOW)
    assert derive.Learned.rebuild(q.store).stale_marks
    _claim(q, "they work at Acme", at=LATER, explicit=False)
    ex.tidy(now="2026-09-05T09:00:00+00:00")
    learned = derive.Learned.rebuild(q.store)
    assert [c.seq for c in learned.claims()] == [job]
    assert learned.stale_marks == {}
    assert learned.claims()[0].note(NOW) == ""


# --- contradictions ---------------------------------------------------------


def _berlin_lisbon(q: EventQueue) -> tuple[int, int]:
    return (
        _claim(q, "They live in Berlin", at=JUN),
        _claim(q, "They live in Lisbon", at=AUG),
    )


def _look(q: EventQueue, ex: Executor, at: str) -> str:
    q.append(episodes.inbound("Nothing new since you last looked.", channel="self", at=at))
    ex.drain()
    return _kinds(q, episodes.TURN_COMPLETED)[-1].get("reply") or ""


def test_a_contradiction_is_recorded_and_shown_on_both_claims(q: EventQueue) -> None:
    berlin, lisbon = _berlin_lisbon(q)
    model = _Model(['{"contradicts": [1]}'])
    ex = _executor(q, model)
    ex.tidy(now=NOW)

    [record] = _kinds(q, episodes.CLAIM_CONTRADICTION)
    assert record["claims"] == [berlin, lisbon]
    assert record["question"] == (
        'I have two things written down that cannot both be true: 1) "They live '
        'in Berlin" (2026-06-01) and 2) "They live in Lisbon" (2026-08-20). '
        "Which is right now? Reply 1 or 2."
    )
    # The model was asked *whether*, never which is right.
    assert "Do not say which is right" in model.learn_prompts[0]
    # Until they answer, both stand and both say so.
    shown = tools.recall(_live(q), at=NOW)
    assert '- They live in Berlin (always) [conflicts with "They live in Lisbon"; not settled yet]' in shown
    assert '- They live in Lisbon (always) [conflicts with "They live in Berlin"; not settled yet]' in shown


def test_a_contradiction_is_asked_exactly_once(q: EventQueue) -> None:
    _berlin_lisbon(q)
    model = _Model(['{"contradicts": [1]}'])
    ex = _executor(q, model)
    ex.tidy(now=NOW)
    [record] = _kinds(q, episodes.CLAIM_CONTRADICTION)

    # Their own message never carries it: that turn is about what they asked.
    q.append(episodes.inbound("what's on today?", channel="tray", at=NOW))
    ex.drain()
    assert record["question"] not in (_kinds(q, episodes.TURN_COMPLETED)[-1]["reply"] or "")

    first = _look(q, ex, NOW)
    assert first == f"Morning.\n\n{record['question']}"
    second = _look(q, ex, LATER)
    assert record["question"] not in second
    assert len(_kinds(q, episodes.CLAIM_ASKED)) == 1

    # And a later pass that judges the same pair a contradiction again does
    # not record it again: the pair is never re-asked.
    _claim(q, "They live in Berlin, Germany", at=LATER)
    model.learn.append('{"contradicts": [1, 2]}')
    ex.tidy(now="2026-09-05T09:00:00+00:00")
    pairs = [tuple(p["claims"]) for p in _kinds(q, episodes.CLAIM_CONTRADICTION)]
    assert len(pairs) == len(set(pairs))


def test_with_a_brief_standing_the_question_waits_for_the_brief(q: EventQueue) -> None:
    q.append(
        episodes.schedule_created(
            id="brief", instruction=tools.brief_instruction(), cron="0 8 *", at=JAN
        )
    )
    _berlin_lisbon(q)
    ex = _executor(q, _Model(['{"contradicts": [1]}']))
    ex.tidy(now=NOW)
    [record] = _kinds(q, episodes.CLAIM_CONTRADICTION)

    assert record["question"] not in _look(q, ex, NOW)
    q.append(
        episodes.inbound(
            tools.brief_instruction(),
            channel="schedule",
            schedule_id="brief",
            schedule_slot=NOW,
            at=NOW,
        )
    )
    ex.drain()
    assert _kinds(q, episodes.TURN_COMPLETED)[-1]["reply"].endswith(record["question"])
    assert len(_kinds(q, episodes.CLAIM_ASKED)) == 1


def test_a_bare_answer_supersedes_the_losing_claim(q: EventQueue) -> None:
    berlin, lisbon = _berlin_lisbon(q)
    model = _Model(['{"contradicts": [1]}'])
    ex = _executor(q, model)
    ex.tidy(now=NOW)
    _look(q, ex, NOW)
    before = _live(q)

    q.append(episodes.inbound("2", channel="tray", at=NOW))
    ex.drain()
    # Sooner than the daily gate: the person has answered something.
    assert ex.tidy(now=NOW) is True

    [resolved] = _kinds(q, episodes.CLAIM_RESOLVED)
    assert (resolved["kept"], resolved["dropped"]) == (lisbon, berlin)
    claims = _live(q)
    assert [c.seq for c in claims] == [lisbon]
    assert claims[0].note(NOW) == ""
    # Read without a model: a bare "2" needs none.
    assert len(model.learn_prompts) == 1
    # Berlin is still in the log, as taught.
    assert any(p["text"] == "They live in Berlin" for p in _kinds(q, episodes.CLAIM_EXTRACTED))
    _only_named_left(before, q)


def test_a_worded_answer_is_read_by_the_model_and_unclear_changes_nothing(
    q: EventQueue,
) -> None:
    berlin, lisbon = _berlin_lisbon(q)
    model = _Model(['{"contradicts": [1]}'])
    ex = _executor(q, model)
    ex.tidy(now=NOW)
    _look(q, ex, NOW)
    before = _live(q)

    q.append(episodes.inbound("hm, why do you ask?", channel="tray", at=NOW))
    ex.drain()
    model.learn.append('{"answer": "unclear"}')
    ex.tidy(now=NOW)
    assert _kinds(q, episodes.CLAIM_RESOLVED) == []
    assert {c.seq for c in _live(q)} == {berlin, lisbon}

    q.append(episodes.inbound("Still Berlin actually, Lisbon fell through", channel="tray", at=NOW))
    ex.drain()
    model.learn.append('{"answer": "1"}')
    ex.tidy(now=NOW)
    [resolved] = _kinds(q, episodes.CLAIM_RESOLVED)
    assert (resolved["kept"], resolved["dropped"]) == (berlin, lisbon)
    # Each message was read once: the unclear one was not asked about again.
    assert len(model.learn_prompts) == 3
    _only_named_left(before, q)


def test_a_bad_judgement_fails_closed_and_is_retried(q: EventQueue) -> None:
    _claim(q, "Keep updates short", at=JAN)
    _claim(q, "keep updates short", at=MAR)
    _berlin_lisbon(q)
    before = _live(q)
    model = _Model(["they look different to me", '{"contradicts": [1]}'])
    ex = _executor(q, model)
    ex.tidy(now=NOW)

    [receipt] = _kinds(q, episodes.MEMORY_TIDIED)
    assert receipt["reason"].startswith("contradiction: answer was not JSON")
    assert receipt["checked_through"] == 0
    assert _kinds(q, episodes.CLAIM_CONTRADICTION) == []
    # The code half needed no model and stands.
    assert receipt["merged"] == 1
    # Nothing was guessed: both claims stand, neither marked.
    assert all(c.note(NOW) == "" for c in _live(q))

    ex.tidy(now=LATER)
    assert len(_kinds(q, episodes.CLAIM_CONTRADICTION)) == 1
    _only_named_left(before, q)


def test_unrelated_claims_never_reach_the_model(q: EventQueue) -> None:
    _claim(q, "Prefers tea over coffee", at=JAN)
    _claim(q, "Uses British spelling", at=JAN)
    model = _Model()
    ex = _executor(q, model)
    ex.tidy(now=NOW)
    assert model.learn_prompts == []
    assert _kinds(q, episodes.MEMORY_TIDIED)[0]["checked_through"] > 0


# --- equivalence ------------------------------------------------------------


def test_with_no_hygiene_episodes_every_surface_renders_as_before(q: EventQueue) -> None:
    _claim(q, "They work at Acme", at=JAN)
    _claim(q, "keep it short", at=JAN, trigger={"any": ["standup"]})
    claims = _live(q)
    assert all(c.stale is None and c.conflicts == () and c.note(NOW) == "" for c in claims)
    assert turn._learned_section(claims, at=NOW) == turn._learned_section(claims)
    assert turn._learned_section(claims) == (
        "What you have learned about working with this person:\n"
        "- They work at Acme\n- keep it short\n\n"
    )
    assert tools.recall(claims, at=NOW) == tools.recall(claims)
    assert tools.recall(claims).splitlines()[1:] == [
        "- They work at Acme (always)",
        "- keep it short (when you mention standup)",
    ]


def test_recall_differs_after_a_pass_only_where_a_record_says(q: EventQueue) -> None:
    said = _claim(q, "Keep status updates short", at=JAN)
    again = _claim(q, "keep status updates short", at=MAR, explicit=False)
    tea = _claim(q, "Prefers tea over coffee", at=JAN)
    job = _claim(q, "They work at Acme", at=JAN)
    berlin, lisbon = _berlin_lisbon(q)
    before = _live(q)
    old_recall = tools.recall(before, at=NOW).splitlines()
    old_section = turn._learned_section(before, at=NOW).splitlines()

    ex = _executor(q, _Model(['{"contradicts": [1]}']))
    ex.tidy(now=NOW)
    after = _live(q)
    new_recall = tools.recall(after, at=NOW).splitlines()
    new_section = turn._learned_section(after, at=NOW).splitlines()

    noted = {c.seq: c.note(NOW) for c in after if c.note(NOW)}
    assert set(noted) == {job, berlin, lisbon}

    def expected(old: list[str], line_of) -> list[str]:
        """The old lines, minus the folded claim, with each noted claim's note
        appended — and nothing else different."""
        out = []
        for line in old:
            [c] = [c for c in before if line_of(c) == line]
            if c.seq == again:
                continue  # folded, and named by the merge
            out.append(f"{line} [{noted[c.seq]}]" if c.seq in noted else line)
        return out

    # recall: a count header, then one line per claim.
    assert old_recall[0] == "You have written down 6 things about this person."
    assert new_recall[0] == "You have written down 5 things about this person."
    assert new_recall[1:] == expected(
        old_recall[1:], lambda c: f"- {c.text} ({learn.when_phrase(c.trigger)})"
    )
    # The learned section: the same heading, the same order, the same trailing
    # blank line.
    assert new_section[0] == old_section[0] and new_section[-1] == old_section[-1] == ""
    assert new_section[1:-1] == expected(old_section[1:-1], lambda c: f"- {c.text}")
    assert tea in {c.seq for c in after} and said in {c.seq for c in after}
    _only_named_left(before, q)


# --- the records -------------------------------------------------------------


def test_the_records_refuse_what_they_cannot_mean() -> None:
    with pytest.raises(episodes.BadPayload):
        episodes.claim_merged(kept=3, folded=[3, 4])
    with pytest.raises(episodes.BadPayload):
        episodes.claim_merged(kept=3, folded=[])
    with pytest.raises(episodes.BadPayload):
        episodes.claim_stale(claim_seq=3, category="mood", basis_at=JAN, age_days=1)
    with pytest.raises(episodes.BadPayload):
        episodes.claim_contradiction(claims=[5, 5], question="?")
    with pytest.raises(episodes.BadPayload):
        episodes.claim_resolved(contradiction_seq=9, kept=5, dropped=5, answer_seq=10)
    with pytest.raises(episodes.BadPayload):
        episodes.memory_tidied(checked_through=-1, heard_through=0)
    for payload in (
        episodes.claim_merged(kept=3, folded=[4]),
        episodes.claim_stale(claim_seq=3, category="job", basis_at=JAN, age_days=200),
        episodes.claim_contradiction(claims=[4, 3], question="which?"),
        episodes.claim_asked(for_seq=7, contradiction_seq=6),
        episodes.claim_resolved(contradiction_seq=6, kept=3, dropped=4, answer_seq=8),
        episodes.memory_tidied(checked_through=0, heard_through=0),
    ):
        assert episodes.decode(episodes.encode(payload)) == payload
        assert payload["kind"] in turn.NOT_RECALLED


def test_a_resolution_naming_a_claim_outside_the_pair_drops_nothing() -> None:
    learned = derive.Learned()
    for seq, text in ((2, "They live in Berlin"), (4, "They live in Lisbon"), (6, "Tea")):
        learned.apply(
            seq,
            episodes.claim_extracted(
                for_seq=seq - 1, text=text, source_seq=seq - 1,
                situation="taught", explicit=True, at=JAN,
            ),
        )
    learned.apply(7, episodes.claim_contradiction(claims=[2, 4], question="which?"))
    learned.apply(8, episodes.claim_resolved(contradiction_seq=7, kept=2, dropped=6, answer_seq=7))
    assert [c.seq for c in learned.claims()] == [2, 4, 6]


def test_the_answer_reader_takes_only_a_bare_choice() -> None:
    assert tidy.read_answer("2") == 2
    assert tidy.read_answer(" 1. ") == 1
    assert tidy.read_answer("option 2") == 2
    assert tidy.read_answer("12") is None
    assert tidy.read_answer("1 or 2, not sure") is None
