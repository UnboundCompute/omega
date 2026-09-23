"""Forgetting a claim — DL-048.

The claim-side sibling of the cancel cases in ``test_schedule_learn.py``, and
the refusal cases matter more here. A cancel that silently no-ops leaves a
reminder firing while the receipt says it stopped, which the next fire reveals.
A retraction that drops the wrong claim produces no event at all — the only
symptom is omega gradually not doing something it was told, months later, with
nothing in the log pointing at the cause. So an id naming nothing believed
fails the whole note.

The fold cases pin the property the whole memory design rests on: a retraction
is an *append*. The ``claim.extracted`` record stays, ``Learned`` stops
including it, and re-deriving the log from scratch reaches the same active set.
"Forget this" sounding like a delete is exactly why that needs a test rather
than a comment.
"""

from __future__ import annotations

import pytest

from omega import derive, episodes, learn, provider, turn
from omega.queue import EventQueue

from tests.teaching import (
    LEARNED_HEADING,
    _a_claim,
    _answer,
    _claim_obj,
    _claims_in,
    _drain,
    _learned_section_of,
    _log,
    _replies,
    _retractions_in,
    _teach_text,
    _Teaching,
)

BLACK = "I take my coffee black."


# --- parsing ----------------------------------------------------------------


def test_a_retract_names_a_claim_omega_currently_believes():
    got = learn.parse_answer(
        _answer(retract=[7]), known=[_claim_obj(7, BLACK)]
    )
    assert got.retract == [7]


def test_a_retract_naming_nothing_believed_is_refused():
    """The rule this whole path exists for. A hallucinated id would drop an
    instruction the person deliberately authored, and nothing would ever fire
    to reveal it."""
    with pytest.raises(learn.NotExtracted, match="not something omega currently"):
        learn.parse_answer(_answer(retract=[99]), known=[_claim_obj(7, BLACK)])


def test_a_retract_with_nothing_believed_at_all_is_refused():
    with pytest.raises(learn.NotExtracted):
        learn.parse_answer(_answer(retract=[7]))


def test_a_repeated_retract_id_is_kept_once():
    got = learn.parse_answer(
        _answer(retract=[7, 7]), known=[_claim_obj(7, BLACK)]
    )
    assert got.retract == [7]


def test_a_boolean_is_not_a_claim_id():
    """``isinstance(True, int)`` is true, so ``True`` would otherwise sail
    through as claim 1 — and claim 1 is a real claim in any log long enough to
    have one."""
    with pytest.raises(learn.NotExtracted, match="not a number"):
        learn.parse_answer(_answer(retract=[True]), known=[_claim_obj(1, BLACK)])


def test_a_retract_that_is_not_a_list_is_refused():
    with pytest.raises(learn.NotExtracted, match='"retract" was not a list'):
        learn.parse_answer('{"claims": [], "retract": 7}', known=[_claim_obj(7, BLACK)])


def test_an_answer_with_no_retract_field_is_fine():
    """Additive on a shape that already existed, exactly as ``cancel`` was."""
    assert learn.parse_answer(_answer()).retract == []


def test_the_prompt_offers_retract_and_supersedes_as_different_things():
    """The model has to be told which is which, or "I've gone vegetarian" and
    "forget that I'm vegetarian" collapse into the same write."""
    messages = learn._messages(
        "forget that", known=[_claim_obj(7, BLACK)], running=[], context=""
    )
    system = messages[0]["content"]
    # The whole distinguishing sentence, not just the field name: the JSON
    # shape line mentions "retract" too, so asserting only that would keep
    # passing with the explanation deleted — and the explanation is the part
    # that stops "I've gone vegetarian" being written as a retraction.
    assert "list of ids of remembered things to forget" in system
    assert "puts nothing" in system
    assert "in its place" in system
    assert "supersedes" in system
    assert "[7]" in system, "the model can only name an id it was shown"


# --- the episode ------------------------------------------------------------


def test_a_retraction_naming_its_own_turn_is_refused():
    """Two seqs mixed up, not a claim anybody meant to forget — the same
    confusion ``supersedes`` guards against, from the other side."""
    with pytest.raises(episodes.BadPayload, match="not the turn"):
        episodes.claim_retracted(for_seq=5, claim_seq=5)


@pytest.mark.parametrize("bad", [0, -1, True, "3", None, 1.0])
def test_a_claim_seq_that_is_not_a_seq_is_refused(bad):
    """A retraction is the one write that makes omega less capable, so the id
    it carries has to be a real seq before the record exists — not discovered
    to be nonsense at fold time, where there is nobody left to tell."""
    with pytest.raises(episodes.BadPayload, match="claim_seq"):
        episodes.claim_retracted(for_seq=9, claim_seq=bad)


def test_a_retraction_is_not_projected():
    """``claim.extracted`` is withheld because ``act.py`` refuses a second wire
    update, so the receipt rides the reply. Identical here."""
    from omega.projection import NOT_PROJECTED, project

    # The *decision*, not the return value. ``project`` answers ``None`` for a
    # kind nobody wired up as readily as for one deliberately withheld, so
    # asserting only the None would pass if this kind were simply forgotten —
    # the distinction ``test_every_m1_kind_is_accounted_for`` exists to keep.
    assert episodes.CLAIM_RETRACTED in NOT_PROJECTED
    assert project(episodes.claim_retracted(for_seq=5, claim_seq=3), 6) is None


# --- the fold ---------------------------------------------------------------


def _folded(*payloads: dict) -> derive.Learned:
    return derive.Learned.fold(enumerate(payloads, start=1))


def _taught(text: str, *, source_seq: int = 1) -> dict:
    return episodes.claim_extracted(
        for_seq=source_seq,
        text=text,
        source_seq=source_seq,
        situation="taught in the tray",
        explicit=True,
    )


def test_a_retracted_claim_leaves_the_active_set():
    learned = _folded(
        _taught(BLACK),
        episodes.claim_retracted(for_seq=2, claim_seq=1),
    )
    assert learned.claims() == []


def test_the_record_survives_the_retraction():
    """The property every memory decision here rests on. If forgetting deleted
    the record, DL-017's rebuild-from-log would stop being true for the one
    kind of memory that changes behaviour silently."""
    payloads = [_taught(BLACK), episodes.claim_retracted(for_seq=2, claim_seq=1)]
    kinds = [p["kind"] for p in payloads]
    assert episodes.CLAIM_EXTRACTED in kinds
    # And a fresh fold over the same log reaches the same answer, which is what
    # "derived, never stored" has to mean.
    assert _folded(*payloads).claims() == []


def test_retracting_one_claim_leaves_the_others_standing():
    learned = _folded(
        _taught(BLACK, source_seq=1),
        _taught("I prefer short replies.", source_seq=2),
        episodes.claim_retracted(for_seq=3, claim_seq=1),
    )
    assert [c.text for c in learned.claims()] == ["I prefer short replies."]


def test_retracting_something_already_gone_is_not_an_error():
    """A derived view does not adjudicate a race in an append-only log. The
    *refusal* lives at extraction, where there is a person to tell."""
    learned = _folded(
        _taught(BLACK),
        episodes.claim_retracted(for_seq=2, claim_seq=1),
        episodes.claim_retracted(for_seq=3, claim_seq=1),
    )
    assert learned.claims() == []


# --- the receipt ------------------------------------------------------------


def test_the_receipt_names_the_sentence_and_not_the_seq():
    """Telling someone claim 41 has been forgotten is a confirmation they
    cannot check — ``cancel_schedules``' argument, and it binds harder here
    because a retraction is the one thing omega does that makes it less
    capable."""
    text = learn.receipt([], forgotten=[_claim_obj(41, BLACK)])
    assert BLACK in text
    assert "41" not in text


def test_a_note_that_only_forgets_is_not_an_empty_extraction():
    """Retracting *is* recording something. If this fell through to "nothing
    was recorded" the person would be told the opposite of what happened."""
    text = learn.receipt([], forgotten=[_claim_obj(41, BLACK)])
    assert "did not find anything" not in text


def test_a_genuinely_empty_note_still_says_nothing_was_recorded():
    assert "did not find anything" in learn.receipt([])


# --- end to end -------------------------------------------------------------


def test_a_teach_that_forgets_removes_the_claim_from_the_next_prompt(store):
    """The capability, graded on the world: not that a record was written, but
    that the belief stops reaching the model."""
    q = EventQueue(store)
    q.append(episodes.inbound(_teach_text("I take my coffee black."), channel="tray"))
    _drain(q, _Teaching(learn_answer=_answer(_a_claim(BLACK, trigger={"any": ["coffee"]}))))

    seq = _claims_in(q)[0][0]

    q.append(episodes.inbound(_teach_text("Forget the coffee thing."), channel="tray"))
    _drain(q, _Teaching(learn_answer=_answer(retract=[seq])))

    assert len(_retractions_in(q)) == 1

    q.append(episodes.inbound("making a coffee round", channel="tray"))
    after = _Teaching(learn_answer="never asked", verdict="SPEAK")
    _drain(q, after)
    assert BLACK not in _learned_section_of(after.last())


def test_the_person_is_told_which_sentence_was_forgotten(store):
    q = EventQueue(store)
    q.append(episodes.inbound(_teach_text("I take my coffee black."), channel="tray"))
    _drain(q, _Teaching(learn_answer=_answer(_a_claim(BLACK))))
    seq = _claims_in(q)[0][0]

    q.append(episodes.inbound(_teach_text("Forget the coffee thing."), channel="tray"))
    _drain(q, _Teaching(learn_answer=_answer(retract=[seq]), reply="Done."))

    assert BLACK in _replies(q)[-1]


# --- the violation ----------------------------------------------------------


def test_an_ordinary_turn_retracts_nothing(store):
    """**The violation metric**: nothing is retracted that was not named. An
    ordinary turn must not reach the learn model at all, so it cannot forget
    anything even if the model would have liked to."""
    q = EventQueue(store)
    q.append(episodes.inbound(_teach_text("I take my coffee black."), channel="tray"))
    _drain(q, _Teaching(learn_answer=_answer(_a_claim(BLACK))))
    seq = _claims_in(q)[0][0]

    q.append(episodes.inbound("what did I have for lunch", channel="tray"))
    ordinary = _Teaching(learn_answer=_answer(retract=[seq]))
    _drain(q, ordinary)

    assert _retractions_in(q) == []
    assert not [c for c in ordinary.calls if c[0] == provider.LEARN]


def test_a_teach_that_asks_for_nothing_retracts_nothing(store):
    q = EventQueue(store)
    q.append(episodes.inbound(_teach_text("I take my coffee black."), channel="tray"))
    _drain(q, _Teaching(learn_answer=_answer(_a_claim(BLACK))))

    q.append(episodes.inbound(_teach_text("I also like tea."), channel="tray"))
    _drain(q, _Teaching(learn_answer=_answer(_a_claim("likes tea"))))

    assert _retractions_in(q) == []


def test_a_note_naming_an_unknown_claim_writes_nothing_at_all(store):
    """All-or-nothing, and the claim in the same note goes down with it. A
    partial write would leave the person confirming a record that is quietly
    short."""
    q = EventQueue(store)
    q.append(episodes.inbound(_teach_text("Forget something."), channel="tray"))
    _drain(q, _Teaching(learn_answer=_answer(_a_claim("this must not be filed"), retract=[999])))

    assert _claims_in(q) == []
    assert _retractions_in(q) == []
    assert "could not write that down" in _replies(q)[-1]
