"""Extraction, and the receipt that checks it — DL-043.

Three groups, and the third is the one that decides whether any of this is
real.

The gate cases pin a string that lives in two languages at once. The parse
cases are mostly about *refusing*, because an extractor that accepts a slightly
wrong answer files a slightly wrong memory and nobody finds out. The end-to-end
cases are the pair `CLAUDE.md` asks for: the capability is that a teach drop
records something that later fires, and the violation it spawns is extraction
happening anywhere else.

One case here is worth more than the rest: a model that says it memorised
everything while the log holds nothing must produce a receipt that says nothing
was recorded. That is *grade the world, not the words* at the only boundary in
omega where the two can disagree without anybody noticing.
"""

from __future__ import annotations

import json

import pytest

from omega import derive, episodes, learn, provider, turn
from omega.queue import EventQueue

from tests.teaching import (
    LEARNED_HEADING,
    TRAY_INSTRUCTION,
    _a_claim,
    _answer,
    _claim_obj,
    _claims_in,
    _drain,
    _learned_section_of,
    _log,
    _replies,
    _teach_text,
    _Teaching,
    _text_of,
)

# The literal instruction the tray builds — `TrayViewModel.swift:523-529`,
# reproduced rather than imported, because the point of the case is that the
# two halves agree while having no way to share a constant.

# --- the gate ---------------------------------------------------------------


def test_the_tray_instruction_marks_a_teach_and_the_note_is_what_follows():
    """The cross-language contract, pinned on this side.

    ``TrayModelTests.swift:28`` asserts the submission contains the same
    fragment. Neither half can be reworded without the other's test failing
    first, which is the whole mitigation for a seam that has no shared
    constant to hold it.
    """
    note = learn.teaching_note(_teach_text("Prefer concise status updates."))
    assert note == "Prefer concise status updates."
    assert learn.TEACH_MARKER in TRAY_INSTRUCTION


def test_an_ordinary_message_is_not_a_teach():
    """The violation metric's first line of defence, at the cheapest layer."""
    assert learn.teaching_note("what did I have for lunch") is None
    assert learn.teaching_note("") is None


def test_an_instruction_with_an_empty_note_is_not_a_teach():
    """Nothing to extract, so nothing runs."""
    assert learn.teaching_note(f"{TRAY_INSTRUCTION}\n\n   \n  ") is None


def test_an_instruction_with_no_note_at_all_degrades_toward_extracting():
    """The deliberate direction of the degradation, stated rather than left to
    whichever branch happened to win.

    A bare instruction has no blank line, so it takes the same path as a
    reworded tray: the note becomes the whole text. That costs one cheap call
    that returns nothing and a receipt saying nothing was recorded — which the
    person can see. The alternative, treating it as not-a-teach, is silent, and
    a teach that vanishes silently is the failure DL-034's done bar exists to
    catch. The tray refuses to send this shape anyway (``canSend`` requires
    text), so the case is about which way the code leans, not about traffic.
    """
    assert learn.teaching_note(TRAY_INSTRUCTION) == TRAY_INSTRUCTION


def test_a_teach_whose_shape_changed_is_still_a_teach():
    """Detection degrades toward extracting one sentence too many, never
    toward silently dropping a teach. If the tray ever stops putting a blank
    line in, the note is the whole text rather than nothing."""
    text = f"{TRAY_INSTRUCTION} Prefer short updates."
    assert learn.teaching_note(text) == text


# --- parsing the model's answer ---------------------------------------------


def test_a_claim_list_parses():
    (claim,) = learn.parse_claims(
        _answer(_a_claim(trigger={"any": ["deploy"]}))
    )
    assert claim["text"] == "keep status updates short"
    assert claim["trigger"] == {"any": ["deploy"]}
    assert claim["supersedes"] is None


def test_a_fenced_answer_parses():
    """The one leniency, and it is a formatting habit rather than a different
    answer."""
    body = _answer(_a_claim())
    (claim,) = learn.parse_claims(f"```json\n{body}\n```")
    assert claim["text"] == "keep status updates short"


def test_an_empty_list_parses_and_means_nothing_to_learn():
    """Distinct from a failure, and the receipt says something different for
    each. Collapsing them would tell a person their note was unusable when the
    model simply found no instruction in it."""
    assert learn.parse_claims('{"claims": []}') == []


@pytest.mark.parametrize(
    "answer",
    [
        "",
        "Sure! I have recorded that for you.",
        '{"result": []}',
        '{"claims": "keep it short"}',
        '{"claims": [["keep it short"]]}',
        '{"claims": [{"situation": "tray"}]}',
        '{"claims": [{"text": "  ", "situation": "tray"}]}',
        '{"claims": [{"text": "x"}]}',
    ],
)
def test_an_answer_that_is_not_a_claim_list_is_refused(answer):
    """Strict like the judge, and for the judge's reason: a parser that hunted
    for a claim inside prose would read a model that ignored the format as
    though it had followed it."""
    with pytest.raises(learn.NotExtracted):
        learn.parse_claims(answer)


def test_too_many_claims_from_one_note_is_refused():
    """One note is one thought. A model returning a dozen has decomposed
    rather than understood, and every fragment would compete for the prompt
    budget on every future turn."""
    many = [_a_claim(f"claim {i}") for i in range(learn.MAX_CLAIMS_PER_NOTE + 1)]
    with pytest.raises(learn.NotExtracted):
        learn.parse_claims(_answer(*many))


def test_an_overlong_claim_is_refused():
    """A claim that is really a paragraph is a permanent tax on prompts that
    have nothing to do with it."""
    long = "x" * (learn.MAX_CLAIM_CHARS + 1)
    with pytest.raises(learn.NotExtracted):
        learn.parse_claims(_answer(_a_claim(long)))


def test_a_supersedes_id_naming_nothing_is_refused():
    """The sharpest one in this section.

    ``Learned.apply`` removes the superseded claim *by key* and succeeds
    silently when the key is absent. So an invented id does not fail anywhere
    — it files a claim that says it replaced something and replaced nothing,
    and the receipt then tells the person a contradiction was resolved that
    was not. Refusing here is the only place that is catchable.
    """
    known = [_claim_obj(4, "an older habit")]
    with pytest.raises(learn.NotExtracted):
        learn.parse_claims(_answer(_a_claim(supersedes=99)), known=known)

    (ok,) = learn.parse_claims(_answer(_a_claim(supersedes=4)), known=known)
    assert ok["supersedes"] == 4


def test_true_is_not_the_claim_at_seq_one():
    """``bool`` is an ``int`` in Python, and a seq of 1 is the commonest seq
    there is."""
    known = [_claim_obj(1, "an older habit")]
    with pytest.raises(learn.NotExtracted):
        learn.parse_claims(_answer(_a_claim(supersedes=True)), known=known)


def test_an_empty_trigger_object_becomes_always():
    """Two spellings of *always* is one too many, and ``episodes`` refuses the
    empty object outright — so it is normalised before it gets there rather
    than turning a reasonable answer into a failed extraction."""
    (claim,) = learn.parse_claims(_answer(_a_claim(trigger={})))
    assert claim["trigger"] is None


# --- the receipt ------------------------------------------------------------


def test_the_receipt_lists_what_was_written_and_when_it_fires():
    """DL-042 #2: the receipt is the precision check, so it has to say the
    thing the person can disagree with — not that something was saved, but
    what it says and when it will apply."""
    written = [_claim_obj(9, "check the changelog", trigger={"any": ["deploy"]})]
    text = learn.receipt(written)
    assert "check the changelog" in text
    assert "deploy" in text


def test_the_receipt_says_so_when_nothing_was_recorded():
    """*Fail closed on empty.* A teach that stored nothing has to say so — the
    failure being guarded against is the person walking away believing omega
    learned something."""
    text = learn.receipt(())
    assert "nothing was recorded" in text
    assert "wrote this down" not in text


def test_the_receipt_says_so_when_extraction_failed():
    """Three-valued, not two. *I recorded nothing* and *I could not record it*
    are different facts and only one of them is worth repeating yourself
    over."""
    failed = learn.receipt((), error="the answer was not JSON")
    empty = learn.receipt(())
    assert "could not" in failed
    assert "not JSON" in failed, "the person cannot act on a failure it does not name"
    assert failed != empty


def test_the_receipt_names_the_claim_a_new_one_replaces():
    """DL-042 #1's escalation, cashed out where the person is already reading.
    Today every claim omega holds was authored deliberately, so every
    supersession is the case the rule wanted escalated."""
    known = [_claim_obj(4, "ignore anything about billing")]
    written = [_claim_obj(9, "flag anything about billing", supersedes=4)]
    text = learn.receipt(written, known=known)
    assert "flag anything about billing" in text
    assert "ignore anything about billing" in text
    assert "replaces" in text


def test_the_receipt_reads_the_same_trigger_fields_the_matcher_does():
    """What makes it a check on the record rather than a second description of
    it. A phrase that omitted a field would let a claim fire on a condition the
    person was never shown."""
    trigger = {"any": ["deploy"], "channel": "tray", "hours": [9, 18]}
    phrase = learn.when_phrase(trigger)
    assert "deploy" in phrase and "tray" in phrase and "9" in phrase
    assert learn.when_phrase(None) == "always"
    assert learn.when_phrase({}) == "always"


# --- the whole path, through a real turn ------------------------------------



def test_a_teach_drop_records_a_claim_that_later_fires(store):
    """**The capability metric.** Not "a claim was filed" — filed is a folder
    (DL-034). The bar is that a later, unrelated-looking turn is changed by
    what the person taught."""
    q = EventQueue(store)
    q.append(episodes.inbound(_teach_text("Check the changelog before deploying."),
                              channel="tray"))
    teaching = _Teaching(
        learn_answer=_answer(
            _a_claim("check the changelog first", trigger={"any": ["deploy"]})
        )
    )
    _drain(q, teaching)

    q.append(episodes.inbound("about to deploy the api", channel="tray"))
    second = _Teaching(learn_answer="never asked", verdict="SPEAK")
    _drain(q, second)

    assert "check the changelog first" in second.last()


def test_an_ordinary_turn_records_nothing(store):
    """**The violation metric**: extraction outside a teach. Structural, needs
    no model, and it is the failure that would poison every later prompt while
    looking like nothing at all."""
    q = EventQueue(store)
    q.append(episodes.inbound("what did I have for lunch", channel="tray"))
    teaching = _Teaching(learn_answer=_answer(_a_claim("this must never be filed")))
    _drain(q, teaching)

    assert _claims_in(q) == []
    assert not [c for c in teaching.calls if c[0] == provider.LEARN], (
        "an ordinary turn must not even ask the learn model"
    )


def test_the_receipt_reaches_the_person_with_the_reply(store):
    """One wire update, carrying both. ``claim.extracted`` is not projected
    (DL-042), so if the receipt did not ride the reply it would not exist."""
    q = EventQueue(store)
    q.append(episodes.inbound(_teach_text("Keep updates short."), channel="tray"))
    teaching = _Teaching(
        learn_answer=_answer(_a_claim("keep status updates short")),
        reply="Got it.",
    )
    _drain(q, teaching)

    (reply,) = _replies(q)
    assert "Got it." in reply
    assert "keep status updates short" in reply


def test_the_receipt_is_rendered_from_the_log_and_not_from_the_model(store):
    """**Grade the world, not the words** — the case this whole path exists to
    survive.

    The tray's instruction ends *"Briefly confirm what you learned"*, so the
    model will happily narrate a successful memorisation whether or not
    anything was written. Here it does exactly that while extraction returns
    nothing, and the receipt has to contradict it.
    """
    q = EventQueue(store)
    q.append(episodes.inbound(_teach_text("Something vague."), channel="tray"))
    teaching = _Teaching(
        learn_answer='{"claims": []}',
        reply="Understood — I have memorised that and will apply it from now on.",
    )
    _drain(q, teaching)

    (reply,) = _replies(q)
    assert "nothing was recorded" in reply
    assert _claims_in(q) == []


def test_a_failed_extraction_does_not_fail_the_turn(store):
    """DL-043 #5. The reply is already composed and is real work; losing it to
    a second call going wrong would make teaching strictly worse than talking,
    and would punish the person for using the feature."""
    q = EventQueue(store)
    q.append(episodes.inbound(_teach_text("Keep updates short."), channel="tray"))
    teaching = _Teaching(learn_answer="Sure, noted!", reply="Got it.")
    _drain(q, teaching)

    outcomes = [
        p.get("outcome")
        for _, p in _log(q)
        if p.get("kind") == episodes.TURN_COMPLETED
    ]
    assert outcomes == ["spoke"]
    (reply,) = _replies(q)
    assert "could not" in reply
    assert "Got it." in reply
    assert _claims_in(q) == []


def test_a_silent_verdict_on_a_teach_still_records_and_still_answers(store):
    """The hole this closes: a teach omega says nothing about is
    indistinguishable from one it dropped.

    Judge calibration is a known open risk, so *the judge went quiet on a
    teach* is a shape that will occur. The record is written anyway and the
    receipt becomes the reply — because the receipt is not conversation, it is
    the record being shown back for checking.
    """
    q = EventQueue(store)
    q.append(episodes.inbound(_teach_text("Keep updates short."), channel="tray"))
    teaching = _Teaching(
        learn_answer=_answer(_a_claim("keep status updates short")),
        verdict="SILENT",
    )
    _drain(q, teaching)

    assert len(_claims_in(q)) == 1
    (reply,) = _replies(q)
    assert "keep status updates short" in reply


def test_a_new_claim_replaces_the_one_it_contradicts(store):
    """DL-042 #3 sited at ingest, end to end: the second teach is told what is
    already known, names the claim it replaces, and the replaced one stops
    firing."""
    q = EventQueue(store)
    q.append(episodes.inbound(_teach_text("Ignore billing."), channel="tray"))
    first = _Teaching(learn_answer=_answer(_a_claim("ignore anything about billing")))
    _drain(q, first)
    ((old_seq, _),) = _claims_in(q)

    q.append(episodes.inbound(_teach_text("Actually, flag billing."), channel="tray"))
    second = _Teaching(
        learn_answer=_answer(
            _a_claim("flag anything about billing", supersedes=old_seq)
        )
    )
    _drain(q, second)

    # The extraction prompt was told what was already known — otherwise the
    # model could not have named it.
    learn_prompts = [
        "\n".join(_text_of(m) for m in messages)
        for role, messages in second.calls
        if role == provider.LEARN
    ]
    assert any("ignore anything about billing" in p for p in learn_prompts)

    assert "replaces" in _replies(q)[-1]

    q.append(episodes.inbound("what about billing", channel="tray"))
    third = _Teaching(learn_answer="never asked")
    _drain(q, third)
    section = _learned_section_of(third.last())
    assert "flag anything about billing" in section
    assert "ignore anything about billing" not in section, (
        "a superseded claim must stop being applied"
    )
    # ...and it is still *in the log*, visible in history, because resolution
    # is append-only (DL-042). Asserting over the whole prompt would have
    # confused "no longer applied" with "erased", and only one of those is true.
    assert "ignore anything about billing" in third.last()


def test_extraction_asks_the_learn_role_and_leaves_the_others_alone(store):
    """DL-043 #7. Reusing ``judge`` would have meant a model change made for
    extraction silently re-calibrating the router that runs on every event."""
    q = EventQueue(store)
    q.append(episodes.inbound(_teach_text("Keep updates short."), channel="tray"))
    teaching = _Teaching(learn_answer=_answer(_a_claim()))
    _drain(q, teaching)

    roles = [role for role, _ in teaching.calls]
    assert roles.count(provider.LEARN) == 1
    assert provider.LEARN in provider.ROLES
