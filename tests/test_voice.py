"""The voice checks, graded against what omega actually said (DL-055).

`evals.py` needs a key and a network, so nothing here runs a scenario. What it
runs is the other half — the :data:`evals.Check` functions, which are pure
functions from an :class:`~omega.evals.Observed` to a grade and so can be
handed a reply directly.

That makes this the place the checks are proven to be *capable of failing*.
Every string below is quoted from ``~/.omega/episodes.log`` on 2026-09-23/24,
not composed for the test: if a check does not fail on the sentence that
motivated it, it would have scored k/k against the live model while the defect
it was written for sat in plain view. Each check is therefore asserted in both
directions — it fires on the real failure, and it stays quiet on a reply in the
register DL-055 settled on.

The one asymmetry is deliberate. ``_a_schedule_reached_the_store`` is the only
check here that reads the world rather than the words, so it is the only one
given a real store — and the case that matters most for it is the unreadable
store, which must come back *undetermined* and never a pass.

No network and no key.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omega import episodes, evals, projection
from omega.memory import MemoryStore
from omega.queue import EventQueue

AT = "2026-09-24T12:00:00+00:00"

# --- quoted from the live store ---------------------------------------------

SAID_IT_FILED = "Reminder set for tomorrow: check whether Albert's chase mechanism is working correctly."
SAID_IT_NOTED = "Reminder noted: **Ask Kaushik for 2 numbers in 2 hours.**"
VENDOR = "I'm Omega, an AI assistant powered by OpenAI."
DISOWNS_MEMORY = (
    "I can remember details during this conversation. Whether I retain them "
    "across separate chats depends on the app's memory settings, so don't "
    "assume I'll remember everything permanently."
)
DECORATED = "Got it—your dog is named **Shiro**, and he's a **Shih Tzu**."
#: The best reply in the corpus, and the register DL-055 is aiming at.
GOOD_DECLINE = "You haven't told me your favorite number yet."


def _observed(reply: str, *, store: Path = Path("/nonexistent")) -> evals.Observed:
    return evals.Observed(
        updates=[
            projection.Update(
                seq=1,
                state="done",
                for_seq=1,
                at=AT,
                kind=episodes.TURN_COMPLETED,
                reply=reply,
                outcome="spoke",
            )
        ],
        said=[],
        store=store,
        seconds=0.1,
    )


# --- claiming a filing the receipt should be making -------------------------


@pytest.mark.parametrize("reply", [SAID_IT_FILED, SAID_IT_NOTED, DECORATED])
def test_prose_that_claims_a_filing_is_caught(reply: str) -> None:
    """All three are real, and all three assert a record in the reply's own
    voice — which is either a duplicate of the receipt beneath it or a lie in
    place of a missing one."""
    assert not evals._left_the_filing_to_the_receipt(_observed(reply)).ok


def test_a_reply_that_leaves_it_to_the_receipt_passes() -> None:
    """The anti-vacuity direction. A check that failed here would fail on
    everything and its score would mean nothing."""
    assert evals._left_the_filing_to_the_receipt(
        _observed("I'll nudge you at 10:16.")
    ).ok


# --- wearing somebody else's identity ---------------------------------------


@pytest.mark.parametrize("reply", [VENDOR, DISOWNS_MEMORY])
def test_the_borrowed_identity_is_caught(reply: str) -> None:
    """Two sides of one failure: calling itself a vendor's assistant, and
    disclaiming a durable log it demonstrably has."""
    assert not evals._not_somebody_elses_product(_observed(reply)).ok


def test_speaking_in_its_own_terms_passes_both_ways() -> None:
    reply = (
        "I'm omega. The model behind this reply can change; what I remember "
        "is on your disk and doesn't."
    )
    assert evals._not_somebody_elses_product(_observed(reply)).ok
    assert evals._speaks_as_omega(_observed(reply)).ok


def test_a_reply_that_never_names_omega_fails_the_capability() -> None:
    """What the scenario's own counter-input is expected to do — asserted here
    so the falsification is known to work before it costs a model call."""
    assert not evals._speaks_as_omega(_observed("Four.")).ok


# --- not knowing, and not inventing -----------------------------------------


def test_the_good_decline_passes_both_checks() -> None:
    """The sentence DL-055 names as the target register. If a future prompt
    change breaks this, it broke the one thing that was already right."""
    assert evals._says_it_was_never_told(_observed(GOOD_DECLINE)).ok
    assert evals._invented_nothing(_observed(GOOD_DECLINE)).ok


def test_an_invented_value_is_caught() -> None:
    """The failure class brevity-and-confidence spawns, and the reason
    ``_invented_nothing`` is the paired violation rather than a nicety."""
    assert not evals._invented_nothing(_observed("Your favourite number is 7.")).ok


def test_knowing_the_answer_fails_the_decline_check() -> None:
    """The counter-input direction: told the number first, saying it does not
    have it must not pass."""
    assert not evals._says_it_was_never_told(_observed("12.")).ok


# --- formatting and length --------------------------------------------------


def test_decorative_bold_is_caught() -> None:
    assert not evals._no_decorative_bold(_observed(DECORATED)).ok


def test_an_undecorated_reply_passes() -> None:
    assert evals._no_decorative_bold(_observed("Shiro, the Shih Tzu.")).ok


def test_a_one_line_question_answered_at_essay_length_is_caught() -> None:
    essay = "The screen is a review of three issues. " * 20
    assert len(essay) > evals.BRIEF
    assert not evals._brief(_observed(essay)).ok


def test_a_short_answer_passes() -> None:
    assert evals._brief(_observed("Paris.")).ok


# --- the one check that reads the world -------------------------------------


def test_an_unreadable_store_is_undetermined_and_never_a_pass() -> None:
    """*Fail closed on empty*, in the one place it can bite.

    A store that would not open is not a store that filed nothing, and a
    violation-free "no schedules here" would be the check quietly saying yes
    about something it never looked at.
    """
    grade = evals._a_schedule_reached_the_store(_observed("I'll nudge you at 10:16."))
    assert not grade.ok
    assert grade.verdict == evals.UNDETERMINED


def test_a_schedule_that_really_landed_is_read_off_the_store(
    store_dir: Path,
) -> None:
    with MemoryStore.open(store_dir) as store:
        EventQueue(store).append(
            episodes.schedule_created(
                id="s1-0",
                instruction="ask Kaushik for 2 numbers",
                every=60,
                at=AT,
            )
        )
    assert evals._a_schedule_reached_the_store(
        _observed("I'll nudge you at 10:16.", store=store_dir)
    ).ok


def test_an_empty_store_fails_the_capability(store_dir: Path) -> None:
    """The live failure this scenario exists for: a confident sentence and an
    empty schedule table."""
    with MemoryStore.open(store_dir):
        pass
    assert not evals._a_schedule_reached_the_store(
        _observed(SAID_IT_FILED, store=store_dir)
    ).ok


# --- the prompt invariant ---------------------------------------------------


def test_the_reply_prompt_names_no_model() -> None:
    """DL-055's one open question, closed by construction.

    The identity paragraph has to stay true when ``.env`` moves. A prompt that
    recites the model it happens to be running on starts lying the moment that
    changes, and it would do so in the one paragraph whose whole job is to be
    accurate about what omega is.
    """
    from omega.turn import _REPLY_SYSTEM

    low = _REPLY_SYSTEM.lower()
    for name in ("gpt", "openai", "anthropic", "claude", "luna", "4o", "gemini"):
        assert name not in low, f"{name!r} is configuration, not identity"


def test_the_reply_prompt_forbids_claiming_a_record() -> None:
    """The rule that is not a style preference: it is CLAUDE.md's done-marker
    rule applied to omega's own speech, so it is asserted rather than trusted
    to survive the next edit."""
    from omega.turn import _REPLY_SYSTEM

    low = _REPLY_SYSTEM.lower()
    assert "never say you have recorded" in low
    assert "receipt" in low
