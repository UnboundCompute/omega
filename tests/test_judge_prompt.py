"""What the judge is actually asked — the shape, not the answer.

The bug this file exists to keep dead, observed on the live store rather than
imagined: typing ``hello`` into the tray failed roughly every other time with

    JudgeUndecided: judge answered 'Hello! How can I help?';
    expected one of ['act', 'act_then_speak', 'nothing', 'reply', 'silent', …]

and that string was exactly the reply the *previous* turn had produced. The
judge was not ignoring its instructions at random. Rendering the prompt for the
failing seq showed a user turn of four thousand characters ending

    you: hello
    omega: Hello! How can I help?

    New event:
    you: hello

— a few-shot demonstration of answering the greeting, with the only instruction
not to do that sitting in a system message far above it. At temperature 0 the
likeliest continuation of that text is the greeting, so the model completed the
conversation instead of classifying it. The alternation was the tell: after a
failed turn the transcript ends ``omega: (turn failed: …)``, which is no
pattern to copy, and the judge answered correctly.

So the invariant is about *position*: the demand for a verdict has to survive
being placed after a transcript that argues against it. These cases assert on
the prompt, because the prompt is where the defect was — every offline test was
green while this was broken, and `FakeProvider` cannot be wrong about a
judgement the way a real model can.

The capability here is "the demand lands last". Its violation metric, which
must not regress, is the last case in the file: the parser stays strict. The
tempting cheap fix for a `JudgeUndecided` is to scan the answer for a verdict
word, and that fix reads "not silent" as silence.

No network and no key.
"""

from __future__ import annotations

import pytest

from omega import episodes, provider
from omega.memory import MemoryStore
from omega.queue import EventQueue
from omega.turn import (
    ActResult,
    JudgeUndecided,
    TurnContext,
    _judge_messages,
    _JUDGE_SUFFIX,
    _reply_messages,
    parse_verdict,
    recall,
    RECALL_N,
)

AT = "2026-09-23T12:00:00+00:00"

#: The pair the live judge copied instead of classifying.
GREETING = "hello"
PARROT = "Hello! How can I help?"


@pytest.fixture
def q(store: MemoryStore) -> EventQueue:
    return EventQueue(store)


def _exchange(q: EventQueue, said: str, replied: str) -> None:
    """One completed turn in the transcript, the way a real one lands."""
    seq = q.append(episodes.inbound(said, channel="tray", at=AT))
    q.claim(seq)
    q.append(
        episodes.completed(
            for_seq=seq, outcome="spoke", reply=replied, tools=[], at=AT
        )
    )


def _ctx_after(q: EventQueue, text: str = GREETING) -> TurnContext:
    """A context for ``text`` that recalls everything already in the log."""
    seq = q.append(episodes.inbound(text, channel="tray", at=AT))
    q.claim(seq)
    pending = q.at(seq)
    return TurnContext(
        seq=pending.seq,
        event=pending.payload,
        recalled=recall(q, n=RECALL_N, before=pending.seq),
        queue=q,
        complete=lambda *a, **k: None,  # never called; the prompt is the subject
    )


def _user_text(messages: list[provider.Message]) -> str:
    content = messages[-1]["content"]
    if isinstance(content, str):
        return content
    return "\n".join(p["text"] for p in content if p.get("type") == "input_text")


# --- the position invariant -------------------------------------------------


def test_the_verdict_demand_is_the_last_thing_the_judge_reads(q: EventQueue) -> None:
    """The fix, stated as the thing that was false before it.

    Not "the prompt contains an instruction" — it did, and the model lost it
    behind the transcript. The claim is that nothing comes after it.
    """
    _exchange(q, GREETING, PARROT)

    text = _user_text(_judge_messages(_ctx_after(q)))

    assert text.endswith(_JUDGE_SUFFIX.rstrip("\n")), text[-200:]


def test_the_live_failure_shape_still_ends_with_the_demand(q: EventQueue) -> None:
    """The exact transcript that broke it, rebuilt.

    The greeting and its parroted reply are both present and in order, so the
    few-shot pressure is really there — this is not a test that passes because
    the transcript is empty.
    """
    _exchange(q, GREETING, PARROT)

    text = _user_text(_judge_messages(_ctx_after(q)))

    assert f"you: {GREETING}" in text
    assert f"omega: {PARROT}" in text
    assert text.index(PARROT) < text.index("Do not answer that message")


def test_the_demand_forbids_the_specific_mistake_not_just_the_format(q: EventQueue) -> None:
    """"Answer with one word" was already said, and lost. What the transcript
    argues for is *replying*, so the correction has to name replying."""
    assert "Do not answer that message" in _JUDGE_SUFFIX
    assert "SPEAK" in _JUDGE_SUFFIX and "ACT" in _JUDGE_SUFFIX and "SILENT" in _JUDGE_SUFFIX


def test_a_long_transcript_does_not_push_the_demand_out(q: EventQueue) -> None:
    """Recall is elided to a budget. The demand sits after the elision, so a
    busy day must not be the thing that removes it."""
    for i in range(RECALL_N):
        _exchange(q, f"message {i}", f"reply {i} " + "padding " * 40)

    text = _user_text(_judge_messages(_ctx_after(q)))

    assert text.endswith(_JUDGE_SUFFIX.rstrip("\n"))


def test_an_empty_log_still_gets_the_demand(q: EventQueue) -> None:
    """The first message omega ever sees. Nothing to copy, but the shape of the
    ask must not depend on that."""
    text = _user_text(_judge_messages(_ctx_after(q)))

    assert text.endswith(_JUDGE_SUFFIX.rstrip("\n"))


# --- what must not pick it up -----------------------------------------------


def test_the_reply_role_is_not_told_to_route(q: EventQueue) -> None:
    """The control, and the reason the suffix is not simply appended in
    ``_event_turn``. ``reply`` exists to answer the message; telling it not to
    would break the turn the judge just approved."""
    _exchange(q, GREETING, PARROT)

    text = _user_text(_reply_messages(_ctx_after(q), ActResult(tools=())))

    assert "Do not answer that message" not in text
    assert _JUDGE_SUFFIX.rstrip("\n") not in text


# --- the violation metric: the parser stays strict --------------------------


def test_a_conversational_answer_is_still_undecided() -> None:
    """The cheap fix that must never land.

    If a future change makes this pass, the judge has started guessing: the
    answer below contains no verdict, and reading one out of it would mean
    reading one out of anything.
    """
    with pytest.raises(JudgeUndecided):
        parse_verdict(PARROT)


def test_a_negated_verdict_is_not_read_as_that_verdict() -> None:
    """Why scanning for a keyword is not available as a repair. "not silent"
    contains "silent" and means its opposite."""
    with pytest.raises(JudgeUndecided):
        parse_verdict("not silent")


def test_the_verdict_words_still_parse() -> None:
    """The floor. Strictness is only defensible while the real answers work."""
    assert parse_verdict("SPEAK").speaks
    assert parse_verdict("ACT").speaks
    assert not parse_verdict("SILENT").speaks
