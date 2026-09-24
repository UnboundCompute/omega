"""The reflection pass — DL-054, and the half of learning DL-043 #2 deferred.

Everything in ``test_learn.py`` is omega being *told*. This is omega noticing.
The pair `CLAUDE.md` asks for is sharper here than anywhere else in the project,
because the capability and the violation are the same mechanism pointed at two
different windows:

    capability  a stretch that shows a pattern files a claim nobody typed
    violation   a stretch that shows nothing files nothing at all

The second one is the named failure of DL-054 — the memory firehose — and it is
the case to protect. A pass that files something from every window would still
score well on the first test and would make omega's learned set unreadable
within a day. `test_a_window_that_shows_nothing_files_nothing` is that guard;
it must not be weakened into "files few things".

The cursor cases are the other half, and they exist because the failure mode of
getting them wrong is a bill rather than a wrong answer: a pass that re-read its
window would call the model on every idle tick forever.
"""

from __future__ import annotations

import json

import pytest

from omega import derive, episodes, executor, learn, provider
from omega.executor import Executor
from omega.memory import MemoryStore
from omega.queue import EventQueue

from tests.teaching import _a_claim, _answer, _log, _Teaching

AT = "2026-09-23T12:00:00+00:00"


@pytest.fixture
def q(store: MemoryStore) -> EventQueue:
    return EventQueue(store)


def _chatter(q: EventQueue, n: int) -> None:
    """``n`` ordinary inbound messages. Ordinary is the point — a reflection
    pass is triggered by *volume of conversation*, not by anything about what
    was said, so the trigger has to be reachable without saying anything
    special."""
    for i in range(n):
        q.append(episodes.inbound(f"message {i}", channel="tray", at=AT))


def _reflections(q: EventQueue) -> list[tuple[int, dict]]:
    return [
        (seq, p) for seq, p in _log(q) if p.get("kind") == episodes.REFLECTION_DONE
    ]


def _claims(q: EventQueue) -> list[tuple[int, dict]]:
    return [
        (seq, p) for seq, p in _log(q) if p.get("kind") == episodes.CLAIM_EXTRACTED
    ]


def _run(q: EventQueue, teaching: _Teaching, **kw) -> Executor:
    ex = Executor(q, complete=teaching.complete)
    ex.recover()
    ex.drain(**kw)
    return ex


def _learn_calls(teaching: _Teaching) -> int:
    """How many times the `learn` role was asked. Counted on the provider
    rather than inferred from the log, because "nothing was filed" is also
    what a pass that ran and found nothing looks like — and several cases
    here are about cost, not output."""
    return sum(1 for role, _ in teaching.calls if role == provider.LEARN)


def _quiet(learn_answer) -> _Teaching:
    """SILENT so the turns themselves cost one call and say nothing. What a
    reflection reads is the conversation, not the replies."""
    return _Teaching(learn_answer=learn_answer, verdict="SILENT")


# --- the trigger ------------------------------------------------------------


def test_a_short_stretch_does_not_reflect_at_all(q: EventQueue) -> None:
    """Under the threshold, the `learn` role is never called.

    Asserted on the provider rather than on the log, because "no claims were
    filed" would also pass if a pass ran and found nothing — and the thing
    being fixed here is cost, not output.
    """
    _chatter(q, executor.REFLECT_EVERY - 1)
    teaching = _quiet(_answer(_a_claim()))

    _run(q, teaching)

    assert _learn_calls(teaching) == 0
    assert _reflections(q) == []
    assert _claims(q) == []


def test_a_long_enough_stretch_reflects_once_when_the_queue_goes_quiet(
    q: EventQueue,
) -> None:
    _chatter(q, executor.REFLECT_EVERY)
    teaching = _quiet(_answer(_a_claim("they work in the evening")))

    _run(q, teaching)

    assert _learn_calls(teaching) == 1
    done = _reflections(q)
    assert len(done) == 1, "a quiet drain reflects exactly once"
    assert done[0][1]["filed"] == 1
    assert done[0][1]["reason"] is None


def test_a_bounded_drain_does_not_reflect(q: EventQueue) -> None:
    """``max_turns`` returns with work still waiting.

    Reflecting there would read a window whose tail has not been handled and
    then move the cursor past it, so the episodes that arrived last would be
    the ones no pass ever looked at.
    """
    _chatter(q, executor.REFLECT_EVERY + 5)
    teaching = _quiet(_answer(_a_claim()))

    ex = Executor(q, complete=teaching.complete)
    ex.recover()
    ex.drain(max_turns=3)

    assert _learn_calls(teaching) == 0
    assert _reflections(q) == []


def test_the_pass_does_not_run_again_over_a_window_it_already_read(
    q: EventQueue,
) -> None:
    """The cursor is the whole point of writing `reflection.done` down.

    A second drain finds the queue quiet again immediately. Without a cursor
    derived from the log, that is a model call every time anything wakes the
    executor, forever.
    """
    _chatter(q, executor.REFLECT_EVERY)
    teaching = _quiet(_answer(_a_claim()))

    ex = _run(q, teaching)
    assert _learn_calls(teaching) == 1

    ex.drain()
    ex.drain()

    assert _learn_calls(teaching) == 1
    assert len(_reflections(q)) == 1


def test_the_cursor_survives_a_restart(q: EventQueue) -> None:
    """A fresh executor over the same log reaches the same conclusion.

    An in-memory counter would reset here and re-file the same conclusions
    after every crash, which is how a derived view becomes wrong (DL-017 —
    the log is the only state).
    """
    _chatter(q, executor.REFLECT_EVERY)
    first = _quiet(_answer(_a_claim()))
    _run(q, first)

    second = _quiet(_answer(_a_claim()))
    _run(q, second)

    assert _learn_calls(second) == 0
    assert len(_reflections(q)) == 1


def test_enough_new_conversation_reflects_a_second_time(q: EventQueue) -> None:
    """The counter resets rather than the threshold rising: the cursor is
    "how much has happened since", so omega keeps noticing for as long as it
    keeps being used."""
    _chatter(q, executor.REFLECT_EVERY)
    teaching = _quiet(_answer(_a_claim("first")))
    ex = _run(q, teaching)

    _chatter(q, executor.REFLECT_EVERY)
    ex.drain()

    assert _learn_calls(teaching) == 2
    done = _reflections(q)
    assert len(done) == 2
    assert done[0][1]["through"] < done[1][1]["through"]


# --- the violation ----------------------------------------------------------


def test_a_window_that_shows_nothing_files_nothing(q: EventQueue) -> None:
    """The memory firehose guard (DL-054), and the case that matters most.

    Most stretches of talking to someone reveal nothing durable about them.
    An empty list has to be an ordinary, cheap answer that leaves no trace in
    the learned set — if this ever becomes "files one or two", the feature is
    worse than not having it.
    """
    _chatter(q, executor.REFLECT_EVERY)
    teaching = _quiet(_answer())

    _run(q, teaching)

    assert _learn_calls(teaching) == 1, "the pass really ran"
    assert _claims(q) == [], "nothing unremarkable was remembered"
    done = _reflections(q)
    assert len(done) == 1 and done[0][1]["filed"] == 0


def test_a_pass_cannot_file_more_than_the_cap(q: EventQueue) -> None:
    """Over the cap is *refused*, not truncated.

    Truncating would keep an arbitrary three of a set the model clearly did
    not mean as considered conclusions, and it would hide the fact that the
    prompt is being read as an invitation to list everything.
    """
    teaching = _quiet(
        _answer(*[_a_claim(f"claim {i}") for i in range(learn.MAX_INFERRED_PER_PASS + 1)])
    )
    _chatter(q, executor.REFLECT_EVERY)

    _run(q, teaching)

    assert _claims(q) == []
    reason = _reflections(q)[0][1]["reason"]
    assert reason and "over the limit" in reason


def test_a_reflection_cannot_create_a_schedule(q: EventQueue) -> None:
    """The asymmetry DL-054 rests on: a wrong inferred claim is a bad sentence
    in a prompt, a wrong inferred schedule is a notification every morning
    forever. So the pass returns claims and the schedule keys are ignored even
    when the model volunteers them."""
    _chatter(q, executor.REFLECT_EVERY)
    answer = json.dumps(
        {
            "claims": [],
            "schedules": [
                {"phrase": "every morning at 9", "intent": "brief me", "hour": 9}
            ],
            "retract": [1],
        }
    )
    teaching = _quiet(answer)

    _run(q, teaching)

    kinds = {p.get("kind") for _, p in _log(q)}
    assert episodes.SCHEDULE_CREATED not in kinds
    assert episodes.CLAIM_RETRACTED not in kinds
    assert _reflections(q)[0][1]["filed"] == 0


# --- what a filed inference is ----------------------------------------------


def test_an_inferred_claim_is_filed_as_not_explicit(q: EventQueue) -> None:
    """DL-042 gave `explicit` one job — whether a contradiction is worth
    interrupting for — and this is the only path that files false. A
    conclusion omega drew on its own is not grounds for stopping someone to
    argue about it."""
    _chatter(q, executor.REFLECT_EVERY)
    teaching = _quiet(_answer(_a_claim("they batch errands on Saturdays")))

    _run(q, teaching)

    filed = _claims(q)
    assert len(filed) == 1
    assert filed[0][1]["explicit"] is False
    assert filed[0][1]["text"] == "they batch errands on Saturdays"


def test_an_inferred_claim_fires_on_a_later_turn(q: EventQueue) -> None:
    """The capability, graded on the world: the claim is in the derived view
    and reaches the prompt of a turn that happens afterwards. Filing something
    nothing ever reads would pass every test above and change nothing."""
    _chatter(q, executor.REFLECT_EVERY)
    teaching = _quiet(_answer(_a_claim("they prefer short answers")))
    ex = _run(q, teaching)

    learned = derive.Learned.fold(_log(q))
    fires = learned.matching(text="anything", channel="tray", at=AT)
    assert [c.text for c in fires] == ["they prefer short answers"]
    assert fires[0].explicit is False

    before = len(teaching.prompts)
    q.append(episodes.inbound("what now", channel="tray", at=AT))
    ex.drain()

    later = "\n".join(teaching.prompts[before:])
    assert "they prefer short answers" in later


def test_the_pass_points_its_claims_at_the_stretch_it_read(q: EventQueue) -> None:
    """There is no turn, so ``for_seq`` names the episode the pass read up to.

    Pointing it at some earlier turn would invent a cause for a conclusion
    that came from the whole window, and `--learned` would then attribute the
    belief to one message that may not have contributed to it.
    """
    _chatter(q, executor.REFLECT_EVERY)
    teaching = _quiet(_answer(_a_claim()))

    _run(q, teaching)

    seq, claim = _claims(q)[0]
    through = _reflections(q)[0][1]["through"]
    assert claim["for_seq"] == claim["source_seq"] == through
    assert through < seq, "the cursor names what was read, not what was written"


# --- when it breaks ---------------------------------------------------------


def test_a_failed_pass_never_takes_down_the_drain(q: EventQueue) -> None:
    """A broken `learn` role must not stop every future turn in the process.

    This is `turn._teach`'s reason carried one step further: there the risk was
    losing a composed reply, here it is the executor thread itself, and there
    is no reply to put a receipt in — so the record is the only place the
    failure can be seen.
    """
    def boom(role, messages):
        raise RuntimeError("400: temperature is not supported")

    _chatter(q, executor.REFLECT_EVERY)
    teaching = _quiet(boom)

    results = None
    ex = Executor(q, complete=teaching.complete)
    ex.recover()
    results = ex.drain()

    assert len(results) == executor.REFLECT_EVERY, "every turn still ran"
    done = _reflections(q)
    assert len(done) == 1
    assert "temperature" in done[0][1]["reason"]
    assert done[0][1]["filed"] == 0


def test_a_failed_pass_still_moves_the_cursor(q: EventQueue) -> None:
    """Otherwise a fault becomes a bill.

    A pass that failed and left the cursor behind would re-read the same
    window on every idle tick for as long as the outage lasted. The window is
    bounded and conversation keeps arriving, so the next pass sees the recent
    part of what this one lost.
    """
    calls: list[int] = []

    def boom(role, messages):
        calls.append(1)
        raise RuntimeError("no")

    _chatter(q, executor.REFLECT_EVERY)
    teaching = _quiet(boom)
    ex = _run(q, teaching)

    ex.drain()
    ex.drain()

    assert len(calls) == 1


def test_a_failed_pass_is_visible_to_the_learned_view(q: EventQueue) -> None:
    """DL-053 said a failed extraction is a record and not only a receipt. A
    failed *reflection* has no receipt at all, so the record is the only
    thing standing between a dead `learn` role and silence."""
    def boom(role, messages):
        raise RuntimeError("the model said no")

    _chatter(q, executor.REFLECT_EVERY)
    _run(q, _quiet(boom))

    learned = derive.Learned.fold(_log(q))
    assert learned.failed == 1
    assert "the model said no" in learned.last_failure

    said = learn.review(
        [], failed=learned.failed, last_failure=learned.last_failure
    )
    assert "not been taught anything" not in said


def test_an_unparseable_answer_is_a_failure_not_a_silent_nothing(
    q: EventQueue,
) -> None:
    """The distinction that hid the DL-052 outage: "found nothing" and "could
    not read the answer" must not render the same way. One is the expected
    case, the other is a broken role."""
    _chatter(q, executor.REFLECT_EVERY)

    _run(q, _quiet("I have reflected on our conversation."))

    done = _reflections(q)[0][1]
    assert done["reason"] is not None
    assert done["filed"] == 0


def test_reflect_before_recovery_does_nothing(q: EventQueue) -> None:
    """§1.2 — nothing may write before the startup report has read the log.
    :meth:`Executor.reflect` is public for the live evals, so it has to hold
    the same line ``drain`` does rather than trusting its caller."""
    _chatter(q, executor.REFLECT_EVERY)
    teaching = _quiet(_answer(_a_claim()))
    ex = Executor(q, complete=teaching.complete)

    assert ex.reflect() is False
    assert _learn_calls(teaching) == 0
    assert _reflections(q) == []
