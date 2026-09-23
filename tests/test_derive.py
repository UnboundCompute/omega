"""What's open, derived from the log (DL-041).

The cases that matter here are not "does a fold fold". They are the two that
would let omega quietly get this wrong in production: discharging on the seq an
answer actually names, and keeping a block visible after it has aged out of
recall.

The second half of the file is the one that decides whether any of this is
real. A correct view nothing reads is a folder, so those cases assert against
the text that actually went to a model — not against the view object, which
was already right before it was wired to anything.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omega import derive, episodes, provider, queue, turn
from omega.executor import Executor
from omega.memory import MemoryStore
from omega.queue import EventQueue


def _blocked(*, for_seq: int, needs: str = "may I fetch that url?"):
    return episodes.blocked(for_seq=for_seq, needs=needs)


def _answer(*, resumes_seq: int, text: str = "yes, go ahead"):
    return episodes.inbound(text, channel="tray", resumes_seq=resumes_seq)


def _typed(text: str = "hello"):
    return episodes.inbound(text, channel="tray")


# --- the fold ------------------------------------------------------------


def test_a_block_opens_an_obligation():
    view = derive.OpenWork.fold([(7, _blocked(for_seq=6, needs="approve the write?"))])
    assert len(view) == 1
    (block,) = view.blocks()
    assert (block.seq, block.for_seq, block.needs) == (7, 6, "approve the write?")


def test_an_answer_discharges_the_block_it_names():
    view = derive.OpenWork.fold(
        [(7, _blocked(for_seq=6)), (8, _answer(resumes_seq=6))]
    )
    assert view.blocks() == [], "an answered question is not still open"


def test_the_key_is_the_event_seq_because_that_is_what_the_tray_sends():
    """The wire contract, pinned.

    ``TrayViewModel.swift:405`` sets the pending resume from ``update.forSeq``,
    so an answer names the *event* whose turn blocked, not the ``turn.blocked``
    episode. Keying the other way would pass a test written from
    ``episodes.inbound``'s prose and then never discharge anything in
    production. This case fails loudly if the key is ever flipped back.
    """
    blocking_seq, event_seq = 7, 6
    naming_the_event = derive.OpenWork.fold(
        [(blocking_seq, _blocked(for_seq=event_seq)), (8, _answer(resumes_seq=event_seq))]
    )
    naming_the_block = derive.OpenWork.fold(
        [(blocking_seq, _blocked(for_seq=event_seq)), (8, _answer(resumes_seq=blocking_seq))]
    )
    assert naming_the_event.blocks() == []
    assert len(naming_the_block) == 1, (
        "resumes_seq names the event, so naming the blocking episode must not "
        "discharge -- if this starts passing, the wire contract moved"
    )


def test_only_the_answered_block_is_discharged():
    view = derive.OpenWork.fold(
        [
            (2, _blocked(for_seq=1, needs="first")),
            (4, _blocked(for_seq=3, needs="second")),
            (5, _answer(resumes_seq=1)),
        ]
    )
    assert [b.needs for b in view.blocks()] == ["second"]


def test_an_answer_that_itself_blocks_opens_a_new_obligation():
    """A different question, not the same one still open."""
    view = derive.OpenWork.fold(
        [
            (2, _blocked(for_seq=1, needs="first")),
            (3, _answer(resumes_seq=1)),
            (4, _blocked(for_seq=3, needs="second")),
        ]
    )
    assert [(b.for_seq, b.needs) for b in view.blocks()] == [(3, "second")]


def test_an_answer_naming_nothing_open_is_not_an_error():
    """It can arrive legitimately; a derived view does not adjudicate input."""
    view = derive.OpenWork.fold([(3, _answer(resumes_seq=999))])
    assert view.blocks() == []


def test_an_ordinary_message_neither_opens_nor_closes_anything():
    view = derive.OpenWork.fold([(2, _blocked(for_seq=1)), (3, _typed())])
    assert len(view) == 1, "a message that answers nothing discharges nothing"


def test_blocks_come_back_oldest_first():
    view = derive.OpenWork.fold(
        [
            (2, _blocked(for_seq=1, needs="first")),
            (4, _blocked(for_seq=3, needs="second")),
            (6, _blocked(for_seq=5, needs="third")),
        ]
    )
    assert [b.needs for b in view.blocks()] == ["first", "second", "third"]


def test_out_of_order_application_is_refused():
    view = derive.OpenWork()
    view.apply(5, _typed())
    with pytest.raises(ValueError, match="in order"):
        view.apply(4, _typed())


def test_through_separates_rebuilt_and_empty_from_never_rebuilt():
    """Fail closed on empty: both answer [] and they are not the same state."""
    never = derive.OpenWork()
    rebuilt = derive.OpenWork.fold([(3, _typed())])
    assert never.blocks() == rebuilt.blocks() == []
    assert never.through == 0
    assert rebuilt.through == 3


# --- against a real store ------------------------------------------------


def test_rebuild_reads_the_whole_log(store):
    event = store.append_episode(episodes.encode(_typed("do the thing")))
    blocked_at = store.append_episode(
        episodes.encode(_blocked(for_seq=event, needs="approve?"))
    )

    view = derive.OpenWork.rebuild(store)

    assert view.through == blocked_at
    assert [(b.for_seq, b.needs) for b in view.blocks()] == [(event, "approve?")]


def test_rebuild_after_the_answer_finds_nothing_open(store):
    event = store.append_episode(episodes.encode(_typed("do the thing")))
    store.append_episode(episodes.encode(_blocked(for_seq=event)))
    store.append_episode(episodes.encode(_answer(resumes_seq=event)))

    assert derive.OpenWork.rebuild(store).blocks() == []


def test_a_block_survives_after_it_ages_out_of_recall(store):
    """The defect this view exists to close.

    ``recall`` is the last ``RECALL_N`` episodes, so a question omega stopped to
    ask falls out of its own context once enough traffic goes by. DL-035
    requires a block to "survive until they look" — after the clock landed,
    nobody need be watching when it is raised.
    """
    event = store.append_episode(episodes.encode(_typed("do the thing")))
    store.append_episode(episodes.encode(_blocked(for_seq=event, needs="approve?")))
    for i in range(turn.RECALL_N + 5):
        store.append_episode(episodes.encode(_typed(f"unrelated chatter {i}")))

    recalled = queue.EventQueue(store).recent(turn.RECALL_N)
    kinds = [p.kind for p in recalled]
    assert episodes.TURN_BLOCKED not in kinds, (
        "precondition: the block must have aged out, or this proves nothing"
    )

    view = derive.OpenWork.rebuild(store)
    assert [b.needs for b in view.blocks()] == ["approve?"], (
        "the obligation outlived the recall window, which is the whole point"
    )


def test_rebuilding_twice_gives_the_same_answer(store):
    """Derived, so it is a function of the log and nothing else (DL-041)."""
    event = store.append_episode(episodes.encode(_typed("do the thing")))
    store.append_episode(episodes.encode(_blocked(for_seq=event, needs="approve?")))

    first = derive.OpenWork.rebuild(store)
    second = derive.OpenWork.rebuild(store)

    assert first.blocks() == second.blocks()
    assert first.through == second.through


def test_advancing_folds_only_what_arrived(store):
    """The steady-state path: O(new), not O(log), on every turn."""
    event = store.append_episode(episodes.encode(_typed("do the thing")))
    view = derive.OpenWork.rebuild(store)
    assert view.blocks() == []

    blocked_at = store.append_episode(
        episodes.encode(_blocked(for_seq=event, needs="approve?"))
    )
    view.advance(store)

    assert view.through == blocked_at
    assert [b.needs for b in view.blocks()] == ["approve?"]


def test_advancing_with_nothing_new_changes_nothing(store):
    """Idempotent, so a caller may advance as often as it likes."""
    event = store.append_episode(episodes.encode(_typed("do the thing")))
    store.append_episode(episodes.encode(_blocked(for_seq=event, needs="approve?")))
    view = derive.OpenWork.rebuild(store)

    before = (view.through, view.blocks())
    view.advance(store)
    view.advance(store)

    assert (view.through, view.blocks()) == before


# --- reaching a prompt ---------------------------------------------------
#
# Everything above proves the view is right. These prove it is *read*, which is
# the whole difference between this and a folder.


class _Watched:
    """A provider that keeps what it was asked, so a test can read the prompt.

    Grading the world rather than the words, at the only boundary where the two
    can differ here: the assertion is over the bytes that went to the model, not
    over the object the turn built on the way there.
    """

    def __init__(self, verdict: str = "SILENT") -> None:
        self.prompts: list[str] = []
        self._fake = provider.FakeProvider(
            {
                provider.JUDGE: self._record(verdict),
                provider.ACT: self._record("ok"),
            }
        )

    def _record(self, answer: str):
        def respond(role, messages):
            self.prompts.append("\n".join(_text_of(m) for m in messages))
            return answer

        return respond

    @property
    def complete(self):
        return self._fake.complete

    def last(self) -> str:
        """The prompt for the final event drained.

        The last one rather than the first, because these cases set up a block
        and then send the event whose turn should see it — and the earlier turns
        are the ones that legitimately saw nothing open yet.
        """
        assert self.prompts, "nothing was asked, so there is no prompt to read"
        return self.prompts[-1]


def _text_of(message: dict) -> str:
    """The text of one provider message, as text.

    Never ``str(message)``: a dict's repr escapes newlines, so a prompt read
    that way matches a heading but not the line under it, and a case asserting
    over the section below the heading silently checks nothing. That is the
    shape of the "passes on empty" check `CLAUDE.md` bans, arrived at by
    accident in a test helper.
    """
    content = message["content"]
    if isinstance(content, str):
        return content
    return "\n".join(
        str(part.get("text", "")) for part in content if isinstance(part, dict)
    )


OPEN_HEADING = "You asked these and have had no answer yet"


def _drained(q: EventQueue, watched: _Watched) -> None:
    ex = Executor(q, complete=watched.complete)
    ex.recover()
    ex.drain()


def test_an_open_question_reaches_the_prompt(store):
    """The point of the whole increment (DL-034: filing alone is a folder)."""
    q = EventQueue(store)
    event = q.append(_typed("do the thing"))
    q.append(_blocked(for_seq=event, needs="may I overwrite notes.md?"))
    q.append(_typed("unrelated, days later"))

    watched = _Watched()
    _drained(q, watched)

    assert OPEN_HEADING in watched.last()
    assert "may I overwrite notes.md?" in watched.last()


def test_nothing_open_leaves_the_prompt_byte_for_byte_as_it_was(store):
    """Not an empty section — and not a stray blank line either.

    Two claims in one case. The first is the design one: a heading that is
    usually empty teaches the model to skip the section before the day it
    matters, so nothing is rendered at all.

    The second is why the eval baselines taken before this change are still
    comparable with ones taken after it. With nothing open the section
    contributes the empty string, so the transcript runs straight into the new
    event exactly as it did before — the prompt is unchanged, not merely
    similar. Asserting the junction rather than the absence of a heading is the
    difference between those two claims, and only the stronger one licenses
    comparing the numbers.
    """
    q = EventQueue(store)
    q.append(_typed("just a message"))

    watched = _Watched()
    _drained(q, watched)
    prompt = watched.last()

    assert OPEN_HEADING not in prompt
    assert "just a message\n\nNew event:" not in prompt, (
        "sanity: the new event is not also the last line of the transcript"
    )
    assert "(nothing yet)\n\nNew event:" in prompt, (
        "an empty section must contribute nothing, not a blank line"
    )


def test_an_answer_is_discharged_before_the_turn_carrying_it_runs(store):
    """The turn handling an answer must not still be asking the question.

    This is why the view advances *through* the claimed event rather than
    stopping short of it the way recall does.
    """
    q = EventQueue(store)
    event = q.append(_typed("do the thing"))
    q.append(_blocked(for_seq=event, needs="may I overwrite notes.md?"))
    q.append(_answer(resumes_seq=event))

    watched = _Watched()
    _drained(q, watched)

    assert OPEN_HEADING not in watched.last(), (
        "the question was answered by the very event this turn is about"
    )


def test_a_turn_is_not_shown_a_question_omega_had_not_asked_yet(store):
    """The fold stops at the claimed episode, not at the head of the log.

    On a backlog — a restart with events already queued — the head runs ahead of
    the turn being handled. Folding to it would put a *later* question into an
    *earlier* turn's prompt: wrong in the one direction nobody thinks to check,
    because a view that is ahead of its reader still looks populated and fresh.
    """
    q = EventQueue(store)
    first = q.append(_typed("do the thing"))
    q.append(_blocked(for_seq=first, needs="may I overwrite notes.md?"))
    q.append(_typed("and another thing"))

    watched = _Watched()
    _drained(q, watched)

    assert OPEN_HEADING not in watched.prompts[0], (
        "the turn for the first event ran before that question was ever asked"
    )
    assert OPEN_HEADING in watched.prompts[-1], (
        "precondition: the later turn must see it, or this proves nothing"
    )


def test_a_block_survives_a_restart_and_reaches_the_next_prompt(store_dir: Path):
    """The case the view exists for, end to end.

    A different process, a different Executor, nothing carried in memory across
    the gap — and the question still arrives in the prompt, because it was
    re-derived from the log rather than remembered.
    """
    with MemoryStore.open(store_dir) as first:
        q = EventQueue(first)
        event = q.append(_typed("do the thing"))
        q.append(_blocked(for_seq=event, needs="may I overwrite notes.md?"))
        Executor(q, complete=_Watched().complete).recover()

    with MemoryStore.open(store_dir) as second:
        q = EventQueue(second)
        q.append(_typed("morning"))
        watched = _Watched()
        _drained(q, watched)

    assert "may I overwrite notes.md?" in watched.last()


def test_a_block_reaches_the_prompt_after_it_has_aged_out_of_recall(store):
    """The defect, closed, measured where it bit: in the prompt.

    The precondition is asserted first — if the block were still inside the
    recall window the transcript would carry it anyway and this case would pass
    for the wrong reason.
    """
    q = EventQueue(store)
    event = q.append(_typed("do the thing"))
    q.append(_blocked(for_seq=event, needs="may I overwrite notes.md?"))
    for i in range(turn.RECALL_N + 5):
        q.append(_typed(f"unrelated chatter {i}"))

    recalled = q.recent(turn.RECALL_N, before=q.head())
    assert episodes.TURN_BLOCKED not in [p.kind for p in recalled], (
        "precondition: the block must have aged out, or this proves nothing"
    )

    watched = _Watched()
    _drained(q, watched)

    assert "may I overwrite notes.md?" in watched.last()


def test_the_open_section_is_bounded_and_says_what_it_dropped(store):
    """A count is not a budget (DL-039) — including for this section."""
    q = EventQueue(store)
    for i in range(40):
        event = q.append(_typed(f"thing {i}"))
        q.append(_blocked(for_seq=event, needs=f"question {i}: " + "x" * 400))
    q.append(_typed("and now this"))

    watched = _Watched()
    _drained(q, watched)
    prompt = watched.last()

    heading, _, rest = prompt.partition(OPEN_HEADING + ":\n")
    section = rest.split("\n\nNew event:")[0]
    assert len(section) <= turn.MAX_OPEN_CHARS + 200, "the section is not bounded"
    assert "not shown here" in section, "a silent drop is the bug, not the fix"
    assert "question 39" in section, "the newest open question must survive"


# --- what omega has been taught (DL-042) ---------------------------------


def _claim(*, text="keep replies short", trigger=None, source_seq=1, for_seq=1,
           explicit=True, supersedes=None, situation="taught while reviewing"):
    return episodes.claim_extracted(
        for_seq=for_seq,
        text=text,
        source_seq=source_seq,
        situation=situation,
        explicit=explicit,
        trigger=trigger,
        supersedes=supersedes,
    )


AT = "2026-09-24T12:00:00+00:00"


def _fires(view, text="anything at all", channel="tray", at=AT):
    return [c.text for c in view.matching(text=text, channel=channel, at=at)]


def test_a_claim_with_no_trigger_applies_to_every_turn():
    """Tone and working style have no situation because they apply to all of
    them. If always-active were not representable, the only way to say "be
    brief" would be to guess every word that might precede a long answer."""
    view = derive.Learned.fold([(1, _claim(text="be brief"))])
    assert _fires(view, text="what is 2+2") == ["be brief"]
    assert _fires(view, text="deploy the thing") == ["be brief"]


def test_a_phrase_trigger_fires_on_the_situation_it_names():
    """The capability half of DL-042's paired metric."""
    view = derive.Learned.fold(
        [(1, _claim(text="check the changelog", trigger={"any": ["deploy"]}))]
    )
    assert _fires(view, text="I'm about to deploy") == ["check the changelog"]


def test_a_phrase_trigger_does_not_fire_on_an_unrelated_event():
    """**The violation half, and the one that must not regress.** A learned
    habit firing on an event it has no business touching is how extraction
    starts changing turns silently -- the false-memory failure class DL-034
    named, observed at the moment it would do damage rather than at ingest."""
    view = derive.Learned.fold(
        [(1, _claim(text="check the changelog", trigger={"any": ["deploy"]}))]
    )
    assert _fires(view, text="what did I have for lunch") == []


def test_matching_is_case_insensitive_in_both_directions():
    """A person types "Deploy" at the start of a sentence. A claim that missed
    that would look broken in the most ordinary case there is."""
    view = derive.Learned.fold(
        [(1, _claim(text="c", trigger={"any": ["DePloY"]}))]
    )
    assert _fires(view, text="Deploying now") == ["c"]


def test_trigger_fields_are_anded_not_ored():
    """A person who names two conditions has narrowed, not widened. Or-ing them
    would make every extra condition make a claim fire *more*, which is the
    opposite of what adding a condition means."""
    view = derive.Learned.fold(
        [(1, _claim(text="c", trigger={"any": ["deploy"], "channel": "tray"}))]
    )
    assert _fires(view, text="deploy", channel="tray") == ["c"]
    assert _fires(view, text="deploy", channel="schedule") == []
    assert _fires(view, text="unrelated", channel="tray") == []


def _at_local_hour(hour: int) -> str:
    """A timestamp whose *local* hour is `hour`, whatever machine this runs on.

    The window is a statement about the clock on the wall, so a case that hard-
    coded UTC would pass in one timezone and fail in another -- and the failure
    would look like a matcher bug rather than a test bug. Built with the local
    offset attached, so the conversion `local_hour` performs is the identity.
    """
    from datetime import datetime as _dt

    local_tz = _dt.now().astimezone().tzinfo
    return _dt(2026, 9, 24, hour, 30, tzinfo=local_tz).isoformat()


def test_an_hour_window_can_wrap_midnight():
    """"At night" is a thing people say, and it crosses midnight. Without
    wrapping it needs two claims, and the second one is the one nobody writes."""
    view = derive.Learned.fold([(1, _claim(text="c", trigger={"hours": [22, 6]}))])
    assert _fires(view, at=_at_local_hour(23)) == ["c"]
    assert _fires(view, at=_at_local_hour(3)) == ["c"]
    assert _fires(view, at=_at_local_hour(12)) == []


def test_an_hour_window_is_read_on_the_wall_clock_not_in_utc():
    """The log writes UTC and the person means their own clock. Reading a naive
    timestamp as *local* would be the other plausible choice and is wrong: it
    would shift every window by the machine's offset, silently, and only on
    machines that are not already UTC."""
    noon_utc = "2026-09-24T12:00:00+00:00"
    from datetime import datetime as _dt, timezone as _tz

    expected = _dt(2026, 9, 24, 12, tzinfo=_tz.utc).astimezone().hour
    assert derive.local_hour(noon_utc) == expected
    assert derive.local_hour("2026-09-24T12:00:00") == expected, (
        "a naive timestamp must be read as UTC, which is what the log writes"
    )


def test_an_unreadable_timestamp_does_not_fire_an_hour_window():
    """Fail closed on empty (`CLAUDE.md`). An hour window that cannot be
    evaluated falling *open* would turn "only while I'm working" into
    "always" -- spurious activation caused by the check itself failing."""
    view = derive.Learned.fold([(1, _claim(text="c", trigger={"hours": [9, 17]}))])
    assert _fires(view, at="not a timestamp") == []
    assert derive.local_hour("not a timestamp") is None


def test_a_superseding_claim_replaces_the_one_it_names():
    view = derive.Learned.fold(
        [
            (1, _claim(text="prefers long answers")),
            (2, _claim(text="prefers short answers", source_seq=2, supersedes=1)),
        ]
    )
    assert [c.text for c in view.claims()] == ["prefers short answers"]


def test_supersession_never_removes_anything_from_the_log():
    """The property the whole append-only rule buys. Re-deriving from the log
    up to the point before the supersession must reach the earlier belief --
    which is what makes "destroy core memory" impossible by construction rather
    than by a model getting a criticality test right (DL-034)."""
    episodes_ = [
        (1, _claim(text="prefers long answers")),
        (2, _claim(text="prefers short answers", source_seq=2, supersedes=1)),
    ]
    assert [c.text for c in derive.Learned.fold(episodes_[:1]).claims()] == [
        "prefers long answers"
    ]


def test_superseding_something_already_gone_is_not_an_error():
    """An ordinary race in an append-only log, not a corruption: the target may
    itself have been superseded. A derived view is not the place to adjudicate
    that."""
    view = derive.Learned.fold(
        [(5, _claim(text="c", source_seq=5, supersedes=1))]
    )
    assert [c.text for c in view.claims()] == ["c"]


def test_a_claim_learned_later_does_not_apply_to_an_earlier_turn(store):
    """The `upto` bound, which is not an optimisation. Folding to the head of
    the log would apply a claim to a turn that happened *before* it was learned
    -- and unlike a stale view, that looks entirely correct from outside."""
    q = EventQueue(store)
    first = q.append(_typed("before"))
    q.append(_claim(text="taught after the fact", for_seq=first, source_seq=first))
    later = q.append(_typed("after"))

    early = derive.Learned().advance(store, upto=first)
    assert early.claims() == [], "a turn was shown a claim omega had not learned yet"
    assert derive.Learned().advance(store, upto=later).claims()

    assert early.through == first


def test_the_view_distinguishes_nothing_learned_from_never_rebuilt():
    """A view answering the empty list for both would be a check that passes on
    empty."""
    assert derive.Learned().through == 0
    assert derive.Learned.fold([(3, _typed("unrelated"))]).through == 3


def test_explicit_is_carried_because_it_decides_escalation(store):
    """DL-042's answer to "what is core memory": not a class of claim, but
    whether the person authored it deliberately. It is a fact about how the
    claim arrived, so it is only available while it is arriving."""
    view = derive.Learned.fold(
        [
            (1, _claim(text="told", explicit=True)),
            (2, _claim(text="inferred", source_seq=2, explicit=False)),
        ]
    )
    by_text = {c.text: c.explicit for c in view.claims()}
    assert by_text == {"told": True, "inferred": False}


def test_claims_survive_a_restart_because_they_are_derived_from_the_log(store_dir):
    """The reason a claim is an episode at all (DL-017). Learned behaviour that
    did not survive a restart would be the one kind of memory whose absence is
    invisible: omega would simply stop doing something, with nothing to read."""
    with MemoryStore.open(store_dir) as store:
        q = EventQueue(store)
        event = q.append(_typed("teach me"))
        q.append(_claim(text="always check the changelog", for_seq=event,
                        source_seq=event))

    with MemoryStore.open(store_dir) as store:
        view = derive.Learned.rebuild(store)
        assert [c.text for c in view.claims()] == ["always check the changelog"]


# --- a learned claim reaching a prompt (DL-042) --------------------------

LEARNED_HEADING = "What you have learned about working with this person"


def test_a_learned_claim_reaches_the_prompt(store):
    """"Filing alone is a folder" (DL-034), applied to this half. The view was
    already correct before anything read it; this is the case that says a
    person teaching omega something changes what the model is given."""
    q = EventQueue(store)
    event = q.append(_typed("remember this"))
    q.append(_claim(text="always check the changelog first", for_seq=event,
                    source_seq=event))
    q.append(_typed("ordinary message"))

    watched = _Watched()
    _drained(q, watched)
    prompt = watched.last()
    assert LEARNED_HEADING in prompt
    assert "always check the changelog first" in prompt


def test_a_claim_that_does_not_fire_never_reaches_the_prompt(store):
    """**The violation metric, measured where it does damage.** A claim that
    leaks into a prompt it does not apply to is a learned habit changing turns
    it has no business touching -- and unlike a wrong answer, nobody sees it
    happen."""
    q = EventQueue(store)
    event = q.append(_typed("remember this"))
    q.append(_claim(text="check the changelog", for_seq=event, source_seq=event,
                    trigger={"any": ["deploy"]}))
    q.append(_typed("what did I have for lunch"))

    watched = _Watched()
    _drained(q, watched)
    assert LEARNED_HEADING not in watched.last()
    assert "changelog" not in watched.last()


def test_nothing_learned_leaves_the_prompt_byte_for_byte_as_it_was(store):
    """The junction, not merely the absence of a heading. Only the stronger
    claim licenses comparing eval numbers taken before this commit with ones
    taken after it -- a person who has taught omega nothing gets exactly the
    prompt they got yesterday."""
    q = EventQueue(store)
    q.append(_typed("hello"))

    watched = _Watched()
    _drained(q, watched)
    with_view = watched.last()

    # The fact that licenses the comparison, stated exactly: the section
    # contributes the *empty string*, not a heading and not a blank line. A
    # blank line is a changed prompt, and "no heading" would not have caught it.
    assert turn._learned_section(()) == ""
    assert turn._learned_section([_a_claim_object()]) != ""

    # And the prompt runs straight from the system message into the history.
    assert LEARNED_HEADING not in with_view
    assert with_view.count("Recent history:") == 1
    assert "\n\nRecent history:" not in with_view.split("Recent history:")[0] + (
        "Recent history:"
    )


def _a_claim_object():
    """A `Claim` built directly, to check the non-empty side of the section
    without routing through a store."""
    return derive.Claim(
        seq=1, text="be brief", trigger=None, situation="s", source_seq=1,
        explicit=True,
    )


def test_a_claim_learned_later_does_not_reach_an_earlier_turn(store):
    """The `upto` bound at the prompt, where it is observable. An over-eager
    question looks wrong; an over-eager habit looks like omega simply behaving
    that way."""
    q = EventQueue(store)
    first = q.append(_typed("before the lesson"))
    q.append(_claim(text="be extremely terse", for_seq=first, source_seq=first))
    q.append(_typed("after the lesson"))

    watched = _Watched()
    _drained(q, watched)
    assert LEARNED_HEADING not in watched.prompts[0], (
        "a turn was shown a claim omega had not been taught yet"
    )
    assert "be extremely terse" in watched.last()


def test_a_claim_survives_a_restart_and_reaches_the_next_prompt(store_dir):
    """Learned behaviour that did not survive a restart is the one kind of
    memory whose absence is invisible: omega just stops doing something."""
    with MemoryStore.open(store_dir) as store:
        q = EventQueue(store)
        event = q.append(_typed("teach"))
        q.append(_claim(text="sign off as omega", for_seq=event, source_seq=event))
        first = _Watched()
        _drained(q, first)

    with MemoryStore.open(store_dir) as store:
        q = EventQueue(store)
        q.append(_typed("after the restart"))
        second = _Watched()
        _drained(q, second)
        assert "sign off as omega" in second.last()


def test_a_claim_reaches_the_prompt_after_it_has_aged_out_of_recall(store):
    """The reason this is a derived view and not just recall. A standing habit
    that expired after forty episodes would be indistinguishable from one omega
    was never taught."""
    q = EventQueue(store)
    event = q.append(_typed("teach"))
    q.append(_claim(text="never use exclamation marks", for_seq=event,
                    source_seq=event))
    for i in range(turn.RECALL_N + 5):
        q.append(_typed(f"filler {i}"))

    watched = _Watched()
    _drained(q, watched)
    assert "never use exclamation marks" in watched.last()


def test_the_learned_section_is_bounded_and_drops_inferred_before_told(store):
    """A count is not a budget (DL-039), and the drop order is the one
    judgement in this section: when it bites, what omega *inferred* goes before
    what the person actually **said**."""
    q = EventQueue(store)
    event = q.append(_typed("teach"))
    for i in range(40):
        q.append(_claim(text=f"inferred {i}: " + "x" * 300, for_seq=event,
                        source_seq=event, explicit=False))
    q.append(_claim(text="TOLD: the one that must survive", for_seq=event,
                    source_seq=event, explicit=True))
    q.append(_typed("ordinary message"))

    watched = _Watched()
    _drained(q, watched)
    prompt = watched.last()
    _, _, rest = prompt.partition(LEARNED_HEADING + ":\n")
    section = rest.split("\n\nRecent history:")[0]
    assert len(section) <= turn.MAX_LEARNED_CHARS + 400, "the section is not bounded"
    assert "not shown here" in section, "a silent drop is the bug, not the fix"
    assert "TOLD: the one that must survive" in section
