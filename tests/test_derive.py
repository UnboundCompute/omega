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


def test_nothing_open_renders_no_heading_at_all(store):
    """Not an empty section. A heading that is usually empty teaches the model
    to skip the section by the time it matters."""
    q = EventQueue(store)
    q.append(_typed("just a message"))

    watched = _Watched()
    _drained(q, watched)

    assert OPEN_HEADING not in watched.last()


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
