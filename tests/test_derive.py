"""What's open, derived from the log (DL-041).

The cases that matter here are not "does a fold fold". They are the two that
would let omega quietly get this wrong in production: discharging on the seq an
answer actually names, and keeping a block visible after it has aged out of
recall.
"""

from __future__ import annotations

import pytest

from omega import derive, episodes, queue, turn


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
