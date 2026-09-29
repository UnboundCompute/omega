"""The second memory — DL-062: what happened, kept apart from what is true.

The pair `CLAUDE.md` asks for is unusually literal here, because this feature
*is* a loosening and the guard is the whole reason it was built the way it was:

    capability  a session that shows unfinished work leaves a moment behind,
                and the situation an hour later is not the one from before
    violation   nothing a moment does may weaken the claim set — not the
                "patterns, not events" instruction, not the per-pass cap, and
                not the lens that was deliberately denied moments

DL-054's memory firehose is the failure this whole path is bounded against, and
a moment is *far* easier for a model to write than a claim: "he is partway
through X" passes a bar that "he always does X" does not. So the bounds are
tested as hard as the behaviour, and every one of them is code rather than
model judgement (DL-014): a per-pass cap here, a horizon and a per-digest cap
in `derive`.

The last test in this file is the one that says why any of this exists. Before
DL-062 two unprompted looks an hour apart produced byte-identical digests, so
the second look could never reach a different verdict than the first — a loop
that ran on time and could never have anything new to say. That test fails if
moments stop reaching the situation, which is the only thing that makes the
digest a function of time rather than a constant.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from omega import derive, episodes, learn, notice, provider
from omega.executor import Executor
from omega.memory import MemoryStore
from omega.queue import EventQueue

from tests.test_transcripts import NOW, _five_messages, _session, _written

AT = "2026-09-23T12:00:00+00:00"
MIDDLE = "partway through the retry logic in the uploader"


def _answer(*, claims: list[dict] | None = None, moments: list[dict] | None = None) -> str:
    body: dict[str, object] = {"claims": claims or []}
    if moments is not None:
        body["moments"] = moments
    return json.dumps(body)


def _a_claim(text: str = "They test before they commit.") -> dict:
    return {"text": text, "situation": "while working in a coding tool"}


def _moments(store) -> list[dict]:
    return _written(store, episodes.MOMENT_NOTICED)


def _run_session(store, tmp_path: Path, answer: str) -> Executor:
    """One transcript on disk, read once by a pass that answers ``answer``.

    The WORK lens on purpose: it is the one moments were built for, because
    `transcript.ingested` keeps a session's id and none of its content, so what
    the person was in the middle of is gone the moment the digest is dropped.
    """
    _session(tmp_path, records=_five_messages())
    q = EventQueue(store)
    # Not scenery: a claim names the episode it came from, so ingestion refuses
    # to run against an empty log (see `_ready` in test_transcripts).
    q.append(episodes.inbound("morning", channel="tray"))
    fp = provider.FakeProvider({provider.LEARN: lambda role, messages: answer})
    ex = Executor(q, complete=fp.complete)
    ex.recover()
    ex.ingest(root=tmp_path, now=NOW)
    return ex


# --- the capability ---------------------------------------------------------


def test_a_work_session_files_what_the_person_is_in_the_middle_of(
    store, tmp_path: Path
) -> None:
    """The capability, with the claim beside it as the control.

    Both halves of one answer are asserted because the interesting failure is
    not "nothing was filed" — it is one list being written and the other
    silently dropped, which a test that only looked at moments would call a
    pass.
    """
    _run_session(
        store,
        tmp_path,
        _answer(claims=[_a_claim()], moments=[{"text": MIDDLE}]),
    )

    filed = _moments(store)
    assert [m["text"] for m in filed] == [MIDDLE]
    # The control: the claim from the same answer landed too, so this is the
    # two memories working side by side rather than one replacing the other.
    assert [c["text"] for c in _written(store, episodes.CLAIM_EXTRACTED)] == [
        _a_claim()["text"]
    ]


def test_a_moment_is_never_filed_as_a_claim(store, tmp_path: Path) -> None:
    """The separation the whole design turns on, asserted at the write.

    An answer carrying a moment and no claim must leave the claim set exactly
    as it was. Kept apart at the episode kind rather than by a flag on one
    kind, because a flag is one refactor away from being ignored and a separate
    kind is not.
    """
    _run_session(store, tmp_path, _answer(claims=[], moments=[{"text": MIDDLE}]))

    # The control for the silence below: the pass ran and did write something,
    # so an empty claim list is a decision rather than a pass that never fired.
    assert [m["text"] for m in _moments(store)] == [MIDDLE]
    assert _written(store, episodes.CLAIM_EXTRACTED) == []


# --- the bounds -------------------------------------------------------------


def test_the_conversation_lens_cannot_file_a_moment(store, tmp_path: Path) -> None:
    """One setting, two outcomes — the pattern `harness-proving` asks for.

    The same answer is offered twice. Through the lens that was denied moments
    it files none; through the lens that was offered them it files one. Without
    the second half this would pass just as happily if moments never worked at
    all.
    """
    offered = _answer(moments=[{"text": MIDDLE}])

    # A lens with moments off was never told the field exists, so a `moments`
    # list coming back from one is an answer to a question nobody asked.
    assert learn.CONVERSATION.moments is False
    dropped = learn.reflect(
        provider.FakeProvider({provider.LEARN: offered}).complete,
        transcript="some chatter",
        known=(),
        observing=learn.CONVERSATION,
    )
    assert dropped.moments == []

    # The control. One field differs and the moment comes through.
    kept = learn.reflect(
        provider.FakeProvider({provider.LEARN: offered}).complete,
        transcript="some chatter",
        known=(),
        observing=learn.WORK,
    )
    assert [m["text"] for m in kept.moments] == [MIDDLE]


def test_more_moments_than_the_cap_is_a_failed_pass(store, tmp_path: Path) -> None:
    """Over the cap the whole pass is refused, not trimmed.

    Refusing rather than truncating for the reason the claim cap gives: a model
    that returned six things worth noticing has not ranked, and keeping the
    first three of an unranked six is keeping an arbitrary three.
    """
    over = [{"text": f"midway through thing {i}"} for i in range(learn.MAX_MOMENTS_PER_PASS + 1)]
    with pytest.raises(learn.NotExtracted):
        learn._parse_moment_list(over)

    # The control: one fewer is accepted, so the refusal above is the cap and
    # not something wrong with the shape of these objects.
    assert len(learn._parse_moment_list(over[:-1])) == learn.MAX_MOMENTS_PER_PASS


def test_a_stale_moment_is_not_shown_and_a_fresh_one_is() -> None:
    """The horizon, with both halves in one case.

    A moment is never retracted and never superseded — it stops being recent,
    and this is the only thing that removes one. So the test that it ages out
    is the test that the view has any bound at all.
    """
    now = "2026-09-29T12:00:00+00:00"
    old = "2026-09-20T12:00:00+00:00"
    view = derive.Moments()
    view.apply(5, episodes.moment_noticed(for_seq=1, text="stale", source_seq=1, at=old))
    view.apply(6, episodes.moment_noticed(for_seq=1, text="fresh", source_seq=1, at=now))

    assert [m.text for m in view.recent(now=now)] == ["fresh"]


def test_only_the_newest_moments_reach_one_situation() -> None:
    """The per-digest cap, and that what survives it is the *newest*.

    This is the bound that protects the prompt: a busy week files far more
    moments than claims, and without it the "Lately" section would crowd out
    the part of the situation that says what is true.
    """
    now = "2026-09-29T12:00:00+00:00"
    view = derive.Moments()
    for seq in range(10, 10 + derive.MAX_MOMENTS_SHOWN + 2):
        view.apply(
            seq,
            episodes.moment_noticed(
                for_seq=1, text=f"thing {seq}", source_seq=1, at=now
            ),
        )

    shown = view.recent(now=now)
    assert len(shown) == derive.MAX_MOMENTS_SHOWN
    # The newest survive, and they read forwards: the cap picks which to show,
    # it does not decide the order they are read in.
    assert [m.text for m in shown] == [
        f"thing {seq}" for seq in range(12, 10 + derive.MAX_MOMENTS_SHOWN + 2)
    ]


# --- why any of this exists -------------------------------------------------


def test_two_looks_an_hour_apart_differ_once_something_has_happened() -> None:
    """The whole point of DL-062, as a check that would notice it being undone.

    Observed on the live log before this was built: two unprompted passes an
    hour apart produced byte-identical digests apart from the clock, because
    every input to the situation was timeless — the same claims, the same
    schedules. A judge handed the same text twice cannot reach a different
    verdict the second time, so the loop ran on schedule and could never have
    anything new to say.

    A moment is the only input that changes on its own. If this test goes green
    with the "Lately" section removed, the loop is back to being a constant.
    """
    first = datetime(2026, 9, 29, 11, 0, 0, tzinfo=timezone.utc)
    second = first + timedelta(hours=1)
    claims = [
        derive.Claim(
            seq=1,
            text="They prefer short answers.",
            trigger=None,
            situation="s",
            source_seq=1,
            explicit=True,
        )
    ]

    def look(at: datetime, moments: list[derive.Moment]) -> str:
        return notice.situation(now=at, claims=claims, moments=moments)

    # Without a moment, an hour changes nothing but the clock. This is the
    # broken behaviour, asserted so the test below means something.
    bare_first = look(first, [])
    bare_second = look(second, [])
    assert _without_clock(bare_first) == _without_clock(bare_second), (
        "the timeless half of the situation is expected to be constant; if this "
        "fails, something else now varies and this test is no longer isolating "
        "moments"
    )

    # With one, it does not.
    happened = [
        derive.Moment(seq=9, text=MIDDLE, source_seq=8, at=second.isoformat())
    ]
    with_moment = look(second, happened)
    assert with_moment != bare_second
    assert MIDDLE in with_moment
    assert MIDDLE not in bare_second


def _without_clock(text: str) -> str:
    """The situation minus any line carrying a time.

    Written out rather than compared whole because the digest legitimately
    carries the clock, and a test that demanded two looks be byte-identical
    would be asserting something false about a correct implementation.
    """
    import re

    return "\n".join(
        line for line in text.splitlines() if not re.search(r"\d{1,2}:\d{2}", line)
    )
