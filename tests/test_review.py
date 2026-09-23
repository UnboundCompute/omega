"""Reading back what omega has been taught — DL-048's other half.

``test_retract.py`` covers taking a claim back. This covers being able to *see*
one in the first place, which is the defect that made retraction hard to use:
the learned set reaches the model only as the claims whose triggers fire on the
current sentence, so until now "what have you learned about me?" was answered
from whatever matched that question, and the whole set was visible nowhere.

Two properties carry most of the weight here. **Reading must not write** — an
audit that appends to the log it is auditing is not an audit — and **the empty
answer must be three-valued**, because "omega has been taught nothing" and "this
view never read the log" are the same empty list and only the first is an
answer.

Nothing here reaches the network or wants a key. That is not incidental: the
moment a person most wants to check what omega believes is the moment omega has
stopped being able to think, so the review path deliberately runs before the
provider is touched and one case pins that.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from omega import __main__ as cli
from omega import derive, episodes, learn
from omega.memory import MemoryStore
from omega.queue import EventQueue
from omega.schedule import Schedule

BLACK = "I take my coffee black."
AT = "2026-09-24T12:00:00+00:00"


def _claim(seq: int, text: str, **kw) -> derive.Claim:
    return derive.Claim(
        seq=seq,
        text=text,
        trigger=kw.get("trigger"),
        situation=kw.get("situation", ""),
        source_seq=kw.get("source_seq", 1),
        explicit=kw.get("explicit", True),
        supersedes=kw.get("supersedes"),
    )


def _schedule(**kw) -> Schedule:
    return Schedule(
        id=kw.get("id", "s1-0"),
        instruction=kw.get("instruction", "brief me"),
        created_at=datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc),
        cron=kw.get("cron", "0 9 *"),
        every=kw.get("every"),
    )


def _lines() -> tuple[list[str], object]:
    out: list[str] = []
    return out, out.append


# --- the empty answer -------------------------------------------------------


def test_a_log_read_to_the_end_with_nothing_in_it_says_so():
    assert learn.review([], current=True) == "I have not been taught anything yet."


def test_a_view_that_has_not_read_the_log_refuses_to_answer():
    """The distinction the whole parameter exists for. A default-constructed
    ``Learned`` and a rebuild over an empty log both report nothing; only one of
    them is evidence that omega has been taught nothing."""
    assert "cannot tell you" in learn.review([], current=False)


def test_a_caller_that_does_not_say_gets_the_uncertain_answer():
    """**Fail closed on the caller's silence.** ``executor.learned`` is empty
    until the first turn folds it, so a future caller passing that view without
    thinking must not get "omega has been taught nothing" — which would be a
    check that passes on empty, wearing a sentence."""
    assert "cannot tell you" in learn.review([])


# --- what it renders --------------------------------------------------------


def test_a_claim_is_shown_with_its_id_and_when_it_fires():
    text = learn.review([_claim(12, BLACK, trigger={"any": ["coffee"]})], current=True)
    assert "[12]" in text
    assert BLACK in text
    assert "when you mention coffee" in text


def test_the_id_is_shown_although_the_receipt_hides_it():
    """Not a contradiction of DL-048 #6. That rule is about *confirmations* —
    "claim 41 has been forgotten" is a sentence the person cannot check. Here
    the id sits against the text it names, so it is checkable, and it is the
    same handle ``supersedes`` and ``retract`` already quote."""
    text = learn.review([_claim(41, BLACK)], current=True)
    assert "[41]" in text
    assert text.index("[41]") < text.index(BLACK)


def test_where_a_claim_was_learned_is_shown_when_it_is_known():
    text = learn.review([_claim(12, BLACK, situation="in the tray, 24 September")],
                        current=True)
    assert "in the tray, 24 September" in text


def test_a_claim_with_no_situation_renders_no_empty_line():
    text = learn.review([_claim(12, BLACK)], current=True)
    assert "    learned" not in text
    assert "\n\n" not in text


def test_a_standing_schedule_is_rendered_from_the_stored_expression():
    """DL-044 #5's rule, and review is where it bites hardest: a receipt read
    weeks ago is the only other place ``0 9 *`` and ``9 0 *`` could have been
    told apart."""
    text = learn.review(
        [],
        running=[_schedule()],
        current=True,
    )
    assert "brief me" in text
    assert "9:00" in text
    assert "[s1-0]" in text


def test_a_schedule_that_cannot_run_is_surfaced_with_its_reason():
    """The worst news the log holds. A schedule whose expression will not parse
    is quarantined at fold time and silently never fires; this is the only
    surface on which that is discoverable."""
    text = learn.review([], broken={"s9-0": "cron has 4 fields, expected 3"},
                        current=True)
    assert "cannot run" in text
    assert "s9-0" in text
    assert "expected 3" in text


def test_a_store_holding_only_a_broken_schedule_does_not_report_nothing():
    """It would be the exact inversion: the one state that most needs saying,
    rendered as "I have not been taught anything yet."."""
    text = learn.review([], broken={"s9-0": "unparseable"}, current=True)
    assert "have not been taught" not in text


def test_the_three_sections_are_separated():
    text = learn.review(
        [_claim(1, BLACK)],
        running=[_schedule()],
        broken={"s9-0": "unparseable"},
        current=True,
    )
    assert text.count("\n\n") == 2


# --- the command ------------------------------------------------------------


def _taught_store(store_dir: Path) -> None:
    with MemoryStore.open(store_dir) as s:
        q = EventQueue(s)
        q.append(episodes.inbound("teach", channel="tray", at=AT))
        q.append(
            episodes.claim_extracted(
                for_seq=1,
                text=BLACK,
                source_seq=1,
                situation="in the tray",
                trigger={"any": ["coffee"]},
                explicit=True,
                at=AT,
            )
        )


def test_the_command_prints_what_omega_believes(store_dir: Path):
    _taught_store(store_dir)
    out, write = _lines()
    assert cli.review(store_dir, write=write) == 0
    assert BLACK in "\n".join(out)


def test_the_command_wants_no_key(store_dir: Path, monkeypatch: pytest.MonkeyPatch):
    """The moment a person most wants to check what omega believes is the moment
    it has stopped being able to think. Answering a question about the log with
    a question about the network would be the wrong failure."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    _taught_store(store_dir)
    # Through `main`, not `review`, because the thing being pinned is the
    # *ordering*: the provider is built a few lines below the branch this takes,
    # and a reordering would fail here and nowhere else.
    assert cli.main(["--store", str(store_dir), "--learned"]) == 0


def test_the_command_reports_a_view_that_is_behind_the_log(
    store_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """The command must *compute* whether its views saw the whole log, not
    assert it. Today ``rebuild`` always reaches the head, so hardcoding ``True``
    would pass every other case here — and the day something caches or truncates
    that fold, the only symptom would be omega confidently reporting it had been
    taught nothing."""
    _taught_store(store_dir)
    monkeypatch.setattr(derive.Learned, "rebuild", staticmethod(lambda store: derive.Learned()))

    out, write = _lines()
    assert cli.review(store_dir, write=write) == 0
    assert "cannot tell you" in "\n".join(out)


def test_a_retracted_claim_is_not_shown(store_dir: Path):
    """The two halves of DL-048 meeting: retraction removes it from the active
    set, and review is where a person can see that it did."""
    _taught_store(store_dir)
    with MemoryStore.open(store_dir) as s:
        EventQueue(s).append(episodes.claim_retracted(for_seq=1, claim_seq=2, at=AT))

    out, write = _lines()
    assert cli.review(store_dir, write=write) == 0
    assert BLACK not in "\n".join(out)


def test_reading_the_record_does_not_change_the_record(store_dir: Path):
    """**The violation metric.** An audit that appends to the log it audits is
    not an audit, and every other episode kind here is written by something that
    looked like a read first."""
    _taught_store(store_dir)
    with MemoryStore.open(store_dir) as s:
        before = EventQueue(s).head()

    out, write = _lines()
    assert cli.review(store_dir, write=write) == 0

    with MemoryStore.open(store_dir) as s:
        assert EventQueue(s).head() == before


def test_the_command_says_omega_is_running_rather_than_raising(store, store_dir: Path):
    """The common case, because the log holds an exclusive lock for the lifetime
    of the open handle. "omega is already running" and "your store is broken"
    want opposite next actions, and the raw error distinguishes neither."""
    out, write = _lines()
    code = cli.review(store_dir, write=write)
    assert code == 2
    assert "already running" in "\n".join(out)


def test_an_unopenable_store_is_a_sentence_and_not_a_traceback(tmp_path: Path):
    out, write = _lines()
    missing = tmp_path / "nope" / "deeper"
    assert cli.review(missing, write=write) == 2
    assert "could not open" in "\n".join(out)


def test_learned_and_serve_are_refused_together():
    """One reads the log and exits, the other never returns. Silently letting
    one win would make the flag a coin toss."""
    with pytest.raises(SystemExit):
        cli.main(["--learned", "--serve"])
