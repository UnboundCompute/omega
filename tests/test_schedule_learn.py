"""A taught time becomes a schedule, not a claim with an hour on it — DL-044.

The defect this closes was live and shipped: DL-042 gave a claim an ``hours``
trigger, DL-043 started writing claims from teach drops, and *"every morning at
nine, remind me to take my meds"* landed as a claim that only applies when the
person is already talking to omega at nine. The receipt then said *"I wrote this
down: ... (between 9:00 and 10:00)"*, which reads as a promise.

So the cases here come in pairs, and the pairing is the point. A note that asks
omega to *act* must produce something the clock fires; a note that merely
*mentions* a time must not. Either one alone would be satisfied by code that is
wrong in the other direction.

The end-to-end cases grade the world: they read the schedule back out of the
log and drive the real :class:`omega.schedule.Scheduler` over it, because
"omega wrote a schedule episode" and "omega will wake up" are different claims
and only the second one is what the person was promised.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest

from omega import episodes, learn, provider, schedule
from omega.queue import EventQueue

from tests.teaching import (
    _a_claim,
    _claim_obj,
    _claims_in,
    _drain,
    _log,
    _replies,
    _teach_text,
    _Teaching,
)


def _answer(*, claims=(), schedules=(), cancel=()) -> str:
    return json.dumps(
        {
            "claims": list(claims),
            "schedules": list(schedules),
            "cancel": list(cancel),
        }
    )


def _a_schedule(instruction="Remind me to take my meds.", cron="0 9 *") -> dict:
    return {"instruction": instruction, "cron": cron}


def _running(sid="s7-0", instruction="Remind me to take my meds.", cron="0 9 *"):
    return schedule.Schedule(
        id=sid,
        instruction=instruction,
        created_at=datetime(2026, 9, 1, 12, 0).astimezone(),
        cron=cron,
    )


def _schedules_in(q: EventQueue) -> list[dict]:
    return [
        p for _, p in _log(q) if p.get("kind") == episodes.SCHEDULE_CREATED
    ]


def _cancels_in(q: EventQueue) -> list[dict]:
    return [
        p for _, p in _log(q) if p.get("kind") == episodes.SCHEDULE_CANCELLED
    ]


# --- describing a cron expression in words ----------------------------------


@pytest.mark.parametrize(
    "cron,expected",
    [
        ("0 9 *", "at 9:00 every day"),
        ("30 6 1,2,3,4,5", "at 6:30 on weekdays"),
        ("0 10 0,6", "at 10:00 at weekends"),
        ("0 9 1", "at 9:00 on Monday"),
        ("*/30 * *", "at :00 and :30 past every hour"),
        ("*/30 * 1,2,3,4,5", "at :00 and :30 past every hour on weekdays"),
        ("* * *", "every minute"),
        ("0 0,6,12,18 *", "at 0:00 and 6:00 and 12:00 and 18:00 every day"),
        ("0 */2 *", "12 times every day"),
    ],
)
def test_a_cron_expression_reads_back_in_words(cron, expected):
    """DL-044 #5. The receipt's whole value is that a person can disagree with
    it, and nobody disagrees with ``30 6 1-5``."""
    assert schedule.describe_cron(cron) == expected


def test_the_two_expressions_that_are_easy_to_swap_read_differently():
    """``0 9 *`` and ``9 0 *`` are both valid and nothing downstream will ever
    notice which one was meant. This is the specific confusion the plain-words
    receipt exists to surface."""
    assert schedule.describe_cron("0 9 *") == "at 9:00 every day"
    assert schedule.describe_cron("9 0 *") == "at 0:09 every day"


def test_an_unusable_expression_is_refused_rather_than_described():
    with pytest.raises(schedule.CronError):
        schedule.validate_cron("0 99 *")


# --- parsing --------------------------------------------------------------


def test_an_answer_carries_claims_and_schedules_together():
    """DL-044 #2: one call, so the extractor decides once whether a sentence is
    a belief or an appointment."""
    got = learn.parse_answer(
        _answer(claims=[_a_claim()], schedules=[_a_schedule()])
    )
    assert [c["text"] for c in got.claims] == ["keep status updates short"]
    assert got.schedules == [
        {"instruction": "Remind me to take my meds.", "cron": "0 9 *"}
    ]


def test_an_answer_with_no_schedules_field_still_parses():
    """The field is additive on an answer shape that already existed, so its
    absence is an old answer and not a broken one."""
    got = learn.parse_answer(json.dumps({"claims": [_a_claim()]}))
    assert got.schedules == []
    assert got.cancel == []


def test_an_answer_with_no_claims_field_is_still_refused():
    """The one field the model is always told to return. An answer missing it
    is an answer in a different format, and DL-043's strictness holds."""
    with pytest.raises(learn.NotExtracted):
        learn.parse_answer(json.dumps({"schedules": [_a_schedule()]}))


@pytest.mark.parametrize(
    "bad,because",
    [
        ({"instruction": "x"}, "no cron at all"),
        ({"instruction": "x", "cron": ""}, "a blank cron"),
        ({"instruction": "", "cron": "0 9 *"}, "a blank instruction"),
        ({"cron": "0 9 *"}, "no instruction"),
        ({"instruction": "x", "cron": "0 99 *"}, "an hour that does not exist"),
        ({"instruction": "x", "cron": "0 9"}, "two fields instead of three"),
        ({"instruction": "x", "cron": "0 9 * *"}, "four fields"),
        ("not an object", "not being an object"),
        ({"instruction": "x", "cron": 900}, "a cron that is not a string"),
        (
            {"instruction": "x" * 501, "cron": "0 9 *"},
            "an instruction over the length cap",
        ),
    ],
)
def test_a_bad_schedule_fails_the_whole_extraction(bad, because):
    """All-or-nothing, extended to schedules. A note yielding a good claim and
    a broken schedule files neither: filing the claim and dropping the schedule
    would produce a receipt that is quietly short in exactly the half the
    person was relying on."""
    with pytest.raises(learn.NotExtracted):
        learn.parse_answer(_answer(claims=[_a_claim()], schedules=[bad]))


def test_more_schedules_than_the_cap_is_refused():
    """Tighter than the claim cap on purpose: a wrong claim costs prompt
    budget, a wrong schedule wakes omega up."""
    many = [_a_schedule(cron=f"{m} 9 *") for m in range(learn.MAX_SCHEDULES_PER_NOTE + 1)]
    with pytest.raises(learn.NotExtracted):
        learn.parse_answer(_answer(schedules=many))


def test_the_finest_expression_this_path_can_write_fires_no_faster_than_the_floor():
    """DL-044 #7, checked rather than asserted.

    Extraction emits cron only, and cron's resolution is one minute; the
    scheduler requires a strictly newer slot than the last fire. So the busiest
    thing a sentence can create fires once a minute, which is exactly
    ``episodes.MIN_EVERY_SECONDS``. The floor on the new path is the grammar,
    not a second check that could drift from the first.
    """
    got = learn.parse_answer(_answer(schedules=[_a_schedule(cron="* * *")]))
    assert got.schedules[0]["cron"] == "* * *"


def test_the_busiest_taught_schedule_fires_once_a_minute(store):
    """The other half of the floor, driven rather than reasoned about.

    ``* * *`` matches every minute, and the scheduler requires a slot strictly
    newer than the last fire — so ticking four times inside one minute produces
    one fire, and the minute boundary produces the next.
    """
    q = EventQueue(store)
    written = learn.file_schedules(
        q, [{"instruction": "x", "cron": "* * *"}], source_seq=1
    )
    clock = schedule.Scheduler(q)
    clock.refresh()

    start = written[0].created_at.replace(second=0, microsecond=0)
    minute = start + timedelta(minutes=1)
    fires = 0
    for second in (0, 15, 30, 45):
        for due in clock.due(minute + timedelta(seconds=second)):
            seq = clock.fire(due)
            # Drain it, so the overlap guard is not what is measured here.
            q.claim(seq)
            q.finish(seq)
            fires += 1
    assert fires == 1

    for due in clock.due(minute + timedelta(minutes=1)):
        clock.fire(due)
        fires += 1
    assert fires == 2


def test_an_every_field_from_the_model_is_ignored_not_honoured():
    """Only ``cron`` is accepted. ``every`` stays a programmatic spelling that
    a sentence cannot reach — which is what keeps the minute floor above from
    being bypassable by a model proposing ``every: 1``."""
    got = learn.parse_answer(
        _answer(schedules=[{"instruction": "x", "cron": "0 9 *", "every": 1}])
    )
    assert got.schedules == [{"instruction": "x", "cron": "0 9 *"}]


# --- cancelling -------------------------------------------------------------


def test_a_cancel_names_a_running_schedule():
    got = learn.parse_answer(
        _answer(cancel=["s7-0"]), running=[_running("s7-0")]
    )
    assert got.cancel == ["s7-0"]


def test_a_cancel_naming_nothing_running_is_refused():
    """``Scheduler._observe`` pops on cancel with a default, so cancelling
    something that is not running succeeds silently — and the receipt would
    then say a reminder had stopped while it went on firing."""
    with pytest.raises(learn.NotExtracted) as caught:
        learn.parse_answer(_answer(cancel=["s99-0"]), running=[_running("s7-0")])
    assert "s99-0" in str(caught.value)


def test_a_cancel_with_nothing_running_at_all_is_refused():
    with pytest.raises(learn.NotExtracted):
        learn.parse_answer(_answer(cancel=["s7-0"]))


def test_a_repeated_cancel_id_is_written_once():
    got = learn.parse_answer(
        _answer(cancel=["s7-0", "s7-0"]), running=[_running("s7-0")]
    )
    assert got.cancel == ["s7-0"]


# --- ids --------------------------------------------------------------------


def test_ids_come_from_the_teaching_episode_never_from_the_model():
    """DL-044's id rule. Two notes that both propose the obvious word cannot
    collide, because neither of them is asked."""
    assert learn.schedule_id(41, 0) == "s41-0"
    assert learn.schedule_id(41, 1) == "s41-1"
    assert learn.schedule_id(42, 0) != learn.schedule_id(41, 0)


def test_a_model_supplied_id_is_not_used(store):
    q = EventQueue(store)
    written = learn.file_schedules(
        q,
        [{"instruction": "x", "cron": "0 9 *", "id": "meds"}],
        source_seq=7,
    )
    assert [s.id for s in written] == ["s7-0"]
    assert _schedules_in(q)[0]["id"] == "s7-0"


# --- the receipt ------------------------------------------------------------


def test_the_receipt_says_what_will_happen_and_when():
    text = learn.receipt(
        (),
        scheduled=[_running("s7-0", "Remind me to take my meds.", "0 9 *")],
    )
    assert "I will wake up and do this:" in text
    assert "Remind me to take my meds." in text
    assert "at 9:00 every day" in text
    assert "s7-0" in text


def test_the_receipt_renders_the_stored_expression_not_the_note():
    """DL-044 #5, and the case that makes it worth having: the note said nine
    in the morning and the model wrote the fields the other way round. The
    receipt has to show what the clock will read, which is the only way the
    person can catch it."""
    text = learn.receipt((), scheduled=[_running(cron="9 0 *")])
    assert "at 0:09 every day" in text
    assert "9:00 every day" not in text


def test_the_receipt_names_what_it_stopped_in_the_persons_words():
    text = learn.receipt((), stopped=[_running("s7-0", "Remind me about meds.")])
    assert "I stopped this:" in text
    assert "Remind me about meds." in text


def test_a_receipt_with_a_claim_and_a_schedule_carries_both():
    text = learn.receipt(
        [_claim_obj(5, "keep updates short")],
        scheduled=[_running()],
    )
    assert "I wrote this down:" in text
    assert "I will wake up and do this:" in text


def test_nothing_at_all_still_says_nothing_was_recorded():
    """Fail closed on empty, unchanged by the new fields."""
    assert "nothing was recorded" in learn.receipt((), scheduled=(), stopped=())


def test_an_error_receipt_beats_every_other_branch():
    text = learn.receipt(
        [_claim_obj(5, "x")], scheduled=[_running()], error="the sky fell"
    )
    assert "could not write that down" in text
    assert "I will wake up" not in text


# --- the whole path ---------------------------------------------------------


def test_a_taught_time_becomes_a_schedule_the_clock_actually_fires(store):
    """**The capability metric.** Not "a schedule episode was written" — the
    bar is that the clock, folding the same log, finds it due and fires it."""
    q = EventQueue(store)
    q.append(
        episodes.inbound(
            _teach_text("Every morning at nine, remind me to take my meds."),
            channel="tray",
        )
    )
    teaching = _Teaching(
        learn_answer=_answer(schedules=[_a_schedule(cron="0 9 *")])
    )
    _drain(q, teaching)

    (written,) = _schedules_in(q)
    assert written["cron"] == "0 9 *"
    assert written["instruction"] == "Remind me to take my meds."

    # Grade the world: drive the real scheduler over the log omega wrote.
    clock = schedule.Scheduler(q)
    clock.refresh()
    assert [s.id for s in clock.schedules] == [written["id"]]

    created = datetime.fromisoformat(written["at"])
    nine = (created + timedelta(days=1)).replace(
        hour=9, minute=0, second=0, microsecond=0
    )
    (due,) = clock.due(nine)
    assert due.schedule.instruction == "Remind me to take my meds."

    fired = clock.fire(due)
    payload = dict(_log(q))[fired]
    assert payload["kind"] == episodes.MESSAGE_INBOUND
    assert payload["channel"] == "schedule"
    assert "take my meds" in payload["text"]


def test_a_preference_that_merely_mentions_an_hour_stays_a_claim(store):
    """**The violation metric.** The inverse defect, and the reason the
    discriminator is *who acts* rather than *does it say a time*: "in the
    mornings I prefer short answers" is a real thing to teach, and turning it
    into a 9am wake-up would be this change failing in the other direction."""
    q = EventQueue(store)
    q.append(
        episodes.inbound(
            _teach_text("In the mornings I prefer short answers."),
            channel="tray",
        )
    )
    teaching = _Teaching(
        learn_answer=_answer(
            claims=[
                _a_claim("keep answers short in the morning", trigger={"hours": [6, 12]})
            ]
        )
    )
    _drain(q, teaching)

    assert len(_claims_in(q)) == 1
    assert _schedules_in(q) == []

    clock = schedule.Scheduler(q)
    clock.refresh()
    assert clock.schedules == []


def test_an_ordinary_turn_creates_no_schedule(store):
    """The gate is the teach drop and nothing else. An ordinary message that
    asks in so many words must not reach the extractor at all."""
    q = EventQueue(store)
    q.append(
        episodes.inbound("remind me every morning at nine", channel="tray")
    )
    teaching = _Teaching(learn_answer=_answer(schedules=[_a_schedule()]))
    _drain(q, teaching)

    assert _schedules_in(q) == []
    assert provider.LEARN not in {role for role, _ in teaching.calls}


def test_the_receipt_reaches_the_person_with_the_reply(store):
    q = EventQueue(store)
    q.append(
        episodes.inbound(
            _teach_text("Every morning at nine, remind me to take my meds."),
            channel="tray",
        )
    )
    teaching = _Teaching(
        learn_answer=_answer(schedules=[_a_schedule(cron="0 9 *")]),
        reply="Got it.",
    )
    _drain(q, teaching)

    (reply,) = _replies(q)
    assert "Got it." in reply
    assert "at 9:00 every day" in reply


def test_a_taught_cancel_stops_the_clock_firing(store):
    """DL-044 #6. A schedule that cannot be stopped is a worse defect than one
    that cannot be started, so this case drives the clock *after* the cancel
    and asserts it finds nothing due at the hour it used to fire."""
    q = EventQueue(store)
    q.append(
        episodes.inbound(
            _teach_text("Every morning at nine, remind me to take my meds."),
            channel="tray",
        )
    )
    _drain(q, _Teaching(learn_answer=_answer(schedules=[_a_schedule("Meds.", "0 9 *")])))
    (created,) = _schedules_in(q)
    sid = created["id"]

    q.append(
        episodes.inbound(
            _teach_text("Stop reminding me about the meds."), channel="tray"
        )
    )
    second = _Teaching(learn_answer=_answer(cancel=[sid]))
    _drain(q, second)

    assert [p["id"] for p in _cancels_in(q)] == [sid]

    clock = schedule.Scheduler(q)
    clock.refresh()
    assert clock.schedules == []
    nine = (datetime.fromisoformat(created["at"]) + timedelta(days=1)).replace(
        hour=9, minute=0, second=0, microsecond=0
    )
    assert clock.due(nine) == []

    (_, reply) = _replies(q)
    assert "I stopped this:" in reply
    assert "Meds." in reply


def test_the_extractor_is_shown_what_is_already_scheduled(store):
    """It cannot name a schedule to cancel without having been shown one, and
    the parser refuses ids it was not shown — so the prompt carrying them is
    load-bearing rather than decorative."""
    q = EventQueue(store)
    q.append(
        episodes.inbound(_teach_text("Every morning, meds."), channel="tray")
    )
    _drain(q, _Teaching(learn_answer=_answer(schedules=[_a_schedule("Meds.", "0 9 *")])))
    (created,) = _schedules_in(q)

    q.append(episodes.inbound(_teach_text("Also water."), channel="tray"))
    second = _Teaching(learn_answer=_answer(claims=[_a_claim()]))
    _drain(q, second)

    prompt = second.last()
    assert "Already scheduled:" in prompt
    assert f"[{created['id']}] Meds." in prompt


def test_a_broken_expression_from_the_model_records_nothing_and_says_so(store):
    """Refused at extraction, not quarantined at fold time (DL-044 #5). The
    person is present, so they get told rather than getting a schedule that
    exists, reports healthy and never fires."""
    q = EventQueue(store)
    q.append(
        episodes.inbound(_teach_text("Every morning at nine, meds."), channel="tray")
    )
    teaching = _Teaching(
        learn_answer=_answer(schedules=[_a_schedule(cron="0 99 *")]),
        reply="Sure.",
    )
    _drain(q, teaching)

    assert _schedules_in(q) == []
    assert _claims_in(q) == []
    (reply,) = _replies(q)
    assert "could not write that down" in reply

    # And the turn still succeeded — DL-043 #5 holds for the new failure too.
    outcomes = [
        p["outcome"] for _, p in _log(q) if p.get("kind") == episodes.TURN_COMPLETED
    ]
    assert outcomes == ["spoke"]


def test_a_note_that_teaches_and_schedules_writes_both_or_neither(store):
    q = EventQueue(store)
    q.append(
        episodes.inbound(
            _teach_text("Keep updates short, and nudge me at nine each day."),
            channel="tray",
        )
    )
    teaching = _Teaching(
        learn_answer=_answer(claims=[_a_claim()], schedules=[_a_schedule()])
    )
    _drain(q, teaching)

    assert len(_claims_in(q)) == 1
    assert len(_schedules_in(q)) == 1


def test_a_second_note_does_not_overwrite_the_first_schedule(store):
    """The id rule, end to end. Two notes about the same subject produce two
    standing schedules rather than one silently replacing the other."""
    q = EventQueue(store)
    for note, cron in (("Meds at nine.", "0 9 *"), ("Meds at six.", "0 18 *")):
        q.append(episodes.inbound(_teach_text(note), channel="tray"))
        _drain(q, _Teaching(learn_answer=_answer(schedules=[_a_schedule("Meds.", cron)])))

    written = _schedules_in(q)
    assert len({p["id"] for p in written}) == 2

    clock = schedule.Scheduler(q)
    clock.refresh()
    assert sorted(s.cron for s in clock.schedules) == ["0 18 *", "0 9 *"]
