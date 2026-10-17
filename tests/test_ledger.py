"""The told-ledger (C1, DL-109): keys from code, admission, and the fold.

Every mail and event here is synthetic: example.test senders, made-up
repos and codes, no real subject lines.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from omega import calendar, episodes, inbox, ledger, tools
from omega.ledger import Ledger
from omega.queue import EventQueue

NOW = datetime(2026, 10, 12, 9, 0, tzinfo=timezone.utc)


def _mail(uid: int, *, sender: str = "Asha Example <asha@example.test>",
          subject: str = "lunch on thursday?", bulk: bool = False, held: bool = False,
          message_id: str = "", text: str = "", date: datetime | None = None) -> inbox.Mail:
    return inbox.Mail(
        sender=sender, subject=subject, date=date or NOW - timedelta(hours=1), snippet="",
        bulk=bulk, unread=True, uid=uid, held=held, message_id=message_id, text=text,
    )


def _ci(uid: int, run: int, *, repo: str = "argus", wf: str = "tests") -> inbox.Mail:
    return _mail(
        uid, sender="GitHub <notifications@github.com>", bulk=True,
        subject=f"[acme/{repo}] Run failed: {wf} - main (abc{run:04d})",
        message_id=f"<run-{run}@example.test>",
        text=f"see https://github.com/acme/{repo}/actions/runs/{9000 + run} for details",
    )


def _fold(payloads) -> ledger.State:
    return Ledger.fold(enumerate(payloads, start=1)).state()


def test_constants_match_their_owners() -> None:
    assert ledger._REMINDER_LEAD == tools.REMINDER_LEAD
    assert ledger.INBOX_SOURCE == tools.INBOX_SOURCE


# --- keys ---------------------------------------------------------------------


def test_mail_key_is_message_id_else_box_and_uid() -> None:
    assert ledger.mail_key(_mail(7, message_id="<a1@example.test>")) == "mail:a1@example.test"
    assert ledger.mail_key(_mail(7)) == "mail:personal#7"
    assert ledger.mail_key(_mail(7), "work") == "mail:work#7"
    odd = ledger.mail_key(_mail(7, message_id="<has space@example.test>"))
    assert odd.startswith("mail:h") and " " not in odd


@pytest.mark.parametrize("subject, key", [
    ("[acme/argus] Run failed: tests - main (abc1234)", "ci:argus/tests"),
    ("[acme/argus] Run failed: Lint And Build - feat/x (abc1234)", "ci:argus/lint-and-build"),
    ("[acme/argus] tests workflow run failed for main branch", "ci:argus/tests"),
])
def test_ci_key_from_the_subject(subject: str, key: str) -> None:
    ci = ledger.ci_of(_mail(1, sender="GitHub <notifications@github.com>", subject=subject))
    assert ci is not None and ci.key == key


@pytest.mark.parametrize("subject", [
    "Run failed: tests - main",                      # no repo
    "[acme/argus] Run succeeded: tests - main (a)",  # not a failure
    "Re: [acme/argus] Run failed: tests - main (a)",  # a reply is a person's mail
    "[acme/ar gus] Run failed: tests - main (a)",    # a name outside the charset
    "[acme/argus] Run failed: rm -rf $HOME; - main (a)",
])
def test_ci_rejects_what_is_not_a_failure_subject(subject: str) -> None:
    m = _mail(1, sender="GitHub <notifications@github.com>", subject=subject)
    assert ledger.ci_of(m) is None


def test_ci_only_from_github() -> None:
    m = _ci(1, 1)
    spoof = _mail(1, sender="x <ci@example.test>", subject=m.subject, text=m.text)
    assert ledger.ci_of(spoof) is None


def test_ci_run_url_only_for_the_repo_the_subject_names() -> None:
    assert ledger.ci_of(_ci(1, 3)).url == "https://github.com/acme/argus/actions/runs/9003"
    other = _mail(1, sender="notifications@github.com", subject="[acme/argus] Run failed: tests - main (a)",
                  text="https://github.com/evil/argus/actions/runs/1 https://github.com/acme/argus/actions/runs/2x")
    assert ledger.ci_of(other).url is None


def test_slack_key_from_the_message_link() -> None:
    m = _mail(1, sender="Slack <notification@slack.com>", bulk=True, subject="new message",
              text="open https://acme.slack.com/archives/C01ABC/p1760000000123456 to reply")
    assert ledger.slack_ts(m) == "1760000000.123456"
    assert ledger.slack_ts(_mail(1, text="https://acme.slack.com/archives/C01/p1760000000123456")) is None


def test_calendar_key_is_uid_at_start() -> None:
    o = calendar.Occurrence(start=NOW, end=NOW, all_day=False, summary="dentist", location="",
                            uid="ev-1@example.test")
    assert ledger.calendar_key(o) == "cal:ev-1@example.test@2026-10-12T09:00:00+00:00"


def test_uid_and_recurring_do_not_change_occurrence_equality() -> None:
    a = calendar.Occurrence(start=NOW, end=NOW, all_day=False, summary="x", location="", uid="1")
    b = calendar.Occurrence(start=NOW, end=NOW, all_day=False, summary="x", location="",
                            uid="2", recurring=True)
    assert a == b and hash(a) == hash(b)


# --- admission ----------------------------------------------------------------


def test_admission_person_ci_travel_open_keys_bulk_is_one_aggregate() -> None:
    mails = [
        _mail(1, message_id="<p1@example.test>"),
        _ci(2, 1),
        _mail(3, sender="Air Example <noreply@air.example.test>", bulk=True,
              subject="Booking confirmed: PNR ZX81QK"),
        _mail(4, sender="News <n@example.test>", bulk=True, subject="weekly digest"),
        _mail(5, sender="Shop <s@example.test>", bulk=True, subject="sale"),
        _mail(6, held=True, subject="your sign-in code"),
    ]
    found = ledger.mail_observations(mails, ledger.State(), NOW)
    by_source = {p["source"]: p for p in found}
    assert set(by_source) == {"mail", "ci", "travel", "bulk"}
    assert by_source["mail"]["key"] == "mail:p1@example.test"
    assert by_source["ci"]["key"] == "ci:argus/tests"
    assert by_source["ci"]["url"] == "https://github.com/acme/argus/actions/runs/9001"
    assert by_source["bulk"]["key"] is None and len(by_source["bulk"]["refs"]) == 2
    assert all(p["untrusted"] == tools.INBOX_SOURCE for p in found if p["key"])
    # the held mail is not read, anywhere
    assert not any("sign-in" in p["what"] for p in found)
    for p in found:
        assert episodes.decode(episodes.encode(p)) == p


def test_the_same_reading_twice_adds_nothing_and_a_second_failure_recurs(store) -> None:
    q = EventQueue(store)
    assert ledger.record_mails(q, [_ci(1, 1), _mail(2)], NOW) == 2
    assert ledger.record_mails(q, [_ci(1, 1), _mail(2)], NOW) == 0
    assert ledger.record_mails(q, [_ci(1, 1), _ci(3, 2)], NOW + timedelta(hours=1)) == 1
    item = Ledger.rebuild(store).state().get("ci:argus/tests")
    assert item.recurrence == 2 and not item.closed


def test_bulk_seen_once_is_not_aggregated_again(store) -> None:
    q = EventQueue(store)
    news = _mail(4, bulk=True, subject="digest", message_id="<d1@example.test>")
    assert ledger.record_mails(q, [news], NOW) == 1
    assert ledger.record_mails(q, [news], NOW) == 0


def test_events_admit_only_non_recurring(store) -> None:
    q = EventQueue(store)
    one = calendar.Occurrence(start=NOW + timedelta(hours=5), end=NOW + timedelta(hours=6),
                              all_day=False, summary="flight ZX 812", location="T2", uid="e1")
    weekly = calendar.Occurrence(start=NOW + timedelta(hours=2), end=NOW + timedelta(hours=3),
                                 all_day=False, summary="standup", location="", uid="e2",
                                 recurring=True)
    assert ledger.record_events(q, [one, weekly], NOW) == 1
    assert ledger.record_events(q, [one, weekly], NOW) == 0
    item = Ledger.rebuild(store).state().get(ledger.calendar_key(one))
    assert item.deadline == one.start.isoformat() and item.untrusted == ()


# --- the fold -----------------------------------------------------------------


def _obs(key: str, ref: str, *, what: str = "mail from Asha: lunch", at: datetime = NOW, **kw):
    return episodes.observation_recorded(key=key, source="mail", what=what, ref=ref,
                                         at=at.isoformat(), **kw)


def test_told_and_considered_and_unkeyed_sends() -> None:
    state = _fold([
        _obs("mail:a", "mail:a"),                                         # 1
        episodes.inbound("look", channel="self", keys=["mail:a"]),        # 2
        episodes.completed(for_seq=2, outcome="spoke", reply="Asha asked about lunch."),  # 3
        episodes.inbound("look", channel="self"),                         # 4
        episodes.completed(for_seq=4, outcome="spoke", reply="disk is full"),  # 5
        episodes.inbound("hi", channel="cli"),                            # 6
        episodes.completed(for_seq=6, outcome="spoke", reply="hello"),    # 7
    ])
    a = state.get("mail:a")
    assert a.told_seq == 2 and a.told_text == "Asha asked about lunch." and not a.fresh
    # unprompted sends count; a prompted reply is not an unprompted send
    assert (state.sends, state.unkeyed, state.repeats) == (2, 1, 0)


def test_a_new_observation_makes_a_told_key_fresh_again_and_a_repeat_counts() -> None:
    state = _fold([
        _obs("ci:argus/tests", "r1"),                                       # 1
        episodes.inbound("x", channel="self", keys=["ci:argus/tests"]),     # 2
        episodes.completed(for_seq=2, outcome="spoke", reply="CI failed."),  # 3
        episodes.inbound("x", channel="self", keys=["ci:argus/tests"]),     # 4
        episodes.completed(for_seq=4, outcome="spoke", reply="CI failed."),  # 5: a repeat
        _obs("ci:argus/tests", "r2"),                                       # 6
    ])
    item = state.get("ci:argus/tests")
    assert state.repeats == 1
    assert item.fresh and item.recurrence == 2
    assert state.repeats_told(["ci:argus/tests"], "CI failed.") is False  # r2 is new


def test_repeats_told_needs_same_words_and_nothing_new() -> None:
    state = _fold([
        _obs("page:p1", "p1"),
        episodes.inbound("x", channel="schedule", keys=["page:p1"]),
        episodes.completed(for_seq=2, outcome="spoke", reply="The page changed."),
    ])
    assert state.repeats_told(["page:p1"], "the page  changed")
    assert not state.repeats_told(["page:p1"], "The page changed again.")
    assert not state.repeats_told([], "anything")
    assert not state.repeats_told(["page:other"], "The page changed.")


def test_considered_is_not_told() -> None:
    state = _fold([
        _obs("mail:a", "mail:a"),
        episodes.inbound("x", channel="self", keys=["mail:a"]),
        episodes.completed(for_seq=2, outcome="silent", judged="nothing to say"),
    ])
    a = state.get("mail:a")
    assert a.told_seq == 0 and a.considered_seq == 2 and not a.fresh
    assert state.sends == 0


def _reminder(sid: str, what: str, *, due_in: int, at: datetime = NOW):
    return episodes.schedule_created(id=sid, instruction=tools.REMINDER_LEAD + what,
                                     every=due_in, once=True, at=at.isoformat())


def test_a_reminder_opens_a_key_and_its_fire_or_cancel_closes_it() -> None:
    state = _fold([
        _reminder("r9-aa", "leave for flight ZX812", due_in=3600),      # 1
        episodes.inbound(tools.REMINDER_LEAD + "x", channel="schedule", schedule_id="r9-aa"),  # 2
        episodes.completed(for_seq=2, outcome="spoke", reply="Time to leave."),  # 3
        _reminder("pr4", "call the bank", due_in=7200),                 # 4
        episodes.schedule_cancelled(id="pr4"),                          # 5
    ])
    assert state.get("reminder:r9-aa").closed
    assert state.get("reminder:pr4").closed


def test_a_reminder_moved_reopens_with_its_new_deadline() -> None:
    state = _fold([
        _reminder("pr4", "call the bank", due_in=3600),
        episodes.schedule_cancelled(id="pr4"),
        _reminder("pr4", "call the bank", due_in=7200),
    ])
    item = state.get("reminder:pr4")
    assert not item.closed and item.deadline == (NOW + timedelta(seconds=7200)).isoformat()


def test_a_failed_fire_does_not_close_the_reminder() -> None:
    state = _fold([
        _reminder("pr1", "x", due_in=60),
        episodes.inbound("x", channel="schedule", schedule_id="pr1"),
        episodes.completed(for_seq=2, outcome="failed", error="boom"),
    ])
    assert not state.get("reminder:pr1").closed


def test_reminder_absorbs_what_it_covers_and_nothing_else() -> None:
    state = _fold([
        _reminder("pr1", "leave for the airport, flight ZX812", due_in=3 * 3600),
        _obs("mail:t1", "mail:t1", what="travel mail from Air: your flight ZX812 web check-in"),
        _obs("mail:a", "mail:a", what="mail from Asha: lunch on thursday?"),
    ])
    waiting = {i.key for i in state.waiting(NOW)}
    assert waiting == {"mail:a"}
    assert state.absorbed(state.get("mail:t1"), NOW).key == "reminder:pr1"
    assert state.covered_by(state.get("reminder:pr1")) == ["mail:t1"]
    # a reminder further out than the window absorbs nothing yet
    far = _fold([_reminder("pr1", "flight ZX812", due_in=30 * 3600),
                 _obs("mail:t1", "mail:t1", what="flight ZX812 check-in")])
    assert {i.key for i in far.waiting(NOW)} == {"mail:t1"}


def test_waiting_skips_stale_and_reminders() -> None:
    state = _fold([
        _obs("mail:old", "mail:old", when=(NOW - timedelta(days=3)).isoformat()),
        _obs("mail:new", "mail:new"),
        _reminder("pr1", "water the plants", due_in=60),
    ])
    assert [i.key for i in state.waiting(NOW)] == ["mail:new"]


def test_stale_draft() -> None:
    state = _fold([
        _obs("mail:a", "mail:a"),                                         # 1
        episodes.inbound("x", channel="self", keys=["mail:a"]),           # 2
        episodes.completed(for_seq=2, outcome="spoke", reply="told"),    # 3
        _reminder("pr1", "x", due_in=60),                                 # 4
        episodes.schedule_cancelled(id="pr1"),                            # 5
    ])
    assert state.stale(["mail:a"], 1) == "they were told about it after it was drafted"
    assert state.stale(["mail:a"], 2) is None
    assert state.stale(["reminder:pr1"], 1) == "its thread closed before it went"
    assert state.stale([], 1) is None
    assert state.stale(["mail:unknown"], 1) is None


def test_omega_authored_reminders() -> None:
    def made(sid: str) -> dict:
        return episodes.schedule_created(
            id=sid, instruction=tools.remind_instruction("x"), every=60, once=True,
        )

    view = Ledger.fold(enumerate([
        episodes.inbound("look", channel="self"),   # 1: omega's own turn
        episodes.inbound("remind me", channel="cli"),  # 2: theirs
        made("r1-abc"), made("r2-abc"),
        # 5: a fire whose seq the id below happens to name; set before it.
        episodes.inbound("a fire", channel="schedule", schedule_id="r5-abc"),
    ], start=1))
    assert view.authored("r1-abc")
    assert not view.authored("r2-abc")
    assert not view.authored("r5-abc")   # not set at all after seq 5
    assert not view.authored("r1-zzz")   # never set: no reminder to own
    assert not view.authored("pr1")      # a promise reminder is set on their yes
    assert not view.authored("weekly")


def test_rebuild_equals_incremental(store) -> None:
    q = EventQueue(store)
    ledger.record_mails(q, [_ci(1, 1), _mail(2)], NOW)
    first = Ledger.rebuild(store)
    ledger.record_mails(q, [_ci(3, 2)], NOW)
    assert first.advance(store).state() == Ledger.rebuild(store).state()


def test_one_reading_reaches_every_writer_even_when_one_fails() -> None:
    from omega import __main__ as main

    got: list = []

    def broken(_):
        raise RuntimeError("down")

    main._each(broken, got.extend)([1, 2])
    assert got == [1, 2]
