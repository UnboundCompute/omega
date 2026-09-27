"""Wake condition (b) — the unprompted pass. DL-011, DL-061.

Every case here exists because of a specific way this could be wrong, and the
two that matter most are not about the rate limit at all:

- an unprompted event rendered as ``you: ...`` makes the judge answer a question
  nobody asked, on every single look, which is the notification firehose running
  at exactly the permitted rate;
- a day-budget check that cannot see this morning's nudge fails *open*, which is
  the one direction this limit must never fail.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from omega import episodes, notice, provider, turn
from omega.executor import Executor
from omega.memory import MemoryStore
from omega.queue import EventQueue

# Local *noon*, in the machine's own zone, for `test_derive.py:533`'s reason:
# the day limit is a statement about the clock on the wall, so a fixture pinned
# to UTC would straddle a local-day boundary on any machine that is not at UTC
# and the boundary tests would pass or fail by geography. Noon leaves room for
# every +/-6h offset below to stay inside one local day, while the two
# deliberate boundary cases (-1d2h, +1d) still land on a different one.
NOW = datetime(2026, 9, 27, 12, 0, tzinfo=datetime.now().astimezone().tzinfo)

# When the person last said something, far enough back that the conversation has
# gone quiet. Most cases below are about a limit other than the quiet gap, and
# they have to clear it to reach the one they are testing.
QUIET = NOW - timedelta(hours=2)


@pytest.fixture
def q(store: MemoryStore) -> EventQueue:
    return EventQueue(store)


def _iso(when: datetime) -> str:
    return when.isoformat()


def _looked(q: EventQueue, when: datetime, *, text: str = "what is open") -> int:
    return q.append(episodes.inbound(text, channel=notice.CHANNEL, at=_iso(when)))


def _spoke(q: EventQueue, for_seq: int, when: datetime) -> int:
    return q.append(
        episodes.completed(
            for_seq=for_seq, outcome="spoke", reply="a thing", at=_iso(when)
        )
    )


def _silent(q: EventQueue, for_seq: int, when: datetime) -> int:
    return q.append(
        episodes.completed(for_seq=for_seq, outcome="silent", at=_iso(when))
    )


# --- the firehose guard -----------------------------------------------------


def test_an_unprompted_event_is_not_rendered_as_the_person_talking() -> None:
    """The bug that would have made this ship as a firehose at the permitted rate.

    `_render_payload` rendered *every* inbound as ``you: <text>``, and the judge
    is told in as many words that "if the person wrote to you, they are talking
    to you: answer them" and that "SILENT is never the right answer to a
    question". An unprompted look wearing that prefix is therefore a message
    addressed to omega, so it would have spoken on every look and been silent on
    none — and the rate limit would have capped the damage at one a day while
    hiding the fact that the judgement never ran.
    """
    theirs = {"kind": episodes.MESSAGE_INBOUND, "text": "hi", "channel": "tray"}
    mine = {"kind": episodes.MESSAGE_INBOUND, "text": "hi", "channel": notice.CHANNEL}

    assert turn._render_payload(theirs) == "you: hi"
    rendered = turn._render_payload(mine)
    assert not rendered.startswith("you:")
    assert "nobody asked" in rendered
    assert "hi" in rendered


def test_the_judge_and_reply_prompts_both_name_the_unprompted_case() -> None:
    """DL-060's lesson, applied before it could bite again: a capability no
    prompt mentions is a function nobody calls. The judge decides whether to
    speak and the reply composes the words, so both have to know that this
    event is not a person asking."""
    assert "nobody asked" in turn._JUDGE_SYSTEM
    assert "nobody asked" in turn._REPLY_SYSTEM


# --- the two limits --------------------------------------------------------


def test_a_first_look_is_allowed(q: EventQueue) -> None:
    q.append(episodes.inbound("hello", channel="tray", at=_iso(QUIET)))
    assert notice.may_look(notice.standing(q.store), now=NOW) is True


def test_a_second_look_inside_the_hour_is_refused(q: EventQueue) -> None:
    seq = _looked(q, NOW - timedelta(minutes=20))
    _silent(q, seq, NOW - timedelta(minutes=20))
    assert notice.may_look(notice.standing(q.store), now=NOW) is False


def test_silence_spends_no_daily_budget(q: EventQueue) -> None:
    """DL-011's load-bearing consequence, as a rate-limit property rather than a
    logging one: a look that chose silence must not cost the day's nudge, or
    omega gets one *look* a day instead of one *nudge* and the pass can never
    learn to be quiet cheaply."""
    seq = _looked(q, NOW - timedelta(hours=3))
    _silent(q, seq, NOW - timedelta(hours=3))
    st = notice.standing(q.store)
    assert st.spoke_on == frozenset()
    assert notice.may_look(st, now=NOW) is True


def test_having_spoken_today_stops_even_looking(q: EventQueue) -> None:
    """Cheaper than looking and being forced quiet: no situation is built and no
    model is called."""
    seq = _looked(q, NOW - timedelta(hours=5))
    _spoke(q, seq, NOW - timedelta(hours=5))
    assert notice.may_look(notice.standing(q.store), now=NOW) is False


def test_the_budget_is_a_local_day_and_yesterday_does_not_spend_today(
    q: EventQueue,
) -> None:
    seq = _looked(q, NOW - timedelta(days=1, hours=2))
    _spoke(q, seq, NOW - timedelta(days=1, hours=2))
    assert notice.may_look(notice.standing(q.store), now=NOW) is True


def test_an_event_whose_turn_has_not_run_stops_another(q: EventQueue) -> None:
    """Two looks stacked on one another would both be judged against a world the
    first one had not finished changing."""
    _looked(q, NOW - timedelta(hours=4))
    st = notice.standing(q.store)
    assert st.outstanding == 1
    assert notice.may_look(st, now=NOW) is False


def test_a_blocked_turn_counts_as_finished(q: EventQueue) -> None:
    """Blocked is over and waiting on a person, not still sitting in the queue —
    so it must not wedge the pass shut forever."""
    seq = _looked(q, NOW - timedelta(hours=4))
    q.append(
        episodes.blocked(
            for_seq=seq, needs="may I?", at=_iso(NOW - timedelta(hours=4))
        )
    )
    st = notice.standing(q.store)
    assert st.outstanding == 0
    assert notice.may_look(st, now=NOW) is True


# --- undeterminable breaks closed ------------------------------------------


def test_an_unreadable_last_look_refuses_rather_than_allows() -> None:
    st = notice.Standing(looked_at="not a timestamp")
    assert notice.may_look(st, now=NOW) is False


def test_an_undeterminable_today_refuses() -> None:
    assert notice.local_date("not a timestamp") is None


def test_a_naive_stamp_is_read_as_utc_and_not_as_local() -> None:
    """`derive.local_hour`'s rule, restated for dates: the log writes UTC, so
    reading a naive stamp as local would shift every day boundary by the
    machine's offset."""
    assert notice.local_date("2026-09-27T15:00:00") == notice.local_date(
        "2026-09-27T15:00:00+00:00"
    )


def test_standing_scans_the_whole_log_because_resuming_would_fail_open(
    q: EventQueue,
) -> None:
    """The optimisation that was deliberately refused (DL-061). A fold resuming
    from a saved seq carries no `spoke_on` from before it, so it would conclude
    an already-spent day was unspent — the cheap version fails open on the one
    limit whose whole purpose is to fail closed. This pins the signature so the
    cursor cannot be reintroduced without reading the reason.
    """
    import inspect

    assert "since" not in inspect.signature(notice.standing).parameters

    seq = _looked(q, NOW - timedelta(hours=6))
    _spoke(q, seq, NOW - timedelta(hours=6))
    for i in range(30):  # plenty of later episodes to scan past
        q.append(episodes.inbound(f"msg {i}", channel="tray", at=_iso(QUIET)))
    assert notice.may_look(notice.standing(q.store), now=NOW) is False


# --- the situation text ----------------------------------------------------


def test_the_situation_says_nobody_asked_and_never_reads_as_a_request() -> None:
    text = notice.situation(now=NOW)
    assert "Nobody asked" in text
    assert "?" not in text


def test_the_situation_carries_open_work_and_standing_schedules() -> None:
    class _Block:
        needs = "may I fetch that page?"

    class _Sched:
        instruction = "ask kaushik for two numbers"

    text = notice.situation(now=NOW, blocks=[_Block()], schedules=[_Sched()])
    assert "may I fetch that page?" in text
    assert "ask kaushik for two numbers" in text


def test_the_situation_is_capped_like_the_prompt_it_is() -> None:
    class _Sched:
        instruction = "x" * 500

    text = notice.situation(now=NOW, schedules=[_Sched() for _ in range(40)])
    assert len(text) <= notice.MAX_L1_CHARS


def test_the_situation_does_not_restate_recent_conversation() -> None:
    """The turn's own recall step already renders it. Writing it twice spends the
    prompt twice to say one thing."""
    text = notice.situation(now=NOW, heard_at=_iso(NOW - timedelta(minutes=30)))
    assert "30 minutes ago" in text
    assert "Recent history" not in text


# --- the executor seam ----------------------------------------------------


def _fake(verdict: str) -> provider.FakeProvider:
    return provider.FakeProvider(
        {
            provider.JUDGE: lambda role, messages: verdict,
            provider.ACT: lambda role, messages: "ok",
        }
    )


def test_notice_on_an_empty_log_does_nothing(q: EventQueue) -> None:
    """As in `ingest` and `digest_usage`: the first thing in an empty log must
    not be omega talking to itself about a person it has not met."""
    ex = Executor(q, complete=_fake("SILENT").complete)
    ex.recover()
    assert ex.notice(now=NOW) is False
    assert q.head() == 0


def test_notice_appends_an_event_the_drain_then_turns_into_a_turn(
    q: EventQueue,
) -> None:
    """The whole point, end to end at the executor: `notice` starts a turn and
    does not run one, and the ordinary drain finishes it."""
    q.append(episodes.inbound("hello", channel="tray", at=_iso(QUIET)))
    ex = Executor(q, complete=_fake("SILENT").complete)
    ex.recover()
    ex.drain()

    assert ex.notice(now=NOW) is True
    raised = [
        p.payload
        for p in q.recent(q.head())
        if p.payload["kind"] == episodes.MESSAGE_INBOUND
        and p.payload["channel"] == notice.CHANNEL
    ]
    assert len(raised) == 1, "no unprompted event was raised"

    ex.drain()
    completed = [
        p.payload
        for p in q.recent(q.head())
        if p.payload["kind"] == episodes.TURN_COMPLETED
    ]
    assert completed[-1]["outcome"] == "silent", completed[-1]
    assert completed[-1]["reply"] is None


def test_notice_returns_true_for_a_look_even_when_the_turn_stays_silent(
    q: EventQueue,
) -> None:
    """`notice` reports whether it *looked*, never whether omega spoke. A caller
    that collapsed the two would count a successful silence as a failed pass,
    which is the regression DL-011 calls the single most important consequence of
    the time-wake."""
    q.append(episodes.inbound("hello", channel="tray", at=_iso(QUIET)))
    ex = Executor(q, complete=_fake("SILENT").complete)
    ex.recover()
    ex.drain()
    assert ex.notice(now=NOW) is True
    ex.drain()
    assert ex.notice(now=NOW + timedelta(hours=2)) is True


def test_notice_will_not_look_twice_before_its_own_turn_has_run(
    q: EventQueue,
) -> None:
    q.append(episodes.inbound("hello", channel="tray", at=_iso(QUIET)))
    ex = Executor(q, complete=_fake("SILENT").complete)
    ex.recover()
    ex.drain()
    assert ex.notice(now=NOW) is True
    assert ex.notice(now=NOW + timedelta(hours=9)) is False


def test_a_person_who_just_spoke_is_not_looked_at(q: EventQueue) -> None:
    """The quiet half of the gap (DL-061 amendment). "Time passed" means the
    conversation went quiet, not that the clock moved: the one moment a nudge is
    certainly unwelcome is while the person is still here, and a look taken a
    second after they spoke re-judges the situation the turn that just ran
    already saw. Found by the suite, not by design - the working pass put an
    extra turn into `test_cli` and `test_evals`, which is the same fact from the
    outside."""
    q.append(episodes.inbound("hello", channel="tray", at=_iso(NOW)))
    assert notice.may_look(notice.standing(q.store), now=NOW) is False
    assert (
        notice.may_look(notice.standing(q.store), now=NOW + timedelta(minutes=20))
        is False
    )
    assert (
        notice.may_look(notice.standing(q.store), now=NOW + timedelta(hours=2)) is True
    )


def test_an_unreadable_heard_stamp_breaks_closed(q: EventQueue) -> None:
    """Same direction as the unreadable look: an undeterminable quiet gap is not
    a gap that has elapsed."""
    q.append(episodes.inbound("hello", channel="tray", at="whenever"))
    st = notice.standing(q.store)
    assert st.heard_at == "whenever"
    assert notice.may_look(st, now=NOW) is False


def test_the_quiet_gap_is_not_confused_with_omegas_own_looks(q: EventQueue) -> None:
    """`heard_at` must track the *person*, not every inbound. If a look counted
    as being heard from, omega would keep resetting its own quiet clock and the
    gap would measure nothing."""
    q.append(episodes.inbound("hello", channel="tray", at=_iso(QUIET)))
    seq = _looked(q, NOW - timedelta(minutes=5))
    _silent(q, seq, NOW - timedelta(minutes=5))
    st = notice.standing(q.store)
    assert st.heard_at == _iso(QUIET), st.heard_at
    assert st.looked_at == _iso(NOW - timedelta(minutes=5))


def test_the_look_is_stamped_with_the_moment_it_was_taken(q: EventQueue) -> None:
    """Found by sabotage, not by the cases above: dropping `at=moment` from the
    raised event left all of them green, because the fixture day and the real
    day happened to agree on the afternoon this was written. `notice.standing`
    charges a spoken nudge to the day of the *look*, so an unstamped look is a
    day budget kept against the wall clock instead of against the clock the
    decision was made on - and a suite that only notices on the wrong date is
    the same geographic fragility as a UTC fixture, one axis over."""
    far = NOW - timedelta(days=200)
    q.append(episodes.inbound("hello", channel="tray", at=_iso(far - timedelta(hours=2))))
    ex = Executor(q, complete=_fake("SILENT").complete)
    ex.recover()
    ex.drain()
    assert ex.notice(now=far) is True

    looks = [
        p.payload
        for p in q.recent(q.head())
        if p.payload["kind"] == episodes.MESSAGE_INBOUND
        and p.payload["channel"] == notice.CHANNEL
    ]
    assert len(looks) == 1
    assert looks[0]["at"] == _iso(far), looks[0]["at"]
    assert notice.standing(q.store).looked_at == _iso(far)


def test_a_spoken_nudge_closes_the_day_at_the_executor(q: EventQueue) -> None:
    q.append(episodes.inbound("hello", channel="tray", at=_iso(QUIET)))
    ex = Executor(q, complete=_fake("SPEAK").complete)
    ex.recover()
    ex.drain()
    assert ex.notice(now=NOW) is True
    ex.drain()
    assert ex.notice(now=NOW + timedelta(hours=6)) is False
    assert ex.notice(now=NOW + timedelta(days=1)) is True
