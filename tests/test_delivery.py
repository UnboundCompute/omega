"""DL-084 — a promise is kept by delivery, not by firing.

Five things, each a way omega went quiet on the live store: a reminder retired
the moment it fired, so a failed turn lost it; their own message answered with
silence; one network blip failing a whole turn; an outage nobody was told
about; and "stop the flight prices watch" refused for want of an id. **No
socket, no key**: every provider here is scripted, and the one real
``OpenAIProvider`` is handed a stub client.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from omega import act, episodes, notice, provider, tools, turn
from omega.executor import Executor
from omega.memory import MemoryStore
from omega.queue import EventQueue
from omega.schedule import OWED_FOR, Scheduler
from omega.tools import TELL_LEAD, ToolBox, ToolError
from omega.turn import run_turn

T0 = datetime(2026, 10, 7, 12, 0, tzinfo=timezone.utc)
AT = T0.isoformat()


@pytest.fixture
def q(store: MemoryStore) -> EventQueue:
    return EventQueue(store)


def _payloads(q: EventQueue) -> list[dict]:
    return [episodes.decode(e.payload) for e in q.store.episodes_since(0)]


def _fires(q: EventQueue, sid: str) -> list[dict]:
    return [p for p in _payloads(q) if p.get("schedule_id") == sid]


def _tick(q: EventQueue, when: datetime) -> list[int]:
    """One tick from a fresh fold: every assertion below is about what the
    log alone says, which is what a restart rebuilds."""
    return Scheduler(q).tick(now=when)


def _ids(q: EventQueue) -> list[str]:
    """What a fresh fold of the log holds as running."""
    clock = Scheduler(q)
    clock.refresh()
    return [s.id for s in clock.schedules]


def _failed(q: EventQueue, seq: int, when: datetime, error: str = "ProviderError: boom") -> None:
    q.append(episodes.completed(for_seq=seq, outcome="failed", error=error, at=when.isoformat()))


def _spoke(q: EventQueue, seq: int, when: datetime) -> None:
    q.append(episodes.completed(for_seq=seq, outcome="spoke", reply="done", at=when.isoformat()))


def _reminder(q: EventQueue, sid: str = "r1") -> None:
    q.append(
        episodes.schedule_created(
            id=sid,
            instruction=tools.remind_instruction("call the bank"),
            every=120,
            once=True,
            at=AT,
        )
    )


# --- #1: discharged by delivery ---------------------------------------------


def test_a_failed_reminder_is_retried_at_1_5_15_minutes_then_every_tick(
    q: EventQueue,
) -> None:
    _reminder(q)
    slot = T0 + timedelta(minutes=2)
    [first] = _tick(q, slot)
    when = slot
    _failed(q, first, when)

    for wait in (1, 5, 15):
        assert _tick(q, when + timedelta(minutes=wait, seconds=-1)) == []
        when = when + timedelta(minutes=wait)
        [again] = _tick(q, when)
        assert q.at(again).payload["schedule_slot"] == slot.isoformat(), (
            "a retry is for the slot it owes, not a new one"
        )
        _failed(q, again, when)

    # After the third backoff: every tick.
    [again] = _tick(q, when)
    _failed(q, again, when)
    [again] = _tick(q, when + timedelta(seconds=30))
    _spoke(q, again, when + timedelta(seconds=30))

    assert len(_fires(q, "r1")) == 6
    later = Scheduler(q)
    assert later.tick(now=when + timedelta(hours=3)) == []
    assert later.schedules == [], "delivered, so retired"
    assert later.owed("r1") is None


def test_a_reminder_that_spoke_never_fires_again(q: EventQueue) -> None:
    _reminder(q)
    [seq] = _tick(q, T0 + timedelta(minutes=2))
    _spoke(q, seq, T0 + timedelta(minutes=2))
    for hours in (0, 1, 30):
        assert _tick(q, T0 + timedelta(minutes=3, hours=hours)) == []
    assert len(_fires(q, "r1")) == 1


def test_an_unfinished_fire_is_awaited_across_a_restart_never_doubled(
    q: EventQueue,
) -> None:
    _reminder(q)
    [seq] = _tick(q, T0 + timedelta(minutes=2))
    # No terminal yet: the turn is running, or the process died under it.
    assert _tick(q, T0 + timedelta(hours=2)) == []
    assert _ids(q) == ["r1"]

    # The restart closes it out as interrupted; that is a failure, so owed.
    q.claim(seq)
    Executor(q, complete=provider.FakeProvider({}).complete).recover()
    clock = Scheduler(q)
    clock.refresh()
    assert clock.owed("r1") is not None
    [again] = clock.tick(now=datetime.now(timezone.utc))
    assert q.at(again).payload["schedule_slot"] == (T0 + timedelta(minutes=2)).isoformat()


def test_a_blocked_fire_is_delivered(q: EventQueue) -> None:
    _reminder(q)
    [seq] = _tick(q, T0 + timedelta(minutes=2))
    q.append(episodes.blocked(for_seq=seq, needs="which bank?", at=AT))
    assert _tick(q, T0 + timedelta(hours=1)) == []
    assert _ids(q) == []


def test_after_a_day_of_failures_they_are_told_once_and_it_is_retired(
    q: EventQueue,
) -> None:
    _reminder(q)
    slot = T0 + timedelta(minutes=2)
    [seq] = _tick(q, slot)
    _failed(q, seq, slot, error="ProviderError: judge call failed: timed out")

    [tell] = _tick(q, slot + OWED_FOR)
    text = q.at(tell).payload["text"]
    assert text.startswith(TELL_LEAD)
    assert "your reminder (call the bank)" in text
    assert "timed out" in text, "the cause is named"
    assert "missed" in text

    assert _tick(q, slot + OWED_FOR + timedelta(minutes=30)) == []
    assert _ids(q) == []

    # The tell is delivered with no model call: it often reports that the
    # model could not be reached.
    q.claim(tell)
    fp = provider.FakeProvider({})
    result = run_turn(q, q.at(tell), complete=fp.complete, at=AT)
    assert result.outcome == "spoke"
    assert result.reply == text[len(TELL_LEAD):].strip()
    assert fp.calls == []


def test_a_failed_brief_slot_is_retried_and_then_waits_for_tomorrow(
    q: EventQueue,
) -> None:
    q.append(
        episodes.schedule_created(
            id="b1", instruction=tools.brief_instruction(), cron="0 9 *",
            at=datetime(2026, 10, 7, 8, 0, tzinfo=timezone.utc).isoformat(),
        )
    )
    nine = datetime(2026, 10, 7, 9, 0, tzinfo=timezone.utc)
    [seq] = _tick(q, nine)
    _failed(q, seq, nine)
    [again] = _tick(q, nine + timedelta(minutes=1))
    assert q.at(again).payload["schedule_slot"] == nine.isoformat()
    _spoke(q, again, nine + timedelta(minutes=2))

    assert _tick(q, nine + timedelta(hours=3)) == []
    assert _ids(q) == ["b1"], "a brief recurs"
    [tomorrow] = _tick(q, nine + timedelta(days=1))
    assert q.at(tomorrow).payload["schedule_slot"] == (nine + timedelta(days=1)).isoformat()


def test_a_watch_that_failed_is_not_owed(q: EventQueue) -> None:
    """Unchanged for what is not a promise: a watch looks again at its own
    interval, and a failed look is not chased."""
    q.append(episodes.schedule_created(id="w1", instruction="look", every=900, at=AT))
    [seq] = _tick(q, T0 + timedelta(minutes=15))
    _failed(q, seq, T0 + timedelta(minutes=15))
    clock = Scheduler(q)
    assert clock.tick(now=T0 + timedelta(minutes=16)) == []
    assert clock.owed("w1") is None
    assert len(clock.tick(now=T0 + timedelta(minutes=30))) == 1


def test_cancelling_an_owed_reminder_stops_the_retries(q: EventQueue) -> None:
    _reminder(q)
    [seq] = _tick(q, T0 + timedelta(minutes=2))
    _failed(q, seq, T0 + timedelta(minutes=2))
    q.append(episodes.schedule_cancelled(id="r1", at=AT))
    assert _tick(q, T0 + timedelta(minutes=10)) == []
    assert _tick(q, T0 + timedelta(days=2)) == []


# --- #2: their own message never ends silent ----------------------------------


def _put(q: EventQueue, text: str, channel: str):
    seq = q.append(episodes.inbound(text, channel=channel, at=AT))
    q.claim(seq)
    return q.at(seq)


@pytest.mark.parametrize("channel", ["tray", "cli", "discord"])
def test_a_silent_verdict_on_their_message_speaks(q: EventQueue, channel: str) -> None:
    fp = provider.FakeProvider({provider.JUDGE: "SILENT", provider.ACT: "Noted."})
    result = run_turn(q, _put(q, "keep checking and let me know", channel), complete=fp.complete, at=AT)
    assert result.outcome == "spoke" and result.reply == "Noted."
    record = q.at(result.record_seq).payload
    assert record["judged"].startswith("SILENT"), "what the judge said is kept"


@pytest.mark.parametrize("channel", ["self", "schedule"])
def test_silence_stays_a_success_where_nobody_asked(q: EventQueue, channel: str) -> None:
    fp = provider.FakeProvider({provider.JUDGE: "SILENT"})
    result = run_turn(q, _put(q, "anything?", channel), complete=fp.complete, at=AT)
    assert result.outcome == "silent"
    assert fp.calls_for(provider.ACT) == []


# --- #3: one retry for a blip, in our code ------------------------------------


class _Flaky:
    """A provider that fails its first ``fails`` calls with ``error``."""

    def __init__(self, error: Exception, fails: int = 1, answer: str = "SPEAK") -> None:
        self.error, self.fails, self.answer, self.calls = error, fails, answer, []

    def complete(self, role, messages, *args, **kwargs):
        self.calls.append(role)
        if len(self.calls) <= self.fails:
            raise self.error
        return provider.Response(text=self.answer, model="m", prompt_tokens=1, completion_tokens=1)


def _blip() -> provider.ProviderError:
    try:
        raise TimeoutError("read timed out")
    except TimeoutError as cause:
        try:
            raise provider.ProviderError("judge call to m failed: timed out") from cause
        except provider.ProviderError as exc:
            return exc


class _Status(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"Error code: {status_code}")
        self.status_code = status_code


def test_a_blip_is_retried_once_and_succeeds() -> None:
    flaky = _Flaky(_blip())
    slept: list[float] = []
    call = provider.retry_blips(flaky.complete, sleep=slept.append)
    assert call(provider.JUDGE, []).text == "SPEAK"
    assert flaky.calls == [provider.JUDGE, provider.JUDGE]
    lo, hi = provider.BLIP_PAUSE_SECONDS
    assert len(slept) == 1 and lo <= slept[0] <= hi, "jittered, once"


def test_a_second_blip_says_it_was_after_a_retry() -> None:
    flaky = _Flaky(_blip(), fails=5)
    call = provider.retry_blips(flaky.complete, sleep=lambda s: None)
    with pytest.raises(provider.ProviderError, match=r"after one retry"):
        call(provider.ACT, [])
    assert len(flaky.calls) == 2, "one retry, never more"


def test_a_4xx_is_an_answer_and_is_not_retried() -> None:
    try:
        raise provider.ProviderError("act call failed: 400") from _Status(400)
    except provider.ProviderError as exc:
        error = exc
    flaky = _Flaky(error)
    call = provider.retry_blips(flaky.complete, sleep=lambda s: pytest.fail("slept"))
    with pytest.raises(provider.ProviderError):
        call(provider.ACT, [])
    assert flaky.calls == [provider.ACT]
    assert provider.is_blip(error) is False


def test_only_judge_and_act_are_retried() -> None:
    flaky = _Flaky(_blip())
    call = provider.retry_blips(flaky.complete, sleep=lambda s: None)
    with pytest.raises(provider.ProviderError):
        call(provider.LEARN, [])
    assert flaky.calls == [provider.LEARN]


def test_the_real_provider_client_blips_through_the_retry(monkeypatch) -> None:
    """The real mapping: the client's timeout is wrapped in ProviderError, the
    wrapper sees through it, and the client itself never retried."""

    class APITimeoutError(Exception):
        pass

    seen = []

    def create(**kwargs):
        seen.append(kwargs)
        if len(seen) == 1:
            raise APITimeoutError("Request timed out.")
        return SimpleNamespace(
            model="m",
            status="completed",
            output=[SimpleNamespace(type="message", content=[SimpleNamespace(type="output_text", text="SPEAK")])],
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        )

    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    real = provider.OpenAIProvider(client=client, models={provider.JUDGE: "m"})
    call = provider.retry_blips(real.complete, sleep=lambda s: None)
    assert call(provider.JUDGE, [provider.user("x")]).text == "SPEAK"
    assert len(seen) == 2


def test_a_turn_records_its_retry_on_the_usage_row(q: EventQueue, monkeypatch) -> None:
    monkeypatch.setattr(provider, "BLIP_PAUSE_SECONDS", (0.0, 0.0))
    flaky = _Flaky(_blip(), answer="SPEAK")
    result = run_turn(q, _put(q, "hello", "tray"), complete=flaky.complete, at=AT)
    assert result.outcome == "spoke"
    usage = q.at(result.record_seq).payload["usage"]
    [judge] = [row for row in usage if row["role"] == provider.JUDGE]
    assert judge["retried"] == 1 and judge["calls"] == 1


def test_a_tool_that_times_out_is_not_retried(q: EventQueue, tmp_path, monkeypatch) -> None:
    """Tool calls never pass through the retry: a timed-out fetch is run
    once and comes back to the model as its result."""
    monkeypatch.setattr(provider, "BLIP_PAUSE_SECONDS", (0.0, 0.0))
    ran = []

    def fetch(url):
        ran.append(url)
        raise TimeoutError("read timed out")

    monkeypatch.setattr(tools, "fetch", fetch)
    root = tmp_path / "box"
    root.mkdir()
    box = ToolBox(store_root=root, append=q.append, for_seq=1)
    with pytest.raises(Exception):
        box.dispatch(box.classify(tools.FETCH, {"url": "https://example.com"}))
    assert ran == ["https://example.com"]


# --- #4: an outage is told, once ----------------------------------------------


NOW = datetime(2026, 9, 27, 12, 0, tzinfo=datetime.now().astimezone().tzinfo)


def _look(q: EventQueue, when: datetime, outcome: str) -> None:
    seq = q.append(episodes.inbound("what is open", channel=notice.CHANNEL, at=when.isoformat()))
    if outcome == "failed":
        _failed(q, seq, when, error="ProviderError: judge call failed: Connection error.")
    else:
        q.append(episodes.completed(for_seq=seq, outcome=outcome, at=when.isoformat()))


def _tells(q: EventQueue) -> list[str]:
    return [
        p["text"]
        for p in _payloads(q)
        if p.get("kind") == episodes.MESSAGE_INBOUND
        and str(p.get("text", "")).startswith(TELL_LEAD)
    ]


def test_three_failed_looks_are_told_once_and_a_success_resets(q: EventQueue) -> None:
    q.append(episodes.inbound("hello", channel="tray", at=(NOW - timedelta(days=1)).isoformat()))
    for h in (6, 5):
        _look(q, NOW - timedelta(hours=h), "failed")
    assert notice.owes_outage(notice.standing(q.store)) is False, "two is not three"
    _look(q, NOW - timedelta(hours=4), "failed")

    st = notice.standing(q.store)
    assert st.failed_looks == 3 and "Connection error" in (st.look_error or "")
    ex = Executor(q, complete=provider.FakeProvider({}).complete)
    ex.recover()
    assert ex.notice(now=NOW) is True
    [tell] = _tells(q)
    assert "last 3 looks failed" in tell and "Connection error" in tell

    # Delivered with no model call, like every tell.
    seq = q.head()
    q.claim(seq)
    fp = provider.FakeProvider({})
    assert run_turn(q, q.at(seq), complete=fp.complete, at=AT).outcome == "spoke"
    assert fp.calls == []

    # Still failing: not told again.
    _look(q, NOW + timedelta(hours=1), "failed")
    assert notice.owes_outage(notice.standing(q.store)) is False
    assert len(_tells(q)) == 1

    # A look that works ends the outage; the next one is told afresh.
    _look(q, NOW + timedelta(hours=2), "silent")
    assert notice.standing(q.store).failed_looks == 0
    for h in (3, 4, 5):
        _look(q, NOW + timedelta(hours=h), "failed")
    assert notice.owes_outage(notice.standing(q.store)) is True


# --- #5: unwatch by title, and the prompt rules -------------------------------


def _running(q: EventQueue):
    clock = Scheduler(q)
    clock.refresh()
    return tuple(clock.schedules)


def _watches(q: EventQueue) -> None:
    q.append(episodes.schedule_created(id="w1", instruction=tools.watch_instruction("flight prices to Goa"), every=900, at=AT))
    q.append(episodes.schedule_created(id="w2", instruction=tools.watch_instruction("the cricket score"), every=900, at=AT))


def _unwatch(q: EventQueue, tmp_path, what: str) -> str:
    root = tmp_path / "box"
    root.mkdir(exist_ok=True)
    box = ToolBox(store_root=root, append=q.append, for_seq=1, running=_running(q))
    return box.dispatch(box.classify(tools.UNWATCH, {"id": what}))


def test_unwatch_resolves_a_title_that_names_exactly_one(q: EventQueue, tmp_path) -> None:
    _watches(q)
    out = _unwatch(q, tmp_path, "Flight Prices")
    assert "flight prices to Goa" in out
    cancelled = [p["id"] for p in _payloads(q) if p["kind"] == episodes.SCHEDULE_CANCELLED]
    assert cancelled == ["w1"]


@pytest.mark.parametrize("what", ["watch", "weather"])
def test_unwatch_refuses_a_title_matching_two_or_none(q: EventQueue, tmp_path, what: str) -> None:
    _watches(q)
    with pytest.raises(ToolError, match=r"w1: .*w2: "):
        _unwatch(q, tmp_path, what)
    assert not [p for p in _payloads(q) if p["kind"] == episodes.SCHEDULE_CANCELLED]


def test_the_judge_sends_keep_checking_to_act() -> None:
    assert (
        "If they ask you to keep checking, monitor, or let them know about "
        "something, or to remind them of something later, choose ACT"
    ) in turn._JUDGE_SYSTEM


def test_the_judge_sends_what_do_you_remember_to_act_for_recall() -> None:
    assert "Asked what you know or remember about them, choose ACT" in turn._JUDGE_SYSTEM


def test_the_act_prompt_forbids_claiming_it_cannot_check_in_the_background() -> None:
    assert "Never tell them you cannot check in the background" in act._ACT_SYSTEM


def test_the_fetch_tool_steers_searching_to_web_search() -> None:
    [schema] = [s for s in tools.schemas() if s["function"]["name"] == tools.FETCH]
    assert "use web_search instead" in schema["function"]["description"]
