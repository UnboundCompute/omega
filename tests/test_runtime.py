"""The resident process — M1_SPEC.md §Q12, §1.6, §1.2; DL-011, DL-016.

Real threads and a real store, no mocks of either: the whole subject here is
what happens when a turn runs on one thread while a person waits on another, and
a fake of that tests the fake. What *is* faked is the model — every case runs
through ``provider.FakeProvider``, so nothing in this file touches the network or
needs a key.
"""

from __future__ import annotations

import hashlib
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence

import pytest

import invariants
from omega import episodes, projection, provider
from omega.channel import ChannelClient
from omega.executor import INTERRUPTED_ERROR
from omega.memory import EPISODES_FILENAME, MemoryStore
from omega.queue import EVENT_KINDS, EventQueue
from omega.runtime import (
    ClockFailed,
    DrainFailed,
    NotRunning,
    Runtime,
    Said,
    TurnTimeout,
)
from omega.turn import ActResult, TurnContext

AT = "2026-09-23T12:00:00+00:00"

# Short enough that a test does not sit on the default poll, long enough that
# the drain is not a spin loop while a case is holding it.
POLL = 0.005


def speaking(reply: str = "ok") -> provider.FakeProvider:
    return provider.FakeProvider(
        {
            provider.JUDGE: lambda role, messages: "SPEAK",
            provider.ACT: lambda role, messages: reply,
        }
    )


def silent() -> provider.FakeProvider:
    return provider.FakeProvider({provider.JUDGE: lambda role, messages: "SILENT"})


def runtime_at(path: Path, fake: provider.FakeProvider, **kwargs) -> Runtime:
    """A runtime with the listener off unless a case is about the listener.

    Off by default because a bound port is the one part of this that can collide
    with another test run, and most cases here have nothing to do with sockets.
    """
    kwargs.setdefault("listen", False)
    kwargs.setdefault("poll", POLL)
    # Off by default for the same reason the listener is: a running clock is a
    # third thread folding the log while these cases hold the drain, and nothing
    # here is about time. The clock's own cases turn it on explicitly.
    kwargs.setdefault("clock", False)
    return Runtime(path, complete=fake.complete, **kwargs)


def new_event(messages: Sequence[dict]) -> str:
    """The prompt's *New event* section, without the recalled history.

    Both prompts render recall above the new event, so a scripted answer that
    matched on the whole body would see every earlier message too — and would
    answer turn three with turn one's reply while still looking correct.
    """
    return messages[-1]["content"].split("New event:\n", 1)[1]


def until(condition, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.005)
    return False


def terminals(path: Path) -> list[dict]:
    """Every terminal record in the log, read with the runtime closed."""
    with MemoryStore.open(path) as store:
        q = EventQueue(store)
        return [
            p.payload for p in q.recent(q.head()) if episodes.is_terminal(p.payload)
        ]


def cursors(path: Path) -> tuple[int, int, int, list[str]]:
    """``(claimed, done, head)`` and the kinds sitting above ``done``.

    A clean stop leaves ``claimed == done`` — that is DL-016's property and what
    makes the next startup report clean. It does **not** leave them at ``head``,
    and asserting that it did would be asserting the wrong thing: the last
    episode written is the terminal record of the last turn, and ``say`` returns
    the instant that record is durable, so the stop usually arrives before the
    drain has skipped it. Skipping a record is bookkeeping that can never cause a
    turn; it costs one pass on the next open. What must never be left above the
    cursors is an *event*.
    """
    with MemoryStore.open(path) as store:
        q = EventQueue(store)
        head, claimed, done = q.head(), q.claimed(), q.done()
        above = [p.kind for p in q.recent(head) if p.seq > done] if head else []
    return claimed, done, head, above


def graded(path: Path) -> invariants.Check:
    with MemoryStore.open(path) as store:
        return invariants.check(EventQueue(store))


# --- green: a turn, end to end ----------------------------------------------


def test_a_line_typed_becomes_an_episode_and_comes_back_as_a_reply(
    store_dir: Path,
) -> None:
    fake = speaking("hi back")
    with runtime_at(store_dir, fake) as rt:
        assert rt.report.clean is True
        said = rt.say("hello omega")

        assert said.spoke is True
        assert said.reply == "hi back"
        assert said.seq == 1, "the inbound episode is the acknowledgement token"
        assert said.record_seq == 2, "its terminal record is the next episode"
        # `say` returns the moment the record is durable, which is *inside* the
        # drain's step -- so the counter is a moment behind it, not equal to it.
        assert until(lambda: rt.turns == 1)

    assert [t["outcome"] for t in terminals(store_dir)] == ["spoke"]
    assert graded(store_dir), str(graded(store_dir))


def test_the_turn_runs_on_the_drain_thread_and_not_in_say(store_dir: Path) -> None:
    """§1.6 — nothing may call the turn directly; the only way to cause one is
    to append an episode. The proof is which thread the model call happened on:
    if ``say`` ran the turn itself, the judge would see the caller's thread."""
    caller = threading.current_thread().name
    judged_on: list[str] = []

    def watch(role: str, messages: list) -> str:
        judged_on.append(threading.current_thread().name)
        return "SPEAK" if role == provider.JUDGE else "ok"

    fake = provider.FakeProvider({provider.JUDGE: watch, provider.ACT: watch})
    with runtime_at(store_dir, fake) as rt:
        rt.say("who ran this?")

    assert judged_on, "the turn must actually have run"
    assert caller not in judged_on
    assert set(judged_on) == {"omega-executor"}


def test_a_silent_verdict_is_a_success_and_not_a_failure(store_dir: Path) -> None:
    """DL-011 — silence is a first-class successful outcome. The two must be
    separately observable, or every later metric counts one as the other."""
    with runtime_at(store_dir, silent()) as rt:
        said = rt.say("fyi, the parcel arrived")

        assert said.silent is True
        assert said.failed is False
        assert said.spoke is False
        assert said.reply is None, "silence is a null reply, never an empty one"
        assert said.state == projection.COMPLETE
        assert said.error is None

    record = terminals(store_dir)[0]
    assert record["outcome"] == "silent" and record["reply"] is None


def test_several_turns_keep_their_order_and_their_answers(store_dir: Path) -> None:
    def answer(role: str, messages: list) -> str:
        if role == provider.JUDGE:
            return "SPEAK"
        # The reply prompt carries the event, so each answer can be tied to the
        # question that caused it -- a test that only counted replies would pass
        # on three copies of the same one.
        asked = new_event(messages)
        return next(w for w in ("first", "second", "third") if f"you: {w}" in asked)

    fake = provider.FakeProvider({provider.JUDGE: answer, provider.ACT: answer})
    with runtime_at(store_dir, fake) as rt:
        said = [rt.say(text) for text in ("first", "second", "third")]

    assert [s.reply for s in said] == ["first", "second", "third"]
    assert [s.seq for s in said] == sorted(s.seq for s in said)
    assert [s.record_seq > s.seq for s in said] == [True, True, True]
    assert graded(store_dir), str(graded(store_dir))


def test_the_listener_wakes_the_drain_rather_than_delivering_the_event(
    store_dir: Path,
) -> None:
    """§Q12 — one process, two threads, and one notification path. A client that
    speaks over the socket must reach the same drain as ``say``, and be served
    the same projection."""
    with runtime_at(store_dir, speaking("over the wire"), listen=True, port=0) as rt:
        assert rt.address is not None
        with ChannelClient(rt.address) as client:
            client.subscribe(0)
            ack = client.say("hello from the tray")

            assert until(lambda: rt.turns == 1), "the append must wake the drain"
            understood, complete = client.read(), client.read()

    assert understood["state"] == projection.UNDERSTOOD
    assert complete["state"] == projection.COMPLETE
    assert complete["for_seq"] == ack["seq"]
    assert complete["reply"] == "over the wire"
    assert graded(store_dir), str(graded(store_dir))


def test_the_blob_store_is_wired_and_lands_beside_the_log(
    store_dir: Path, tmp_path: Path
) -> None:
    """DL-027 — the runtime opens the blob store where the log is.

    Proven through the socket rather than through ``rt.blobs`` alone, because
    the thing that can be wrong is the *assembly*: a channel handed no store, or
    handed one rooted somewhere else, would pass every unit test in
    ``test_blobs`` and fail the first time a tray attached anything.
    """
    content = b"staged before the message exists"
    source = tmp_path / "staged.png"
    source.write_bytes(content)

    with runtime_at(store_dir, speaking("seen"), listen=True, port=0) as rt:
        assert rt.address is not None
        with ChannelClient(rt.address) as client:
            attached = client.attach(str(source))

        assert attached["blob"] == "sha256:" + hashlib.sha256(content).hexdigest()
        assert rt.blobs.root == store_dir / "blobs"
        assert rt.blobs.path_for(attached["blob"]).read_bytes() == content
        assert rt.turns == 0, "ingesting bytes is not an event and causes no turn"

    assert (store_dir / "blobs").is_dir()
    assert (store_dir / EPISODES_FILENAME).is_file()
    # No ``graded`` here, and that is the point: nothing was logged, so the
    # invariant check answers *undetermined* rather than pass — which is the
    # right answer and the wrong assertion to build a test on.
    assert cursors(store_dir)[2] == 0, "attach writes bytes, never episodes"


# --- red: the model, and the missing key ------------------------------------


def test_a_provider_outage_lands_as_failed_and_the_drain_keeps_going(
    store_dir: Path,
) -> None:
    """§2.4's spirit for the model rather than a tool: surface, never swallow —
    and never wedge. A failed turn still records, still advances DONE, and the
    next event is still drained."""
    calls: list[str] = []

    def flaky(role: str, messages: list) -> str:
        calls.append(role)
        if len([c for c in calls if c == provider.JUDGE]) == 1:
            raise provider.ProviderError("the model is down")
        return "SPEAK" if role == provider.JUDGE else "recovered"

    fake = provider.FakeProvider({provider.JUDGE: flaky, provider.ACT: flaky})
    with runtime_at(store_dir, fake) as rt:
        broke = rt.say("while it is down")
        worked = rt.say("and after it comes back")

        assert broke.failed is True
        assert broke.silent is False, "a failure is not a silence (DL-011)"
        assert broke.state == projection.FAILED
        assert "the model is down" in (broke.error or "")
        assert worked.spoke is True and worked.reply == "recovered"
        assert until(lambda: rt.turns == 2)
        assert rt.drain_error is None, "a failed turn must not kill the drain"

    outcomes = [t["outcome"] for t in terminals(store_dir)]
    assert outcomes == ["failed", "spoke"]
    assert graded(store_dir), str(graded(store_dir))


def test_using_a_runtime_after_it_stops_is_refused_by_name(store_dir: Path) -> None:
    rt = runtime_at(store_dir, speaking())
    rt.start()
    rt.say("one")
    rt.stop()
    rt.stop()  # idempotent

    with pytest.raises(NotRunning):
        rt.say("after the end")
    with pytest.raises(NotRunning):
        rt.queue.head()


def test_starting_twice_is_refused_rather_than_taking_the_lock_again(
    store_dir: Path,
) -> None:
    rt = runtime_at(store_dir, speaking())
    rt.start()
    try:
        with pytest.raises(NotRunning):
            rt.start()
    finally:
        rt.stop()


# --- yellow: stopping, and the threads -------------------------------------


def test_a_clean_stop_leaves_the_next_open_with_a_clean_report(
    store_dir: Path,
) -> None:
    """DL-016's restart test on the path a person actually takes. Ctrl-C is not
    a crash, so it must not manufacture the shape a crash makes."""
    with runtime_at(store_dir, speaking()) as rt:
        rt.say("one")
        rt.say("two")

    with runtime_at(store_dir, speaking()) as again:
        assert again.report.clean is True
        assert again.report.interrupted is None
        assert again.report.lines() == []
        assert again.report.claimed == again.report.done

    claimed, done, _head, above = cursors(store_dir)
    assert claimed == done, "a clean stop leaves nothing claimed"
    assert not [k for k in above if k in EVENT_KINDS], (
        "no event may be left above the cursors; only records may be"
    )
    assert len(terminals(store_dir)) == 2
    assert graded(store_dir), str(graded(store_dir))


def test_stopping_mid_turn_finishes_that_turn_before_letting_go(
    store_dir: Path,
) -> None:
    """The dangerous half of a clean stop: the drain is *inside* a turn when the
    stop arrives. Abandoning it would leave the turn claimed with no terminal
    record — which is precisely what §1.2 reserves for a real crash, and would
    make every ordinary quit look like one."""
    entered, release = threading.Event(), threading.Event()

    def slow(role: str, messages: list) -> str:
        if role == provider.JUDGE:
            entered.set()
            release.wait(5)
        return "SPEAK" if role == provider.JUDGE else "eventually"

    fake = provider.FakeProvider({provider.JUDGE: slow, provider.ACT: slow})
    rt = runtime_at(store_dir, fake)
    rt.start()
    try:
        # Appended, not said: `say` would block this thread on the same turn the
        # stop has to interrupt.
        seq = rt.append(episodes.inbound("the slow one", channel="test"))
        assert entered.wait(5), "the drain must have claimed the turn"
        # Let the turn finish a moment *after* stop() starts waiting, so the
        # wait is real rather than already satisfied.
        threading.Timer(0.05, release.set).start()
        rt.stop()
    finally:
        release.set()

    with runtime_at(store_dir, speaking()) as again:
        assert again.report.clean is True, "a mid-turn stop is still a clean stop"

    records = terminals(store_dir)
    assert len(records) == 1
    assert records[0]["for_seq"] == seq
    assert records[0]["outcome"] == "spoke", "the turn finished; it was not abandoned"
    assert records[0]["error"] != INTERRUPTED_ERROR
    assert graded(store_dir), str(graded(store_dir))


class Detonate(BaseException):
    """A bug that escapes the turn's own recording.

    Deliberately **not** an ``Exception``: ``run_turn`` catches those and records
    them as a failed turn, which is the path ``test_a_provider_outage...``
    covers. This stands in for the class that gets past even that — a defect in
    the loop itself — because the question here is what the *thread* does with
    it, not what the turn does.
    """


def detonating_act(ctx: TurnContext) -> ActResult:
    raise Detonate("the act step exploded")


def test_a_drain_thread_that_dies_is_surfaced_by_stop_not_swallowed(
    store_dir: Path,
) -> None:
    fake = provider.FakeProvider({provider.JUDGE: lambda role, messages: "ACT"})
    rt = runtime_at(store_dir, fake, act=detonating_act)
    rt.start()
    rt.append(episodes.inbound("go on then", channel="test"))

    assert until(lambda: rt.drain_error is not None), "the drain must have died"
    assert isinstance(rt.drain_error, Detonate)

    with pytest.raises(DrainFailed) as raised:
        rt.stop()
    assert isinstance(raised.value.__cause__, Detonate)
    assert "Detonate" in str(raised.value)

    # And the turn it died inside is visible as interrupted rather than lost —
    # the claim was durable before anything ran (§1.2).
    with runtime_at(store_dir, speaking()) as again:
        assert again.report.interrupted is not None
        assert "go on then" in again.report.interrupted.question
    assert terminals(store_dir)[0]["error"] == INTERRUPTED_ERROR


def test_a_dead_drain_fails_say_instead_of_leaving_it_waiting(
    store_dir: Path,
) -> None:
    """A caller parked on a turn nothing will ever run is the worst shape of
    this bug: the loop is dead and the only symptom is an assistant that has
    gone quiet. ``say`` must learn about it, not time out."""
    fake = provider.FakeProvider({provider.JUDGE: lambda role, messages: "ACT"})
    rt = runtime_at(store_dir, fake, act=detonating_act, turn_timeout=30.0)
    rt.start()
    try:
        started = time.monotonic()
        with pytest.raises(DrainFailed):
            rt.say("this one kills it")
        assert time.monotonic() - started < 10, "it must not have waited out the timeout"
    finally:
        with pytest.raises(DrainFailed):
            rt.stop()


def test_say_waits_for_its_own_turn_while_the_drain_is_busy_with_an_earlier_one(
    store_dir: Path,
) -> None:
    """The queue is serial, so a `say` issued while an earlier event is in flight
    is the *ordinary* case. It must match on ``for_seq`` and not return the
    record of the turn that happened to finish first."""
    entered, release = threading.Event(), threading.Event()

    def answer(role: str, messages: list) -> str:
        if "you: the slow one" in new_event(messages):
            if role == provider.JUDGE:
                entered.set()
                release.wait(5)
            return "SPEAK" if role == provider.JUDGE else "answered the slow one"
        return "SPEAK" if role == provider.JUDGE else "answered the second"

    fake = provider.FakeProvider({provider.JUDGE: answer, provider.ACT: answer})
    with runtime_at(store_dir, fake) as rt:
        first = rt.append(episodes.inbound("the slow one", channel="test"))
        assert entered.wait(5)
        threading.Timer(0.05, release.set).start()

        said = rt.say("the second")

        assert said.seq > first
        assert said.reply == "answered the second"
        assert said.record_seq > first

    records = {t["for_seq"]: t for t in terminals(store_dir)}
    assert records[first]["reply"] == "answered the slow one"
    assert graded(store_dir), str(graded(store_dir))


def test_giving_up_on_a_turn_does_not_cancel_it(store_dir: Path) -> None:
    """A timeout is a caller that stopped watching, not a turn that stopped
    running. The distinction matters because the opposite reading would make a
    slow turn look like a lost one, and the log would then disagree with the
    person."""
    release = threading.Event()

    def slow(role: str, messages: list) -> str:
        release.wait(5)
        return "SPEAK" if role == provider.JUDGE else "late but real"

    fake = provider.FakeProvider({provider.JUDGE: slow, provider.ACT: slow})
    with runtime_at(store_dir, fake) as rt:
        with pytest.raises(TurnTimeout) as raised:
            rt.say("take your time", timeout=0.05)
        assert raised.value.seq == 1
        release.set()
        assert until(lambda: rt.turns == 1)

    assert terminals(store_dir)[0]["reply"] == "late but real"
    assert graded(store_dir), str(graded(store_dir))


# --- the M1 violation metric ------------------------------------------------


def test_no_episode_is_processed_twice_and_no_claim_is_left_without_a_record(
    store_dir: Path,
) -> None:
    """M1_SPEC.md §"The M1 violation metric", across a stop/reopen cycle.

    The capability is *turns complete and land in the log*; this is the paired
    violation metric that must not regress. It is graded by ``invariants.check``,
    which is three-valued and fails closed — a log in which nothing was processed
    reports *undetermined*, and this case asserts a real ``pass``.

    The judge-call count is the "never twice" half said directly: four inbound
    events across two processes must be exactly four judgements.
    """
    judgements: list[int] = []

    def counting(role: str, messages: list) -> str:
        if role == provider.JUDGE:
            judgements.append(1)
            return "SPEAK"
        return "ok"

    first = provider.FakeProvider({provider.JUDGE: counting, provider.ACT: counting})
    with runtime_at(store_dir, first) as rt:
        for text in ("one", "two", "three"):
            rt.say(text)

    mid = graded(store_dir)
    assert mid.verdict == invariants.PASS, str(mid)
    assert mid.events == 3 and mid.terminals == 3

    second = provider.FakeProvider({provider.JUDGE: counting, provider.ACT: counting})
    with runtime_at(store_dir, second) as again:
        assert again.report.clean is True
        again.say("four")

    final = graded(store_dir)
    assert final.verdict == invariants.PASS, str(final)
    assert final.events == 4 and final.terminals == 4
    assert final.claimed == final.done
    assert not [k for k in cursors(store_dir)[3] if k in EVENT_KINDS]
    assert len(judgements) == 4, "one judgement per event, across both processes"


def test_the_grader_used_here_can_still_fail(store_dir: Path) -> None:
    """Its control. Every case above asserts the grader passes, which proves
    nothing unless the grader can fail — and unless an untouched store reports
    *undetermined* rather than a pass."""
    empty = graded(store_dir)
    assert empty.verdict == invariants.UNDETERMINED
    assert not empty, "undetermined must never read as a pass"


# --- Said: the shape the caller renders -------------------------------------


def test_said_is_built_from_the_projection_the_tray_also_reads(
    store_dir: Path,
) -> None:
    """Constraint: the CLI must see exactly what the tray sees. So ``Said`` is
    built from a ``projection.Update`` and nothing else — if the two ever
    disagreed, a bug would reproduce on one surface and not the other."""
    with runtime_at(store_dir, speaking("shared")) as rt:
        said = rt.say("hello")
        updates = list(projection.updates_since(rt.queue, said.seq))

    terminal = [u for u in updates if u.kind in episodes.TERMINAL_KINDS]
    assert len(terminal) == 1
    assert Said.from_update(said.seq, terminal[0]) == said
    assert said.state in projection.STATES, "a Said carries a real wire state"


# --- the heartbeat: omega acting without being asked (DL-035) ---------------


def every_minute(instruction: str = "check the thing", sid: str = "s1") -> dict:
    """A schedule already overdue by construction: created a day ago, so the
    first tick has a slot to discharge and the test does not wait a minute."""
    return episodes.schedule_created(
        id=sid,
        instruction=instruction,
        every=episodes.MIN_EVERY_SECONDS,
        at=(
            datetime.fromisoformat(AT) - timedelta(days=1)
        ).isoformat(),
    )


def test_a_schedule_fires_a_real_turn_with_nobody_typing(store_dir: Path) -> None:
    """DL-035's whole claim, end to end on the real threads: an episode nobody
    sent produces a reply nobody asked for. Everything else in this section is
    about the ways that can go wrong."""
    with runtime_at(store_dir, speaking("brief"), clock=False) as rt:
        rt.append(every_minute("write the morning brief"))

    with runtime_at(store_dir, speaking("brief"), clock=True, tick=0.05) as rt:
        assert until(lambda: rt.turns >= 1), "no turn ran from the clock alone"
        assert rt.fires == 1
        fired = [
            u
            for u in projection.updates_since(rt.queue, 0)
            if u.kind == episodes.MESSAGE_INBOUND
        ]

    assert [u.text for u in fired] == ["write the morning brief"]


def test_the_clock_appends_and_does_not_run_the_turn_itself(store_dir: Path) -> None:
    """The constraint `runtime.py` names in its own docstring. If the clock ran
    the turn, the model call would happen on the clock thread — so the thread
    name is the evidence, not the fact that a turn happened at all."""
    judged_on: list[str] = []

    def watch(role: str, messages: list) -> str:
        judged_on.append(threading.current_thread().name)
        return "SILENT"

    fake = provider.FakeProvider({provider.JUDGE: watch})
    with runtime_at(store_dir, fake, clock=False) as rt:
        rt.append(every_minute())

    with runtime_at(store_dir, fake, clock=True, tick=0.05) as rt:
        assert until(lambda: bool(judged_on))

    assert judged_on[0] == "omega-executor"
    assert "omega-clock" not in judged_on


def test_a_fire_is_indistinguishable_to_the_loop_from_a_typed_message(
    store_dir: Path,
) -> None:
    """The drain must need no knowledge of the clock. It gets that for free only
    because a fire is a `message.inbound`, so this asserts the kind rather than
    trusting the design note."""
    with runtime_at(store_dir, speaking("ok"), clock=False) as rt:
        rt.append(every_minute())

    with runtime_at(store_dir, speaking("ok"), clock=True, tick=0.05) as rt:
        assert until(lambda: rt.fires >= 1)
        seq = rt.queue.head()
        while not rt.queue.at(seq).payload.get("schedule_id"):
            seq -= 1
        pending = rt.queue.at(seq)

    assert pending.payload["kind"] == episodes.MESSAGE_INBOUND
    assert pending.payload["kind"] in EVENT_KINDS
    assert pending.is_event


def test_a_clock_with_nothing_scheduled_fires_nothing(store_dir: Path) -> None:
    """The control. A heartbeat that ticked something into existence on an empty
    log would make every other case here unfalsifiable."""
    with runtime_at(store_dir, speaking("x"), clock=True, tick=0.02) as rt:
        assert until(lambda: False, 0.3) is False  # let it tick
        assert rt.fires == 0
        assert rt.turns == 0
        assert rt.scheduler is not None
        assert rt.scheduler.schedules == []


def test_a_disabled_clock_leaves_no_scheduler_and_no_thread(store_dir: Path) -> None:
    before = {t.name for t in threading.enumerate()}
    with runtime_at(store_dir, speaking("x"), clock=False) as rt:
        rt.append(every_minute())
        assert rt.scheduler is None
        during = {t.name for t in threading.enumerate()} - before

    assert "omega-clock" not in during
    assert rt.fires == 0


def test_stopping_is_not_delayed_by_a_long_tick(store_dir: Path) -> None:
    """A tick measured in minutes must not become a floor on how long Ctrl-C
    takes — which it would if the clock slept instead of waiting on the stop."""
    rt = runtime_at(store_dir, speaking("x"), clock=True, tick=300.0)
    rt.start()
    started = time.monotonic()
    rt.stop()

    assert time.monotonic() - started < 5.0


def test_a_dead_clock_is_reported_and_does_not_take_the_loop_down(
    store_dir: Path,
) -> None:
    """The asymmetry `ClockFailed` exists to state. A broken clock must not stop
    omega answering, and must not stay invisible either — a heartbeat that died
    looks exactly like one with nothing to do."""
    rt = runtime_at(store_dir, speaking("still here"), clock=True, tick=0.02)
    rt.start()
    assert rt.scheduler is not None

    def boom() -> list[int]:
        raise RuntimeError("the clock broke")

    rt.scheduler.tick = boom  # type: ignore[method-assign]
    assert until(lambda: rt.clock_error is not None), "a dead clock said nothing"

    said = rt.say("are you still there")
    assert said.reply == "still here", "a dead clock must not stop the loop"

    with pytest.raises(ClockFailed):
        rt.stop()


def test_an_unparseable_cron_breaks_one_schedule_and_not_the_clock(
    store_dir: Path,
) -> None:
    """The schema checks `cron` is a string, not that it parses, so a bad one
    reaches the log. Before quarantining, `previous_match` raised on every tick
    forever and one mistyped field took the whole clock down with it."""
    with runtime_at(store_dir, speaking("fine"), clock=False) as rt:
        rt.append(
            episodes.schedule_created(id="bad", instruction="never", cron="0 99 *")
        )
        rt.append(every_minute("the good one", sid="good"))

    with runtime_at(store_dir, speaking("fine"), clock=True, tick=0.05) as rt:
        assert until(lambda: rt.fires >= 1), "the good schedule never fired"
        assert rt.clock_error is None, "one bad expression killed the heartbeat"
        assert rt.scheduler is not None
        assert list(rt.scheduler.broken) == ["bad"]
        assert [s.id for s in rt.scheduler.schedules] == ["good"]
