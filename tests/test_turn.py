"""One turn, six steps — M1_SPEC.md §1.4, §1.5; DL-011, DL-024.

The assertion this whole file is built around: **silence is a successful
outcome and empty content is a failure, and the log can always tell them
apart.** Every case here runs against `provider.FakeProvider` — no network, no
key, no spend — which is what makes "the judge chose silence" assertable at all.
"""

from __future__ import annotations

import pytest

from omega import episodes, provider, turn
from omega.memory import MemoryStore, WriteKeyConflict
from omega.queue import EventQueue
from omega.turn import (
    ACT_THEN_SPEAK,
    SPEAK,
    STAY_SILENT,
    ActResult,
    JudgeUndecided,
    NotAnEvent,
    TurnContext,
    Verdict,
    parse_verdict,
    perceive,
    recall,
    run_turn,
    write_memory,
)

AT = "2026-09-23T12:00:00+00:00"


@pytest.fixture
def q(store: MemoryStore) -> EventQueue:
    return EventQueue(store)


def put(q: EventQueue, text: str = "what's on my plate?"):
    """Enqueue an inbound event and claim it, exactly as the drain would."""
    seq = q.append(episodes.inbound(text, channel="tray", at=AT))
    q.claim(seq)
    return q.at(seq)


def fake(judge: object = "SPEAK", act: object = "here you go") -> provider.FakeProvider:
    return provider.FakeProvider({provider.JUDGE: judge, provider.ACT: act})


def last_payload(q: EventQueue) -> dict:
    return q.at(q.head()).payload


# --- green: the three ways a turn ends well ---------------------------------


def test_a_speaking_turn_records_what_it_said(q: EventQueue) -> None:
    p = put(q, "summarise my day")
    fp = fake(judge="SPEAK", act="Three things are open.")

    result = run_turn(q, p, complete=fp.complete, at=AT)

    assert result.outcome == "spoke" and result.spoke
    assert result.reply == "Three things are open."
    record = last_payload(q)
    assert record["kind"] == episodes.TURN_COMPLETED
    assert record["outcome"] == "spoke"
    assert record["reply"] == "Three things are open."
    assert record["for_seq"] == p.seq
    assert record["error"] is None
    # Both roles were used, once each, and the judge went first.
    assert [role for role, _ in fp.calls] == [provider.JUDGE, provider.ACT]


def test_a_silent_turn_is_a_success_and_costs_no_reply_call(q: EventQueue) -> None:
    """DL-011's load-bearing property. Silence is an outcome, not an absence:
    the record exists, ``reply`` is null rather than empty, and the expensive
    role is never called."""
    p = put(q, "fyi, moved the meeting")
    fp = fake(judge="SILENT", act=["should never be used"])

    result = run_turn(q, p, complete=fp.complete, at=AT)

    assert result.outcome == "silent" and result.silent
    assert result.reply is None
    assert result.failed is False
    record = last_payload(q)
    assert record["kind"] == episodes.TURN_COMPLETED
    assert record["outcome"] == "silent"
    assert record["reply"] is None
    assert record["error"] is None
    assert fp.calls_for(provider.ACT) == [], "silence must not pay for the act model"


def test_act_then_speak_runs_the_act_step_and_records_its_tools(q: EventQueue) -> None:
    p = put(q, "check the build")
    fp = fake(judge="ACT", act="The build is green.")

    def act(ctx: TurnContext) -> ActResult:
        assert ctx.seq == p.seq
        assert ctx.event["text"] == "check the build"
        return ActResult(tools=("shell",), stop_reason="done, verified")

    result = run_turn(q, p, complete=fp.complete, act=act, at=AT)

    assert result.outcome == "spoke"
    assert result.tools == ("shell",)
    assert result.verdict is not None and result.verdict.choice == ACT_THEN_SPEAK
    assert last_payload(q)["tools"] == ["shell"]


def test_the_answer_the_act_loop_composed_is_the_answer_the_person_gets(
    q: EventQueue,
) -> None:
    """The sub-loop saw every tool result; nothing downstream did.

    Measured against a real model before this existed: the loop wrote the file,
    read it back and answered "the number 7 has been successfully written";
    :func:`reply` threw that away, re-asked a model holding tool *names* and no
    results, and omega told the person "I can't perform the action right now"
    about work it had just finished. Composing again is not a second opinion,
    it is a first guess by the only party that did not watch.
    """
    p = put(q, "write 7 into count.txt")
    fp = fake(judge="ACT", act=["a blind recomposition that must not happen"])
    answered = "The number 7 was written and read back."

    def act(ctx: TurnContext) -> ActResult:
        return ActResult(
            tools=("write_file", "read_file"),
            stop_reason="the model stopped asking for tools",
            text=answered,
        )

    result = run_turn(q, p, complete=fp.complete, act=act, at=AT)

    assert result.outcome == "spoke"
    assert result.reply == answered
    assert last_payload(q)["reply"] == answered
    assert fp.calls_for(provider.ACT) == [], "the loop already answered; don't pay twice"


def test_a_loop_that_ended_without_an_answer_still_gets_one_composed(
    q: EventQueue,
) -> None:
    """The control. A stall, a block or a cap leaves ``text`` unset, and those
    turns still owe the person a reply — so the early return has to key on an
    answer actually being there, not on the act step having run."""
    p = put(q, "check the build")
    fp = fake(judge="ACT", act="I stopped after eight passes without finishing.")

    def act(ctx: TurnContext) -> ActResult:
        return ActResult(tools=("shell",), stop_reason="pass cap reached")

    result = run_turn(q, p, complete=fp.complete, act=act, at=AT)

    assert result.reply == "I stopped after eight passes without finishing."
    assert len(fp.calls_for(provider.ACT)) == 1


def test_a_blocked_turn_is_its_own_kind_not_a_failure(q: EventQueue) -> None:
    """§2.1 — stopping to ask is correct behaviour. Folding it into `failed`
    would make the tray show a stall as an error and lose the distinction the
    whole initiative feature rests on."""
    p = put(q, "book the flight")
    fp = fake(judge="ACT", act=["should never be used"])

    def act(ctx: TurnContext) -> ActResult:
        return ActResult(blocked_on="which airport?", stop_reason="not moving")

    result = run_turn(q, p, complete=fp.complete, act=act, at=AT)

    assert result.outcome == "blocked" and result.blocked
    assert result.failed is False
    assert result.needs == "which airport?"
    record = last_payload(q)
    assert record["kind"] == episodes.TURN_BLOCKED
    assert record["needs"] == "which airport?"
    assert record["for_seq"] == p.seq
    assert fp.calls_for(provider.ACT) == []


def test_a_turn_can_run_over_a_detached_work_event(q: EventQueue) -> None:
    """§1.5 — long work re-enters as an ordinary enqueued event, so it is a
    turn like any other and not a second entry path."""
    seq = q.append(episodes.work_finished(for_seq=1, summary="index rebuilt", at=AT))
    q.claim(seq)
    fp = fake(judge="SPEAK", act="Index is rebuilt.")

    result = run_turn(q, q.at(seq), complete=fp.complete, at=AT)

    assert result.outcome == "spoke"
    assert last_payload(q)["for_seq"] == seq


# --- green: the steps, separately -------------------------------------------


def test_perceive_accepts_events_and_refuses_records(q: EventQueue) -> None:
    p = put(q)
    assert perceive(p)["text"] == "what's on my plate?"

    record_seq = q.append(
        episodes.completed(for_seq=p.seq, outcome="silent", at=AT),
        episodes.turn_write_key(p.seq),
    )
    with pytest.raises(NotAnEvent):
        perceive(q.at(record_seq))


def test_recall_is_the_last_n_newest_last_and_excludes_the_event(q: EventQueue) -> None:
    for i in range(6):
        q.append(episodes.inbound(f"m{i}", channel="tray", at=AT))
    current = q.append(episodes.inbound("now", channel="tray", at=AT))

    got = recall(q, before=current, n=3)
    assert [p.payload["text"] for p in got] == ["m3", "m4", "m5"]
    assert current not in [p.seq for p in got]

    assert recall(q, before=1, n=40) == []


def test_write_memory_uses_the_turn_write_key(q: EventQueue) -> None:
    p = put(q)
    record = write_memory(q, seq=p.seq, outcome="silent", at=AT)
    assert q.at(record).write_key == episodes.turn_write_key(p.seq)
    assert q.at(record).write_key == f"turn:{p.seq}"


def test_run_turn_does_not_advance_done(q: EventQueue) -> None:
    """§1.2 — the record lands first, the cursor moves after, and the two are
    different jobs. run_turn owning DONE would put the crash window on the
    wrong side of the write."""
    p = put(q)
    run_turn(q, p, complete=fake(judge="SILENT").complete, at=AT)
    assert q.claimed() == p.seq
    assert q.done() == 0


@pytest.mark.parametrize(
    "answer,expected",
    [
        ("SPEAK", SPEAK),
        ("speak", SPEAK),
        ("SPEAK.", SPEAK),
        ("  speak  ", SPEAK),
        ("**SPEAK**", SPEAK),
        ("reply", SPEAK),
        ("ACT", ACT_THEN_SPEAK),
        ("act_then_speak", ACT_THEN_SPEAK),
        ("SILENT", STAY_SILENT),
        ("stay_silent", STAY_SILENT),
        ("nothing", STAY_SILENT),
        ("SILENT - nothing to add", STAY_SILENT),
    ],
)
def test_parse_verdict_accepts_the_answers_the_prompt_asks_for(
    answer: str, expected: str
) -> None:
    assert parse_verdict(answer).choice == expected


@pytest.mark.parametrize(
    "answer",
    [
        "",
        "   ",
        "\n\n",
        "I think we should stay silent",  # first-word rule: "not silent" traps
        "not silent",
        "maybe",
        "acknowledge",  # a prefix rule would read this as ACT
        "{'choice': 'speak'}",
        "1",
    ],
)
def test_parse_verdict_refuses_everything_else(answer: str) -> None:
    """Undecided is never defaulted. Reading an unparseable answer as `speak`
    makes silence unreachable by accident; reading it as `silent` turns a broken
    judge into a quiet one, which DL-024 says no other test would catch."""
    with pytest.raises(JudgeUndecided):
        parse_verdict(answer)


def test_parse_verdict_refuses_a_non_string() -> None:
    with pytest.raises(JudgeUndecided):
        parse_verdict(None)  # type: ignore[arg-type]


# --- red: the ways a turn ends badly, each still ending ---------------------


def test_a_provider_outage_fails_the_turn_and_records_why(q: EventQueue) -> None:
    """A turn that cannot reach the model still ends. The alternative leaves
    DONE stuck behind CLAIMED forever, which stops the single consumer."""
    p = put(q)
    dead = provider.FakeProvider({})  # nothing scripted: every call raises

    result = run_turn(q, p, complete=dead.complete, at=AT)

    assert result.outcome == "failed" and result.failed
    assert result.reply is None
    assert "ProviderError" in (result.error or "")
    record = last_payload(q)
    assert record["kind"] == episodes.TURN_COMPLETED
    assert record["outcome"] == "failed"
    assert record["error"], "a failed turn with no error is a silent fault"
    assert record["reply"] is None


def test_an_outage_in_the_reply_call_still_ends_the_turn(q: EventQueue) -> None:
    """The judge succeeded and the reply model died — the half-way failure."""
    p = put(q)
    half_dead = provider.FakeProvider({provider.JUDGE: "SPEAK"})

    result = run_turn(q, p, complete=half_dead.complete, at=AT)

    assert result.outcome == "failed"
    assert result.verdict is not None and result.verdict.choice == SPEAK
    assert last_payload(q)["outcome"] == "failed"


def test_an_unparseable_verdict_fails_the_turn_rather_than_guessing(
    q: EventQueue,
) -> None:
    p = put(q)
    fp = fake(judge="I'd rather not say", act=["should never be used"])

    result = run_turn(q, p, complete=fp.complete, at=AT)

    assert result.outcome == "failed"
    assert "JudgeUndecided" in (result.error or "")
    assert fp.calls_for(provider.ACT) == []
    # The three outcomes it must NOT have become.
    assert result.outcome not in ("silent", "spoke", "blocked")


def test_empty_model_content_is_a_failure_never_silence(q: EventQueue) -> None:
    """DL-011, the sharpest edge in the milestone. A model that **returns no
    content** failed; a model that **decided not to speak** succeeded. If these
    ever merge, every metric that counts silence is measuring outages."""
    p = put(q)
    fp = fake(judge="SPEAK", act="   ")

    result = run_turn(q, p, complete=fp.complete, at=AT)

    assert result.outcome == "failed"
    assert result.silent is False
    record = last_payload(q)
    assert record["outcome"] == "failed"
    assert record["reply"] is None
    assert "empty content" in record["error"]


def test_an_exploding_act_step_fails_the_turn(q: EventQueue) -> None:
    p = put(q)
    fp = fake(judge="ACT", act=["should never be used"])

    def act(ctx: TurnContext) -> ActResult:
        raise ZeroDivisionError("a tool did something silly")

    result = run_turn(q, p, complete=fp.complete, act=act, at=AT)

    assert result.outcome == "failed"
    assert "ZeroDivisionError" in (result.error or "")
    assert last_payload(q)["outcome"] == "failed"


def test_an_act_step_reporting_an_error_fails_the_turn(q: EventQueue) -> None:
    p = put(q)
    fp = fake(judge="ACT", act=["should never be used"])

    def act(ctx: TurnContext) -> ActResult:
        return ActResult(tools=("shell",), error="the tool never came back")

    result = run_turn(q, p, complete=fp.complete, act=act, at=AT)

    assert result.outcome == "failed"
    assert "never came back" in (result.error or "")
    assert result.tools == ("shell",)
    assert last_payload(q)["tools"] == ["shell"]


def test_running_a_turn_over_a_record_is_refused_before_anything_is_written(
    q: EventQueue,
) -> None:
    p = put(q)
    record_seq = q.append(
        episodes.completed(for_seq=p.seq, outcome="silent", at=AT),
        episodes.turn_write_key(p.seq),
    )
    head_before = q.head()

    with pytest.raises(NotAnEvent):
        run_turn(q, q.at(record_seq), complete=fake().complete, at=AT)
    assert q.head() == head_before


def test_running_one_turn_twice_is_rejected_by_the_log(q: EventQueue) -> None:
    """The write key as a live invariant check (§2.1): a turn cannot be
    recorded twice, and the refusal comes from the log rather than from a
    convention in the loop."""
    p = put(q)
    run_turn(q, p, complete=fake(judge="SPEAK", act="first").complete, at=AT)

    with pytest.raises(WriteKeyConflict):
        run_turn(q, p, complete=fake(judge="SPEAK", act="second").complete, at=AT)

    payloads = [x.payload for x in q.recent(10)]
    terminals = [x for x in payloads if episodes.is_terminal(x)]
    assert len(terminals) == 1
    assert terminals[0]["reply"] == "first"


def test_write_memory_refuses_a_blocked_turn_with_nothing_to_ask(q: EventQueue) -> None:
    p = put(q)
    with pytest.raises(ValueError):
        write_memory(q, seq=p.seq, outcome="blocked", needs=None, at=AT)
    with pytest.raises(ValueError):
        write_memory(q, seq=p.seq, outcome="blocked", needs="", at=AT)


# --- yellow: the distinctions that must survive -----------------------------


def test_silence_failure_and_speech_are_three_distinguishable_records(
    q: EventQueue,
) -> None:
    """The violation metric in one test: run all three and assert the log can
    tell them apart on the fields a later metric would read."""
    outcomes = {}
    for text, judge_says, act_says in [
        ("speak", "SPEAK", "said it"),
        ("silent", "SILENT", None),
        ("failed", "nonsense", None),
    ]:
        p = put(q, text)
        fp = provider.FakeProvider(
            {provider.JUDGE: judge_says, provider.ACT: act_says or "unused"}
        )
        result = run_turn(q, p, complete=fp.complete, at=AT)
        outcomes[text] = q.at(result.record_seq).payload

    assert outcomes["speak"]["outcome"] == "spoke"
    assert outcomes["silent"]["outcome"] == "silent"
    assert outcomes["failed"]["outcome"] == "failed"

    assert outcomes["speak"]["reply"] == "said it"
    assert outcomes["silent"]["reply"] is None
    assert outcomes["failed"]["reply"] is None

    assert outcomes["silent"]["error"] is None
    assert outcomes["failed"]["error"]
    # The pair that must never be equal, spelled out.
    assert outcomes["silent"]["outcome"] != outcomes["failed"]["outcome"]


def test_a_turn_runs_with_nothing_in_recall(q: EventQueue) -> None:
    """The first turn omega ever takes. An empty corpus is the resting state of
    a new install, not an error."""
    p = put(q, "hello")
    assert recall(q, before=p.seq) == []

    result = run_turn(q, p, complete=fake(judge="SPEAK", act="hi").complete, at=AT)
    assert result.outcome == "spoke"


def test_recall_carries_earlier_silence_into_the_prompt(q: EventQueue) -> None:
    """A silent turn is part of the history. Hiding it would make the judge
    re-decide an event it has already answered."""
    first = put(q, "moved the meeting")
    run_turn(q, first, complete=fake(judge="SILENT").complete, at=AT)
    q.finish(first.seq)

    second = put(q, "still ok?")
    fp = fake(judge="SPEAK", act="Yes.")
    run_turn(q, second, complete=fp.complete, at=AT)

    judge_prompt = fp.calls_for(provider.JUDGE)[0][-1]["content"]
    assert "moved the meeting" in judge_prompt
    assert "stayed silent" in judge_prompt


def test_the_judge_prompt_names_all_three_verdicts(q: EventQueue) -> None:
    """A judge prompt that never mentions silence is a judge that never chooses
    it — the rubber-stamp regression DL-024 warns about, one layer earlier."""
    p = put(q)
    fp = fake(judge="SILENT")
    run_turn(q, p, complete=fp.complete, at=AT)

    system = fp.calls_for(provider.JUDGE)[0][0]["content"]
    assert "SPEAK" in system and "ACT" in system and "SILENT" in system
    assert "successful" in system.lower()


def test_the_default_act_step_is_empty_and_says_so(q: EventQueue) -> None:
    """Step 4 is deliberately empty until step 7. A turn that judged ACT still
    completes, and the log shows no tools rather than pretending some ran."""
    p = put(q, "do the thing")
    fp = fake(judge="ACT", act="Done.")

    result = run_turn(q, p, complete=fp.complete, at=AT)

    assert result.outcome == "spoke"
    assert result.tools == ()
    assert last_payload(q)["tools"] == []


def test_a_verdict_keeps_the_raw_answer_it_was_parsed_from() -> None:
    v = parse_verdict("SILENT - nothing worth saying")
    assert v == Verdict(choice=STAY_SILENT, raw="SILENT - nothing worth saying")
    assert v.speaks is False


# --- context budgets: a count is not a budget (DL-039) -----------------------


def _prompt(q: EventQueue, pending) -> str:
    """The judge's user message, as text. The judge rather than act or reply
    because it fires on *every* event including every clock tick, so it is
    where an unbounded transcript costs the most."""
    ctx = TurnContext(
        seq=pending.seq,
        event=pending.payload,
        recalled=recall(q, before=pending.seq),
        queue=q,
        complete=lambda *a, **k: None,
    )
    message = turn._judge_messages(ctx)[-1]
    content = message["content"]
    return content if isinstance(content, str) else content[0]["text"]


def test_an_ordinary_conversation_is_not_elided_at_all(q):
    """The cap must be invisible in normal use. A budget that fires on a
    two-sentence message would teach the model that every line is truncated,
    which is worse than the problem it was added to solve."""
    for i in range(5):
        seq = q.append(episodes.inbound(f"message {i}", channel="tray", at=AT))
        q.claim(seq)
        q.append(episodes.completed(for_seq=seq, outcome="spoke", reply=f"reply {i}", at=AT))
        q.finish(seq)
    text = _prompt(q, put(q))
    assert "not shown here" not in text
    assert "message 0" in text and "reply 4" in text


def test_one_pasted_document_does_not_ride_along_on_every_later_turn(q):
    """The measured defect. A 100 KB paste rendered in full into every later
    prompt — ~27k tokens, re-sent up to forty times, including into every
    unattended clock tick."""
    paste = "lorem ipsum dolor sit amet " * 4000
    seq = q.append(episodes.inbound(paste, channel="tray", at=AT))
    q.claim(seq)
    q.append(episodes.completed(for_seq=seq, outcome="spoke", reply="noted", at=AT))
    q.finish(seq)

    text = _prompt(q, put(q))
    assert len(text) < 10_000, f"transcript is {len(text):,} chars"
    assert "not shown here" in text


def test_an_elision_says_how_much_is_missing_and_where_to_find_it(q):
    """An elision that reads like the end of the sentence teaches the model the
    person stopped mid-thought. Naming the seq keeps this a bounded *view* of a
    complete record — something omega can be asked to go and fetch."""
    seq = q.append(episodes.inbound("x" * 50_000, channel="tray", at=AT))
    q.claim(seq)
    q.append(episodes.completed(for_seq=seq, outcome="spoke", reply="ok", at=AT))
    q.finish(seq)

    text = _prompt(q, put(q))
    assert f"at seq {seq} in the log" in text
    assert "more characters not shown here" in text


def test_the_full_text_is_still_in_the_log_after_it_is_elided(q):
    """*Grade the world.* The budget bounds a derived view; it must not touch
    the record. If this ever fails, elision has become deletion."""
    paste = "y" * 50_000
    seq = q.append(episodes.inbound(paste, channel="tray", at=AT))
    q.claim(seq)
    q.append(episodes.completed(for_seq=seq, outcome="spoke", reply="ok", at=AT))
    q.finish(seq)

    _prompt(q, put(q))
    assert q.at(seq).payload["text"] == paste


def test_the_new_event_keeps_a_far_looser_budget_than_recall(q):
    """Cutting the new event is cutting the question. A 10 KB message is
    history-sized when recalled and subject-sized when it is what was just
    asked, and one budget for both would have to be wrong for one of them."""
    message = "z" * 10_000
    pending = q.at(q.append(episodes.inbound(message, channel="tray", at=AT)))
    q.claim(pending.seq)
    text = _prompt(q, q.at(pending.seq))
    new_event = text.split("New event:\n", 1)[1]
    assert message in new_event

    # The same message, now history: cut.
    q.append(
        episodes.completed(
            for_seq=pending.seq, outcome="spoke", reply="ok", at=AT
        )
    )
    q.finish(pending.seq)
    history = _prompt(q, put(q)).split("New event:", 1)[0]
    assert message not in history
    assert "not shown here" in history


def test_a_message_too_large_even_for_the_new_event_budget_is_cut(q):
    """The looser budget is still a budget. One pasted book should cost a
    degraded turn, not a failed request."""
    pending = q.at(q.append(episodes.inbound("q" * 200_000, channel="tray", at=AT)))
    q.claim(pending.seq)
    text = _prompt(q, q.at(pending.seq))
    assert len(text) < turn.MAX_NEW_EVENT_CHARS + 5_000
    assert "not shown here" in text


def test_many_medium_lines_are_bounded_by_the_transcript_budget(q):
    """The shape a per-line cap alone misses: forty lines each just under it.
    Without the transcript budget this is ~40 x 2,000 chars of context that no
    single line is responsible for."""
    for i in range(turn.RECALL_N):
        seq = q.append(
            episodes.inbound(f"{i} " + "w" * 1_900, channel="tray", at=AT)
        )
        q.claim(seq)
        q.append(episodes.completed(for_seq=seq, outcome="spoke", reply="ok", at=AT))
        q.finish(seq)

    history = _prompt(q, put(q)).split("New event:", 1)[0]
    assert len(history) < turn.MAX_RECALL_CHARS + 2_000, f"{len(history):,} chars"


def test_dropping_oldest_lines_is_announced_and_never_silent(q):
    """A window that silently got shorter is a model answering "you never
    mentioned that" — right about its context and wrong about the conversation."""
    for i in range(turn.RECALL_N):
        seq = q.append(
            episodes.inbound(f"{i} " + "v" * 1_900, channel="tray", at=AT)
        )
        q.claim(seq)
        q.append(episodes.completed(for_seq=seq, outcome="spoke", reply="ok", at=AT))
        q.finish(seq)

    history = _prompt(q, put(q)).split("New event:", 1)[0]
    assert "earlier event(s) not shown here" in history


def test_the_newest_lines_survive_when_the_budget_bites(q):
    """When something has to go it is the least recent thing, because that is
    the one whose absence is least likely to be the answer."""
    for i in range(turn.RECALL_N):
        seq = q.append(
            episodes.inbound(f"marker{i} " + "u" * 1_900, channel="tray", at=AT)
        )
        q.claim(seq)
        q.append(episodes.completed(for_seq=seq, outcome="spoke", reply="ok", at=AT))
        q.finish(seq)

    history = _prompt(q, put(q)).split("New event:", 1)[0]
    assert f"marker{turn.RECALL_N - 1}" in history
    assert "marker0" not in history


def test_elide_leaves_a_line_at_its_budget_plus_only_the_note():
    """Fail closed on the cap itself: a cap that can be exceeded by the note it
    adds is not a cap."""
    line = "a" * 10_000
    out = turn._elide(line, 100, at=7)
    assert out.startswith("a" * 100)
    assert len(out) < 100 + 120
    assert "at seq 7 in the log" in out


def test_elide_does_not_touch_a_line_already_within_budget():
    assert turn._elide("short", 100, at=1) == "short"
