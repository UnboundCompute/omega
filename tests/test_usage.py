"""DL-070 — every pass records what it spent on the model.

Costing needs the tokens per call, split the way they are billed: input,
the cached part of input, output, and the reasoning part of output. The
provider already read the first two and dropped them on the floor; nothing
in the log said what a turn cost.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from omega import episodes, executor, provider
from omega.memory import MemoryStore
from omega.queue import EventQueue
from omega.turn import run_turn

from tests.teaching import _a_claim, _answer, _log, _Teaching

AT = "2026-10-07T12:00:00+00:00"


@pytest.fixture
def q(store: MemoryStore) -> EventQueue:
    return EventQueue(store)


def _priced(answers: dict[str, list[str]], tokens=(100, 20, 40, 5)):
    """A ``complete`` that answers by role and reports token counts the way
    the real provider does: (input, output, cached, reasoning)."""
    left = {role: list(texts) for role, texts in answers.items()}

    def complete(role, messages, tools=None):
        text = left[role].pop(0)
        if isinstance(text, Exception):
            raise text
        return provider.Response(
            text=text,
            model=f"model-{role}",
            prompt_tokens=tokens[0],
            completion_tokens=tokens[1],
            cached_tokens=tokens[2],
            reasoning_tokens=tokens[3],
        )

    return complete


def _turn(q: EventQueue, complete) -> dict:
    seq = q.append(episodes.inbound("hello", channel="tray", at=AT))
    q.claim(seq)
    run_turn(q, q.at(seq), complete=complete, at=AT)
    [record] = [p for _, p in _log(q) if p["kind"] in episodes.TERMINAL_KINDS]
    return record


# --- the meter --------------------------------------------------------------


def test_the_meter_tallies_per_role_and_model_split_the_way_it_is_billed() -> None:
    meter = provider.Meter(_priced({"judge": ["SPEAK"], "act": ["a", "b"]}))
    assert meter.usage() is None, "nothing called, nothing to say"

    meter("judge", [])
    meter("act", [])
    meter("act", [], None)

    assert meter.usage() == [
        {"role": "act", "model": "model-act", "calls": 2, "input": 200,
         "cached": 80, "output": 40, "reasoning": 10, "unmetered": 0},
        {"role": "judge", "model": "model-judge", "calls": 1, "input": 100,
         "cached": 40, "output": 20, "reasoning": 5, "unmetered": 0},
    ]


def test_a_response_without_counts_is_unmetered_not_free() -> None:
    """Counting it as zero would make a total look complete when it is not."""
    fp = provider.FakeProvider({provider.JUDGE: "SILENT"})
    meter = provider.Meter(fp.complete)
    meter(provider.JUDGE, [provider.user("hi")])

    [row] = meter.usage()
    assert row["calls"] == 1 and row["unmetered"] == 1
    assert row["input"] == row["output"] == 0


def test_the_real_provider_reads_cached_and_reasoning_counts() -> None:
    raw = SimpleNamespace(
        model="gpt-x",
        status="completed",
        output=[SimpleNamespace(
            type="message",
            content=[SimpleNamespace(type="output_text", text="hi")],
        )],
        usage=SimpleNamespace(
            input_tokens=1200,
            output_tokens=300,
            input_tokens_details=SimpleNamespace(cached_tokens=1024),
            output_tokens_details=SimpleNamespace(reasoning_tokens=256),
        ),
    )
    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **_: raw))
    response = provider.OpenAIProvider(client=client).complete(provider.ACT, [provider.user("hi")])

    assert (response.prompt_tokens, response.completion_tokens) == (1200, 300)
    assert (response.cached_tokens, response.reasoning_tokens) == (1024, 256)


# --- turns ------------------------------------------------------------------


def test_a_spoken_turn_records_what_every_call_in_it_spent(q: EventQueue) -> None:
    record = _turn(q, _priced({"judge": ["SPEAK"], "act": ["hi there"]}))

    assert record["outcome"] == "spoke"
    assert {(r["role"], r["calls"], r["input"]) for r in record["usage"]} == {
        ("judge", 1, 100),
        ("act", 1, 100),
    }


def test_a_silent_turn_still_paid_for_its_judge(q: EventQueue) -> None:
    record = _turn(q, _priced({"judge": ["SILENT"]}))

    assert record["outcome"] == "silent"
    [row] = record["usage"]
    assert row["role"] == "judge" and row["calls"] == 1


def test_a_failed_turn_keeps_what_it_spent_before_failing(q: EventQueue) -> None:
    """The judge was billed even though the reply call blew up."""
    record = _turn(
        q,
        _priced({"judge": ["SPEAK"], "act": [provider.ProviderError("boom")]}),
    )

    assert record["outcome"] == "failed"
    [row] = record["usage"]
    assert row["role"] == "judge" and row["calls"] == 1


# --- background passes ------------------------------------------------------


def test_a_reflection_receipt_says_what_the_pass_spent(q: EventQueue) -> None:
    for i in range(executor.REFLECT_EVERY):
        q.append(episodes.inbound(f"message {i}", channel="tray", at=AT))
    teaching = _Teaching(learn_answer=_answer(_a_claim()), verdict="SILENT")
    ex = executor.Executor(q, complete=teaching.complete)
    ex.recover()
    ex.drain()

    [done] = [p for _, p in _log(q) if p["kind"] == episodes.REFLECTION_DONE]
    assert [r["role"] for r in done["usage"]] == [provider.LEARN]
    assert done["usage"][0]["calls"] == 1


# --- the schema -------------------------------------------------------------


def _row(**over):
    return {"role": "judge", "model": "m", "calls": 1, "input": 10, "cached": 0,
            "output": 2, "reasoning": 0, "unmetered": 0, **over}


def test_a_record_without_usage_is_byte_for_byte_what_it_was() -> None:
    payload = episodes.completed(for_seq=1, outcome="silent", at=AT)
    assert "usage" not in payload


@pytest.mark.parametrize(
    "usage",
    [
        [],
        [_row(calls=0)],
        [_row(input=True)],
        [_row(cached=11)],
        [_row(reasoning=3)],
        [_row(unmetered=2)],
        [_row(model="")],
        ["judge"],
    ],
)
def test_a_malformed_usage_is_refused(usage) -> None:
    with pytest.raises(episodes.BadPayload):
        episodes.completed(for_seq=1, outcome="silent", usage=usage, at=AT)


def test_only_a_pass_ending_record_may_carry_usage() -> None:
    payload = episodes.tool_returned(for_seq=1, tool="fetch", ok=True, at=AT)
    payload["usage"] = [_row()]
    with pytest.raises(episodes.BadPayload):
        episodes.decode(json.dumps(payload).encode("utf-8"))
