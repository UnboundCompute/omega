"""DL-094 — look_back: search over the whole past conversation.

Fakes only: a real store in a temp dir, the fake provider, pinned clocks.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from omega import episodes, lookback, provider, tools
from omega.act import act_loop
from omega.memory import MemoryStore
from omega.provider import ToolCall
from omega.queue import EventQueue
from omega.tools import ToolBox, ToolRejected
from omega.turn import run_turn

NOW = datetime(2026, 10, 12, 12, 0, tzinfo=timezone.utc)
AT = NOW.isoformat()


def at(month: int, day: int, hour: int = 12) -> str:
    # Midday UTC, so the local date is the same in every zone the suite runs in.
    return datetime(2026, month, day, hour, 0, tzinfo=timezone.utc).isoformat()


def local_day(stamp: str) -> str:
    return f"{datetime.fromisoformat(stamp).astimezone():%d %b %Y}"


@pytest.fixture
def q(store: MemoryStore) -> EventQueue:
    return EventQueue(store)


def said(q: EventQueue, text: str, when: str, *, reply: str | None = None,
         channel: str = "tray") -> int:
    seq = q.append(episodes.inbound(text, channel=channel, at=when))
    if reply is not None:
        q.append(episodes.completed(for_seq=seq, outcome="spoke", reply=reply, at=when))
    return seq


def call(tool: str, **arguments: Any) -> ToolCall:
    return ToolCall(id="c1", name=tool, arguments=dict(arguments))


def turn_with(q: EventQueue, text: str, act: list[Any], *, judge: str = "ACT",
              channel: str = "tray", **extra: Any) -> provider.FakeProvider:
    seq = q.append(episodes.inbound(text, channel=channel, at=AT, **extra))
    q.claim(seq)
    fp = provider.FakeProvider({provider.JUDGE: [judge], provider.ACT: list(act)})
    run_turn(q, q.at(seq), complete=fp.complete, act=act_loop, at=AT)
    q.finish(seq)
    return fp


def offered(fp: provider.FakeProvider, n: int = 0) -> list[str]:
    return [t["function"]["name"] for t in fp.offers_for(provider.ACT)[n] or ()
            if "function" in t]


def results(fp: provider.FakeProvider) -> str:
    return "\n".join(str(m["content"]) for c in fp.calls_for(provider.ACT)
                     for m in c if m.get("role") == "tool")


def box_for(q: EventQueue, before: int = 0) -> ToolBox:
    return ToolBox(
        store_root=q.store.root, for_seq=before,
        history=lambda: lookback.past(q.store, before=before),
    )


def ask(box: ToolBox, **args: Any) -> str:
    return str(box.dispatch(box.classify(tools.LOOK_BACK, args)))


# --- the turn ----------------------------------------------------------------------


def test_a_turn_finds_what_they_said_in_august_with_its_date(q: EventQueue) -> None:
    said(q, "the landlord wants the lease signed by Friday", at(8, 14),
         reply="I'll remind you Thursday about the lease.")
    said(q, "dinner at the office later", at(9, 2), reply="Noted.")
    fp = turn_with(q, "what did I say about the lease in August?",
                   [call(tools.LOOK_BACK, query="lease", since="2026-08-01",
                         until="2026-08-31"), "On 14 Aug you said…"])
    out = results(fp)
    assert "2 of 2 past messages since 14 Aug 2026" in out
    assert "you said: the landlord wants the lease signed by Friday" in out
    assert "omega replied: I'll remind you Thursday about the lease." in out
    assert local_day(at(8, 14)) in out and "dinner" not in out
    # Its own question is never a hit: only what came before this message.
    assert "what did I say" not in out


@pytest.mark.parametrize("channel,judge", [("tray", "ACT"), ("discord", "ACT"),
                                           ("schedule", "ACT"), ("self", "CHECK")])
def test_offered_on_every_turn_kind_that_has_recall(q: EventQueue, channel: str, judge: str) -> None:
    extra = {"schedule_id": "brief"} if channel == "schedule" else {}
    fp = turn_with(q, "anything", ["Nothing."], judge=judge, channel=channel, **extra)
    names = offered(fp)
    assert tools.RECALL in names and tools.LOOK_BACK in names


def test_the_offer_never_depends_on_their_words(q: EventQueue) -> None:
    offers = [offered(turn_with(q, words, ["Ok."])) for words in
              ("hello", "what did I say in August", "remind me at 5", "")]
    assert all(tools.LOOK_BACK in o for o in offers)
    assert offers[0] == offers[1] == offers[2] == offers[3]


def test_the_description_says_to_use_it_and_to_state_the_date(q: EventQueue) -> None:
    schema = next(s for s in box_for(q).schemas() if s["function"]["name"] == tools.LOOK_BACK)
    text = schema["function"]["description"]
    assert "earlier than the conversation you can see" in text
    assert "say when it was said" in text


# --- equivalence -------------------------------------------------------------------


def test_a_box_without_history_is_exactly_what_it_was(q: EventQueue) -> None:
    plain = ToolBox(store_root=q.store.root)
    assert plain.schemas() == tools.schemas()
    assert tools.LOOK_BACK not in tools.TOOL_NAMES
    with pytest.raises(ToolRejected, match="no tool called 'look_back'"):
        plain.classify(tools.LOOK_BACK, {"query": "x"})


def test_it_is_exploration_and_writes_nothing(q: EventQueue) -> None:
    said(q, "the lease", at(8, 14))
    box = box_for(q)
    decision = box.classify(tools.LOOK_BACK, {"query": "lease"})
    assert decision.tier == tools.EXPLORATION
    head = q.store.head()
    box.dispatch(decision)
    assert q.store.head() == head


# --- matching ----------------------------------------------------------------------


def test_every_term_case_and_accent_insensitive(q: EventQueue) -> None:
    said(q, "Meet at the Café Flore on Rue de Rivoli", at(8, 3))
    said(q, "the cafe near the station was closed", at(8, 4))
    said(q, "flore is a nice name", at(8, 5))
    box = box_for(q)
    assert "3 of 3" not in ask(box, query="CAFE")
    assert "2 of 3 past messages" in ask(box, query="CAFE")
    # The accent in the query, not the text: still both.
    assert "2 of 3 past messages" in ask(box, query="Café")
    out = ask(box, query="café flore")
    assert "1 of 3 past messages" in out and "Café Flore" in out
    assert "1 of 3" in ask(box, query="RUE   rivoli")


def test_newest_first_with_a_limit(q: EventQueue) -> None:
    for day in range(1, 26):
        said(q, f"gym note {day}", at(7, day))
    box = box_for(q)
    out = ask(box, query="gym", limit=3).splitlines()
    assert out[0].startswith("25 of 25 past messages") and "showing 3" in out[0]
    assert out[1].endswith("gym note 25") and out[3].endswith("gym note 23")
    assert "22 older matches not shown" in out[-1]
    assert len([l for l in ask(box, query="gym").splitlines() if l.startswith("- ")]) == 10


def test_dates_are_inclusive_local_days(q: EventQueue) -> None:
    for day in (13, 14, 15, 16):
        said(q, f"budget {day}", at(8, day))
    box = box_for(q)
    out = ask(box, query="budget", since="2026-08-14", until="2026-08-15")
    assert "2 of 2 past messages since 14 Aug 2026 (searched 14 Aug 2026 to 15 Aug 2026)" in out
    assert "budget 14" in out and "budget 15" in out and "budget 13" not in out
    # Dates alone, no words: what was said on that day.
    assert "budget 16" in ask(box, since="2026-08-16")


def test_excerpts_are_short_and_centred_on_the_match(q: EventQueue) -> None:
    filler = "word " * 200
    said(q, filler + "the deposit is two months " + filler, at(8, 1))
    line = next(l for l in ask(box_for(q), query="deposit").splitlines() if l.startswith("- "))
    assert "the deposit is two months" in line
    assert line.count("…") == 2 and len(line) < 2 * lookback.EXCERPT_CHARS + 80


@pytest.mark.parametrize("args,why", [
    ({"limit": 21}, "1 to 20"), ({"limit": -1}, "1 to 20"), ({"limit": "5"}, "a number"),
    ({"since": "14 Aug"}, "a date like"), ({"until": 5}, "a date like"),
    ({"since": "2026-09-01", "until": "2026-08-01"}, "is after"), ({"query": 3}, "words"),
])
def test_bad_arguments_are_refused(q: EventQueue, args: dict, why: str) -> None:
    with pytest.raises(ToolRejected, match=why):
        box_for(q).classify(tools.LOOK_BACK, args)


# --- empty results -----------------------------------------------------------------


def test_no_history_is_not_the_same_as_no_match(q: EventQueue) -> None:
    empty = ask(box_for(q), query="lease")
    assert empty.startswith("There is no past conversation in the log yet")
    said(q, "dinner plans", at(8, 14))
    said(q, "gym plans", at(8, 20))
    none = ask(box_for(q), query="lease")
    assert none == "No matches in 2 past messages since 14 Aug 2026 for 'lease'."
    ranged = ask(box_for(q), query="lease", since="2026-09-01")
    assert ranged.startswith("No matches in 0 past messages (searched from 01 Sep 2026)")
    assert "holds 2 past messages since 14 Aug 2026" in ranged


# --- what counts as conversation ---------------------------------------------------


def test_who_said_it_and_what_is_left_out(q: EventQueue) -> None:
    said(q, "lease question", at(8, 1), reply="lease answer")
    look = said(q, "lease look prompt", at(8, 2), channel="self")
    q.append(episodes.completed(for_seq=look, outcome="spoke", reply="lease nudge", at=at(8, 2)))
    fire = said(q, "lease fire prompt", at(8, 3), channel="schedule")
    q.append(episodes.completed(for_seq=fire, outcome="silent", at=at(8, 3)))
    asked = said(q, "send the lease", at(8, 4))
    q.append(episodes.blocked(for_seq=asked, needs="May I email the lease?", at=at(8, 4)))
    q.append(episodes.tool_returned(for_seq=asked, tool="fetch", ok=True,
                                    result="lease in a tool result", at=at(8, 4)))
    out = ask(box_for(q), query="lease")
    assert "you said: lease question" in out and "omega replied: lease answer" in out
    assert "omega told them: lease nudge" in out
    assert "omega asked: May I email the lease?" in out and "you said: send the lease" in out
    for absent in ("look prompt", "fire prompt", "tool result"):
        assert absent not in out
    assert out.startswith("5 of 5 past messages")


def test_a_held_mail_turn_is_redacted(q: EventQueue) -> None:
    seq = said(q, "open the held mail", at(8, 1))
    q.append(episodes.tool_returned(
        for_seq=seq, tool="mail", ok=True,
        result=f"{tools.HELD_LEAD}from bank: code — 123456", at=at(8, 1)))
    q.append(episodes.completed(for_seq=seq, outcome="spoke",
                                reply="Your recovery code is 482913, see https://example.com/r",
                                at=at(8, 1)))
    said(q, "my recovery code worked", at(8, 2), reply="Good, code 777777 noted.")
    out = ask(box_for(q), query="code")
    assert "482913" not in out and "example.com" not in out
    assert "Your recovery code is •••, see [link]" in out
    # Only that turn: an ordinary reply is shown as it was said.
    assert "Good, code 777777 noted." in out


# --- the index ---------------------------------------------------------------------


def test_the_index_follows_the_log_and_matches_a_fresh_scan(q: EventQueue) -> None:
    said(q, "first lease note", at(8, 1))
    first = lookback.past(q.store, before=0)
    assert [s.text for s in first] == ["first lease note"]
    later = said(q, "second lease note", at(8, 2), reply="lease reply")
    grown = lookback.past(q.store, before=0)
    assert [s.text for s in grown] == ["first lease note", "second lease note", "lease reply"]
    fresh = lookback.Index()
    fresh.extend(q.store.episodes_since(0))
    assert fresh.lines == grown
    # Strictly before the turn's own event.
    assert [s.text for s in lookback.past(q.store, before=later)] == ["first lease note"]


def test_a_new_store_never_sees_another_stores_index(tmp_path) -> None:
    for name, text in (("a", "alpha lease"), ("b", "beta lease")):
        d = tmp_path / name
        d.mkdir()
        with MemoryStore.open(d) as store:
            EventQueue(store).append(episodes.inbound(text, channel="tray", at=at(8, 1)))
            assert [s.text for s in lookback.past(store, before=0)] == [text]


def test_a_blank_limit_is_the_default(q: EventQueue) -> None:
    # A model fills every offered field; 0 is a blank, as everywhere (DL-079).
    assert box_for(q).classify(tools.LOOK_BACK, {"limit": 0}).args["limit"] == lookback.DEFAULT_LIMIT
