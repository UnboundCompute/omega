"""DL-087 #6 — the topics they keep asking about, and keeping up with them.

Counting is code: a word crosses once, on the message that brings it to three
asking messages on two days within a fortnight, and only then is one LEARN
call made. Without the ``topics`` grant nothing is called and nothing is
written; an explicit follow needs no grant. **No socket, no key**: every
provider and search is scripted, and every clock is pinned.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from omega import episodes, grants, mcp, provider, tools, topics, turn as turn_mod
from omega.act import act_loop
from omega.executor import Executor
from omega.memory import MemoryStore
from omega.provider import ToolCall
from omega.queue import EventQueue
from omega.topics import Topics

NOW = datetime(2026, 10, 11, 12, 0, tzinfo=timezone.utc)
AT = NOW.isoformat()


def day(n: int) -> str:
    """Noon UTC ``n`` days before NOW: a distinct local day in any zone."""
    return (NOW - timedelta(days=n)).isoformat()


@pytest.fixture(autouse=True)
def pinned(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tools, "_now", lambda: NOW.astimezone())
    monkeypatch.delenv("OMEGA_MCP_SERVERS", raising=False)
    monkeypatch.delenv("OMEGA_WEB_SEARCH", raising=False)


@pytest.fixture
def q(store: MemoryStore) -> EventQueue:
    return EventQueue(store)


def payloads(q: EventQueue) -> list[dict]:
    return [episodes.decode(e.payload) for e in q.store.episodes_since(0)]


def kinds(q: EventQueue, kind: str) -> list[dict]:
    return [p for p in payloads(q) if p.get("kind") == kind]


def say(q: EventQueue, text: str, at: str = AT, channel: str = "tray", **extra: Any) -> int:
    return q.append(episodes.inbound(text, channel=channel, at=at, **extra))


def give(q: EventQueue, at: str = AT) -> None:
    seq = say(q, "you have the topics grant", at=at)
    q.append(episodes.grant_given(for_seq=seq, scope=grants.TOPICS, at=at))
    q.claim(seq)
    q.finish(seq)


def drop_gate(q: EventQueue) -> None:
    seq = say(q, "stop asking, just do it")
    p = q.append(episodes.grant_proposed(
        for_seq=seq, scope=grants.TOPICS, proposes="drop_gate", actions=[], at=AT
    ))
    q.append(episodes.grant_answered(
        for_seq=seq, scope=grants.TOPICS, proposal=p, proposed_for=seq, yes=True, at=AT
    ))
    q.claim(seq)
    q.finish(seq)


def follow(q: EventQueue, name: str, query: str = "") -> None:
    seq = say(q, f"keep up with {name}")
    q.claim(seq)
    q.finish(seq)
    q.append(episodes.topic_followed(name=name, query=query or f"{name} news", since=seq, at=AT))


def call(tool: str, **arguments: Any) -> ToolCall:
    return ToolCall(id="c1", name=tool, arguments=dict(arguments))


def named(name: str = "Rust 2027 edition", follow: bool = True) -> str:
    return json.dumps({"topic": {"name": name, "query": f"{name} news", "follow": follow}})


def found(*urls: str) -> str:
    return json.dumps({"items": [{"title": f"story {u}", "url": u} for u in urls]})


class Script:
    """Scripted provider: the naming LEARN calls, the searches (hosted ACT
    or MCP-read LEARN) and the act answers are counted apart."""

    def __init__(
        self, *names: str, searches: list[str] | None = None, act: list[Any] | None = None
    ) -> None:
        self.names = list(names)
        self.searches = list(searches or [])
        self.acts = list(act) if act is not None else None
        self.naming: list[list[dict]] = []
        self.searched: list[list[dict]] = []
        self.fp = provider.FakeProvider({
            provider.JUDGE: lambda role, m: "ACT",
            provider.ACT: self._act,
            provider.LEARN: self._learn,
        })

    def _act(self, role: str, messages: list[dict]) -> Any:
        if "Search the web for the latest news" in str(messages[0]["content"]):
            self.searched.append(list(messages))
            return self.searches.pop(0)
        if self.acts is not None:
            return self.acts.pop(0)
        return "Okay."

    def _learn(self, role: str, messages: list[dict]) -> str:
        head = str(messages[0]["content"])
        if "keep asking about" in head:
            self.naming.append(list(messages))
            return self.names.pop(0)
        if "Below are web search results" in head:
            self.searched.append(list(messages))
            return self.searches.pop(0)
        return "{}"


def executor(q: EventQueue, script: Script) -> Executor:
    ex = Executor(q, complete=script.fp.complete, act=act_loop)
    ex.recover()
    return ex


def run(ex: Executor, q: EventQueue) -> Any:
    last = None
    while next(q.pending(), None) is not None:
        result = ex.step()
        last = result if result is not None else last
    return last


def asks_rust(q: EventQueue, ex: Executor, days: tuple[int, ...] = (6, 4, 2)) -> None:
    for n in days:
        say(q, "any news on the rust 2027 edition?", at=day(n))
        run(ex, q)


def view(q: EventQueue) -> topics.State:
    return Topics.rebuild(q.store).state()


# --- the code pre-filter and the counting -------------------------------------


def test_asking_and_words() -> None:
    assert topics.asking("any news on the rust edition?")
    assert topics.asking("what's the price of copper")
    assert not topics.asking("I made pasta tonight")
    assert topics.words("Any news on the Rust 2027 edition?") == {"rust", "2027", "edition"}


def test_digest_ignores_scheme_case_www_slash_and_fragment() -> None:
    a = topics.digest("https://www.Example.com/post/1/#top")
    assert a == topics.digest("http://example.com/post/1")
    assert a != topics.digest("https://example.com/post/2")


def fold(*asked: tuple[int, str]) -> tuple[topics.State, list[int]]:
    """A fold over asking messages ``(days before NOW, text)``."""
    view_ = Topics()
    seqs = []
    for i, (n, text) in enumerate(asked, start=1):
        view_.apply(i, episodes.inbound(text, channel="tray", at=day(n)))
        seqs.append(i)
    return view_.state(), seqs


def test_a_word_crosses_on_its_third_ask_on_two_days() -> None:
    state, seqs = fold((6, "rust edition?"), (4, "rust edition?"), (2, "rust edition?"))
    assert [w for w, _ in state.crossing(seqs[2])] == ["edition", "rust"]
    assert state.crossing(seqs[1]) == [], "two asks are not enough"


def test_three_asks_on_one_day_do_not_cross() -> None:
    state, seqs = fold((2, "rust edition?"), (2, "rust edition?"), (2, "rust edition?"))
    assert state.crossing(seqs[2]) == []


def test_asks_outside_the_fortnight_do_not_count() -> None:
    state, seqs = fold((20, "rust edition?"), (18, "rust edition?"), (2, "rust edition?"))
    assert state.crossing(seqs[2]) == []


def test_a_word_crosses_once() -> None:
    state, seqs = fold(*((n, "rust edition?") for n in (8, 6, 4, 2)))
    assert state.crossing(seqs[2]) and state.crossing(seqs[3]) == []


def test_only_asking_messages_of_theirs_count() -> None:
    view_ = Topics()
    view_.apply(1, episodes.inbound("rust edition?", channel="tray", at=day(6)))
    view_.apply(2, episodes.inbound("I read about the rust edition", channel="tray", at=day(4)))
    view_.apply(3, episodes.inbound("rust edition?", channel="schedule", at=day(3),
                                    schedule_id="s1"))
    view_.apply(4, episodes.inbound("rust edition?", channel="tray", at=day(2)))
    assert view_.state().crossing(4) == []


# --- no grant: nothing at all --------------------------------------------------


def test_without_the_grant_nothing_is_called_written_or_offered(q: EventQueue) -> None:
    script = Script()
    ex = executor(q, script)
    asks_rust(q, ex)
    assert script.naming == [] and script.searched == []
    assert kinds(q, episodes.GRANT_PROPOSED) == []
    assert not any(p["kind"].startswith("topic.") for p in payloads(q))
    assert ex.keep_up(now=AT) is False and script.searched == []
    offered = {s["function"]["name"] for o in script.fp.offers if o for s in o if "function" in s}
    # Offered on every turn of theirs (no word list decides it); offering it
    # is not calling it, and nothing above was followed or written.
    assert tools.TOPICS in offered, "every turn of theirs is offered topics"


def test_a_revoked_grant_stops_the_offers(q: EventQueue) -> None:
    give(q)
    q.append(episodes.grant_revoked(for_seq=1, scope=grants.TOPICS, at=AT))
    script = Script()
    asks_rust(q, executor(q, script))
    assert script.naming == [] and kinds(q, episodes.GRANT_PROPOSED) == []


# --- the offer -----------------------------------------------------------------


def test_the_third_ask_is_one_learn_call_and_one_offer(q: EventQueue) -> None:
    give(q, at=day(7))
    script = Script(named())
    ex = executor(q, script)
    asks_rust(q, ex, days=(6, 4))
    assert script.naming == [] and kinds(q, episodes.GRANT_PROPOSED) == []
    asks_rust(q, ex, days=(2,))
    assert len(script.naming) == 1
    assert "any news on the rust 2027 edition?" in str(script.naming[0][-1]["content"])
    [proposal] = kinds(q, episodes.GRANT_PROPOSED)
    assert proposal["scope"] == grants.TOPICS and proposal["proposes"] == "action"
    [action] = proposal["actions"]
    assert action["tool"] == tools.TOPICS
    assert action["args"]["do"] == "follow" and action["args"]["name"] == "Rust 2027 edition"
    tell = payloads(q)[proposal["for_seq"] - 1]
    assert tools.is_tell(tell) and tools.asks(tell)
    assert "I've noticed you keep asking about Rust 2027 edition" in tell["text"]
    assert kinds(q, episodes.TOPIC_FOLLOWED) == [], "nothing followed before their yes"


def test_the_offer_is_said_with_its_id_and_a_yes_follows(q: EventQueue) -> None:
    give(q, at=day(7))
    script = Script(named())
    ex = executor(q, script)
    asks_rust(q, ex)
    pseq = next(e.seq for e in q.store.episodes_since(0)
                if episodes.decode(e.payload)["kind"] == episodes.GRANT_PROPOSED)
    told = [r for r in kinds(q, episodes.TURN_COMPLETED) if "keep asking about" in str(r.get("reply"))]
    assert told and f'Reply "yes p{pseq}" or "no p{pseq}".' in told[-1]["reply"]
    say(q, f"yes p{pseq}")
    run(ex, q)
    [followed] = kinds(q, episodes.TOPIC_FOLLOWED)
    assert followed["name"] == "Rust 2027 edition"
    [acted] = kinds(q, episodes.GRANT_ACTED)
    assert acted["scope"] == grants.TOPICS and acted["undo"] == {"drop": "Rust 2027 edition"}
    assert set(view(q).followed) == {"rust 2027 edition"}


def test_a_no_is_never_offered_again(q: EventQueue) -> None:
    give(q, at=day(13))
    script = Script(named())
    ex = executor(q, script)
    asks_rust(q, ex, days=(12, 11, 10))
    pseq = next(e.seq for e in q.store.episodes_since(0)
                if episodes.decode(e.payload)["kind"] == episodes.GRANT_PROPOSED)
    say(q, f"no p{pseq}", at=day(9))
    run(ex, q)
    assert kinds(q, episodes.TOPIC_FOLLOWED) == []
    # Asked about again, more than enough: no second LEARN call, no offer.
    asks_rust(q, ex, days=(5, 3, 1, 0))
    assert len(script.naming) == 1 and len(kinds(q, episodes.GRANT_PROPOSED)) == 1


def test_a_one_off_fact_is_not_offered(q: EventQueue) -> None:
    give(q, at=day(7))
    script = Script(named("Rust 2027 edition", follow=False))
    asks_rust(q, executor(q, script))
    assert len(script.naming) == 1 and kinds(q, episodes.GRANT_PROPOSED) == []


@pytest.mark.parametrize("bad", ["not json", '{"topic": {"name": "x"}}', '{"other": 1}'])
def test_a_bad_answer_offers_nothing_and_is_not_retried(q: EventQueue, bad: str) -> None:
    give(q, at=day(7))
    script = Script(bad)
    ex = executor(q, script)
    asks_rust(q, ex)
    assert len(script.naming) == 1 and kinds(q, episodes.GRANT_PROPOSED) == []


def test_with_the_gate_dropped_it_follows_with_its_undo(q: EventQueue) -> None:
    give(q, at=day(7))
    drop_gate(q)
    script = Script(named())
    asks_rust(q, executor(q, script))
    [followed] = kinds(q, episodes.TOPIC_FOLLOWED)
    assert followed["name"] == "Rust 2027 edition"
    [acted] = kinds(q, episodes.GRANT_ACTED)
    assert acted["proposal"] is None and acted["undo"] == {"drop": "Rust 2027 edition"}


def test_at_five_followed_nothing_more_is_offered(q: EventQueue) -> None:
    give(q, at=day(7))
    for i in range(topics.MAX_FOLLOWED):
        follow(q, f"subject {i}")
    script = Script(named())
    asks_rust(q, executor(q, script))
    assert kinds(q, episodes.GRANT_PROPOSED) == []


# --- the topics tool -----------------------------------------------------------


def test_an_explicit_follow_needs_no_grant(q: EventQueue) -> None:
    script = Script(act=[call("topics", do="follow", name="Copper price",
                              query="copper price today"), "Will do."])
    say(q, "keep up with the copper price for me")
    run(executor(q, script), q)
    [followed] = kinds(q, episodes.TOPIC_FOLLOWED)
    assert followed["name"] == "Copper price" and followed["query"] == "copper price today"
    assert script.naming == []


def test_follow_refuses_a_sixth(q: EventQueue) -> None:
    for i in range(topics.MAX_FOLLOWED):
        follow(q, f"subject {i}")
    script = Script(act=[call("topics", do="follow", name="one more"), "No room."])
    say(q, "keep up with one more")
    run(executor(q, script), q)
    assert len(view(q).followed) == topics.MAX_FOLLOWED
    [returned] = [r for r in kinds(q, episodes.TOOL_RETURNED) if r["tool"] == "topics"]
    assert returned["ok"] is False or "5" in str(returned.get("error") or returned.get("result"))


def test_drop_stops_it_and_list_names_them(q: EventQueue) -> None:
    follow(q, "Copper price")
    follow(q, "Rust 2027 edition")
    script = Script(act=[call("topics", do="drop", name="copper"), "Dropped."])
    say(q, "stop keeping up with copper")
    run(executor(q, script), q)
    [dropped] = kinds(q, episodes.TOPIC_DROPPED)
    assert dropped["name"] == "Copper price"
    assert set(view(q).followed) == {"rust 2027 edition"}
    box = tools.ToolBox(store_root=q.store.root, prompted=True, topics=view(q))
    assert "Rust 2027 edition" in box.dispatch(box.classify("topics", {"do": "list"}))


def test_now_looks_once_a_day(q: EventQueue) -> None:
    follow(q, "Rust 2027 edition")
    script = Script(
        searches=[found("https://a.example/1")],
        act=[call("topics", do="now", name="rust"), "Here.",
             call("topics", do="now", name="rust"), "Same."],
    )
    ex = executor(q, script)
    say(q, "what's new on the rust edition you keep up with?")
    run(ex, q)
    say(q, "and now, on the rust edition?")
    run(ex, q)
    assert len(script.searched) == 1, "the second ask today reads the first look"
    [update] = kinds(q, episodes.TOPIC_UPDATE)
    assert update["for_seq"] and [i["url"] for i in update["items"]] == ["https://a.example/1"]
    results = [r["result"] for r in kinds(q, episodes.TOOL_RETURNED) if r["tool"] == "topics"]
    assert "already looked" in results[-1] and "a.example/1" in results[-1]


# --- the daily look ------------------------------------------------------------


def test_keep_up_once_a_day_and_only_new_items(q: EventQueue) -> None:
    follow(q, "Rust 2027 edition")
    script = Script(searches=[
        found("https://a.example/1", "https://b.example/2"),
        found("https://www.a.example/1/", "https://c.example/3"),
    ])
    ex = executor(q, script)
    assert ex.keep_up(now=day(1)) is True
    assert ex.keep_up(now=day(1)) is False, "once a day"
    assert ex.keep_up(now=AT) is True
    first, second = kinds(q, episodes.TOPIC_UPDATE)
    assert [i["url"] for i in first["items"]] == ["https://a.example/1", "https://b.example/2"]
    assert [i["url"] for i in second["items"]] == ["https://c.example/3"], "deduped"
    assert len(script.searched) == 2 and first["usage"]


def test_keep_up_with_nothing_followed_is_nothing(q: EventQueue) -> None:
    give(q)
    script = Script()
    ex = executor(q, script)
    calls = len(script.fp.calls)
    assert ex.keep_up(now=AT) is False
    assert len(script.fp.calls) == calls and kinds(q, episodes.TOPIC_UPDATE) == []


def test_keep_up_stops_at_the_daily_budget(q: EventQueue) -> None:
    for i in range(3):
        follow(q, f"subject {i}")
    q.append(episodes.topic_update(
        name="subject 0", items=[], at=AT,
        usage=[{"role": "act", "model": "m", "calls": 1, "input": topics.DAILY_TOKENS,
                "cached": 0, "output": 0, "reasoning": 0, "unmetered": 0}],
    ))
    script = Script()
    ex = executor(q, script)
    assert ex.keep_up(now=AT) is False and script.searched == []


def test_keep_up_without_any_search_writes_down_why(q: EventQueue, monkeypatch) -> None:
    monkeypatch.setenv("OMEGA_WEB_SEARCH", "off")
    follow(q, "Rust 2027 edition")
    script = Script()
    ex = executor(q, script)
    assert ex.keep_up(now=AT) is True
    [update] = kinds(q, episodes.TOPIC_UPDATE)
    assert update["items"] == [] and "no web search" in update["reason"]
    assert ex.keep_up(now=AT) is False, "a failed look is still the day's look"


class FakeRemote:
    def __init__(self) -> None:
        self.ran: list[Any] = []

    def schemas(self) -> list[dict]:
        return [{"type": "function", "function": {"name": "exa__web_search_exa"}}]

    def classify(self, name: str, args: dict) -> Any:
        return (name, args)

    def dispatch(self, decision: Any) -> str:
        self.ran.append(decision)
        return "Result: Rust 2027 lands https://blog.example/rust-2027"


def test_keep_up_uses_their_mcp_search_when_configured(q: EventQueue, monkeypatch) -> None:
    remote = FakeRemote()
    monkeypatch.setattr(mcp, "remote", lambda *a, **k: remote)
    follow(q, "Rust 2027 edition", "rust 2027 edition")
    script = Script(searches=[found("https://blog.example/rust-2027")])
    ex = executor(q, script)
    assert ex.keep_up(now=AT) is True
    assert remote.ran == [("exa__web_search_exa", {"query": "rust 2027 edition"})]
    assert script.fp.calls_for(provider.ACT) == [], "no hosted search"
    [update] = kinds(q, episodes.TOPIC_UPDATE)
    assert [i["url"] for i in update["items"]] == ["https://blog.example/rust-2027"]


# --- the brief -----------------------------------------------------------------


def brief(q: EventQueue, at: str = AT) -> int:
    return say(q, tools.brief_instruction(), channel="schedule", schedule_id="b1-0001", at=at)


def test_the_brief_lists_only_topics_with_news_once(q: EventQueue) -> None:
    follow(q, "Rust 2027 edition")
    follow(q, "Copper price")
    script = Script(searches=[found("https://a.example/1"), found()])
    ex = executor(q, script)
    ex.keep_up(now=AT)
    ex.keep_up(now=AT)
    brief(q)
    result = run(ex, q)
    assert result.reply.endswith(
        "Keeping up with:\n- Rust 2027 edition: story https://a.example/1 (https://a.example/1)"
    )
    assert "Copper price" not in result.reply
    brief(q)
    assert turn_mod.KEEPING_UP not in run(ex, q).reply, "each item once"


def test_without_news_the_brief_has_no_section(q: EventQueue, monkeypatch) -> None:
    follow(q, "Rust 2027 edition")
    script = Script(searches=[found()])
    ex = executor(q, script)
    ex.keep_up(now=AT)
    brief(q)
    with_fold = run(ex, q)
    monkeypatch.setattr(turn_mod, "_keeping", lambda ctx: "")
    brief(q)
    without = run(ex, q)
    assert with_fold.reply == without.reply == "Okay."


def test_equivalence_no_grant_no_followed_changes_nothing(tmp_path, monkeypatch) -> None:
    """The same log, run with the topics fold and with it forced to nothing,
    asks the provider exactly the same and writes the same records."""

    def scenario(root, patch: bool) -> tuple[list, list, list]:
        root.mkdir()
        s = MemoryStore.open(root)
        try:
            qq = EventQueue(s)
            script = Script()
            with monkeypatch.context() as m:
                if patch:
                    m.setattr(topics.Topics, "state", lambda self: None)
                    m.setattr(Executor, "_notice_topics", lambda self, p: None)
                ex = executor(qq, script)
                asks_rust(qq, ex)
                brief(qq)
                run(ex, qq)
                ex.keep_up(now=AT)
            recs = [
                {k: v for k, v in p.items() if k not in ("at", "elapsed_ms", "ms")}
                for p in payloads(qq)
            ]
            def same(x: Any) -> str:  # the store's own path aside
                return json.dumps(x, sort_keys=True, default=str).replace(str(root), "<root>")

            return same(script.fp.calls), same(script.fp.offers), same(recs)
        finally:
            s.close()

    a = scenario(tmp_path / "a", patch=False)
    b = scenario(tmp_path / "b", patch=True)
    assert a[0] == b[0] and a[1] == b[1]
    assert a[2] == b[2]


# --- restart -------------------------------------------------------------------


def test_the_fold_survives_a_restart_and_is_incremental(q: EventQueue) -> None:
    give(q, at=day(7))
    follow(q, "Copper price")
    script = Script(searches=[found("https://a.example/1")])
    ex = executor(q, script)
    ex.keep_up(now=AT)
    brief(q)
    run(ex, q)
    incremental = Topics()
    incremental.advance(q.store)
    assert incremental.state() == view(q)
    # A new executor on the same log (a restart) does not look again today.
    ex2 = executor(q, Script())
    assert ex2.keep_up(now=AT) is False
    assert view(q).seen["copper price"] == {topics.digest("https://a.example/1")}


def test_topics_is_offered_on_every_turn_of_theirs_and_never_unprompted() -> None:
    # No word list decides it: any phrasing of theirs is offered the tool,
    # nothing followed or granted; a turn they did not start never is.
    for said in ("keep up with the rust edition", "can u watch rust news 4 me", "hello", ""):
        box = tools.ToolBox(store_root=".", prompted=True, said=said)
        assert tools.TOPICS in {s["function"]["name"] for s in box.schemas()}, said
    box = tools.ToolBox(store_root=".", prompted=False, said="keep up with rust")
    assert tools.TOPICS not in {s["function"]["name"] for s in box.schemas()}
    with pytest.raises(tools.ToolRejected):
        box.classify(tools.TOPICS, {"do": "list"})
