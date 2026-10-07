"""Searching the web through the provider's hosted search — DL-067.

Three groups, in the order a search travels.

The **offer**: hosted search is offered beside Ring 1 and never inside it, so
the closed tool set stays closed and the gate never sees a call it cannot
dispatch. It can be switched off, because a model that does not support it
rejects every act call and the remedy must not need code.

The **reading**: a ``web_search_call`` item becomes a :class:`HostedCall`. The
fixture is the shape the live API returned on 2026-10-07, not one written from
the docs. The rule the group pins is that reading the *record* of a search never
fails the call — the answer built on that search is in the same response.

The **log**: the condition hosted search was chosen on. Every search lands as a
``tool.called``/``tool.returned`` pair, so "what did omega look up and what did
it read" is answered from the log like any other tool. This is the violation
check: a hosted search that ran and left no trace is exactly what the choice of
hosted over our own tool was conditional on not happening.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from omega import episodes, provider, tools
from omega.act import act_loop
from omega.memory import MemoryStore
from omega.queue import EventQueue
from omega.tools import ToolBox
from omega.turn import TurnContext

AT = "2026-10-07T06:38:00+00:00"


def _search_item(*, status="completed", action=None):
    """A ``web_search_call`` as the live API returned it (trimmed)."""
    if action is None:
        action = SimpleNamespace(
            type="search",
            query="Bigg Boss 20 latest news October 2026",
            queries=[
                "Bigg Boss 20 latest news October 2026",
                "Bigg Boss season 20 latest updates India",
            ],
            sources=[
                SimpleNamespace(type="url", url="https://www.ndtv.com/topic/bigg-boss"),
                SimpleNamespace(type="url", url="https://www.hindustantimes.com/topic/bigg-boss/news"),
                SimpleNamespace(type="url", url="https://www.ndtv.com/topic/bigg-boss"),
            ],
        )
    return SimpleNamespace(type="web_search_call", id="ws_1", status=status, action=action)


def _message(text):
    return SimpleNamespace(
        type="message", content=[SimpleNamespace(type="output_text", text=text)]
    )


def _raw(*items):
    return SimpleNamespace(
        model="m",
        status="completed",
        output=list(items),
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
    )


class _Client:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.seen = response, error, []
        self.responses = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.seen.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


# --- the offer ----------------------------------------------------------------


def test_search_is_offered_beside_ring_one_not_in_it(monkeypatch) -> None:
    """Ring 1 is what omega dispatches and it is closed (DL-014). A hosted
    search never reaches the gate, so letting it into the set would make the
    gate's own claim — every tool is classified before it runs — false."""
    monkeypatch.delenv("OMEGA_WEB_SEARCH", raising=False)

    assert tools.hosted() == [{"type": "web_search"}]
    assert tools.WEB_SEARCH not in tools.TOOL_NAMES
    with pytest.raises(tools.ToolRejected):
        ToolBox(store_root=Path(".")).classify(tools.WEB_SEARCH, {"query": "x"})


@pytest.mark.parametrize("value", ["off", "0", "false", "OFF", " no "])
def test_search_can_be_switched_off_without_code(monkeypatch, value) -> None:
    monkeypatch.setenv("OMEGA_WEB_SEARCH", value)
    assert tools.hosted() == []


def test_the_sources_are_asked_for_whenever_search_is_offered() -> None:
    """Without ``include`` the item carries the query but not what was read."""
    client = _Client(_raw(_message("hi")))
    p = provider.OpenAIProvider(client=client, models={provider.ACT: "m"})

    p.complete(provider.ACT, [provider.user("x")], tools.schemas() + tools.hosted())
    p.complete(provider.ACT, [provider.user("x")], tools.schemas())

    assert client.seen[0]["include"] == ["web_search_call.action.sources"]
    assert {"type": "web_search"} in client.seen[0]["tools"]
    assert "include" not in client.seen[1]


def test_a_model_that_rejects_search_names_the_switch() -> None:
    """The failure turns *every* act call into a 400, so the error has to say
    which line of `.env` ends it."""
    client = _Client(error=RuntimeError("Tool 'web_search' is not supported with this model."))
    p = provider.OpenAIProvider(client=client, models={provider.ACT: "m"})

    with pytest.raises(provider.ProviderError, match="OMEGA_WEB_SEARCH=off"):
        p.complete(provider.ACT, [provider.user("x")], tools.hosted())


# --- the reading --------------------------------------------------------------


def test_a_search_comes_back_as_what_was_asked_and_what_was_read() -> None:
    client = _Client(_raw(_search_item(), _message("Rhiti was evicted.")))
    p = provider.OpenAIProvider(client=client, models={provider.ACT: "m"})

    got = p.complete(provider.ACT, [provider.user("x")], tools.hosted())

    assert got.text == "Rhiti was evicted."
    assert got.tool_calls == ()
    (search,) = got.hosted
    assert search.tool == "web_search" and search.ok
    assert search.args["action"] == "search"
    assert search.args["query"] == "Bigg Boss 20 latest news October 2026"
    assert len(search.args["queries"]) == 2
    # Deduplicated, in the order they were read.
    assert search.result.splitlines() == [
        "https://www.ndtv.com/topic/bigg-boss",
        "https://www.hindustantimes.com/topic/bigg-boss/news",
    ]


def test_a_page_open_is_logged_by_its_url() -> None:
    item = _search_item(
        action={"type": "open_page", "url": "https://example.org/a"}
    )
    (search,) = provider.OpenAIProvider(
        client=_Client(_raw(item, _message("ok"))), models={provider.ACT: "m"}
    ).complete(provider.ACT, [provider.user("x")], tools.hosted()).hosted

    assert search.args == {"action": "open_page", "url": "https://example.org/a"}
    assert search.result == "(no sources reported)"


def test_a_failed_search_is_recorded_as_failed_not_dropped() -> None:
    (search,) = provider.OpenAIProvider(
        client=_Client(_raw(_search_item(status="failed"), _message("could not"))),
        models={provider.ACT: "m"},
    ).complete(provider.ACT, [provider.user("x")], tools.hosted()).hosted

    assert not search.ok
    assert search.error == "search failed"


def test_an_unfamiliar_search_shape_never_fails_the_answer() -> None:
    """The search already happened and the answer is in the same response;
    failing the call over the shape of the *record* would discard a good
    answer to protect a log line."""
    weird = SimpleNamespace(type="web_search_call", status="completed", action=None)

    got = provider.OpenAIProvider(
        client=_Client(_raw(weird, _message("answer"))), models={provider.ACT: "m"}
    ).complete(provider.ACT, [provider.user("x")], tools.hosted())

    assert got.text == "answer"
    assert got.hosted[0].args == {"action": "unknown"}


def test_a_search_with_no_answer_is_still_no_content() -> None:
    """A search is not an answer. A response holding only a search record and
    no message must still be refused, or an outage reads as silence."""
    with pytest.raises(provider.ProviderError, match="no content"):
        provider.OpenAIProvider(
            client=_Client(_raw(_search_item())), models={provider.ACT: "m"}
        ).complete(provider.ACT, [provider.user("x")], tools.hosted())


# --- the log ------------------------------------------------------------------


@pytest.fixture
def q(store: MemoryStore) -> EventQueue:
    return EventQueue(store)


def _ctx(q: EventQueue, fp: provider.FakeProvider) -> TurnContext:
    seq = q.append(episodes.inbound("any news?", channel="tray", at=AT))
    q.claim(seq)
    return TurnContext(seq=seq, event=q.at(seq).payload, recalled=[], queue=q, complete=fp.complete)


def _tool_episodes(q: EventQueue, seq: int) -> list[dict]:
    return [
        p.payload
        for p in q.recent(50)
        if p.payload.get("kind", "").startswith("tool.") and p.payload.get("for_seq") == seq
    ]


def test_every_search_lands_in_the_log_as_a_tool_pair(q: EventQueue, tmp_path: Path) -> None:
    """The violation check. Hosted was chosen over our own tool on the
    condition that a search leaves the same trace any tool does."""
    answer = provider.Response(
        text="Rhiti was evicted.",
        model="m",
        hosted=(
            provider.HostedCall(
                tool="web_search", args={"action": "search", "query": "bigg boss 20"},
                result="https://www.ndtv.com/topic/bigg-boss",
            ),
            provider.HostedCall(
                tool="web_search", args={"action": "search", "query": "rhiti"},
                ok=False, error="search failed",
            ),
        ),
    )
    ctx = _ctx(q, provider.FakeProvider({provider.ACT: [answer]}))

    result = act_loop(ctx, box=ToolBox(store_root=tmp_path))

    assert result.text == "Rhiti was evicted."
    assert result.tools == ("web_search",)  # only the search that worked
    logged = _tool_episodes(q, ctx.seq)
    assert [(e["kind"], e["tool"]) for e in logged] == [
        (episodes.TOOL_CALLED, "web_search"),
        (episodes.TOOL_RETURNED, "web_search"),
        (episodes.TOOL_CALLED, "web_search"),
        (episodes.TOOL_RETURNED, "web_search"),
    ]
    assert logged[0]["args"]["query"] == "bigg boss 20"
    assert logged[1]["ok"] and "ndtv.com" in logged[1]["result"]
    assert not logged[3]["ok"] and logged[3]["error"] == "search failed"


def test_searches_are_logged_on_a_pass_that_also_calls_tools(
    q: EventQueue, tmp_path: Path
) -> None:
    """Search then read: the search must not be lost because the pass went on
    to ask for a Ring 1 tool."""
    (tmp_path / "notes.txt").write_text("hello")
    first = provider.Response(
        text="",
        model="m",
        tool_calls=(provider.ToolCall(id="c1", name="read_file", arguments={"path": str(tmp_path / "notes.txt")}),),
        hosted=(provider.HostedCall(tool="web_search", args={"action": "search", "query": "q"}, result="u"),),
    )
    ctx = _ctx(q, provider.FakeProvider({provider.ACT: [first, "done"]}))

    act_loop(ctx, box=ToolBox(store_root=tmp_path))

    tools_logged = [e["tool"] for e in _tool_episodes(q, ctx.seq) if e["kind"] == episodes.TOOL_CALLED]
    assert tools_logged == ["web_search", "read_file"]
