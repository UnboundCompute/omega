"""DL-101a — the judge is told which hands are on, so a Slack ask acts.

Live, with Slack on: "post X to #channel" from Discord was judged SPEAK, and
the reply path (which holds no tools) said truthfully that it could not send
Slack messages from there. The judge's "you have a body" sentence never named
Slack. These cases pin the fix: the judge and the reply are told what this
turn's act box will hold, decided from structure (their own message, a token
set, a server configured) and never from the message's words.

No network and no key: Slack is :class:`tests.test_slack.FakeSlack` behind
the module's one HTTPS seam, and every model answer is `FakeProvider`'s.
"""

from __future__ import annotations

import hashlib
from typing import Any

import pytest

from omega import act as act_mod
from omega import episodes, mcp, provider, slack, tools, turn
from omega.act import act_loop
from omega.derive import OpenWork
from omega.grants import Grants
from omega.memory import MemoryStore
from omega.provider import ToolCall
from omega.queue import EventQueue
from omega.slack import Client
from omega.turn import ActResult, TurnContext, recall, run_turn

from test_slack import LINK, TOKEN, FakeSlack

AT = "2026-10-11T09:00:00+00:00"

#: sha256 of origin/main's (30a4f53) `_JUDGE_SYSTEM` and `_REPLY_SYSTEM`,
#: the prompts before this change. Nothing configured must render exactly these.
MAIN_JUDGE = "d69895d31bc2f5d0bde6b9bee72bdc947fc7c41ae64a8914d373369994336870"
MAIN_REPLY = "47684c1996fe8ef15cb897c26fb105247458bad4521be3ebbf740e5153f4f80d"

SLACK_WORDS = "post to their work Slack as them"


@pytest.fixture(autouse=True)
def bare(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(episodes, "now", lambda: AT)
    monkeypatch.delenv(slack.ENV_TOKEN, raising=False)
    for key in list(__import__("os").environ):
        if key.startswith("OMEGA_MCP_"):
            monkeypatch.delenv(key)


@pytest.fixture
def q(store: MemoryStore) -> EventQueue:
    return EventQueue(store)


@pytest.fixture
def fake() -> FakeSlack:
    return FakeSlack()


@pytest.fixture
def on(monkeypatch: pytest.MonkeyPatch, fake: FakeSlack) -> Client:
    """Their token set, with the process's client fake-backed."""
    held = Client(TOKEN, http=fake, sleep=lambda _s: None)
    monkeypatch.setenv(slack.ENV_TOKEN, TOKEN)
    monkeypatch.setattr(slack, "client", lambda environ=None: held)
    return held


def _maps_and_github(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(mcp.ENV_SERVERS, "maps,github")
    monkeypatch.setenv("OMEGA_MCP_MAPS_URL", "https://maps.example.com/mcp")
    monkeypatch.setenv("OMEGA_MCP_MAPS_TOOLS", "search_places,compute_routes")
    monkeypatch.setenv("OMEGA_MCP_GITHUB_URL", "https://github.example.com/mcp")
    monkeypatch.setenv("OMEGA_MCP_GITHUB_TOOLS", "list_pull_requests")


def _ctx(q: EventQueue, text: str = "post the plan to #team", channel: str = "discord",
         **more: Any) -> TurnContext:
    seq = q.append(episodes.inbound(text, channel=channel, at=AT, **more))
    q.claim(seq)
    pending = q.at(seq)
    return TurnContext(
        seq=pending.seq, event=pending.payload,
        recalled=recall(q, before=pending.seq), queue=q,
        complete=lambda *a, **k: None,
    )


def _system(messages: list[provider.Message]) -> str:
    return str(messages[0]["content"])


def _judged(ctx: TurnContext) -> str:
    return _system(turn._judge_messages(ctx))


def _replied(ctx: TurnContext) -> str:
    return _system(turn._reply_messages(ctx, ActResult(tools=())))


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# --- equivalence -----------------------------------------------------------------


@pytest.mark.parametrize("channel,more", [
    ("discord", {}), ("tray", {}), ("self", {}), ("self", {"thinking": "what is open today"}), ("schedule", {}),
])
def test_nothing_configured_renders_mains_prompts_byte_for_byte(
    q: EventQueue, channel: str, more: dict,
) -> None:
    ctx = _ctx(q, channel=channel, **more)
    assert turn.hands(ctx) == []
    assert _sha(_judged(ctx)) == MAIN_JUDGE and _sha(turn._JUDGE_SYSTEM) == MAIN_JUDGE
    assert _sha(_replied(ctx)) == MAIN_REPLY and _sha(turn._REPLY_SYSTEM) == MAIN_REPLY


# --- slack ------------------------------------------------------------------------


@pytest.mark.parametrize("channel", ["discord", "tray"])
def test_their_own_message_with_the_token_names_slack(q: EventQueue, on: Client, channel: str) -> None:
    ctx = _ctx(q, channel=channel)
    judged = _judged(ctx)
    assert SLACK_WORDS in judged and "choose ACT" in judged.split(SLACK_WORDS)[1]
    assert "send something to Slack" in judged
    assert SLACK_WORDS in _replied(ctx)
    # Everything main said is still said, around the addition.
    assert judged.replace(judged.split("and set a reminder. ")[1].split("ACT is the verdict")[0], "") \
        == turn._JUDGE_SYSTEM


@pytest.mark.parametrize("channel,more", [
    ("self", {}), ("self", {"thinking": "what is open today"}), ("schedule", {}), ("task", {}),
])
def test_slack_is_not_named_on_a_turn_they_did_not_start(
    q: EventQueue, on: Client, channel: str, more: dict,
) -> None:
    ctx = _ctx(q, channel=channel, **more)
    assert "Slack" not in _judged(ctx) and "Slack" not in _replied(ctx)
    assert _sha(_judged(ctx)) == MAIN_JUDGE and _sha(_replied(ctx)) == MAIN_REPLY


def test_slack_is_not_named_without_the_token(q: EventQueue) -> None:
    assert slack.client() is None
    ctx = _ctx(q)
    assert "Slack" not in _judged(ctx) and "Slack" not in _replied(ctx)


@pytest.mark.parametrize("channel,more", [
    ("discord", {}), ("tray", {}), ("self", {}), ("self", {"thinking": "what is open today"}), ("schedule", {}),
])
def test_the_judge_names_slack_exactly_where_the_act_box_seats_it(
    q: EventQueue, on: Client, channel: str, more: dict,
) -> None:
    ctx = _ctx(q, channel=channel, **more)
    assert (SLACK_WORDS in _judged(ctx)) == (act_mod._slack_seat(ctx) is not None)


def test_the_offer_reads_structure_not_words(q: EventQueue, on: Client) -> None:
    plain = _ctx(q, text="what time is it")
    assert SLACK_WORDS in _judged(plain)
    look = _ctx(q, text="post to slack #team slack slack", channel="self")
    assert SLACK_WORDS not in _judged(look)


# --- other configured hands -------------------------------------------------------


def test_mcp_servers_are_named_only_when_configured(
    q: EventQueue, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = _ctx(q, text="how long to the office")
    assert "Maps" not in _judged(ctx) and "GitHub" not in _judged(ctx)
    _maps_and_github(monkeypatch)
    judged, replied = _judged(ctx), _replied(ctx)
    for words in ("find places and routes on Maps", "read GitHub",
                  "tell them when to leave by live traffic"):
        assert words in judged and words in replied
    assert "Slack" not in judged


def test_a_server_with_nothing_allowed_is_not_a_hand(
    q: EventQueue, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(mcp.ENV_SERVERS, "notes")
    monkeypatch.setenv("OMEGA_MCP_NOTES_URL", "https://notes.example.com/mcp")
    ctx = _ctx(q)
    assert turn.hands(ctx) == [] and _sha(_judged(ctx)) == MAIN_JUDGE
    monkeypatch.setenv("OMEGA_MCP_NOTES_TOOLS", "search")
    assert turn.hands(ctx) == ["use their notes tools"]


def test_the_server_tools_are_named_on_a_look_but_the_leave_alarm_is_not(
    q: EventQueue, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The box is handed MCP tools on every turn, and `leave_by` only on
    their own message; the judge is told the same."""
    _maps_and_github(monkeypatch)
    look = _ctx(q, channel="self")
    assert turn.hands(look) == ["find places and routes on Maps", "read GitHub"]
    monkeypatch.setenv("OMEGA_MCP_MAPS_TOOLS", "search_places")
    assert "tell them when to leave by live traffic" not in turn.hands(_ctx(q))


# --- the whole path ---------------------------------------------------------------


def _judge_that_reads_its_body(role: str, messages: list[provider.Message]) -> str:
    """A judge that acts only when its system prompt says Slack is a hand."""
    return "ACT" if SLACK_WORDS in _system(list(messages)) else "SPEAK"


def _run(q: EventQueue, seq: int, act: list[Any]):
    q.claim(seq)
    fp = provider.FakeProvider({provider.JUDGE: _judge_that_reads_its_body, provider.ACT: list(act)})
    result = run_turn(
        q, q.at(seq), complete=fp.complete, act=act_loop,
        open_work=OpenWork().advance(q.store, upto=seq).blocks(),
        grants=Grants().advance(q.store, upto=seq).state(),
        at=AT,
    )
    q.finish(seq)
    return result, fp


def _kinds(q: EventQueue, kind: str) -> list[dict]:
    return [p for p in (episodes.decode(e.payload) for e in q.store.episodes_since(0))
            if p.get("kind") == kind]


def test_a_slack_ask_from_discord_acts_and_posts(q: EventQueue, on: Client, fake: FakeSlack) -> None:
    seq = q.append(episodes.inbound("post the plan link to #team", channel="discord", at=AT))
    call = ToolCall(id="c1", name=tools.SLACK,
                    arguments={"do": "post", "to": "#team", "text": "plan: https://example.com/plan"})
    result, fp = _run(q, seq, [call, f"Posted: {LINK}"])
    [sent] = [p for m, _, p in fake.calls if m == "chat.postMessage"]
    assert sent == {"channel": "C0TEAM001", "text": "plan: https://example.com/plan"}
    [rec] = _kinds(q, episodes.SLACK_POSTED)
    assert rec["for_seq"] == seq and rec["permalink"] == LINK
    assert result.outcome == "spoke" and LINK in result.reply


def test_without_the_token_the_same_ask_speaks_and_posts_nothing(q: EventQueue, fake: FakeSlack) -> None:
    seq = q.append(episodes.inbound("post the plan link to #team", channel="discord", at=AT))
    result, fp = _run(q, seq, ["I can't post to Slack from here."])
    assert fake.calls == [] and _kinds(q, episodes.SLACK_POSTED) == []
    assert fp.offers_for(provider.ACT) == [None]
