"""DL-097 — the final handoff: links and documents, never a payment.

**No socket, no key.** Every link is built in code and compared as a string;
documents go into a temporary store; the Discord side is the real channel over
loopback with a fake DM.
"""

from __future__ import annotations

import base64
import socket
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

import pytest

from omega import discord as adapter
from omega import episodes, handoff, mcp, tools, watcher
from omega.blobs import BlobStore
from omega.channel import Channel, ChannelClient
from omega.memory import MemoryStore
from omega.queue import EventQueue
from omega.tools import EXPLORATION, LOCAL, ToolBox, ToolRejected

NOW = datetime(2026, 10, 11, 12, 0, tzinfo=timezone.utc)
AT = NOW.isoformat()


# --- links ---------------------------------------------------------------------


def test_directions_are_a_google_maps_link() -> None:
    assert handoff.directions("MG Road Metro, Bengaluru") == (
        "https://www.google.com/maps/dir/?api=1&destination=MG%20Road%20Metro%2C%20Bengaluru"
        "&travelmode=driving"
    )
    assert handoff.directions("the office", origin="home", mode="transit") == (
        "https://www.google.com/maps/dir/?api=1&origin=home&destination=the%20office&travelmode=transit"
    )
    with pytest.raises(handoff.HandoffError):
        handoff.directions("x", mode="teleport")
    with pytest.raises(handoff.HandoffError):
        handoff.directions("  ")


def test_each_service_has_its_search_page() -> None:
    assert handoff.search("amazon", "usb c cable") == "https://www.amazon.in/s?k=usb%20c%20cable"
    assert handoff.search("zomato", "biryani") == "https://www.zomato.com/search?q=biryani"
    assert handoff.search("zomato", "biryani", city="New Delhi") == (
        "https://www.zomato.com/new-delhi/restaurants?q=biryani")
    assert handoff.search("district", "comedy show") == "https://www.district.in/search?q=comedy%20show"
    assert handoff.search("ixigo", origin="blr", destination="DEL", when="2026-10-25") == (
        "https://www.ixigo.com/search/result/flight?from=BLR&to=DEL&date=25102026"
        "&adults=1&children=0&infants=0&class=e"
    )
    for bad in ({"origin": "Bangalore", "destination": "DEL", "when": "2026-10-25"},
                {"origin": "BLR", "destination": "DEL", "when": "25/10/2026"}):
        with pytest.raises(handoff.HandoffError):
            handoff.search("ixigo", **bad)


def test_an_item_link_is_kept_only_on_the_services_own_https_domain() -> None:
    assert handoff.item("amazon", "https://www.amazon.in/dp/B0EXAMPLE") == "https://www.amazon.in/dp/B0EXAMPLE"
    assert handoff.item("zomato", "https://www.zomato.com/bangalore/a-cafe/order")
    for service, url in (("amazon", "http://www.amazon.in/dp/x"),
                         ("amazon", "https://amazon.in.example.com/dp/x"),
                         ("zomato", "https://evil.example/zomato.com"),
                         ("ixigo", "javascript:alert(1)")):
        with pytest.raises(handoff.HandoffError):
            handoff.item(service, url)


def test_every_link_is_plain_https() -> None:
    made = [
        handoff.build("directions", {"destination": "the office"}),
        handoff.build("amazon", {"query": "kettle"}),
        handoff.build("zomato", {"query": "dosa", "city": "Bengaluru"}),
        handoff.build("district", {"query": "concert"}),
        handoff.build("ixigo", {"from": "BOM", "to": "GOI", "date": "2026-12-20"}),
        handoff.build("amazon", {"url": "https://www.amazon.in/dp/B0EXAMPLE"}),
        handoff.build("link", {"url": "https://tickets.example/show"}),
    ]
    for link in made:
        assert link.startswith("https://") and " " not in link and "\n" not in link, link


# --- the money hard rule, in one place --------------------------------------------


@pytest.mark.parametrize("kind", ["pay", "checkout", "book_flight", "place_order", "buy_now"])
def test_a_money_handoff_is_refused_with_the_one_sentence(kind: str) -> None:
    with pytest.raises(handoff.HandoffError) as raised:
        handoff.build(kind, {"query": "x"})
    assert str(raised.value) == mcp.money_refusal(kind)


@pytest.fixture
def no_reach(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every way out of the process, spied: a handoff may call none of them."""
    calls: list[str] = []

    def spy(name: str):
        def called(*a: Any, **k: Any) -> Any:
            calls.append(name)
            raise AssertionError(f"a handoff reached {name}")
        return called

    monkeypatch.setattr(mcp.Remote, "classify", spy("mcp.classify"))
    monkeypatch.setattr(mcp.Remote, "dispatch", spy("mcp.dispatch"))
    monkeypatch.setattr(tools, "fetch", spy("fetch"))
    monkeypatch.setattr(urllib.request, "urlopen", spy("urlopen"))
    monkeypatch.setattr(socket.socket, "connect", spy("socket.connect"))
    monkeypatch.setattr(socket, "create_connection", spy("create_connection"))
    return calls


def test_no_handoff_path_calls_any_tool(no_reach: list[str], tmp_path: Path, store: MemoryStore) -> None:
    """Each kind built straight and through the ``handoff`` tool: no MCP tool,
    no fetch and no socket is reached, so no checkout, payment, order or
    booking tool can be. The tool names a handoff could have reached would
    all be refused by :func:`omega.mcp.money_refusal` anyway."""
    for name in ("shop__checkout", "pay", "orders__place_order", "trains__book_ticket"):
        assert mcp.money_refusal(name) is not None
    q = EventQueue(store)
    seq = q.append(episodes.inbound("send me the link", channel="tray", at=AT))
    box = ToolBox(store_root=store.root, append=q.append, for_seq=seq, prompted=True, at=AT)
    every = {
        "directions": {"destination": "the office", "origin": "home", "mode": "walking"},
        "amazon": {"query": "kettle"},
        "zomato": {"query": "dosa"},
        "district": {"query": "concert"},
        "ixigo": {"from": "BLR", "to": "DEL", "date": "2026-10-25"},
        "link": {"url": "https://shop.example/item"},
        "document": {"name": "plan", "text": "1. pack\n2. leave at 6"},
    }
    assert set(every) == set(handoff.KINDS)
    for kind, args in every.items():
        out = handoff.build(kind, args, store_root=tmp_path)
        assert out
        decision = box.classify(tools.HANDOFF, {"kind": kind, **args})
        assert decision.tier == (LOCAL if kind == "document" else EXPLORATION)
        assert box.dispatch(decision)
    assert no_reach == []
    assert handoff.page("https://shop.example/x") == "https://shop.example/x"
    assert no_reach == []


def test_the_module_holds_no_way_to_act() -> None:
    """By construction: handoff imports no tool runner, client or socket, and
    takes nothing from :mod:`omega.mcp` but its money refusal."""
    import ast

    tree = ast.parse(Path(handoff.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            names = {a.name for a in node.names}
            imported |= {f"{node.module}.{n}" for n in names}
    assert imported == {
        "__future__.annotations", "re", "tempfile", "dataclasses.dataclass",
        "datetime.date", "pathlib.Path", "typing.Any", "typing.Optional",
        "urllib.parse.quote", "urllib.parse.urlencode", "urllib.parse.urlsplit",
        "omega.blobs", "omega.mcp.money_refusal",
    }


# --- documents ------------------------------------------------------------------


def test_a_document_is_a_blob_and_a_marker(tmp_path: Path) -> None:
    made = handoff.document(tmp_path, "trip plan", "Day 1: Goa\nDay 2: beach")
    assert made.name == "trip plan.md" and made.bytes == len("Day 1: Goa\nDay 2: beach")
    store = BlobStore.open(tmp_path)
    assert store.path_for(made.digest).read_text(encoding="utf-8") == "Day 1: Goa\nDay 2: beach"
    assert handoff.markers(f"Here you go. {made.marker}") == [(made.name, made.digest)]
    assert handoff.document(tmp_path, "../../etc/passwd.txt", "x").name == "passwd.txt"
    for name, text in (("", "x"), ("a", ""), ("a", "x" * (handoff.MAX_DOCUMENT_BYTES + 1))):
        with pytest.raises(handoff.HandoffError):
            handoff.document(tmp_path, name, text)


def test_the_tool_is_offered_on_their_own_message_only(store: MemoryStore) -> None:
    q = EventQueue(store)
    names = lambda b: [s["function"]["name"] for s in b.schemas() if "function" in s]  # noqa: E731
    assert tools.HANDOFF in names(ToolBox(store_root=store.root, append=q.append, prompted=True))
    assert tools.HANDOFF not in names(ToolBox(store_root=store.root, append=q.append, prompted=False))
    assert tools.HANDOFF not in names(ToolBox(store_root=store.root, prompted=True))
    with pytest.raises(ToolRejected):
        ToolBox(store_root=store.root, append=q.append, prompted=False).classify(
            tools.HANDOFF, {"kind": "amazon", "query": "x"})
    box = ToolBox(store_root=store.root, append=q.append, prompted=True)
    for bad in ({"kind": "pay"}, {"kind": "amazon"}, {"kind": "ixigo", "from": "BLR"},
                {"kind": "link", "url": "http://x.example"}, {"kind": "document", "name": "x"}):
        with pytest.raises(ToolRejected):
            box.classify(tools.HANDOFF, bad)


# --- the page watcher's alert link comes from here --------------------------------


def test_the_watchers_alert_link_is_the_handoffs(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []
    real = handoff.open_line
    monkeypatch.setattr(handoff, "open_line", lambda link: seen.append(link) or real(link))
    w = watcher.Watch(
        id="w1", name="shop.example", adapter={"kind": "fetch", "url": "https://shop.example/p"},
        condition={"kind": "below", "number": 400}, every_minutes=60, max_per_day=24,
        until=(NOW + timedelta(days=30)).isoformat(), summarize=False,
        link=watcher.link_for({"kind": "fetch", "url": "https://shop.example/p"}), since=AT,
    )
    assert w.link == handoff.page("https://shop.example/p")
    words = watcher._alert_words(w, 399.0, [], "")
    assert seen == ["https://shop.example/p"] and words.endswith("\nOpen it: https://shop.example/p")
    assert watcher.link_for({"kind": "fetch", "url": "ftp://x"}) is None


# --- Discord: the document goes as a file -------------------------------------------


@pytest.fixture
def q(store: MemoryStore) -> EventQueue:
    return EventQueue(store)


@pytest.fixture
def channel(q: EventQueue, store_dir: Path) -> Iterator[Channel]:
    ch = Channel(q, BlobStore.open(store_dir), port=0, poll=0.005)
    ch.start()
    try:
        yield ch
    finally:
        ch.stop()


class Sent:
    def __init__(self) -> None:
        self.messages: list[str] = []
        self.files: list[tuple[str, bytes]] = []

    def __call__(self, text: str) -> None:
        self.messages.append(text)

    def file(self, name: str, data: bytes) -> None:
        self.files.append((name, data))


def _out(channel: Channel, tmp_path: Path, sent: Sent, files: bool = True) -> adapter.Outbound:
    state = tmp_path / "state"
    state.mkdir(exist_ok=True)
    return adapter.Outbound(
        channel.address, adapter.Cursor(state / adapter.CURSOR_FILE), sent,
        send_file=sent.file if files else None,
        connect=lambda a: ChannelClient(a, timeout=5.0), sleep=lambda s: None, log=lambda line: None,
    )


def _reply(q: EventQueue, reply: str) -> dict[str, Any]:
    from omega import projection

    asked = q.append(episodes.inbound("plan it", channel="discord"))
    seq = q.append(episodes.completed(for_seq=asked, outcome="spoke", reply=reply))
    (pending,) = [p for p in q.recent(q.head()) if p.seq == seq]
    return projection.project_pending(pending).wire()


def test_a_document_marker_goes_to_discord_as_a_file(
    channel: Channel, q: EventQueue, store_dir: Path, tmp_path: Path
) -> None:
    made = handoff.document(store_dir, "plan", "1. pack\n2. leave at 6")
    sent = Sent()
    update = _reply(q, f"Here is the plan.\n{made.marker}")
    with ChannelClient(channel.address, timeout=5.0) as c:
        assert _out(channel, tmp_path, sent).handle(c, update)
    assert sent.messages == ["Here is the plan."]
    assert sent.files == [("plan.md", b"1. pack\n2. leave at 6")]


def test_without_file_delivery_or_a_marker_the_dm_is_unchanged(
    channel: Channel, q: EventQueue, store_dir: Path, tmp_path: Path
) -> None:
    made = handoff.document(store_dir, "plan", "x")
    text = f"Here is the plan.\n{made.marker}"
    sent = Sent()
    with ChannelClient(channel.address, timeout=5.0) as c:
        _out(channel, tmp_path, sent, files=False).handle(c, _reply(q, text))
        _out(channel, tmp_path, sent).handle(c, _reply(q, "no file here"))
    assert sent.messages == [text, "no file here"] and sent.files == []


def test_a_marker_for_a_missing_blob_stays_as_words(
    channel: Channel, q: EventQueue, tmp_path: Path
) -> None:
    text = "Plan: [attached: plan.md sha256:" + "a" * 64 + "]"
    sent = Sent()
    with ChannelClient(channel.address, timeout=5.0) as c:
        _out(channel, tmp_path, sent).handle(c, _reply(q, text))
    assert sent.messages == [text] and sent.files == []


def test_the_blob_op_serves_bytes_and_refuses_the_rest(channel: Channel, store_dir: Path) -> None:
    made = handoff.document(store_dir, "plan", "hello")
    with ChannelClient(channel.address, timeout=5.0) as c:
        answer = c.blob(made.digest)
        assert answer["op"] == "blob" and base64.b64decode(answer["data"]) == b"hello"
        assert c.blob("sha256:" + "0" * 64)["op"] == "error"
        assert c.blob("../../etc/passwd")["op"] == "error"


def test_a_turn_from_discord_hands_back_a_document_as_a_file(
    channel: Channel, q: EventQueue, store_dir: Path, tmp_path: Path
) -> None:
    """Production wiring end to end: their DM, the model's one ``handoff``
    call, its reply carrying the marker, the projection, and the DM with the
    document attached."""
    from omega import projection, provider
    from omega.act import act_loop
    from omega.executor import Executor
    from omega.provider import ToolCall

    def act(role: str, messages: list) -> Any:
        said = [str(m.get("content") or "") for m in messages if m.get("role") == "tool"]
        found = [m for s in said for m in handoff.MARKER.finditer(s)]
        if not found:
            return ToolCall(id="c1", name=tools.HANDOFF, arguments={
                "kind": "document", "name": "packing list", "text": "- charger\n- passport"})
        return f"Here is your list.\n{found[0].group(0)}"

    fp = provider.FakeProvider({provider.JUDGE: lambda role, m: "ACT", provider.ACT: act})
    asked = q.append(episodes.inbound("make me a packing list", channel="discord"))
    ex = Executor(q, complete=fp.complete, act=act_loop)
    ex.recover()
    ex.drain()
    (done,) = [p for p in q.recent(q.head()) if p.seq > asked and
               p.payload.get("kind") == episodes.TURN_COMPLETED]
    update = projection.project_pending(done).wire()
    sent = Sent()
    with ChannelClient(channel.address, timeout=5.0) as c:
        assert _out(channel, tmp_path, sent).handle(c, update)
    assert sent.messages == ["Here is your list."]
    assert sent.files == [("packing list.md", b"- charger\n- passport")]
