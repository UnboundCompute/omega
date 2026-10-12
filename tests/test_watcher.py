"""DL-096 — page watches: a deterministic check of a page, alerting on a condition.

The model sets a watch up on their own message; every check after that is a
read and a comparison in code. **No socket, no key**: the page is a scripted
``fetch``, the MCP servers are fakes, every provider is scripted and every
clock is pinned.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from omega import access, episodes, mcp, provider, tools, watcher
from omega.act import act_loop
from omega.executor import Executor
from omega.memory import MemoryStore
from omega.provider import ToolCall
from omega.queue import EventQueue
from omega.tools import EXPLORATION, LOCAL, Decision, ToolBox, ToolError, ToolRejected

NOW = datetime(2026, 10, 11, 12, 0, tzinfo=timezone.utc)
AT = NOW.isoformat()


def later(minutes: float) -> str:
    return (NOW + timedelta(minutes=minutes)).isoformat()


class Page:
    """The scripted web: one text per url, swapped by a test; counts reads."""

    def __init__(self, text: str = "") -> None:
        self.text = text
        self.reads: list[str] = []
        self.fail: str = ""

    def __call__(self, url: str, **_: Any) -> str:
        self.reads.append(url)
        if self.fail:
            raise ToolError(self.fail)
        return f"200 {url}\n\n{self.text}"


@pytest.fixture
def page(monkeypatch: pytest.MonkeyPatch) -> Page:
    p = Page()
    monkeypatch.setattr(tools, "fetch", p)
    monkeypatch.setattr(tools, "_now", lambda: NOW.astimezone())
    monkeypatch.delenv("OMEGA_MCP_SERVERS", raising=False)
    return p


@pytest.fixture
def q(store: MemoryStore) -> EventQueue:
    return EventQueue(store)


def payloads(q: EventQueue) -> list[dict]:
    return [episodes.decode(e.payload) for e in q.store.episodes_since(0)]


def kinds(q: EventQueue, kind: str) -> list[dict]:
    return [p for p in payloads(q) if p.get("kind") == kind]


def tells(q: EventQueue) -> list[str]:
    return [tools.told(p) for p in kinds(q, episodes.MESSAGE_INBOUND) if tools.is_tell(p)]


def box(q: EventQueue, tmp_path: Path, *, remote: Any = None, prompted: bool = True) -> ToolBox:
    seq = q.append(episodes.inbound("tell me when it drops", channel="tray", at=AT))
    root = tmp_path / "box"
    root.mkdir(exist_ok=True)
    return ToolBox(
        store_root=root, append=q.append, for_seq=seq, prompted=prompted, at=AT,
        remote=remote, pages=lambda: watcher.fold(q.store),
    )


def set_watch(q: EventQueue, tmp_path: Path, remote: Any = None, **args: Any) -> str:
    b = box(q, tmp_path, remote=remote)
    return b.dispatch(b.classify(tools.WATCH_PAGE, {"do": "set", **args}))


def executor(q: EventQueue, fp: provider.FakeProvider | None = None) -> Executor:
    ex = Executor(q, complete=(fp or provider.FakeProvider({})).complete, act=act_loop)
    ex.recover()
    return ex


def only(q: EventQueue) -> watcher.Watch:
    [w] = watcher.fold(q.store).oldest()
    return w


# --- the rules: all code --------------------------------------------------------


@pytest.mark.parametrize(
    "text, label, value",
    [
        ("Deal of the day. Price: ₹1,23,456 incl. taxes", None, 123456.0),
        ("Now Rs. 499 only", None, 499.0),
        ("Was $15, now $12.50", None, 15.0),
        ("1,299 INR\nMRP 1,999", None, 1299.0),
        ("Rating 4.5\nSale price ₹38,990\nMRP ₹45,000", "sale price", 38990.0),
        ("Seats left: 12 of 300", "seats left", 12.0),
        ("no money here, only 42 words", None, None),
        ("Seats gone", "seats left", None),
    ],
)
def test_number_rule(text: str, label: str | None, value: float | None) -> None:
    assert watcher.number_in(text, label) == value


def test_digest_ignores_spacing_and_blank_lines_only() -> None:
    assert watcher.digest("a  b\n\n c \n") == watcher.digest("a b\nc")
    assert watcher.digest("a b\nc") != watcher.digest("a b\nd")


def test_conditions_hold_in_code() -> None:
    assert watcher.holds({"kind": "contains", "text": "Tickets  ON sale"}, "x\ntickets on sale now")[0]
    assert watcher.holds({"kind": "disappears", "text": "sold out"}, "book now")[0]
    assert watcher.holds({"kind": "below", "number": 40000}, "₹39,999") == (True, 39999.0)
    assert watcher.holds({"kind": "above", "number": 40000}, "₹39,999") == (False, 39999.0)
    assert watcher.holds({"kind": "below", "number": 40000}, "free") == (False, None)


# --- setting one up: the one turn with a model ------------------------------------


def test_set_reads_once_now_and_writes_the_watch(q: EventQueue, tmp_path: Path, page: Page) -> None:
    page.text = "Phone X\nPrice ₹45,000"
    out = set_watch(q, tmp_path, url="https://shop.example/phone", condition="below", number=40000)
    [made] = kinds(q, episodes.WATCHER_SET)
    assert made["adapter"] == {"kind": "fetch", "url": "https://shop.example/phone"}
    assert made["condition"] == {"kind": "below", "number": 40000}
    assert (made["every_minutes"], made["max_per_day"]) == (60, 24)
    assert made["until"] == (NOW + timedelta(days=30)).isoformat()
    assert made["link"] == "https://shop.example/phone"
    [first] = kinds(q, episodes.WATCHER_CHECKED)
    assert first["ok"] and first["hit"] is False and first["value"] == 45000.0
    assert first["for_seq"] == made["for_seq"]
    assert kinds(q, episodes.WATCHER_ALERTED) == []
    assert "45,000" in out and made["id"] in out
    assert page.reads == ["https://shop.example/phone"]


def test_a_rule_that_holds_at_set_is_said_in_the_reply_not_told(
    q: EventQueue, tmp_path: Path, page: Page
) -> None:
    page.text = "Price ₹35,000"
    out = set_watch(q, tmp_path, url="https://shop.example/phone", condition="below", number="₹40,000")
    [alert] = kinds(q, episodes.WATCHER_ALERTED)
    assert "for_seq" in alert and "35,000" in alert["text"] and "35,000" in out
    assert tells(q) == []
    # Already said: the next check, the same page, says nothing again.
    ex = executor(q)
    assert ex.watch_pages(now=later(61)) is True
    assert tells(q) == []


def test_set_refuses_what_cannot_be_watched(q: EventQueue, tmp_path: Path, page: Page) -> None:
    page.text = "no number here"
    with pytest.raises(ToolError, match="no price"):
        set_watch(q, tmp_path, url="https://shop.example/x", condition="below", number=10)
    page.fail = "https://shop.example/x answered 404 Not Found"
    with pytest.raises(ToolError, match="404"):
        set_watch(q, tmp_path, url="https://shop.example/x", condition="changed")
    assert kinds(q, episodes.WATCHER_SET) == [], "fails closed: nothing set"
    b = box(q, tmp_path)
    for bad in (
        {"do": "set", "condition": "changed"},
        {"do": "set", "url": "ftp://x", "condition": "changed"},
        {"do": "set", "url": "https://x.example", "condition": "nearly"},
        {"do": "set", "url": "https://x.example", "condition": "contains"},
        {"do": "set", "url": "https://x.example", "condition": "below"},
        {"do": "set", "url": "https://x.example", "condition": "changed", "every_minutes": 5},
        {"do": "set", "url": "https://x.example", "condition": "changed", "max_per_day": 500},
        {"do": "stop", "id": "w999"},
        {"do": "pay"},
    ):
        with pytest.raises(ToolRejected):
            b.classify(tools.WATCH_PAGE, bad)


def test_offered_only_on_their_own_message_with_the_log(q: EventQueue, tmp_path: Path) -> None:
    names = lambda b: [s["function"]["name"] for s in b.schemas() if "function" in s]  # noqa: E731
    assert tools.WATCH_PAGE in names(box(q, tmp_path))
    assert tools.WATCH_PAGE not in names(box(q, tmp_path, prompted=False))
    bare = ToolBox(store_root=tmp_path / "box", append=q.append, prompted=True)
    assert tools.WATCH_PAGE not in names(bare)
    with pytest.raises(ToolRejected, match="no tool called"):
        box(q, tmp_path, prompted=False).classify(tools.WATCH_PAGE, {"do": "list"})
    assert tools.WATCH_PAGE not in tools.TOOL_NAMES


def test_list_stop_and_renew(q: EventQueue, tmp_path: Path, page: Page) -> None:
    page.text = "Sold out"
    set_watch(q, tmp_path, url="https://tickets.example/show", condition="appears",
              text="tickets on sale", name="the show")
    b = box(q, tmp_path)
    listed = b.classify(tools.WATCH_PAGE, {"do": "list"})
    assert listed.tier == EXPLORATION
    wid = only(q).id
    assert wid in b.dispatch(listed) and "the show" in b.dispatch(listed)
    renewed = b.classify(tools.WATCH_PAGE, {"do": "renew", "id": wid})
    assert renewed.tier == LOCAL
    b.dispatch(renewed)
    assert only(q).until == (NOW + timedelta(days=30)).isoformat()
    assert only(q).last_hit is False, "renewing keeps what the checks saw"
    b.dispatch(b.classify(tools.WATCH_PAGE, {"do": "stop", "id": wid}))
    assert watcher.fold(q.store).watches == {}
    [stopped] = kinds(q, episodes.WATCHER_STOPPED)
    assert stopped["reason"] == "asked"


def test_set_through_the_act_loop_end_to_end(q: EventQueue, page: Page) -> None:
    """Production wiring: their message, the model's one call, the record."""
    page.text = "Sold out"
    fp = provider.FakeProvider({
        provider.JUDGE: lambda role, m: "ACT",
        provider.ACT: [
            ToolCall(id="c1", name=tools.WATCH_PAGE, arguments={
                "do": "set", "url": "https://tickets.example/show",
                "condition": "appears", "text": "tickets on sale",
            }),
            "I'll tell you when tickets open.",
        ],
    })
    q.append(episodes.inbound("tell me when tickets open on tickets.example/show",
                              channel="tray", at=AT))
    ex = executor(q, fp)
    while next(q.pending(), None) is not None:
        ex.step()
    [made] = kinds(q, episodes.WATCHER_SET)
    assert made["condition"] == {"kind": "appears", "text": "tickets on sale"}
    offered = fp.offers_for(provider.ACT)[0]
    assert tools.WATCH_PAGE in [s["function"]["name"] for s in offered if "function" in s]
    calls = len(fp.calls)
    # Every check after that is code: no model call at all.
    for n in range(1, 6):
        ex.watch_pages(now=later(61 * n))
    assert len(fp.calls) == calls
    assert len(kinds(q, episodes.WATCHER_CHECKED)) == 6


# --- checks: code only ---------------------------------------------------------------


def test_a_price_drop_alerts_once_with_the_link(q: EventQueue, tmp_path: Path, page: Page) -> None:
    page.text = "Price ₹45,000"
    set_watch(q, tmp_path, url="https://shop.example/phone", condition="below", number=40000)
    ex = executor(q)
    assert ex.watch_pages(now=later(30)) is False, "not due before its interval"
    page.text = "Price ₹38,990"
    assert ex.watch_pages(now=later(60)) is True
    [told] = tells(q)
    assert "38,990" in told and "below 40,000" in told
    assert "https://shop.example/phone" in told
    page.text = "Price ₹37,500 today"
    ex.watch_pages(now=later(120))
    assert len(tells(q)) == 1, "still below: no second alert for the same crossing"
    page.text = "Price ₹41,000"
    ex.watch_pages(now=later(180))
    page.text = "Price ₹39,000"
    ex.watch_pages(now=later(240))
    assert len(tells(q)) == 2, "a new crossing alerts again"


def test_the_same_page_crossing_again_is_not_told_twice(q: EventQueue, tmp_path: Path, page: Page) -> None:
    page.text = "Price ₹45,000"
    set_watch(q, tmp_path, url="https://shop.example/phone", condition="below", number=40000)
    ex = executor(q)
    page.text = "Price ₹38,990"
    ex.watch_pages(now=later(60))
    page.text = "Price ₹41,000"
    ex.watch_pages(now=later(120))
    page.text = "Price ₹38,990"
    ex.watch_pages(now=later(180))
    assert len(tells(q)) == 1, "deduped by content digest: the same page, already told"
    assert len(kinds(q, episodes.WATCHER_ALERTED)) == 1


def test_changed_baselines_then_alerts_each_new_digest(q: EventQueue, tmp_path: Path, page: Page) -> None:
    page.text = "Menu\nSoup"
    set_watch(q, tmp_path, url="https://cafe.example/menu", condition="changed")
    [baseline] = kinds(q, episodes.WATCHER_CHECKED)
    assert baseline["hit"] is False, "the first read is the baseline, not a change"
    ex = executor(q)
    ex.watch_pages(now=later(60))
    assert tells(q) == [], "the same page is no change"
    page.text = "Menu\nSoup\nNew: pie"
    ex.watch_pages(now=later(120))
    [told] = tells(q)
    assert "New: pie" in told
    page.text = "Menu\n  Soup\n\nNew: pie  "
    ex.watch_pages(now=later(180))
    assert len(tells(q)) == 1, "spacing is not a change"


def test_appears_needs_a_crossing_and_disappears_too(q: EventQueue, tmp_path: Path, page: Page) -> None:
    page.text = "Tickets on sale"
    set_watch(q, tmp_path, url="https://tickets.example/a", condition="appears", text="tickets on sale")
    assert kinds(q, episodes.WATCHER_ALERTED) == [], "already there is not an appearance"
    ex = executor(q)
    page.text = "Sold out"
    ex.watch_pages(now=later(60))
    page.text = "Tickets on sale!"
    ex.watch_pages(now=later(120))
    assert len(tells(q)) == 1 and "appeared" in tells(q)[0]


def test_errors_back_off_and_are_told_once(q: EventQueue, tmp_path: Path, page: Page) -> None:
    page.text = "ok"
    set_watch(q, tmp_path, url="https://flaky.example", condition="changed", every_minutes=15)
    ex = executor(q)
    page.fail = "could not fetch https://flaky.example: timed out"
    t = 0.0
    waits = []
    for _ in range(6):
        step = 15
        while not ex.watch_pages(now=later(t + step)):
            step += 15
        t += step
        waits.append(step)
    assert waits == [15, 30, 60, 120, 240, 480], "doubling after each error"
    failed = [c for c in kinds(q, episodes.WATCHER_CHECKED) if not c["ok"]]
    assert len(failed) == 6 and "timed out" in failed[0]["error"]
    [told] = tells(q)
    assert "failed 5 checks" in told
    page.fail = ""
    assert ex.watch_pages(now=later(t + 960))
    assert only(q).errors == 0 and len(tells(q)) == 1


def test_backoff_is_capped_at_a_day() -> None:
    w = watcher.Watch(
        id="w1", name="x", adapter={"kind": "fetch", "url": "https://x.example"},
        condition={"kind": "changed"}, every_minutes=60, max_per_day=24,
        until=(NOW + timedelta(days=30)).isoformat(), summarize=False, link=None,
        since=AT, checked_at=AT, errors=9,
    )
    assert not watcher.due(w, NOW + timedelta(hours=23))
    assert watcher.due(w, NOW + timedelta(hours=24))


def test_at_most_max_per_day_checks(q: EventQueue, tmp_path: Path, page: Page) -> None:
    page.text = "x"
    set_watch(q, tmp_path, url="https://x.example", condition="changed",
              every_minutes=15, max_per_day=3)
    ex = executor(q)
    ran = [ex.watch_pages(now=later(15 * n)) for n in range(1, 6)]
    # The set's own check is the first of the day's three.
    assert ran == [True, True, False, False, False]
    assert len(kinds(q, episodes.WATCHER_CHECKED)) == 3


def test_expires_after_thirty_days_with_one_tell(q: EventQueue, tmp_path: Path, page: Page) -> None:
    page.text = "x"
    set_watch(q, tmp_path, url="https://x.example", condition="changed")
    ex = executor(q)
    assert ex.watch_pages(now=(NOW + timedelta(days=30)).isoformat()) is True
    [stopped] = kinds(q, episodes.WATCHER_STOPPED)
    assert stopped["reason"] == "expired"
    [told] = tells(q)
    assert "30 days" in told and "renew" in told
    assert ex.watch_pages(now=(NOW + timedelta(days=31)).isoformat()) is False
    assert len(tells(q)) == 1


def test_summarize_is_one_learn_call_and_only_on_an_alert(q: EventQueue, tmp_path: Path, page: Page) -> None:
    page.text = "Menu\nSoup"
    set_watch(q, tmp_path, url="https://cafe.example/menu", condition="changed", summarize=True)
    learned: list[Any] = []
    fp = provider.FakeProvider({provider.LEARN: lambda role, m: learned.append(m) or "Pie is on the menu now."})
    ex = executor(q, fp)
    ex.watch_pages(now=later(60))
    assert learned == [], "no change, no call"
    page.text = "Menu\nSoup\nPie"
    ex.watch_pages(now=later(120))
    assert len(learned) == 1
    [told] = tells(q)
    assert told.startswith("Pie is on the menu now.")
    page.text = "Menu\nSoup\nPie\nTart"
    fp2 = provider.FakeProvider({provider.LEARN: lambda role, m: (_ for _ in ()).throw(RuntimeError("down"))})
    ex2 = executor(q, fp2)
    ex2.watch_pages(now=later(180))
    assert tells(q)[-1].startswith("cafe.example changed"), "a failed summary sends it plain"


def test_restart_neither_repeats_nor_forgets(q: EventQueue, tmp_path: Path, page: Page) -> None:
    page.text = "Price ₹45,000"
    set_watch(q, tmp_path, url="https://shop.example/phone", condition="below", number=40000)
    page.text = "Price ₹38,000"
    executor(q).watch_pages(now=later(60))
    incremental = watcher.Watches()
    incremental.advance(q.store)
    assert incremental.state() == watcher.fold(q.store)
    ex2 = executor(q)
    assert ex2.watch_pages(now=later(90)) is False, "the interval survives a restart"
    ex2.watch_pages(now=later(120))
    assert len(tells(q)) == 1


# --- MCP: the person's scrape tool, gated as any MCP read ----------------------------


class FakeRemote:
    def __init__(self, tools_: tuple[str, ...] = ("fire__firecrawl_scrape",)) -> None:
        self.names = tools_
        self.ran: list[Decision] = []
        self.refuse = False
        self.text = "Price ₹45,000"

    def schemas(self) -> list[dict]:
        return [{"type": "function", "function": {"name": n}} for n in self.names]

    def classify(self, name: str, args: dict) -> Any:
        server, _, bare = name.partition("__")
        why = mcp.money_refusal(bare)
        if why:
            raise ToolRejected(why)
        if name not in self.names:
            return None
        return Decision(tool=name, tier=EXPLORATION, why="read", args=dict(args))

    def knows(self, name: str) -> bool:
        return name in self.names

    def dispatch(self, decision: Decision) -> str:
        self.ran.append(decision)
        server = decision.tool.split("__")[0]
        if self.refuse:
            raise ToolError(f"the {server} MCP server refused the credentials (401); check its token")
        return mcp.RESULT_LEAD.format(server=server) + "\n" + self.text


def test_a_configured_scrape_tool_reads_the_page(
    q: EventQueue, tmp_path: Path, page: Page, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote = FakeRemote()
    monkeypatch.setattr(mcp, "remote", lambda *a, **k: remote)
    set_watch(q, tmp_path, remote=remote, url="https://shop.example/phone",
              condition="below", number=40000)
    assert only(q).adapter == {
        "kind": "mcp", "tool": "fire__firecrawl_scrape",
        "args": {"url": "https://shop.example/phone", "formats": ["markdown"]},
    }
    assert page.reads == [], "the built-in fetch is not used"
    remote.text = "Price ₹39,000"
    executor(q).watch_pages(now=later(60))
    assert len(remote.ran) == 2 and "39,000" in tells(q)[0]
    assert "https://shop.example/phone" in tells(q)[0]


def test_mcp_refusals_feed_access_and_pause_the_watch(
    q: EventQueue, tmp_path: Path, page: Page, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote = FakeRemote()
    monkeypatch.setattr(mcp, "remote", lambda *a, **k: remote)
    set_watch(q, tmp_path, remote=remote, url="https://shop.example/phone", condition="changed")
    remote.refuse = True
    ex = executor(q)
    assert ex.watch_pages(now=later(60)) is True
    state = access.Access.rebuild(q.store).state()
    assert [o.origin for o in state.owed] == ["mcp:fire"]
    ex.notice(now=NOW + timedelta(minutes=61))
    assert access.Access.rebuild(q.store).state().lost.keys() == {"mcp:fire"}
    ran = len(remote.ran)
    assert ex.watch_pages(now=later(60 * 24 * 3)) is False, "paused while the sign-in is lost"
    assert len(remote.ran) == ran


def test_an_mcp_read_tool_and_never_a_money_one(
    q: EventQueue, tmp_path: Path, page: Page, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote = FakeRemote(("shop__get_price", "shop__checkout_cart"))
    remote.text = "₹500"
    monkeypatch.setattr(mcp, "remote", lambda *a, **k: remote)
    set_watch(q, tmp_path, remote=remote, tool="shop__get_price", args={"sku": "a1"},
              condition="below", number=400)
    assert only(q).adapter == {"kind": "mcp", "tool": "shop__get_price", "args": {"sku": "a1"}}
    assert only(q).link is None
    b = box(q, tmp_path, remote=remote)
    with pytest.raises(ToolRejected, match="moves money"):
        b.classify(tools.WATCH_PAGE, {"do": "set", "tool": "shop__checkout_cart",
                                      "condition": "changed"})
    with pytest.raises(ToolRejected):
        b.classify(tools.WATCH_PAGE, {"do": "set", "tool": "shop__unlisted", "condition": "changed"})


# --- equivalence: existing watches and turns are what they were -----------------------

#: The ``watch`` and ``unwatch`` schemas as they were before DL-096, pinned by
#: hash: the LLM watch is untouched.
_WATCH_SCHEMAS = "b388242d"


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:8]


def test_the_llm_watch_is_byte_identical() -> None:
    ring = {s["function"]["name"]: s for s in tools.schemas()}
    assert _hash([ring["watch"], ring["unwatch"]]) == _WATCH_SCHEMAS
    assert tools.WATCH_PAGE not in ring
    assert tools.watch_instruction("x").startswith(tools.WATCH_LEAD)


def test_no_watch_standing_is_a_fold_and_nothing_else(q: EventQueue, page: Page) -> None:
    fp = provider.FakeProvider({})
    ex = executor(q, fp)
    before = payloads(q)
    assert ex.watch_pages(now=later(60)) is False
    assert payloads(q) == before and fp.calls == [] and page.reads == []


def test_an_existing_watch_turn_and_its_fire_are_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, page: Page
) -> None:
    """The same LLM ``watch`` set and fired, with page watches wired and with
    them forced off: the same records and the same calls; the prompted turn's
    offer differs only by ``watch_page`` appended, and the fire's not at all."""

    def scenario(root: Path, off: bool) -> tuple[str, list, list]:
        root.mkdir()
        s = MemoryStore.open(root)
        try:
            qq = EventQueue(s)
            fp = provider.FakeProvider({
                provider.JUDGE: lambda role, m: "ACT",
                provider.ACT: [
                    ToolCall(id="c1", name="watch", arguments={"what": "the build", "every_minutes": 60}),
                    "Watching.",
                    "Nothing new.",
                ],
            })
            with monkeypatch.context() as m:
                if off:
                    m.setattr(tools, "_watching_pages", lambda box: False)
                    m.setattr(Executor, "watch_pages", lambda self, now=None: False)
                ex = executor(qq, fp)
                qq.append(episodes.inbound("keep an eye on the build", channel="tray", at=AT))
                while next(qq.pending(), None) is not None:
                    ex.step()
                [made] = [p for p in payloads(qq) if p["kind"] == episodes.SCHEDULE_CREATED]
                qq.append(episodes.inbound(made["instruction"], channel="schedule",
                                           schedule_id=made["id"], at=later(60)))
                while next(qq.pending(), None) is not None:
                    ex.step()
                ex.watch_pages(now=later(61))
            recs = [
                {k: v for k, v in p.items() if k not in ("at", "elapsed_ms", "ms")}
                for p in payloads(qq)
            ]
            same = lambda x: json.dumps(x, sort_keys=True, default=str).replace(str(root), "<root>")  # noqa: E731
            return same(recs), [same(c) for c in fp.calls], fp.offers_for(provider.ACT)
        finally:
            s.close()

    on_recs, on_calls, on_offers = scenario(tmp_path / "on", off=False)
    off_recs, off_calls, off_offers = scenario(tmp_path / "off", off=True)
    assert on_recs == off_recs and on_calls == off_calls
    assert len(on_offers) == len(off_offers) == 3
    page_schema = [s for s in on_offers[0] if s.get("function", {}).get("name") == tools.WATCH_PAGE]
    assert len(page_schema) == 1
    for on, off in zip(on_offers[:2], off_offers[:2]):
        assert [s for s in on if s not in page_schema] == off, "their turn: watch_page appended, nothing else"
    assert on_offers[2] == off_offers[2], "the fire is unprompted: offered exactly as before"
