"""DL-087 #3 — sign-ins and access.

An expired Google sign-in, an IMAP login the server refuses, or an MCP server
answering 401/403 is told on the first refusal, once, with the fix; what reads
from it is paused (and not counted as a failure) until it works again; the
recovery is told once; and a brief lists what is still lost. **No socket, no
key**: refusals are written to the log as the act loop writes them, and every
login, refresh and listing below is a fake.
"""

from __future__ import annotations

import imaplib
import io
import json
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from omega import access, episodes, followups, gcal, inbox, mcp, provider, tools
from omega.access import Access
from omega.act import act_loop
from omega.executor import Executor
from omega.followups import Followups
from omega.grants import Grants
from omega.memory import MemoryStore
from omega.queue import EventQueue
from omega.turn import TurnContext, run_turn
from tests.test_inbox import ENV as IMAP_ENV, PASSWORD, server

NOW = datetime(2026, 10, 11, 12, 0, tzinfo=timezone.utc)
AT = NOW.isoformat()
TOKEN = "ghp_SECRETSECRETSECRET1234"

IMAP_ERROR = (
    "could not read the inbox: login refused (check OMEGA_IMAP_USER and the "
    f"app password). {inbox.IMAP_FIX}"
)
GOOGLE_ERROR = f"could not read the calendar: {gcal.SIGN_IN_AGAIN}"
MCP_ERROR = "the github MCP server refused the credentials (401); check its token"


@pytest.fixture
def q(store: MemoryStore) -> EventQueue:
    return EventQueue(store)


def payloads(q: EventQueue) -> list[dict]:
    return [episodes.decode(e.payload) for e in q.store.episodes_since(0)]


def kinds(q: EventQueue, kind: str) -> list[dict]:
    return [p for p in payloads(q) if p.get("kind") == kind]


def tells(q: EventQueue) -> list[str]:
    return [
        p["text"] for p in kinds(q, episodes.MESSAGE_INBOUND)
        if str(p.get("text", "")).startswith(tools.TELL_LEAD)
    ]


def say(q: EventQueue, text: str, channel: str = "tray", **extra: Any) -> int:
    return q.append(episodes.inbound(text, channel=channel, at=AT, **extra))


def returned(q: EventQueue, seq: int, tool: str, *, ok: bool,
             result: str | None = None, error: str | None = None) -> None:
    q.append(episodes.tool_called(for_seq=seq, tool=tool, args={}, at=AT))
    q.append(episodes.tool_returned(
        for_seq=seq, tool=tool, ok=ok, result=result, error=error, at=AT,
    ))


def asked(q: EventQueue, tool: str, *, ok: bool, result: str | None = None,
          error: str | None = None) -> int:
    """Their own message, one read, and its turn."""
    seq = say(q, "check it")
    returned(q, seq, tool, ok=ok, result=result, error=error)
    q.append(episodes.completed(for_seq=seq, outcome="spoke", reply="ok", at=AT))
    return seq


def watch(q: EventQueue, sid: str = "w1-abcd", what: str = "the inbox") -> None:
    q.append(episodes.schedule_created(
        id=sid, instruction=tools.watch_instruction(what), every=900, at=AT
    ))


def fire(q: EventQueue, sid: str = "w1-abcd", what: str = "the inbox", *,
         tool: str | None = "inbox", ok: bool = False, error: str | None = IMAP_ERROR,
         result: str | None = None, outcome: str = "failed") -> int:
    """One watch fire that read once, and how its turn ended."""
    seq = say(q, tools.watch_instruction(what), channel="schedule", schedule_id=sid)
    if tool is not None:
        returned(q, seq, tool, ok=ok, result=result, error=error)
    q.append(episodes.completed(
        for_seq=seq, outcome=outcome,
        error="RuntimeError: could not read" if outcome == "failed" else None, at=AT,
    ))
    return seq


def pending_fire(q: EventQueue, sid: str = "w1-abcd", what: str = "the inbox") -> int:
    return say(q, tools.watch_instruction(what), channel="schedule", schedule_id=sid)


def executor(q: EventQueue, **kw: Any) -> Executor:
    kw.setdefault("complete", provider.FakeProvider({}).complete)
    ex = Executor(q, act=act_loop, **kw)
    ex.recover()
    return ex


def drain(ex: Executor) -> None:
    """Close every pending turn without a model: tells deliver as written,
    and anything else is closed by hand so the queue is empty."""
    while True:
        pend = next(iter(ex.queue.pending()), None)
        if pend is None:
            return
        if not pend.is_event or tools.is_tell(pend.payload):
            ex.step()
            continue
        ex.queue.claim(pend.seq)
        ex.queue.append(episodes.completed(for_seq=pend.seq, outcome="silent", at=AT),
                        episodes.turn_write_key(pend.seq))
        ex.queue.finish(pend.seq)


def view(q: EventQueue) -> access.State:
    return Access.rebuild(q.store).state()


# --- the classifier ---------------------------------------------------------------


def _ret(tool: str, *, ok: bool, result: str | None = None, error: str | None = None) -> dict:
    return episodes.tool_returned(for_seq=1, tool=tool, ok=ok, result=result, error=error, at=AT)


@pytest.mark.parametrize("payload, want", [
    (_ret("inbox", ok=False, error=IMAP_ERROR), ("inbox", True)),
    (_ret("mail", ok=False, error=f"could not read the mail: x. {inbox.IMAP_FIX}"), ("inbox", True)),
    (_ret("inbox", ok=True, result="2 mails"), ("inbox", False)),
    (_ret("inbox", ok=True, result=inbox.NOT_CONNECTED), None),
    (_ret("inbox", ok=False, error="could not read the inbox: login refused (check "
          "OMEGA_IMAP_USER and the app password)"), None),
    (_ret("inbox", ok=False, error="could not read the inbox: timed out"), None),
    (_ret("calendar", ok=False, error=GOOGLE_ERROR), ("google", True)),
    (_ret("calendar", ok=True, result=f"today 10:00 x\ncouldn't read Google calendar: "
          f"{gcal.SIGN_IN_AGAIN}"), ("google", True)),
    (_ret("calendar", ok=True, result="today 10:00 standup"), ("google", False)),
    (_ret("calendar", ok=True, result="x\ncouldn't read Google calendar: timed out"), None),
    (_ret("calendar", ok=False, error="could not read the calendar: timed out"), None),
    (_ret("github__search", ok=False, error=MCP_ERROR), ("mcp:github", True)),
    (_ret("github__search", ok=False, error="the github MCP server refused the credentials "
          "(403); check its token"), ("mcp:github", True)),
    (_ret("github__search", ok=True, result="[]"), ("mcp:github", False)),
    (_ret("github__search", ok=False, error="(Result from the github MCP server, x)"),
     ("mcp:github", False)),
    (_ret("github__search", ok=False, error="the github MCP server answered HTTP 500"), None),
    (_ret("github__search", ok=False, error="the github MCP server could not be reached"), None),
    (_ret("github__search", ok=False, error="refused: money never goes through omega"), None),
    (_ret("remember", ok=False, error="refused the credentials"), None),
])
def test_only_explicit_refusals_are_access_failures(payload: dict, want: Any) -> None:
    assert access.classify(payload) == want


def test_the_mirrored_strings_match_what_they_mirror() -> None:
    server = mcp.Server(name="github", url="https://x", allow=frozenset(), token=TOKEN)
    refused = mcp.Client(server)._fail("refused the credentials (401); check its token")
    assert str(refused).startswith(access._MCP_REFUSED.format(server="github"))
    assert followups._MCP_ANSWERED == access._MCP_ANSWERED
    assert {tools.INBOX, tools.MAIL} == access._INBOX_TOOLS
    assert tools.CALENDAR == access._CALENDAR_TOOL
    assert mcp.SEP == access._SEP


def test_each_fix_is_concrete_and_never_carries_a_key() -> None:
    assert access.fix_for("inbox") == "IMAP refused the login: check the app password"
    assert access.fix_for("google") == gcal.SIGN_IN_AGAIN
    assert "python -m omega calendar-auth" in access.fix_for("google")
    assert access.fix_for("mcp:github") == (
        "the github MCP server refused its token: replace OMEGA_MCP_GITHUB_TOKEN"
    )
    for origin in ("inbox", "google", "mcp:github"):
        episodes.access_lost(origin=origin, fix=access.fix_for(origin), at=AT)
    with pytest.raises(episodes.BadPayload):
        episodes.access_lost(origin="dropbox", fix="x", at=AT)
    with pytest.raises(episodes.BadPayload):
        episodes.access_paused(origin="inbox", for_seq=0, at=AT)


# --- the IMAP refusal --------------------------------------------------------------


def test_an_explicit_imap_refusal_is_marked_and_keeps_its_words() -> None:
    err = imaplib.IMAP4.error(f"b'[AUTHENTICATIONFAILED] Invalid credentials {PASSWORD}'")
    with pytest.raises(inbox.InboxUnavailable) as caught:
        inbox.read(inbox.settings(IMAP_ENV), connect=server(fail={"login": err}))
    assert caught.value.refused is True
    assert str(caught.value) == "login refused (check OMEGA_IMAP_USER and the app password)"
    out = inbox.look_lines(NOW, environ=IMAP_ENV, connect=server(fail={"login": err}))
    assert out == [f"couldn't read the inbox: {caught.value}. {inbox.IMAP_FIX}"]
    assert PASSWORD not in out[0]


@pytest.mark.parametrize("text", ["nope", "[UNAVAILABLE] try later", "IMAP access disabled"])
def test_any_other_login_failure_stays_a_plain_failure(text: str) -> None:
    connect = server(fail={"login": imaplib.IMAP4.error(text)})
    with pytest.raises(inbox.InboxUnavailable) as caught:
        inbox.read(inbox.settings(IMAP_ENV), connect=connect)
    assert caught.value.refused is False
    assert inbox.refusal_suffix(caught.value) == ""


def test_the_tools_add_the_fix_only_to_a_refusal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    def refuse(**kw: Any) -> str:
        raise inbox.InboxUnavailable("login refused (x)", refused=True)

    def plain(**kw: Any) -> str:
        raise inbox.InboxUnavailable("timed out")

    box = tools.ToolBox(store_root=tmp_path, prompted=True)
    for fn, tail in ((refuse, f"login refused (x). {inbox.IMAP_FIX}"), (plain, "timed out")):
        monkeypatch.setattr(inbox, "tool_text", fn)
        with pytest.raises(tools.ToolError) as caught:
            tools._inbox(box, {"from": None, "days": 1, "unread_only": True})
        assert str(caught.value) == f"could not read the inbox: {tail}"


# --- the probes ----------------------------------------------------------------------


def test_the_inbox_probe_logs_in_and_reads_nothing() -> None:
    assert inbox.probe({}, connect=server()) is None
    assert inbox.probe(IMAP_ENV, connect=server()) is True
    from tests.test_inbox import fake

    assert [c[0] for c in fake().calls] == ["LOGIN", "SELECT", "LOGOUT"]
    refused = imaplib.IMAP4.error("[AUTHENTICATIONFAILED] Invalid credentials")
    assert inbox.probe(IMAP_ENV, connect=server(fail={"login": refused})) is False
    assert inbox.probe(IMAP_ENV, connect=server(fail={"login": OSError("down")})) is None


GCAL_ENV = {gcal.ENV_CLIENT_ID: "cid", gcal.ENV_CLIENT_SECRET: "cs",
            gcal.ENV_REFRESH_TOKEN: "refresh-probe"}


def test_the_google_probe_is_the_ordinary_refresh() -> None:
    gcal._TOKENS.clear()
    seen: list[str] = []

    def answer(status: int, body: dict):
        def send(method, url, form, headers):
            seen.append(url)
            return status, json.dumps(body).encode()
        return send

    assert gcal.probe({}, send=answer(200, {})) is None
    assert gcal.probe(GCAL_ENV, send=answer(400, {"error": "invalid_grant"})) is False
    assert gcal.probe(GCAL_ENV, send=answer(500, {})) is None
    assert gcal.probe(GCAL_ENV, send=answer(200, {"access_token": "a", "expires_in": 3600})) is True
    assert set(seen) == {gcal.TOKEN_URL}
    gcal._TOKENS.clear()


def test_the_mcp_probe_lists_and_calls_nothing() -> None:
    from tests.test_mcp import FakeServer, env

    fake = FakeServer([])
    assert mcp.probe("github", env(), post=fake) is True
    assert "tools/call" not in fake.methods
    fake.status = 401
    assert mcp.probe("github", env(), post=fake) is False
    fake.status = 500
    assert mcp.probe("github", env(), post=fake) is None
    assert mcp.probe("gitlab", env(), post=fake) is None
    assert access.probe("dropbox", {}) is None


# --- the first refusal: one tell, with the fix ------------------------------------------


@pytest.mark.parametrize("tool, error, origin", [
    ("inbox", IMAP_ERROR, "inbox"),
    ("calendar", GOOGLE_ERROR, "google"),
    ("github__search", MCP_ERROR, "mcp:github"),
])
def test_the_first_refusal_is_told_once_with_its_fix(
    q: EventQueue, tool: str, error: str, origin: str
) -> None:
    say(q, "hello")
    asked(q, tool, ok=False, error=error)
    assert [e.origin for e in view(q).owed] == [origin]
    ex = executor(q)
    assert ex.notice(now=NOW) is True
    [tell] = tells(q)
    assert tell == tools.tell_instruction(access.lost_words(origin))
    assert access.fix_for(origin) in tell
    [lost] = kinds(q, episodes.ACCESS_LOST)
    assert lost["origin"] == origin and lost["fix"] == access.fix_for(origin)
    assert TOKEN not in json.dumps(payloads(q))

    # Delivered as written, no model call.
    drain(ex)
    # Refused again, and after a restart: never told twice.
    asked(q, tool, ok=False, error=error)
    executor(q).notice(now=NOW + timedelta(minutes=1))
    assert len(tells(q)) == 1 and len(kinds(q, episodes.ACCESS_LOST)) == 1
    assert list(view(q).lost) == [origin]


def test_a_refusal_that_works_before_its_tell_is_not_told(q: EventQueue) -> None:
    say(q, "hello")
    asked(q, "inbox", ok=False, error=IMAP_ERROR)
    asked(q, "inbox", ok=True, result="no mail")
    assert not view(q)
    assert executor(q, online=lambda: False).notice(now=NOW) is False
    assert tells(q) == []


def test_refusals_are_never_counted_as_plain_failures(q: EventQueue) -> None:
    say(q, "hello")
    watch(q)
    for _ in range(4):
        asked(q, "github__search", ok=False, error=MCP_ERROR)
        fire(q)
    state = Followups.rebuild(q.store).state()
    assert not state.entries and state.owed() == ()
    # A plain failure still counts as before.
    for _ in range(3):
        asked(q, "github__search", ok=False, error="the github MCP server answered HTTP 500")
    assert Followups.rebuild(q.store).state().entries["mcp:github"].count == 3


# --- pausing what reads from it ------------------------------------------------------------


def test_a_watch_refused_once_is_paused_while_lost_and_not_counted(q: EventQueue) -> None:
    say(q, "hello")
    watch(q)
    fire(q)
    ex = executor(q)
    assert ex.notice(now=NOW) is True
    drain(ex)
    assert view(q).pausing("w1-abcd") == "inbox"

    for _ in range(3):
        seq = pending_fire(q)
        fp = provider.FakeProvider({})
        ex = executor(q, complete=fp.complete)
        [result] = ex.drain()
        assert result.seq == seq and result.outcome == "silent"
        assert fp.calls == [], "no model call, no read"
    paused = kinds(q, episodes.ACCESS_PAUSED)
    assert [p["origin"] for p in paused] == ["inbox"] * 3
    assert all(p["kind"] != episodes.TOOL_CALLED or p["for_seq"] < paused[0]["for_seq"]
               for p in payloads(q))
    # Neither failed nor reset, so no follow-up tell is owed.
    assert not Followups.rebuild(q.store).state()
    assert len(tells(q)) == 1

    # Other watches, briefs and reminders are not paused.
    other = pending_fire(q, sid="w2-ffff", what="the flights")
    assert executor(q)._pausing(q.at(other).payload) is None
    b = say(q, tools.brief_instruction(), channel="schedule", schedule_id="w1-abcd")
    assert executor(q)._pausing(q.at(b).payload) is None


def test_recovery_is_told_once_and_the_watch_runs_again(q: EventQueue) -> None:
    say(q, "hello")
    watch(q)
    fire(q)
    ex = executor(q)
    ex.notice(now=NOW)
    drain(ex)
    asked(q, "inbox", ok=True, result="2 mails")
    assert view(q).recovering == ("inbox",)
    ex = executor(q)
    assert ex.notice(now=NOW) is True
    [rec] = kinds(q, episodes.ACCESS_RESTORED)
    assert rec["origin"] == "inbox"
    assert tells(q)[-1] == tools.tell_instruction("Your inbox works again; what was paused is running again.")
    drain(ex)
    assert not view(q)
    assert view(q).pausing("w1-abcd") is None
    executor(q, online=lambda: False).notice(now=NOW)
    assert len(kinds(q, episodes.ACCESS_RESTORED)) == 1 and len(tells(q)) == 2


def test_a_lost_origin_is_probed_at_most_hourly(q: EventQueue) -> None:
    say(q, "hello")
    asked(q, "github__search", ok=False, error=MCP_ERROR)
    probed: list[str] = []
    answers = [False, True]

    def probe(origin: str) -> bool:
        probed.append(origin)
        return answers.pop(0)

    ex = executor(q, probe=probe, online=lambda: True)
    ex.notice(now=NOW)  # the tell
    drain(ex)
    assert probed == []
    ex.notice(now=NOW + timedelta(minutes=1))
    assert probed == ["mcp:github"]
    ex.notice(now=NOW + timedelta(minutes=30))
    assert probed == ["mcp:github"], "not again within the hour"
    assert ex.notice(now=NOW + timedelta(minutes=62)) is True
    assert probed == ["mcp:github"] * 2
    assert [p["origin"] for p in kinds(q, episodes.ACCESS_RESTORED)] == ["mcp:github"]
    assert "github MCP server works again" in tells(q)[-1]


def test_no_probe_while_offline(q: EventQueue) -> None:
    say(q, "hello")
    asked(q, "inbox", ok=False, error=IMAP_ERROR)
    probed: list[str] = []
    ex = executor(q, probe=lambda o: probed.append(o), online=lambda: False)
    ex.notice(now=NOW)
    drain(ex)
    ex.notice(now=NOW + timedelta(hours=2))
    assert probed == []


# --- the look ------------------------------------------------------------------------------


def test_a_look_refused_tells_at_once_and_then_skips_that_read(q: EventQueue) -> None:
    say(q, "hello")
    reads: list[datetime] = []
    refused = [f"couldn't read the inbox: login refused (x). {inbox.IMAP_FIX}"]

    def sense(now: datetime) -> list[str]:
        reads.append(now)
        return refused

    ex = executor(q, inbox=sense, online=lambda: True)
    assert ex.notice(now=NOW, look_every_seconds=0) is True
    assert len(tells(q)) == 1 and kinds(q, episodes.ACCESS_LOST)[0]["origin"] == "inbox"
    drain(ex)
    ex.notice(now=NOW + timedelta(hours=1), look_every_seconds=0)
    assert len(reads) == 1, "the lost inbox is not read again by the look"
    look = [p for p in kinds(q, episodes.MESSAGE_INBOUND) if p["channel"] == "self"][-1]
    assert access.paused_line("inbox") in look["text"]
    [paused] = kinds(q, episodes.ACCESS_PAUSED)
    assert paused["origin"] == "inbox"
    assert len(tells(q)) == 1


# --- the brief -------------------------------------------------------------------------------


def brief_turn(q: EventQueue, *, with_access: bool = True):
    seq = say(q, tools.brief_instruction(), channel="schedule", schedule_id="b1-0001")
    q.claim(seq)
    fp = provider.FakeProvider({provider.JUDGE: ["ACT"], provider.ACT: ["Your day is clear."]})
    result = run_turn(
        q, q.at(seq), complete=fp.complete, act=act_loop,
        grants=Grants().advance(q.store, upto=seq).state(),
        access=Access().advance(q.store, upto=seq).state() if with_access else None,
        at=AT,
    )
    q.finish(seq)
    return result


# The date in the line is a local rendering of NOW, so the zone is pinned.
@pytest.mark.usefixtures("pinned_zone")
def test_the_brief_lists_each_lost_access_under_waiting_on_you(q: EventQueue) -> None:
    say(q, "hello")
    asked(q, "calendar", ok=False, error=GOOGLE_ERROR)
    ex = executor(q)
    ex.notice(now=NOW)
    drain(ex)
    result = brief_turn(q)
    assert result.reply == (
        "Your day is clear.\n\nWaiting on you:\n"
        f"- your Google calendar — {gcal.SIGN_IN_AGAIN}, since Sun 11 Oct."
    )
    assert "refresh-probe" not in result.reply


# --- equivalence ------------------------------------------------------------------------------


def test_with_no_refusal_nothing_changes(q: EventQueue) -> None:
    say(q, "hello")
    watch(q)
    fire(q, outcome="silent", ok=True, error=None, result="2 mails")
    asked(q, "github__search", ok=False, error="the github MCP server answered HTTP 500")
    asked(q, "calendar", ok=True, result="today 10:00 standup")
    assert not view(q)
    probed: list[str] = []
    before = q.head()
    ex = executor(q, probe=lambda o: probed.append(o), online=lambda: False)
    assert ex.notice(now=NOW) is False
    assert q.head() == before and probed == []

    # The brief reads exactly as it does without the view.
    a, b = brief_turn(q, with_access=True), brief_turn(q, with_access=False)
    assert a.reply == b.reply == "Your day is clear."

    # A fire runs as before rather than being paused.
    seq = pending_fire(q)
    assert executor(q)._pausing(q.at(seq).payload) is None


def test_the_fold_is_the_same_incrementally_and_from_scratch(q: EventQueue) -> None:
    say(q, "hello")
    watch(q)
    fire(q)
    asked(q, "github__search", ok=False, error=MCP_ERROR)
    q.append(episodes.access_lost(origin="inbox", fix=access.fix_for("inbox"), at=AT))
    asked(q, "inbox", ok=True, result="x")
    step = Access()
    for e in q.store.episodes_since(0):
        step.apply(e.seq, episodes.decode(e.payload))
    assert step.state() == view(q)
    assert step.state().recovering == ("inbox",)
    assert [e.origin for e in step.state().owed] == ["mcp:github"]


# The date in the line is a local rendering of NOW, so the zone is pinned.
@pytest.mark.usefixtures("pinned_zone")
def test_lost_rows_join_the_proposals_under_one_heading(
    q: EventQueue, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    from omega import grants as granted, turn

    say(q, "hello")
    asked(q, "inbox", ok=False, error=IMAP_ERROR)
    q.append(episodes.access_lost(origin="inbox", fix=access.fix_for("inbox"), at=AT))
    listed = 'Waiting on you:\n- p7 — retire watch. Reply "yes p7" or "no p7".\n\nNo longer asking p3.'
    monkeypatch.setattr(granted, "briefing", lambda *a, **k: listed)
    event = episodes.inbound(tools.brief_instruction(), channel="schedule", at=AT, schedule_id="b1-0001")
    ctx = SimpleNamespace(event=event, grants=None, seq=99, access=view(q))
    assert turn._waiting(ctx).splitlines() == [
        "Waiting on you:",
        '- p7 — retire watch. Reply "yes p7" or "no p7".',
        f"- your inbox — {inbox.IMAP_FIX}, since Sun 11 Oct.",
        "",
        "No longer asking p3.",
    ]
    for none in (None, access.State()):
        ctx.access = none
        assert turn._waiting(ctx) == listed
