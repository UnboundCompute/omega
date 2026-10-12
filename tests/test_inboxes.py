"""DL-092 — a second, read-only mailbox beside the first.

``OMEGA_INBOXES`` names the extra mailboxes; each has its own
``OMEGA_IMAP_<NAME>_*``. **No network and no key**: every mailbox below is
the ``test_mail_cursor`` fake, one per host, and every turn runs on a fake
provider through the real executor and act loop.

The equivalence test at the end is the one that matters most: with no
``OMEGA_INBOXES`` the tool schemas, the decisions, the sweep's and the
look's records are byte-for-byte what they were before DL-092, checked
against digests taken from the code before it.
"""

from __future__ import annotations

import functools
import hashlib
import imaplib
import json
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import pytest

from omega import access, episodes, inbox, mailcursor, notice, provider, tools
from omega.access import Access
from omega.act import act_loop
from omega.executor import Executor
from omega.memory import MemoryStore
from omega.queue import EventQueue
from tests.test_inbox import ENV, MUTATING
from tests.test_mail_cursor import Mailbox, StampedIMAP, _sweep

WORK_HOST = "imap.work.example.com"
WORK_PASSWORD = "work-app-password-123"
WORK = {
    "OMEGA_INBOXES": "work",
    "OMEGA_IMAP_WORK_USER": "user@example.com",
    "OMEGA_IMAP_WORK_PASSWORD": WORK_PASSWORD,
    "OMEGA_IMAP_WORK_HOST": WORK_HOST,
}
ENV2 = {**ENV, **WORK}
NOW = datetime(2026, 10, 9, 9, 0, tzinfo=timezone.utc)
REFUSED = imaplib.IMAP4.error("[AUTHENTICATIONFAILED] Invalid credentials")


class Box(Mailbox):
    """A :class:`Mailbox` whose login can also be refused."""

    def __init__(self) -> None:
        super().__init__()
        self.refused = False
        self.sessions: list[StampedIMAP] = []

    def connect(self, host: str, port: int, timeout: float | None = None) -> StampedIMAP:
        if self.refused:
            self.connections += 1
            imap = StampedIMAP(host, port, timeout, mails=self.mails, fail={"login": REFUSED})
            imap.mailbox = self
        else:
            imap = super().connect(host, port, timeout)
        self.sessions.append(imap)
        return imap


class Boxes:
    """Both mailboxes, each behind its own host: a UID means nothing across them."""

    def __init__(self) -> None:
        self.personal = Box()
        self.work = Box()

    def connect(self, host: str, port: int, timeout: float | None = None) -> StampedIMAP:
        box = {inbox.DEFAULT_HOST: self.personal, WORK_HOST: self.work}[host]
        return box.connect(host, port, timeout)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _clear(monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    for key in list(os.environ):
        if key.startswith("OMEGA_IMAP") or key == inbox.ENV_INBOXES:
            monkeypatch.delenv(key)


@pytest.fixture
def boxes(monkeypatch: pytest.MonkeyPatch) -> Boxes:
    """Both mailboxes configured in the process environment, as on the host."""
    _clear(monkeypatch)
    for key, value in ENV2.items():
        monkeypatch.setenv(key, value)
    both = Boxes()
    monkeypatch.setattr(inbox, "tool_text_multi", functools.partial(
        inbox.tool_text_multi, connect=both.connect))
    monkeypatch.setattr(inbox, "open_mail", functools.partial(
        inbox.open_mail, connect=both.connect))
    return both


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------


def test_extra_mailboxes_are_read_from_the_environment() -> None:
    said: list[str] = []
    env = {
        **ENV,
        inbox.ENV_INBOXES: " Work, side2 ,personal, 9x ,work,averyveryverylongname,half,",
        "OMEGA_IMAP_WORK_USER": "user@example.com",
        "OMEGA_IMAP_WORK_PASSWORD": "pw",
        "OMEGA_IMAP_SIDE2_USER": "other@example.com",
        "OMEGA_IMAP_SIDE2_PASSWORD": "pw2",
        "OMEGA_IMAP_SIDE2_HOST": "imap.example.com",
        "OMEGA_IMAP_HALF_USER": "half@example.com",
    }
    found = inbox.extras(env, warn=said.append)
    assert [(s.name, s.user, s.host) for s in found] == [
        ("work", "user@example.com", inbox.DEFAULT_HOST),
        ("side2", "other@example.com", "imap.example.com"),
    ]
    assert "pw" not in repr(found)
    # personal, 9x, the repeat, the long name, and the half-configured one:
    # one line each, and the half one names exactly what is missing.
    assert len(said) == 5
    assert "'half'" in said[-1] and "OMEGA_IMAP_HALF_PASSWORD" in said[-1]
    assert inbox.names(env) == ["personal", "work", "side2"] and inbox.multi(env)
    assert inbox.extras(env) == found, "every other caller reads quietly"
    for quiet in ({}, ENV, {inbox.ENV_INBOXES: ""}, {inbox.ENV_INBOXES: ",, ,"}):
        assert inbox.extras(quiet, warn=said.append) == [] and not inbox.multi(quiet)
    assert len(said) == 5
    # The default mailbox may be unset while an extra one is not.
    assert inbox.names(WORK) == ["work"]


def test_startup_names_each_mailbox_and_says_what_it_skipped(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from omega import __main__ as entry

    _clear(monkeypatch)
    before = entry._account_lines(ENV)
    assert capsys.readouterr().err == ""
    assert not any(line.startswith("inbox work") for line in before)
    lines = entry._account_lines({**ENV2, inbox.ENV_INBOXES: "work,Bad-Name"})
    assert lines[:2] == before[:2]
    assert "inbox work: reading user@example.com at imap.work.example.com (read-only)" in lines
    assert WORK_PASSWORD not in "\n".join(lines)
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and "omega: inbox: skipping inbox 'Bad-Name'" in err


# --------------------------------------------------------------------------
# the tools
# --------------------------------------------------------------------------


def _props(schemas: list[dict[str, Any]], name: str) -> dict[str, Any]:
    [found] = [s for s in schemas if s["function"]["name"] == name]
    return found["function"]["parameters"]


def test_the_inbox_argument_is_offered_only_beside_another_mailbox(
    boxes: Boxes, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    box = tools.ToolBox(store_root=tmp_path, prompted=True)
    offered = box.schemas()
    listing, opening = _props(offered, "inbox"), _props(offered, "mail")
    assert listing["properties"]["inbox"]["enum"] == ["personal", "work", "all"]
    assert "inbox" not in listing["required"]
    assert opening["properties"]["inbox"]["enum"] == ["personal", "work"]
    assert opening["required"] == ["id", "inbox"]
    # The module's own list, and every other tool, are untouched.
    assert "inbox" not in _props(tools.schemas(), "inbox")["properties"]
    plain = {s["function"]["name"]: s for s in tools.schemas()}
    for schema in offered:
        if schema["function"]["name"] not in ("inbox", "mail"):
            assert schema == plain.get(schema["function"]["name"], schema)

    # One mailbox, whichever it is, offers no choice.
    monkeypatch.delenv(inbox.ENV_USER)
    one = tools.ToolBox(store_root=tmp_path, prompted=True).schemas()
    assert "inbox" not in _props(one, "inbox")["properties"]
    assert "inbox" not in _props(one, "mail")["properties"]


def test_the_offer_never_depends_on_what_they_said(boxes: Boxes, tmp_path: Path) -> None:
    """Never trigger words: the offer is the configuration, whatever the turn."""
    a = tools.ToolBox(store_root=tmp_path, prompted=True).schemas()
    b = tools.ToolBox(store_root=tmp_path, prompted=True, remembered=("work mail",)).schemas()
    assert _props(a, "inbox") == _props(b, "inbox")
    assert _props(a, "mail") == _props(b, "mail")


def _read(box: tools.ToolBox, **args: Any) -> str:
    return box.dispatch(box.classify("inbox", args))


def test_every_line_says_its_mailbox_and_uids_do_not_collide(
    boxes: Boxes, tmp_path: Path
) -> None:
    boxes.personal.arrive(1, _now() - timedelta(minutes=20), subject="dinner on friday")
    boxes.work.arrive(1, _now() - timedelta(minutes=30), subject="quarterly review")
    box = tools.ToolBox(store_root=tmp_path, prompted=True)
    decided = box.classify("inbox", {})
    assert decided.args["inbox"] == ["personal", "work"]
    assert decided.why.endswith("in the personal and work inboxes")
    text = box.dispatch(decided)
    lines = text.split("\n")
    assert all(re.match(r"^\[(personal|work)\] ", line) for line in lines)
    assert any(line.startswith("[personal] ") and "dinner on friday" in line and "#1" in line
               for line in lines)
    assert any(line.startswith("[work] ") and "quarterly review" in line and "#1" in line
               for line in lines)

    only = _read(box, inbox="work")
    assert "quarterly review" in only and "dinner" not in only
    assert _read(box, inbox="ALL") == text
    with pytest.raises(tools.ToolRejected, match="'inbox' must be one of"):
        box.classify("inbox", {"inbox": "school"})

    opened = box.dispatch(box.classify("mail", {"id": 1, "inbox": "work"}))
    assert opened.startswith("[work] mail #1\n") and "quarterly review" in opened
    assert "dinner" not in opened
    mine = box.dispatch(box.classify("mail", {"id": 1, "inbox": "personal"}))
    assert "dinner on friday" in mine
    with pytest.raises(tools.ToolRejected, match="needs 'inbox'"):
        box.classify("mail", {"id": 1})

    # Read-only, on both servers.
    for b in (boxes.personal, boxes.work):
        sent = [c for imap in b.sessions for c in imap.calls]
        assert ("SELECT", "INBOX", True) in sent
        assert not {c[0] for c in sent} & MUTATING
    assert {i.host for i in boxes.work.sessions} == {WORK_HOST}


def test_one_mailbox_failing_never_hides_the_other(boxes: Boxes, tmp_path: Path) -> None:
    boxes.personal.arrive(1, _now() - timedelta(minutes=20), subject="dinner on friday")
    boxes.work.down = True
    box = tools.ToolBox(store_root=tmp_path, prompted=True)
    text = _read(box)
    first, *rest = text.split("\n")
    assert first.startswith("[work] couldn't read the inbox: ")
    assert any("dinner on friday" in line for line in rest)
    boxes.personal.down = True
    with pytest.raises(tools.ToolError) as caught:
        _read(box)
    assert str(caught.value).split("\n")[1:] == [
        "[personal] couldn't read the inbox: error: server unavailable",
        "[work] couldn't read the inbox: error: server unavailable",
    ]
    assert WORK_PASSWORD not in str(caught.value)


# --------------------------------------------------------------------------
# the sweep: one cursor per mailbox
# --------------------------------------------------------------------------


def test_each_mailbox_keeps_its_own_sweep_cursor(store: MemoryStore, boxes: Boxes) -> None:
    q = EventQueue(store)
    boxes.personal.arrive(1, _now() - timedelta(minutes=20), subject="dinner on friday")
    boxes.work.arrive(1, _now() - timedelta(minutes=25), subject="quarterly review")
    boxes.work.down = True

    first = _sweep(q)
    assert first["ok"] is True
    assert "dinner on friday" in first["result"]
    assert first["result"].startswith("[work] couldn't read the inbox")
    assert mailcursor.last_read(store, "schedule:w1", unread_only=True) is not None
    assert mailcursor.last_read(store, "schedule:w1", unread_only=True, mailbox="work") is None

    boxes.work.down = False
    second = _sweep(q)
    assert "quarterly review" in second["result"], (
        "the work mailbox's failed read must not have moved its cursor, and "
        "the personal mailbox's read must not stand in for it"
    )
    assert "dinner on friday" not in second["result"]
    assert "[personal] no new unread mail since" in second["result"]

    third = _sweep(q)
    assert "quarterly review" not in third["result"] and "dinner" not in third["result"]
    assert "[work] no new unread mail since" in third["result"]

    # Mail arriving in one mailbox shows once, in that mailbox only.
    time.sleep(1.1)
    boxes.work.arrive(2, _now(), subject="the offsite")
    time.sleep(1.1)
    fourth = _sweep(q)
    assert [line for line in fourth["result"].split("\n") if "the offsite" in line][0].startswith("[work] ")
    assert "the offsite" not in _sweep(q)["result"]

    called = [p.payload for p in q.recent(200) if p.payload.get("kind") == episodes.TOOL_CALLED]
    assert all(c["args"]["inbox"] == ["personal", "work"] for c in called)


def test_a_one_mailbox_read_covers_only_that_mailbox(store: MemoryStore, boxes: Boxes) -> None:
    q = EventQueue(store)
    boxes.personal.arrive(1, _now() - timedelta(minutes=20), subject="dinner on friday")
    boxes.work.arrive(1, _now() - timedelta(minutes=25), subject="quarterly review")
    assert "quarterly review" in _sweep(q, args={"inbox": "work"})["result"]
    both = _sweep(q)["result"]
    assert "dinner on friday" in both and "quarterly review" not in both


# --------------------------------------------------------------------------
# the look
# --------------------------------------------------------------------------


def _looks(q: EventQueue, ex: Executor, when: datetime) -> dict[str, Any]:
    assert ex.notice(now=when) is True
    ex.drain()
    return [
        e.payload for e in q.recent(200)
        if e.payload.get("kind") == episodes.MESSAGE_INBOUND
        and e.payload.get("channel") == notice.CHANNEL
    ][-1]


def _look_executor(q: EventQueue, boxes: Boxes, **kw: Any) -> Executor:
    def mail(now: datetime, after: Any = None) -> list[str]:
        return inbox.look_lines(now, after=after, environ=ENV2, connect=boxes.connect)

    q.append(episodes.inbound("hello", channel="tray", at=(NOW - timedelta(hours=3)).isoformat()))
    fp = provider.FakeProvider({provider.JUDGE: "SILENT", provider.ACT: ["unused"]})
    ex = Executor(q, complete=fp.complete, inbox=mail, inbox_names=("personal", "work"), **kw)
    ex.recover()
    ex.drain()
    return ex


def test_the_look_reads_each_mailbox_hourly_from_its_own_cursor(
    store: MemoryStore, boxes: Boxes
) -> None:
    q = EventQueue(store)
    boxes.personal.arrive(1, NOW - timedelta(minutes=30), subject="dinner on friday")
    boxes.work.arrive(1, NOW - timedelta(minutes=20), subject="quarterly review")
    boxes.work.down = True
    ex = _look_executor(q, boxes)

    first = _looks(q, ex, NOW)
    assert "[personal] " in first["text"] and "dinner on friday" in first["text"]
    assert "[work] couldn't read the inbox" in first["text"]
    assert first["mail_at"] == NOW.isoformat()
    assert "mail_inboxes" not in first, "a failed read stamps nothing"

    boxes.work.down = False
    half = _looks(q, ex, NOW + timedelta(minutes=31))
    assert boxes.personal.connections == 1, "the personal mailbox waits out its hour"
    assert "[personal] mail is read once an hour" in half["text"]
    assert "quarterly review" in half["text"], "the work mailbox is retried at once"
    assert "mail_at" not in half
    assert half["mail_inboxes"] == {"work": (NOW + timedelta(minutes=31)).isoformat()}

    boxes.personal.arrive(2, NOW + timedelta(minutes=50), subject="the plumber")
    hour = _looks(q, ex, NOW + timedelta(minutes=62))
    assert "the plumber" in hour["text"] and "dinner" not in hour["text"]
    assert "[work] mail is read once an hour" in hour["text"]
    assert "quarterly review" not in hour["text"]
    assert hour["mail_at"] == (NOW + timedelta(minutes=62)).isoformat()
    assert "mail_inboxes" not in hour


# --------------------------------------------------------------------------
# access, per mailbox
# --------------------------------------------------------------------------


def _tells(q: EventQueue) -> list[str]:
    return [
        episodes.decode(e.payload)["text"] for e in q.store.episodes_since(0)
        if episodes.decode(e.payload).get("kind") == episodes.MESSAGE_INBOUND
        and str(episodes.decode(e.payload).get("text", "")).startswith(tools.TELL_LEAD)
    ]


def _kinds(q: EventQueue, kind: str) -> list[dict[str, Any]]:
    return [p for p in (episodes.decode(e.payload) for e in q.store.episodes_since(0))
            if p.get("kind") == kind]


def test_the_work_mailboxs_fix_names_its_password_and_the_admin() -> None:
    fix = access.fix_for("inbox:work")
    assert "OMEGA_IMAP_WORK_PASSWORD" in fix and "app password" in fix and "admin" in fix
    assert inbox.IMAP_FIX not in fix
    assert access.fix_for("inbox") == inbox.IMAP_FIX
    assert access.label_for("inbox:work") == "your work inbox"
    assert access.label_for("inbox", ENV) == "your inbox"
    assert access.label_for("inbox", ENV2) == "your personal inbox"
    assert access.inbox_origin("personal") == "inbox"


def test_a_refused_work_login_loses_only_the_work_mailbox(
    store: MemoryStore, boxes: Boxes, tmp_path: Path
) -> None:
    q = EventQueue(store)
    boxes.personal.arrive(1, _now() - timedelta(minutes=20), subject="dinner on friday")
    boxes.work.refused = True
    swept = _sweep(q)
    assert swept["ok"] is True and "dinner on friday" in swept["result"]
    assert access.classify_all(swept) == [("inbox", False), ("inbox:work", True)]

    view = Access.rebuild(store).state()
    assert [e.origin for e in view.owed] == ["inbox:work"]
    probed: list[str] = []
    ex = Executor(q, act=act_loop, complete=provider.FakeProvider({}).complete,
                  probe=lambda o: probed.append(o) or False, online=lambda: True,
                  inbox_names=("personal", "work"))
    ex.recover()
    assert ex.notice(now=NOW) is True
    [tell] = _tells(q)
    assert "your work inbox" in tell and "admin" in tell
    [lost] = _kinds(q, episodes.ACCESS_LOST)
    assert lost["origin"] == "inbox:work"
    assert list(Access.rebuild(store).state().lost) == ["inbox:work"]
    assert WORK_PASSWORD not in json.dumps(_kinds(q, episodes.ACCESS_LOST))

    # The watch that read it is paused; one that read only the personal
    # mailbox is not.
    assert Access.rebuild(store).state().pausing("w1") == "inbox:work"
    _sweep(q, sid="w2", args={"inbox": "personal"})
    assert Access.rebuild(store).state().pausing("w2") is None


def test_the_look_pauses_only_the_lost_mailbox_and_probes_it(
    store: MemoryStore, boxes: Boxes
) -> None:
    q = EventQueue(store)
    boxes.personal.arrive(1, NOW - timedelta(minutes=30), subject="dinner on friday")
    boxes.work.refused = True
    answers = [False, True]
    probed: list[str] = []

    def probe(origin: str) -> bool:
        probed.append(origin)
        return answers.pop(0)

    ex = _look_executor(q, boxes, probe=probe, online=lambda: True)
    first = _looks(q, ex, NOW)
    assert "dinner on friday" in first["text"]
    assert "OMEGA_IMAP_WORK_PASSWORD" in first["text"]
    [lost] = _kinds(q, episodes.ACCESS_LOST)
    assert lost["origin"] == "inbox:work" and lost["fix"] == access.fix_for("inbox:work")
    ex.drain()

    work_reads = boxes.work.connections
    later = _looks(q, ex, NOW + timedelta(minutes=70))
    assert boxes.work.connections == work_reads, "the lost mailbox is not read"
    assert "[work] " + access.paused_line("inbox:work") in later["text"]
    assert "[personal] no new mail from people since" in later["text"]
    assert [p["origin"] for p in _kinds(q, episodes.ACCESS_PAUSED)] == ["inbox:work"]
    assert probed == ["inbox:work"]

    ex.notice(now=NOW + timedelta(minutes=140))
    ex.drain()
    assert probed == ["inbox:work"] * 2
    assert [p["origin"] for p in _kinds(q, episodes.ACCESS_RESTORED)] == ["inbox:work"]
    assert any("Your work inbox works again" in t for t in _tells(q))


def test_the_work_probe_logs_into_the_work_mailbox_only() -> None:
    both = Boxes()
    assert inbox.probe(ENV2, name="work", connect=both.connect) is True
    assert both.personal.connections == 0 and both.work.connections == 1
    both.work.refused = True
    assert inbox.probe(ENV2, name="work", connect=both.connect) is False
    assert inbox.probe(ENV2, connect=both.connect) is True, "the default is unaffected"
    assert inbox.probe(ENV, name="work", connect=both.connect) is None


def test_the_brief_lists_the_lost_mailbox_by_name(store: MemoryStore, boxes: Boxes) -> None:
    from tests.test_access import brief_turn

    q = EventQueue(store)
    boxes.work.refused = True
    _sweep(q)
    ex = Executor(q, act=act_loop, complete=provider.FakeProvider({}).complete,
                  inbox_names=("personal", "work"))
    ex.recover()
    ex.notice(now=NOW)
    from tests.test_access import drain

    drain(ex)
    reply = brief_turn(q).reply
    assert "Waiting on you:\n- your work inbox — " + access.fix_for("inbox:work") in reply


# --------------------------------------------------------------------------
# equivalence: no OMEGA_INBOXES, nothing changed
# --------------------------------------------------------------------------

#: Digests of :func:`_equivalence_record`, taken from the code before DL-092
#: (and again from this code, which must match them). ``schemas`` was
#: retaken when ``topics`` came to be offered on every prompted turn; with
#: ``topics`` left out of ``box.schemas()`` it is still the pre-DL-092
#: digest, 6664b2a47e611bb4fec4ceced8f8a778abc87697ac15b5079a39296f568a7703.
GOLDEN = {
    "schemas": "bbfdd0ed13461bcf17136abc3ed7c7b8edd1e781eb2d9553d41725ecb7a4aba6",
    "decisions": "2c4dce3c91de59c6a04cae8410eff9fc2289959ec56142f4804f9d1e0fb8c190",
    "mail": "804f1d2668f4ae0779d57ed156443b465043318c19e29ae63d20f391362fad0a",
    "sweep": "a82b1e92db42f1f10e7e4cf8f5565991002744e5368e81feb200250f970d5da0",
    # A look that lists mail fences it as data (DL-064); with that fence taken
    # out, the look's records digest to the value this golden had before it.
    "look": "df81b5def046c6e98e9eea7a11f9f25e70a2d5fc2726ee00ad2a956734583ed3",
}

_ISO = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?(?:[+-]\d\d:\d\d|Z)?")
_HM = re.compile(r"(?:yesterday )?(?<!\d)\d\d:\d\d(?!\d)")


def _normal(value: Any) -> str:
    """The wall clock taken out: the sweep runs on the real one, since the
    mail cursor stamps reads with it. Everything else is kept."""
    text = json.dumps(value, sort_keys=True, ensure_ascii=False)
    return _HM.sub("<hm>", _ISO.sub("<at>", text))


def _digest(value: Any) -> str:
    return hashlib.sha256(_normal(value).encode()).hexdigest()


def _records(store: MemoryStore, *, drop: tuple[str, ...] = ("ms",)) -> list[dict[str, Any]]:
    out = []
    for e in store.episodes_since(0):
        payload = episodes.decode(e.payload)
        out.append({k: v for k, v in payload.items() if k not in drop})
    return out


def _equivalence_record(root: Path, *, clear: bool = True) -> dict[str, str]:
    """What DL-092 must leave byte-identical without ``OMEGA_INBOXES``: the
    offered schemas, the inbox decisions, a sweep's records and a look's.
    Runnable against the code before DL-092, which is where the golden
    digests came from."""
    import os

    with pytest.MonkeyPatch.context() as mp:
        for key in list(os.environ):
            if clear and (key.startswith("OMEGA_IMAP") or key == "OMEGA_INBOXES"):
                mp.delenv(key)
        mp.setenv("TZ", "Asia/Kolkata")
        time.tzset()
        try:
            out: dict[str, str] = {}
            box = tools.ToolBox(store_root=root, prompted=True)
            out["schemas"] = _digest([tools.schemas(), box.schemas(),
                                      tools.ToolBox(store_root=root, prompted=False).schemas()])
            decisions = []
            for name, args in (("inbox", {}), ("inbox", {"from": "Kaushik", "hours": 3}),
                               ("inbox", {"unread_only": False, "days": 5}), ("mail", {"id": 7})):
                d = box.classify(name, args)
                decisions.append([d.tool, d.tier, d.why, d.args])
            out["decisions"] = _digest(decisions)

            mailbox = Mailbox()
            mp.setattr(inbox, "tool_text", functools.partial(
                inbox.tool_text, environ=ENV, connect=mailbox.connect))
            mp.setattr(inbox, "open_mail", functools.partial(
                inbox.open_mail, environ=ENV, connect=mailbox.connect))
            mailbox.arrive(1, _now() - timedelta(minutes=20), subject="lunch?")
            mailbox.arrive(2, _now() - timedelta(minutes=200), subject="older")
            opened = box.dispatch(box.classify("mail", {"id": 1}))
            out["mail"] = _digest(opened if isinstance(opened, str) else [opened.model, opened.log])
            with MemoryStore.open(root / "sweep") as store:
                q = EventQueue(store)
                _sweep(q)
                _sweep(q)
                _sweep(q, args={"from": "Kaushik"})
                mailbox.down = True
                _sweep(q)
                out["sweep"] = _digest(_records(store))

            look = Mailbox()
            look.arrive(1, NOW - timedelta(minutes=30), subject="lunch?")
            look.arrive(2, NOW - timedelta(hours=5), subject="last night")
            with MemoryStore.open(root / "look") as store:
                q = EventQueue(store)

                def mail(now: datetime, after: Optional[datetime] = None) -> list[str]:
                    return inbox.look_lines(now, after=after, environ=ENV, connect=look.connect)

                q.append(episodes.inbound("hello", channel="tray",
                                          at=(NOW - timedelta(hours=3)).isoformat()))
                fp = provider.FakeProvider({provider.JUDGE: "SILENT", provider.ACT: ["unused"]})
                ex = Executor(q, complete=fp.complete, inbox=mail)
                ex.recover()
                ex.drain()
                for minutes in (0, 31, 62):
                    ex.notice(now=NOW + timedelta(minutes=minutes))
                    ex.drain()
                look.down = True
                ex.notice(now=NOW + timedelta(minutes=130))
                ex.drain()
                # The look's records keep their pinned stamps: no normalizing.
                looks = [
                    p for p in _records(store)
                    if p.get("kind") == episodes.MESSAGE_INBOUND and p.get("channel") == notice.CHANNEL
                ]
                out["look"] = hashlib.sha256(
                    json.dumps(looks, sort_keys=True, ensure_ascii=False).encode()
                ).hexdigest()
            return out
        finally:
            mp.undo()
            time.tzset()


def test_without_omega_inboxes_nothing_changes(tmp_path: Path) -> None:
    assert _equivalence_record(tmp_path) == GOLDEN


def test_an_unusable_omega_inboxes_changes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Set but naming nothing usable, it is as if it were unset."""
    _clear(monkeypatch)
    monkeypatch.setenv(inbox.ENV_INBOXES, "personal, half ,Bad-Name")
    monkeypatch.setenv("OMEGA_IMAP_HALF_USER", "half@example.com")
    assert _equivalence_record(tmp_path, clear=False) == GOLDEN
