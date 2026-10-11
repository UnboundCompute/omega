"""DL-090 — the Gmail sweep reads hourly and only what arrived since it last read.

No network and no account: the mailbox is :class:`StampedIMAP`, the
``test_inbox`` fake taught to answer INTERNALDATE, and every turn runs on a
:class:`omega.provider.FakeProvider` through the real executor and act loop,
so the cursor is read out of the same log a restart would fold.
"""

from __future__ import annotations

import functools
import imaplib
import time
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from pathlib import Path
from typing import Any, Optional

import pytest

from omega import episodes, inbox, mailcursor, notice, provider, tools
from omega.act import act_loop
from omega.executor import Executor
from omega.memory import MemoryStore
from omega.provider import ToolCall
from omega.queue import EventQueue
from tests.test_inbox import ENV, FakeIMAP, message

_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def _internaldate(when: datetime) -> str:
    when = when.astimezone(timezone.utc)
    return f"{when.day:02d}-{_MONTHS[when.month - 1]}-{when.year} {when:%H:%M:%S} +0000"


class StampedIMAP(FakeIMAP):
    """The ``test_inbox`` fake, answering ``INTERNALDATE`` when it is asked for."""

    def _fetch(self, uid: int, what: str):
        self.mailbox.fetches.append(what)
        typ, data = super()._fetch(uid, what)
        if "INTERNALDATE" in what and isinstance(data[0], tuple):
            meta, literal = data[0]
            stamp = _internaldate(self.mails[uid]["arrived"]).encode()
            meta = meta.replace(b" BODY[HEADER", b' INTERNALDATE "' + stamp + b'" BODY[HEADER', 1)
            data = [(meta, literal), *data[1:]]
        return typ, data


class Mailbox:
    """A mailbox that mail can arrive in between sweeps, and that can go down."""

    def __init__(self) -> None:
        self.mails: dict[int, dict[str, Any]] = {}
        self.down = False
        self.connections = 0
        self.fetches: list[str] = []

    def arrive(self, uid: int, arrived: datetime, *, subject: str,
               date: Optional[datetime] = None, sender: str = "Kaushik R <k@x.com>") -> None:
        raw = message(sender=sender, subject=subject, date=format_datetime(date or arrived))
        self.mails[uid] = {"raw": raw, "arrived": arrived}

    def connect(self, host: str, port: int, timeout: float | None = None) -> StampedIMAP:
        self.connections += 1
        fail = {"search": imaplib.IMAP4.error("server unavailable")} if self.down else None
        imap = StampedIMAP(host, port, timeout, mails=self.mails, fail=fail)
        imap.mailbox = self
        return imap


@pytest.fixture
def box(monkeypatch: pytest.MonkeyPatch) -> Mailbox:
    mailbox = Mailbox()
    real = inbox.tool_text
    monkeypatch.setattr(inbox, "tool_text", functools.partial(
        real, environ=ENV, connect=mailbox.connect))
    return mailbox


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _between(box: Mailbox, uid: int, *, subject: str) -> None:
    """A mail arriving strictly between two reads.

    INTERNALDATE has whole seconds and a read's stamp is kept to the second,
    so a mail in the same second as a read is shown again rather than lost;
    a second either side keeps this test off that deliberate edge.
    """
    time.sleep(1.1)
    box.arrive(uid, _now(), subject=subject)
    time.sleep(1.1)


def _act(role: str, messages: list[dict[str, Any]], *, args: dict[str, Any]) -> Any:
    """Read the inbox once, then answer with nothing new."""
    if any(m.get("role") == "tool" for m in messages):
        return tools.NOTHING_NEW
    return ToolCall(id="c1", name="inbox", arguments=dict(args))


def _turn(q: EventQueue, event: dict[str, Any], *, args: Optional[dict] = None,
          outcome: str = "silent") -> dict[str, Any]:
    """One turn through the real act loop; returns its ``tool.returned``."""
    fp = provider.FakeProvider({provider.ACT: functools.partial(_act, args=args or {})})
    seq = q.append(event)
    q.claim(seq)
    from omega.turn import TurnContext

    ctx = TurnContext(seq=seq, event=q.at(seq).payload, recalled=[], queue=q,
                      complete=fp.complete)
    act_loop(ctx)
    [returned] = [p.payload for p in q.recent(50)
                  if p.payload.get("kind") == episodes.TOOL_RETURNED
                  and p.payload.get("for_seq") == seq]
    q.append(episodes.completed(
        for_seq=seq, outcome=outcome, tools=["inbox"],
        error="model went away" if outcome == "failed" else None,
    ))
    return returned


def _sweep(q: EventQueue, *, sid: str = "w1", **kw: Any) -> dict[str, Any]:
    """The person's Gmail watch firing."""
    fire = episodes.inbound(tools.watch_instruction("new mail in their Gmail"),
                            channel="schedule", schedule_id=sid)
    return _turn(q, fire, **kw)


def _asked(q: EventQueue, **kw: Any) -> dict[str, Any]:
    """Their own message asking about mail."""
    return _turn(q, episodes.inbound("did Kaushik write back?", channel="tray"), **kw)


# --------------------------------------------------------------------------
# a watch
# --------------------------------------------------------------------------


def test_a_repeat_sweep_returns_nothing_new(store: MemoryStore, box: Mailbox) -> None:
    q = EventQueue(store)
    box.arrive(1, _now() - timedelta(minutes=20), subject="lunch?")
    first = _sweep(q)
    assert first["ok"] is True and "lunch?" in first["result"]
    assert first["result"].startswith("new since ")
    again = _sweep(q)
    assert again["ok"] is True
    assert "lunch?" not in again["result"]
    assert again["result"].startswith("no new unread mail since "), (
        "an empty cursor read must say so, never read as a failure"
    )


def test_a_mail_arriving_between_sweeps_appears_once(store: MemoryStore, box: Mailbox) -> None:
    q = EventQueue(store)
    box.arrive(1, _now() - timedelta(minutes=20), subject="lunch?")
    _sweep(q)
    _between(box, 2, subject="the docs")
    second = _sweep(q)
    assert "the docs" in second["result"] and "lunch?" not in second["result"]
    third = _sweep(q)
    assert "the docs" not in third["result"] and "lunch?" not in third["result"]


def test_the_first_run_covers_the_last_hour(store: MemoryStore, box: Mailbox) -> None:
    q = EventQueue(store)
    box.arrive(1, _now() - timedelta(minutes=50), subject="within the hour")
    box.arrive(2, _now() - timedelta(minutes=70), subject="before the hour")
    first = _sweep(q)
    assert "within the hour" in first["result"]
    assert "before the hour" not in first["result"]


def test_arrival_is_internaldate_not_the_date_header(store: MemoryStore, box: Mailbox) -> None:
    q = EventQueue(store)
    # Claims to be fresh, arrived three hours ago: a forged or skewed Date.
    box.arrive(1, _now() - timedelta(hours=3), subject="backdated arrival",
               date=_now())
    # Claims to be from last week, arrived just now.
    box.arrive(2, _now() - timedelta(minutes=5), subject="old date header",
               date=_now() - timedelta(days=7))
    result = _sweep(q)["result"]
    assert "old date header" in result
    assert "backdated arrival" not in result
    assert any("INTERNALDATE" in f for f in box.fetches)


def test_a_failed_read_does_not_advance_the_cursor(store: MemoryStore, box: Mailbox) -> None:
    q = EventQueue(store)
    box.arrive(1, _now() - timedelta(minutes=20), subject="lunch?")
    _sweep(q)
    _between(box, 2, subject="the docs")
    box.down = True
    failed = _sweep(q)
    assert failed["ok"] is False and "could not read the inbox" in failed["error"]
    box.down = False
    # A read that worked, in a turn that then failed, does not count either.
    lost = _sweep(q, outcome="failed")
    assert "the docs" in lost["result"]
    after = _sweep(q)
    assert "the docs" in after["result"], "nothing that arrived was lost"
    assert "the docs" not in _sweep(q)["result"]


def test_a_restart_keeps_the_cursor(store_dir: Path, box: Mailbox) -> None:
    box.arrive(1, _now() - timedelta(minutes=20), subject="lunch?")
    with MemoryStore.open(store_dir) as s:
        assert "lunch?" in _sweep(EventQueue(s))["result"]
    with MemoryStore.open(store_dir) as s:
        q = EventQueue(s)
        assert mailcursor.last_read(s, "schedule:w1", unread_only=True) is not None
        assert "lunch?" not in _sweep(q)["result"]


def test_each_origin_keeps_its_own_cursor(store: MemoryStore, box: Mailbox) -> None:
    """One watch reading a mail must not swallow it for another, or a brief."""
    q = EventQueue(store)
    box.arrive(1, _now() - timedelta(minutes=20), subject="lunch?")
    assert "lunch?" in _sweep(q, sid="w1")["result"]
    assert "lunch?" in _sweep(q, sid="w2")["result"]
    assert "lunch?" not in _sweep(q, sid="w1")["result"]


def test_a_sender_narrowed_read_only_covers_that_sender(store: MemoryStore, box: Mailbox) -> None:
    q = EventQueue(store)
    box.arrive(1, _now() - timedelta(minutes=20), subject="lunch?")
    _sweep(q, args={"from": "Kaushik"})
    assert "lunch?" in _sweep(q)["result"], "a narrowed read did not show everyone's mail"
    assert "lunch?" not in _sweep(q, args={"from": "kaushik"})["result"]


def test_hours_narrows_below_a_day(store: MemoryStore, box: Mailbox) -> None:
    q = EventQueue(store)
    box.arrive(1, _now() - timedelta(minutes=30), subject="this hour")
    box.arrive(2, _now() - timedelta(hours=3), subject="this morning")
    text = _asked(q, args={"hours": 2})["result"]
    assert "this hour" in text and "this morning" not in text
    box.mails.clear()
    assert _asked(q, args={"hours": 1, "from": "nobody@x.com"})["result"] == (
        "no unread mail from nobody@x.com in the last hour"
    )
    box_ = tools.ToolBox(store_root=store.root, prompted=True)
    # A zero is a blank the model filled in (DL-079), not a request for none.
    assert "hours" not in box_.classify("inbox", {"hours": 0}).args
    for bad in (49, -1, "two", True):
        with pytest.raises(tools.ToolRejected):
            box_.classify("inbox", {"hours": bad})
    decided = box_.classify("inbox", {"hours": 3})
    assert decided.args["hours"] == 3 and decided.why.endswith("of the last 3 hours")
    assert "hours" not in box_.classify("inbox", {}).args


# --------------------------------------------------------------------------
# their own question is unchanged
# --------------------------------------------------------------------------


def test_prompted_reads_are_unchanged(store: MemoryStore, box: Mailbox) -> None:
    q = EventQueue(store)
    box.arrive(1, _now() - timedelta(minutes=20), subject="lunch?")
    box.arrive(2, _now() - timedelta(hours=20), subject="yesterday's thread")
    _sweep(q)
    _sweep(q)
    box.fetches.clear()
    asked = _asked(q)["result"]
    expected = inbox.tool_text()  # no cursor, no hours: the old call exactly
    assert asked == expected
    assert "lunch?" in asked and "yesterday's thread" in asked
    assert not asked.startswith("new since")
    assert all("INTERNALDATE" not in f for f in box.fetches), (
        "a prompted read sends the commands it always sent"
    )
    assert _asked(q)["result"] == expected, "and asking again repeats it, as asked"


def test_still_read_only(store: MemoryStore, box: Mailbox) -> None:
    calls: list[tuple] = []
    real = box.connect

    def connect(*a: Any, **kw: Any) -> StampedIMAP:
        imap = real(*a, **kw)
        calls.append(imap.calls)
        return imap

    box.connect = connect  # type: ignore[method-assign]
    inbox.tool_text.keywords["connect"] = connect  # type: ignore[attr-defined]
    box.arrive(1, _now() - timedelta(minutes=20), subject="lunch?")
    _sweep(EventQueue(store))
    sent = [c for log in calls for c in log]
    assert ("SELECT", "INBOX", True) in sent
    assert not {c[0] for c in sent} & {"STORE", "COPY", "MOVE", "EXPUNGE", "APPEND"}
    fetched = [c[3] for c in sent if c[:2] == ("UID", "FETCH")]
    assert fetched and all("BODY.PEEK[" in f for f in fetched)


def test_held_mail_is_still_held_on_a_cursor_read(store: MemoryStore, box: Mailbox) -> None:
    q = EventQueue(store)
    box.arrive(1, _now() - timedelta(minutes=10), subject="Reset your password",
               sender="Apple <appleid@id.apple.com>")
    result = _sweep(q)["result"]
    assert inbox.HELD_MARK in result


# --------------------------------------------------------------------------
# the look
# --------------------------------------------------------------------------

NOW = datetime(2026, 10, 9, 9, 0, tzinfo=timezone.utc)


def _looks(q: EventQueue, ex: Executor, when: datetime) -> dict[str, Any]:
    assert ex.notice(now=when) is True
    ex.drain()
    return [
        e.payload for e in q.recent(200)
        if e.payload.get("kind") == episodes.MESSAGE_INBOUND
        and e.payload.get("channel") == notice.CHANNEL
    ][-1]


def test_the_looks_mail_part_runs_hourly(store: MemoryStore) -> None:
    q = EventQueue(store)
    box = Mailbox()
    swept: list[int] = []
    cal: list[datetime] = []
    box.arrive(1, NOW - timedelta(minutes=30), subject="lunch?")
    box.arrive(2, NOW - timedelta(hours=5), subject="last night")

    def mail(now: datetime, after: Optional[datetime] = None) -> list[str]:
        return inbox.look_lines(now, after=after, environ=ENV, connect=box.connect,
                                on_mails=lambda found: swept.append(len(found)))

    def calendar(now: datetime) -> list[str]:
        cal.append(now)
        return ["nothing on the calendar in the next 12 hours"]

    q.append(episodes.inbound("hello", channel="tray", at=(NOW - timedelta(hours=3)).isoformat()))
    fp = provider.FakeProvider({provider.JUDGE: "SILENT", provider.ACT: ["unused"]})
    ex = Executor(q, complete=fp.complete, inbox=mail, calendar=calendar)
    ex.recover()
    ex.drain()

    first = _looks(q, ex, NOW)
    assert "lunch?" in first["text"] and "last night" not in first["text"]
    assert first["mail_at"] == NOW.isoformat()
    assert box.connections == 1

    half = _looks(q, ex, NOW + timedelta(minutes=31))
    assert box.connections == 1, "the mail part waits out the hour"
    assert "mail is read once an hour; last read at" in half["text"]
    assert "lunch?" not in half["text"]
    assert "mail_at" not in half

    box.arrive(3, NOW + timedelta(minutes=40), subject="the docs")
    hour = _looks(q, ex, NOW + timedelta(minutes=62))
    assert box.connections == 2
    assert "the docs" in hour["text"] and "lunch?" not in hour["text"]

    quiet = _looks(q, ex, NOW + timedelta(minutes=124))
    assert "no new mail from people since" in quiet["text"]
    assert len(cal) == 4, "the calendar keeps the look's own pace"
    assert swept == [2, 3, 3], "finance still gets the whole reading"


def test_a_looks_failed_mail_read_is_retried_next_look(store: MemoryStore) -> None:
    q = EventQueue(store)
    box = Mailbox()
    box.arrive(1, NOW - timedelta(minutes=10), subject="lunch?")
    box.down = True

    def mail(now: datetime, after: Optional[datetime] = None) -> list[str]:
        return inbox.look_lines(now, after=after, environ=ENV, connect=box.connect)

    q.append(episodes.inbound("hello", channel="tray", at=(NOW - timedelta(hours=3)).isoformat()))
    fp = provider.FakeProvider({provider.JUDGE: "SILENT", provider.ACT: ["unused"]})
    ex = Executor(q, complete=fp.complete, inbox=mail)
    ex.recover()
    ex.drain()
    failed = _looks(q, ex, NOW)
    assert "couldn't read the inbox" in failed["text"] and "mail_at" not in failed
    box.down = False
    retried = _looks(q, ex, NOW + timedelta(minutes=31))
    assert "lunch?" in retried["text"]
