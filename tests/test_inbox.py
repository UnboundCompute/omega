"""The inbox sense — Gmail over IMAP, read-only — DL-080.

No network and no real account: :class:`FakeIMAP` stands in for
``imaplib.IMAP4_SSL``, answers in imaplib's own tuple shapes, and records
every command so the read-only invariant is checked against what was actually
sent, not against what the code meant to send.
"""

from __future__ import annotations

import imaplib
import re
import socket
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from omega import inbox

PASSWORD = "abcd efgh ijkl mnop"
ENV = {inbox.ENV_USER: "me@gmail.com", inbox.ENV_PASSWORD: PASSWORD}
IST = timezone(timedelta(hours=5, minutes=30))
NOW = datetime(2026, 10, 9, 15, 0, tzinfo=IST)  # a Friday

MUTATING = {"STORE", "COPY", "MOVE", "EXPUNGE", "APPEND", "DELETE", "CREATE", "RENAME"}


@pytest.fixture
def ist(monkeypatch: pytest.MonkeyPatch):
    import time

    monkeypatch.setenv("TZ", "Asia/Kolkata")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


# --------------------------------------------------------------------------
# the fake server
# --------------------------------------------------------------------------


def message(
    *,
    sender: str = "Kaushik R <k@x.com>",
    subject: str = "lunch?",
    date: str = "Fri, 09 Oct 2026 14:05:00 +0530",
    body: bytes | None = b"Are we on for lunch tomorrow?",
    content_type: str = "text/plain",
    extra: str = "",
    raw: bytes | None = None,
) -> bytes:
    if raw is not None:
        return raw
    head = f"From: {sender}\r\nSubject: {subject}\r\nDate: {date}\r\n{extra}"
    head += f"MIME-Version: 1.0\r\nContent-Type: {content_type}; charset=utf-8\r\n\r\n"
    return head.encode() + (body or b"")


def _header_block(raw: bytes) -> bytes:
    """What the server sends for BODY[HEADER.FIELDS (...)]: only those fields."""
    head = raw.split(b"\r\n\r\n", 1)[0]
    wanted = set(inbox._HEADER_FIELDS.lower().split())
    keep: list[bytes] = []
    current = False
    for line in head.split(b"\r\n"):
        if line[:1] in (b" ", b"\t"):
            if current:
                keep.append(line)
            continue
        current = line.split(b":", 1)[0].strip().lower().decode() in wanted
        if current:
            keep.append(line)
    return b"\r\n".join(keep) + b"\r\n\r\n"


class FakeIMAP:
    """Answers like ``imaplib.IMAP4_SSL`` and records every call."""

    instances: list["FakeIMAP"] = []

    def __init__(self, host: str, port: int, timeout: float | None = None,
                 *, mails: dict[int, dict[str, Any]] | None = None,
                 fail: dict[str, BaseException] | None = None) -> None:
        self.host, self.port, self.timeout = host, port, timeout
        self.mails = mails or {}
        self.fail = fail or {}
        self.calls: list[tuple] = []
        FakeIMAP.instances.append(self)

    def _maybe_fail(self, name: str) -> None:
        if name in self.fail:
            raise self.fail[name]

    def login(self, user: str, password: str):
        self.calls.append(("LOGIN", user, password))
        self._maybe_fail("login")
        return "OK", [b"me@gmail.com authenticated (Success)"]

    def select(self, mailbox: str = "INBOX", readonly: bool = False):
        self.calls.append(("SELECT", mailbox, readonly))
        self._maybe_fail("select")
        return "OK", [str(len(self.mails)).encode()]

    def uid(self, command: str, *args: Any):
        self.calls.append(("UID", command.upper(), *args))
        if command.upper() == "SEARCH":
            self._maybe_fail("search")
            return "OK", [b" ".join(str(u).encode() for u in sorted(self.mails))]
        if command.upper() == "FETCH":
            self._maybe_fail("fetch")
            return self._fetch(int(args[0]), args[1])
        raise AssertionError(f"unexpected UID {command}")

    def _fetch(self, uid: int, what: str):
        m = self.mails[uid]
        raw = m["raw"]
        if "HEADER.FIELDS" in what:
            header = _header_block(raw)
            flags = " ".join(m.get("flags", ()))
            meta = (f"{uid} (UID {uid} FLAGS ({flags}) RFC822.SIZE "
                    f"{m.get('size', len(raw))} BODY[HEADER.FIELDS "
                    f"({inbox._HEADER_FIELDS})] {{{len(header)}}}").encode()
            return "OK", [(meta, header), b")"]
        if what == "(BODY.PEEK[])":
            return "OK", [(f"{uid} (UID {uid} BODY[] {{{len(raw)}}}".encode(), raw), b")"]
        raise AssertionError(f"unexpected fetch {what}")

    def logout(self):
        self.calls.append(("LOGOUT",))
        return "BYE", [b"LOGOUT Requested"]

    # anything mutating would land here and be recorded
    def __getattr__(self, name: str):
        def record(*args: Any, **kw: Any):
            self.calls.append((name.upper(), *args))
            return "OK", [b""]
        return record


def server(mails: dict[int, dict[str, Any]] | None = None, **kw: Any):
    """A ``connect`` that builds a FakeIMAP over ``mails`` (uid -> spec)."""
    built = {u: {"raw": message(**spec.pop("msg", {})), **spec}
             for u, spec in (mails or {}).items()}

    def connect(host: str, port: int, timeout: float | None = None):
        return FakeIMAP(host, port, timeout, mails=built, **kw)

    FakeIMAP.instances.clear()
    return connect


def fake() -> FakeIMAP:
    return FakeIMAP.instances[-1]


SETTINGS = inbox.Settings(user="me@gmail.com", password=PASSWORD)


# --------------------------------------------------------------------------
# settings
# --------------------------------------------------------------------------


def test_settings_none_unless_both_user_and_password() -> None:
    assert inbox.settings({}) is None
    assert inbox.settings({inbox.ENV_USER: "me@gmail.com"}) is None
    assert inbox.settings({inbox.ENV_PASSWORD: PASSWORD}) is None
    assert inbox.settings({inbox.ENV_USER: " ", inbox.ENV_PASSWORD: PASSWORD}) is None


def test_settings_full_and_repr_hides_password() -> None:
    s = inbox.settings({**ENV, inbox.ENV_HOST: "imap.example.com"})
    assert s == inbox.Settings("me@gmail.com", PASSWORD, "imap.example.com", 993)
    assert inbox.settings(ENV).host == "imap.gmail.com"
    assert PASSWORD not in repr(s) and "password" not in repr(s)
    assert PASSWORD not in str(s)


# --------------------------------------------------------------------------
# read-only invariant
# --------------------------------------------------------------------------


def test_readonly_select_peek_fetches_and_nothing_mutating() -> None:
    connect = server({1: {}, 2: {"flags": ["\\Seen"]}})
    inbox.read(SETTINGS, unread_only=False, connect=connect, now=NOW)
    calls = fake().calls
    assert ("SELECT", "INBOX", True) in calls
    assert not [c for c in calls if c[0] == "SELECT" and c[2] is not True]
    fetches = [c[3] for c in calls if c[:2] == ("UID", "FETCH")]
    assert len(fetches) == 4  # header + body for each
    for what in fetches:
        assert "BODY.PEEK[" in what
        assert not re.search(r"(?<!PEEK)\bBODY\[", what.replace("BODY.PEEK[", ""))
        assert "RFC822" not in what.replace("RFC822.SIZE", "")
    names = {c[0] if c[0] != "UID" else c[1] for c in calls}
    assert names <= {"LOGIN", "SELECT", "SEARCH", "FETCH", "LOGOUT"}
    assert not names & MUTATING
    assert fake().timeout == 15 and (fake().host, fake().port) == ("imap.gmail.com", 993)


# --------------------------------------------------------------------------
# search criteria
# --------------------------------------------------------------------------


def _search(connect) -> tuple:
    return next(c for c in fake().calls if c[:2] == ("UID", "SEARCH"))[2:]


def test_search_since_and_unseen(ist) -> None:
    connect = server()
    inbox.read(SETTINGS, connect=connect, now=NOW)
    assert _search(connect) == (None, "SINCE", "07-Oct-2026", "UNSEEN")


def test_search_since_uses_local_date(ist) -> None:
    # 20:00 UTC on the 8th is already the 9th in India.
    connect = server()
    inbox.read(SETTINGS, days=1, unread_only=False, connect=connect,
               now=datetime(2026, 10, 8, 20, 0, tzinfo=timezone.utc))
    assert _search(connect) == (None, "SINCE", "08-Oct-2026")


def test_search_from_is_quoted() -> None:
    connect = server()
    inbox.read(SETTINGS, sender='Kaushik "K" \\R', connect=connect, now=NOW)
    terms = _search(connect)
    assert terms[-2:] == ("FROM", '"Kaushik \\"K\\" \\\\R"')


@pytest.mark.parametrize("bad", ["k@x.com\r\nSTORE 1 +FLAGS (\\Deleted)",
                                 "a\nb", "x" * 101, "Kaushik é"])
def test_bad_sender_refused_before_connecting(bad: str) -> None:
    connect = server()
    with pytest.raises(ValueError):
        inbox.read(SETTINGS, sender=bad, connect=connect, now=NOW)
    assert FakeIMAP.instances == []


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------


def test_rfc2047_subject_and_sender_decoded() -> None:
    connect = server({1: {"msg": {
        "sender": "=?utf-8?q?K=C3=A4ushik_R?= <k@x.com>",
        "subject": "=?utf-8?b?4KSo4KSu4KS44KWN4KSk4KWHIGhlbGxv?=",
    }}})
    [m] = inbox.read(SETTINGS, connect=connect, now=NOW)
    assert m.sender == "Käushik R <k@x.com>"
    assert m.subject == "नमस्ते hello"
    assert m.date == datetime(2026, 10, 9, 14, 5, tzinfo=IST)
    assert m.unread and not m.bulk


def test_control_chars_stripped_and_truncated() -> None:
    connect = server({1: {"msg": {
        "sender": "Evil\x1b[31m ‮Name <e@x.com>",
        "subject": "a\tb   " + "s" * 300,
        "body": b"line one\r\n\r\n  line\x07 two " + b"w" * 500,
    }}})
    [m] = inbox.read(SETTINGS, connect=connect, now=NOW)
    assert "\x1b" not in m.sender and "‮" not in m.sender
    assert m.subject.startswith("a b s") and len(m.subject) <= 140
    assert m.snippet.startswith("line one line two ") and len(m.snippet) <= 200
    assert "\x07" not in m.snippet and "\n" not in m.snippet
    long = server({1: {"msg": {"sender": "N" * 200 + " <n@x.com>"}}})
    [m] = inbox.read(SETTINGS, connect=long, now=NOW)
    assert len(m.sender) <= 80


def test_html_only_body_tags_stripped() -> None:
    html = (b"<html><head><style>p{color:red}</style></head><body>"
            b"<p>Hi&nbsp;there &amp; welcome</p><script>alert(1)</script></body></html>")
    connect = server({1: {"msg": {"body": html, "content_type": "text/html"}}})
    [m] = inbox.read(SETTINGS, connect=connect, now=NOW)
    assert m.snippet == "Hi\xa0there & welcome" or m.snippet == "Hi there & welcome"
    assert "<" not in m.snippet and "color" not in m.snippet and "alert" not in m.snippet


def test_multipart_prefers_plain() -> None:
    raw = (b"From: K <k@x.com>\r\nSubject: s\r\nDate: Fri, 09 Oct 2026 14:05:00 +0530\r\n"
           b"MIME-Version: 1.0\r\nContent-Type: multipart/alternative; boundary=XX\r\n\r\n"
           b"--XX\r\nContent-Type: text/html; charset=utf-8\r\n\r\n<b>html version</b>\r\n"
           b"--XX\r\nContent-Type: text/plain; charset=utf-8\r\n\r\nplain version\r\n"
           b"--XX--\r\n")
    connect = server({1: {"msg": {"raw": raw}}})
    [m] = inbox.read(SETTINGS, connect=connect, now=NOW)
    assert m.snippet == "plain version"


def test_oversized_message_body_not_fetched() -> None:
    connect = server({1: {"size": 256_001}, 2: {"size": 256_000}})
    mails = inbox.read(SETTINGS, connect=connect, now=NOW)
    fetches = [c for c in fake().calls if c[:2] == ("UID", "FETCH")]
    bodies = [c[2] for c in fetches if c[3] == "(BODY.PEEK[])"]
    assert bodies == [b"2"]
    assert sorted(m.snippet for m in mails) == ["", "Are we on for lunch tomorrow?"]


@pytest.mark.parametrize("extra, bulk", [
    ("List-Unsubscribe: <mailto:u@x.com>\r\n", True),
    ("List-Id: <news.x.com>\r\n", True),
    ("Precedence: bulk\r\n", True),
    ("Precedence: list\r\n", True),
    ("Precedence: junk\r\n", True),
    ("Auto-Submitted: auto-generated\r\n", True),
    ("Auto-Submitted: no\r\n", False),
    ("Precedence: first-class\r\n", False),
    ("", False),
])
def test_bulk_detection(extra: str, bulk: bool) -> None:
    connect = server({1: {"msg": {"extra": extra}}})
    [m] = inbox.read(SETTINGS, connect=connect, now=NOW)
    assert m.bulk is bulk


def test_unparseable_date_is_none() -> None:
    connect = server({1: {"msg": {"date": "sometime last week"}}})
    [m] = inbox.read(SETTINGS, connect=connect, now=NOW)
    assert m.date is None


def test_newest_first_and_limit() -> None:
    connect = server({
        u: {"msg": {"subject": f"s{u}", "date": f"Fri, 09 Oct 2026 1{u}:00:00 +0530"}}
        for u in range(1, 6)
    })
    mails = inbox.read(SETTINGS, limit=3, connect=connect, now=NOW)
    assert [m.subject for m in mails] == ["s5", "s4", "s3"]
    fetched = {c[2] for c in fake().calls if c[:2] == ("UID", "FETCH")}
    assert fetched == {b"3", b"4", b"5"}


def test_seen_flag_marks_read() -> None:
    connect = server({1: {"flags": ["\\Seen", "\\Answered"]}, 2: {"flags": []}})
    mails = inbox.read(SETTINGS, unread_only=False, connect=connect, now=NOW)
    assert sorted(m.unread for m in mails) == [False, True]


# --------------------------------------------------------------------------
# failures
# --------------------------------------------------------------------------


def test_auth_failure_reason_without_password() -> None:
    err = imaplib.IMAP4.error(f"b'[AUTHENTICATIONFAILED] Invalid credentials {PASSWORD}'")
    connect = server(fail={"login": err})
    with pytest.raises(inbox.InboxUnavailable) as caught:
        inbox.read(SETTINGS, connect=connect, now=NOW)
    assert str(caught.value) == "login refused (check OMEGA_IMAP_USER and the app password)"
    assert PASSWORD not in repr(caught.value)
    assert caught.value.__cause__ is None and caught.value.__suppress_context__
    assert fake().calls[-1] == ("LOGOUT",)


def test_timeout() -> None:
    connect = server({1: {}}, fail={"fetch": socket.timeout("timed out")})
    with pytest.raises(inbox.InboxUnavailable, match="^timed out$"):
        inbox.read(SETTINGS, connect=connect, now=NOW)
    assert fake().calls[-1] == ("LOGOUT",)


def test_connect_failure_reason() -> None:
    def connect(host: str, port: int, timeout: float | None = None):
        raise ConnectionRefusedError(61, "Connection refused")

    with pytest.raises(inbox.InboxUnavailable, match="ConnectionRefusedError"):
        inbox.read(SETTINGS, connect=connect, now=NOW)


def test_server_error_scrubbed_and_logout_always() -> None:
    connect = server(fail={"search": imaplib.IMAP4.error(f"SEARCH bad {PASSWORD}")})
    with pytest.raises(inbox.InboxUnavailable) as caught:
        inbox.read(SETTINGS, connect=connect, now=NOW)
    assert str(caught.value).startswith("error: ")
    assert PASSWORD not in str(caught.value)
    assert fake().calls[-1] == ("LOGOUT",)


# --------------------------------------------------------------------------
# wording
# --------------------------------------------------------------------------


def _mail(date: datetime | None, **kw: Any) -> inbox.Mail:
    base = dict(sender="Kaushik R <k@x.com>", subject="lunch?", date=date,
                snippet="are we on?", bulk=False, unread=True)
    return inbox.Mail(**{**base, **kw})


def test_lines_today_yesterday_weekday(ist) -> None:
    out = inbox.lines([
        _mail(datetime(2026, 10, 9, 14, 5, tzinfo=IST)),
        _mail(datetime(2026, 10, 8, 12, 50, tzinfo=timezone.utc), snippet=""),
        _mail(datetime(2026, 10, 6, 9, 0, tzinfo=IST), unread=False),
        _mail(None),
    ], NOW)
    assert out == [
        "14:05 from Kaushik R <k@x.com>: lunch? — are we on?",
        "yesterday 18:20 from Kaushik R <k@x.com>: lunch?",
        "Tue 06 Oct from Kaushik R <k@x.com>: lunch? — are we on? (read)",
        "undated from Kaushik R <k@x.com>: lunch? — are we on?",
    ]


def test_look_lines_not_configured() -> None:
    assert inbox.look_lines(NOW, environ={}, connect=server()) == []
    assert FakeIMAP.instances == []


def test_look_lines_failure() -> None:
    connect = server(fail={"login": imaplib.IMAP4.error("nope")})
    out = inbox.look_lines(NOW, environ=ENV, connect=connect)
    assert out == ["couldn't read the inbox: login refused "
                   "(check OMEGA_IMAP_USER and the app password)"]


def test_look_lines_people_then_bulk_count(ist) -> None:
    connect = server({
        1: {"msg": {"sender": "News <n@x.com>", "extra": "List-Id: <n.x.com>\r\n"}},
        2: {"msg": {"sender": "Bot <b@x.com>", "extra": "Auto-Submitted: auto-replied\r\n"}},
        3: {},
    })
    out = inbox.look_lines(NOW, environ=ENV, connect=connect)
    assert out == [
        "14:05 from Kaushik R <k@x.com>: lunch? — Are we on for lunch tomorrow?",
        "(and 2 unread newsletters/notifications, not listed)",
    ]


def test_look_lines_none_from_people(ist) -> None:
    connect = server()
    assert inbox.look_lines(NOW, environ=ENV, connect=connect) == [
        "no unread mail from people in the last 2 days"]
    connect = server({1: {"msg": {"extra": "Precedence: bulk\r\n"}}})
    assert inbox.look_lines(NOW, environ=ENV, connect=connect) == [
        "no unread mail from people in the last 2 days",
        "(and 1 unread newsletters/notifications, not listed)",
    ]


def test_tool_text_not_configured() -> None:
    assert inbox.tool_text(environ={}, connect=server()) == (
        "no inbox is connected: set OMEGA_IMAP_USER and OMEGA_IMAP_PASSWORD "
        "(a Gmail app password, myaccount.google.com/apppasswords; needs 2-Step "
        "Verification) on the host")


def test_tool_text_empty_names_the_query() -> None:
    text = inbox.tool_text(sender="Kaushik", days=3, now=NOW, environ=ENV, connect=server())
    assert "Kaushik" in text and "3 days" in text and "unread" in text
    text = inbox.tool_text(unread_only=False, now=NOW, environ=ENV, connect=server())
    assert text == "no mail in the last 2 days"


def test_tool_text_tags_bulk_and_raises_on_failure(ist) -> None:
    connect = server({1: {"msg": {"extra": "Precedence: bulk\r\n", "body": b""}},
                      2: {"flags": ["\\Seen"], "msg": {"body": b""}}})
    text = inbox.tool_text(unread_only=False, now=NOW, environ=ENV, connect=connect)
    assert text.splitlines() == [
        "14:05 from Kaushik R <k@x.com>: lunch? (read)",  # same Date: higher UID first
        "14:05 from Kaushik R <k@x.com>: lunch? [bulk]",
    ]
    bad = server(fail={"login": imaplib.IMAP4.error(PASSWORD)})
    with pytest.raises(inbox.InboxUnavailable) as caught:
        inbox.tool_text(now=NOW, environ=ENV, connect=bad)
    assert PASSWORD not in str(caught.value)
