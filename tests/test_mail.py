"""Held mail and opening one mail in full — DL-082.

The inbox's fake IMAP server comes from ``test_inbox``: no network, no
account, and every command the code sent is there to check. The act loop runs
on ``provider.FakeProvider``, so what the model was handed and what the log
kept can be compared directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from omega import episodes, inbox, provider, tools, turn
from omega.act import act_loop
from omega.memory import MemoryStore
from omega.provider import ToolCall
from omega.queue import EventQueue
from omega.tools import EXPLORATION, ToolBox, ToolError, ToolRejected, ToolText
from omega.turn import TurnContext
from tests.test_inbox import (  # noqa: F401  (ist is a fixture)
    ENV,
    MUTATING,
    NOW,
    SETTINGS,
    fake,
    ist,
    server,
)

CODE = "482913"
LINK = "https://appleid.apple.com/reset?t=Zx81"
RESET = f"Your verification code is {CODE}.\r\nOr reset your password here: {LINK}\r\n"
APPLE = "Apple <appleid@id.apple.com>"


# --------------------------------------------------------------------------
# the classifier
# --------------------------------------------------------------------------


@pytest.mark.parametrize("sender, subject, body", [
    ("GitHub <noreply@github.com>", "[GitHub] Please reset your password", ""),
    ("Shop <help@shop.in>", "Password reset request", ""),
    ("Shop <help@shop.in>", "Forgot your password?", ""),
    ("Bank <care@bank.in>", "Forgotten password", ""),
    ("Google <no-reply@accounts.google.com>", "Account recovery request", ""),
    ("Google <no-reply@accounts.google.com>", "Your recovery email was changed", ""),
    ("Site <a@site.io>", "Here are your backup codes", ""),
    ("Site <a@site.io>", "Unlock your account", ""),
    ("Site <a@site.io>", "Recover your account", ""),
    ("Site <a@site.io>", "Your new recovery key", ""),
    ("Site <a@site.io>", "Review your recovery options", ""),
    ("Site <a@site.io>", "Recovery phone added", ""),
    ("Site <a@site.io>", "Hello", "Thanks!\n\n" + "x " * 300 + "\nTo RESET YOUR\nPASSWORD, click"),
    (APPLE, "Your receipt", ""),
    ("Microsoft account team <account-security-noreply@accountprotection.microsoft.com>",
     "Your monthly summary", ""),
    ("Apple <no_reply@email.apple.com>", "Your Apple Account was used to sign in", ""),
    ("Microsoft <msa@communication.microsoft.com>", "Microsoft account security code", ""),
    ("Microsoft <msa@microsoft.com>", "Verify your email address", ""),
    ("Apple <x@apple.com>", "Two-factor authentication is on", ""),
])
def test_held(sender: str, subject: str, body: str) -> None:
    assert inbox.is_held(sender, subject, body) is True


@pytest.mark.parametrize("sender, subject, body", [
    ("HDFC Bank <alerts@hdfcbank.net>", "OTP for your transaction",
     "Your OTP is 482913 for a payment of Rs 500. Do not share it."),
    ("Zomato <noreply@zomato.com>", "Your login code is 1234", "Use 1234 to sign in."),
    ("ICICI <alerts@icicibank.com>", "Rs 500 debited from your account",
     "Rs 500.00 debited from a/c XX1234 on 09-Oct. Not you? Call us."),
    ("Google <no-reply@accounts.google.com>", "Security alert",
     "A new sign-in on Mac. You're getting this email to make sure it was you. "
     "If not, we'll help you secure your account. Check activity."),
    ("Google <no-reply@accounts.google.com>", "New sign-in to your account", ""),
    ("Google <noreply@google.com>", "Your Google verification code", "Code: 123456"),
    ("Apple <news@insideapple.apple.com>", "The new iPhone is here", "Pre-order today."),
    ("Not Apple <a@notapple.com>", "Your account password", ""),
    ("Kaushik R <k@x.com>", "lunch?", "Are we on for lunch tomorrow?"),
    ("", "", ""),
])
def test_not_held(sender: str, subject: str, body: str) -> None:
    assert inbox.is_held(sender, subject, body) is False


def test_redact_masks_digits_links_and_addresses() -> None:
    text = (f"Code {CODE}, or open {LINK} or www.x.com/a and write help@x.co.in. "
            "Order 12 ok.")
    assert inbox.redact(text) == (
        "Code •••, or open [link] or [link] and write [address]. Order 12 ok.")


# --------------------------------------------------------------------------
# held mail in a listing
# --------------------------------------------------------------------------


def test_held_on_the_whole_body_and_never_snippeted(ist) -> None:
    body = ("Hi,\r\n\r\n" + "Lots of preamble. " * 30 + "\r\n" + RESET).encode()
    connect = server({1: {"msg": {"sender": "Site <a@site.io>", "subject": "Hello", "body": body}},
                      2: {}})
    held, plain = sorted(inbox.read(SETTINGS, connect=connect, now=NOW), key=lambda m: m.uid)
    assert held.held and held.snippet == "" and held.uid == 1
    assert not plain.held and plain.snippet
    out = inbox.look_lines(NOW, environ=ENV, connect=connect)
    assert "14:05 from Site <a@site.io>: Hello [held: opens only when they ask] #1" in out
    assert not [line for line in out if CODE in line or "preamble" in line]


def test_held_too_large_to_fetch_decided_on_subject_and_sender() -> None:
    connect = server({1: {"size": 300_000, "msg": {"body": RESET.encode()}},
                      2: {"size": 300_000, "msg": {"sender": APPLE}}})
    mails = {m.uid: m for m in inbox.read(SETTINGS, connect=connect, now=NOW)}
    assert mails[1].held is False  # its body would have held it; nobody read it
    assert mails[2].held is True


# --------------------------------------------------------------------------
# opening one mail
# --------------------------------------------------------------------------


def test_open_mail_is_read_only_and_marks_nothing_read() -> None:
    connect = server({5: {"flags": [], "msg": {"body": b"hello"}}})
    opened = inbox.open_mail(5, environ=ENV, connect=connect)
    assert opened.text == "hello" and opened.uid == 5
    calls = fake().calls
    assert ("SELECT", "INBOX", True) in calls
    fetches = [c for c in calls if c[:2] == ("UID", "FETCH")]
    assert [c[2] for c in fetches] == [b"5", b"5"]
    assert all("BODY.PEEK[" in c[3] for c in fetches)
    assert not [c for c in fetches if "BODY[" in c[3].replace("BODY.PEEK[", "")]
    names = {c[0] if c[0] != "UID" else c[1] for c in calls}
    assert not names & MUTATING
    assert fake().mails[5].get("flags") == []  # still unread
    assert calls[-1] == ("LOGOUT",)


def test_open_mail_keeps_paragraphs_and_strips_control_chars() -> None:
    body = b"Hi  there,\r\n\r\n\r\n\r\nLine\x07 two\xe2\x80\xae\r\n   \r\nBye\r\n"
    connect = server({1: {"msg": {"body": body}}})
    assert inbox.open_mail(1, environ=ENV, connect=connect).text == (
        "Hi there,\n\nLine two\n\nBye")


def test_open_mail_html_paragraphs_and_plain_preferred() -> None:
    page = (b"<html><head><style>p{}</style></head><body><p>One &amp; done</p>"
            b"<p>Two<br>three</p><script>x()</script></body></html>")
    connect = server({1: {"msg": {"body": page, "content_type": "text/html"}}})
    assert inbox.open_mail(1, environ=ENV, connect=connect).text == "One & done\nTwo\nthree"
    raw = (b"From: K <k@x.com>\r\nSubject: s\r\nDate: Fri, 09 Oct 2026 14:05:00 +0530\r\n"
           b"MIME-Version: 1.0\r\nContent-Type: multipart/mixed; boundary=XX\r\n\r\n"
           b"--XX\r\nContent-Type: text/plain; charset=utf-8\r\n\r\nsee attached\r\n"
           b"--XX\r\nContent-Type: application/pdf\r\nContent-Disposition: attachment; "
           b"filename=a.pdf\r\nContent-Transfer-Encoding: base64\r\n\r\nJVBERi0xLjQK\r\n"
           b"--XX--\r\n")
    connect = server({1: {"msg": {"raw": raw}}})
    opened = inbox.open_mail(1, environ=ENV, connect=connect)
    assert opened.text == "see attached"
    assert opened.sender == "K <k@x.com>" and opened.subject == "s"


def test_open_mail_truncates() -> None:
    connect = server({1: {"msg": {"body": b"w" * 30_000}}})
    text = inbox.open_mail(1, environ=ENV, connect=connect).text
    assert text == "w" * inbox.OPEN_CHARS + "\n… (truncated)"


def test_open_mail_held_is_classified_on_open() -> None:
    connect = server({1: {"msg": {"body": RESET.encode()}}})
    opened = inbox.open_mail(1, environ=ENV, connect=connect)
    assert opened.held and CODE in opened.text and LINK in opened.text


def test_open_mail_refusals() -> None:
    with pytest.raises(inbox.InboxUnavailable, match="^no mail #9 in the inbox$"):
        inbox.open_mail(9, environ=ENV, connect=server({1: {}}))
    connect = server({1: {"size": 6_000_000}})
    with pytest.raises(inbox.InboxUnavailable, match="too large to open"):
        inbox.open_mail(1, environ=ENV, connect=connect)
    assert not [c for c in fake().calls if c[:2] == ("UID", "FETCH") and c[3] == "(BODY.PEEK[])"]
    with pytest.raises(inbox.InboxUnavailable, match="no inbox is connected"):
        inbox.open_mail(1, environ={}, connect=server())


# --------------------------------------------------------------------------
# the tool
# --------------------------------------------------------------------------


def _served(monkeypatch: pytest.MonkeyPatch, mails: dict[int, dict[str, Any]]) -> None:
    connect = server(mails)
    real = inbox.open_mail
    monkeypatch.setattr(inbox, "open_mail", lambda uid: real(uid, environ=ENV, connect=connect))


@pytest.fixture
def room(tmp_path: Path) -> Path:
    path = tmp_path / "store"
    path.mkdir()
    return path


@pytest.mark.parametrize("args", [{}, {"id": "3"}, {"id": True}, {"id": 0}, {"id": -1},
                                  {"id": 2.5}, {"id": None}])
def test_mail_refuses_an_id_that_is_not_a_whole_number(room: Path, args: dict) -> None:
    with pytest.raises(ToolRejected):
        ToolBox(store_root=room).classify("mail", args)


def test_ordinary_mail_opens_in_any_turn_and_is_plain_text(
    room: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _served(monkeypatch, {3: {}})
    box = ToolBox(store_root=room)  # not prompted
    decision = box.classify("mail", {"id": 3})
    assert decision.tier == EXPLORATION and decision.args == {"id": 3}
    text = box.dispatch(decision)
    assert isinstance(text, str)
    assert text.splitlines()[:2] == ["From: Kaushik R <k@x.com>", "Subject: lunch?"]
    assert tools.MAIL_LEAD in text and text.endswith("Are we on for lunch tomorrow?")


def test_held_mail_refused_unless_their_own_message_asked(
    room: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _served(monkeypatch, {4: {"msg": {"sender": APPLE, "subject": "Reset your password",
                                      "body": RESET.encode()}}})
    with pytest.raises(ToolError, match="held .* opens only when they ask for it"):
        ToolBox(store_root=room).dispatch(ToolBox(store_root=room).classify("mail", {"id": 4}))
    box = ToolBox(store_root=room, prompted=True)
    result = box.dispatch(box.classify("mail", {"id": 4}))
    assert isinstance(result, ToolText)
    assert CODE in result.model and LINK in result.model
    assert result.log.startswith(tools.HELD_LEAD + "from Apple <appleid@id.apple.com>: ")
    assert CODE not in result.log and "https://" not in result.log and "•••" in result.log


def test_mail_unavailable_is_a_tool_error(room: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _served(monkeypatch, {})
    box = ToolBox(store_root=room, prompted=True)
    with pytest.raises(ToolError, match="^could not read the mail: no mail #9 in the inbox$"):
        box.dispatch(box.classify("mail", {"id": 9}))


def test_inbox_tool_output_is_capped(room: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(inbox, "tool_text", lambda **kw: "x" * 20_000)
    box = ToolBox(store_root=room)
    text = box.dispatch(box.classify("inbox", {}))
    assert len(text) < 9_000 and text.endswith("[truncated at 8000 characters]")


# --------------------------------------------------------------------------
# the act loop: the gate from the turn's origin, and what the log keeps
# --------------------------------------------------------------------------


@pytest.fixture
def q(store: MemoryStore) -> EventQueue:
    return EventQueue(store)


def _run(q: EventQueue, channel: str) -> tuple[provider.FakeProvider, dict]:
    fp = provider.FakeProvider({provider.ACT: [ToolCall(id="c1", name="mail",
                                                        arguments={"id": 4}), ""]})
    seq = q.append(episodes.inbound("what's that apple code?", channel=channel,
                                    at="2026-10-09T09:00:00+00:00"))
    q.claim(seq)
    ctx = TurnContext(seq=seq, event=q.at(seq).payload, recalled=[], queue=q,
                      complete=fp.complete)
    act_loop(ctx)  # builds its own box, so `prompted` comes from the turn
    [returned] = [p.payload for p in q.recent(50)
                  if p.payload.get("kind") == episodes.TOOL_RETURNED
                  and p.payload.get("for_seq") == seq]
    return fp, returned


def _handed(fp: provider.FakeProvider) -> str:
    second = fp.calls_for(provider.ACT)[1]
    return "\n".join(m["content"] for m in second if m.get("role") == "tool")


def test_their_message_opens_held_mail_and_the_log_keeps_only_a_digest(
    q: EventQueue, monkeypatch: pytest.MonkeyPatch
) -> None:
    _served(monkeypatch, {4: {"msg": {"sender": APPLE, "subject": "Your code",
                                      "body": RESET.encode()}}})
    fp, returned = _run(q, "tray")
    assert CODE in _handed(fp) and LINK in _handed(fp)
    assert returned["ok"] is True
    assert returned["result"].startswith(tools.HELD_LEAD)
    assert CODE not in returned["result"] and LINK not in returned["result"]
    assert "https://" not in returned["result"]


def test_an_unprompted_turn_cannot_open_held_mail(
    q: EventQueue, monkeypatch: pytest.MonkeyPatch
) -> None:
    _served(monkeypatch, {4: {"msg": {"sender": APPLE, "body": RESET.encode()}}})
    for channel in turn.UNPROMPTED_CHANNELS:
        fp, returned = _run(q, channel)
        assert returned["ok"] is False and "held" in returned["error"]
        assert CODE not in _handed(fp) and "held" in _handed(fp)


def test_ordinary_opened_mail_is_logged_in_full(
    q: EventQueue, monkeypatch: pytest.MonkeyPatch
) -> None:
    _served(monkeypatch, {4: {}})
    fp, returned = _run(q, "self")
    # The log keeps what came back; the model saw it fenced as data (DL-064).
    assert returned["ok"] is True and tools.fence("mail #4", returned["result"]) == _handed(fp)


# --------------------------------------------------------------------------
# recall renders an opened mail as one line
# --------------------------------------------------------------------------


def _returned(result: str, tool: str = "mail", ok: bool = True) -> dict:
    return episodes.tool_returned(for_seq=1, tool=tool, ok=ok,
                                  result=result if ok else None,
                                  error=None if ok else "nope")


def test_recall_renders_mail_as_a_digest() -> None:
    full = "From: Kaushik R <k@x.com>\nSubject: lunch?\nDate: Fri\n\n(lead)\n" + "body " * 500
    assert turn._render_payload(_returned(full)) == (
        "tool mail ok: opened mail from Kaushik R <k@x.com>: lunch?")
    digest = tools.HELD_LEAD + "from Apple <a@id.apple.com>: Your code — Code ••• " + "x" * 400
    rendered = turn._render_payload(_returned(digest))
    assert rendered.startswith("tool mail ok: " + tools.HELD_LEAD)
    assert len(rendered) <= len("tool mail ok: ") + turn.MAIL_DIGEST_CHARS
    assert turn._render_payload(_returned("", ok=False)) == "tool mail failed: nope"
    assert turn._render_payload(_returned(full, tool="inbox")) == "tool inbox ok"
