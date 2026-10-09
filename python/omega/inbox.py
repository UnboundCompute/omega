"""The inbox sense — the person's Gmail, read over IMAP, never written. DL-080.

A helper beside the person would notice "Kaushik wrote back" without being
asked, and could answer "did he reply?" when asked. This module is that and
nothing more: the look gets a few lines about unread mail from *people*, and a
tool gets the same reading on demand, narrowed to a sender.

**Strictly read-only, enforced in the protocol, not by intent.** The mailbox
is opened with ``EXAMINE`` (``select(readonly=True)``), every fetch is
``BODY.PEEK[...]`` — a plain ``BODY[...]`` or ``RFC822`` would set ``\\Seen``
and quietly mark the person's mail as read — and no ``STORE``, ``COPY``,
``EXPUNGE`` or ``APPEND`` is ever issued. Looking must leave the inbox exactly
as the person left it.

**The password is an app password, and it goes only to ``login``.** It never
appears in a :class:`InboxUnavailable` reason, a returned line, or a repr:
those reach the log and the console. Reasons are scrubbed of it as a last
guard, not as the plan.

**Everything read here is other people's text.** Senders, subjects and bodies
are written by strangers and land in a prompt, so each is decoded (RFC 2047),
stripped of control and bidi-format characters, collapsed to one line and cut
short before it leaves this module.

**Three-valued, never empty-as-all-clear.** :func:`look_lines` says "not
connected" (nothing), "couldn't read" (a line saying so) or what it found —
including, explicitly, that it found nothing. An inbox that failed to load
must never read as an inbox with nothing in it.
"""

from __future__ import annotations

import email
import email.policy
import html
import imaplib
import os
import re
import socket
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Mapping, Optional, Sequence

__all__ = [
    "ENV_USER",
    "ENV_PASSWORD",
    "ENV_HOST",
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "LOOK_DAYS",
    "MAX_MAILS",
    "MAX_BODY_BYTES",
    "InboxUnavailable",
    "Settings",
    "Mail",
    "settings",
    "read",
    "lines",
    "look_lines",
    "tool_text",
]

ENV_USER = "OMEGA_IMAP_USER"
ENV_PASSWORD = "OMEGA_IMAP_PASSWORD"
ENV_HOST = "OMEGA_IMAP_HOST"

DEFAULT_HOST = "imap.gmail.com"
DEFAULT_PORT = 993

#: How far back the look reads. Two days covers "since yesterday morning"
#: without dredging up a week of mail the person has already decided to ignore.
LOOK_DAYS = 2

#: At most this many mails are fetched per reading, newest first.
MAX_MAILS = 20

#: Above this size a message's body is not fetched at all — it is almost always
#: an attachment, and a snippet is not worth pulling megabytes for.
MAX_BODY_BYTES = 256_000

#: Per connection. The look runs on a clock; a hung server must not hold it.
TIMEOUT_SECONDS = 15

SENDER_CHARS = 80
SUBJECT_CHARS = 140
SNIPPET_CHARS = 200

#: A ``FROM`` search term longer than this is not a name or an address.
MAX_SENDER_QUERY = 100

#: The only header fields fetched for the listing.
_HEADER_FIELDS = "FROM SUBJECT DATE LIST-UNSUBSCRIBE LIST-ID PRECEDENCE AUTO-SUBMITTED"
_META_FETCH = f"(FLAGS RFC822.SIZE BODY.PEEK[HEADER.FIELDS ({_HEADER_FIELDS})])"
_BODY_FETCH = "(BODY.PEEK[])"

_AUTH_REASON = f"login refused (check {ENV_USER} and the app password)"

_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


class InboxUnavailable(Exception):
    """The inbox could not be read. The message is a short reason, never the password."""


@dataclass(frozen=True)
class Settings:
    user: str
    password: str = field(repr=False)
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT


@dataclass(frozen=True)
class Mail:
    sender: str               # "Kaushik R <k@x.com>", sanitized
    subject: str
    date: Optional[datetime]  # tz-aware; None when the Date header is unparseable
    snippet: str              # start of the text body, "" when none or too large
    bulk: bool                # a list, newsletter or machine-sent notification
    unread: bool


# --------------------------------------------------------------------------
# settings
# --------------------------------------------------------------------------


def settings(environ: Mapping[str, str] = os.environ) -> Optional[Settings]:
    """The IMAP settings, or ``None`` unless both user and password are set."""
    user = (environ.get(ENV_USER) or "").strip()
    password = (environ.get(ENV_PASSWORD) or "").strip()
    if not user or not password:
        return None
    host = (environ.get(ENV_HOST) or "").strip() or DEFAULT_HOST
    return Settings(user=user, password=password, host=host)


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------


def read(
    s: Settings,
    *,
    days: int = LOOK_DAYS,
    unread_only: bool = True,
    sender: Optional[str] = None,
    limit: int = MAX_MAILS,
    connect: Callable[..., Any] = imaplib.IMAP4_SSL,
    now: Optional[datetime] = None,
) -> list[Mail]:
    """Up to ``limit`` mails from the last ``days`` days, newest first.

    Raises :class:`ValueError` for a sender that cannot be searched safely and
    :class:`InboxUnavailable` for anything the server or the network did.
    """
    criteria = _criteria(days, unread_only, sender, now)
    imap = None
    phase = "connect"
    try:
        imap = connect(s.host, s.port, timeout=TIMEOUT_SECONDS)
        phase = "login"
        imap.login(s.user, s.password)
        phase = "read"
        _ok(imap.select("INBOX", readonly=True), "select")
        typ, data = imap.uid("SEARCH", None, *criteria)
        _ok((typ, data), "search")
        uids = _uids(data)[-limit:] if limit > 0 else []
        mails = [_fetch(imap, uid) for uid in reversed(uids)]
    except (imaplib.IMAP4.error, OSError) as exc:
        raise InboxUnavailable(_reason(exc, phase, s.password)) from None
    finally:
        if imap is not None:
            _logout(imap)
    return _newest_first([m for m in mails if m is not None])


def _criteria(days: int, unread_only: bool, sender: Optional[str],
              now: Optional[datetime]) -> list[str]:
    """The SEARCH terms. ``SINCE`` is a date in IMAP, so local midnight counts."""
    local = (now or datetime.now(timezone.utc)).astimezone()
    since = (local - timedelta(days=days)).date()
    terms = ["SINCE", f"{since.day:02d}-{_MONTHS[since.month - 1]}-{since.year}"]
    if unread_only:
        terms.append("UNSEEN")
    if sender is not None:
        terms += ["FROM", _quote(sender)]
    return terms


def _quote(sender: str) -> str:
    """An IMAP quoted string, refusing anything that could break out of one."""
    if "\r" in sender or "\n" in sender:
        raise ValueError("sender must be one line")
    if len(sender) > MAX_SENDER_QUERY:
        raise ValueError(f"sender must be at most {MAX_SENDER_QUERY} characters")
    if not sender.isascii():
        # imaplib sends arguments as ASCII; a non-ASCII term would need a
        # literal and a CHARSET, which is more protocol than a name is worth.
        raise ValueError("sender must be ASCII (search by address instead)")
    escaped = sender.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _ok(response: tuple[str, Any], what: str) -> None:
    typ, data = response
    if typ != "OK":
        raise imaplib.IMAP4.error(f"{what} {typ}: {_text(data)}")


def _uids(data: Sequence[Any]) -> list[bytes]:
    found: list[bytes] = []
    for chunk in data or []:
        if isinstance(chunk, bytes):
            found += [u for u in chunk.split() if u.isdigit()]
    return sorted(found, key=int)  # ascending UID = oldest first


def _fetch(imap: Any, uid: bytes) -> Optional[Mail]:
    """One mail: flags, size and the listing headers, then the body if small."""
    typ, data = imap.uid("FETCH", uid, _META_FETCH)
    _ok((typ, data), "fetch")
    meta, header = _parts(data, b"HEADER.FIELDS")
    if header is None:
        return None  # expunged between search and fetch
    flags = re.search(rb"FLAGS \(([^)]*)\)", meta)
    size = re.search(rb"RFC822\.SIZE (\d+)", meta)
    headers = email.message_from_bytes(header, policy=email.policy.default)

    snippet = ""
    if size is not None and int(size.group(1)) <= MAX_BODY_BYTES:
        typ, data = imap.uid("FETCH", uid, _BODY_FETCH)
        _ok((typ, data), "fetch")
        _, raw = _parts(data, b"BODY[]")
        if raw is not None:
            snippet = _snippet(raw)

    return Mail(
        sender=_clean(_header(headers, "From"), SENDER_CHARS),
        subject=_clean(_header(headers, "Subject"), SUBJECT_CHARS),
        date=_date(_header(headers, "Date")),
        snippet=snippet,
        bulk=_is_bulk(headers),
        unread=flags is None or b"\\Seen" not in flags.group(1).split(),
    )


def _parts(data: Sequence[Any], marker: bytes) -> tuple[bytes, Optional[bytes]]:
    """Split a FETCH response into its metadata text and the literal after ``marker``.

    imaplib returns ``[(b'1 (UID 1 FLAGS (...) BODY[...] {n}', literal), b')']``;
    servers may put FLAGS after the literal instead, in the trailing bytes, so
    every non-literal piece is joined into the metadata.
    """
    meta = b""
    literal = None
    for chunk in data or []:
        if isinstance(chunk, tuple) and len(chunk) == 2:
            meta += chunk[0] + b" "
            if marker in chunk[0] and literal is None:
                literal = chunk[1]
        elif isinstance(chunk, bytes):
            meta += chunk + b" "
    return meta, literal


def _header(msg: Any, name: str) -> str:
    """A header's decoded value; a malformed one falls back to its raw text."""
    try:
        value = msg.get(name)
    except Exception:  # a header the policy cannot parse
        value = None
        for key, raw in msg.raw_items():
            if key.lower() == name.lower():
                value = raw
                break
    return "" if value is None else str(value)


def _date(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        return None
    if parsed is None:
        return None
    # "-0000" means "zone unknown" and parses naive; UTC is the honest reading.
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _is_bulk(msg: Any) -> bool:
    """A mailing list, a newsletter, or a machine — not a person writing."""
    if _header(msg, "List-Unsubscribe") or _header(msg, "List-Id"):
        return True
    if _header(msg, "Precedence").strip().lower() in ("bulk", "list", "junk"):
        return True
    auto = _header(msg, "Auto-Submitted").strip().lower()
    return bool(auto) and auto != "no"


def _snippet(raw: bytes) -> str:
    """The start of the text body: plain preferred, else HTML with tags stripped."""
    try:
        msg = email.message_from_bytes(raw, policy=email.policy.default)
        assert isinstance(msg, EmailMessage)
        part = msg.get_body(preferencelist=("plain", "html"))
        if part is None:
            return ""
        text = part.get_content()
    except Exception:  # an undecodable body is just a mail without a snippet
        return ""
    if not isinstance(text, str):
        return ""
    if part.get_content_subtype() == "html":
        text = _strip_html(text)
    return _clean(text, SNIPPET_CHARS)


def _strip_html(text: str) -> str:
    text = re.sub(r"(?is)<(script|style|head)\b.*?</\1\s*>", " ", text)
    text = re.sub(r"(?s)<!--.*?-->", " ", text)
    text = re.sub(r"(?s)<[^>]*>", " ", text)
    return html.unescape(text)


def _clean(text: str, limit: int) -> str:
    """One line of someone else's text: no control characters, bounded length."""
    text = "".join(
        " " if ch.isspace() else ch
        for ch in text
        if ch.isspace() or unicodedata.category(ch) not in ("Cc", "Cf")
    )
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _newest_first(mails: list[Mail]) -> list[Mail]:
    """By Date when there is one; undated mail keeps its UID order, at the end."""
    floor = datetime.min.replace(tzinfo=timezone.utc)
    return sorted(mails, key=lambda m: m.date or floor, reverse=True)


def _logout(imap: Any) -> None:
    try:
        imap.logout()
    except Exception:  # a dead connection cannot be logged out of; nothing to do
        pass


def _reason(exc: BaseException, phase: str, password: str) -> str:
    """A short, password-free reason for :class:`InboxUnavailable`."""
    if isinstance(exc, (socket.timeout, TimeoutError)):
        return "timed out"
    if phase == "login" and isinstance(exc, imaplib.IMAP4.error):
        return _AUTH_REASON
    text = _clean(_text(exc.args[0] if exc.args else ""), 120)
    if password:
        text = text.replace(password, "***")
    name = type(exc).__name__
    return f"{name}: {text}" if text else name


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, (list, tuple)):
        return " ".join(_text(v) for v in value)
    return str(value)


# --------------------------------------------------------------------------
# wording
# --------------------------------------------------------------------------


def lines(mails: Sequence[Mail], now: datetime) -> list[str]:
    """One line per mail; already-read ones are marked " (read)"."""
    return [_line(m, now) for m in mails]


def _line(mail: Mail, now: datetime, *, tag_bulk: bool = False) -> str:
    text = f"{_when(mail.date, now)} from {mail.sender or 'unknown sender'}: "
    text += mail.subject or "(no subject)"
    if mail.snippet:
        text += f" — {mail.snippet}"
    if not mail.unread:
        text += " (read)"
    if tag_bulk and mail.bulk:
        text += " [bulk]"
    return text


def _when(date: Optional[datetime], now: datetime) -> str:
    """'14:05' today, 'yesterday 18:20', else 'Tue 06 Oct' — all local time."""
    if date is None:
        return "undated"
    local, today = date.astimezone(), now.astimezone().date()
    if local.date() == today:
        return local.strftime("%H:%M")
    if local.date() == today - timedelta(days=1):
        return "yesterday " + local.strftime("%H:%M")
    return local.strftime("%a %d %b")


def look_lines(
    now: Optional[datetime] = None,
    *,
    environ: Mapping[str, str] = os.environ,
    connect: Callable[..., Any] = imaplib.IMAP4_SSL,
) -> list[str]:
    """What the look says about the inbox: nothing, a failure, or what is there."""
    s = settings(environ)
    if s is None:
        return []
    now = now or datetime.now(timezone.utc)
    try:
        mails = read(s, days=LOOK_DAYS, unread_only=True, connect=connect, now=now)
    except InboxUnavailable as exc:
        return [f"couldn't read the inbox: {exc}"]
    people = [m for m in mails if not m.bulk]
    bulk = len(mails) - len(people)
    out = lines(people, now) if people else [
        f"no unread mail from people in the last {LOOK_DAYS} days"
    ]
    if bulk:
        out.append(f"(and {bulk} unread newsletters/notifications, not listed)")
    return out


NOT_CONNECTED = (
    f"no inbox is connected: set {ENV_USER} and {ENV_PASSWORD} (a Gmail app "
    "password, myaccount.google.com/apppasswords; needs 2-Step Verification) "
    "on the host"
)


def tool_text(
    *,
    sender: Optional[str] = None,
    days: int = LOOK_DAYS,
    unread_only: bool = True,
    now: Optional[datetime] = None,
    environ: Mapping[str, str] = os.environ,
    connect: Callable[..., Any] = imaplib.IMAP4_SSL,
) -> str:
    """The inbox as a tool answer. Raises :class:`InboxUnavailable` on failure."""
    s = settings(environ)
    if s is None:
        return NOT_CONNECTED
    now = now or datetime.now(timezone.utc)
    mails = read(s, days=days, unread_only=unread_only, sender=sender,
                 connect=connect, now=now)
    if not mails:
        kind = "unread mail" if unread_only else "mail"
        who = f" from {_clean(sender, SENDER_CHARS)}" if sender else ""
        span = "day" if days == 1 else f"{days} days"
        return f"no {kind}{who} in the last {span}"
    return "\n".join(_line(m, now, tag_bulk=True) for m in mails)
