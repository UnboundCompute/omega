"""The inbox sense — the person's Gmail, read over IMAP, never written. DL-080.

A helper beside the person would notice "Kaushik wrote back" without being
asked, and could answer "did he reply?" when asked. This module is that and
nothing more: the look gets a few lines about recent mail from *people*, and a
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

**Held mail, and opening one mail in full (DL-082).** Recovery mail of any
kind, and Apple ID / Microsoft account mail, is *held*: :func:`is_held`
decides it here, in code, before any model sees a word, and a held mail shows
only sender, time and subject anywhere it is listed. :func:`open_mail` reads
one mail whole by its UID — the ``#id`` every listed line ends with — on the
same read-only select with the same peek. Whether a held mail may be opened is
the tool's gate, not this module's; :func:`redact` is what lets the log keep
that it was opened without keeping the code or the link in it.
"""

from __future__ import annotations

import email
import email.policy
import html
import imaplib
import sys
import os
import re
import socket
import unicodedata
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from email.utils import parseaddr, parsedate_to_datetime
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence

__all__ = [
    "ENV_USER",
    "ENV_PASSWORD",
    "ENV_HOST",
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "LOOK_DAYS",
    "MAX_MAILS",
    "MAX_BODY_BYTES",
    "MAX_OPEN_BYTES",
    "OPEN_CHARS",
    "HELD_MARK",
    "TEXT_CHARS",
    "InboxUnavailable",
    "Settings",
    "Mail",
    "Opened",
    "settings",
    "read",
    "is_held",
    "open_mail",
    "redact",
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

#: Opening one mail (DL-082) fetches more than a listing would, but not
#: anything: above this the mail is an attachment carrying a message.
MAX_OPEN_BYTES = 5_000_000

#: An opened mail's text body is cut here. A letter fits; a mailing-list
#: digest is cut and says so.
OPEN_CHARS = 20_000

#: What a held mail shows in place of its snippet (DL-082 #2).
HELD_MARK = "[held: opens only when they ask]"

#: Per connection. The look runs on a clock; a hung server must not hold it.
TIMEOUT_SECONDS = 15

SENDER_CHARS = 80
SUBJECT_CHARS = 140
SNIPPET_CHARS = 200

#: How much of a listed mail's text body :class:`Mail` keeps beside the
#: snippet, for the finance sweep (DL-083): a bank alert's amount is often
#: past the first 200 characters, rarely past the first 4000.
TEXT_CHARS = 4_000
MESSAGE_ID_CHARS = 200

#: A ``FROM`` search term longer than this is not a name or an address.
MAX_SENDER_QUERY = 100

#: The only header fields fetched for the listing.
_HEADER_FIELDS = (
    "FROM SUBJECT DATE MESSAGE-ID LIST-UNSUBSCRIBE LIST-ID PRECEDENCE AUTO-SUBMITTED"
)
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
    snippet: str              # start of the text body; "" when none, too large or held
    bulk: bool                # a list, newsletter or machine-sent notification
    unread: bool
    uid: int                  # the IMAP UID, shown as "#uid" and what `mail` opens
    held: bool                # recovery or Apple/Microsoft account mail (DL-082)
    # DL-083, both defaulted so a Mail built by hand still reads as before.
    message_id: str = ""      # the Message-ID header, sanitized; "" when absent
    text: str = ""            # the text body's start, one line, TEXT_CHARS; "" when held


@dataclass(frozen=True)
class Opened:
    """One mail read whole by :func:`open_mail` (DL-082)."""

    uid: int
    sender: str
    subject: str
    date: Optional[datetime]
    held: bool
    text: str                 # the text body, paragraphs kept, capped at OPEN_CHARS


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
    with _session(s, connect) as imap:
        typ, data = imap.uid("SEARCH", None, *criteria)
        _ok((typ, data), "search")
        uids = _uids(data)[-limit:] if limit > 0 else []
        mails = [_fetch(imap, uid) for uid in reversed(uids)]
    return _newest_first([m for m in mails if m is not None])


@contextmanager
def _session(s: Settings, connect: Callable[..., Any]) -> Iterator[Any]:
    """A logged-in connection with INBOX selected **read-only**, always logged out.

    Anything the server or the network does inside it becomes
    :class:`InboxUnavailable` with a password-free reason.
    """
    imap = None
    phase = "connect"
    try:
        imap = connect(s.host, s.port, timeout=TIMEOUT_SECONDS)
        phase = "login"
        imap.login(s.user, s.password)
        phase = "read"
        _ok(imap.select("INBOX", readonly=True), "select")
        yield imap
    except (imaplib.IMAP4.error, OSError) as exc:
        raise InboxUnavailable(_reason(exc, phase, s.password)) from None
    finally:
        if imap is not None:
            _logout(imap)


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

    body = ""
    if size is not None and int(size.group(1)) <= MAX_BODY_BYTES:
        typ, data = imap.uid("FETCH", uid, _BODY_FETCH)
        _ok((typ, data), "fetch")
        _, raw = _parts(data, b"BODY[]")
        if raw is not None:
            body = _body_text(raw)

    # Held is decided on the whole body, not the snippet: "reset your
    # password" is rarely in the first 200 characters. Too large to fetch
    # means subject and sender only.
    held = is_held(_header(headers, "From"), _header(headers, "Subject"), body)
    return Mail(
        sender=_clean(_header(headers, "From"), SENDER_CHARS),
        subject=_clean(_header(headers, "Subject"), SUBJECT_CHARS),
        date=_date(_header(headers, "Date")),
        snippet="" if held else _clean(body, SNIPPET_CHARS),
        bulk=_is_bulk(headers),
        unread=flags is None or b"\\Seen" not in flags.group(1).split(),
        uid=int(uid),
        held=held,
        message_id=_clean(_header(headers, "Message-ID"), MESSAGE_ID_CHARS),
        text="" if held else _clean(body, TEXT_CHARS),
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


# --------------------------------------------------------------------------
# held mail (DL-082)
# --------------------------------------------------------------------------

#: Recovery of any kind, in a subject or a body. Not OTPs, not transaction
#: mail, not "new sign-in" alerts: the person settled that those are fine.
_RECOVERY = re.compile(
    r"\b(?:password\s+reset"
    r"|reset\s+(?:your\s+|my\s+|the\s+)?password"
    r"|forgot(?:ten)?\s+(?:your\s+|my\s+|the\s+)?password"
    r"|account\s+recovery"
    r"|recover\s+(?:your|my|the)\s+account"
    r"|recovery\s+(?:codes?|e-?mails?|phones?|keys?|options?)"
    r"|backup\s+codes?"
    r"|unlock\s+(?:your|my|the)\s+account)\b",
    re.IGNORECASE,
)

#: Senders whose every mail is about the account itself.
_ACCOUNT_DOMAINS = ("id.apple.com", "accountprotection.microsoft.com")

#: Senders whose account mail is held when the subject says it is one.
_VENDOR_DOMAINS = ("apple.com", "microsoft.com")
_ACCOUNT_SUBJECT = re.compile(
    r"account|sign[- ]?in|password|security|verif|two[- ]factor|2[- ]step|code",
    re.IGNORECASE,
)


def is_held(sender: str, subject: str, body_text: str) -> bool:
    """Is this mail held — shown by subject only, opened only when they ask?

    Deterministic, and decided before any model reads the mail (DL-082 #1):
    an LLM judging sensitivity would have to read the thing it protects. A
    spoofed sender can only make more mail held, never less.
    """
    if _RECOVERY.search(subject or "") or _RECOVERY.search(body_text or ""):
        return True
    domain = parseaddr(sender or "")[1].rpartition("@")[2].strip().lower().rstrip(".")
    if not domain:
        return False
    if any(_within(domain, d) for d in _ACCOUNT_DOMAINS):
        return True
    return any(_within(domain, d) for d in _VENDOR_DOMAINS) and bool(
        _ACCOUNT_SUBJECT.search(subject or "")
    )


def _within(domain: str, parent: str) -> bool:
    return domain == parent or domain.endswith("." + parent)


_URL = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
_ADDRESS = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_DIGITS = re.compile(r"\d{3,}")


def redact(text: str) -> str:
    """Text with what a recovery mail's secret is made of masked: links,
    addresses and runs of three or more digits. What the log keeps of an
    opened held mail (DL-082 #5)."""
    text = _URL.sub("[link]", text)
    text = _ADDRESS.sub("[address]", text)
    return _DIGITS.sub("•••", text)


# --------------------------------------------------------------------------
# opening one mail (DL-082)
# --------------------------------------------------------------------------


def open_mail(
    uid: int,
    *,
    environ: Mapping[str, str] = os.environ,
    connect: Callable[..., Any] = imaplib.IMAP4_SSL,
) -> Opened:
    """One mail, whole, by UID — on a read-only select, with peeks only.

    Raises :class:`InboxUnavailable` when there is no inbox, no such mail, a
    mail too large to open, or anything the server or the network did.
    """
    s = settings(environ)
    if s is None:
        raise InboxUnavailable(NOT_CONNECTED)
    key = str(int(uid)).encode()
    with _session(s, connect) as imap:
        typ, data = imap.uid("FETCH", key, _META_FETCH)
        _ok((typ, data), "fetch")
        meta, header = _parts(data, b"HEADER.FIELDS")
        if header is None:
            raise InboxUnavailable(f"no mail #{uid} in the inbox")
        size = re.search(rb"RFC822\.SIZE (\d+)", meta)
        if size is not None and int(size.group(1)) > MAX_OPEN_BYTES:
            mb = int(size.group(1)) / 1_000_000
            raise InboxUnavailable(
                f"mail #{uid} is {mb:.0f} MB, too large to open "
                f"(at most {MAX_OPEN_BYTES // 1_000_000} MB)"
            )
        typ, data = imap.uid("FETCH", key, _BODY_FETCH)
        _ok((typ, data), "fetch")
        _, raw = _parts(data, b"BODY[]")
    headers = email.message_from_bytes(header, policy=email.policy.default)
    body = _body_text(raw) if raw is not None else ""
    return Opened(
        uid=int(uid),
        sender=_clean(_header(headers, "From"), SENDER_CHARS),
        subject=_clean(_header(headers, "Subject"), SUBJECT_CHARS),
        date=_date(_header(headers, "Date")),
        held=is_held(_header(headers, "From"), _header(headers, "Subject"), body),
        text=_paragraphs(body),
    )


def _body_text(raw: bytes) -> str:
    """The whole text body: plain preferred, else HTML with tags stripped.

    Not yet cleaned: :func:`_clean` makes a snippet of it, :func:`_paragraphs`
    an opened mail. Attachments are never a body, so never read.
    """
    try:
        msg = email.message_from_bytes(raw, policy=email.policy.default)
        assert isinstance(msg, EmailMessage)
        part = msg.get_body(preferencelist=("plain", "html"))
        if part is None:
            return ""
        text = part.get_content()
    except Exception:  # an undecodable body is just a mail without a text body
        return ""
    if not isinstance(text, str):
        return ""
    if part.get_content_subtype() == "html":
        text = _strip_html(text)
    return text


def _strip_html(text: str) -> str:
    text = re.sub(r"(?is)<(script|style|head)\b.*?</\1\s*>", " ", text)
    text = re.sub(r"(?s)<!--.*?-->", " ", text)
    # Block ends become line breaks so an opened mail keeps its paragraphs.
    text = re.sub(r"(?i)<br\s*/?>|</(?:p|div|tr|li|h[1-6]|table|blockquote)\s*>", "\n", text)
    text = re.sub(r"(?s)<[^>]*>", " ", text)
    return html.unescape(text)


def _paragraphs(text: str, limit: int = OPEN_CHARS) -> str:
    """Someone else's text kept as lines: no control or format characters,
    each line's spaces collapsed, runs of blank lines made one, bounded."""
    text = "".join(
        ch for ch in text
        if ch.isspace() or unicodedata.category(ch) not in ("Cc", "Cf")
    )
    out: list[str] = []
    for line in text.splitlines():
        line = " ".join(line.split())
        if line or (out and out[-1]):
            out.append(line)
    text = "\n".join(out).strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "\n… (truncated)"


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
    """One listed mail, ending in its ``#uid`` so the model can open it."""
    text = f"{_when(mail.date, now)} from {mail.sender or 'unknown sender'}: "
    text += mail.subject or "(no subject)"
    if mail.held:
        text += f" {HELD_MARK}"  # never the snippet (DL-082 #2)
    elif mail.snippet:
        text += f" — {mail.snippet}"
    if not mail.unread:
        text += " (read)"
    if tag_bulk and mail.bulk:
        text += " [bulk]"
    return f"{text} #{mail.uid}"


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
    on_mails: Optional[Callable[[Sequence[Mail]], None]] = None,
) -> list[str]:
    """What the look says about the inbox: nothing, a failure, or what is there.

    Recent mail read or not (DL-082 #6): mail they opened on their phone first
    is still part of their day, and is marked " (read)".

    ``on_mails`` is handed every mail this reading found, bulk included, and
    only when the reading succeeded — the finance sweep (DL-083) rides the
    look's one IMAP read rather than making its own. It never costs the look
    its lines: whatever it raises is dropped here.
    """
    s = settings(environ)
    if s is None:
        return []
    now = now or datetime.now(timezone.utc)
    try:
        mails = read(s, days=LOOK_DAYS, unread_only=False, connect=connect, now=now)
    except InboxUnavailable as exc:
        return [f"couldn't read the inbox: {exc}"]
    if on_mails is not None:
        try:
            on_mails(mails)
        except Exception as exc:  # noqa: BLE001 - a sweep never kills the look
            # ...but it is never silent either: a sweep that broke records
            # nothing, and only this line says so (DL-084 #4).
            print(f"omega: mail sweep failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    people = [m for m in mails if not m.bulk]
    bulk = len(mails) - len(people)
    out = lines(people, now) if people else [
        f"no mail from people in the last {LOOK_DAYS} days"
    ]
    if bulk:
        out.append(f"(and {bulk} newsletters/notifications, not listed)")
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
