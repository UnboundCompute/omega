"""Finance from transaction mail — read, recorded, never acted on. DL-083.

The person wants omega to know how their money is going, from what their bank
already mails them, starting now. This module is that and nothing more: a
deterministic parser that turns a bank or card alert into a record, a sweep
that rides the look's one reading of the inbox and appends those records to
the log, and the wording the ``finance`` tool answers with.

**The log is the ledger.** A ``finance.started`` marks the instant tracking
began; every transaction read is one ``finance.recorded``; every mail that
looked like a transaction but would not parse is one ``finance.skipped``.
Nothing is written beside the log, and the tool reads only the log — "what did
I spend this week" is answered from what was recorded, not from IMAP.

**From now on, never backfilled.** Mail dated before ``finance.started`` is
never recorded, by the person's own word ("no need for backfill").

**No model, and fail closed.** A model inventing an amount is worse than a
missing one, so extraction is regular expressions over the mail's own text. No
amount, or no single direction (debited *and* credited, or neither), and there
is no record — but the mail is counted, so the tool never hands over a
confident total built on silently dropped mail.

**Bulk is not excluded; held is.** Bank alerts routinely carry
``List-Unsubscribe`` and so read as bulk to :mod:`omega.inbox`, which is
exactly the mail this exists for. Held mail (DL-082) is never parsed: it opens
only when they ask, and a sweep is nobody asking.

**No action, ever.** It reads mail the look already read and appends to the
log. It never pays, replies, flags, moves or marks anything.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from email.utils import parseaddr
from typing import Any, Iterable, Optional, Sequence

from omega import episodes

__all__ = [
    "KINDS",
    "DEFAULT_DAYS",
    "MAX_DAYS",
    "TOP_COUNTERPARTIES",
    "NOT_STARTED",
    "Record",
    "looks_like_transaction",
    "parse",
    "key_for",
    "sweep",
    "ledger",
    "record",
    "tool_text",
]

#: The episode kinds this module reads and writes.
KINDS = frozenset(
    {episodes.FINANCE_STARTED, episodes.FINANCE_RECORDED, episodes.FINANCE_SKIPPED}
)

DEFAULT_DAYS = 7
MAX_DAYS = 90
TOP_COUNTERPARTIES = 5
COUNTERPARTY_CHARS = 60

NOT_STARTED = "finance tracking starts with the next look; nothing recorded yet"


@dataclass(frozen=True)
class Record:
    amount: str            # a Decimal as text, two places, no commas
    currency: str          # INR, USD, EUR, GBP, else the code as written
    direction: str         # "debit" | "credit"
    counterparty: str      # best effort, else the sender's name
    account: Optional[str]  # at most the last four digits; None when absent
    when: str              # ISO date: stated in the mail if parsed, else its Date
    sender: str
    subject: str
    message_id: str


# --------------------------------------------------------------------------
# reading one mail
# --------------------------------------------------------------------------

_SYMBOLS = {"₹": "INR", "rs": "INR", "rs.": "INR", "$": "USD", "€": "EUR", "£": "GBP"}

#: Codes taken as currency when written beside a number. A closed list, so
#: "UPI 6283..." or "REF 1234" is never read as money.
_CODES = (
    "INR", "USD", "EUR", "GBP", "AED", "SGD", "AUD", "CAD", "JPY", "CHF",
    "CNY", "HKD", "NZD", "SAR", "QAR", "THB", "MYR", "LKR", "NPR",
)
_CODE = "|".join(_CODES)
_NUMBER = r"\d{1,3}(?:,\d{2,3})+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?"
_AMOUNT = re.compile(
    rf"(?P<cur>₹|(?<![A-Za-z])Rs\.?|\$|€|£|(?<![A-Za-z])(?:{_CODE})(?![A-Za-z]))\s*"
    rf"(?P<num>{_NUMBER})(?![\d,])"
    rf"|(?P<num2>{_NUMBER})\s*(?P<cur2>(?<![A-Za-z])(?:{_CODE})(?![A-Za-z]))",
    re.IGNORECASE,
)

_DEBIT = re.compile(
    # "for using your Credit Card ending 4321 for Rs 1,299" is how a common
    # Indian card alert spends, with no verb of its own; missed, it would be
    # dropped without even counting as unread (DL-083 #5).
    r"\b(?:debited|spent|paid|sent|withdrawn|purchase[sd]?|charged"
    r"|using\s+your\s+(?:\w+\s+){0,2}card)\b", re.IGNORECASE
)
_CREDIT = re.compile(r"\b(?:credited|received|refund(?:ed)?|deposited)\b", re.IGNORECASE)

#: A one-time code is not a transaction, even one naming an amount it would
#: authorise: the payment has not happened.
_OTP = re.compile(
    r"\bOTP\b|one[- ]time (?:password|passcode|code)|verification code", re.IGNORECASE
)

#: How far around an amount its direction is looked for. A footer saying
#: "this email was sent to you" must not decide what a transaction was.
_BEFORE, _AFTER = 120, 200

_ACCOUNT = re.compile(
    r"(?i)\b(?:a/c|acct|account|card)"
    r"(?:\s*(?:no\.?|number|ending(?:\s+(?:in|with))?))?"
    r"\s*[:\-]?\s*[x*•.]*\s*(\d{2,})"
)

_STOP = r"(?=\s+(?:on|via|using|with|for|ref|txn|upi|dated|is|has|from|at)\b|[.,;:()\n]|\s*$)"
_NAME = r"([A-Za-z][A-Za-z0-9&.'@\-_ ]{0,80}?)"
_VPA = re.compile(r"(?i)\b(?:to|from|by)\s+VPA\s+([\w.\-]+@[\w.\-]+)")
_DEBIT_PARTY = (
    re.compile(rf"\bat\s+(?![Yy]our\b){_NAME}{_STOP}"),
    re.compile(rf"\btowards\s+(?![Yy]our\b){_NAME}{_STOP}"),
    re.compile(rf"\bto\s+(?![Yy]our\b|VPA\b|you\b|the\b|be\b)([A-Z][A-Za-z0-9&.'@\-_ ]{{0,80}}?){_STOP}"),
)
_CREDIT_PARTY = (
    re.compile(rf"\bfrom\s+(?![Yy]our\b|account\b|a/c\b|A/c\b|card\b){_NAME}{_STOP}"),
)

_MONTHS = ("jan", "feb", "mar", "apr", "may", "jun",
           "jul", "aug", "sep", "oct", "nov", "dec")
_DMY = re.compile(r"\b(\d{1,2})[-/.](\d{1,2})[-/.](\d{4}|\d{2})\b")
_DMONY = re.compile(
    r"(?i)\b(\d{1,2})[- ]?(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[-, ]\s*(\d{4}|\d{2})\b"
)
_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")

#: A stated date is believed only this close to the mail's own Date: an alert
#: is written days after a transaction at most, never weeks, never ahead.
_DATE_SLACK_BEFORE = timedelta(days=31)
_DATE_SLACK_AFTER = timedelta(days=1)


def _text(subject: str, body: str) -> str:
    return f"{subject or ''}\n{body or ''}"


def _amounts(text: str) -> Iterable[tuple[re.Match[str], str, str]]:
    for m in _AMOUNT.finditer(text):
        cur = m.group("cur") or m.group("cur2")
        num = m.group("num") or m.group("num2")
        yield m, cur, num


#: Mail boilerplate that would otherwise read as "sent": a footer saying the
#: mail itself was sent to them is not money moving.
_BOILERPLATE = re.compile(
    r"(?i)\b(?:this|the)\s+(?:e-?mail|message|mail|alert)\s+(?:was|is|has\s+been)\s+sent\b"
    r"|\bsent\s+(?:to\s+)?you\b"
)


def _directions(window: str) -> set[str]:
    window = _BOILERPLATE.sub(" ", window)
    found = set()
    if _DEBIT.search(window):
        found.add("debit")
    if _CREDIT.search(window):
        found.add("credit")
    return found


def _first_directed(subject: str, body: str) -> Optional[tuple[re.Match[str], str, str, set[str]]]:
    """The first amount with direction wording near it (or in the subject)."""
    text = _text(subject, body)
    in_subject = _directions(subject or "")
    for m, cur, num in _amounts(text):
        window = text[max(0, m.start() - _BEFORE): m.end() + _AFTER]
        found = _directions(window) | in_subject
        if found:
            return m, cur, num, found
    return None


def looks_like_transaction(subject: str, body: str) -> bool:
    """An amount with debited/credited/spent/paid/received/refund wording near
    it, and not a one-time code. Says nothing about whether it will parse."""
    if _OTP.search(_text(subject, body)):
        return False
    return _first_directed(subject, body) is not None


def _currency(token: str) -> str:
    return _SYMBOLS.get(token.lower(), token.upper())


def _amount(num: str) -> Optional[str]:
    try:
        value = Decimal(num.replace(",", ""))
    except InvalidOperation:
        return None
    if not value.is_finite() or value <= 0:
        return None
    return f"{value:.2f}"


def _account(text: str) -> Optional[str]:
    m = _ACCOUNT.search(text)
    return m.group(1)[-4:] if m else None  # never more than the last four


def _trim(name: str) -> str:
    name = " ".join(name.split()).strip(" .,-")
    return name[:COUNTERPARTY_CHARS].rstrip()


def _sender_name(sender: str) -> str:
    name, address = parseaddr(sender or "")
    return _trim(name or address or sender or "")


def _counterparty(text: str, direction: str, sender: str) -> str:
    """Who the money went to or came from, read just after the amount — the
    earliest phrase that names one wins, so a footer's "call us at" never
    does. Else the sender's name."""
    text = text[:_AFTER]
    m = _VPA.search(text)
    if m:
        return _trim(m.group(1))
    hits = [
        m for pattern in (_DEBIT_PARTY if direction == "debit" else _CREDIT_PARTY)
        for m in [pattern.search(text)] if m and _trim(m.group(1))
    ]
    if hits:
        return _trim(min(hits, key=lambda m: m.start()).group(1))
    return _sender_name(sender)


def _stated(text: str, mailed: Optional[date]) -> Optional[date]:
    """The first date written in the mail that sits believably near its Date."""
    found: list[tuple[int, date]] = []
    for m in _ISO.finditer(text):
        found += _candidate(m.start(), int(m.group(1)), int(m.group(2)), int(m.group(3)))
    for m in _DMY.finditer(text):
        year = int(m.group(3))
        found += _candidate(m.start(), year + 2000 if year < 100 else year,
                            int(m.group(2)), int(m.group(1)))
    for m in _DMONY.finditer(text):
        year = int(m.group(3))
        found += _candidate(m.start(), year + 2000 if year < 100 else year,
                            _MONTHS.index(m.group(2).lower()) + 1, int(m.group(1)))
    for _, day in sorted(found, key=lambda pair: pair[0]):
        if mailed is None or (
            mailed - _DATE_SLACK_BEFORE <= day <= mailed + _DATE_SLACK_AFTER
        ):
            return day
    return None


def _candidate(at: int, year: int, month: int, day: int) -> list[tuple[int, date]]:
    try:
        return [(at, date(year, month, day))]
    except ValueError:
        return []


def parse(mail: Any) -> Optional[Record]:
    """One mail to one record, or ``None`` — never a guess.

    ``mail`` is anything with :class:`omega.inbox.Mail`'s fields: ``sender``,
    ``subject``, ``date``, ``message_id``, ``held`` and the body as ``text``
    (``snippet`` when there is no ``text``). Held mail is never parsed.
    """
    if getattr(mail, "held", False):
        return None
    subject = getattr(mail, "subject", "") or ""
    body = getattr(mail, "text", "") or getattr(mail, "snippet", "") or ""
    if not looks_like_transaction(subject, body):
        return None
    found = _first_directed(subject, body)
    if found is None:
        return None
    m, cur, num, directions = found
    if len(directions) != 1:
        return None  # both or neither: ambiguous, so no record
    amount = _amount(num)
    if amount is None:
        return None
    direction = next(iter(directions))
    text = _text(subject, body)
    after = text[m.start():]
    sent: Optional[datetime] = getattr(mail, "date", None)
    mailed = sent.astimezone().date() if sent is not None else None
    when = _stated(after, mailed) or _stated(text, mailed) or mailed
    if when is None:
        return None
    sender = getattr(mail, "sender", "") or ""
    return Record(
        amount=amount,
        currency=_currency(cur),
        direction=direction,
        counterparty=_counterparty(after, direction, sender) or _sender_name(sender),
        account=_account(text),
        when=when.isoformat(),
        sender=sender,
        subject=subject,
        message_id=getattr(mail, "message_id", "") or "",
    )


# --------------------------------------------------------------------------
# the sweep
# --------------------------------------------------------------------------


def key_for(mail: Any) -> str:
    """What dedupes a mail (DL-083 #4): its Message-ID, else a digest of
    sender, Date and subject."""
    message_id = (getattr(mail, "message_id", "") or "").strip()
    if message_id:
        return message_id
    sent = getattr(mail, "date", None)
    raw = "|".join([
        getattr(mail, "sender", "") or "",
        sent.isoformat() if sent is not None else "",
        getattr(mail, "subject", "") or "",
    ])
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def sweep(
    mails: Sequence[Any], existing: Iterable[dict[str, Any]], now: datetime
) -> tuple[list[dict[str, Any]], int]:
    """The episodes one look's mail adds to the ledger, and how many of them
    are mail that looked like a transaction but would not parse.

    ``existing`` is the finance episodes already in the log. With no
    ``finance.started`` among them this sweep starts tracking at ``now``, and
    — the look having read the last two days — records nothing it read, which
    is the no-backfill rule doing its job rather than a gap.
    """
    started: Optional[datetime] = None
    seen: set[str] = set()
    for payload in existing:
        kind = payload.get("kind")
        if kind == episodes.FINANCE_STARTED and started is None:
            started = datetime.fromisoformat(str(payload["at"]))
        elif kind in (episodes.FINANCE_RECORDED, episodes.FINANCE_SKIPPED):
            seen.add(str(payload["key"]))

    out: list[dict[str, Any]] = []
    at = now.isoformat()
    if started is None:
        started = now
        out.append(episodes.finance_started(at=at))

    skipped = 0
    for mail in sorted(mails, key=lambda m: m.date or now):
        if getattr(mail, "held", False):
            continue
        sent = getattr(mail, "date", None)
        if sent is None or sent < started:
            continue  # undated mail cannot be shown to be after the start
        key = key_for(mail)
        if key in seen:
            continue
        subject = getattr(mail, "subject", "") or ""
        body = getattr(mail, "text", "") or getattr(mail, "snippet", "") or ""
        if not looks_like_transaction(subject, body):
            continue
        seen.add(key)
        rec = parse(mail)
        if rec is None:
            skipped += 1
            out.append(episodes.finance_skipped(
                key=key,
                sender=getattr(mail, "sender", "") or "",
                subject=subject,
                when=sent.astimezone().date().isoformat(),
                message_id=getattr(mail, "message_id", "") or "",
                at=at,
            ))
            continue
        out.append(episodes.finance_recorded(
            key=key,
            amount=rec.amount,
            currency=rec.currency,
            direction=rec.direction,
            counterparty=rec.counterparty,
            account=rec.account,
            when=rec.when,
            sender=rec.sender,
            subject=rec.subject,
            message_id=rec.message_id,
            at=at,
        ))
    return out, skipped


def ledger(store: Any) -> list[dict[str, Any]]:
    """Every finance episode in the log, in log order.

    A full scan, as :func:`omega.notice.standing` makes once a look: the
    ledger has no other home, and a cursor would be a second one.
    """
    out = []
    for episode in store.episodes_since(0):
        payload = episodes.decode(episode.payload)
        if payload.get("kind") in KINDS:
            out.append(payload)
    return out


def record(queue: Any, mails: Sequence[Any], now: datetime) -> int:
    """Run the sweep against the log and append what it found. Returns the
    number of episodes appended."""
    found, _ = sweep(mails, ledger(queue.store), now)
    for payload in found:
        queue.append(payload)
    return len(found)


# --------------------------------------------------------------------------
# the tool's answer
# --------------------------------------------------------------------------


def tool_text(found: Sequence[dict[str, Any]], *, days: int, now: datetime) -> str:
    """What the ``finance`` tool says about the last ``days`` days."""
    start = next(
        (p for p in found if p.get("kind") == episodes.FINANCE_STARTED), None
    )
    if start is None:
        return NOT_STARTED
    began = datetime.fromisoformat(str(start["at"])).astimezone().date()
    since = (now.astimezone() - timedelta(days=days)).date()
    span = "day" if days == 1 else f"{days} days"

    def within(p: dict[str, Any]) -> bool:
        return date.fromisoformat(str(p["when"])) >= since

    records = sorted(
        (p for p in found if p.get("kind") == episodes.FINANCE_RECORDED and within(p)),
        key=lambda p: str(p["when"]),
    )
    unread = sum(
        1 for p in found if p.get("kind") == episodes.FINANCE_SKIPPED and within(p)
    )

    lines = [f"Transactions read from their mail in the last {span}"
             f" (tracking began {began.isoformat()}; nothing before is recorded):"]
    if not records:
        lines.append("- none recorded")
    else:
        spent: dict[str, Decimal] = {}
        received: dict[str, Decimal] = {}
        by_party: dict[tuple[str, str], Decimal] = {}
        for p in records:
            value = Decimal(str(p["amount"]))
            cur = str(p["currency"])
            if p["direction"] == "debit":
                spent[cur] = spent.get(cur, Decimal(0)) + value
                party = (str(p["counterparty"]) or "unknown", cur)
                by_party[party] = by_party.get(party, Decimal(0)) + value
            else:
                received[cur] = received.get(cur, Decimal(0)) + value
        for cur in sorted(set(spent) | set(received)):
            lines.append(
                f"Total {cur}: spent {spent.get(cur, Decimal(0)):.2f}, "
                f"received {received.get(cur, Decimal(0)):.2f}"
            )
        top = sorted(by_party.items(), key=lambda kv: (-kv[1], kv[0]))[:TOP_COUNTERPARTIES]
        if top:
            lines.append("Most spent with:")
            lines += [f"- {party}: {value:.2f} {cur}" for (party, cur), value in top]
        lines.append("Records:")
        for p in records:
            sign = "-" if p["direction"] == "debit" else "+"
            line = f"- {p['when']} {sign}{p['amount']} {p['currency']} {p['counterparty']}"
            if p.get("account"):
                line += f" ••{p['account']}"
            lines.append(line)
    if unread:
        noun = "mail" if unread == 1 else "mails"
        lines.append(
            f"{unread} transaction-looking {noun} I couldn't read the amount of"
            " (not in the totals above)"
        )
    return "\n".join(lines)
