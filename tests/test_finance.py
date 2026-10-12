"""Finance from transaction mail — DL-083.

Offline throughout: mail is built by hand or served by test_inbox's FakeIMAP,
and the ledger is a real log in a temporary store. What is checked is what
landed in the log and what the tool says from it, never a marker that the
sweep "ran".
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import pytest

from omega import episodes, finance, inbox, tools
from omega.queue import EventQueue
from omega.tools import EXPLORATION, ToolBox, ToolError, ToolRejected

from test_inbox import ENV, FakeIMAP, server

IST = timezone(timedelta(hours=5, minutes=30))
# Midday IST, so the local date is the same in IST and UTC; mail arrives an
# hour before the sweep that reads it.
NOW = datetime(2026, 10, 11, 12, 0, tzinfo=IST)
SENT = NOW - timedelta(hours=1)
BANK = "HDFC Bank InstaAlerts <alerts@hdfcbank.net>"


@dataclass(frozen=True)
class M:
    """Mail-shaped, with only what the sweep reads."""

    subject: str
    text: str
    sender: str = BANK
    date: Optional[datetime] = SENT
    message_id: str = ""
    held: bool = False
    bulk: bool = True
    snippet: str = ""


UPI = M(
    "You have done a UPI txn. Check details!",
    "Dear Customer, Rs.500.00 has been debited from account **1234 to VPA "
    "abc@ybl ABC STORE on 11-10-26. Your UPI transaction reference number is "
    "628312345678.",
    message_id="<upi@hdfc>",
)
CREDIT = M(
    "Credit alert",
    "INR 25,000.00 credited to your A/c XX9876 on 10-Oct-2026 from Suresh P. "
    "Available balance INR 1,02,345.00.",
    message_id="<cr@hdfc>",
)
CARD = M(
    "Card transaction alert",
    "Rs 1,299.00 spent on your Credit Card ending 4321 at AMAZON PAY INDIA "
    "on 2026-10-10 18:22:01.",
    message_id="<card@hdfc>",
)
USD = M(
    "Your receipt",
    "You were charged $12.99 for GitHub Copilot. Card 5555444433332222.",
    sender="GitHub <billing@github.com>",
    message_id="<usd@gh>",
)
REFUND = M(
    "Refund processed",
    "A refund of ₹349.00 for your order has been credited to your original "
    "payment method.",
    sender="Swiggy <noreply@swiggy.in>",
    message_id="<ref@swiggy>",
)


# --------------------------------------------------------------------------
# reading one mail
# --------------------------------------------------------------------------


def test_upi_debit() -> None:
    r = finance.parse(UPI)
    assert r is not None
    assert (r.amount, r.currency, r.direction) == ("500.00", "INR", "debit")
    assert r.counterparty == "abc@ybl"
    assert r.account == "1234"
    assert r.when == "2026-10-11"
    assert r.message_id == "<upi@hdfc>" and r.sender == BANK


def test_inr_credit_with_indian_grouping() -> None:
    r = finance.parse(CREDIT)
    assert r is not None
    assert (r.amount, r.currency, r.direction) == ("25000.00", "INR", "credit")
    assert r.counterparty == "Suresh P"
    assert r.account == "9876"
    assert r.when == "2026-10-10"


def test_card_spend() -> None:
    r = finance.parse(CARD)
    assert r is not None
    assert (r.amount, r.currency, r.direction) == ("1299.00", "INR", "debit")
    assert r.counterparty == "AMAZON PAY INDIA"
    assert r.account == "4321"
    assert r.when == "2026-10-10"


def test_usd_charge_falls_back_to_the_mail_date() -> None:
    r = finance.parse(USD)
    assert r is not None
    assert (r.amount, r.currency, r.direction) == ("12.99", "USD", "debit")
    assert r.when == SENT.astimezone().date().isoformat()


def test_refund_is_a_credit() -> None:
    r = finance.parse(REFUND)
    assert r is not None
    assert (r.amount, r.currency, r.direction) == ("349.00", "INR", "credit")
    assert r.counterparty == "Swiggy"  # no party in the text: the sender's name


@pytest.mark.parametrize("mail", [UPI, CREDIT, CARD, USD, REFUND])
def test_account_is_never_more_than_four_digits(mail: M) -> None:
    r = finance.parse(mail)
    assert r is not None
    assert r.account is None or (r.account.isdigit() and 1 <= len(r.account) <= 4)
    # and the long number itself is nowhere in what gets recorded
    assert "5555444433332222" not in repr(r)


@pytest.mark.parametrize(
    "mail",
    [
        M("Your OTP", "OTP 482913 for a transaction of Rs 2,000.00 at FLIPKART. "
                      "Do not share it. It will be debited only after you confirm."),
        M("Big Diwali sale!", "Save Rs 500 on your next order. Shop now. "
                              "This email was sent to you because you subscribed.",
          sender="Myntra <news@myntra.com>"),
        M("Statement ready", "Your balance is Rs 12,000.00 as of today."),
        M("Hello", "lunch tomorrow?", sender="Kaushik R <k@x.com>"),
    ],
    ids=["otp", "newsletter", "balance-no-direction", "no-amount"],
)
def test_not_a_transaction(mail: M) -> None:
    assert not finance.looks_like_transaction(mail.subject, mail.text)
    assert finance.parse(mail) is None


def test_both_directions_is_no_record_but_still_looks_like_one() -> None:
    mail = M("Alert", "Rs 200.00 debited and Rs 200.00 credited back to your account.")
    assert finance.looks_like_transaction(mail.subject, mail.text)
    assert finance.parse(mail) is None


def test_held_mail_is_never_parsed() -> None:
    held = M(UPI.subject, UPI.text, held=True, message_id="<h@x>")
    assert finance.parse(held) is None
    found, unread = finance.sweep([held], [_started(NOW - timedelta(days=1))], NOW)
    assert found == [] and unread == 0


# --------------------------------------------------------------------------
# the sweep
# --------------------------------------------------------------------------


def _started(at: datetime) -> dict[str, Any]:
    return episodes.finance_started(at=at.isoformat())


def test_first_sweep_starts_tracking_and_backfills_nothing() -> None:
    found, unread = finance.sweep([UPI, CREDIT], [], NOW)
    assert [p["kind"] for p in found] == [episodes.FINANCE_STARTED]
    assert found[0]["at"] == NOW.isoformat() and unread == 0


def test_mail_before_the_start_is_ignored_and_start_is_written_once() -> None:
    start = _started(SENT - timedelta(minutes=1))
    before = M(UPI.subject, UPI.text, date=SENT - timedelta(hours=1), message_id="<old@x>")
    found, _ = finance.sweep([before, CARD], [start], NOW)
    assert [p["kind"] for p in found] == [episodes.FINANCE_RECORDED]
    assert found[0]["message_id"] == "<card@hdfc>"


def test_undated_mail_is_ignored() -> None:
    undated = M(UPI.subject, UPI.text, date=None, message_id="<nodate@x>")
    assert finance.sweep([undated], [_started(NOW - timedelta(days=1))], NOW) == ([], 0)


def test_dedupe_across_sweeps_by_message_id() -> None:
    log = [_started(NOW - timedelta(days=1))]
    ambiguous = M("Alert", "Rs 200.00 debited and Rs 200.00 credited back.",
                  message_id="<amb@x>")
    first, unread = finance.sweep([UPI, CREDIT, ambiguous], log, NOW)
    assert [p["kind"] for p in first] == [
        episodes.FINANCE_RECORDED, episodes.FINANCE_RECORDED, episodes.FINANCE_SKIPPED
    ]
    assert unread == 1
    log += first
    again, unread = finance.sweep([UPI, CREDIT, ambiguous], log, NOW + timedelta(hours=1))
    assert again == [] and unread == 0


def test_dedupe_without_message_id_by_digest() -> None:
    log = [_started(NOW - timedelta(days=1))]
    bare = M(UPI.subject, UPI.text, message_id="")
    first, _ = finance.sweep([bare], log, NOW)
    assert len(first) == 1 and first[0]["key"].startswith("sha256:")
    assert first[0]["key"] == finance.key_for(bare)
    assert finance.sweep([bare], log + first, NOW) == ([], 0)
    other = M(UPI.subject, UPI.text, message_id="", date=SENT + timedelta(minutes=1))
    assert finance.key_for(other) != finance.key_for(bare)


def test_bulk_is_not_excluded() -> None:
    found, _ = finance.sweep([M(CARD.subject, CARD.text, bulk=True, message_id="<b@x>")],
                             [_started(NOW - timedelta(days=1))], NOW)
    assert [p["kind"] for p in found] == [episodes.FINANCE_RECORDED]


def test_every_episode_the_sweep_makes_validates() -> None:
    found, _ = finance.sweep([UPI, CREDIT, CARD, USD, REFUND], [], NOW - timedelta(days=1))
    found2, _ = finance.sweep([UPI, CREDIT, CARD, USD, REFUND], found, NOW)
    for p in found + found2:
        assert episodes.decode(episodes.encode(p)) == p


def test_record_appends_to_a_real_log_and_ledger_reads_it_back(store) -> None:
    q = EventQueue(store)
    assert finance.record(q, [], NOW - timedelta(days=1)) == 1  # the start only
    assert finance.record(q, [UPI, CARD], NOW) == 2
    assert finance.record(q, [UPI, CARD], NOW) == 0
    kinds = [p["kind"] for p in finance.ledger(store)]
    assert kinds == [episodes.FINANCE_STARTED] + [episodes.FINANCE_RECORDED] * 2


# --------------------------------------------------------------------------
# riding the look
# --------------------------------------------------------------------------


def test_look_hands_every_mail_with_message_id_and_text_to_the_hook() -> None:
    connect = server({
        1: {"msg": {"sender": BANK, "subject": UPI.subject,
                    "date": "Sun, 11 Oct 2026 11:00:00 +0530",
                    "extra": "Message-ID: <upi@hdfc>\r\nList-Unsubscribe: <mailto:u@x>\r\n",
                    "body": UPI.text.encode()}},
        2: {"msg": {}},
    })
    seen: list[Any] = []
    lines = inbox.look_lines(NOW, environ=ENV, connect=connect, on_mails=seen.extend)
    assert lines  # the look still says what it says
    by_id = {m.message_id: m for m in seen}
    assert set(by_id) == {"<upi@hdfc>", ""}
    bank = by_id["<upi@hdfc>"]
    assert bank.bulk and "Rs.500.00 has been debited" in bank.text
    assert len(bank.text) <= inbox.TEXT_CHARS
    mutating = {"STORE", "COPY", "MOVE", "EXPUNGE", "APPEND", "DELETE"}
    assert not [c for f in FakeIMAP.instances for c in f.calls
                if c[0] in mutating or (c[0] == "UID" and c[1] in mutating)]


def test_a_failing_hook_never_costs_the_look_its_lines(capsys: Any) -> None:
    def boom(_: Any) -> None:
        raise RuntimeError("sweep broke")

    connect = server({2: {"msg": {}}})
    quiet = inbox.look_lines(NOW, environ=ENV, connect=connect)
    connect = server({2: {"msg": {}}})
    assert inbox.look_lines(NOW, environ=ENV, connect=connect, on_mails=boom) == quiet
    # Kept out of the look, never out of sight (DL-084 #4).
    assert "mail sweep failed: RuntimeError: sweep broke" in capsys.readouterr().err


def test_hook_not_called_when_the_read_fails() -> None:
    import socket

    called: list[Any] = []
    connect = server({}, fail={"login": socket.timeout()})
    lines = inbox.look_lines(NOW, environ=ENV, connect=connect, on_mails=called.append)
    assert lines and lines[0].startswith("couldn't read the inbox") and called == []


# --------------------------------------------------------------------------
# the tool
# --------------------------------------------------------------------------


def _ledger() -> list[dict[str, Any]]:
    log = [_started(NOW - timedelta(days=3))]
    ambiguous = M("Alert", "Rs 200.00 debited and Rs 200.00 credited back.",
                  message_id="<amb@x>")
    swiggy = M("Order paid", "Rs 450.00 paid to Swiggy on 11-10-2026.",
               sender="Swiggy <noreply@swiggy.in>", message_id="<sw1@x>")
    swiggy2 = M("Order paid", "Rs 150.00 paid to Swiggy on 11-10-2026.",
                sender="Swiggy <noreply@swiggy.in>", message_id="<sw2@x>")
    found, _ = finance.sweep(
        [UPI, CREDIT, CARD, USD, REFUND, ambiguous, swiggy, swiggy2], log, NOW
    )
    return log + found


def test_tool_text_totals_top_counterparties_and_unread_count() -> None:
    text = finance.tool_text(_ledger(), days=7, now=NOW)
    lines = text.splitlines()
    assert lines[0].startswith("Transactions read from their mail in the last 7 days")
    assert "Total INR: spent 2399.00, received 25349.00" in lines
    assert "Total USD: spent 12.99, received 0.00" in lines
    top = lines[lines.index("Most spent with:") + 1:lines.index("Records:")]
    assert top[0] == "- AMAZON PAY INDIA: 1299.00 INR"
    assert "- Swiggy: 600.00 INR" in top and "- abc@ybl: 500.00 INR" in top
    assert len(top) <= finance.TOP_COUNTERPARTIES
    assert "- 2026-10-11 -500.00 INR abc@ybl ••1234" in lines
    assert "- 2026-10-10 +25000.00 INR Suresh P ••9876" in lines
    assert lines[-1] == ("1 transaction-looking mail I couldn't read the amount of"
                         " (not in the totals above)")


def test_tool_text_window_excludes_older_records() -> None:
    text = finance.tool_text(_ledger(), days=1, now=NOW + timedelta(days=2))
    assert "- none recorded" in text and "Total" not in text


def test_tool_text_not_started_and_empty() -> None:
    assert finance.tool_text([], days=7, now=NOW) == finance.NOT_STARTED
    assert finance.NOT_STARTED == (
        "finance tracking starts with the next look; nothing recorded yet"
    )
    empty = finance.tool_text([_started(NOW)], days=7, now=NOW)
    assert empty.splitlines()[1:] == ["- none recorded"]


def test_tool_through_the_box(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The ledger is dated around NOW; the tool's window must be too.
    monkeypatch.setattr(tools, "_now", lambda: NOW)
    box = ToolBox(store_root=tmp_path, ledger=_ledger)
    decision = box.classify("finance", {})
    assert decision.tier == EXPLORATION and decision.args == {"days": 7}
    text = box.dispatch(decision)
    assert "Total INR: spent 2399.00" in text and "couldn't read the amount of" in text
    assert box.dispatch(box.classify("finance", {"days": 90})).startswith("Transactions")
    nothing = ToolBox(store_root=tmp_path, ledger=lambda: [])
    assert nothing.dispatch(nothing.classify("finance", {})) == finance.NOT_STARTED


def test_tool_without_a_ledger_is_a_tool_error(tmp_path) -> None:
    box = ToolBox(store_root=tmp_path)
    with pytest.raises(ToolError):
        box.dispatch(box.classify("finance", {}))


@pytest.mark.parametrize("args", [{"days": 91}, {"days": -1}, {"days": True},
                                  {"days": "7"}, {"days": [7]}])
def test_tool_rejects_bad_days(tmp_path, args: dict) -> None:
    with pytest.raises(ToolRejected):
        ToolBox(store_root=tmp_path, ledger=_ledger).classify("finance", args)


@pytest.mark.parametrize("given, days", [({"days": 0}, 7), ({"days": None}, 7),
                                         ({"days": 1}, 1), ({"days": 90}, 90)])
def test_tool_days_follow_the_house_rule(tmp_path, given: dict, days: int) -> None:
    # 0 reads as blank, as for every other `days` (a model fills a blank with 0).
    box = ToolBox(store_root=tmp_path, ledger=_ledger)
    assert box.classify("finance", given).args == {"days": days}


def test_tool_output_is_capped(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(finance, "tool_text", lambda found, **kw: "x" * 20_000)
    box = ToolBox(store_root=tmp_path, ledger=lambda: [])
    text = box.dispatch(box.classify("finance", {}))
    assert len(text) < 9_000 and text.endswith("[truncated at 8000 characters]")


def test_finance_is_a_tool() -> None:
    assert tools.FINANCE == "finance" and tools.FINANCE in tools.TOOL_NAMES


def test_a_card_alert_with_no_verb_of_its_own_is_still_a_debit():
    # The common Indian shape that slipped through first: "using your card ...
    # for Rs ... at ..." spends without saying spent (DL-083 #5).
    subject = "Alert : Update on your HDFC Bank Credit Card"
    body = "Thank you for using your Credit Card ending 4321 for Rs 1,299.00 at AMAZON on 11-10-2026."
    assert finance.looks_like_transaction(subject, body)


def test_the_card_alert_parses_to_a_whole_record():
    from types import SimpleNamespace
    from datetime import datetime, timezone
    mail = SimpleNamespace(
        sender="HDFC Bank <alerts@hdfcbank.net>",
        subject="Alert : Update on your HDFC Bank Credit Card",
        date=datetime(2026, 10, 11, 9, 0, tzinfo=timezone.utc),
        message_id="<card-1@hdfc>",
        held=False,
        text="Thank you for using your Credit Card ending 4321 for Rs 1,299.00 at AMAZON on 11-10-2026.",
        snippet="",
    )
    record = finance.parse(mail)
    assert record is not None
    assert (record.amount, record.currency, record.direction, record.account) == ("1299.00", "INR", "debit", "4321")
    assert "AMAZON" in record.counterparty
