"""Thinking time that thinks (C4, M2; DL-111, DL-118).

A thinking slot is one turn with read-only tools (``look_back``, ``projects``,
``calendar``, ``recall``; the brief's slot also ``inbox`` and ``schedules``)
and an explicit spend cap. Its shape is fixed: what changed, at most three
questions, insights that each cite the seqs they rest on, and a decision for
every open key it was shown: act, ask, fold or let go. It ends as one
composed message, or as a silence whose ``why`` names what it weighed (the
suppression log). Either way it writes one ``journal.written``, append-only.

Code, not the model, holds every line that matters:

* a cited seq stands only if the slot was shown it; any other is dropped and
  counted as ``fabricated``, and an insight left citing nothing is dropped;
* a key it was shown and did not decide about is folded, marked ``by: code``;
  a key it names that it was not shown is ignored;
* a tool outside the read-only set, or one that is not a read, is refused;
* an offer (a held ``delegate`` start for a CI key, ``handoff`` directions for
  a calendar key) is filed by the executor from structured fields only, for
  their yes; the model only names the key;
* where they roughly are is shown to it and never said (D8): a message that
  names it, when no open key it was shown does, is not sent.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

from omega import episodes, journal, ledger, provider
from omega.tools import (
    CALENDAR, EXPLORATION, INBOX, LOOK_BACK, PROJECTS, RECALL, SCHEDULES,
    ToolRejected, fence, hidden, untrusted_source,
)

#: The per-slot spend cap (DL-111), in tokens: about 15k [A], set by
#: ``OMEGA_SLOT_TOKENS``. Past half of it the slot gets one last call with no
#: tools; a slot that reaches it without an answer is written down as failed.
SLOT_TOKENS = 15_000
SLOT_TOKENS_ENV = "OMEGA_SLOT_TOKENS"
CAP_FAILED = "slot spend cap"
#: Read-only tools a thinking slot is offered, and the brief's two more.
READ_TOOLS = (LOOK_BACK, PROJECTS, CALENDAR, RECALL)
BRIEF_TOOLS = (*READ_TOOLS, INBOX, SCHEDULES)
MAX_PASSES = 4
#: Open keys one slot is shown, the least recently weighed first, so every
#: open key comes up for a decision within a day or two.
MAX_ITEMS = 16
MAX_OFFERS = 1
DECISIONS = ("act", "ask", "fold", "let_go")
#: Why an act with no lookup was folded (DL-121 #8).
NO_LOOKUP = "act with no lookup this slot, so folded"
_SILENT = ("SILENT", "NOTHING_NEW", "NOTHING NEW")
_JSON = re.compile(r"\{.*\}", re.S)

SYSTEM = (
    "This is your thinking time: nobody asked you anything. You are a person's "
    "second brain, looking over what is open in their life and deciding what, "
    "if anything, is worth one message now.\n"
    "Work in this order. What changed since your last journal entry. At most "
    "three questions worth asking yourself. Insights, each citing the seq "
    "numbers (shown as 'seq N' or '[journal N]') it rests on; cite nothing you "
    "were not shown. Then a decision for every open thread: act (worth telling "
    "or doing something about now), ask (worth a question to them), fold (keep "
    "it, nothing to say yet), let_go (it no longer needs them).\n"
    "An act needs a lookup: call a read-only tool on it this slot, or it is "
    "folded.\n"
    "You may look things up with the read-only tools offered. You cannot do "
    "anything outward; to offer help on a thread, list it under offers (a "
    "'delegate' for a failing CI, 'handoff' for directions to an event) and "
    "omega will ask them first.\n"
    "Where they roughly are is for your thinking only: never say it.\n"
    "Answer with one JSON object and nothing else:\n"
    '{"changed": "...", "questions": ["..."], '
    '"insights": [{"text": "...", "seqs": [12]}], '
    '"decisions": [{"key": "...", "decision": "act|ask|fold|let_go", "why": "..."}], '
    '"offers": [{"key": "...", "tool": "delegate|handoff"}], '
    '"message": "the one message to send them, or null", '
    '"silence_why": "if message is null: why nothing is worth saying"}'
)


class SlotCapExceeded(RuntimeError):
    """The slot reached its spend cap with no answer (DL-111)."""


@dataclass(frozen=True)
class Thought:
    text: Optional[str]
    tools: tuple[str, ...] = ()
    tainted: tuple[str, ...] = ()
    journal_seq: int = 0
    spent: int = 0
    #: The keys the message speaks for (C1): what it decided to act on or
    #: ask about, or offered help with; empty leaves the turn the keys its
    #: inbound carried (the items waiting to be composed).
    keys: tuple[str, ...] = ()


#: ``offer(item, tool, for_seq) -> ask text or None``, filed by the executor.
Offer = Callable[[ledger.Item, str, int], Optional[str]]


def slot_tokens(env: Mapping[str, str] = os.environ) -> int:
    raw = env.get(SLOT_TOKENS_ENV, "").strip()
    try:
        value = int(raw) if raw else SLOT_TOKENS
    except ValueError:
        return SLOT_TOKENS
    return value if value > 0 else SLOT_TOKENS


def shown_items(state: "ledger.State") -> list[ledger.Item]:
    """The open keys a slot decides about: not closed, not a reminder of
    theirs (it tells itself, DL-075), least recently weighed first."""
    open_ = [i for i in state.items.values() if not i.closed and not i.reminder]
    open_.sort(key=lambda i: (max(i.told_seq, i.considered_seq), i.last_seq))
    return open_[:MAX_ITEMS]


def _spent(response: provider.Response) -> int:
    # A provider that reports no counts cannot be capped per slot; the day's
    # budget still charges it (outbox.THINKING_DAILY_TOKENS).
    return int(response.prompt_tokens or 0) + int(response.completion_tokens or 0)


def _suffix(slot: str, items: Sequence[ledger.Item], entries: Sequence[journal.Entry],
            where: str) -> str:
    parts = [f"\n\nThinking slot: {slot}."]
    if items:
        lines = []
        for i in items:
            told = f"; last told {i.told_at[:16]}" if i.told_at else ""
            seen = f"; seen {i.recurrence} times" if i.recurrence > 1 else ""
            lines.append(f"- [{i.key}] {i.what} (seq {i.last_seq}{seen}{told})")
        block = "\n".join(lines)
        taint = ledger.taint(items)
        if taint:
            block = fence(taint[0], block)
        parts.append(f"Open threads, each needing a decision:\n{block}")
    else:
        parts.append("No open threads.")
    if entries:
        block = "\n\n".join(journal.render(e) for e in entries)
        if any(e.tainted for e in entries):
            block = fence(journal.SOURCE, block)
        parts.append(f"Your journal, latest last:\n{block}")
    if where:
        parts.append(f"Roughly where they are (for your thinking only; never say it): {where}")
    return "\n\n".join(parts)


def _parse(text: str) -> dict[str, Any]:
    raw = text.strip()
    match = _JSON.search(raw)
    if match:
        try:
            found = json.loads(match.group(0))
        except ValueError:
            found = None
        if isinstance(found, dict):
            return found
    if not raw or raw.upper().startswith(_SILENT):
        return {"message": None, "silence_why": raw.partition(" ")[2].strip()}
    return {"message": raw}


def _sentence(value: Any, cap: int) -> str:
    if not isinstance(value, str):
        return ""
    flat = " ".join(value.split())
    return flat if len(flat) <= cap else flat[: cap - 1].rstrip() + "…"


def think(
    ctx: Any,
    *,
    slot: str,
    items: Sequence[ledger.Item] = (),
    entries: Sequence[journal.Entry] = (),
    where: str = "",
    brief: bool = False,
    offer: Optional[Offer] = None,
    cap: Optional[int] = None,
    box: Any = None,
) -> Thought:
    """Run one thinking slot over ``ctx`` and write its journal entry."""
    from omega import act as acting
    from omega.turn import _event_turn, read_untrusted

    cap = cap or slot_tokens()
    box = box or acting.build_box(ctx)
    allowed = set(BRIEF_TOOLS if brief else READ_TOOLS)
    offered = [
        s for s in box.schemas(hidden(box))
        if s.get("function", {}).get("name") in allowed
    ]
    tainted: list[str] = list(read_untrusted(ctx))
    for t in (*ledger.taint(items), *(t for e in entries for t in e.tainted)):
        if t not in tainted:
            tainted.append(t)
    messages: list[provider.Message] = [
        provider.system(SYSTEM),
        _event_turn(ctx, images=False, lessons=True,
                    suffix=_suffix(slot, items, entries, where)),
    ]
    used: list[str] = []
    seen: set[tuple[str, str]] = set()
    spent = 0
    answer: Optional[provider.Response] = None
    for n in range(MAX_PASSES):
        last = n == MAX_PASSES - 1 or spent * 2 >= cap
        response = ctx.complete(provider.ACT, messages, None if last or not offered else offered)
        spent += _spent(response)
        if not response.tool_calls or last:
            answer = response
            break
        if spent >= cap:
            break
        messages.append(provider.assistant_tool_calls(response.tool_calls, response.text))
        for call in response.tool_calls:
            sig = (call.name, repr(sorted(call.arguments.items())))
            text = _run(ctx, box, call, allowed, sig in seen, used, tainted)
            seen.add(sig)
            messages.append(provider.tool_result(call.id, text))
    if answer is None:
        ctx.queue.append(episodes.thinking_failed(
            slot=slot, why=CAP_FAILED, at=str(ctx.event.get("at") or "") or None,
        ))
        raise SlotCapExceeded(f"thinking slot {slot} reached its {cap}-token cap")
    return _settle(ctx, slot=slot, parsed=_parse(answer.text), items=items, entries=entries,
                   where=where, brief=brief, offer=offer, used=used, tainted=tainted,
                   spent=spent)


def _run(ctx: Any, box: Any, call: provider.ToolCall, allowed: set, repeat: bool,
         used: list[str], tainted: list[str]) -> str:
    from omega import act as acting

    def refuse(why: str) -> str:
        acting._called(ctx, call.name, dict(call.arguments))
        acting._returned(ctx, call.name, ok=False, error=f"refused: {why}")
        return why

    if call.name not in allowed:
        return refuse(f"thinking time may only read: {sorted(allowed)}")
    if repeat:
        return refuse("asked again with the same arguments; it would say the same")
    try:
        decision = box.classify(call.name, call.arguments)
    except ToolRejected as exc:
        return refuse(str(exc))
    if decision.tier != EXPLORATION:
        return refuse(f"{call.name} would not be a read here")
    acting._called(ctx, decision.tool, decision.args)
    text, logged, ok = acting._dispatch(box, decision)
    acting._returned(ctx, decision.tool, ok=ok, result=logged if ok else None,
                     error=None if ok else text)
    source = untrusted_source(decision.tool, decision.args)
    if source is not None and source not in tainted:
        tainted.append(source)
    if ok:
        used.append(decision.tool)
        if source is not None:
            text = fence(source, text)
    return text


def _settle(ctx: Any, *, slot: str, parsed: dict[str, Any], items: Sequence[ledger.Item],
            entries: Sequence[journal.Entry], where: str, brief: bool,
            offer: Optional[Offer], used: list[str], tainted: list[str],
            spent: int) -> Thought:
    cap_text = episodes.MAX_JOURNAL_TEXT
    shown = {i.key: i for i in items}
    cited = {int(ctx.seq), *(i.last_seq for i in items), *(i.first_seq for i in items),
             *(e.seq for e in entries)}
    fabricated = 0
    insights = []
    for raw in parsed.get("insights") or ():
        if not isinstance(raw, dict):
            continue
        text = _sentence(raw.get("text"), cap_text)
        given = raw.get("seqs") if isinstance(raw.get("seqs"), list) else []
        kept = [s for s in dict.fromkeys(
            s for s in given if isinstance(s, int) and not isinstance(s, bool)
        ) if s in cited]
        fabricated += len(given) - len(kept)
        if text and kept and len(insights) < episodes.MAX_JOURNAL_INSIGHTS:
            insights.append({"text": text, "seqs": kept})
    questions = [q for q in (_sentence(x, cap_text) for x in parsed.get("questions") or ()) if q]
    questions = questions[: episodes.MAX_JOURNAL_QUESTIONS]
    said: dict[str, dict[str, Any]] = {}
    for raw in parsed.get("decisions") or ():
        if not isinstance(raw, dict):
            continue
        key, choice = raw.get("key"), raw.get("decision")
        if key in shown and key not in said and choice in DECISIONS:
            said[key] = {"key": key, "decision": choice, "by": "model",
                         "why": _sentence(raw.get("why"), episodes.MAX_OUTBOX_WHY) or choice}
    decisions = [
        said.get(k) or {"key": k, "decision": "fold", "by": "code",
                        "why": "no decision given; folded until next time"}
        for k in shown
    ]
    changed = _sentence(parsed.get("changed"), cap_text)
    message = parsed.get("message")
    message = message.strip() if isinstance(message, str) and message.strip() else None
    why = _sentence(parsed.get("silence_why"), 600)
    if message and where and _names(message, where) and not any(
        _names(i.what, where) for i in items
    ):
        message, why = None, "it named where they are, which is never volunteered (D8)"
    asks: list[str] = []
    offered_keys: list[str] = []
    if offer is not None:
        for raw in parsed.get("offers") or ():
            if len(asks) >= MAX_OFFERS:
                break
            if not isinstance(raw, dict) or raw.get("key") not in shown:
                continue
            ask = offer(shown[raw["key"]], str(raw.get("tool")), int(ctx.seq))
            if ask:
                asks.append(ask)
                offered_keys.append(raw["key"])
    # DL-121 #8: an act is a thread worth doing something about now, which
    # takes a look first. One with no lookup this slot (and no offer on its
    # key) is folded, and the journal says why.
    downgraded = 0
    for k, d in said.items():
        if d["decision"] == "act" and not used and k not in offered_keys:
            d["decision"] = "fold"
            d["why"] = _sentence(f"{NO_LOOKUP}: {d['why']}", episodes.MAX_OUTBOX_WHY)
            downgraded += 1
    if downgraded and message and not asks and not brief and not any(
        d["decision"] in ("act", "ask") for d in said.values()
    ):
        message, why = None, NO_LOOKUP
    if asks:
        message = "\n\n".join([*( [message] if message else []), *asks])
    if brief and message is None:
        # The brief is a promise (DL-080, DL-114): it says something, or the
        # turn fails and the brief stays owed (gap 0a).
        if not changed:
            raise RuntimeError("the brief's thinking slot said nothing")
        message = changed
    if message is None and not why:
        why = "nothing here is worth a message now"
    considered = list(shown)
    if message is None:
        named = ", ".join(considered) if considered else "no open threads"
        why = _sentence(f"{why} (weighed: {named})", cap_text)
    seq = ctx.queue.append(episodes.journal_written(
        for_seq=int(ctx.seq), slot=slot, changed=changed, questions=questions,
        insights=insights, decisions=decisions, considered=considered,
        outcome="message" if message else "silent", why="" if message else why,
        fabricated=fabricated, tainted=tuple(tainted),
        at=str(ctx.event.get("at") or "") or None,
    ))
    spoke_for = [d["key"] for d in decisions if d["decision"] in ("act", "ask")]
    spoke_for = list(dict.fromkeys((*spoke_for, *offered_keys)))
    return Thought(text=message, tools=tuple(used), tainted=tuple(tainted),
                   journal_seq=seq, spent=spent,
                   keys=tuple(spoke_for[: episodes.MAX_KEYS]) if message else ())


def _names(text: str, where: str) -> bool:
    return bool(where) and where.casefold() in text.casefold()
