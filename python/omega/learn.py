"""Turning a teaching note into claims — DL-043, the third leg of DL-034.

DL-042 settled what a learned claim *is*: a record with a trigger, filed in the
log, folded into a view, rendered into the prompt when it fires. This is where
one gets written.

Three things here are load-bearing.

**The gate is a sentence, and that is on purpose.** A teach drop carries no
marker. The tray wraps the note in an instruction and sends an ordinary inbound
episode, because DL-034 deliberately made a teach indistinguishable *in kind* so
the tray would not have to know anything about learning. So the only thing this
module can key on is the instruction itself — :data:`TEACH_MARKER`, a fragment
the tray's own test already pins. Two tests in two languages hold one string,
and a reword on either side fails on that side first.

**The receipt renders what was written, never what the model said it wrote.**
The tray's instruction ends *"Briefly confirm what you learned"*, and a model
will write that confirmation just as fluently when extraction returned nothing,
failed, or never ran. That is precisely the done-marker `CLAUDE.md` bans, so
:func:`receipt` is built from the claims that were actually appended and is
three-valued in the way *fail closed on empty* requires: here is what I
recorded / I recorded nothing / I could not record it.

**Extraction is all-or-nothing.** A note that yields three claims of which one
fails to validate does not file two. Partial success is worse than either
outcome: the receipt would list two claims and say nothing about the third, and
the person would confirm a record that is quietly short. The whole extraction
fails and the receipt says so.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Optional, Sequence

from omega import episodes, provider
from omega.derive import Claim

__all__ = [
    "TEACH_MARKER",
    "MAX_CLAIMS_PER_NOTE",
    "MAX_CLAIM_CHARS",
    "NotExtracted",
    "teaching_note",
    "extract",
    "parse_claims",
    "file_claims",
    "receipt",
    "when_phrase",
]

#: The fragment of the tray's teaching instruction that marks a teach drop.
#:
#: Pinned on the Swift side by ``TrayModelTests.swift:28`` and on this side by
#: this module's tests. It is a contract between two languages with no shared
#: constant to hold it, which is the cost of DL-034 having made a teach drop an
#: ordinary episode; the mitigation is that both halves are tested, so a reword
#: cannot pass silently on either.
TEACH_MARKER = "remember and apply in future conversations"

#: More claims than this from one note is treated as a failed extraction rather
#: than as a productive teach. One note is one thought; a model returning a
#: dozen claims has decomposed rather than understood, and every one of them
#: would go on to compete for the prompt budget on every future turn.
MAX_CLAIMS_PER_NOTE = 8

#: A claim longer than this is refused for the same reason: the learned section
#: is rendered on turns that have nothing to do with it, so a claim that is
#: really a paragraph is a permanent tax on every prompt.
MAX_CLAIM_CHARS = 300


class NotExtracted(RuntimeError):
    """The model's answer was not a usable claim list.

    Deliberately does not distinguish *unparseable* from *invalid* from *too
    many*: nothing reacts differently to those, and the receipt says the same
    thing to the person in all three cases. What it does carry is a reason
    short enough to put in front of them.
    """


# --- the gate ---------------------------------------------------------------


def teaching_note(text: str) -> Optional[str]:
    """The note the person typed, or ``None`` if this was not a teach drop.

    Detection and stripping are one function so they cannot disagree. The tray
    builds ``instruction\\n\\n{note}`` (``TrayViewModel.swift:523-529``), so the
    note is what follows the first blank line — and if the text does not have
    that shape, the whole thing is the note rather than nothing, because
    dropping a teach on a formatting change is worse than extracting from one
    extra sentence.

    An instruction with an empty note returns ``None``. There is nothing to
    extract from it, and the tray refuses to send one, so treating it as "not a
    teach" costs nothing and keeps the caller's branch single-valued.
    """
    if TEACH_MARKER not in text:
        return None
    head, sep, rest = text.partition("\n\n")
    note = rest.strip() if (sep and TEACH_MARKER in head) else text.strip()
    return note or None


# --- extraction -------------------------------------------------------------


def extract(
    complete: Callable[..., provider.Response],
    *,
    note: str,
    known: Sequence[Claim] = (),
    context: str = "",
) -> list[dict[str, Any]]:
    """Ask the ``learn`` role what to remember. Raises :class:`NotExtracted`.

    ``known`` is the whole learned set, not the subset that fires on this
    event: the model is being asked which existing claim a new one *replaces*,
    and a claim can only be contradicted by one it was never going to fire
    alongside. Putting the contradiction check in this call rather than in a
    second one is DL-042 #3 sited where DL-043 #6 puts it — at ingest, on a
    call that is already being made, O(claims) once per teach rather than per
    turn.
    """
    response = complete(provider.LEARN, _messages(note, known, context))
    return parse_claims(response.text, known=known)


def _messages(
    note: str, known: Sequence[Claim], context: str
) -> list[provider.Message]:
    """The extraction prompt.

    States the shape and stops. It does not explain what omega will do with a
    claim, or that a claim with no trigger applies to every turn — DL-033's
    rule is that a prompt explaining the gate is a prompt that routes around
    it, and the failure mode here is a model reaching for ``null`` because it
    was told null is the powerful option.
    """
    lines = [
        "You turn a teaching note into claims to remember.",
        "",
        'Answer with JSON and nothing else: {"claims": [...]}.',
        "Each claim is an object:",
        '  "text"       what to remember, one sentence, written as an',
        "               instruction to yourself",
        '  "situation"  what was going on when this was taught',
        '  "trigger"    when the claim applies, or null',
        '  "supersedes" the id of a claim this one replaces, or null',
        "",
        "A trigger is an object with any of these fields, combined with AND.",
        "No other field is allowed.",
        '  {"any": ["phrase", ...]}  the message mentions one of these',
        '  {"channel": "tray"}       the message arrived on this channel',
        '  {"hours": [9, 18]}        the local hour is at or after the first',
        "                            and before the second",
        "",
        'Return {"claims": []} if the note asks you to remember nothing.',
    ]
    if known:
        lines += ["", "Already remembered:"]
        lines += [f"  [{c.seq}] {c.text}" for c in known]
    system = provider.system("\n".join(lines))

    body = f"Teaching note:\n{note}"
    if context:
        body = f"{context}\n\n{body}"
    return [system, provider.user(body)]


def parse_claims(
    text: str, *, known: Sequence[Claim] = ()
) -> list[dict[str, Any]]:
    """Strict parse of the extraction answer. Raises :class:`NotExtracted`.

    The one leniency is a fenced code block, stripped before parsing, because a
    fence is a formatting habit rather than a different answer. Everything else
    is refused: a parser that hunted for a JSON object inside prose would be
    reading a model that did not follow the format as though it had.

    ``supersedes`` is checked against ``known``. An id naming no claim is
    refused rather than dropped — :meth:`omega.derive.Learned.apply` removes
    the superseded claim by key and silently succeeds when the key is absent,
    so an invented id would file a claim that claims to replace something and
    replaces nothing.
    """
    body = _unfence(text).strip()
    if not body:
        raise NotExtracted("the learn model returned no content")
    try:
        parsed = json.loads(body)
    except ValueError as exc:
        raise NotExtracted(f"the answer was not JSON: {exc}") from exc
    if not isinstance(parsed, dict) or "claims" not in parsed:
        raise NotExtracted('the answer had no "claims" field')
    raw = parsed["claims"]
    if not isinstance(raw, list):
        raise NotExtracted('"claims" was not a list')
    if len(raw) > MAX_CLAIMS_PER_NOTE:
        raise NotExtracted(
            f"{len(raw)} claims from one note, over the limit of "
            f"{MAX_CLAIMS_PER_NOTE}"
        )

    seqs = {c.seq for c in known}
    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            raise NotExtracted("a claim was not an object")
        claim_text = item.get("text")
        if not isinstance(claim_text, str) or not claim_text.strip():
            raise NotExtracted("a claim had no text")
        if len(claim_text) > MAX_CLAIM_CHARS:
            raise NotExtracted(
                f"a claim was {len(claim_text)} characters, over the limit of "
                f"{MAX_CLAIM_CHARS}"
            )
        situation = item.get("situation")
        if not isinstance(situation, str) or not situation.strip():
            raise NotExtracted("a claim had no situation")
        trigger = item.get("trigger")
        if trigger is not None and not isinstance(trigger, dict):
            raise NotExtracted("a trigger was neither an object nor null")
        if trigger == {}:
            # `episodes` refuses this too, and for the reason that matters: an
            # empty object reads as "no conditions", which is the same as
            # always, and two spellings of always is one too many.
            trigger = None
        supersedes = item.get("supersedes")
        if supersedes is not None:
            if isinstance(supersedes, bool) or not isinstance(supersedes, int):
                raise NotExtracted("a supersedes id was not a number")
            if supersedes not in seqs:
                raise NotExtracted(
                    f"a claim said it supersedes {supersedes}, which is not "
                    f"something omega has been taught"
                )
        out.append(
            {
                "text": claim_text.strip(),
                "situation": situation.strip(),
                "trigger": trigger,
                "supersedes": supersedes,
            }
        )
    return out


def _unfence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) < 2:
        return stripped
    body = lines[1:]
    if body and body[-1].strip().startswith("```"):
        body = body[:-1]
    return "\n".join(body)


# --- writing it down --------------------------------------------------------


def file_claims(
    queue: Any,
    claims: Sequence[dict[str, Any]],
    *,
    for_seq: int,
    source_seq: int,
    at: Optional[str] = None,
) -> list[Claim]:
    """Append each claim and return what was written, in order.

    ``explicit=True`` for all of them: everything this module files came from a
    note the person deliberately typed into a composer whose placeholder asks
    *What should omega learn?*. Inferred claims are DL-043 #2's deferred half
    and there is no path here that produces one.

    **No write key**, unlike the turn's terminal record. A key would make a
    replay of this window raise ``WriteKeyConflict`` — the payloads differ by
    their timestamp — and the caller turns a raised extraction into *"I could
    not record it"*, which would be a lie about claims that are in the log.
    Duplicate claims after a crash mid-window are visible in the receipt and
    can be superseded; a false failure report cannot be corrected by anyone.
    """
    written: list[Claim] = []
    for item in claims:
        payload = episodes.claim_extracted(
            for_seq=for_seq,
            text=item["text"],
            source_seq=source_seq,
            situation=item["situation"],
            explicit=True,
            trigger=item["trigger"],
            supersedes=item["supersedes"],
            at=at,
        )
        seq = queue.append(payload)
        written.append(
            Claim(
                seq=seq,
                text=item["text"],
                trigger=item["trigger"],
                situation=item["situation"],
                source_seq=source_seq,
                explicit=True,
                supersedes=item["supersedes"],
            )
        )
    return written


# --- the receipt ------------------------------------------------------------


def receipt(
    written: Sequence[Claim],
    *,
    known: Sequence[Claim] = (),
    error: Optional[str] = None,
) -> str:
    """What to append to the reply so the person can check the record.

    DL-042 #2: the receipt is the precision check, run by the only party
    holding ground truth, at the only moment it is cheap. So it says what will
    fire and when, in the person's terms, and it never says anything that was
    not appended.

    Naming a replaced claim *is* DL-042 #1's escalation, and today it is
    unconditional rather than keyed on ``explicit`` — because today every claim
    omega holds was authored deliberately by the person, so every supersession
    is the case the rule wanted escalated. The quiet path arrives with inferred
    claims and would be dead code before then.
    """
    if error:
        return (
            f"I could not write that down — {error}. Nothing was recorded, so "
            f"tell me again if it matters."
        )
    if not written:
        return "I did not find anything to remember in that, so nothing was recorded."

    by_seq = {c.seq: c for c in known}
    lines = ["I wrote this down:"]
    for claim in written:
        lines.append(f"- {claim.text} ({when_phrase(claim.trigger)})")
        replaced = by_seq.get(claim.supersedes) if claim.supersedes else None
        if replaced is not None:
            lines.append(f"  replaces what you told me before: {replaced.text}")
    return "\n".join(lines)


def when_phrase(trigger: Optional[dict[str, Any]]) -> str:
    """The trigger in words, so the person can disagree with it.

    Reads the same fields :meth:`omega.derive.Claim.fires_on` matches on, in
    the same order, which is what makes the receipt a check on the record
    rather than a second description of it.
    """
    if not trigger:
        return "always"
    parts: list[str] = []
    phrases = trigger.get("any")
    if phrases:
        parts.append("when you mention " + " or ".join(str(p) for p in phrases))
    channel = trigger.get("channel")
    if channel:
        parts.append(f"on {channel}")
    window = trigger.get("hours")
    if window:
        parts.append(f"between {int(window[0])}:00 and {int(window[1])}:00")
    return ", ".join(parts) if parts else "always"
