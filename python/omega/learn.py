"""Turning a teaching note into claims and schedules — DL-043, DL-044.

DL-042 settled what a learned claim *is*: a record with a trigger, filed in the
log, folded into a view, rendered into the prompt when it fires. This is where
one gets written. DL-044 added the other thing a teaching note can be.

**A taught time is a schedule, not a claim with an hour on it.** A claim's
``hours`` trigger is a filter on *rendering* — it decides whether the claim
joins a prompt omega is already building. So *"every morning at nine, remind me
to take my meds"* filed as a claim applies only when the person is already
talking to omega at nine, which is the one circumstance in which they did not
need reminding. The discriminator is therefore not whether a note mentions a
time but **whether it asks omega to act unprompted**: *"in the mornings I prefer
short answers"* is a real thing to teach and must stay a claim.

Four things here are load-bearing.

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
fails and the receipt says so. Schedules join the same window: a note that would
file a claim and a broken schedule files neither.

**Ids are omega's, never the model's.** A schedule id names something
cancellable, and :meth:`omega.schedule.Scheduler._observe` redefines by id
without complaint — so a model free to choose ids could silently overwrite one
standing intention with another by reaching for the same obvious word twice.
Ids are derived from the episode that taught them, and the model names an
existing schedule only by quoting an id it was shown.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional, Sequence

from omega import episodes, provider, schedule as scheduling
from omega.derive import Claim
from omega.schedule import Schedule

__all__ = [
    "TEACH_MARKER",
    "MAX_CLAIMS_PER_NOTE",
    "MAX_INFERRED_PER_PASS",
    "MAX_CLAIM_CHARS",
    "MAX_SCHEDULES_PER_NOTE",
    "MAX_INSTRUCTION_CHARS",
    "Extraction",
    "NotExtracted",
    "Lens",
    "CONVERSATION",
    "WORK",
    "DAY",
    "teaching_note",
    "extract",
    "reflect",
    "parse_answer",
    "parse_claims",
    "file_claims",
    "file_schedules",
    "cancel_schedules",
    "retract_claims",
    "schedule_id",
    "receipt",
    "review",
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

#: Schedules from one note. Lower than the claim cap and deliberately so: a
#: claim that is wrong costs prompt budget, and a schedule that is wrong wakes
#: omega up. DL-035 named unattended repetition as the price of proactivity, so
#: the bound on how much of it one sentence can buy is tighter.
MAX_SCHEDULES_PER_NOTE = 4

#: Claims one reflection pass may file (DL-054). Lower than a teach's cap, and
#: the gap is the point: a teach is a sentence the person deliberately typed, a
#: reflection is omega's own reading of a stretch of conversation, and the
#: lower-confidence source gets the tighter budget. Three is also small enough
#: that the model must choose — asked for at most three things worth keeping out
#: of forty turns, it has to rank, and ranking is most of what makes the
#: difference between a memory and a transcript.
MAX_INFERRED_PER_PASS = 3


@dataclass(frozen=True)
class Lens:
    """What a reflection pass is looking at, in the two words the prompt needs.

    Reflection reads two different objects now: a stretch of omega's own
    conversation (DL-054) and a digest of work the person did somewhere else
    (DL-057). Everything about the pass is the same — the same refusals, the
    same cap, the same claim shape, the same fold — except the sentence naming
    what is being read, and the label above it.

    Two fields rather than two prompts, because the parts that matter are the
    parts both share: *most stretches show nothing*, and *patterns, not events*.
    A second prompt would be a second place for those to drift.
    """

    #: How the system prompt opens, naming what is about to be read.
    opening: tuple[str, ...]
    #: What the material is called when it is handed over.
    label: str


#: Reflecting over omega's own recent conversation (DL-054).
CONVERSATION = Lens(
    opening=(
        "You are reviewing a stretch of your own recent conversation to see",
        "whether it shows anything worth remembering about this person.",
    ),
    label="The conversation",
)

#: Reflecting over a digest of work the person did in another agent (DL-057).
#: It says *watched*, not *took part in*, because omega did not: reading a
#: record of somebody working is weaker evidence than talking to them, and the
#: prompt should not imply a memory omega does not have.
WORK = Lens(
    opening=(
        "You are reviewing a record of work this person did in a coding tool,",
        "which you watched but did not take part in. Look for what it shows",
        "about how they work — not for what the work was about.",
    ),
    label="The session",
)

#: Reflecting over one day of the Mac's own usage record (DL-059). The weakest
#: evidence of the three and the opening says so twice — *did not see*, and
#: *rhythm, not content* — because this is the lens most likely to produce a
#: confident sentence about a person from a table of numbers. It knows when
#: they were at the machine and which app was in front; it knows nothing
#: whatever about what they were doing in it, and a claim that implies
#: otherwise is the invention this whole path is bounded against.
DAY = Lens(
    opening=(
        "You are reviewing a record of which applications this person had open",
        "during one day, which you did not see and were not part of. Look for",
        "the rhythm of their day — when they start, when they stop, when they",
        "are deep in one thing. You do not know what they were working on.",
    ),
    label="The day",
)

#: A schedule's instruction is the whole text of a future turn, so it has room
#: to be a sentence rather than a phrase — but not room to be a document that
#: nobody will see again until it fires.
MAX_INSTRUCTION_CHARS = 500


@dataclass(frozen=True)
class Extraction:
    """What one teaching note turned into, before any of it is written.

    Four lists rather than one because they are four different writes, and
    keeping them apart until :func:`file_claims` / :func:`file_schedules` /
    :func:`cancel_schedules` / :func:`retract_claims` is what lets validation
    reject the whole note without having appended part of it.
    """

    claims: list[dict[str, Any]] = field(default_factory=list)
    schedules: list[dict[str, Any]] = field(default_factory=list)
    #: Ids of standing schedules the note asks to retire.
    cancel: list[str] = field(default_factory=list)
    #: Seqs of active claims the note asks omega to forget (DL-048). The
    #: counterpart of ``cancel``, which claims went without for one entry.
    retract: list[int] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.claims or self.schedules or self.cancel or self.retract)


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
    running: Sequence[Schedule] = (),
    context: str = "",
) -> Extraction:
    """Ask the ``learn`` role what to remember. Raises :class:`NotExtracted`.

    ``known`` is the whole learned set, not the subset that fires on this
    event: the model is being asked which existing claim a new one *replaces*,
    and a claim can only be contradicted by one it was never going to fire
    alongside. Putting the contradiction check in this call rather than in a
    second one is DL-042 #3 sited where DL-043 #6 puts it — at ingest, on a
    call that is already being made, O(claims) once per teach rather than per
    turn.

    ``running`` is the standing schedules, shown for the same reason and used
    the same way: it is what makes *"stop reminding me about the meds"* a thing
    the extractor can express (DL-044 #6).

    One call returns both kinds (DL-044 #2). Two calls would let each decide
    independently that a sentence belonged to it, and a note that produced both
    a claim and a schedule for the same sentence would fire *and* nag.
    """
    response = complete(provider.LEARN, _messages(note, known, running, context))
    return parse_answer(response.text, known=known, running=running)


def _messages(
    note: str,
    known: Sequence[Claim],
    running: Sequence[Schedule],
    context: str,
) -> list[provider.Message]:
    """The extraction prompt.

    States the shape and stops. It does not explain what omega will do with a
    claim, or that a claim with no trigger applies to every turn — DL-033's
    rule is that a prompt explaining the gate is a prompt that routes around
    it, and the failure mode here is a model reaching for ``null`` because it
    was told null is the powerful option.
    """
    lines = [
        "You turn a teaching note into things to remember and things to do.",
        "",
        "Answer with JSON and nothing else:",
        '  {"claims": [...], "schedules": [...], "cancel": [...],',
        '   "retract": [...]}',
        "",
        "A claim is something to bear in mind while answering.",
        "A schedule wakes you up at a time and gives you something to do,",
        "when nobody has said anything. Use a schedule only when the note",
        "asks you to act on your own; a preference about mornings is a claim",
        "with an hours trigger, not a schedule.",
        "",
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
        "",
        "Each schedule is an object:",
        '  "instruction"  what to do when you wake, addressed to yourself',
        '  "cron"         "minute hour day-of-week", where day-of-week is',
        "                 0 for Sunday. Each field is *, a number, A-B, A,B",
        "                 or */N. Local time.",
        "",
        '"cancel" is a list of ids of standing schedules to stop.',
        '"retract" is a list of ids of remembered things to forget. Use it',
        "when the note asks you to stop believing something and puts nothing",
        "in its place; if it replaces one thing with another, write the new",
        'claim with "supersedes" instead.',
        "",
        'Return empty lists if the note asks you for nothing.',
    ]
    if known:
        lines += ["", "Already remembered:"]
        lines += [f"  [{c.seq}] {c.text}" for c in known]
    if running:
        lines += ["", "Already scheduled:"]
        lines += [f"  [{s.id}] {s.instruction}" for s in running]
    system = provider.system("\n".join(lines))

    body = f"Teaching note:\n{note}"
    if context:
        body = f"{context}\n\n{body}"
    return [system, provider.user(body)]


def reflect(
    complete: Callable[..., provider.Response],
    *,
    transcript: str,
    known: Sequence[Claim] = (),
    observing: Lens = CONVERSATION,
) -> list[dict[str, Any]]:
    """What this stretch of conversation showed. Raises :class:`NotExtracted`.

    DL-054, and the half of learning DL-043 #2 deferred. Everything
    :func:`extract` files came from a sentence the person deliberately typed
    into a composer; this is omega reading what it has already been through and
    keeping what it noticed. The output is the same claim shape and goes through
    the same :func:`file_claims`, differing only in ``explicit``.

    **A window rather than a message, and that is the whole design.** The things
    worth inferring about a person — how they work, what they keep coming back
    to, what to raise when they are doing a particular thing — are patterns, and
    a pattern is not visible in one sentence. An extractor handed a single
    message and asked what is worth remembering must either invent something or
    file nothing, and a model asked that question will invent: omega has been
    observed answering *"Black, no sugar"* for a coffee preference it was never
    told. Showing it the stretch instead is what makes *not enough evidence* an
    available answer.

    ``known`` is shown for :func:`extract`'s reason and one more. There it stops
    a new claim contradicting an old one silently; here it also stops the pass
    re-filing what it already concluded last time, which is the way this path
    would otherwise fill the learned set with near-duplicates of one true fact.

    ``schedules``, ``cancel`` and ``retract`` are deliberately *not* reachable
    from here — this returns claims and nothing else. Every one of those is
    omega doing something on the person's behalf that they did not ask for in
    that moment: inferring its way into waking them at nine, or into dropping a
    belief they taught it on purpose. A wrong inferred claim is a bad sentence
    in a prompt. A wrong inferred schedule is a notification at nine every
    morning forever, which is the failure DL-011 names as terminal.
    """
    response = complete(
        provider.LEARN, _reflection_messages(transcript, known, observing)
    )
    raw = _unfence(response.text)
    try:
        answer = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise NotExtracted(f"the answer was not JSON: {exc}") from exc
    if not isinstance(answer, dict):
        raise NotExtracted("the answer was not a JSON object")
    return _parse_claim_list(
        answer.get("claims", []), known, limit=MAX_INFERRED_PER_PASS
    )


def _reflection_messages(
    transcript: str, known: Sequence[Claim], observing: Lens = CONVERSATION
) -> list[provider.Message]:
    """The reflection prompt.

    Two things in here are load-bearing and both are about *refusing*.

    **It says the expected answer is none.** A model asked to find what is worth
    remembering in forty turns will find something, because that is what it was
    asked to do, and the result is a learned set that grows without bound while
    every individual claim looks locally plausible. DL-011 names the
    notification firehose as the loop's terminal failure; the memory firehose is
    the same failure with a longer fuse, and worse, because the person never
    sees the moment a claim is filed — only a slow fog of things omega
    half-believes about them. Saying *most stretches contain nothing* up front
    is the cheapest guard available, and the eval carries the check that it
    works.

    **It asks for a pattern, not an event.** *"He mentioned the Tuesday standup"*
    is a fact already in the log and recall will find it; *"he prepares for the
    Tuesday standup on Monday evenings"* is a thing to bear in mind that recall
    cannot reconstruct. The first is what a model reaches for by default, so the
    distinction is drawn explicitly and with both examples.
    """
    lines = [
        *observing.opening,
        "",
        "Most stretches show nothing. Returning an empty list is the normal",
        "and expected answer. Only write something down if the conversation",
        "shows it more than once, or states it as a standing fact.",
        "",
        "Remember patterns, not events. What happened is already recorded and",
        "you can look it up. Write down only what would change how you answer",
        "a future question:",
        '  bad   "He asked about the Tuesday standup."',
        '  good  "He prepares for the Tuesday standup on Monday evenings —',
        '         raise anything standup-related before then."',
        "",
        "Never write down anything you are guessing at. If the conversation",
        "hints at a preference without stating it, that is not evidence.",
        "",
        "Answer with JSON and nothing else:",
        '  {"claims": [...]}',
        "",
        f"At most {MAX_INFERRED_PER_PASS} claims. Each is an object:",
        '  "text"       what to remember, one sentence, written as an',
        "               instruction to yourself",
        '  "situation"  what was going on when you noticed this',
        '  "trigger"    when the claim applies, or null',
        '  "supersedes" the id of a claim this one replaces, or null',
        "",
        "A trigger is an object with any of these fields, combined with AND.",
        "No other field is allowed.",
        '  {"any": ["phrase", ...]}  the message mentions one of these',
        '  {"channel": "tray"}       the message arrived on this channel',
        '  {"hours": [9, 18]}        the local hour is at or after the first',
        "                            and before the second",
    ]
    if known:
        lines += [
            "",
            "You already believe these. Do not write any of them down again;",
            "if this conversation refines one, use its id in supersedes.",
        ]
        lines += [f"  [{c.seq}] {c.text}" for c in known]
    system = provider.system("\n".join(lines))
    return [system, provider.user(f"{observing.label}:\n{transcript}")]


def parse_answer(
    text: str,
    *,
    known: Sequence[Claim] = (),
    running: Sequence[Schedule] = (),
) -> Extraction:
    """Strict parse of the extraction answer. Raises :class:`NotExtracted`.

    The one leniency is a fenced code block, stripped before parsing, because a
    fence is a formatting habit rather than a different answer. Everything else
    is refused: a parser that hunted for a JSON object inside prose would be
    reading a model that did not follow the format as though it had.

    ``schedules``, ``cancel`` and ``retract`` may be absent — each was additive
    on an answer shape that already existed — but ``claims`` may not, because
    the model is always told to return it and an answer missing it is an answer
    in a different format.
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
    return Extraction(
        claims=_parse_claim_list(parsed["claims"], known),
        schedules=_parse_schedule_list(parsed.get("schedules")),
        cancel=_parse_cancel_list(parsed.get("cancel"), running),
        retract=_parse_retract_list(parsed.get("retract"), known),
    )


def parse_claims(
    text: str, *, known: Sequence[Claim] = ()
) -> list[dict[str, Any]]:
    """The claims of :func:`parse_answer`, for callers that want only those."""
    return parse_answer(text, known=known).claims


def _parse_claim_list(
    raw: Any, known: Sequence[Claim], limit: int = MAX_CLAIMS_PER_NOTE
) -> list[dict[str, Any]]:
    """``supersedes`` is checked against ``known``. An id naming no claim is
    refused rather than dropped — :meth:`omega.derive.Learned.apply` removes
    the superseded claim by key and silently succeeds when the key is absent,
    so an invented id would file a claim that claims to replace something and
    replaces nothing.
    """
    if not isinstance(raw, list):
        raise NotExtracted('"claims" was not a list')
    if len(raw) > limit:
        raise NotExtracted(
            f"{len(raw)} claims from one note, over the limit of {limit}"
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


def _parse_schedule_list(raw: Any) -> list[dict[str, Any]]:
    """Validate proposed schedules, including that the expression parses.

    DL-044 #5: a cron expression is wrong in a way nothing downstream notices,
    so it is checked here rather than quarantined at fold time. DL-035 accepted
    fold-time quarantine because episodes can predate the parser; that is not a
    licence to write one we already know is broken while the person is present
    to be told.

    Only ``cron`` is accepted, never ``every`` (DL-044 #7). Cron's resolution is
    one minute and `Scheduler._slot_for` requires a strictly newer slot, so the
    busiest expression this path can write fires once a minute — which is
    exactly `episodes.MIN_EVERY_SECONDS`. The floor on the new path is the
    grammar rather than a second check that could drift from the first.
    """
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise NotExtracted('"schedules" was not a list')
    if len(raw) > MAX_SCHEDULES_PER_NOTE:
        raise NotExtracted(
            f"{len(raw)} schedules from one note, over the limit of "
            f"{MAX_SCHEDULES_PER_NOTE}"
        )
    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            raise NotExtracted("a schedule was not an object")
        instruction = item.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            raise NotExtracted("a schedule had no instruction")
        if len(instruction) > MAX_INSTRUCTION_CHARS:
            raise NotExtracted(
                f"a schedule instruction was {len(instruction)} characters, "
                f"over the limit of {MAX_INSTRUCTION_CHARS}"
            )
        cron = item.get("cron")
        if not isinstance(cron, str) or not cron.strip():
            raise NotExtracted("a schedule had no cron expression")
        try:
            scheduling.validate_cron(cron)
        except scheduling.CronError as exc:
            raise NotExtracted(f"a schedule's timing was unusable: {exc}") from exc
        out.append({"instruction": instruction.strip(), "cron": cron.strip()})
    return out


def _parse_cancel_list(raw: Any, running: Sequence[Schedule]) -> list[str]:
    """An id naming nothing standing is refused, for DL-043 #6's reason.

    :meth:`omega.schedule.Scheduler._observe` pops on cancel with a default, so
    cancelling something that is not running succeeds silently — and the
    receipt would then tell the person a reminder had stopped while it went on
    firing. That is the one failure this whole path exists to avoid, arriving
    from the other direction.
    """
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise NotExtracted('"cancel" was not a list')
    live = {s.id for s in running}
    out: list[str] = []
    for item in raw:
        if not isinstance(item, str) or not item.strip():
            raise NotExtracted("a cancel id was not a string")
        sid = item.strip()
        if sid not in live:
            raise NotExtracted(
                f"asked to stop {sid!r}, which is not something omega has "
                f"scheduled"
            )
        if sid not in out:
            out.append(sid)
    return out


def _parse_retract_list(raw: Any, known: Sequence[Claim]) -> list[int]:
    """An id naming nothing currently believed is refused (DL-048 #4).

    The same rule as :func:`_parse_cancel_list`, and it binds harder here. A
    cancel that silently no-ops leaves a reminder firing while the receipt says
    it stopped — annoying, and visible the next time it fires. A *retract* that
    named the wrong claim would quietly drop an instruction the person
    deliberately authored, and nothing would ever fire to reveal it: the only
    symptom is omega gradually not doing something it was told. Destructive to
    the derived view beats annoying, so an unknown id fails the whole note
    rather than being skipped.
    """
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise NotExtracted('"retract" was not a list')
    live = {c.seq for c in known}
    out: list[int] = []
    for item in raw:
        # ``bool`` first: ``isinstance(True, int)`` is true, and ``True`` would
        # otherwise sail through as claim 1.
        if isinstance(item, bool) or not isinstance(item, int):
            raise NotExtracted("a retract id was not a number")
        if item not in live:
            raise NotExtracted(
                f"asked to forget claim {item}, which is not something omega "
                f"currently believes"
            )
        if item not in out:
            out.append(item)
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
    explicit: bool = True,
) -> list[Claim]:
    """Append each claim and return what was written, in order.

    ``explicit`` defaults to true because the teach path is the one that must
    not get this wrong by omission: a note the person deliberately typed into a
    composer whose placeholder asks *What should omega learn?* is explicit by
    definition, and a default of false would silently downgrade every taught
    claim if a caller forgot the argument. :func:`reflect`'s caller passes false
    and is the only thing that does (DL-054).

    The flag is not a confidence score and nothing treats it as one. DL-042 gave
    it one job — deciding whether a contradiction is worth interrupting the
    person about — and the reason it can stay that cheap is that neither kind of
    claim can destroy anything: both leave the active set by supersession or
    retraction, both stay in the log, and re-derivation reaches the earlier
    state either way.

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
            explicit=explicit,
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
                explicit=explicit,
                supersedes=item["supersedes"],
            )
        )
    return written


def schedule_id(source_seq: int, index: int) -> str:
    """The id for the ``index``-th schedule taught by episode ``source_seq``.

    Derived rather than chosen, because ids are the handle by which a standing
    intention is cancelled and :meth:`omega.schedule.Scheduler._observe`
    redefines by id without complaint. A model free to name its own would reach
    for the same obvious word twice across two notes and silently replace one
    reminder with another; deriving from the teaching episode makes collision
    impossible without anyone having to be careful.
    """
    return f"s{source_seq}-{index}"


def file_schedules(
    queue: Any,
    schedules: Sequence[dict[str, Any]],
    *,
    source_seq: int,
    at: Optional[str] = None,
) -> list[Schedule]:
    """Append each schedule definition and return what was written, in order.

    No write key, for :func:`file_claims`' reason. Cron only, so the ``every``
    field of the episode stays what it was — a programmatic spelling, not
    something a sentence can reach.
    """
    written: list[Schedule] = []
    for index, item in enumerate(schedules):
        sid = schedule_id(source_seq, index)
        payload = episodes.schedule_created(
            id=sid,
            instruction=item["instruction"],
            cron=item["cron"],
            at=at,
        )
        queue.append(payload)
        written.append(
            Schedule(
                id=sid,
                instruction=item["instruction"],
                created_at=datetime.fromisoformat(payload["at"]),
                cron=item["cron"],
            )
        )
    return written


def cancel_schedules(
    queue: Any,
    ids: Sequence[str],
    *,
    running: Sequence[Schedule] = (),
    at: Optional[str] = None,
) -> list[Schedule]:
    """Retire each schedule and return the definitions that were stopped.

    Returns the *definitions* rather than the ids so the receipt can say what
    stopped in the person's words. Telling someone ``s41-0`` has been cancelled
    is a confirmation they cannot check, which is the same defect as a
    done-marker wearing different clothes.
    """
    by_id = {s.id: s for s in running}
    stopped: list[Schedule] = []
    for sid in ids:
        queue.append(episodes.schedule_cancelled(id=sid, at=at))
        existing = by_id.get(sid)
        if existing is not None:
            stopped.append(existing)
    return stopped


def retract_claims(
    queue: Any,
    seqs: Sequence[int],
    *,
    for_seq: int,
    known: Sequence[Claim] = (),
    at: Optional[str] = None,
) -> list[Claim]:
    """Forget each claim and return the ones that were dropped (DL-048).

    Returns the :class:`~omega.derive.Claim` objects rather than the seqs for
    :func:`cancel_schedules`' reason, which applies word for word: telling
    someone claim ``41`` has been forgotten is a confirmation they cannot
    check. The receipt has to say the sentence.
    """
    by_seq = {c.seq: c for c in known}
    dropped: list[Claim] = []
    for seq in seqs:
        queue.append(
            episodes.claim_retracted(for_seq=for_seq, claim_seq=seq, at=at)
        )
        existing = by_seq.get(seq)
        if existing is not None:
            dropped.append(existing)
    return dropped


# --- the receipt ------------------------------------------------------------


def receipt(
    written: Sequence[Claim],
    *,
    known: Sequence[Claim] = (),
    scheduled: Sequence[Schedule] = (),
    stopped: Sequence[Schedule] = (),
    forgotten: Sequence[Claim] = (),
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

    ``forgotten`` is the retraction half (DL-048) and it is named in the
    person's own sentence rather than by seq, for the reason
    :func:`cancel_schedules` gives about ids. It matters more here than
    anywhere else in this receipt: a retraction is the one thing omega does
    that makes it *less* capable, so if the wrong claim was dropped this line
    is the only place it will ever be visible.
    """
    if error:
        return (
            f"I could not write that down — {error}. Nothing was recorded, so "
            f"tell me again if it matters."
        )
    if not (written or scheduled or stopped or forgotten):
        return "I did not find anything to remember in that, so nothing was recorded."

    by_seq = {c.seq: c for c in known}
    lines: list[str] = []
    if written:
        lines.append("I wrote this down:")
        for claim in written:
            lines.append(f"- {claim.text} ({when_phrase(claim.trigger)})")
            replaced = by_seq.get(claim.supersedes) if claim.supersedes else None
            if replaced is not None:
                lines.append(f"  replaces what you told me before: {replaced.text}")
    if scheduled:
        if lines:
            lines.append("")
        lines.append("I will wake up and do this:")
        for item in scheduled:
            # Rendered from the stored expression, not from the note (DL-044
            # #5). `0 9 *` and `9 0 *` are both valid and only one of them is
            # nine in the morning, so the phrasing has to come from the thing
            # the clock will actually read.
            lines.append(
                f"- {item.instruction} ({_cron_phrase(item.cron)}) [{item.id}]"
            )
    if stopped:
        if lines:
            lines.append("")
        lines.append("I stopped this:")
        lines += [f"- {item.instruction}" for item in stopped]
    if forgotten:
        if lines:
            lines.append("")
        lines.append("I forgot this:")
        lines += [f"- {claim.text}" for claim in forgotten]
    return "\n".join(lines)


def _cron_phrase(cron: Optional[str]) -> str:
    """The expression in words, degrading to the expression itself.

    A schedule written by this module has already parsed, so the fallback is
    unreachable from :func:`file_schedules` — it exists because the receipt must
    never be the thing that raises. Losing the plain-English phrasing costs the
    person some clarity; losing the reply costs them the whole turn.
    """
    if not cron:
        return "on a timer"
    try:
        return scheduling.describe_cron(cron)
    except scheduling.CronError:
        return cron


def review(
    claims: Sequence[Claim],
    *,
    running: Sequence[Schedule] = (),
    broken: Optional[dict[str, str]] = None,
    current: Optional[bool] = None,
    failed: int = 0,
    last_failure: str = "",
) -> str:
    """Everything omega is currently carrying, in the person's own words (DL-048).

    The counterpart of :func:`receipt` and the opposite question. A receipt says
    *what this note just changed*; this says *what is standing right now*. Until
    it existed there was no answer to the second at all: the learned set reaches
    the model only as the handful of claims whose triggers fire on the current
    sentence (``turn.py:220``), so asking omega what it had learned was answered
    from whatever happened to match that question — and the whole set was
    visible nowhere.

    **Ids are shown here although** :func:`receipt` **hides them.** Not a
    contradiction: that rule is about *confirmations*, where "claim 41 has been
    forgotten" is a sentence the person cannot check. In a listing the id sits
    against the text it names, so it is checkable by construction, and it gives
    the person and the model the same stable handle — the one ``supersedes`` and
    ``retract`` already quote.

    **Schedules are here too, and the broken ones especially.** A schedule whose
    expression will not parse is quarantined at fold time (DL-035) and is
    otherwise invisible: it was written down, it was confirmed, and it silently
    never fires. Review is the only surface on which that is ever discoverable,
    so leaving it out would have made this a report that hides its own worst
    news.

    ``failed`` is that same rule one layer up and is the whole of DL-053. An
    extraction that fails says so in a receipt and then nothing keeps it, so a
    store whose `learn` role has been returning 400 on every call reads exactly
    like a store nobody has taught — *"I have not been taught anything yet"* was
    the literal output of a log holding thirteen failed teach drops. When the
    count is non-zero that sentence is replaced rather than decorated: omega has
    not been taught nothing, it has been taught and could not write it down, and
    those are different enough that the person's next action differs.

    ``current`` makes the empty answer three-valued in the way *fail closed on
    empty* requires. "Folded the whole log and omega has been taught nothing"
    and "folded none of it" are the same empty list, and only the first is an
    answer. It is a claim about the *view*, not about the counts: a
    default-constructed :class:`~omega.derive.Learned` reports ``through == 0``
    and so does a rebuild over a log with nothing in it yet, and those two must
    not render the same way. The caller holds both numbers, so the caller says.

    ``None`` — the caller did not say — renders as the uncertain answer rather
    than the reassuring one, so silence from a future caller cannot turn into
    "omega has been taught nothing" on a view that was simply never advanced.
    That case is live, not hypothetical: ``executor.learned`` is empty until the
    first turn folds it.
    """
    broken = broken or {}

    def _failures() -> list[str]:
        if failed <= 0:
            return []
        times = "once" if failed == 1 else f"{failed} times"
        out = [f"I was taught something and could not record it ({times}):"]
        if last_failure:
            out.append(f"- {last_failure}")
        return out

    if not (claims or running or broken):
        if failed > 0:
            # Before the `current` branch on purpose. A view that has not folded
            # the whole log cannot say what omega knows, but it can say that
            # what it *has* read contains failures — and that is the more urgent
            # of the two things, because it is actionable and the other is not.
            return "\n".join(_failures())
        if current:
            return "I have not been taught anything yet."
        return "I have not read the whole log, so I cannot tell you what I know."

    lines: list[str] = _failures()
    if lines:
        lines.append("")
    if claims:
        lines.append("What I believe about you:")
        for claim in claims:
            lines.append(f"- [{claim.seq}] {claim.text} ({when_phrase(claim.trigger)})")
            if claim.situation:
                lines.append(f"    learned {claim.situation}")
    if running:
        if lines:
            lines.append("")
        lines.append("What I will wake up and do:")
        for item in running:
            # From the stored expression, never from the note that taught it —
            # DL-044 #5's rule, and review is the place a person would catch the
            # `0 9 *` / `9 0 *` transposition that a receipt read past weeks ago.
            lines.append(
                f"- [{item.id}] {item.instruction} ({_cron_phrase(item.cron)})"
            )
    if broken:
        if lines:
            lines.append("")
        lines.append("Written down, but I cannot run these:")
        for schedule_id_, reason in sorted(broken.items()):
            lines.append(f"- [{schedule_id_}] {reason}")
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
