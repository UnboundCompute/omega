"""One turn — M1 step 3, spec `agent/M1_SPEC.md` §1.4, §1.5; DL-011.

**perceive -> recall -> judge -> act -> reply -> write memory.** One function,
six named steps, in that order, each separately testable.

Two things here are load-bearing rather than structural:

**`judge` is its own step and is a real model call.** DL-011 puts the decision
to say nothing in `judge` precisely so that it is not inside an action path;
merging it into `act` is how silence turns from an outcome into a failure mode.
DL-024 then makes it a *real* call, because a judge that always returns
``speak`` compiles, passes every other test in the milestone, and never once
exercises the property the loop exists for.

**Silence is a success.** A turn that decides not to speak writes
``turn.completed{outcome: "silent", reply: null}`` and is as complete as one
that spoke. A model that returns *no content* is a different thing entirely —
the provider raises there (see `provider.py`), and the turn records
``outcome: "failed"`` with the error. The two must never collapse into one
recorded outcome, because every later metric that counts turns reads this field
and M5 becomes a notification firehose if it cannot tell them apart.

`act` is a **sub-loop, never a sub-agent** (§1.5) and lands at M1 step 7. It is
a parameter here rather than a hole: the turn calls whatever act step it is
given, so step 7 plugs in without touching the terminal-record path that the
restart test depends on.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

from omega import blobs, derive, episodes, provider
from omega.queue import EVENT_KINDS, EventQueue, Pending

__all__ = [
    "SPEAK",
    "ACT_THEN_SPEAK",
    "STAY_SILENT",
    "VERDICTS",
    "RECALL_N",
    "MAX_EVENT_CHARS",
    "MAX_NEW_EVENT_CHARS",
    "MAX_RECALL_CHARS",
    "MAX_OPEN_CHARS",
    "MAX_LEARNED_CHARS",
    "MAX_ACT_PASSES",
    "JudgeUndecided",
    "NotAnEvent",
    "Verdict",
    "ActResult",
    "TurnContext",
    "TurnResult",
    "perceive",
    "recall",
    "judge",
    "no_act_loop_yet",
    "reply",
    "write_memory",
    "run_turn",
]

#: The three answers `judge` may give (§1.4). Closed: a fourth would be a new
#: shape of turn, not a new string.
SPEAK = "speak"
ACT_THEN_SPEAK = "act_then_speak"
STAY_SILENT = "stay_silent"
VERDICTS = frozenset({SPEAK, ACT_THEN_SPEAK, STAY_SILENT})

#: §2.2 — recall is the last N episodes, newest last, no scoring. Deliberately
#: dumb: DL-019 says making recall good first trades a date that cannot be
#: recovered for a quality that can. **N is a labelled guess**, tuned in M3
#: against the corpus M1 is here to start.
RECALL_N = 40

#: **A count is not a budget** (DL-039). ``RECALL_N`` bounds how many episodes
#: come back, which was mistaken for bounding how much *context* they cost.
#: One pasted document renders in full, on every later turn, into every role's
#: prompt — including the judge, which now fires on every clock tick with
#: nobody watching. Measured: a single 100 KB paste is ~27k tokens re-sent
#: forty times.
#:
#: Two budgets, because history and the live message answer different
#: questions. A recalled line is *context* — enough to know the thing was said
#: and roughly what it was. The new event is the *subject*, and cutting it is
#: cutting the thing omega was asked about, so its budget is far looser and it
#: exists only to stop one paste taking the whole turn down.
#:
#: Guesses, labelled as such, and cheap to change: nothing is lost by cutting
#: too hard, because the log keeps the original and the elision note says where.
MAX_EVENT_CHARS = 2_000
MAX_NEW_EVENT_CHARS = 32_000

#: The whole recalled transcript, after per-line elision. Backstop for the
#: shape per-line caps miss: forty lines each just under the line cap. Reached
#: oldest-first, because when something has to go the least recent thing is the
#: one whose absence is least likely to be the answer.
MAX_RECALL_CHARS = 24_000

#: What's open, after per-line elision. Small next to recall, and deliberately:
#: this section is a *list of unanswered questions*, not a transcript, and a
#: person who has accumulated enough open questions to fill 24k characters has a
#: problem no budget fixes. Dropped oldest-first like recall, which here means
#: the questions that survive are the ones asked most recently — an unanswered
#: question from months ago is likelier to be dead than overdue. The drop is
#: announced, for the same reason recall's is (DL-039).
MAX_OPEN_CHARS = 2_000

#: What omega has been taught that applies to this turn, after per-line elision
#: (DL-042). Larger than the open-questions budget and far smaller than recall,
#: because a learned claim is one sentence and forty of them firing at once is
#: already a sign the triggers are too loose rather than a sign the budget is too
#: small. The drop is announced, like every other budget here, and its order is
#: the one thing about this section that is a *judgement* — see
#: :func:`_learned_section`.
MAX_LEARNED_CHARS = 3_000

#: §2.3 — a fixed cap on `act` passes. Not an answer to "where does the stop
#: threshold sit"; a floor that makes the question answerable with real stalls
#: instead of a guess. The sub-loop that consumes it lands at step 7.
MAX_ACT_PASSES = 8


class JudgeUndecided(RuntimeError):
    """The judge answered something that is not one of the three verdicts.

    **Not defaulted.** Reading an unparseable answer as ``speak`` would make
    silence unreachable by accident, and reading it as ``stay_silent`` would
    turn a broken judge into a quiet one — the failure DL-024 says would be
    invisible to every other test. So it fails the turn, loudly and in the log.
    """


class NotAnEvent(ValueError):
    """A turn was asked for over a record the loop itself wrote.

    Only :data:`omega.queue.EVENT_KINDS` cause turns. A ``turn.completed``
    flowing into ``perceive`` means the drain's skip guard is broken, and that
    is the bug that loops forever (§1.1).
    """


@dataclass(frozen=True)
class Verdict:
    """What `judge` decided, and the raw text it decided it with.

    ``raw`` is kept because the verdict is the one field a later metric reads
    to count silences (DL-024), and a count is only trustworthy if the thing it
    counted can still be read back.
    """

    choice: str
    raw: str

    @property
    def speaks(self) -> bool:
        return self.choice in (SPEAK, ACT_THEN_SPEAK)


@dataclass(frozen=True)
class ActResult:
    """What the `act` sub-loop did. Empty at M1; step 7 fills it.

    ``blocked_on`` is where "is this moving?" answering **no** surfaces
    (§1.5, §2.1). It ends the turn with ``turn.blocked`` rather than a failure,
    because stopping to ask is correct behaviour and not an error — and because
    a blocked turn that stayed claimed would freeze the single consumer for as
    long as the human took to answer.

    ``text`` is the answer the sub-loop **already composed** — the message the
    model stopped on when it stopped asking for tools. §1.5 makes that the
    done-signal, and the same message is the reply: it is the only text in the
    turn written by something that could see what the tools returned.

    Throwing it away is not neutral. Measured on a real store: the loop wrote
    the file, read it back, and answered "the number 7 has been successfully
    written ... reading it back confirms it"; :func:`reply` then discarded that,
    re-asked a model holding tool *names* and no results, and omega told the
    person "I can't perform the action right now" about work it had just
    finished. A second call cannot re-derive what the first one saw.
    """

    tools: tuple[str, ...] = ()
    blocked_on: Optional[str] = None
    error: Optional[str] = None
    stop_reason: str = ""
    text: Optional[str] = None


@dataclass(frozen=True)
class TurnContext:
    """Everything a step is allowed to see. Passed rather than imported, so a
    test can drive any step without a process-wide provider."""

    seq: int
    event: dict[str, Any]
    recalled: Sequence[Pending]
    queue: EventQueue
    complete: Callable[..., provider.Response]

    #: The other half of DL-019's "recent N + what's open" — questions omega
    #: asked and has no answer to yet, derived from the log rather than stored
    #: (DL-041). Defaulted to empty so every existing caller keeps working and
    #: so a step can be driven without a store; empty means *nothing is open*,
    #: which is also what it means when the view has genuinely found nothing.
    #: Those two are the same claim here on purpose — the difference between
    #: "rebuilt and empty" and "never rebuilt" belongs to whoever builds the
    #: view (``OpenWork.through``), not to a prompt that can only render what
    #: it is handed.
    open_work: Sequence[derive.OpenBlock] = ()

    #: The claims that fire on *this* event (DL-042) — already matched, not the
    #: whole learned set. Matching happens outside the prompt for the same
    #: reason recall does: the prompt renders what it is handed, and a section
    #: that decided for itself what applied would be a second place where
    #: retrieval policy lived. Empty means nothing fired, which is the common
    #: case and renders as nothing at all.
    learned: Sequence[derive.Claim] = ()


@dataclass(frozen=True)
class TurnResult:
    """How the turn ended, mirroring what was written to the log.

    ``outcome`` has four values and ``blocked`` is one of them, even though the
    codec spells it as a separate *kind*: a caller asking "how did that turn
    end" wants one answer, and collapsing blocked into failed is exactly what
    §2.1 refuses.
    """

    seq: int
    outcome: str
    record_seq: int
    reply: Optional[str] = None
    tools: tuple[str, ...] = ()
    error: Optional[str] = None
    needs: Optional[str] = None
    verdict: Optional[Verdict] = None

    @property
    def spoke(self) -> bool:
        return self.outcome == "spoke"

    @property
    def silent(self) -> bool:
        """A success. Never read this as a degraded ``spoke`` (DL-011)."""
        return self.outcome == "silent"

    @property
    def failed(self) -> bool:
        return self.outcome == "failed"

    @property
    def blocked(self) -> bool:
        return self.outcome == "blocked"


# --- the six steps ----------------------------------------------------------


def perceive(pending: Pending) -> dict[str, Any]:
    """Step 1 — read the claimed episode; decode its payload.

    Decoding already happened at the queue, so what is left is the check that
    this episode is something a turn may run over at all.
    """
    if pending.kind not in EVENT_KINDS:
        raise NotAnEvent(
            f"seq {pending.seq} is a {pending.kind!r}, which is a record the loop "
            f"wrote; only {sorted(EVENT_KINDS)} cause turns"
        )
    return pending.payload


def recall(queue: EventQueue, *, before: int, n: int = RECALL_N) -> list[Pending]:
    """Step 2 — the last ``n`` episodes, newest last, nothing else (§2.2).

    Reads through the queue (and so through ``episodes_since``), never through
    diagnostics. ``before`` excludes the event being handled: it arrives via
    `perceive`, and a turn that saw its own trigger twice would be reasoning
    over a transcript that never happened.

    The honest limitation, per §2.2: DL-019 describes M1 recall as "recent N +
    what's open". *What's open* is derived and lands in M2. This is the
    recent-N half, and it does not pretend otherwise.
    """
    return queue.recent(n, before=before)


def judge(ctx: TurnContext) -> Verdict:
    """Step 3 — decide: speak, act then speak, or stay silent (DL-024).

    A real model call through the provider seam, on the ``judge`` role, which
    exists separately from ``act`` because this one fires on *every* event
    including every future idle tick and its answer is usually "nothing".

    Parsing is deliberately strict: the **first** word of the answer must be
    one of the three, or it is :class:`JudgeUndecided`. A lenient parser that
    scanned the whole answer for a keyword would read "not silent" as silence.
    """
    response = ctx.complete(provider.JUDGE, _judge_messages(ctx))
    return parse_verdict(response.text)


def no_act_loop_yet(ctx: TurnContext) -> ActResult:
    """Step 4 — the `act` sub-loop's slot. Empty at M1, by build order.

    §1.5's two questions — *are we done, verified?* and *is this moving?* — and
    the pass cap are step 7's work. Until then a turn that judged
    ``act_then_speak`` runs no tool passes and composes its reply directly,
    which is visible in the log as a ``turn.completed`` carrying no tools rather
    than as a turn that silently did nothing.
    """
    return ActResult(stop_reason="no act sub-loop at M1 step 3; lands at step 7")


def reply(ctx: TurnContext, verdict: Verdict, acted: ActResult) -> Optional[str]:
    """Step 5 — emit outbound, or nothing on ``stay_silent``.

    Returns ``None`` for silence, which is *not* an empty reply: the caller
    records the difference as an outcome, because ``""`` and "chose not to
    speak" must never be the same row.

    **Where the text comes from is thinly specified.** The spec names the step
    and the verdict set but never says which role composes the words, so this
    takes the smallest reading consistent with the seam: composing a reply is
    real work, and real work is the ``act`` role's model. Flagged rather than
    assumed — see the report accompanying this commit.

    There is no transport call here on purpose. Outbound at M1 is the
    projection of the episode stream (§Q10, step 5), which reads the record
    this turn is about to write. A second output path would be the thing §1.6
    refuses.
    """
    if not verdict.speaks:
        return None
    if acted.text is not None and acted.text.strip():
        # The sub-loop already answered, holding every tool result. Composing
        # again here would ask a model that cannot see any of them to describe
        # work it did not watch — which is exactly how omega came to write a
        # file and then say it could not (see :class:`ActResult`). The cheapest
        # correct reply is the one already written.
        return acted.text
    response = ctx.complete(provider.ACT, _reply_messages(ctx, acted))
    if not response.text.strip():
        # DL-011, stated as sharply as it deserves: a model that **returns no
        # content** is a failure, and a model that **decides not to speak** is a
        # success. Returning "" here would hand the caller the same `None` the
        # silent path uses and merge the two forever.
        raise provider.ProviderError(
            "the reply model returned empty content; that is a failed turn, "
            "not a decision to stay silent"
        )
    return response.text


def write_memory(
    queue: EventQueue,
    *,
    seq: int,
    outcome: str,
    reply_text: Optional[str] = None,
    tools: Sequence[str] = (),
    error: Optional[str] = None,
    needs: Optional[str] = None,
    at: Optional[str] = None,
) -> int:
    """Step 6 — append the turn's terminal record and return its seq.

    Carries ``write_key = "turn:N"`` (:func:`omega.episodes.turn_write_key`), so
    a second terminal record for the same turn is rejected **by the log** with
    ``WriteKeyConflict`` rather than by a convention in the loop. That is the
    restart-safety mechanism, not decoration: it is what lets the startup path
    tell "this turn was cut off" from "this turn finished and the crash landed
    before the cursor moved".

    Deliberately does **not** advance ``DONE``. The record lands first and the
    cursor moves after (§1.2); crashing between them over-reports a finished
    turn as interrupted, which is the safe direction, and the reverse order
    loses turns silently.
    """
    if outcome == "blocked":
        if not needs:
            raise ValueError("a blocked turn must say what it needs")
        payload = episodes.blocked(for_seq=seq, needs=needs, at=at)
    else:
        payload = episodes.completed(
            for_seq=seq,
            outcome=outcome,
            reply=reply_text,
            tools=list(tools),
            error=error,
            at=at,
        )
    return queue.append(payload, episodes.turn_write_key(seq))


# --- the turn ---------------------------------------------------------------


def run_turn(
    queue: EventQueue,
    pending: Pending,
    *,
    complete: Optional[Callable[..., provider.Response]] = None,
    act: Callable[[TurnContext], ActResult] = no_act_loop_yet,
    recall_n: int = RECALL_N,
    open_work: Sequence[derive.OpenBlock] = (),
    learned: Sequence[derive.Claim] = (),
    at: Optional[str] = None,
) -> TurnResult:
    """Run one turn over an already-claimed episode and record how it ended.

    **The caller claims before calling this and advances ``DONE`` after it
    returns.** That split is §1.2: the claim has to be durable before anything
    happens, and ``DONE`` has to move after the terminal record is durable.

    Every path through here ends with exactly one terminal record. A failing
    judge, a provider outage, an unparseable verdict and an outright bug all
    land as ``turn.completed{outcome: "failed"}`` with the reason in ``error``,
    because a claimed turn that ends without a record is the one shape the
    restart test cannot distinguish from a crash — and it would leave ``DONE``
    stuck behind ``CLAIMED`` forever, which stops the single consumer.

    The one exception is a failure to *append the record itself*: there is
    nothing left to write it with, so it propagates and the turn is genuinely
    in-flight for the startup report to find.

    ``open_work`` is passed in rather than derived here. The view is a fold over
    the whole log and its whole value is that it is *kept*, so a turn that built
    its own would replay the log once per turn and pay O(log) for a fact its
    caller already has. It defaults to empty so a test can drive one turn
    without a view — which is also why the default is a silent empty section
    rather than a claim that nothing is open: see :func:`_open_section`.

    ``learned`` arrives the same way and **already matched** against this event.
    Matching here instead would mean a turn could not be driven without a store,
    and would put the decision about what applies inside the thing that renders
    it — the same separation recall has kept since M1.
    """
    complete = complete or provider.complete
    event = perceive(pending)
    recalled = recall(queue, before=pending.seq, n=recall_n)
    ctx = TurnContext(
        seq=pending.seq,
        event=event,
        recalled=recalled,
        queue=queue,
        complete=complete,
        open_work=tuple(open_work),
        learned=tuple(learned),
    )

    verdict: Optional[Verdict] = None
    acted = ActResult()
    try:
        verdict = judge(ctx)
        if verdict.choice == ACT_THEN_SPEAK:
            acted = act(ctx)
            if acted.blocked_on:
                record = write_memory(
                    queue,
                    seq=pending.seq,
                    outcome="blocked",
                    needs=acted.blocked_on,
                    at=at,
                )
                return TurnResult(
                    seq=pending.seq,
                    outcome="blocked",
                    record_seq=record,
                    tools=tuple(acted.tools),
                    needs=acted.blocked_on,
                    verdict=verdict,
                )
            if acted.error:
                raise RuntimeError(acted.error)
        text = reply(ctx, verdict, acted)
    except Exception as exc:  # noqa: BLE001 - see the docstring: every path records
        error = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
        record = write_memory(
            queue,
            seq=pending.seq,
            outcome="failed",
            tools=acted.tools,
            error=error,
            at=at,
        )
        return TurnResult(
            seq=pending.seq,
            outcome="failed",
            record_seq=record,
            tools=tuple(acted.tools),
            error=error,
            verdict=verdict,
        )

    if text is None:
        # Silence. A success, recorded as one, with `reply` null rather than
        # empty so that "said nothing" can never be read as "said ''".
        record = write_memory(
            queue, seq=pending.seq, outcome="silent", tools=acted.tools, at=at
        )
        return TurnResult(
            seq=pending.seq,
            outcome="silent",
            record_seq=record,
            tools=tuple(acted.tools),
            verdict=verdict,
        )

    record = write_memory(
        queue,
        seq=pending.seq,
        outcome="spoke",
        reply_text=text,
        tools=acted.tools,
        at=at,
    )
    return TurnResult(
        seq=pending.seq,
        outcome="spoke",
        record_seq=record,
        reply=text,
        tools=tuple(acted.tools),
        verdict=verdict,
    )


# --- verdict parsing --------------------------------------------------------

#: What the judge is allowed to say, and what each utterance means. Aliases are
#: listed explicitly rather than matched by prefix: a prefix rule would make
#: "acknowledge" mean ACT.
_VERDICT_WORDS = {
    "speak": SPEAK,
    "reply": SPEAK,
    "act": ACT_THEN_SPEAK,
    "act_then_speak": ACT_THEN_SPEAK,
    "silent": STAY_SILENT,
    "stay_silent": STAY_SILENT,
    "nothing": STAY_SILENT,
}


def parse_verdict(raw: str) -> Verdict:
    """The judge's answer as a verdict, or :class:`JudgeUndecided`.

    Only the first word counts, and it must be a known one. An empty answer is
    undecided rather than silence — a model that said nothing and a model that
    *decided* on nothing are the distinction this milestone exists to keep.
    """
    if not isinstance(raw, str):
        raise JudgeUndecided(f"judge answered a {type(raw).__name__}, not text")

    cleaned = "".join(c if (c.isalnum() or c == "_") else " " for c in raw.strip())
    words = cleaned.split()
    if not words:
        raise JudgeUndecided(f"judge answered nothing usable: {raw!r}")

    choice = _VERDICT_WORDS.get(words[0].lower())
    if choice is None:
        raise JudgeUndecided(
            f"judge answered {raw.strip()[:80]!r}; expected one of "
            f"{sorted(set(_VERDICT_WORDS))}"
        )
    return Verdict(choice=choice, raw=raw)


# --- prompts ----------------------------------------------------------------
# Deliberately the smallest thing that puts the three verdicts in front of the
# model. §Q11 leaves prompt content open on purpose: it needs a real corpus to
# argue from, which is what M1 exists to start producing.

_JUDGE_SYSTEM = (
    "You are omega, a second brain. Decide what this event deserves.\n"
    "Answer with exactly one word and nothing else:\n"
    "SPEAK - answer now, no work needed first\n"
    "ACT - do some work first, then answer\n"
    "SILENT - say nothing\n"
    "You have a body: you can read and write files, run read-only shell "
    "commands, and fetch a URL. ACT is the verdict that reaches them, and it "
    "is the only one that does. Choose ACT whenever answering well means "
    "looking something up on this machine, changing a file, or reading a "
    "page - do not guess at an answer you could go and check.\n"
    "If the person wrote to you, they are talking to you: answer them. "
    "SPEAK or ACT is right there even for a greeting, a short question, or "
    "something you think is obvious. Do not stay silent on a message addressed "
    "to you. A question about this machine, this project or a file on it is "
    "still a question for you; SILENT is never the right answer to a question.\n"
    "SILENT is for events nobody asked you about - your own idle ticks, "
    "background noise, things already handled. Staying silent there is a "
    "correct and successful answer, not a failure.\n"
    "Judge this event on its own. That you stayed silent before is not a "
    "reason to stay silent now."
)

_REPLY_SYSTEM = (
    "You are omega, a second brain. Answer the latest message directly and "
    "briefly. Do not narrate what you are doing.\n"
    "You have a body: you can read and write files, run read-only shell "
    "commands, and fetch a URL. Never tell the person you cannot reach their "
    "filesystem or the network - that is false - and never hand them a shell "
    "command to run themselves in place of doing it.\n"
    "You are not holding those tools in this particular message. If answering "
    "properly would need one, say what you would need to check. Do not guess "
    "an answer, and do not claim you are unable."
)


def _event_turn(
    ctx: TurnContext, *, images: bool, suffix: str = ""
) -> provider.Message:
    """The one user turn every role is given: recall, then the new event.

    ``images`` is the whole of DL-031's cheap/capable split at the prompt
    boundary. With it false the turn is a plain string and any attachment is
    named but not shown; with it true the attachments on *this* event — never
    the ones in recall — ride along as real bytes. `judge` passes false
    because it fires on every event and only needs to know a picture is there
    to route; `act` and `reply` pass true because they are the roles expected
    to answer about it.
    """
    # The new event gets its own, far looser budget (DL-039): it is the thing
    # omega was asked about, so cutting it is cutting the question. It is
    # bounded at all only because one pasted document should cost a degraded
    # turn rather than a failed request.
    event = _elide(_render_event(ctx.event), MAX_NEW_EVENT_CHARS)
    text = (
        f"{_learned_section(ctx.learned)}"
        f"Recent history:\n{_transcript(ctx.recalled)}\n\n"
        f"{_open_section(ctx.open_work)}"
        f"New event:\n{event}{suffix}"
    )
    parts = _image_parts(ctx) if images else []
    if not parts:
        return provider.user(text)
    return provider.user([provider.text_part(text), *parts])


def _judge_messages(ctx: TurnContext) -> list[provider.Message]:
    return [
        provider.system(_JUDGE_SYSTEM),
        _event_turn(ctx, images=False),
    ]


def _reply_messages(ctx: TurnContext, acted: ActResult) -> list[provider.Message]:
    work = f"\n\nWork done this turn: {', '.join(acted.tools)}" if acted.tools else ""
    return [
        provider.system(_REPLY_SYSTEM),
        _event_turn(ctx, images=True, suffix=work),
    ]


def _learned_section(learned: Sequence[derive.Claim]) -> str:
    """What omega has been taught that applies here, above the history (DL-042).

    **Above** it, not below, and that is the placement decision. These are
    standing instructions about *how to behave*, not facts about what happened,
    and recall's budget is twelve times this one — putting them after the
    transcript would bury a rule about tone under twenty-four thousand
    characters of conversation.

    Absent when nothing fires, for DL-041's reason: a heading that is usually
    empty teaches the model to skip it. It also means a person who has taught
    omega nothing gets a prompt that is **byte-identical** to the one before
    this existed, which is what keeps earlier eval numbers comparable.

    It states the claims and stops, per DL-033 — no explanation of what a
    learned claim is, no instruction about how strongly to weigh one. A prompt
    that explains its own mechanism is one the model reasons about instead of
    from.

    *Ordering is least-important-first, which is also the drop order.*
    ``_within`` drops from the front, so putting inferred claims before
    explicit ones and older before newer means that when the budget bites, what
    omega merely *inferred* goes before what the person actually **said** — the
    only ranking available here that is a fact rather than a guess, and the one
    DL-042 already gave meaning to. It has a second effect worth having: the
    claims the person stated sit nearest the event they apply to.
    """
    if not learned:
        return ""
    ordered = sorted(learned, key=lambda c: (c.explicit, c.seq))
    lines = [_elide(f"- {claim.text}", MAX_EVENT_CHARS, at=claim.seq) for claim in ordered]
    lines = _within(lines, MAX_LEARNED_CHARS, noun="learned item")
    body = "\n".join(lines)
    return f"What you have learned about working with this person:\n{body}\n\n"


def _open_section(open_work: Sequence[derive.OpenBlock]) -> str:
    """What's open, as its own block between history and the new event.

    Absent entirely when nothing is open, rather than present and empty. A
    standing heading with nothing under it teaches the model that the section is
    usually noise, and by the time it matters it has been trained to skip it.

    **It states the fact and stops.** No instruction to chase the answer, no
    explanation of what being blocked means for what it may do next. DL-033 is
    the reason: a prompt that explains a mechanism is a prompt the model reasons
    *about* rather than from, and the failure there was a model that talked
    around a gate instead of using it. What omega does about an unanswered
    question is a judgement made from the fact, so the fact is all that is
    supplied.

    A block still inside the recall window is rendered twice — once in the
    transcript as something omega asked, once here. That redundancy is the
    point: the transcript says the question was asked, and only this section
    says it was never answered. The two lines together are the distinction, and
    suppressing the duplicate would delete it.
    """
    if not open_work:
        return ""
    lines = [
        _elide(f"- {block.needs}", MAX_EVENT_CHARS, at=block.seq)
        for block in open_work
    ]
    lines = _within(lines, MAX_OPEN_CHARS, noun="question")
    body = "\n".join(lines)
    return f"You asked these and have had no answer yet:\n{body}\n\n"


def _transcript(recalled: Sequence[Pending]) -> str:
    """Recall rendered newest-last. Records the loop wrote are included: a turn
    that stayed silent is part of the history, and hiding it would make the
    model re-decide the same event without knowing it already answered.

    Nothing labels the lines. Ordering is what line order already says, and a
    seq number in the margin is scaffolding the model cannot tell apart from
    something the person typed — asked "which number did I talk about", it
    answered with the range of its own recall window. The new event is rendered
    unlabelled too, so history and the live line look alike.

    Both budgets live here rather than in the renderer's callers because this
    is the only place that sees the *whole* window; a per-line cap alone misses
    forty lines each just under it (DL-039).
    """
    lines = [
        _elide(_render_event(p.payload), MAX_EVENT_CHARS, at=p.seq)
        for p in recalled
    ]
    lines = _within(lines, MAX_RECALL_CHARS)
    return "\n".join(lines) if lines else "(nothing yet)"


def _within(lines: list[str], budget: int, noun: str = "event") -> list[str]:
    """Drop whole lines, oldest first, until the transcript fits.

    Dropping *whole* lines rather than shaving every line proportionally: a
    transcript of forty half-sentences is worse than one of twenty sentences,
    and the second at least leaves what survives intelligible.

    The drop is **announced**. A window that silently got shorter is the shape
    of a model confidently answering "you never mentioned that" — and it would
    be right about its context and wrong about the conversation, which is the
    worst combination available.

    ``noun`` only names what was dropped. It exists because this is shared with
    the what's-open section, where "earlier event(s) not shown" would describe
    the lines as history — and a dropped *question* read as a dropped event is
    exactly the misreading the announcement is here to prevent.
    """
    total = sum(len(line) + 1 for line in lines)
    if total <= budget:
        return lines
    dropped = 0
    while lines and total > budget:
        total -= len(lines[0]) + 1
        lines.pop(0)
        dropped += 1
    return [
        f"[{dropped} earlier {noun}(s) not shown here — they are in the log]",
        *lines,
    ]


def _elide(line: str, budget: int, *, at: Optional[int] = None) -> str:
    """Cut a rendered line to ``budget``, saying so and saying where.

    The note is the whole point. An elision that reads like the end of the
    sentence teaches the model the person stopped talking mid-thought; one that
    names the log position is a thing omega can be asked to go and fetch, which
    keeps this a *bounded view* of a complete record rather than a lossy one.
    """
    if len(line) <= budget:
        return line
    where = f", at seq {at} in the log" if at is not None else ", in the log"
    return (
        f"{line[:budget]}… "
        f"[{len(line) - budget:,} more characters not shown here{where}]"
    )


def _render_event(payload: dict[str, Any]) -> str:
    """One event as one line, plus a name for anything attached to it.

    The attachment names are appended here rather than at the one call site
    that can show pixels, because this function feeds *three* roles and the
    whole of recall. An attachment that only the capable role learned about
    would be invisible to the judge deciding whether to wake that role at all,
    and invisible again the next turn when the same event comes back through
    :func:`_transcript` — which is the shape of the bug DL-031 records.

    Pure, and deliberately so: no disk, no blob store. It runs ``RECALL_N``
    times per turn, and a renderer that reads files turns a text prompt into
    forty stat calls.
    """
    line = _render_payload(payload)
    notes = " ".join(_attachment_note(item) for item in _context_of(payload))
    return f"{line} {notes}" if notes else line


def _context_of(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Attachments on an event, tolerantly.

    ``episodes.inbound`` validates this list strictly on the way in, so a
    stored event has the shape below. This reads it defensively anyway: the
    log outlives the code that wrote it, and a renderer is the wrong place to
    discover that an old record is one field short.
    """
    context = payload.get("context")
    if not isinstance(context, list):
        return []
    return [item for item in context if isinstance(item, dict)]


def _attachment_note(item: dict[str, Any], reason: str = "") -> str:
    """How an attachment reads when it is named rather than shown.

    ``kind`` is the label, not the dispatch — what decides whether pixels get
    sent is ``mime`` (DL-031). Here the label is the useful half: "image" and
    "file" mean different things to a person, and the model is being told what
    the person thinks they sent.
    """
    kind = str(item.get("kind") or "attachment")
    title = str(item.get("title") or item.get("id") or "untitled")
    tail = f" — {reason}" if reason else ""
    return f"[{kind}: {title}{tail}]"


def _render_payload(payload: dict[str, Any]) -> str:
    kind = payload.get("kind")
    if kind == episodes.MESSAGE_INBOUND:
        return f"you: {payload.get('text', '')}"
    if kind == episodes.TURN_COMPLETED:
        outcome = payload.get("outcome")
        if outcome == "silent":
            return "omega: (stayed silent)"
        if outcome == "failed":
            return f"omega: (turn failed: {payload.get('error')})"
        return f"omega: {payload.get('reply', '')}"
    if kind == episodes.TURN_BLOCKED:
        return f"omega asked: {payload.get('needs', '')}"
    if kind == episodes.WORK_FINISHED:
        return f"work finished: {payload.get('summary', '')}"
    if kind == episodes.TOOL_CALLED:
        return f"tool {payload.get('tool')} called"
    if kind == episodes.TOOL_RETURNED:
        ok = "ok" if payload.get("ok") else f"failed: {payload.get('error')}"
        return f"tool {payload.get('tool')} {ok}"
    return str(kind)


#: Caps on what travels as pixels (DL-031). A starting guess, not a
#: measurement: a retina screenshot lands around 2–5 MiB, so one of them fits
#: and a photo library does not. The per-turn total exists because ten images
#: can each pass the per-image check and blow the budget together.
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_TURN_IMAGE_BYTES = 10 * 1024 * 1024


def _is_image(item: dict[str, Any]) -> bool:
    """Does this attachment have pixels omega could show?

    Decided by ``mime`` and the presence of a blob, never by ``kind``. ``kind``
    is the word a client chose for a tray row; ``mime`` is a fact about the
    bytes, recorded for the same reason DL-027 keeps the media type in the
    episode instead of in a filename anything on the box could rename. A
    ``screen`` capture is an image; a ``file`` that happens to be a PNG is too.
    """
    mime = str(item.get("mime") or "").lower()
    return mime.startswith("image/") and bool(item.get("blob"))


def _image_parts(ctx: TurnContext) -> list[provider.Part]:
    """Attachments on the new event, as content blocks for the model.

    Every failure here degrades to a text part that *says* what could not be
    shown, and none of them fails the turn. An oversized screenshot, a blob
    missing from the store, an unreadable file — each is a thing omega should
    tell the person it cannot show, not a turn that dies on the way to the
    model. The degraded line is also visible in the log afterwards, which the
    silent drop this replaces never was.

    Only the new event's images. Recall renders them as names via
    :func:`_render_event`, because ``RECALL_N`` is 40 and re-sending every
    picture every turn makes the cost of a conversation grow with its length.
    """
    items = [item for item in _context_of(ctx.event) if _is_image(item)]
    if not items:
        return []

    try:
        store = blobs.BlobStore.open(ctx.queue.store.root)
    except Exception as exc:  # noqa: BLE001
        return [
            provider.text_part(_attachment_note(item, f"not shown: {exc}"))
            for item in items
        ]

    parts: list[provider.Part] = []
    remaining = MAX_TURN_IMAGE_BYTES
    for item in items:
        part, spent = _one_image(store, item, remaining)
        parts.append(part)
        remaining -= spent
    return parts


def _one_image(
    store: blobs.BlobStore, item: dict[str, Any], remaining: int
) -> tuple[provider.Part, int]:
    """One attachment as a part, and what it spent of the turn's budget.

    The size that counts is the one on disk, not the ``bytes`` field the sender
    declared. `CLAUDE.md` grades the world rather than the words, and here the
    two really can differ: the field is a claim made by whatever built the
    event, and the budget is being spent on the actual file.
    """
    digest = str(item.get("blob") or "")
    mime = str(item.get("mime") or "application/octet-stream")
    try:
        path = store.path_for(digest)
        size = path.stat().st_size
    except Exception:  # noqa: BLE001
        return provider.text_part(_attachment_note(item, "not shown: missing")), 0

    if size > MAX_IMAGE_BYTES:
        return provider.text_part(_attachment_note(item, "not shown: too large")), 0
    if size > remaining:
        return (
            provider.text_part(
                _attachment_note(item, "not shown: too many images this turn")
            ),
            0,
        )

    try:
        raw = path.read_bytes()
    except OSError as exc:
        return provider.text_part(_attachment_note(item, f"not shown: {exc}")), 0

    encoded = base64.b64encode(raw).decode("ascii")
    return provider.image_part(f"data:{mime};base64,{encoded}"), size
