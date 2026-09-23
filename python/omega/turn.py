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

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

from omega import episodes, provider
from omega.queue import EVENT_KINDS, EventQueue, Pending

__all__ = [
    "SPEAK",
    "ACT_THEN_SPEAK",
    "STAY_SILENT",
    "VERDICTS",
    "RECALL_N",
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
    """

    tools: tuple[str, ...] = ()
    blocked_on: Optional[str] = None
    error: Optional[str] = None
    stop_reason: str = ""


@dataclass(frozen=True)
class TurnContext:
    """Everything a step is allowed to see. Passed rather than imported, so a
    test can drive any step without a process-wide provider."""

    seq: int
    event: dict[str, Any]
    recalled: Sequence[Pending]
    queue: EventQueue
    complete: Callable[..., provider.Response]


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
    "If the person wrote to you, they are talking to you: answer them. "
    "SPEAK or ACT is right there even for a greeting, a short question, or "
    "something you think is obvious. Do not stay silent on a message addressed "
    "to you.\n"
    "SILENT is for events nobody asked you about - your own idle ticks, "
    "background noise, things already handled. Staying silent there is a "
    "correct and successful answer, not a failure.\n"
    "Judge this event on its own. That you stayed silent before is not a "
    "reason to stay silent now."
)

_REPLY_SYSTEM = (
    "You are omega, a second brain. Answer the latest message directly and "
    "briefly. Do not narrate what you are doing."
)


def _judge_messages(ctx: TurnContext) -> list[provider.Message]:
    return [
        provider.system(_JUDGE_SYSTEM),
        provider.user(
            f"Recent history:\n{_transcript(ctx.recalled)}\n\n"
            f"New event:\n{_render_event(ctx.event)}"
        ),
    ]


def _reply_messages(ctx: TurnContext, acted: ActResult) -> list[provider.Message]:
    work = f"\n\nWork done this turn: {', '.join(acted.tools)}" if acted.tools else ""
    return [
        provider.system(_REPLY_SYSTEM),
        provider.user(
            f"Recent history:\n{_transcript(ctx.recalled)}\n\n"
            f"New event:\n{_render_event(ctx.event)}{work}"
        ),
    ]


def _transcript(recalled: Sequence[Pending]) -> str:
    """Recall rendered newest-last. Records the loop wrote are included: a turn
    that stayed silent is part of the history, and hiding it would make the
    model re-decide the same event without knowing it already answered.

    Nothing labels the lines. Ordering is what line order already says, and a
    seq number in the margin is scaffolding the model cannot tell apart from
    something the person typed — asked "which number did I talk about", it
    answered with the range of its own recall window. The new event is rendered
    unlabelled too, so history and the live line look alike.
    """
    lines = [_render_event(p.payload) for p in recalled]
    return "\n".join(lines) if lines else "(nothing yet)"


def _render_event(payload: dict[str, Any]) -> str:
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
