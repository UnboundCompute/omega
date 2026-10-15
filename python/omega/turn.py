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
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Callable, Optional, Sequence

from omega import blobs, derive, episodes, grants as granted, learn, lessons as self_taught, notice, provider, schedule
from omega import delegate, leave, mcp, slack
from omega import location as whereabouts
from omega import tasks as _tasks
from omega import outbox
from omega.tools import (
    INBOX_SOURCE,
    MAIL,
    NOTHING_NEW,
    FOLLOW_UP,
    REPORT,
    FOLLOWUP_ASK,
    PROMISE_ASK,
    asks,
    is_brief,
    is_reminder,
    is_research,
    is_tell,
    is_watch,
    fence,
    fence_open,
    must_speak,
    told,
)
from omega.queue import EVENT_KINDS, EventQueue, Pending

__all__ = [
    "SPEAK",
    "ACT_THEN_SPEAK",
    "STAY_SILENT",
    "VERDICTS",
    "RECALL_N",
    "NOT_RECALLED",
    "MAX_EVENT_CHARS",
    "MAX_NEW_EVENT_CHARS",
    "MAX_RECALL_CHARS",
    "MAX_OPEN_CHARS",
    "MAX_LEARNED_CHARS",
    "MAX_ACT_PASSES",
    "UNLOOKED",
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

#: Kinds that never take a recall slot. A sense report is what the Mac read, not
#: something said (DL-072); presence alone arrives every five minutes while the
#: person works, so counting reports would push the conversation out of
#: ``RECALL_N`` within hours. What the reports taught arrives as claims instead.
#: The finance ledger (DL-083) is out for the same reason: one record per bank
#: alert would crowd the conversation out, and the `finance` tool reads it.
NOT_RECALLED = frozenset({
    episodes.SENSE_REPORTED,
    episodes.FINANCE_STARTED,
    episodes.FINANCE_RECORDED,
    episodes.FINANCE_SKIPPED,
    # Memory hygiene (DL-087 #4): bookkeeping about claims, not conversation.
    # The question a contradiction asks is already in recall, inside the reply
    # that carried it.
    episodes.CLAIM_MERGED,
    episodes.CLAIM_STALE,
    episodes.CLAIM_CONTRADICTION,
    episodes.CLAIM_ASKED,
    episodes.CLAIM_RESOLVED,
    episodes.MEMORY_TIDIED,
    # Where a coding project was left (DL-087 #7): derived from a digest, and
    # read through the `projects` tool, not replayed as conversation.
    episodes.PROJECT_STATE,
    # Standing grants (DL-085) reach the prompt through their own section,
    # read off the fold, rather than as a line each in the conversation.
    *episodes.GRANT_KINDS,
    # A withdrawn proposal (DL-087 #8) reaches them as one line of the next
    # brief, read off the fold, not as a line of conversation.
    episodes.GRANT_WITHDRAWN,
    # Follow-ups on omega's own failures (DL-087 #2): the tell is already in
    # the conversation as its own inbound; the record is bookkeeping.
    *episodes.FAILURE_KINDS,
    # Lost sign-ins (DL-087 #3): the tell is its own inbound, the brief names
    # what is still lost; the records are bookkeeping.
    *episodes.ACCESS_KINDS,
    # Promises they made (DL-087 #1): their words are already in the
    # conversation as their own message; the brief names the open ones.
    *episodes.PROMISE_KINDS,
    # Lessons and skills (DL-086) reach the judge and act prompts through
    # their own section, read off the fold, never as lines of conversation.
    *episodes.LESSON_KINDS,
    # Where they are (DL-091): never conversation, never a thing to learn. A
    # prompted turn sees it as one line, read off the fold.
    *episodes.LOCATION_KINDS,
    # Picking the moment (DL-098): a held draft is shown to the look and the
    # thinking turn in their own situation text, and a delivered one is
    # already in the conversation as the reply that carried it.
    *episodes.OUTBOX_KINDS,
    # Topics kept up with (DL-087 #6): their news reaches them in the brief,
    # read off the fold; the records are bookkeeping.
    *episodes.TOPIC_KINDS,
    # A task's records (DL-095): derived-view state, read off the fold into the
    # brief's running/waiting section. Each step's reply is already in the
    # conversation; the task.* records are bookkeeping, not conversation.
    *episodes.TASK_KINDS,
    # Page watches (DL-096): an alert reaches them as a tell, which is
    # conversation; the set, the checks and the stops are bookkeeping, and
    # the watch tool lists them off the fold when asked.
    *episodes.WATCHER_KINDS,
    # The leave-now alarm (DL-093): every check, ping and close. A ping
    # reaches the conversation as its own tell; a check is one of up to a
    # dozen per alarm and would crowd out what they said. ``leave.set`` is
    # recalled: one per request, beside the message that asked for it, and
    # rendered as its kind alone, so a later turn sees an alarm was set
    # without where they are.
    episodes.LEAVE_CHECKED,
    episodes.LEAVE_PINGED,
    episodes.LEAVE_CANCELLED,
    episodes.LEAVE_DONE,
    # Sharing to Slack (DL-101): a post reaches them as the reply that
    # carries its link, which is conversation; the records are bookkeeping a
    # delete reads off the fold, and a failure is the turn's own tool result.
    *episodes.SLACK_KINDS,
    # Handing work to Claude Code (DL-104): the brief is long and lives in
    # its record; every report reaches them as a tell, which is conversation
    # and carries the result's words and its `[attached: ...]` files, so a
    # later "post that to Slack" recalls the tell, not the bookkeeping.
    *episodes.DELEGATE_KINDS,
})

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
    #: What code says after the reply, whatever the model wrote (DL-087 #8):
    #: the question a proposal asks, with its id and how to answer it, and
    #: "that one's closed" for an answer that came too late. Never blocks.
    told: str = ""
    #: The untrusted content the turn read (DL-064), by source: a URL, an MCP
    #: server, their inbox. Empty for a clean turn, which is every turn that
    #: read none.
    tainted: tuple[str, ...] = ()


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

    #: The *whole* learned set, which ``learned`` is a subset of. It reaches no
    #: prompt — rendering ninety-one claims on every turn is the memory
    #: firehose pointed inward (DL-060) — and exists so the `recall` tool can
    #: answer from what the turn already holds rather than reopening a store
    #: this process has locked.
    known: Sequence[derive.Claim] = ()

    #: Schedules standing when this turn began, so the act loop's ``unwatch``
    #: can name what it is stopping (DL-068) and ``schedules`` can list them
    #: from the same snapshot (DL-077).
    running: Sequence[schedule.Schedule] = ()

    #: The standing grants as of this turn (DL-085), a frozen
    #: :class:`omega.grants.State`. ``None`` is none at all, and renders as
    #: nothing: a turn with no grants reads exactly as it did before them.
    grants: Any = None

    #: Omega's own failures as of this turn (DL-087 #2), a frozen
    #: :class:`omega.followups.State`. ``None`` or empty renders as nothing,
    #: so a turn over a log with no failures reads exactly as before.
    followups: Any = None

    #: Sign-ins and access lost as of this turn (DL-087 #3), a frozen
    #: :class:`omega.access.State`. ``None`` or one with nothing lost renders
    #: as nothing, so a brief over a log with no refusals reads as before.
    access: Any = None

    #: The promises they made, as of this turn (DL-087 #1), a frozen
    #: :class:`omega.promises.State`. ``None``, or one with no live grant,
    #: renders as nothing: a turn without the grant reads exactly as before.
    promises: Any = None

    #: omega's own lessons and skills as of this turn (DL-086), a frozen
    #: :class:`omega.lessons.State`. ``None``, or one with none standing,
    #: renders as nothing and offers nothing: a turn before the first review
    #: reads exactly as before.
    lessons: Any = None

    #: Where they are, as of this turn (DL-091), a frozen
    #: :class:`omega.location.State`. ``None`` or one with no fix renders as
    #: nothing, and so does every unprompted turn: a turn over a log with no
    #: location reads exactly as before.
    location: Any = None
    #: The topics kept up with, as of this turn (DL-087 #6), a frozen
    #: :class:`omega.topics.State`. ``None``, or one with nothing followed and
    #: no grant, renders as nothing and offers nothing.
    topics: Any = None
    #: The tasks as of this turn (DL-095), a frozen :class:`omega.tasks.State`.
    #: ``None``, or one with nothing running, renders as nothing in the brief
    #: and offers nothing, so a log with no tasks reads exactly as before.
    tasks: Any = None


#: Channels whose events nobody typed: omega's own look (DL-061) and the clock
#: (DL-035). Only on these may a turn that checked end in silence (DL-068 #2).
UNPROMPTED_CHANNELS = notice.UNPROMPTED_CHANNELS


def unprompted(ctx: "TurnContext") -> bool:
    """Did nobody type this? True on the unprompted channels, and for any
    event that is not an inbound at all: work that finished (DL-085's
    research report) is omega's own, and a turn it causes must never pass
    for the person's message — that is what lets ``grant`` refuse it."""
    if ctx.event.get("kind", episodes.MESSAGE_INBOUND) != episodes.MESSAGE_INBOUND:
        return True
    return str(ctx.event.get("channel", "")) in UNPROMPTED_CHANNELS


#: What a turn opening its own mail reads back in recall (`_mail_digest`):
#: one line of From and Subject, which the sender wrote.
RECALLED_MAIL = "a mail opened earlier"


def read_untrusted(ctx: "TurnContext") -> tuple[str, ...]:
    """The untrusted content a turn holds before it calls anything (DL-064).

    A look that listed their mail, as its own event or recalled, and a mail
    opened in a recalled turn, whose From and Subject recall renders. Read
    from structure only: the look's text is omega's own (its inbox block is
    fenced by :func:`omega.notice.situation`), and a person's message is never
    inspected, so nothing they type can taint or clean a turn.
    """
    found: list[str] = []
    marker = fence_open(INBOX_SOURCE)
    for payload in [ctx.event, *(p.payload for p in ctx.recalled)]:
        kind = payload.get("kind")
        if (
            kind == episodes.MESSAGE_INBOUND
            and str(payload.get("channel", "")) == notice.CHANNEL
            and marker in str(payload.get("text", ""))
        ):
            found.append(INBOX_SOURCE)
        elif kind == episodes.TOOL_RETURNED and payload.get("ok") and payload.get("tool") == MAIL:
            found.append(RECALLED_MAIL)
        elif kind == episodes.MESSAGE_INBOUND and payload.get("untrusted"):
            # A tell carrying a watched page's words (DL-096), as it says.
            found.append(str(payload["untrusted"]))
    return tuple(dict.fromkeys(found))


def is_look(ctx: "TurnContext") -> bool:
    """Is this omega's own look rather than something that arrived?"""
    return str(ctx.event.get("channel", "")) == notice.CHANNEL


def look_reason(ctx: "TurnContext", verdict: "Verdict") -> str:
    """The reason a look's judge gave after its first word, or ``""``."""
    if not is_look(ctx):
        return ""
    raw = (verdict.raw or "").strip()
    parts = raw.split(None, 1)
    return parts[1].strip(" :-\n") if len(parts) > 1 else ""


def _judged(verdict: Optional["Verdict"]) -> Optional[str]:
    """What the judge said, for the record (DL-076), capped like a log line."""
    if verdict is None or not isinstance(verdict.raw, str) or not verdict.raw.strip():
        return None
    return " ".join(verdict.raw.split())[:300]


#: The error a turn fails with when it said "nothing new" having looked at
#: nothing (DL-089 #2).
UNLOOKED = "answered nothing new without looking"


def quiet_allowed(ctx: "TurnContext", text: str, acted: "ActResult") -> bool:
    """May this reply end the turn in silence? Only an unprompted check that
    found nothing, never a reminder or a brief coming due (DL-075, DL-080),
    and only after a real look: at least one tool call that came back
    (DL-089). ``acted.tools`` lists exactly those, the calls whose
    ``tool.returned`` was ok, hosted searches included; a refused or failed
    call is not a look.

    An unprompted turn that answers the sentinel without having looked is not
    an all-clear, and it is not silence either: it raises, so the turn is
    recorded as failed with :data:`UNLOOKED` in its error.

    A research follow-up may be quiet only once it has set the next one or
    reported (DL-085): otherwise silence would end their task with nobody
    told, so it speaks instead.
    """
    if not (unprompted(ctx) and not must_speak(ctx.event) and nothing_new(text)):
        return False
    if is_research(ctx.event) and not ({FOLLOW_UP, REPORT} & set(acted.tools)):
        return False
    if not acted.tools:
        raise RuntimeError(UNLOOKED)
    return True


def nothing_new(text: str) -> bool:
    """Is this the sentinel, allowing for the quoting and punctuation a model
    wraps around an answer it was told to give exactly?"""
    core = text.strip().strip("`*\"'. ").lower()
    return core == NOTHING_NEW.strip("()").lower() or core == NOTHING_NEW.lower()


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
    return queue.recent(n, before=before, skipping=NOT_RECALLED)


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
    told_ = acted.told or None
    if unprompted(ctx) and REPORT in acted.tools:
        # The report is its own event, and the turn it causes tells them
        # (DL-085): saying it here too would tell them twice.
        return told_
    if acted.text is not None and acted.text.strip():
        if quiet_allowed(ctx, acted.text, acted):
            # Checked, and found nothing worth interrupting them for. On a
            # channel nobody typed on that is a decision, not an empty answer.
            # A proposal it filed is still asked: that is not news, it is a
            # question (DL-087 #8).
            return told_
        if told_:
            return f"{acted.text}\n\n{told_}"
        # The sub-loop already answered, holding every tool result. Composing
        # again here would ask a model that cannot see any of them to describe
        # work it did not watch — which is exactly how omega came to write a
        # file and then say it could not (see :class:`ActResult`). The cheapest
        # correct reply is the one already written.
        return acted.text
    if told_:
        # Code already has the words (a proposal's question, a closed
        # answer); a model asked to restate them could only lose the id.
        return told_
    response = ctx.complete(
        provider.ACT, _reply_messages(ctx, acted, look_reason(ctx, verdict))
    )
    if quiet_allowed(ctx, response.text, acted):
        return None
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
    usage: Optional[list[dict[str, Any]]] = None,
    judged: Optional[str] = None,
    at: Optional[str] = None,
    tainted: Sequence[str] = (),
    reply_to: Optional[str] = None,
) -> int:
    """Step 6 — append the turn's terminal record and return its seq.

    ``reply_to`` is the channel a reply goes back to when the event names one
    (DL-104: a Claude Code session's result goes where it was asked for);
    absent, the record is the one it always was.

    ``tainted`` is what untrusted content the turn read (DL-064); empty, the
    record is the one it always was.

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
        payload = episodes.blocked(
            for_seq=seq, needs=needs, usage=usage, at=at, tainted=list(tainted)
        )
    else:
        asked, settled = _asked_and_settled(queue, seq)
        payload = episodes.completed(
            for_seq=seq,
            outcome=outcome,
            reply=reply_text,
            tools=list(tools),
            error=error,
            usage=usage,
            judged=judged,
            at=at,
            tainted=list(tainted),
            # Only a reply can carry a question to answer; a silent or failed
            # turn's proposal still waits, and is asked by the next brief.
            proposals=asked if outcome == "spoke" else None,
            settled=settled,
            reply_to=reply_to if outcome == "spoke" else None,
        )
    return queue.append(payload, episodes.turn_write_key(seq))


def _asked_and_settled(
    queue: EventQueue, seq: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The proposals turn ``seq`` asked and the ones it answered (DL-103),
    read from the records it wrote, never from its words.

    A proposal names the turn that asked it (``grant.proposed.for_seq``),
    whether act filed it, the task tool did, or the executor filed it beside
    the tell this turn speaks; an answer names the turn it came in
    (``grant.answered.for_seq``). Both land after the turn's own event and
    before its terminal record, so the records since ``seq`` hold them all.
    """
    asked: list[dict[str, Any]] = []
    settled: list[dict[str, Any]] = []
    for episode in queue.store.episodes_since(seq):
        payload = episodes.decode(episode.payload)
        if payload.get("for_seq") != seq:
            continue
        kind = payload.get("kind")
        if kind == episodes.GRANT_PROPOSED:
            proposal = granted.proposal_of(episode.seq, payload)
            asked.append({"id": proposal.pid, "summary": granted.button_summary(proposal)})
        elif kind == episodes.GRANT_ANSWERED:
            settled.append({"id": granted.pid(int(payload["proposal"])), "yes": bool(payload["yes"])})
    return asked, settled


# --- the turn ---------------------------------------------------------------


def _teach(
    ctx: TurnContext,
    pending: Pending,
    *,
    known: Sequence[derive.Claim],
    running: Sequence[schedule.Schedule],
    at: Optional[str],
) -> Optional[str]:
    """Extract a teach drop and return its receipt, or ``None`` if it was not one.

    **Never raises.** DL-043 #5: the reply is already composed and is real
    work, and losing it to a failed second call would make teaching strictly
    worse than talking. A failure becomes a receipt that says so, which is the
    one place the person can act on it, and the turn still records ``spoke``.
    This is the only step in the turn whose failure is not
    ``outcome: "failed"``.
    """
    note = learn.teaching_note(str(pending.payload.get("text", "")))
    if note is None:
        return None

    where = str(pending.payload.get("channel", "")).strip()
    when = str(pending.payload.get("at", "")).strip()
    context = " ".join(
        part
        for part in (
            f"The note arrived on the {where} channel." if where else "",
            f"It is {when}." if when else "",
        )
        if part
    )
    try:
        extracted = learn.extract(
            ctx.complete,
            note=note,
            known=known,
            running=running,
            context=context,
        )
        # Cancels first: a note that retires one reminder and sets another in
        # its place should not leave both standing for the width of this
        # window, because a crash inside it would leave the old one firing
        # forever beside the new one.
        stopped = learn.cancel_schedules(
            ctx.queue, extracted.cancel, running=running, at=at
        )
        # Retractions next, for the same reason and one more: a note that
        # forgets one claim and teaches a replacement must not leave both
        # active, because the window between them is exactly where omega would
        # hold two contradictory beliefs and act on whichever rendered first.
        forgotten = learn.retract_claims(
            ctx.queue, extracted.retract, for_seq=pending.seq, known=known, at=at
        )
        written = learn.file_claims(
            ctx.queue,
            extracted.claims,
            for_seq=pending.seq,
            source_seq=pending.seq,
            at=at,
        )
        scheduled = learn.file_schedules(
            ctx.queue, extracted.schedules, source_seq=pending.seq, at=at
        )
    except Exception as exc:  # noqa: BLE001 - see the docstring
        reason = str(exc) or type(exc).__name__
        try:
            ctx.queue.append(
                episodes.claim_extraction_failed(for_seq=pending.seq, reason=reason, at=at)
            )
        except Exception:  # noqa: BLE001
            # The record is the diagnosis, not the work. If even the append
            # fails the receipt below still reaches the person, and turning a
            # failed *write about* a failure into a failed turn would lose the
            # reply this function exists to protect (DL-043 #5).
            pass
        return learn.receipt((), error=reason)
    return learn.receipt(
        written,
        known=known,
        scheduled=scheduled,
        stopped=stopped,
        forgotten=forgotten,
    )


def run_turn(
    queue: EventQueue,
    pending: Pending,
    *,
    complete: Optional[Callable[..., provider.Response]] = None,
    act: Callable[[TurnContext], ActResult] = no_act_loop_yet,
    recall_n: int = RECALL_N,
    open_work: Sequence[derive.OpenBlock] = (),
    learned: Sequence[derive.Claim] = (),
    known: Sequence[derive.Claim] = (),
    running: Sequence[schedule.Schedule] = (),
    ask: Sequence[tuple[int, str]] = (),
    grants: Any = None,
    followups: Any = None,
    at: Optional[str] = None,
    promises: Any = None,
    lessons: Any = None,
    access: Any = None,
    location: Any = None,
    topics: Any = None,
    tasks: Any = None,
    gate: Any = None,
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

    ``known`` is the *whole* learned set and is used for two things that do not
    touch each other: deciding what a newly taught claim replaces (DL-043 #6),
    and answering the `recall` tool (DL-060). It is separate from ``learned``
    because those and ``learned`` ask opposite questions — ``learned`` is what
    fires on this event, a claim can only be contradicted by one that was never
    going to fire alongside it, and *what do you remember about me* is a
    question about the set rather than about the turn.

    ``gate`` is the outbox's decision for a non-urgent unprompted turn
    (DL-098, :class:`omega.outbox.Gate`), or ``None`` for every other turn,
    which then runs exactly as before. A turn it holds ends silent with its
    words written down as a draft; one it lets speak is counted.
    """
    # Metered here, once, so every model call the turn makes — judge, act
    # loop, reply and teach extraction all go through `ctx.complete` — lands
    # on the one terminal record the turn writes, whichever path ends it
    # (DL-070).
    meter = provider.Meter(complete or provider.complete)
    # DL-084 #3: one retry for a judge or act call that blipped in transport,
    # counted on the meter so the record says it happened.
    complete = provider.retry_blips(meter, on_retry=meter.retried)
    event = perceive(pending)
    if is_tell(event):
        # DL-084 #1, #4: omega telling them about itself — a promise it could
        # not keep, or an outage. Delivered as written, with no model call:
        # what it reports is often that the model could not be reached.
        words = told(event)
        if asks(event):
            # DL-087 #2, #1: a follow-up that proposes pausing a watch, or a
            # promise's reminder, is a question. Filed, not blocked (DL-087
            # #8): it is said with its id and how to answer in place of the
            # bare "Reply yes or no.", and a plain "yes" is still read against
            # it while it is the newest thing omega asked.
            asked = _proposal_for(queue, pending.seq)
            if asked is not None and words.endswith(_REPLY_YES_NO):
                words = (
                    words[: -len(_REPLY_YES_NO)]
                    + f" {granted.how_to_answer(asked)}"
                )
        if gate is not None and outbox.settle(queue, pending.seq, gate, words):
            record = write_memory(queue, seq=pending.seq, outcome="silent", at=at)
            return TurnResult(seq=pending.seq, outcome="silent", record_seq=record)
        record = write_memory(
            queue, seq=pending.seq, outcome="spoke", reply_text=words, at=at,
            # A watched page's words (DL-096): the record says whose, so
            # recall fences the reply as it fences the tell.
            tainted=(str(event["untrusted"]),) if event.get("untrusted") else (),
            reply_to=event.get("reply_to"),
        )
        return TurnResult(
            seq=pending.seq, outcome="spoke", record_seq=record, reply=words
        )
    if self_taught.is_review(event):
        return _review(
            queue, pending, complete, meter, lessons=lessons, at=at, gate=gate
        )
    recalled = recall(queue, before=pending.seq, n=recall_n)
    ctx = TurnContext(
        seq=pending.seq,
        event=event,
        recalled=recalled,
        queue=queue,
        complete=complete,
        open_work=tuple(open_work),
        learned=tuple(learned),
        known=tuple(known),
        running=tuple(running),
        grants=grants,
        followups=followups,
        promises=promises,
        lessons=lessons,
        access=access,
        location=location,
        topics=topics,
        tasks=tasks,
    )

    verdict: Optional[Verdict] = None
    acted = ActResult()
    # DL-064: what untrusted content the turn read lasts the whole turn and
    # lands on its record — what it held before acting, and what acting read.
    held = read_untrusted(ctx)

    def tainted() -> tuple[str, ...]:
        return tuple(dict.fromkeys((*held, *acted.tainted)))

    try:
        verdict = judge(ctx)
        if verdict.choice != ACT_THEN_SPEAK and not unprompted(ctx) and granted.answers(
            ctx.grants, ctx.event, ctx.open_work
        ):
            # Their yes or no to a proposal (DL-085 #3). The act step is where
            # an answer is settled, so it must run whatever the judge saw.
            verdict = Verdict(choice=ACT_THEN_SPEAK, raw=verdict.raw)
        elif verdict.choice != ACT_THEN_SPEAK and is_research(ctx.event):
            # A research follow-up carries their task on: like a brief, it
            # has looking to do, and a judge that let it lapse would drop the
            # task with nobody told (DL-085 #5).
            verdict = Verdict(choice=ACT_THEN_SPEAK, raw=verdict.raw)
        elif (
            verdict.choice == STAY_SILENT
            and ctx.event.get("kind") == episodes.WORK_FINISHED
        ):
            # Work that finished was asked for, so it is told. It used to read
            # as their message, which had the same effect; now that it does
            # not (see `unprompted`), it says so here.
            verdict = Verdict(choice=SPEAK, raw=verdict.raw)
        elif verdict.choice == STAY_SILENT and is_reminder(ctx.event):
            # A reminder coming due is a promise, and the prompt saying "never
            # SILENT" was not enough: live, the judge chose silence on a 6pm
            # walk reminder and nobody heard it (DL-075). Code keeps the promise
            # the prompt only asked for; SPEAK, because a judge that saw nothing
            # to do saw nothing to look up. ``raw`` keeps what it said.
            verdict = Verdict(choice=SPEAK, raw=verdict.raw)
        elif verdict.choice == STAY_SILENT and not unprompted(ctx):
            # Their own message never ends silent (DL-084 #2). The prompt says
            # so; live, "keep checking ... and let me know" was still answered
            # with nothing, twice. SPEAK, as for a reminder; ``raw`` keeps
            # what the judge said.
            verdict = Verdict(choice=SPEAK, raw=verdict.raw)
        elif verdict.choice != ACT_THEN_SPEAK and is_brief(ctx.event):
            # A brief is the same promise with looking built in (DL-080): a
            # brief that only spoke would brief them from nothing, so any
            # verdict but ACT becomes ACT.
            verdict = Verdict(choice=ACT_THEN_SPEAK, raw=verdict.raw)
        elif verdict.choice != ACT_THEN_SPEAK and is_watch(ctx.event):
            # A watch is a promise to look (DL-089). Live, calendar and inbox
            # watches were judged a bare SILENT in 1-2s with no tool call, so
            # they never opened what they were watching. Any verdict but ACT
            # becomes ACT, as for a brief; the watch may still end quiet, but
            # only after it has looked. ``raw`` keeps what the judge said.
            verdict = Verdict(choice=ACT_THEN_SPEAK, raw=verdict.raw)
        elif verdict.choice != ACT_THEN_SPEAK and _tasks.obligation_of(ctx.event) is not None:
            # A driven task step carries an obligation (DL-095): it must do the
            # step and end by calling `step`. A judge that let it speak would
            # be the DL-040 gate-leak — work that owed a gated action ending in
            # prose. ACT, so the step's own tool is reached; the executor still
            # checks afterwards that it was called.
            verdict = Verdict(choice=ACT_THEN_SPEAK, raw=verdict.raw)
        if verdict.choice == ACT_THEN_SPEAK:
            acted = act(ctx)
            if acted.blocked_on:
                waiting = _brief_tail(ctx)
                if waiting:
                    # A brief that stops on a "May I?" still lists them.
                    acted = replace(acted, blocked_on=f"{acted.blocked_on}\n\n{waiting}")
                record = write_memory(
                    queue,
                    seq=pending.seq,
                    outcome="blocked",
                    needs=acted.blocked_on,
                    usage=meter.usage(),
                    at=at,
                    tainted=tainted(),
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
            usage=meter.usage(),
            judged=_judged(verdict),
            at=at,
            tainted=tainted(),
        )
        return TurnResult(
            seq=pending.seq,
            outcome="failed",
            record_seq=record,
            tools=tuple(acted.tools),
            error=error,
            verdict=verdict,
        )

    # DL-043: a teach drop is extracted here, between the reply and the record.
    # After the `except` because a turn that already failed has nothing to show
    # a receipt with; before `write_memory` because `claim.extracted` is not
    # projected (DL-042), so the receipt has nowhere to ride but the one reply
    # this turn is about to record.
    receipt = _teach(ctx, pending, known=known, running=running, at=at)

    brief = _brief_tail(ctx)
    if brief:
        # DL-087 #8: every proposal still waiting on them, and each withdrawn
        # since the last brief, listed by code so none is ever dropped; and
        # (#6) what is new on the topics kept up with.
        text = brief if text is None else f"{text}\n\n{brief}"

    if text is None and receipt is None:
        # Silence. A success, recorded as one, with `reply` null rather than
        # empty so that "said nothing" can never be read as "said ''".
        if gate is not None:
            outbox.dropped(queue, pending.seq, gate)
        record = write_memory(
            queue,
            seq=pending.seq,
            outcome="silent",
            tools=acted.tools,
            usage=meter.usage(),
            judged=_judged(verdict),
            at=at,
            tainted=tainted(),
        )
        return TurnResult(
            seq=pending.seq,
            outcome="silent",
            record_seq=record,
            tools=tuple(acted.tools),
            verdict=verdict,
        )

    if receipt is not None:
        # The receipt is not conversation, so it does not replace a reply that
        # exists — and when the judge chose silence on a teach drop it *is* the
        # reply, because a teach omega says nothing about is indistinguishable
        # from one it dropped, which is the failure DL-034's done bar exists to
        # catch. The judge still governs every other turn.
        text = receipt if text is None else f"{text}\n\n{receipt}"

    question = _question_for(ctx, ask, running) if text is not None else None
    if question is not None:
        # DL-087 #4: a contradiction is asked once, folded into a turn that is
        # already speaking unprompted — never an interrupt of its own. The
        # record lands before the terminal one, like a teach receipt's claims,
        # and it is what stops the same pair being asked again.
        contradiction, words = question
        queue.append(
            episodes.claim_asked(
                for_seq=pending.seq, contradiction_seq=contradiction, at=at
            )
        )
        text = f"{text}\n\n{words}"

    if gate is not None and outbox.settle(queue, pending.seq, gate, text):
        # DL-098: not now. The words wait as a draft for a better moment;
        # the turn itself ends quiet, with what the judge said kept.
        record = write_memory(
            queue,
            seq=pending.seq,
            outcome="silent",
            tools=acted.tools,
            usage=meter.usage(),
            judged=_judged(verdict),
            at=at,
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
        usage=meter.usage(),
        judged=_judged(verdict),
        at=at,
        tainted=tainted(),
    )
    return TurnResult(
        seq=pending.seq,
        outcome="spoke",
        record_seq=record,
        reply=text,
        tools=tuple(acted.tools),
        verdict=verdict,
    )


def _waiting(ctx: "TurnContext") -> str:
    """A brief's list of proposals waiting on them and of withdrawals (DL-087
    #8), and of the sign-ins still lost (#3), each with its fix; empty for any
    other turn, and for a brief with none of these."""
    if not is_brief(ctx.event):
        return ""
    now = str(ctx.event.get("at", "")) or episodes.now()
    text = granted.briefing(ctx.grants, ctx.seq, now=now)
    lost = ctx.access.lost if ctx.access is not None else {}
    if not lost:
        return text
    rows = [
        f"- {e.label} — {e.fix}, since {granted.since_words(e.since)}."
        for e in sorted(lost.values(), key=lambda e: (e.since, e.origin))
    ]
    lines = text.splitlines() if text else []
    if lines and lines[0] == _WAITING:
        end = 1
        while end < len(lines) and lines[end].startswith("- "):
            end += 1
        lines[end:end] = rows
    else:
        lines[0:0] = [_WAITING, *rows]
    return "\n".join(lines)


def _brief_tail(ctx: "TurnContext") -> str:
    """What code appends to a brief: :func:`_waiting`, what is new on the
    topics kept up with (DL-087 #6), then the tasks under way (DL-095). Each
    part empty when there is nothing, so no section is ever empty."""
    return "\n\n".join(
        part for part in (_waiting(ctx), _keeping(ctx), _running_tasks(ctx)) if part
    )


#: The heading a brief gives the tasks under way (DL-095).
TASKS_RUNNING = "Tasks:"


def _running_tasks(ctx: "TurnContext") -> str:
    """A brief's running and waiting tasks (DL-095): each task started, not
    finished, not stopped, with where its run has got to. Empty on any other
    turn and when nothing is running, so a brief over a log with no tasks reads
    exactly as before."""
    if not is_brief(ctx.event) or ctx.tasks is None:
        return ""
    running = ctx.tasks.running()
    if not running:
        return ""
    lines = [TASKS_RUNNING]
    for t in running:
        where = "waiting" if t.waiting else f"on step {t.cursor + 1} of {t.total}"
        lines.append(f"- {t.plan.goal}: {where}")
    return "\n".join(lines)


#: The heading a brief gives the topics with news (DL-087 #6).
KEEPING_UP = "Keeping up with:"


def _keeping(ctx: "TurnContext") -> str:
    """A brief's news on the topics kept up with (DL-087 #6): only the topics
    with something new since the brief before, each item once. Empty for any
    other turn and for a brief with no news, so no section is ever empty."""
    if not is_brief(ctx.event) or ctx.topics is None:
        return ""
    news = ctx.topics.briefing(ctx.seq)
    if not news:
        return ""
    lines = [KEEPING_UP]
    for name, items in news:
        listed = "; ".join(f"{i['title']} ({i['url']})" for i in items)
        lines.append(f"- {name}: {listed}")
    return "\n".join(lines)


#: The heading :func:`omega.grants.briefing` gives what waits on them.
_WAITING = "Waiting on you:"


#: How every asking tell ends (:data:`omega.tools.FOLLOWUP_ASK`,
#: :data:`omega.tools.PROMISE_ASK`); swapped for the proposal's own answer
#: form once it is filed.
_REPLY_YES_NO = " Reply yes or no."
assert FOLLOWUP_ASK.endswith(_REPLY_YES_NO) and PROMISE_ASK.endswith(_REPLY_YES_NO)


def _proposal_for(queue: EventQueue, seq: int) -> Optional[str]:
    """The id of the proposal a follow-up tell at ``seq`` carries, or
    ``None``. The executor writes it straight after the tell, so it is among
    the next few records; a turn's grants fold stops at its own event and
    cannot see it."""
    for n, episode in enumerate(queue.store.episodes_since(seq)):
        if n >= 8:
            break
        payload = episodes.decode(episode.payload)
        if payload.get("kind") == episodes.GRANT_PROPOSED and payload.get("for_seq") == seq:
            return granted.pid(episode.seq)
    return None


def _question_for(
    ctx: "TurnContext",
    ask: Sequence[tuple[int, str]],
    running: Sequence[schedule.Schedule],
) -> Optional[tuple[int, str]]:
    """The one contradiction question this turn may carry, or ``None``.

    Folded into the brief when there is one (DL-080): a brief is the moment the
    person expects omega to raise things. With no brief standing, a spoken look
    carries it instead. Never a reply to their own message — that turn is about
    what they asked — and at most one per turn, so a backlog of contradictions
    becomes one question a day rather than a quiz.
    """
    if not ask:
        return None
    from omega.tools import BRIEF_LEAD

    briefed = any(item.instruction.startswith(BRIEF_LEAD) for item in running)
    if is_brief(ctx.event) or (is_look(ctx) and not briefed):
        return ask[0]
    return None


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
    # A look's words (DL-076).
    "nudge": SPEAK,
    "check": ACT_THEN_SPEAK,
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
    "commands, search the web, read a page, look at the notes you have "
    "written about this person, keep a standing watch on something for "
    "them, and set a reminder. ACT is the verdict that reaches them, and it is the only one that "
    "does. Choose ACT whenever answering well means looking something up on "
    "this machine, changing a file, or searching the web - do not guess at an "
    "answer you could go and check.\n"
    "Asked what you know or remember about them, choose ACT. Your notes are "
    "not in front of you when you decide this, and only a small part of them "
    "is ever in front of you at all, so answering from what you can see means "
    "reporting an empty memory while holding a full one.\n"
    "If the person wrote to you, they are talking to you: answer them. "
    "SPEAK or ACT is right there even for a greeting, a short question, or "
    "something you think is obvious. Do not stay silent on a message addressed "
    "to you. A question about this machine, this project or a file on it is "
    "still a question for you; SILENT is never the right answer to a question.\n"
    "If they ask you to keep checking, monitor, or let them know about "
    "something, or to remind them of something later, choose ACT: you can "
    "set a watch or a reminder that wakes you on its own.\n"
    "SILENT is a successful outcome, not a failure, for events that need "
    "nothing - background noise, things already handled.\n"
    "An event marked 'nobody asked' is your own look at their machine and "
    "what is open, not the person writing to you, and the question it asks "
    "is: do they need a nudge right now? A nudge is something they would "
    "want to act on soon and would be sorry not to have heard: something due "
    "or overdue, something waiting on them or on you, their machine needing "
    "attention (a reading flagged LOW, above all one that is new since your "
    "last look or worse than when it was first flagged), an event on their "
    "calendar starting within about half an hour, new mail from a person "
    "that needs their answer or attention, or a real change in "
    "something they asked you to track. Trivia, news they did not ask for, "
    "newsletters and notifications, events hours away, "
    "and anything you already told them that has not got worse or more "
    "urgent are not nudges. If a nudge may be due but you would have to "
    "check something first, check. A look is answered in the form its end "
    "asks for, with a reason.\n"
    "A standing watch firing is a check you promised them: choose ACT. A "
    "reminder firing is a promise coming due: never SILENT. If keeping it "
    "means looking something up or doing something first (the weather, a "
    "score, a file), choose ACT; otherwise choose SPEAK and remind them.\n"
    "A daily brief firing is a promise too: choose ACT, look at their "
    "calendar, mail and what is open, then brief them.\n"
    "Judge this event on its own. That you stayed silent before is not a "
    "reason to stay silent now."
)

# Every paragraph below is here because of a specific thing omega said on the
# live store, not because it sounded like a good rule (DL-055). The old version
# of this prompt was three sentences and named no self, and the model filled the
# vacuum from its pretraining prior: asked what it was it answered "an AI
# assistant powered by OpenAI", and asked whether it remembers it said that
# depended on "the app's memory settings" — describing the vendor's product
# while sitting on an append-only log it had just used to recall a fact across a
# restart.
#
# No model name appears here on purpose. The identity has to stay true when
# `.env` changes, and a prompt that recites its own configuration is a prompt
# that starts lying the moment that configuration moves.
_REPLY_SYSTEM = (
    "You are omega, one person's second brain, running on their Mac.\n"
    "You are not a chat assistant, and you are not the product of whichever "
    "model happens to be answering - that can change and you do not. What you "
    "remember is written to an append-only log on this person's own disk, so "
    "it survives restarts, days, and new conversations. Never tell them your "
    "memory is limited to this conversation or depends on a setting "
    "somewhere; that is false. If you were told something and it is not in "
    "front of you now, say you don't have it - not that you cannot keep it.\n"
    "You have a body: you can read and write files, run read-only shell "
    "commands, search the web, read a page, read their calendar and their "
    "inbox (read-only, when connected), and look at the notes you have "
    "written about this "
    "person. Never tell the person you cannot reach their filesystem or the "
    "network - that is false - and never hand them a shell command to run "
    "themselves in place of doing it.\n"
    "You also run in the background: you look at their machine and what is "
    "open on your own, you can keep a standing watch on something and "
    "tell them when there is news, you can remind them of something "
    "later, and you can brief them on their day at a set time. Never tell them you cannot check in the "
    "background or message them on your own - that is false.\n"
    "Only a few of your notes are ever in front of you: the ones that matched "
    "this message. There are many more. So 'I don't have any details about "
    "you' is wrong - what is true is that none are in front of you right now, "
    "and that you can go and look.\n"
    "You are not holding those tools in this particular message. If answering "
    "properly would need one, say what you would need to check. Do not guess "
    "an answer, and do not claim you are unable.\n"
    "Never say you have recorded, saved, scheduled, remembered, or noted "
    "anything. You are not the part of omega that does that. A sentence "
    "claiming it is either a duplicate of the receipt printed under your "
    "reply, or a lie standing in for a receipt that is missing. Answer the "
    "message and let the receipt speak for the filing.\n"
    "When the event says nobody asked, you are opening the conversation, not "
    "answering one. Do not thank them, do not refer to a question, and do not "
    "explain that you were looking - say the one thing that made speaking worth "
    "it and stop. One or two sentences. They did not ask, so earn it. If it "
    f"turns out there is nothing worth it, answer exactly {NOTHING_NEW}\n"
    "How you talk: like someone who knows this person and is not performing. "
    "Short, first person, contractions, dry. Answer the thing asked and stop - "
    "no 'Got it', no repeating their message back to them, no offer to help at "
    "the end. Not knowing is said plainly and briefly, with no apology and no "
    "hedge.\n"
    "You render in a small panel, so most answers are a sentence or three. Be "
    "longer when the content really is that long - walking through a "
    "screenshot, or a set of genuinely separate things - and never drop part "
    "of an answer in order to be brief. Formatting has to earn itself: no bold "
    "for emphasis, and no bullet list unless the content is a list."
)


#: Where :func:`_with_hands` adds to each prompt's "you have a body" sentence.
_JUDGE_BODY_END = "and set a reminder. "
_REPLY_BODY_END = "written about this person. "
_SLACK_HAND = "post to their work Slack as them"
_DELEGATE_HAND = "hand work to a Claude Code session on their Mac"
#: How a configured MCP server's tools read as an ability; any other server
#: is named by the name the person gave it.
_SERVER_ABILITY = {
    leave.SERVER: "find places and routes on Maps",
    "github": "read GitHub",
}


def hands(ctx: "TurnContext") -> list[str]:
    """The configured or seat-only abilities this turn's act box will hold,
    as words for the judge and the reply (DL-101a).

    Structural only, never the message's words: Slack where
    :func:`omega.act._slack_seat` would seat it (their own message and a
    token set), a Claude Code session where :func:`omega.act._delegate_seat`
    would seat one (their own message and a worker on their Mac polling), each configured MCP server with tools allowed on every turn
    as the box is handed them, and the leave-now alarm where
    :func:`omega.tools._leaving` could hold, from the configuration alone so
    that judging never lists a server over the network. Empty when nothing
    is configured, which is what keeps both prompts as they were.
    """
    prompted = not unprompted(ctx)
    out: list[str] = []
    if prompted and slack.client() is not None:
        out.append(_SLACK_HAND)
    if prompted and delegate.available():
        # DL-104: a worker on their Mac, with folders allowed, polled lately.
        out.append(_DELEGATE_HAND)
    servers = [s for s in mcp.servers() if s.allow]
    out.extend(_SERVER_ABILITY.get(s.name, f"use their {s.name} tools") for s in servers)
    if prompted and any(leave.FULL in {f"{s.name}{mcp.SEP}{t}" for t in s.allow} for s in servers):
        out.append("tell them when to leave by live traffic")
    return out


def _abilities(hands_: Sequence[str]) -> str:
    if len(hands_) == 1:
        return hands_[0]
    return ", ".join(hands_[:-1]) + " and " + hands_[-1]


def _with_hands(system: str, anchor: str, extra: str) -> str:
    assert system.count(anchor) == 1, anchor
    return system.replace(anchor, anchor + extra, 1)


def judge_system(ctx: "TurnContext") -> str:
    """:data:`_JUDGE_SYSTEM`, plus what is switched on this turn and the one
    rule that sends an ask for it to ACT; byte-identical with nothing on."""
    on = hands(ctx)
    if not on:
        return _JUDGE_SYSTEM
    reaches = [
        "post, share or send something to Slack" if hand == _SLACK_HAND
        else "ask Claude or Claude Code to make, write or fix something"
        if hand == _DELEGATE_HAND
        else "look something up on a service named here"
        for hand in on
    ]
    return _with_hands(
        _JUDGE_SYSTEM, _JUDGE_BODY_END,
        f"This turn you can also {_abilities(on)}. If they ask you to do "
        f"something one of these reaches ({', '.join(dict.fromkeys(reaches))}), "
        "choose ACT: only ACT holds the tools that do it. ",
    )


def reply_system(ctx: "TurnContext") -> str:
    """:data:`_REPLY_SYSTEM`, plus what is switched on this turn;
    byte-identical with nothing on."""
    on = hands(ctx)
    if not on:
        return _REPLY_SYSTEM
    return _with_hands(
        _REPLY_SYSTEM, _REPLY_BODY_END, f"You can also {_abilities(on)}. "
    )


def _event_turn(
    ctx: TurnContext, *, images: bool, suffix: str = "", lessons: bool = False
) -> provider.Message:
    """The one user turn every role is given: recall, then the new event.

    ``images`` is the whole of DL-031's cheap/capable split at the prompt
    boundary. With it false the turn is a plain string and any attachment is
    named but not shown; with it true the attachments on *this* event — never
    the ones in recall — ride along as real bytes. `judge` passes false
    because it fires on every event and only needs to know a picture is there
    to route; `act` and `reply` pass true because they are the roles expected
    to answer about it.

    ``lessons`` puts omega's own lessons (DL-086) under the claims: the judge
    and act steps decide how to work, so they read them; the reply step only
    words what was done. Empty when none stand, so the turn is unchanged.
    """
    # The new event gets its own, far looser budget (DL-039): it is the thing
    # omega was asked about, so cutting it is cutting the question. It is
    # bounded at all only because one pasted document should cost a degraded
    # turn rather than a failed request.
    event = _elide(_render_event(ctx.event), MAX_NEW_EVENT_CHARS)
    text = (
        f"{_clock_line(ctx.event)}"
        f"{_location_section(ctx)}"
        f"{_learned_section(ctx.learned, at=str(ctx.event.get('at', '')))}"
        f"{self_taught.section(ctx.lessons) if lessons else ''}"
        f"Recent history:\n{_transcript(ctx.recalled)}\n\n"
        f"{_open_section(ctx.open_work)}"
        f"{_grants_section(ctx)}"
        f"{_followups_section(ctx)}"
        f"{_promises_section(ctx)}"
        f"New event:\n{event}{suffix}"
    )
    parts = _image_parts(ctx) if images else []
    if not parts:
        return provider.user(text)
    return provider.user([provider.text_part(text), *parts])


def _location_section(ctx: TurnContext) -> str:
    """Where they are (DL-091), one line under the clock, so a maps tool can
    answer "near me" and route questions.

    Only on their own message: a brief, a watch, a look or a tell never
    mentions it. Fresh for three hours; after that it is their *last known*
    location and says so, never presented as current. Absent with no fix, so
    a turn over a log without one renders exactly as before.
    """
    if ctx.location is None or unprompted(ctx):
        return ""
    return whereabouts.context_line(ctx.location, ctx.event.get("at"))


def _clock_line(event: dict[str, Any]) -> str:
    """When this event arrived, in the person's zone (DL-079).

    Every role used to infer "now" from whatever look last printed a time; at
    11:43 IST the newest one said 05:52 (UTC, before the VM had a zone), and
    "remind me at 1" became 428 minutes. Read off the event's own ``at``, not
    the wall clock, so a replayed turn sees the moment it saw live.
    """
    raw = event.get("at")
    if not isinstance(raw, str) or not raw.strip():
        return ""
    try:
        moment = datetime.fromisoformat(raw.strip())
    except ValueError:
        return ""
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    local = moment.astimezone()
    return f"It is now {local:%A %d %B %Y, %H:%M}, {schedule.zone_label(local)}.\n"


#: Repeated at the very end of the judge's user turn, after the transcript.
#:
#: Measured, not guessed. On the live store, `hello` failed roughly every other
#: time with ``JudgeUndecided: judge answered 'Hello! How can I help?'`` — and
#: that string is exactly the reply the *previous* turn had given. Rendering the
#: prompt for the failing seq showed why: it ends
#:
#:     you: hello
#:     omega: Hello! How can I help?
#:
#:     New event:
#:     you: hello
#:
#: which is a few-shot demonstration of answering the greeting, with the only
#: instruction not to do that sitting four thousand characters earlier in the
#: system message. At temperature 0 the likeliest continuation of that text is
#: the greeting, so the model completed the conversation instead of classifying
#: it. The alternation was the tell: after a *failed* turn the transcript ends
#: ``omega: (turn failed: …)``, which is no pattern to copy, and the judge
#: answered correctly.
#:
#: So the format demand is repeated where the model actually generates, and the
#: first sentence names the specific error rather than the general rule —
#: "answer with one word" was already said and lost; "do not answer that
#: message" is the thing the transcript is arguing for.
#:
#: **The parser is deliberately not loosened to accept this.** Scanning a
#: sentence for a verdict word would read "not silent" as silence, and a router
#: that guesses is worse than one that says it could not decide.
_JUDGE_SUFFIX = (
    "\n\n---\n"
    "Do not answer that message. You are routing it, not replying to it.\n"
    "Your entire reply is one word - SPEAK, ACT, or SILENT."
)

#: The end of the judge's turn for a look (DL-076). A look gets its own
#: question - does the person need a nudge right now - and must give its
#: reason on the same line. The reason is the guard on both sides: a nudge has
#: to name what needs their attention, which "here is some news" cannot, and a
#: silence is recorded with why, which the log could never say before.
#: ``NUDGE`` reads as SPEAK and ``CHECK`` as ACT; the first word still decides.
_LOOK_JUDGE_SUFFIX = (
    "\n\n---\n"
    "This is your own look, not a message to answer. Do they need a nudge "
    "right now?\n"
    "Answer on one line: NUDGE, CHECK, or SILENT as the first word, then a "
    "short reason - what needs their attention, what you will check, or why "
    "nothing does."
)


def _review(
    queue: EventQueue,
    pending: Pending,
    complete: Callable[..., provider.Response],
    meter: provider.Meter,
    *,
    lessons: Any,
    at: Optional[str],
    gate: Any = None,
) -> TurnResult:
    """omega's weekly self-review (DL-086), in place of judge, act and reply.

    One LEARN call over evidence code built (:mod:`omega.lessons`). It speaks
    a short report when something changed and is quiet when nothing did,
    which is a review that read the log and found nothing to learn, not a
    look that skipped looking: this path never reaches :func:`reply`, where
    the unlooked rule lives. An answer that was not the shape writes nothing
    and fails the turn, so the failure follow-up counts it.
    """
    try:
        done = self_taught.review(
            queue, complete, upto=pending.seq, state=lessons, at=at,
        )
    except Exception as exc:  # noqa: BLE001 - every path records, as run_turn's
        error = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
        record = write_memory(
            queue, seq=pending.seq, outcome="failed", error=error,
            usage=meter.usage(), at=at,
        )
        return TurnResult(
            seq=pending.seq, outcome="failed", record_seq=record, error=error
        )
    if done.report is None or (
        gate is not None and outbox.settle(queue, pending.seq, gate, done.report)
    ):
        record = write_memory(
            queue, seq=pending.seq, outcome="silent", usage=meter.usage(), at=at
        )
        return TurnResult(seq=pending.seq, outcome="silent", record_seq=record)
    record = write_memory(
        queue, seq=pending.seq, outcome="spoke", reply_text=done.report,
        usage=meter.usage(), at=at,
    )
    return TurnResult(
        seq=pending.seq, outcome="spoke", record_seq=record, reply=done.report
    )


def _judge_messages(ctx: TurnContext) -> list[provider.Message]:
    suffix = _LOOK_JUDGE_SUFFIX if is_look(ctx) else _JUDGE_SUFFIX
    return [
        provider.system(judge_system(ctx)),
        _event_turn(ctx, images=False, suffix=suffix, lessons=True),
    ]


def _reply_messages(
    ctx: TurnContext, acted: ActResult, reason: str = ""
) -> list[provider.Message]:
    work = f"\n\nWork done this turn: {', '.join(acted.tools)}" if acted.tools else ""
    why = f"\n\nYou decided to speak up because: {reason}" if reason else ""
    return [
        provider.system(reply_system(ctx)),
        _event_turn(ctx, images=True, suffix=work + why),
    ]


def _learned_section(learned: Sequence[derive.Claim], *, at: str = "") -> str:
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
    lines = [
        _elide(f"- {claim.text}{_hygiene(claim, at)}", MAX_EVENT_CHARS, at=claim.seq)
        for claim in ordered
    ]
    lines = _within(lines, MAX_LEARNED_CHARS, noun="learned item")
    body = "\n".join(lines)
    return f"What you have learned about working with this person:\n{body}\n\n"


def _hygiene(claim: derive.Claim, at: str) -> str:
    """What memory hygiene noted (DL-087 #4), bracketed, or ``""`` when it
    noted nothing — which is what keeps an untouched set byte-identical."""
    note = claim.note(at or None)
    return f" [{note}]" if note else ""


def _grants_section(ctx: TurnContext) -> str:
    """The standing grants (DL-085), when there are any.

    Absent when nothing is granted or pending, as the open section is, so a
    turn with no grants renders exactly as it did before they existed.
    """
    state = ctx.grants
    if not state:
        return ""
    lines = []
    for scope in sorted(state.grants):
        g = state.grants[scope]
        gate = (
            f"you ask before each action ({g.streak} yes in a row)"
            if g.gated else "they agreed you need not ask first; each action goes in the next brief"
        )
        lines.append(f"- {scope} (given in message {g.for_seq}): {gate}")
    for p in state.pending:
        lines.append(f"- waiting on their yes or no to {p.pid}: {p.words()}")
    if is_brief(ctx.event) and (state.pending or state.withdrawn_for_brief(ctx.seq)):
        lines.append(
            "The list of what waits on their answer, and of what you stopped "
            "asking, is added after your brief in code; do not repeat it."
        )
    for task, what in sorted(state.tasks.items()):
        lines.append(f"- research task {task} under way: {what}")
    if is_brief(ctx.event):
        done = state.done_for_brief(ctx.seq)
        if done:
            lines.append("Done without asking since the last brief; tell them each, and how to undo it:")
            lines.extend(f"  - {a.why} ({a.undo_words()})" for a in done)
    body = "\n".join(lines)
    return f"Standing grants they gave you:\n{body}\n\n"


def _followups_section(ctx: TurnContext) -> str:
    """What a brief owes them about omega's own failures (DL-087 #2).

    Only in a brief, and only when there is something: a subject they were
    told kept failing that works again (one line, not an interrupt), and a
    research task whose follow-up failed ("couldn't finish X — here's why").
    Absent otherwise, so a log with no failures renders as it always did.
    """
    state = ctx.followups
    if not state or not is_brief(ctx.event):
        return ""
    owed = state.briefing(ctx.seq)
    if not owed:
        return ""
    lines = []
    for e in owed.recovered:
        lines.append(
            f"- working again: {e.label}, which had failed {e.count} times in a "
            "row and which you told them about; say so in one line"
        )
    research = ctx.grants is not None and ctx.grants.live(granted.RESEARCH) is not None
    for e in owed.research:
        carry = (
            f" You may carry it on with {FOLLOW_UP} (task {e.ref}); otherwise say so."
            if research else ""
        )
        lines.append(
            f"- you couldn't finish {e.label}: here's why — {e.last_error}. "
            f"Tell them.{carry}"
        )
    body = "\n".join(lines)
    return f"Your own follow-ups, for this brief:\n{body}\n\n"


def _promises_section(ctx: TurnContext) -> str:
    """The open promises a brief names (DL-087 #1).

    Only in a brief, only under a live ``promises`` grant, and only for
    promises no earlier brief named: each is named once, up to
    :data:`omega.promises.BRIEF_NAMES` oldest first, then "+N more" (which the
    next brief names). Absent otherwise, so a turn without the grant renders
    exactly as it did.
    """
    state = ctx.promises
    if not state or not is_brief(ctx.event):
        return ""
    owed = state.briefing(ctx.seq)
    if not owed:
        return ""
    lines = [f"- {p.words()}, said in message {p.source_seq} (promise {p.seq})" for p in owed.named]
    if owed.more:
        lines.append(f"+{owed.more} more")
    body = "\n".join(lines)
    return (
        "Open promises they made, for this brief (name each, oldest first, "
        "with \"+N more\" as written; they are theirs to keep, never yours to "
        f"act on):\n{body}\n\n"
    )


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
        if payload.get("channel") == notice.CHANNEL:
            # **Not a person talking**, and this prefix is the only thing that
            # says so (DL-061). Without it an unprompted pass renders as
            # `you: ...`, the judge is told plainly that "if the person wrote to
            # you, they are talking to you: answer them", and omega would speak
            # on every single look while never once being silent - the firehose
            # at exactly the permitted rate. It matters in recall too: omega
            # reading its own looks back as the person's words would build a
            # memory of things they never said.
            return f"(nobody asked - you looked at what was open)\n{payload.get('text', '')}"
        if payload.get("untrusted"):
            # A tell carrying a watched page's words (DL-096), fenced as
            # DL-064 fences every untrusted result.
            return f"you: {fence(str(payload['untrusted']), str(payload.get('text', '')))}"
        return f"you: {payload.get('text', '')}"
    if kind == episodes.TURN_COMPLETED:
        outcome = payload.get("outcome")
        if outcome == "silent":
            return "omega: (stayed silent)"
        if outcome == "failed":
            return f"omega: (turn failed: {payload.get('error')})"
        tainted = payload.get("tainted")
        if isinstance(tainted, list) and tainted:
            # A reply from a turn that read someone else's words may repeat
            # them (a watched page's alert does, DL-096): fenced in recall,
            # as the words were where it read them (DL-064).
            sources = ", ".join(str(t) for t in tainted)
            return f"omega: {fence(sources, str(payload.get('reply', '')))}"
        return f"omega: {payload.get('reply', '')}"
    if kind == episodes.TURN_BLOCKED:
        return f"omega asked: {payload.get('needs', '')}"
    if kind == episodes.WORK_FINISHED:
        return f"work finished: {payload.get('summary', '')}"
    if kind == episodes.TOOL_CALLED:
        return f"tool {payload.get('tool')} called"
    if kind == episodes.TOOL_RETURNED:
        if payload.get("ok") and payload.get("tool") == "mail":
            return f"tool mail ok: {_mail_digest(str(payload.get('result') or ''))}"
        ok = "ok" if payload.get("ok") else f"failed: {payload.get('error')}"
        return f"tool {payload.get('tool')} {ok}"
    return str(kind)


#: How long a recalled ``mail`` result may run (DL-082 #5).
MAIL_DIGEST_CHARS = 200


def _mail_digest(result: str) -> str:
    """One line saying which mail was opened, so what was read stays in
    context without its body. A held mail's result is already its redacted
    digest; an ordinary one is reduced to its From and Subject lines."""
    head = dict(
        line.split(": ", 1) for line in result.splitlines()[:3] if ": " in line
    )
    if "From" in head:
        line = f"opened mail from {head['From']}: {head.get('Subject', '(no subject)')}"
    else:
        line = " ".join(result.split())
    if len(line) > MAIL_DIGEST_CHARS:
        line = line[: MAIL_DIGEST_CHARS - 1].rstrip() + "…"
    return line


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
