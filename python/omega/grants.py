"""Standing grants, and the gate they sit behind (DL-085).

A grant is authority the person handed over in their own message, for one of
four closed scopes: **research** (carry its own research tasks to the end),
**tidy** (tidy their reminders and watches), **promises** (DL-087 #1: hold
them to what they say they will do, with a reminder) and **topics** (DL-087
#6: notice what they keep asking about and offer to keep up with it). Nothing
else can be granted, and calendar, mail and anything that moves money are
outside all four.

Everything here is a fold over the log, like :class:`omega.derive.OpenWork`.
The grant set is never stored beside the log and never mutated; a turn is
handed a frozen :class:`State` read off the fold at its own seq.

**One path, with a gate.** Under a grant an action is first *proposed*
("should I do X?"), and runs only on their yes. Each yes counts; any no resets
the count. After :data:`GATE_DROP_AFTER` yeses in a row for one scope omega
*proposes* dropping the gate, and the gate drops only on their explicit yes to
that. Nothing drops it on its own, and there is no shadow mode in which omega
acts quietly while pretending to ask. They put the gate back with one sentence
(the ``grant`` tool's ``restore_gate``). What runs with the gate dropped is
listed in the next daily brief, with how to undo it.

**Reversibility tiers** classify every action omega can take, beside DL-014's
dispatch tiers rather than instead of them. DL-014 decides whether a call may
run without asking; the reversibility tier decides whether a grant may ever
cover it. Irreversible actions are exactly DL-014's ``external`` ones, so they
are always confirmed and a grant cannot reach them. Money is refused outright
(DL-088), before any tier is read, and no grant lifts that.

**Approvals that do not stall (DL-087 #8).** A proposal is an episode with an
id (its own seq, shown as ``p<seq>``). Filing one ends the turn normally, as a
reply that shows the id and how to answer it; several can be pending at once.
:func:`answers` reads the person's reply: ``yes p12``, ``no p12 p14``, or a
bare yes or no, which answers only the one proposal pending and only while it
is the newest thing omega said. A proposal that no longer applies (its grant
taken back, its gate put back, its watch stopped or changed, its research
finished) *lapses* here at once, so no later yes can run it, and the executor
writes it down as ``grant.withdrawn`` (:func:`withdraw_lapsed`), which the next
brief reports in one line. An answer to a lapsed, withdrawn or already
answered proposal is told "that one's closed" and runs nothing. Every brief
lists what is still waiting (:func:`briefing`), oldest marked old after
:data:`OLD_AFTER_DAYS`, until it is answered or withdrawn.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Iterable, Optional, Sequence

from omega import episodes

if TYPE_CHECKING:  # pragma: no cover
    from omega.memory import MemoryStore

__all__ = [
    "RESEARCH",
    "TIDY",
    "PROMISES",
    "TOPICS",
    "SCOPES",
    "GIVABLE",
    "GATE_DROP_AFTER",
    "GATED_TOOLS",
    "PROPOSED_TOOLS",
    "READ",
    "OWN",
    "OUTWARD",
    "IRREVERSIBLE",
    "MONEY",
    "REVERSIBILITY",
    "tier_of",
    "is_money",
    "money_refusal",
    "Grant",
    "Proposal",
    "Acted",
    "State",
    "Grants",
    "reading",
    "answers",
    "Answer",
    "how_to_answer",
    "proposal_of",
    "button_summary",
    "briefing",
    "since_words",
    "is_old",
    "withdraw_lapsed",
    "OLD_AFTER_DAYS",
    "undo_for",
    "pid",
]

RESEARCH = "research"
TIDY = "tidy"
PROMISES = "promises"
TOPICS = "topics"
TASKS = "tasks"
SCOPES = episodes.GRANT_SCOPES
assert {RESEARCH, TIDY, PROMISES, TOPICS, TASKS} == SCOPES
#: The scopes a person can give, revoke or re-gate with the `grant` tool.
#: `tasks` is not one: it is only the queue a plan's approval is filed under
#: (DL-095), approved per plan and never held as a standing grant.
GIVABLE = SCOPES - {TASKS}

#: Yeses in a row, for one scope, before omega asks whether it may stop
#: asking. Asks, never drops.
GATE_DROP_AFTER = 10

#: Days a proposal waits before the brief marks it old. Marked, never dropped:
#: it is listed until they answer it or it is withdrawn.
OLD_AFTER_DAYS = 7

#: The tools a grant covers, and so the only ones a gate sits in front of.
#: Every other tool is either a read (no gate needed) or outside every grant.
GATED_TOOLS = {"tidy": TIDY, "follow_up": RESEARCH}

#: Tools a grant covers only when *code* proposes them under it, never when
#: the model asks (DL-087 #1): a reminder for a promise they made is proposed
#: under ``promises``, while a ``remind`` they ask for in so many words runs as
#: it always has. Kept apart from :data:`GATED_TOOLS` for exactly that reason:
#: listing ``remind`` there would put a gate in front of every reminder.
#: ``topics`` likewise (DL-087 #6): code offers to follow a subject they keep
#: asking about under ``topics``; a follow they ask for runs ungated.
PROPOSED_TOOLS = {"remind": PROMISES, "topics": TOPICS}

# --- reversibility ----------------------------------------------------------

#: Reads. Nothing changes anywhere.
READ = "read"
#: Writes only omega's own log or store, and can be undone from it.
OWN = "own"
#: Reaches outside omega but can be taken back. ``slack`` is the first one
#: here (DL-101): a post reaches their work Slack, and ``slack delete`` takes
#: back any post omega made.
OUTWARD = "outward"
#: Cannot be taken back, or reaches outside omega in a way it cannot. Always
#: confirmed; never covered by a grant. Identical to DL-014's ``external``.
IRREVERSIBLE = "irreversible"
#: Moves money. Refused outright (DL-088).
MONEY = "money"

#: Every tool omega can take, by name. ``write_file`` and ``run_code`` depend
#: on their arguments, so their entry is resolved by :func:`tier_of`.
REVERSIBILITY: dict[str, str] = {
    "read_file": READ,
    "recall": READ,
    "schedules": READ,
    "calendar": READ,
    "inbox": READ,
    "mail": READ,
    "finance": READ,
    "projects": READ,  # reads the log only (DL-087 #7)
    "pending": READ,  # lists open proposals off the fold (DL-087 #8)
    "look_back": READ,  # searches the past conversation in the log (DL-094)
    "journal": READ,  # reads omega's thinking journal back (DL-118)
    "fetch": READ,
    "web_search": READ,
    "watch": OWN,
    "unwatch": OWN,
    "remind": OWN,
    "brief": OWN,
    "grant": OWN,
    "undo": OWN,
    "tidy": OWN,
    "follow_up": OWN,
    "report": OWN,
    "promise": OWN,  # closes one of their noted promises (DL-087 #1)
    "forget": OWN,  # strikes one of omega's own lessons or skills (DL-086)
    "location": OWN,  # notes or names where they are, in the log (DL-091)
    "topics": OWN,  # follows, drops or looks up a topic (DL-087 #6)
    "task": OWN,  # drafts, cancels or reports on a task's plan (DL-095)
    "step": OWN,  # records a driven step's terminal state (DL-095)
    "watch_page": OWN,  # sets or stops a page watch in the log (DL-096)
    "slack": OUTWARD,  # posts as them; omega's own posts can be deleted (DL-101)
    "handoff": OWN,  # a link, or a document into the store; never a payment (DL-097)
    "write_file": OWN,  # inside the store; outside it is irreversible
    "run_code": READ,  # read-only argv; anything else is irreversible
}


def tier_of(tool: str, dispatch_tier: str) -> str:
    """The reversibility tier of one classified call.

    ``dispatch_tier`` is DL-014's tier for the same call. An ``external``
    call is irreversible whatever the table says, which is what makes
    "irreversible is always confirmed" a fact and not a second list to keep
    in step: the confirmation is DL-014's own block.
    """
    if is_money(tool):
        return MONEY
    if dispatch_tier == "external":
        return IRREVERSIBLE
    known = REVERSIBILITY.get(tool)
    if known is not None:
        return known
    # A tool this table has not met (an adapter's, say): a read if DL-014
    # lets it run as one, otherwise omega's own write at most.
    return READ if dispatch_tier == "exploration" else OWN


_MONEY_WORDS = frozenset(
    {
        "pay", "payment", "payments", "payout", "transfer", "checkout",
        "purchase", "buy", "money", "withdraw", "refund", "invoice", "wire",
        "upi", "venmo", "paypal", "stripe", "tip", "donate", "send_money",
    }
)


def is_money(tool: str) -> bool:
    """Does this tool's name say it moves money?

    By the name's words, so ``stripe_create_payment`` and ``pay`` are both
    caught and ``finance`` (which only reads the ledger) is not. Wide on
    purpose: a false refusal costs one tool, a false pass costs money.
    """
    words = {w for w in re.split(r"[^a-z0-9]+", tool.lower()) if w}
    if words & _MONEY_WORDS:
        return True
    # And whatever the MCP adapter's own, broader name check refuses (DL-088):
    # one answer to "does this move money" for every tool, local or remote.
    # Imported here because the adapter imports the tools that import this.
    from omega.mcp import money_refusal as remote_refusal

    return remote_refusal(tool) is not None


def money_refusal(tool: str) -> str:
    return (
        f"{tool!r} moves money, and omega refuses every tool that does (DL-088): "
        "no grant, approval or gate can change that. Tell them to do it themselves."
    )


# --- the fold ---------------------------------------------------------------


@dataclass(frozen=True)
class Grant:
    """One live grant."""

    scope: str
    for_seq: int  # the message that gave it
    at: str
    gated: bool = True
    streak: int = 0  # yeses in a row since the gate was last reset


@dataclass(frozen=True)
class Proposal:
    """A question omega asked under a grant and has no answer to yet."""

    seq: int  # its id
    for_seq: int  # the turn that asked
    scope: str
    proposes: str  # "action", "drop_gate" or "task"
    actions: tuple[dict[str, Any], ...] = ()
    at: str = ""  # when it was asked
    #: Why a held call was held (DL-064): "this turn read ...". Empty otherwise.
    reason: str = ""
    task: Optional[int] = None  # the task.proposed seq, when proposes == "task"
    summary: str = ""  # the task's goal, for the words it asks (DL-095)

    @property
    def pid(self) -> str:
        return pid(self.seq)

    def words(self) -> str:
        """What it asks, in words a person can answer."""
        if self.proposes == "drop_gate":
            return f"stop asking first before {self.scope} actions"
        if self.proposes == "task":
            return f"run the plan: {self.summary}" if self.summary else "run the plan"
        words = "; ".join(str(a.get("why", a.get("tool"))) for a in self.actions)
        return f"{words} ({self.reason})" if self.reason else words


@dataclass(frozen=True)
class Acted:
    """Something done under a grant."""

    seq: int
    for_seq: int
    scope: str
    tool: str
    why: str
    args: dict[str, Any]
    undo: dict[str, Any]
    gated: bool
    at: str

    def undo_words(self) -> str:
        if "restore" in self.undo:
            return f"undo(action={self.seq}) puts {self.undo['restore'].get('id')} back as it was"
        if "drop" in self.undo:
            return f"undo(action={self.seq}) stops keeping up with {self.undo['drop']}"
        return f"undo(action={self.seq}) stops {self.undo.get('cancel')}"


def pid(seq: int) -> str:
    return f"p{seq}"


def proposal_of(seq: int, payload: dict[str, Any]) -> Proposal:
    """The proposal a ``grant.proposed`` record at ``seq`` files."""
    task = payload.get("task")
    return Proposal(
        seq=seq,
        for_seq=int(payload["for_seq"]),
        scope=payload["scope"],
        proposes=payload["proposes"],
        actions=tuple(dict(a) for a in payload["actions"]),
        at=str(payload.get("at", "")),
        reason=str(payload.get("reason", "")),
        task=int(task) if isinstance(task, int) and not isinstance(task, bool) else None,
        summary=str(payload.get("summary", "")),
    )


def button_summary(proposal: Proposal) -> str:
    """One line saying what a yes to ``proposal`` would do (DL-103): the tools
    it runs, then its words, cut to :data:`episodes.MAX_PROPOSAL_SUMMARY`.
    Shown beside a button; the button itself is driven by the id alone."""
    tools = list(dict.fromkeys(str(a.get("tool")) for a in proposal.actions if a.get("tool")))
    words = proposal.words()
    line = f"{', '.join(tools)}: {words}" if tools else words
    line = " ".join(line.split())
    cap = episodes.MAX_PROPOSAL_SUMMARY
    return line if len(line) <= cap else line[: cap - 1].rstrip() + "…"


def how_to_answer(proposal_id: str) -> str:
    """How every proposal message says it is answered (DL-087 #8)."""
    return f'Reply "yes {proposal_id}" or "no {proposal_id}".'


def _moment(raw: str) -> Optional[datetime]:
    try:
        moment = datetime.fromisoformat(str(raw).strip())
    except ValueError:
        return None
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=timezone.utc)


def since_words(at: str) -> str:
    """When a proposal was asked, as a brief says it: ``Sat 11 Oct``."""
    moment = _moment(at)
    return f"{moment.astimezone():%a %d %b}" if moment is not None else "a while ago"


def is_old(proposal: "Proposal", now: str) -> bool:
    """Has it waited :data:`OLD_AFTER_DAYS` or more by ``now``?"""
    asked, then = _moment(proposal.at), _moment(now)
    if asked is None or then is None:
        return False
    return (then - asked).total_seconds() >= OLD_AFTER_DAYS * 86400


@dataclass(frozen=True)
class State:
    """The grants as of one seq. Frozen, so a turn cannot change what it read."""

    through: int = 0
    grants: dict[str, Grant] = field(default_factory=dict)
    pending: tuple[Proposal, ...] = ()
    acted: dict[int, Acted] = field(default_factory=dict)
    #: Brief seq -> what was done with the gate dropped since the brief before.
    briefed: dict[int, tuple[Acted, ...]] = field(default_factory=dict)
    #: Research tasks under way: task seq -> what was last asked of it.
    tasks: dict[int, str] = field(default_factory=dict)
    #: Proposals that can no longer be answered: seq -> (the turn that asked,
    #: why). Answered, withdrawn, or lapsed and not yet written down.
    closed: dict[int, tuple[int, str]] = field(default_factory=dict)
    #: Proposals that stopped applying, not yet recorded as withdrawn.
    lapsed: tuple[tuple[Proposal, str], ...] = ()
    #: Brief seq -> proposals withdrawn since the brief before, with why.
    withdrawn: dict[int, tuple[tuple[Proposal, str], ...]] = field(default_factory=dict)
    #: The turn behind the newest thing omega said to them (a reply or a
    #: question), so a bare yes can tell whether a proposal is still the
    #: newest open question.
    said: int = 0

    def live(self, scope: str) -> Optional[Grant]:
        return self.grants.get(scope)

    def gated(self, scope: str) -> bool:
        """Is the gate on for ``scope``? True for a scope with no grant, so a
        caller that forgot to check the grant still asks."""
        g = self.grants.get(scope)
        return True if g is None else g.gated

    def proposal(self, seq: int) -> Optional[Proposal]:
        for p in self.pending:
            if p.seq == seq:
                return p
        return None

    def done_for_brief(self, seq: int) -> tuple[Acted, ...]:
        return self.briefed.get(seq, ())

    def withdrawn_for_brief(self, seq: int) -> tuple[tuple[Proposal, str], ...]:
        return self.withdrawn.get(seq, ())

    def __bool__(self) -> bool:
        # What is closed or lapsed is not on its own a reason to render
        # anything, so a log whose proposals are all settled reads as before.
        return bool(
            self.grants or self.pending or any(self.briefed.values())
            or any(self.withdrawn.values())
        )


class Grants:
    """The fold. ``apply`` in seq order; :meth:`state` snapshots it."""

    __slots__ = (
        "_through", "_grants", "_pending", "_acted", "_unreported", "_briefed", "_tasks",
        "_closed", "_lapsed", "_unwithdrawn", "_withdrawn", "_said",
    )

    def __init__(self) -> None:
        self._through = 0
        self._grants: dict[str, Grant] = {}
        self._pending: dict[int, Proposal] = {}
        self._acted: dict[int, Acted] = {}
        self._unreported: list[Acted] = []
        self._briefed: dict[int, tuple[Acted, ...]] = {}
        self._tasks: dict[int, str] = {}
        self._closed: dict[int, tuple[int, str]] = {}
        self._lapsed: dict[int, tuple[Proposal, str]] = {}
        self._unwithdrawn: list[tuple[Proposal, str]] = []
        self._withdrawn: dict[int, tuple[tuple[Proposal, str], ...]] = {}
        self._said = 0

    @property
    def through(self) -> int:
        return self._through

    def apply(self, seq: int, payload: dict[str, Any]) -> None:
        if seq <= self._through:
            raise ValueError(
                f"episodes must be folded in order: got seq {seq} after {self._through}"
            )
        kind = payload.get("kind")
        if kind == episodes.GRANT_GIVEN:
            scope = payload["scope"]
            self._grants[scope] = Grant(
                scope=scope, for_seq=int(payload["for_seq"]), at=str(payload["at"])
            )
        elif kind == episodes.GRANT_REVOKED:
            scope = payload["scope"]
            self._grants.pop(scope, None)
            # What was proposed under a grant that is gone has nothing to run
            # under: a later yes to it must not run it. Lapsed, not dropped
            # silently: it is withdrawn, and the next brief says so.
            self._lapse(
                lambda p: p.scope == scope, f"you took back the {scope} grant"
            )
            if scope == RESEARCH:
                self._tasks.clear()
        elif kind == episodes.GRANT_RESTORED:
            scope = payload["scope"]
            self._reset(scope, gated=True)
            self._lapse(
                lambda p: p.scope == scope and p.proposes == "drop_gate",
                f"you put the {scope} gate back on",
            )
        elif kind == episodes.GRANT_PROPOSED:
            self._pending[seq] = proposal_of(seq, payload)
        elif kind == episodes.GRANT_ANSWERED:
            p = self._pending.pop(int(payload["proposal"]), None)
            gone = self._lapsed.pop(int(payload["proposal"]), None)
            if p is not None or gone is not None:
                said = "yes" if payload["yes"] else "no"
                self._closed[int(payload["proposal"])] = (
                    int(payload["proposed_for"]), f"you already said {said} to it"
                )
            g = self._grants.get(payload["scope"])
            if p is not None and g is not None:
                if p.proposes == "drop_gate":
                    self._reset(g.scope, gated=not payload["yes"])
                elif payload["yes"]:
                    self._grants[g.scope] = Grant(
                        scope=g.scope, for_seq=g.for_seq, at=g.at,
                        gated=g.gated, streak=g.streak + 1,
                    )
                else:
                    self._reset(g.scope, gated=g.gated)
        elif kind == episodes.GRANT_ACTED:
            acted = Acted(
                seq=seq,
                for_seq=int(payload["for_seq"]),
                scope=payload["scope"],
                tool=payload["tool"],
                why=payload["why"],
                args=dict(payload["args"]),
                undo=dict(payload["undo"]),
                gated=bool(payload["gated"]),
                at=str(payload["at"]),
            )
            self._acted[seq] = acted
            if not acted.gated:
                self._unreported.append(acted)
            if acted.tool == "follow_up":
                task = acted.args.get("task")
                if isinstance(task, int) and RESEARCH in self._grants:
                    self._tasks[task] = str(acted.args.get("what", ""))
        elif kind == episodes.GRANT_WITHDRAWN:
            n = int(payload["proposal"])
            hit = self._lapsed.pop(n, None)
            p = hit[0] if hit is not None else self._pending.pop(n, None)
            if p is not None:
                reason = str(payload["reason"])
                self._closed[n] = (p.for_seq, reason)
                self._unwithdrawn.append((p, reason))
        elif kind == episodes.WORK_FINISHED:
            task = int(payload["for_seq"])
            self._tasks.pop(task, None)
            self._lapse(
                lambda p: any(
                    a.get("tool") == "follow_up" and (a.get("args") or {}).get("task") == task
                    for a in p.actions
                ),
                f"research task {task} is finished",
            )
        elif kind in (episodes.SCHEDULE_CANCELLED, episodes.SCHEDULE_CREATED):
            # A tidy proposal is about one definition of one schedule; once
            # that schedule is stopped or redefined, a yes has nothing to run.
            sid = payload["id"]
            why = (
                f"{sid} is no longer running" if kind == episodes.SCHEDULE_CANCELLED
                else f"{sid} changed since I asked"
            )
            self._lapse(
                lambda p: any(
                    a.get("tool") == "tidy" and (a.get("args") or {}).get("id") == sid
                    for a in p.actions
                ),
                why,
            )
        elif kind == episodes.PROMISE_CLOSED:
            # A promise's reminder is proposed under ``promises`` (DL-087 #1);
            # once that promise is closed (kept, dropped, or reminded), a yes
            # must not set a reminder for it. A proposal naming only closed
            # promises is withdrawn; one naming others as well loses just the
            # closed promise's reminder, and a yes runs the rest as proposed.
            done = int(payload["promise"])

            def names(a: dict[str, Any]) -> bool:
                return (a.get("args") or {}).get("promise") == done

            self._lapse(
                lambda p: p.scope == PROMISES and bool(p.actions)
                and all(names(a) for a in p.actions),
                {
                    "done": f"you kept promise {done}",
                    "dropped": f"you dropped promise {done}",
                }.get(str(payload.get("how")), f"promise {done} is already closed"),
            )
            for n, p in list(self._pending.items()):
                if p.scope == PROMISES and any(names(a) for a in p.actions):
                    self._pending[n] = replace(
                        p, actions=tuple(a for a in p.actions if not names(a))
                    )
        elif kind == episodes.TURN_COMPLETED:
            if payload.get("outcome") == "spoke":
                self._said = int(payload["for_seq"])
        elif kind == episodes.TURN_BLOCKED:
            self._said = int(payload["for_seq"])
        elif kind == episodes.MESSAGE_INBOUND:
            from omega.tools import is_brief  # lazy: tools imports this module

            if is_brief(payload):
                self._briefed[seq] = tuple(self._unreported)
                self._unreported = []
                self._withdrawn[seq] = tuple(self._unwithdrawn)
                self._unwithdrawn = []
        self._through = seq

    def _lapse(self, which: Any, reason: str) -> None:
        """Take every pending proposal ``which`` matches out of reach of a
        yes, now, and keep it to be written down as withdrawn."""
        for p in [p for p in self._pending.values() if which(p)]:
            del self._pending[p.seq]
            self._lapsed[p.seq] = (p, reason)
            self._closed[p.seq] = (p.for_seq, reason)

    def _reset(self, scope: str, *, gated: bool) -> None:
        g = self._grants.get(scope)
        if g is not None:
            self._grants[scope] = Grant(
                scope=scope, for_seq=g.for_seq, at=g.at, gated=gated, streak=0
            )

    def state(self) -> State:
        return State(
            through=self._through,
            grants=dict(self._grants),
            pending=tuple(sorted(self._pending.values(), key=lambda p: p.seq)),
            acted=dict(self._acted),
            # Only the latest brief's list is ever asked for; keeping one keeps
            # the snapshot bounded by what is in flight, not by the log.
            briefed=dict(list(self._briefed.items())[-1:]),
            tasks=dict(self._tasks),
            closed=dict(self._closed),
            lapsed=tuple(self._lapsed[k] for k in sorted(self._lapsed)),
            withdrawn=dict(list(self._withdrawn.items())[-1:]),
            said=self._said,
        )

    @classmethod
    def fold(cls, decoded: Iterable[tuple[int, dict[str, Any]]]) -> "Grants":
        view = cls()
        for seq, payload in decoded:
            view.apply(seq, payload)
        return view

    def advance(self, store: "MemoryStore", *, upto: Optional[int] = None) -> "Grants":
        """Fold what the log gained since :attr:`through`, to ``upto`` (as
        :meth:`omega.derive.OpenWork.advance`)."""
        for episode in store.episodes_since(self._through):
            if upto is not None and episode.seq > upto:
                break
            self.apply(episode.seq, episodes.decode(episode.payload))
        return self

    @classmethod
    def rebuild(cls, store: "MemoryStore") -> "Grants":
        return cls().advance(store)


# --- reading an answer --------------------------------------------------------

_YES = frozenset(
    {"yes", "y", "yep", "yeah", "yup", "sure", "ok", "okay", "approve",
     "approved", "go", "fine", "do"}
)
_NO = frozenset(
    {"no", "n", "nope", "nah", "don't", "dont", "reject", "rejected",
     "decline", "not", "stop"}
)
_FILLER = frozenset(
    {"please", "thanks", "thank", "you", "do", "it", "ahead", "that", "this",
     "one", "go", "sounds", "good", "now", "to", "for", "proposal", "the"}
)
_PID = re.compile(r"^p(\d+)$")


#: The most words a clean answer may run to. Room for a few ids beside the
#: yes or no ("yes p12 p14 p15 please"), and still far short of a sentence.
MAX_ANSWER_WORDS = 16


def _answer(text: str) -> Optional[tuple[bool, tuple[int, ...]]]:
    """``(yes, named ids)`` if ``text`` is a clean yes or no, else ``None``.

    Clean means: it opens with a yes or a no word, and every other word is
    filler or a ``p<id>``. "yes but only the first" is not clean, and is left
    to the model, because acting on half a sentence is acting on a guess.
    Several ids may be named; one yes or no covers all of them.
    """
    words = [w for w in re.split(r"[\s,.!;:]+", text.strip().lower()) if w]
    if not words or len(words) > MAX_ANSWER_WORDS:
        return None
    named: list[int] = []
    rest: list[str] = []
    for w in words:
        m = _PID.match(w)
        if m:
            if int(m.group(1)) not in named:
                named.append(int(m.group(1)))
        else:
            rest.append(w)
    if not rest:
        return None
    head, tail = rest[0], rest[1:]
    if head in _YES:
        yes = True
    elif head in _NO:
        yes = False
    else:
        return None
    if any(w not in _FILLER and w not in (_YES if yes else _NO) for w in tail):
        return None
    return yes, tuple(named)


@dataclass(frozen=True)
class Answer:
    """Their reply, read against the proposals.

    ``open`` are the proposals it answers, each with the yes or no;
    ``closed`` the ones it names that can no longer be answered, each with
    why. Truthy when it answers or names anything, so a turn that holds one
    settles it in code before any model is asked.
    """

    open: tuple[tuple[Proposal, bool], ...] = ()
    closed: tuple[tuple[str, str], ...] = ()

    def __bool__(self) -> bool:
        return bool(self.open or self.closed)


def answers(
    state: Optional[State],
    event: dict[str, Any],
    open_work: Sequence[Any] = (),
) -> Optional[Answer]:
    """Is this message their answer to proposals, and to which?

    Only a person's own message answers, and only when the words are a clean
    yes or no. Which proposals, in order: the ones it names (``p<id>``); the
    one whose turn it replies to (``resumes_seq``, as the tray sends);
    otherwise the only one pending, and only while it is the newest open
    question: the newest thing omega said was the message that asked it,
    and no question is open that is newer. That last condition is what keeps
    a "yes" to some other "May I?" from being read as a yes to a proposal
    asked earlier.

    A proposal it names, or whose question it plainly answers, that has been
    answered or withdrawn or no longer applies comes back in ``closed``, with
    why, so the reply can say so and run nothing.
    """
    if state is None or not (state.pending or state.closed):
        return None
    if event.get("kind") != episodes.MESSAGE_INBOUND:
        return None
    from omega.notice import UNPROMPTED_CHANNELS  # lazy, as above

    if str(event.get("channel", "")) in UNPROMPTED_CHANNELS:
        return None
    read = _answer(str(event.get("text", "")))
    if read is None:
        return None
    yes, named = read

    def closed(seq: int) -> tuple[str, str]:
        return pid(seq), state.closed[seq][1]

    if named:
        hits = [(state.proposal(n), n) for n in named]
        out = Answer(
            open=tuple((p, yes) for p, _ in hits if p is not None),
            closed=tuple(closed(n) for p, n in hits if p is None and n in state.closed),
        )
        return out or None
    resumes = event.get("resumes_seq")
    if isinstance(resumes, int):
        hits = [p for p in state.pending if p.for_seq == resumes]
        if hits:
            return Answer(open=((hits[-1], yes),))
        gone = [n for n, (asked, _) in state.closed.items() if asked == resumes]
        return Answer(closed=(closed(max(gone)),)) if gone else None
    def newer_question(after: int, asked: int) -> bool:
        # An open question from another turn, asked after this proposal.
        return any(
            getattr(b, "seq", 0) > after and getattr(b, "for_seq", None) != asked
            for b in open_work
        )

    if len(state.pending) == 1:
        [only] = state.pending
        if only.for_seq != state.said or newer_question(only.seq, only.for_seq):
            return None
        return Answer(open=((only, yes),))
    if state.pending:
        return None
    # Nothing pending: a plain yes to a question omega asked last, which has
    # since closed, is told so rather than handed to the model as a yes.
    gone = [n for n, (asked, _) in state.closed.items() if asked == state.said]
    if not gone or newer_question(max(gone), state.said):
        return None
    return Answer(closed=(closed(max(gone)),))


def reading(
    state: Optional[State],
    event: dict[str, Any],
    open_work: Sequence[Any] = (),
) -> Optional[tuple[Proposal, bool]]:
    """The one proposal this message answers, when it answers exactly one
    open proposal (DL-085's reading; :func:`answers` is the whole of it)."""
    got = answers(state, event, open_work)
    if got is None or len(got.open) != 1 or got.closed:
        return None
    return got.open[0]


def briefing(state: Optional[State], seq: int, *, now: str) -> str:
    """What a brief adds, in code, about proposals (DL-087 #8): every one
    still waiting on them, and each withdrawn since the last brief, one line
    apiece. Empty when there is neither, so a brief over a log with no
    proposals reads exactly as before."""
    if state is None:
        return ""
    lines: list[str] = []
    if state.pending:
        lines.append("Waiting on you:")
        for p in state.pending:
            old = " (old)" if is_old(p, now) else ""
            lines.append(
                f"- {p.pid} — {p.words()}, since {since_words(p.at)}{old}. "
                f"{how_to_answer(p.pid)}"
            )
    for p, reason in state.withdrawn_for_brief(seq):
        lines.append(f"No longer asking {p.pid} ({p.words()}): {reason}.")
    return "\n".join(lines)


def withdraw_lapsed(
    append: Any, state: State, *, at: Optional[str] = None
) -> list[int]:
    """Write ``grant.withdrawn`` for every proposal that lapsed (DL-087 #8).

    ``state`` must be read at the log's head, by a fold of its own: the one a
    turn holds stops at that turn's event. Returns the seqs written. The fold
    already keeps a lapsed proposal from being run; this is what makes its
    withdrawal a fact in the log, for the next brief to tell.
    """
    written = []
    for p, reason in state.lapsed:
        written.append(append(episodes.grant_withdrawn(proposal=p.seq, reason=reason, at=at)))
    return written


def undo_for(tool: str, args: dict[str, Any]) -> dict[str, Any]:
    """How to take back one granted action, from the arguments it ran with."""
    if tool == "tidy":
        return {"restore": dict(args["was"])}
    if tool == "follow_up":
        return {"cancel": str(args["id"])}
    if tool == "remind" and args.get("id"):
        # A promise's reminder (DL-087 #1): set under a known id, so taking it
        # back is cancelling that schedule.
        return {"cancel": str(args["id"])}
    if tool == "topics" and args.get("do") == "follow" and args.get("name"):
        # A topic followed on their yes (DL-087 #6): taken back by dropping it.
        return {"drop": str(args["name"])}
    raise ValueError(f"{tool} is not an action a grant covers")
