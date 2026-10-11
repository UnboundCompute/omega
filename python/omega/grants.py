"""Standing grants, and the gate they sit behind (DL-085).

A grant is authority the person handed over in their own message, for one of
three closed scopes: **research** (carry its own research tasks to the end),
**tidy** (tidy their reminders and watches) and **promises** (DL-087 #1: hold
them to what they say they will do, with a reminder). Nothing else can be
granted, and calendar, mail and anything that moves money are outside all
three.

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

**The seam for the approval queue (DL-087 #8).** Proposals are episodes with
ids (their own seq, shown as ``p<seq>``), and :func:`reading` resolves one from
the person's reply. Discord and the tray reach it the same way, as an inbound;
a queue that lists pending proposals and answers them by id needs only to
write that inbound (or name ``p<seq>`` in it). Nothing here assumes the queue
exists.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable, Optional, Sequence

from omega import episodes

if TYPE_CHECKING:  # pragma: no cover
    from omega.memory import MemoryStore

__all__ = [
    "RESEARCH",
    "TIDY",
    "PROMISES",
    "SCOPES",
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
    "undo_for",
    "pid",
]

RESEARCH = "research"
TIDY = "tidy"
PROMISES = "promises"
SCOPES = episodes.GRANT_SCOPES
assert {RESEARCH, TIDY, PROMISES} == SCOPES

#: Yeses in a row, for one scope, before omega asks whether it may stop
#: asking. Asks, never drops.
GATE_DROP_AFTER = 10

#: The tools a grant covers, and so the only ones a gate sits in front of.
#: Every other tool is either a read (no gate needed) or outside every grant.
GATED_TOOLS = {"tidy": TIDY, "follow_up": RESEARCH}

#: Tools a grant covers only when *code* proposes them under it, never when
#: the model asks (DL-087 #1): a reminder for a promise they made is proposed
#: under ``promises``, while a ``remind`` they ask for in so many words runs as
#: it always has. Kept apart from :data:`GATED_TOOLS` for exactly that reason:
#: listing ``remind`` there would put a gate in front of every reminder.
PROPOSED_TOOLS = {"remind": PROMISES}

# --- reversibility ----------------------------------------------------------

#: Reads. Nothing changes anywhere.
READ = "read"
#: Writes only omega's own log or store, and can be undone from it.
OWN = "own"
#: Reaches outside omega but can be taken back. No tool is here today: it is
#: named so that the first one has a place other than "irreversible".
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
    proposes: str  # "action" or "drop_gate"
    actions: tuple[dict[str, Any], ...] = ()

    @property
    def pid(self) -> str:
        return pid(self.seq)

    def words(self) -> str:
        """What it asks, in words a person can answer."""
        if self.proposes == "drop_gate":
            return f"stop asking first before {self.scope} actions"
        return "; ".join(str(a.get("why", a.get("tool"))) for a in self.actions)


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
        return f"undo(action={self.seq}) stops {self.undo.get('cancel')}"


def pid(seq: int) -> str:
    return f"p{seq}"


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

    def __bool__(self) -> bool:
        return bool(self.grants or self.pending or any(self.briefed.values()))


class Grants:
    """The fold. ``apply`` in seq order; :meth:`state` snapshots it."""

    __slots__ = ("_through", "_grants", "_pending", "_acted", "_unreported", "_briefed", "_tasks")

    def __init__(self) -> None:
        self._through = 0
        self._grants: dict[str, Grant] = {}
        self._pending: dict[int, Proposal] = {}
        self._acted: dict[int, Acted] = {}
        self._unreported: list[Acted] = []
        self._briefed: dict[int, tuple[Acted, ...]] = {}
        self._tasks: dict[int, str] = {}

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
            # under: a later yes to it must not run it.
            for p in [p for p in self._pending.values() if p.scope == scope]:
                del self._pending[p.seq]
            if scope == RESEARCH:
                self._tasks.clear()
        elif kind == episodes.GRANT_RESTORED:
            self._reset(payload["scope"], gated=True)
            for p in [
                p for p in self._pending.values()
                if p.scope == payload["scope"] and p.proposes == "drop_gate"
            ]:
                del self._pending[p.seq]
        elif kind == episodes.GRANT_PROPOSED:
            self._pending[seq] = Proposal(
                seq=seq,
                for_seq=int(payload["for_seq"]),
                scope=payload["scope"],
                proposes=payload["proposes"],
                actions=tuple(dict(a) for a in payload["actions"]),
            )
        elif kind == episodes.GRANT_ANSWERED:
            p = self._pending.pop(int(payload["proposal"]), None)
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
        elif kind == episodes.WORK_FINISHED:
            self._tasks.pop(int(payload["for_seq"]), None)
        elif kind == episodes.MESSAGE_INBOUND:
            from omega.tools import is_brief  # lazy: tools imports this module

            if is_brief(payload):
                self._briefed[seq] = tuple(self._unreported)
                self._unreported = []
        self._through = seq

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


def _answer(text: str) -> Optional[tuple[bool, Optional[int]]]:
    """``(yes, named id)`` if ``text`` is a clean yes or no, else ``None``.

    Clean means: it opens with a yes or a no word, and every other word is
    filler or a ``p<id>``. "yes but only the first" is not clean, and is left
    to the model, because acting on half a sentence is acting on a guess.
    """
    words = [w for w in re.split(r"[\s,.!;:]+", text.strip().lower()) if w]
    if not words or len(words) > 8:
        return None
    named: Optional[int] = None
    rest: list[str] = []
    for w in words:
        m = _PID.match(w)
        if m:
            named = int(m.group(1))
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
    return yes, named


def reading(
    state: Optional[State],
    event: dict[str, Any],
    open_work: Sequence[Any] = (),
) -> Optional[tuple[Proposal, bool]]:
    """Is this message their answer to a pending proposal, and which one?

    Only a person's own message answers, and only when the words are a clean
    yes or no. Which proposal, in order: one it names (``p<id>``); the one
    whose turn it replies to (``resumes_seq``, as the tray sends); otherwise
    the only one pending, and only if it is also the newest open question.
    That last condition is what keeps a "yes" to some other "May I?" from
    being read as a yes to a proposal asked earlier.
    """
    if state is None or not state.pending:
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
    if named is not None:
        p = state.proposal(named)
        return (p, yes) if p is not None else None
    resumes = event.get("resumes_seq")
    if isinstance(resumes, int):
        hits = [p for p in state.pending if p.for_seq == resumes]
        return (hits[-1], yes) if hits else None
    if len(state.pending) != 1:
        return None
    [only] = state.pending
    blocks = sorted(open_work, key=lambda b: getattr(b, "seq", 0))
    if not blocks or getattr(blocks[-1], "for_seq", None) != only.for_seq:
        return None
    return only, yes


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
    raise ValueError(f"{tool} is not an action a grant covers")
