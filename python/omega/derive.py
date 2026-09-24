"""Derived views over the log — rebuilt, never stored (DL-041).

Everything here is a pure function of the episode log. Nothing in this module
persists anything, and nothing outside it may write what it computes back to
disk: the log is the only thing that survives a restart, and a derived view
that outlived the process would be a second source of truth that could disagree
with it.

DL-041 measured what that costs before choosing it — replay runs at roughly a
quarter of a million episodes a second, so a year of heavy use rebuilds in
about a third of a second. The benchmark lives in ``bench/replay.py`` so the
number can be re-taken rather than believed.

Two views live here.

:class:`OpenWork` derives *what's open*: the half of DL-019's "recent N + what's
open" that :func:`omega.turn.recall` has always named in its docstring and
never had. Recall is the last ``RECALL_N`` episodes, so until now a question
omega stopped to ask simply **aged out of its own context** once forty episodes
went by — and DL-035 requires the opposite, that a block "survive until they
look", because after the clock landed there may be nobody watching when it is
raised.

:class:`Learned` derives *what omega has been taught* (DL-042). Same shape, and
it is the same shape for a reason that is not tidiness: a learned habit changes
how omega behaves on turns that have nothing to do with the material that taught
it, so of everything in the system it is the thing that least deserves to live
somewhere a rebuild cannot reach. Deriving it from ``claim.extracted`` records
means there is no learned state to migrate, no learned state to corrupt, and no
way for what omega believes to drift from what the log says it was told.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Iterable, Optional

from . import episodes

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .memory import MemoryStore

__all__ = ["OpenBlock", "OpenWork", "Claim", "Learned", "local_hour"]


@dataclass(frozen=True)
class OpenBlock:
    """A turn that stopped to ask, with no answer yet.

    ``seq`` is the ``turn.blocked`` episode; ``for_seq`` is the event whose turn
    blocked. Both are kept because they answer different questions — ``seq``
    says when omega asked, ``for_seq`` says what it was doing — and because
    only one of them is what an answer names. See :meth:`OpenWork.apply`.
    """

    seq: int
    for_seq: int
    needs: str
    at: str


class OpenWork:
    """What omega is still waiting on a person for.

    Built by folding the log forward. Two episode kinds matter: a
    ``turn.blocked`` opens an obligation, and an inbound carrying
    ``resumes_seq`` discharges one.

    **Obligations are keyed by ``for_seq``, not by the blocking episode's own
    seq, and that is a wire fact rather than a preference.** The tray sets its
    pending resume from ``update.forSeq`` (``TrayViewModel.swift:405``), so the
    ``resumes_seq`` that comes back over the channel names *the event whose turn
    blocked* — not the ``turn.blocked`` record. ``episodes.inbound``'s docstring
    says an answer "answers an earlier ``turn.blocked``", which reads the other
    way and is the reason this is spelled out here: keying on the blocking seq
    would typecheck, pass any test written from the prose, and then never
    discharge anything in production, so omega would nag about questions that
    had already been answered. The key is whatever an answer actually names.

    One ``for_seq`` can carry at most one open block, because an event causes
    exactly one turn and a turn blocks at most once — ``turn.blocked`` is
    terminal. An answer that itself blocks opens a *new* obligation under the
    answer's own seq, which is correct: that is a different question.
    """

    __slots__ = ("_blocks", "_through")

    def __init__(self) -> None:
        self._blocks: dict[int, OpenBlock] = {}
        self._through = 0

    @property
    def through(self) -> int:
        """The last seq folded in. 0 means nothing has been applied.

        Exposed so a caller can tell "rebuilt and found nothing open" from
        "never rebuilt" — a view that answered the empty list for both would be
        a check that passes on empty, which is the one thing `CLAUDE.md` says a
        check may not do.
        """
        return self._through

    def apply(self, seq: int, payload: dict[str, Any]) -> None:
        """Fold one episode in. Must be called in seq order.

        Out-of-order application is refused rather than tolerated: the fold is
        order-dependent — an answer applied before the block it discharges
        would leave the block open forever — and silently accepting it would
        turn a caller's bug into omega nagging about a settled question.
        """
        if seq <= self._through:
            raise ValueError(
                f"episodes must be folded in order: got seq {seq} after "
                f"{self._through}"
            )
        kind = payload.get("kind")
        if kind == episodes.TURN_BLOCKED:
            for_seq = int(payload["for_seq"])
            self._blocks[for_seq] = OpenBlock(
                seq=seq,
                for_seq=for_seq,
                needs=str(payload.get("needs", "")),
                at=str(payload.get("at", "")),
            )
        elif kind == episodes.MESSAGE_INBOUND and "resumes_seq" in payload:
            # ``pop`` with a default: an answer naming something that was never
            # blocked is not an error here. It can arrive legitimately after a
            # restart, and a derived view is not the place to adjudicate the
            # channel's input.
            self._blocks.pop(int(payload["resumes_seq"]), None)
        self._through = seq

    def blocks(self) -> list[OpenBlock]:
        """Every open obligation, oldest first.

        Ordered by when omega asked, because that is the order a person owes
        answers in, and because a stable order keeps the rendered transcript
        diffable between turns.
        """
        return sorted(self._blocks.values(), key=lambda b: b.seq)

    def __len__(self) -> int:
        return len(self._blocks)

    @classmethod
    def fold(cls, decoded: Iterable[tuple[int, dict[str, Any]]]) -> "OpenWork":
        """Build from ``(seq, payload)`` pairs. The pure core, for tests."""
        view = cls()
        for seq, payload in decoded:
            view.apply(seq, payload)
        return view

    def advance(self, store: "MemoryStore", *, upto: Optional[int] = None) -> "OpenWork":
        """Fold what the log gained since :attr:`through`, to ``upto``. Returns self.

        This is the steady-state path, and it is what keeps the view off the
        per-turn cost sheet: ``episodes_since`` is exclusive of its argument, so
        a turn folds only what actually arrived rather than replaying the log —
        the difference between O(new) and O(log) on every turn, next to
        durability code where the second asymptote would not stay unnoticed.

        **``upto`` is not an optimisation.** Without it the fold runs to the
        head of the log, and a turn handling a backlog would be shown questions
        omega had not asked yet at that point in the conversation — worse than
        stale, because a view that runs ahead of the turn reading it is wrong in
        a direction nobody checks for. A caller handling one episode passes that
        episode's seq; a caller that genuinely wants "everything so far" omits
        it and says so by omitting it.

        Idempotent when nothing new is in range: the iterator yields nothing or
        is cut at once, no ``apply`` runs, and ``through`` does not move. So a
        caller may advance as often as it likes.
        """
        for episode in store.episodes_since(self._through):
            if upto is not None and episode.seq > upto:
                break
            self.apply(episode.seq, episodes.decode(episode.payload))
        return self

    @classmethod
    def rebuild(cls, store: "MemoryStore") -> "OpenWork":
        """Replay the whole log. This is the boot path, and the only one.

        Deliberately a full replay rather than a resume from a checkpoint: the
        checkpoints beside the log are the *queue's* cursors, and a derived
        view that advanced them would be the second writer the queue's design
        exists to exclude.
        """
        return cls().advance(store)


# --- what omega has been taught (DL-042) -------------------------------------


def local_hour(at: str) -> Optional[int]:
    """The local-clock hour of an episode timestamp, or ``None`` if unreadable.

    ``None`` is a real answer, and the caller must treat it as *does not match*
    rather than as *matches anything*. An hour window that cannot be evaluated
    falling open would turn "only while I'm working" into "always", which is
    precisely the spurious activation DL-042 pairs against the capability — and
    `CLAUDE.md`'s fail-closed-on-empty rule says which way an undeterminable
    check has to break.

    Timestamps are written in UTC by :func:`omega.episodes.now`, and the window
    is a *human* one — "while I'm at work" is a statement about the clock on the
    wall, not about UTC. So this converts. A naive timestamp is read as UTC,
    because that is what the log writes; it is not read as local, which would
    silently shift every window by the machine's offset.
    """
    try:
        parsed = datetime.fromisoformat(at)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone().hour


@dataclass(frozen=True)
class Claim:
    """One thing omega has been taught, and the condition it applies under.

    ``seq`` is the ``claim.extracted`` record. ``source_seq`` is the episode it
    was learned *from*, which is usually the event of the same turn and not
    always. ``situation`` is what was going on at the time — the provenance
    DL-034 asked for, kept because a contradiction cannot be re-decided without
    it and it cannot be reconstructed after the fact.

    ``explicit`` says the person authored this deliberately rather than omega
    inferring it, and it is what decides whether a contradiction is worth
    interrupting for (DL-042). It is carried on the claim rather than computed
    later because it is a fact about how the claim arrived, and that fact is
    only available while it is arriving.
    """

    seq: int
    text: str
    trigger: Optional[dict[str, Any]]
    situation: str
    source_seq: int
    explicit: bool
    supersedes: Optional[int] = None

    def fires_on(self, *, text: str, channel: str, hour: Optional[int]) -> bool:
        """Does this claim apply to the situation described?

        No trigger means always — the right representation for tone and working
        style, which have no situation because they apply to all of them. A
        trigger's fields are ANDed: a claim that names both a phrase and an hour
        window means *both*, because a person who names two conditions has
        narrowed, not widened.

        Matching is substring over a lowercased form, and that is as clever as
        it gets on purpose (DL-042). It will miss a situation described in words
        the claim did not anticipate. The mitigation is not a better matcher
        here — it is that the person sees the trigger in the receipt while they
        still remember what they meant, and can say so.
        """
        trigger = self.trigger
        if not trigger:
            return True

        phrases = trigger.get("any")
        if phrases is not None:
            haystack = text.lower()
            if not any(p.strip().lower() in haystack for p in phrases):
                return False

        wanted = trigger.get("channel")
        if wanted is not None and wanted != channel:
            return False

        window = trigger.get("hours")
        if window is not None:
            if hour is None:
                # Undeterminable, so it does not fire. See `local_hour`.
                return False
            start, end = int(window[0]), int(window[1])
            # Half-open, and wrapping is a real window: [22, 6] is the night,
            # which is a thing people say and would otherwise need two claims.
            inside = start <= hour < end if start < end else (hour >= start or hour < end)
            if not inside:
                return False

        return True


class Learned:
    """Everything omega has been taught that is still standing.

    Folded from ``claim.extracted`` records the same way :class:`OpenWork` is
    folded from blocks, and kept as a separate view rather than a second field
    on that one because the two answer different questions and have different
    lifetimes: an obligation is discharged by an answer and gone, a claim
    persists until something contradicts it.

    A claim leaves the active set two ways. **Supersession** is the model
    replacing a belief while writing a better one down; **retraction**
    (``claim.retracted``, DL-048) is the person removing one and putting nothing
    in its place. Only the second can be asked for, which is why it needed a
    kind of its own rather than a supersession with an empty replacement.

    **Both remove a claim from the active set and from nowhere else.** The
    record stays in the log, so re-derivation can always reach the earlier state
    and "destroy core memory" is impossible by construction rather than by a
    model getting a criticality test right (DL-034, DL-042). That is the
    property that lets ``explicit`` be a cheap static flag about *escalation*
    instead of a safety mechanism carrying weight it could not hold — and it is
    why retraction had to be an append even though "forget this" sounds like a
    delete.
    """

    __slots__ = ("_claims", "_through", "_failed", "_last_failure")

    def __init__(self) -> None:
        self._claims: dict[int, Claim] = {}
        self._through = 0
        self._failed = 0
        self._last_failure = ""

    @property
    def through(self) -> int:
        """The last seq folded in. 0 means nothing has been applied.

        Same reason :attr:`OpenWork.through` exists: "rebuilt and found nothing
        learned" and "never rebuilt" must be distinguishable, or the empty
        answer is a check that passes on empty.
        """
        return self._through

    def apply(self, seq: int, payload: dict[str, Any]) -> None:
        """Fold one episode in. Must be called in seq order.

        Order matters here for the same reason it does in :class:`OpenWork` and
        with a worse failure: a superseding claim applied before the claim it
        replaces would leave both active, so omega would hold two contradictory
        beliefs and act on whichever the renderer happened to reach first.
        """
        if seq <= self._through:
            raise ValueError(
                f"episodes must be folded in order: got seq {seq} after "
                f"{self._through}"
            )
        if payload.get("kind") == episodes.CLAIM_EXTRACTED:
            supersedes = payload.get("supersedes")
            if supersedes is not None:
                # ``pop`` with a default, as in ``OpenWork``: a claim naming a
                # supersession target that is not currently active is not an
                # error a derived view should adjudicate. The target may itself
                # have been superseded already, which is an ordinary race in an
                # append-only log and not a corruption.
                self._claims.pop(int(supersedes), None)
            self._claims[seq] = Claim(
                seq=seq,
                text=str(payload["text"]),
                trigger=payload.get("trigger"),
                situation=str(payload.get("situation", "")),
                source_seq=int(payload["source_seq"]),
                explicit=bool(payload.get("explicit", False)),
                supersedes=None if supersedes is None else int(supersedes),
            )
            # A claim reached the log, so whatever was failing is not failing
            # now. Counting *since the last success* rather than for all time is
            # what keeps this a live signal instead of a scar: a fault that was
            # found and fixed should stop being reported, or the line teaches the
            # person to skip it and the next real outage scrolls past inside it.
            self._failed = 0
            self._last_failure = ""
        elif payload.get("kind") == episodes.CLAIM_EXTRACTION_FAILED:
            self._failed += 1
            self._last_failure = str(payload["reason"])
        elif payload.get("kind") == episodes.CLAIM_RETRACTED:
            # ``pop`` with a default for the same reason supersession uses one:
            # naming a claim that is no longer active is an ordinary race in an
            # append-only log, not a corruption a derived view should adjudicate.
            # The *refusal* to retract an unknown id lives at extraction, where
            # there is a person to tell (DL-048 #4); by the time a record exists
            # the decision was already made and this only replays it.
            self._claims.pop(int(payload["claim_seq"]), None)
        self._through = seq

    def claims(self) -> list[Claim]:
        """Every active claim, oldest first."""
        return sorted(self._claims.values(), key=lambda c: c.seq)

    @property
    def failed(self) -> int:
        """Extractions that failed since the last one that filed a claim (DL-053).

        Zero is the ordinary state and also the state of a log that has never
        been taught anything, which is fine — the two are only confusable if you
        read this as "learning is healthy" rather than as what it says, which is
        a count. The question it answers is the one DL-052 could not:
        ``--learned`` reporting nothing learned means *nothing was taught* when
        this is 0 and *nothing could be recorded* when it is not.

        One known bias, and it is deliberate. A teach whose extraction succeeds
        but legitimately files nothing — a note that asks omega for nothing —
        writes no record and so does not clear this, leaving a repaired fault
        reported slightly too long. Over-reporting was chosen because the fault
        underneath is total and silent for as long as it lasts, and a count that
        lingers a teach too long is cheaper than one that goes quiet early.
        """
        return self._failed

    @property
    def last_failure(self) -> str:
        """Why the most recent extraction failed, or ``""``.

        The provider's own message, kept whole. It is the part that made DL-052
        a one-line repair instead of an investigation: it named the rejected
        parameter, the model that rejected it, and the environment variable that
        turns it off. Summarising that into a category would throw away the only
        part a person can act on.
        """
        return self._last_failure

    def matching(
        self, *, text: str, channel: str, at: str
    ) -> list[Claim]:
        """The claims that fire on this event, oldest first.

        This runs on **every** turn (DL-034), which is the whole reason the
        trigger vocabulary is structural: it is a few substring tests per claim
        against text omega already has in hand, with no model call and no
        retrieval. What it costs is proportional to how much omega has learned,
        which is the one growth curve that is affordable here.
        """
        hour = local_hour(at)
        return [
            claim
            for claim in self.claims()
            if claim.fires_on(text=text, channel=channel, hour=hour)
        ]

    def __len__(self) -> int:
        return len(self._claims)

    @classmethod
    def fold(cls, decoded: Iterable[tuple[int, dict[str, Any]]]) -> "Learned":
        """Build from ``(seq, payload)`` pairs. The pure core, for tests."""
        view = cls()
        for seq, payload in decoded:
            view.apply(seq, payload)
        return view

    def advance(self, store: "MemoryStore", *, upto: Optional[int] = None) -> "Learned":
        """Fold what the log gained since :attr:`through`, to ``upto``. Returns self.

        ``upto`` carries the same meaning and the same warning as it does on
        :meth:`OpenWork.advance`: a turn handling a backlog passes its own
        episode's seq, so the claims it is shown are the ones omega had actually
        been taught by that point. Folding to the head would apply a claim to a
        turn that happened before it was learned — which, unlike a stale view,
        looks entirely correct from the outside.
        """
        for episode in store.episodes_since(self._through):
            if upto is not None and episode.seq > upto:
                break
            self.apply(episode.seq, episodes.decode(episode.payload))
        return self

    @classmethod
    def rebuild(cls, store: "MemoryStore") -> "Learned":
        """Replay the whole log. The boot path, and the only one."""
        return cls().advance(store)
