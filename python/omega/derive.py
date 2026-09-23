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

What is derived here is *what's open*: the half of DL-019's "recent N + what's
open" that :func:`omega.turn.recall` has always named in its docstring and
never had. Recall is the last ``RECALL_N`` episodes, so until now a question
omega stopped to ask simply **aged out of its own context** once forty episodes
went by — and DL-035 requires the opposite, that a block "survive until they
look", because after the clock landed there may be nobody watching when it is
raised.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable

from . import episodes

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .memory import MemoryStore

__all__ = ["OpenBlock", "OpenWork"]


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

    @classmethod
    def rebuild(cls, store: "MemoryStore") -> "OpenWork":
        """Replay the whole log. This is the boot path, and the only one.

        Deliberately a full replay rather than a resume from a checkpoint: the
        checkpoints beside the log are the *queue's* cursors, and a derived
        view that advanced them would be the second writer the queue's design
        exists to exclude.
        """
        view = cls()
        for episode in store.episodes_since(0):
            view.apply(episode.seq, episodes.decode(episode.payload))
        return view
