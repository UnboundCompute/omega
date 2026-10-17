"""The thinking journal (C4, M2): what each thinking slot thought.

One ``journal.written`` per slot, append-only, folded here for two readers:
the next slot, which is shown the last few entries (the Sunday review the
week's), and the ``journal`` tool, which reads them back on their own message
when they ask (DL-118). Nothing here pushes an entry to them.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Iterable, Optional, Sequence

from omega import episodes

if TYPE_CHECKING:
    from omega.memory import MemoryStore

#: How the journal is named as a source: an entry can repeat mail (DL-108).
SOURCE = "omega's journal"
#: Entries a slot is shown; the Sunday review is shown the week's.
RECENT = 3
WEEK_DAYS = 7
MAX_WEEK = 21


@dataclass(frozen=True)
class Entry:
    seq: int
    payload: dict

    @property
    def at(self) -> str:
        return str(self.payload.get("at", ""))

    @property
    def tainted(self) -> tuple[str, ...]:
        return tuple(self.payload.get("tainted") or ())


class Journal:
    """The fold, kept at the log's head as the executor's other views."""

    __slots__ = ("_through", "_entries")

    def __init__(self) -> None:
        self._through = 0
        self._entries: list[Entry] = []

    def apply(self, seq: int, payload: dict[str, Any]) -> None:
        self._through = seq
        if payload.get("kind") == episodes.JOURNAL_WRITTEN:
            self._entries.append(Entry(seq=seq, payload=payload))

    def advance(self, store: "MemoryStore", *, upto: Optional[int] = None) -> "Journal":
        for episode in store.episodes_since(self._through):
            if upto is not None and episode.seq > upto:
                break
            self.apply(episode.seq, episodes.decode(episode.payload))
        return self

    def entries(self) -> tuple[Entry, ...]:
        return tuple(self._entries)

    @classmethod
    def fold(cls, decoded: Iterable[tuple[int, dict[str, Any]]]) -> "Journal":
        view = cls()
        for seq, payload in decoded:
            view.apply(seq, payload)
        return view


def entries(store: "MemoryStore") -> tuple[Entry, ...]:
    return Journal().advance(store).entries()


def recent(found: Sequence[Entry], *, now: datetime, week: bool = False) -> list[Entry]:
    """The entries a slot is shown: the last :data:`RECENT`, or for the
    Sunday review every one of the past week (at most :data:`MAX_WEEK`)."""
    if not week:
        return list(found[-RECENT:])
    since = now - timedelta(days=WEEK_DAYS)
    out = []
    for e in found:
        try:
            at = datetime.fromisoformat(e.at)
        except ValueError:
            continue
        if at.tzinfo is not None and at >= since:
            out.append(e)
    return out[-MAX_WEEK:]


def render(e: Entry) -> str:
    """One entry as text, each insight with the seqs it cites."""
    p = e.payload
    lines = [f"[journal {e.seq}] {p.get('slot')} ({e.at[:16]}), {p.get('outcome')}"]
    if p.get("changed"):
        lines.append(f"  changed: {p['changed']}")
    for q in p.get("questions") or ():
        lines.append(f"  asked myself: {q}")
    for i in p.get("insights") or ():
        lines.append(f"  insight: {i['text']} (seqs {', '.join(map(str, i['seqs']))})")
    for d in p.get("decisions") or ():
        lines.append(f"  {d['key']}: {d['decision']} ({d['why']})")
    if p.get("outcome") == "silent" and p.get("why"):
        lines.append(f"  stayed quiet: {p['why']}")
    return "\n".join(lines)


def tool_text(found: Sequence[Entry], *, limit: int) -> str:
    """What the ``journal`` tool answers: the newest ``limit`` entries."""
    shown = list(found[-limit:])
    if not shown:
        return "The journal is empty: no thinking slot has written in it yet."
    return "\n\n".join(render(e) for e in shown)
