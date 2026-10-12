"""Looking back over the whole conversation (DL-094).

Recall is the last :data:`omega.turn.RECALL_N` episodes, and the ``recall``
tool searches learned claims. Neither reaches what was *said* three months
ago, so "what did I say about the lease in August" was answered from nothing.
This is the read that reaches it: keyword and date search over every
conversational episode in the log, newest first, each hit carrying when it was
said so the answer can say "on 14 Aug you said…".

**What counts as conversation.** Their own messages (``message.inbound`` on a
channel a person types on), what omega said back (``turn.completed`` with a
reply: *replied* to their message, or *told* them unprompted), and what it
stopped to ask (``turn.blocked``). A look's or a fire's inbound is omega's own
prompt, never their words, and is left out for the reason recall labels it.

**Read from the log, through the store the turn already holds.** The running
process holds the log's exclusive lock, so nothing here opens it; it reads
through the store's own ``episodes_since``.

**An index, because a scan was measured too slow.** On a synthetic log of
50,000 episodes (about 14,000 of them conversation), a full scan per call took
~340 ms, over the 200 ms bar, even with the byte pre-filter that skips decoding
anything that is not conversation. So the scan is kept, folded once, in
:class:`Index`: conversational lines with their search form, held in this
process's memory beside the store and extended from where it stopped on each
call. A query is then a substring test per line, a few milliseconds. It is
derived from the log and from nothing else, is never written anywhere, and is
rebuilt from the log whenever the process starts or the log looks shorter than
what it has read — so it can lag nothing and drift from nothing.

**No embeddings.** Matching is every term, case- and accent-insensitive, as
words a person would type to find their own sentence. Embeddings stay a
deferred decision.

**Redaction.** A turn that opened held mail (DL-082 #4) may have read a
recovery code out loud. The log keeps only a redacted digest of the mail, and
what omega said in that turn is redacted here the same way before it is
indexed.
"""

from __future__ import annotations

import bisect
import unicodedata
import weakref
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Iterable, Optional

from omega import episodes
from omega import inbox as inboxes

__all__ = [
    "DEFAULT_LIMIT",
    "MAX_LIMIT",
    "EXCERPT_CHARS",
    "Said",
    "Index",
    "past",
    "fold",
    "tool_text",
]

#: Hits one answer carries by default, and at most.
DEFAULT_LIMIT = 10
MAX_LIMIT = 20

#: Characters of context kept on each side of the match.
EXCERPT_CHARS = 90

_INBOUND = f'"kind":"{episodes.MESSAGE_INBOUND}"'.encode()
_COMPLETED = f'"kind":"{episodes.TURN_COMPLETED}"'.encode()
_BLOCKED = f'"kind":"{episodes.TURN_BLOCKED}"'.encode()
_RETURNED = f'"kind":"{episodes.TOOL_RETURNED}"'.encode()

YOU = "you said"
REPLIED = "omega replied"
TOLD = "omega told them"
ASKED = "omega asked"


@dataclass(frozen=True)
class Said:
    """One thing said, by them or by omega, and when."""

    seq: int
    at: Optional[datetime]
    who: str
    text: str
    #: ``text`` as searched: see :func:`fold`.
    folded: str = field(default="", compare=False, repr=False)


def fold(text: str) -> str:
    """Case- and accent-insensitive form: "Café" and "CAFE" both read "cafe"."""
    if text.isascii():
        return text.lower()
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold()


class Index:
    """The conversation in one log, folded once and extended as it grows.

    Fed by :meth:`extend` from ``episodes_since(upto)``, so each call reads
    only what was appended since the last. The kind is read off the encoded
    bytes before anything is decoded (:func:`omega.episodes.encode` writes
    sorted, compact JSON), so the bulk of a long log — tool calls, sense
    reports, ledger rows — costs a substring test each and nothing more.
    """

    def __init__(self) -> None:
        self.upto = 0
        self.lines: list[Said] = []
        self._seqs: list[int] = []
        self._prompted: set[int] = set()
        self._held: set[int] = set()

    def extend(self, history: Iterable[Any]) -> None:
        # Local: `notice` and `tools` both import this module's importers.
        from omega.notice import UNPROMPTED_CHANNELS
        from omega.tools import HELD_LEAD

        held_bytes = HELD_LEAD.encode()
        for episode in history:
            seq = episode.seq
            if seq <= self.upto:
                continue
            self.upto = seq
            raw = episode.payload
            if _INBOUND in raw:
                payload = episodes.decode(raw)
                if payload.get("kind") != episodes.MESSAGE_INBOUND:
                    continue
                if payload.get("channel") in UNPROMPTED_CHANNELS:
                    continue
                self._prompted.add(seq)
                self._add(seq, payload, YOU, str(payload.get("text") or ""))
            elif _COMPLETED in raw:
                payload = episodes.decode(raw)
                if payload.get("kind") != episodes.TURN_COMPLETED:
                    continue
                reply = payload.get("reply")
                if payload.get("outcome") != "spoke" or not isinstance(reply, str):
                    continue
                who = REPLIED if payload.get("for_seq") in self._prompted else TOLD
                self._add(seq, payload, who, reply)
            elif _BLOCKED in raw:
                payload = episodes.decode(raw)
                if payload.get("kind") != episodes.TURN_BLOCKED:
                    continue
                self._add(seq, payload, ASKED, str(payload.get("needs") or ""))
            elif _RETURNED in raw and held_bytes in raw:
                payload = episodes.decode(raw)
                result = payload.get("result")
                if (
                    payload.get("kind") == episodes.TOOL_RETURNED
                    and payload.get("ok")
                    and isinstance(result, str)
                    and result.startswith(HELD_LEAD)
                ):
                    self._held.add(payload.get("for_seq"))

    def _add(self, seq: int, payload: dict[str, Any], who: str, text: str) -> None:
        if not text.strip():
            return
        if who != YOU and payload.get("for_seq") in self._held:
            text = inboxes.redact(text)
        self.lines.append(Said(seq, _when(payload), who, text, fold(text)))
        self._seqs.append(seq)

    def before(self, seq: int) -> list[Said]:
        """The lines strictly before ``seq``, in log order. ``0`` is all."""
        if not seq:
            return list(self.lines)
        return self.lines[: bisect.bisect_left(self._seqs, seq)]


#: One index per open store, gone when the store is. Process memory only.
_INDEXES: "weakref.WeakKeyDictionary[Any, Index]" = weakref.WeakKeyDictionary()


def past(store: Any, *, before: int) -> list[Said]:
    """The conversation in ``store``'s log strictly before ``before``.

    The episode the turn is answering, and everything the turn wrote, are at
    or after ``before`` and never answer it: "what did I say about X" finding
    its own question is the one hit that is never wanted.
    """
    index = _INDEXES.get(store)
    head = store.head()
    if index is None or index.upto > head:
        # First read in this process, or a log shorter than what was read:
        # either way the index is rebuilt from the log, never trusted.
        index = Index()
        _INDEXES[store] = index
    if index.upto < head:
        index.extend(store.episodes_since(index.upto))
    return index.before(before)


def _when(payload: dict[str, Any]) -> Optional[datetime]:
    raw = payload.get("at")
    if not isinstance(raw, str):
        return None
    try:
        moment = datetime.fromisoformat(raw.strip())
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    # The person's zone, as the clock line reads it (DL-079).
    return moment.astimezone()


def terms(query: str) -> list[str]:
    return [t for t in (fold(w) for w in query.split()) if t]


def matches(
    found: list[Said],
    *,
    words: list[str],
    since: Optional[date] = None,
    until: Optional[date] = None,
) -> tuple[list[Said], list[Said]]:
    """``(searched, hits)``: the lines in the date range, and those of them
    holding every term. Both in log order."""
    if since is None and until is None:
        searched = found
    else:
        searched = [
            s for s in found
            if s.at is not None
            and (since is None or s.at.date() >= since)
            and (until is None or s.at.date() <= until)
        ]
    if not words:
        return searched, searched
    return searched, [s for s in searched if all(w in s.folded for w in words)]


def excerpt(text: str, words: list[str], width: int = EXCERPT_CHARS) -> str:
    """A short window of ``text`` around the earliest match, on one line."""
    flat = " ".join(text.split())
    where = min((p for p in (_locate(flat, w) for w in words) if p >= 0), default=0)
    start = max(0, where - width)
    end = min(len(flat), where + width)
    if start > 0:
        # Begin on a word, not inside one.
        space = flat.find(" ", start)
        start = space + 1 if 0 <= space < where else start
    if end < len(flat):
        space = flat.rfind(" ", where, end)
        end = space if space > where else end
    return ("…" if start > 0 else "") + flat[start:end] + ("…" if end < len(flat) else "")


def _locate(text: str, word: str) -> int:
    """Where ``word`` starts in ``text``, as an index into ``text`` itself
    (folding can change length, so the folded index is mapped back)."""
    if text.isascii():
        return text.lower().find(word)
    owners: list[int] = []
    pieces: list[str] = []
    for i, c in enumerate(text):
        f = fold(c)
        pieces.append(f)
        owners.extend([i] * len(f))
    k = "".join(pieces).find(word)
    return owners[k] if 0 <= k < len(owners) else -1


def _day(moment: Optional[datetime]) -> str:
    return f"{moment:%d %b %Y}" if moment is not None else "an unknown date"


def _stamp(moment: Optional[datetime]) -> str:
    return f"{moment:%a %d %b %Y, %H:%M}" if moment is not None else "(undated)"


def _span(since: Optional[date], until: Optional[date]) -> str:
    if since and until:
        return f" (searched {since:%d %b %Y} to {until:%d %b %Y})"
    if since:
        return f" (searched from {since:%d %b %Y})"
    if until:
        return f" (searched up to {until:%d %b %Y})"
    return ""


def tool_text(
    found: list[Said],
    *,
    query: str = "",
    since: Optional[date] = None,
    until: Optional[date] = None,
    limit: int = DEFAULT_LIMIT,
) -> str:
    """What the ``look_back`` tool answers: the newest ``limit`` hits, each
    dated, or an empty result that says how much was searched."""
    if not found:
        # Distinct from "nothing matched", as recall's empty memory is: an
        # empty log and a failed search read alike to a model about to say
        # one or the other.
        return "There is no past conversation in the log yet: nothing was said before this message."
    words = terms(query)
    searched, hits = matches(found, words=words, since=since, until=until)
    span = _span(since, until)
    if not hits:
        if not searched:
            return (
                f"No matches in 0 past messages{span}. The whole conversation "
                f"holds {len(found)} past messages since {_day(found[0].at)}."
            )
        what = f" for {query.strip()!r}" if words else ""
        return (
            f"No matches in {len(searched)} past messages since "
            f"{_day(searched[0].at)}{span}{what}."
        )
    shown = list(reversed(hits[-limit:]))
    about = f" mention {query.strip()!r}" if words else ""
    head = (
        f"{len(hits)} of {len(searched)} past messages since "
        f"{_day(searched[0].at)}{span}{about}; newest first"
        + (f", showing {len(shown)}" if len(shown) < len(hits) else "")
        + ":"
    )
    lines = [f"- {_stamp(s.at)} — {s.who}: {excerpt(s.text, words)}" for s in shown]
    if len(shown) < len(hits):
        lines.append(
            f"... {len(hits) - len(shown)} older matches not shown; narrow the "
            f"search or the dates to reach them."
        )
    return "\n".join([head, *lines])
