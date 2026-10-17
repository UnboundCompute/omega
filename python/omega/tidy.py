"""Omega keeping its own memory clean (DL-087 #4).

Three jobs, one periodic pass, and every one of them an *append*:

- **Duplicates are merged.** Two claims whose words normalise to the same
  sentence under the same trigger are one belief said twice. Decided in code,
  never by a model: the schema has no subject or predicate to compare, so the
  only test that is a fact rather than a judgement is "is it the same sentence".
  A ``claim.merged`` names the survivor and the folded seqs; the originals stay
  in the log.
- **Changeable state goes stale.** A claim about where they work, where they
  live or what they plan is marked old once it is past its category's threshold
  (``claim.stale``), and recall shows it *with its age* rather than hiding it.
  A durable fact — a preference, a way of working — is never aged.
- **A contradiction is asked once, never guessed.** The model is asked only
  whether two standing claims cannot both be true; it never picks the winner.
  A ``claim.contradiction`` records the pair and the question; the question
  rides the next brief or spoken look (`turn.run_turn`), is recorded as asked,
  and is never asked again. Only the person's answer drops a claim
  (``claim.resolved``). Until then both stand, each marked as conflicting.

The pass is bounded (``MAX_*`` below) and idempotent: its cursors and marks live
in the log, so a rerun over the same log finds nothing left to do.

**The violation metric this is paired with**: no claim leaves the active set
unless a ``claim.merged`` folded it or a ``claim.resolved`` dropped it.
``tests/test_tidy.py`` checks it over every pass it runs.
"""

from __future__ import annotations

import json
import re
import unicodedata
from datetime import timedelta
from typing import Any, Callable, Optional, Sequence

from omega import episodes, provider
from omega.derive import Claim, Learned, _days_between, _parse_at

__all__ = [
    "STALE_AFTER_DAYS",
    "TIDY_EVERY",
    "MAX_MERGES_PER_PASS",
    "MAX_PAIRS_PER_PASS",
    "MAX_ANSWERS_PER_PASS",
    "NotJudged",
    "normalize",
    "category",
    "duplicates",
    "stale_due",
    "candidate_pairs",
    "question",
    "read_answer",
    "due",
    "run",
]

#: Days after which a changeable-state claim is marked old, by category.
#: **Labelled guesses**, unmeasured: a plan is usually about the next few weeks,
#: and a job or a home changes on the scale of years but is worth re-checking
#: after half of one. Marking is cheap to get wrong in either direction, because
#: a mark only adds the age to what recall shows — nothing is hidden.
STALE_AFTER_DAYS: dict[str, int] = {"plans": 30, "job": 180, "location": 180}

#: How often the pass runs. Daily, a labelled guess: memory changes at the pace
#: of conversation, and a pass is cheap when there is nothing new.
TIDY_EVERY = timedelta(hours=24)

#: Work bounds per pass, labelled guesses. The pair cap is what bounds the one
#: model call contradiction-checking makes; the rest waits for the next pass
#: behind a cursor in the log.
MAX_MERGES_PER_PASS = 25
MAX_PAIRS_PER_PASS = 12
MAX_ANSWERS_PER_PASS = 3

#: Words that say a claim is about changeable state, by category. Matched as
#: whole words over the normalised text. A keyword list rather than a model,
#: because staleness has to be deterministic and re-derivable; what it misses is
#: a claim that is simply not aged, which is the safe direction.
_CATEGORIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "job",
        (
            "works at", "work at", "working at", "works for", "work for",
            "working for", "job", "employer", "employed", "role at",
            "position at", "intern at", "joined",
        ),
    ),
    (
        "location",
        (
            "lives in", "live in", "living in", "based in", "moved to",
            "moving to", "located in", "staying in", "home is in", "address",
            "relocating",
        ),
    ),
    (
        "plans",
        (
            "plans to", "plan to", "planning to", "planning", "going to",
            "intends to", "intend to", "next week", "next month", "this weekend",
            "upcoming", "trip", "travelling", "traveling",
        ),
    ),
)

_ARTICLES = frozenset({"a", "an", "the"})
#: Too common to say two claims are about the same thing.
_STOP = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "to", "of", "in", "on", "at",
        "for", "with", "is", "are", "was", "be", "it", "they", "them", "their",
        "he", "she", "his", "her", "i", "me", "my", "you", "your", "when",
        "that", "this", "as", "by", "from", "not", "do", "does", "likes",
        "like", "prefers", "prefer", "wants", "want", "person", "user",
    }
)


class NotJudged(RuntimeError):
    """The model's answer about a contradiction or a reply was not usable."""


# --- deterministic halves ---------------------------------------------------


def normalize(text: str) -> str:
    """The claim as words to compare: NFKC, casefolded, punctuation and the
    three articles dropped, whitespace collapsed. Word order is kept, because
    reordering is where "same words" becomes "same meaning", and that is a
    judgement this function must not make."""
    folded = unicodedata.normalize("NFKC", text).casefold()
    words = re.sub(r"[^\w\s]", " ", folded).split()
    return " ".join(w for w in words if w not in _ARTICLES)


def _trigger_key(trigger: Optional[dict[str, Any]]) -> Any:
    if not trigger:
        return None
    phrases = trigger.get("any")
    hours = trigger.get("hours")
    return (
        trigger.get("channel"),
        None if hours is None else (int(hours[0]), int(hours[1])),
        None if phrases is None else frozenset(normalize(str(p)) for p in phrases),
    )


#: Channels nobody types on (omega's look, the clock, its tasks). Named here
#: rather than imported from `notice`, which imports this module's neighbours.
_UNPROMPTED = frozenset({"self", "schedule", "task"})


# The receipt each learning pass closes its claims with. A claim belongs to
# the first of these written after it: the executor is the single writer, and
# every pass files its claims, then its moments and lessons, then its receipt.
_RECEIPTS = (
    episodes.REFLECTION_DONE,
    episodes.TRANSCRIPT_INGESTED,
    episodes.USAGE_DIGESTED,
    episodes.AUDIO_CAPTURED,
)


def conversation_claims(log: Sequence[tuple[int, dict[str, Any]]]) -> set[int]:
    """Seqs of the claims a conversation reflection pass filed.

    Read off the receipt that closed the claim's pass: ``reflection.done``
    for the CONVERSATION lens, a transcript, usage or audio receipt for the
    digest lenses. The digest lenses anchor on the log's head because no turn
    is behind them, so after a look that head is the look's own record — an
    anchor that says nothing about whose words taught the claim. A claim no
    receipt closed (a crash mid-pass) belongs to no pass and is left out.
    """
    found: set[int] = set()
    pending: list[int] = []
    for seq, payload in log:
        kind = payload.get("kind")
        if kind == episodes.CLAIM_EXTRACTED:
            pending.append(seq)
        elif kind in _RECEIPTS:
            if kind == episodes.REFLECTION_DONE:
                found.update(pending)
            pending = []
    return found


def misattributed(store: Any, claims: Sequence[Claim]) -> list[tuple[Claim, int]]:
    """Inferred claims a reflection pass anchored on omega's own fire (C0),
    each with the fire it was anchored on.

    Before C0 a pass read every channel and named the log's head as its
    cause, so omega's own watch rules came back as things it had learned
    about the person. The class is read from structure only: an inferred
    (``explicit=False``) claim a conversation reflection pass filed
    (:func:`conversation_claims`) whose ``for_seq`` or ``source_seq`` is an
    inbound on an unprompted channel, or the turn record that answered one.
    A claim the person taught is explicit and never matches. A claim learned
    from a transcript, a recording or the day's usage never matches either,
    even when the head it was anchored on was a look's record: its pass
    closed with a digest receipt, not ``reflection.done``. Reflection now
    anchors on the last person message it read, so no new claim can match.
    """
    inferred = [c for c in claims if not c.explicit]
    if not inferred:
        return []
    log = [(e.seq, episodes.decode(e.payload)) for e in store.episodes_since(0)]
    payloads = dict(log)
    reflected = conversation_claims(log)
    return [
        (claim, hit)
        for claim in inferred
        if claim.seq in reflected
        and (hit := fire_anchor(payloads, payloads.get(claim.seq, {}).get("for_seq"),
                                claim.source_seq)) is not None
    ]


def fire_anchor(payloads: dict[int, dict[str, Any]], *seqs: Any) -> Optional[int]:
    """The unprompted inbound the first of ``seqs`` names, directly or
    through the turn record that answered it, else ``None``."""
    anchors = list(seqs)
    for seq in seqs:
        record = payloads.get(seq) if isinstance(seq, int) else None
        if record and record.get("kind") in (episodes.TURN_COMPLETED, episodes.TURN_BLOCKED):
            anchors.append(record.get("for_seq"))
    for seq in anchors:
        p = payloads.get(seq) if isinstance(seq, int) else None
        if p and p.get("kind") == episodes.MESSAGE_INBOUND and p.get("channel") in _UNPROMPTED:
            return seq
    return None


def duplicates(claims: Sequence[Claim]) -> list[tuple[int, list[int]]]:
    """Groups of claims that are one sentence said twice, as ``(kept, folded)``.

    Same normalised text *and* the same trigger: "keep it short" in the morning
    and "keep it short" always are two instructions. The survivor is the most
    authoritative — an explicit claim over an inferred one — and then the
    newest, so the merge never replaces what the person said with what omega
    guessed.
    """
    groups: dict[Any, list[Claim]] = {}
    for claim in claims:
        key = (normalize(claim.text), _trigger_key(claim.trigger))
        if not key[0]:
            continue
        groups.setdefault(key, []).append(claim)
    merges = []
    for group in groups.values():
        if len(group) < 2:
            continue
        kept = max(group, key=lambda c: (c.explicit, c.seq))
        merges.append((kept.seq, sorted(c.seq for c in group if c.seq != kept.seq)))
    return sorted(merges)


def category(text: str) -> Optional[str]:
    """Which changeable state a claim is about, or ``None`` for a durable one."""
    words = f" {normalize(text)} "
    for name, cues in _CATEGORIES:
        if any(f" {cue} " in words for cue in cues):
            return name
    return None


def stale_due(
    claims: Sequence[Claim], *, marked: Any, now: str
) -> list[tuple[Claim, str, str, int]]:
    """Claims past their category's threshold and not already marked, as
    ``(claim, category, basis_at, age_days)``. An undatable claim is not aged:
    "could not tell how old" is not "old"."""
    due = []
    for claim in claims:
        name = category(claim.text)
        if name is None or claim.seq in marked or not claim.at:
            continue
        age = _days_between(claim.at, now)
        if age is not None and age >= STALE_AFTER_DAYS[name]:
            due.append((claim, name, claim.at, age))
    return due


def _content(text: str) -> set[str]:
    return {w for w in normalize(text).split() if w not in _STOP and len(w) > 2}


#: How alike two claims' content words must be (Dice) to count as one claim
#: said twice (C0). A starting guess [A]: exact-text dedup let paraphrases of
#: one watch rule through three times; revise from the live misses.
SIMILAR = 0.6


def similarity(a: str, b: str) -> float:
    """Dice overlap of two texts' content words, 0 to 1."""
    x, y = _content(a), _content(b)
    if not x or not y:
        return 0.0
    return 2 * len(x & y) / (len(x) + len(y))


def novel(
    claims: Sequence[dict[str, Any]], known: Sequence[Claim], *, threshold: float = SIMILAR
) -> list[dict[str, Any]]:
    """``claims`` (proposed, not yet filed) without the ones that restate a
    known claim or an earlier one in the same batch. A claim that names what
    it supersedes is a refinement, so it is not checked against that one."""
    kept: list[dict[str, Any]] = []
    for item in claims:
        text = str(item.get("text", ""))
        against = [c.text for c in known if c.seq != item.get("supersedes")]
        against += [str(k["text"]) for k in kept]
        if any(similarity(text, other) >= threshold for other in against):
            continue
        kept.append(item)
    return kept


def candidate_pairs(
    claims: Sequence[Claim],
    *,
    checked_through: int,
    recorded: Any,
    limit: int = MAX_PAIRS_PER_PASS,
) -> tuple[list[tuple[Claim, Claim]], int]:
    """Pairs worth asking the model about, and the cursor they take it to.

    Each pair is considered once, when its newer claim is past the cursor. A
    pair is a candidate when both are the same kind of changeable state or they
    share at least two content words — a prefilter, so the model is never
    shown the whole set squared. Claims are taken whole, in seq order, while
    the batch has room; the cursor stops before the first one that does not fit,
    so nothing is skipped. One claim with more candidates than the cap on its
    own takes its closest ``limit`` and the cursor passes it: a labelled loss,
    bounded, and better than a cursor that never moves.
    """
    ordered = sorted(claims, key=lambda c: c.seq)
    batch: list[tuple[Claim, Claim]] = []
    cursor = checked_through
    for index, claim in enumerate(ordered):
        if claim.seq <= checked_through:
            continue
        mine = category(claim.text)
        words = _content(claim.text)
        found = []
        for other in ordered[:index]:
            if frozenset((other.seq, claim.seq)) in recorded:
                continue
            shared = len(words & _content(other.text))
            same_state = mine is not None and category(other.text) == mine
            if same_state or shared >= 2:
                found.append((shared, other))
        if len(batch) + len(found) > limit:
            if batch:
                break
            found = sorted(found, key=lambda f: (-f[0], -f[1].seq))[:limit]
        batch.extend((other, claim) for _, other in sorted(found, key=lambda f: f[1].seq))
        cursor = claim.seq
    return batch, cursor


def question(older: Claim, newer: Claim) -> str:
    """The one question a contradiction is asked as. Rendered by code, so what
    the person is asked is exactly the two claims and nothing a model added."""
    return (
        f'I have two things written down that cannot both be true: '
        f'1) "{older.text}" ({_day(older.at)}) and 2) "{newer.text}" '
        f"({_day(newer.at)}). Which is right now? Reply 1 or 2."
    )


def _day(at: Optional[str]) -> str:
    parsed = _parse_at(at)
    return "date unknown" if parsed is None else parsed.date().isoformat()


_BARE = re.compile(r"^\s*(?:#|no\.?\s*|number\s+|option\s+)?([12])\s*[.!)]*\s*$", re.I)


def read_answer(text: str) -> Optional[int]:
    """A bare "1" or "2", read without a model, or ``None``."""
    match = _BARE.match(text)
    return int(match.group(1)) if match else None


# --- the model halves -------------------------------------------------------


def _parse_json(text: str) -> Any:
    from omega.learn import _unfence  # local: one fence-stripper, not two

    try:
        return json.loads(_unfence(text))
    except (TypeError, ValueError) as exc:
        raise NotJudged(f"answer was not JSON: {exc}") from exc


def judge_pairs(
    complete: Callable[..., provider.Response],
    pairs: Sequence[tuple[Claim, Claim]],
) -> set[int]:
    """Which of ``pairs`` (0-based) cannot both be true. Raises :class:`NotJudged`.

    The model is asked *whether*, never *which is right*: picking a winner is
    the guess DL-087 #4 rules out, and only the person can make it.
    """
    listed = "\n".join(
        f'{n}. "{a.text}" / "{b.text}"' for n, (a, b) in enumerate(pairs, start=1)
    )
    messages = [
        provider.system(
            "Each numbered line is two things a person told their assistant at "
            "different times. List the pairs that cannot both be true now: they "
            "give different values for the same changeable fact (where they "
            "live, where they work, what they plan) or directly opposite "
            "preferences. Two claims that are merely different, or where one "
            "adds detail to the other, are not a contradiction. Do not say which "
            "is right. Answer with JSON only, exactly "
            '{"contradicts": [numbers]}. An empty list is an ordinary answer.'
        ),
        provider.user(listed),
    ]
    answer = _parse_json(complete(provider.LEARN, messages).text)
    if not isinstance(answer, dict) or not isinstance(answer.get("contradicts"), list):
        raise NotJudged('answer must be {"contradicts": [numbers]}')
    found = set()
    for value in answer["contradicts"]:
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= len(pairs):
            raise NotJudged(f"contradicts names {value!r}, not a listed pair")
        found.add(value - 1)
    return found


def judge_answer(
    complete: Callable[..., provider.Response], asked: str, reply: str
) -> Optional[int]:
    """1 or 2 if ``reply`` picks one, else ``None``. Raises :class:`NotJudged`."""
    messages = [
        provider.system(
            "You asked the person the question below, and their next message "
            "follows it. Did their message pick claim 1, claim 2, neither, or is "
            "it not an answer to the question at all? Answer with JSON only, "
            'exactly {"answer": "1"} or "2", "neither" or "unclear".'
        ),
        provider.user(f"Question: {asked}\n\nTheir message: {reply}"),
    ]
    answer = _parse_json(complete(provider.LEARN, messages).text)
    if not isinstance(answer, dict) or answer.get("answer") not in ("1", "2", "neither", "unclear"):
        raise NotJudged('answer must be {"answer": "1" | "2" | "neither" | "unclear"}')
    return int(answer["answer"]) if answer["answer"] in ("1", "2") else None


# --- the pass ---------------------------------------------------------------


def due(learned: Learned, now: str) -> bool:
    """Should a pass run? When there is something to tidy and either a day has
    passed since the last one, or the person has said something since a
    question went out that has not been read for an answer yet."""
    if not len(learned):
        return False
    if any(heard for _, _, _, heard in learned.awaiting()):
        return True
    last = _parse_at(learned.tidied_at)
    current = _parse_at(now)
    if last is None or current is None:
        return last is None
    return current - last >= TIDY_EVERY


def run(
    queue: Any,
    learned: Learned,
    complete: Callable[..., provider.Response],
    *,
    now: str,
) -> dict[str, int]:
    """One pass. Appends its records and a ``memory.tidied``; returns the counts.

    The code halves run first and always; the model halves can fail, and a
    failure is the receipt's ``reason`` with its cursor left where it was, so
    the next pass retries what this one could not judge (DL-053's rule: a pass
    nobody watches must leave a trace when it breaks).
    """
    store = queue.store
    learned.advance(store)
    counts = {"merged": 0, "staled": 0, "contradictions": 0, "resolved": 0}

    # First, so a misattributed claim is neither merged into nor checked
    # against the real ones. Attached to the fire it was anchored on.
    for claim, fire in misattributed(store, learned.claims()):
        queue.append(episodes.claim_retracted(for_seq=fire, claim_seq=claim.seq, at=now))
    learned.advance(store)

    for kept, folded in duplicates(learned.claims())[:MAX_MERGES_PER_PASS]:
        queue.append(episodes.claim_merged(kept=kept, folded=folded, at=now))
        counts["merged"] += len(folded)
    learned.advance(store)

    for claim, name, basis, age in stale_due(
        learned.claims(), marked=learned.stale_marks, now=now
    ):
        queue.append(
            episodes.claim_stale(
                claim_seq=claim.seq, category=name, basis_at=basis, age_days=age, at=now
            )
        )
        counts["staled"] += 1
    learned.advance(store)

    meter = provider.Meter(complete)
    reasons: list[str] = []
    heard_through = learned.heard_through
    try:
        heard_through = _hear(queue, learned, meter, now=now, counts=counts)
    except Exception as exc:  # noqa: BLE001 - recorded on the receipt
        reasons.append(f"answer: {str(exc) or type(exc).__name__}")
        heard_through = getattr(exc, "heard_through", learned.heard_through)
    learned.advance(store)

    checked_through = learned.checked_through
    pairs, cursor = candidate_pairs(
        learned.claims(),
        checked_through=learned.checked_through,
        recorded=learned.pairs_recorded,
    )
    try:
        found = judge_pairs(meter, pairs) if pairs else set()
        for index in sorted(found):
            older, newer = pairs[index]
            queue.append(
                episodes.claim_contradiction(
                    claims=[older.seq, newer.seq], question=question(older, newer), at=now
                )
            )
            counts["contradictions"] += 1
        checked_through = cursor
    except Exception as exc:  # noqa: BLE001 - recorded on the receipt
        reasons.append(f"contradiction: {str(exc) or type(exc).__name__}")

    queue.append(
        episodes.memory_tidied(
            checked_through=checked_through,
            heard_through=heard_through,
            reason="; ".join(reasons) or None,
            usage=meter.usage(),
            at=now,
            **counts,
        )
    )
    learned.advance(store)
    return counts


class _Stopped(RuntimeError):
    def __init__(self, cause: BaseException, heard_through: int) -> None:
        super().__init__(str(cause) or type(cause).__name__)
        self.heard_through = heard_through


def _hear(
    queue: Any, learned: Learned, complete: Any, *, now: str, counts: dict[str, int]
) -> int:
    """Read the person's messages since a question for its answer.

    Message by message, oldest first, so ``heard_through`` only ever passes a
    message every open question has been checked against. A bare "1" or "2" is
    read without a model; anything else is one bounded model call. "Neither"
    and "unclear" change nothing — the pair stays marked conflicting.
    """
    pending: dict[int, list[tuple[int, tuple[int, int], str]]] = {}
    for seq, pair, asked, heard in learned.awaiting():
        for message_seq, _ in heard:
            pending.setdefault(message_seq, []).append((seq, pair, asked))
    texts = {m: t for _, _, _, heard in learned.awaiting() for m, t in heard}
    heard_through = learned.heard_through
    calls = 0
    settled: set[int] = set()
    for message_seq in sorted(pending):
        for contradiction, pair, asked in pending[message_seq]:
            if contradiction in settled:
                continue
            text = texts[message_seq]
            choice = read_answer(text)
            if choice is None:
                if calls >= MAX_ANSWERS_PER_PASS:
                    return heard_through
                calls += 1
                try:
                    choice = judge_answer(complete, asked, text)
                except Exception as exc:  # noqa: BLE001 - re-raised with the cursor
                    raise _Stopped(exc, heard_through) from exc
            if choice is None:
                continue
            kept, dropped = (pair[0], pair[1]) if choice == 1 else (pair[1], pair[0])
            queue.append(
                episodes.claim_resolved(
                    contradiction_seq=contradiction,
                    kept=kept,
                    dropped=dropped,
                    answer_seq=message_seq,
                    at=now,
                )
            )
            settled.add(contradiction)
            counts["resolved"] += 1
        heard_through = message_seq
    return heard_through
