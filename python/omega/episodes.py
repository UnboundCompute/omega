"""The episode payload codec — M1 step 1, spec `agent/M1_SPEC.md` §2.1.

M0 made ``payload`` **opaque bytes** on purpose, so what an episode *means* is
decided here, in Python (DL-018), and nothing below the memory seam knows this
file exists. Changing the schema is a re-derive from the log, not a migration
(DL-017), which is why JSON is affordable: we pay nothing structurally for
readability during exactly the phase where we will be reading raw logs to debug
the loop.

**This is not the channel wire format.** That stays deferred. This is the
*storage* shape of an episode. Keeping the two apart is the whole lesson of
DL-006 — a stub transport's shape nearly became the contract by default.

The module is deliberately all functions and constants: an episode payload is a
plain ``dict`` that round-trips through JSON, and wrapping it in a class
hierarchy would buy nothing the closed ``KINDS`` set does not already give.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional

from omega.memory import MAX_BODY

__all__ = [
    "VERSION",
    "KINDS",
    "TERMINAL_KINDS",
    "OUTCOMES",
    "MESSAGE_INBOUND",
    "TURN_BLOCKED",
    "TURN_COMPLETED",
    "TOOL_CALLED",
    "TOOL_RETURNED",
    "WORK_FINISHED",
    "BadPayload",
    "UnsupportedPayloadVersion",
    "encode",
    "decode",
    "is_terminal",
    "turn_write_key",
    "inbound",
    "blocked",
    "completed",
    "tool_called",
    "tool_returned",
    "work_finished",
    "now",
]

#: Payload schema version. Bumped when a change would make an *old* payload
#: decode wrongly — not when a field is merely added, since an unknown field is
#: ignored on read and every M1 consumer tolerates that.
VERSION = 1

MESSAGE_INBOUND = "message.inbound"
TURN_BLOCKED = "turn.blocked"
TURN_COMPLETED = "turn.completed"
TOOL_CALLED = "tool.called"
TOOL_RETURNED = "tool.returned"
WORK_FINISHED = "work.finished"

#: The complete M1 kind set (§2.1: "and that is all of them").
#:
#: Closed on purpose. §1.1 requires the drain to *advance past* record kinds it
#: does not process — otherwise the executor treats its own output as a new
#: event and loops forever. "Advance past" is only safe when an unrecognised
#: kind can be told apart from a kind we forgot to handle, so an unknown kind
#: is an error here rather than a shrug at the call site.
KINDS = frozenset(
    {
        MESSAGE_INBOUND,
        TURN_BLOCKED,
        TURN_COMPLETED,
        TOOL_CALLED,
        TOOL_RETURNED,
        WORK_FINISHED,
    }
)

#: The two kinds that end a claimed turn and let ``DONE`` advance (§1.2).
#:
#: ``turn.blocked`` is in here, and that is the substantive part: a blocked turn
#: waits on a human, which is unbounded, so if it stayed claimed the single
#: consumer would stop draining and one question to the user would freeze every
#: other event. The block is durable because it is *in the log*, not because a
#: thread is parked on it.
TERMINAL_KINDS = frozenset({TURN_COMPLETED, TURN_BLOCKED})

#: How a turn ended. ``silent`` is a **success**, not a degraded ``spoke`` —
#: DL-011 makes deciding to say nothing the load-bearing property of the loop.
OUTCOMES = frozenset({"spoke", "silent", "failed"})

_CONTEXT_KINDS = frozenset({"file", "image", "text", "link", "screen"})
_URGENCIES = frozenset({"normal", "timely"})


class BadPayload(ValueError):
    """A payload that is not a valid episode.

    Raised on the way *in* (a kind outside ``KINDS``, a missing required field,
    a payload too large for the log) and on the way *out* (bytes that are not
    JSON, or are JSON that is not an episode). Never returns ``None`` for a bad
    payload: a decoder that answers ``None`` makes "no episodes" and "every
    episode was unreadable" the same observation, and `CLAUDE.md` calls a check
    that passes on empty not a check.
    """


class UnsupportedPayloadVersion(BadPayload):
    """The payload is a well-formed episode from a schema we cannot read.

    Distinct from :class:`BadPayload` because the two demand different
    responses: garbage means the log is damaged, while a future ``v`` means
    this code is old. Only the second is fixed by upgrading.
    """


def now() -> str:
    """The current instant, UTC, ISO 8601, second resolution.

    Injectable everywhere it is used, because the clock is a dependency and a
    test that cannot pin it is a test that measures the wall clock.
    """
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# --- the wire ---------------------------------------------------------------


def encode(payload: dict[str, Any]) -> bytes:
    """Validate ``payload`` and render it as the bytes M0 will store.

    Validation happens *here*, before the append, so a malformed episode is
    rejected while the caller still has a stack to blame. The alternative —
    writing first and discovering it on read — puts the error in the one place
    it cannot be fixed, since the log is append-only.

    Encoding is deterministic (sorted keys, no incidental whitespace) so the
    same episode is the same bytes. That is not cosmetic: it makes the content
    of a re-derived log comparable byte-for-byte against the original, which is
    how DL-017's rebuild-from-log gets to be checkable rather than trusted.
    """
    _validate(payload)
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    if len(raw) > MAX_BODY:
        # Caught here rather than at append: the store's error names a byte
        # limit, which tells the caller nothing about *which* episode, and by
        # then the useful context is gone.
        raise BadPayload(
            f"episode is {len(raw)} bytes, over the log's {MAX_BODY}-byte limit"
        )
    return raw


def decode(raw: bytes) -> dict[str, Any]:
    """Parse stored bytes back into an episode payload.

    Fails closed on anything it cannot vouch for. An episode that does not
    decode is a real problem — M0 guarantees the bytes are *intact*, so
    unreadable bytes mean the writer wrote nonsense, not that the disk rotted.
    """
    try:
        payload = json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise BadPayload(f"payload is not UTF-8: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise BadPayload(f"payload is not JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise BadPayload(
            f"payload is a {type(payload).__name__}, not an episode object"
        )

    version = payload.get("v")
    if version != VERSION:
        if isinstance(version, int) and version > VERSION:
            raise UnsupportedPayloadVersion(
                f"payload is schema v{version}; this build reads v{VERSION}"
            )
        raise BadPayload(f"payload has version {version!r}, expected {VERSION}")

    _validate(payload)
    return payload


def is_terminal(payload: dict[str, Any]) -> bool:
    """Does this record end the turn it belongs to? (§1.2)

    A claimed turn ends with ``turn.completed`` **or** ``turn.blocked``; both
    advance ``DONE``, and only their *absence* means the turn was interrupted.
    """
    return payload.get("kind") in TERMINAL_KINDS


def turn_write_key(for_seq: int) -> str:
    """The write key a terminal record carries for inbound episode ``for_seq``.

    Lives here so the loop cannot spell it two ways. With it, a double-write is
    rejected *by the log itself* (M0's `WriteKeyConflict`) rather than by a
    convention someone has to remember — which turns M0's dedup from an unused
    feature into a live invariant check on the executor's most dangerous bug.
    """
    if not isinstance(for_seq, int) or isinstance(for_seq, bool) or for_seq < 1:
        raise BadPayload(f"for_seq must be a positive int, got {for_seq!r}")
    return f"turn:{for_seq}"


# --- constructors -----------------------------------------------------------
# One per kind. They exist so no call site assembles a dict by hand and gets a
# key subtly wrong -- a payload with "kind" misspelled is a valid JSON object
# and would sail into the log, where it is permanent.


def inbound(
    text: str,
    *,
    channel: str,
    context: Optional[list[dict[str, Any]]] = None,
    urgency: str = "normal",
    resumes_seq: Optional[int] = None,
    at: Optional[str] = None,
) -> dict[str, Any]:
    """Something arrived that omega may need to respond to.

    ``context`` entries carry a stable ``id`` because the tray needs identity to
    survive delivery *and failure recovery* — ids live in the payload, not in the
    tray's memory, or a ``kill -9`` on either side strands the staged items. The
    tray keeps the previews; the log keeps the identities.

    ``resumes_seq`` is set when this message answers an earlier
    ``turn.blocked``. The answer is an ordinary turn, not a resumption of the
    blocked one — the blocked record simply lands in its recall.
    """
    payload: dict[str, Any] = {
        "v": VERSION,
        "kind": MESSAGE_INBOUND,
        "text": text,
        "channel": channel,
        "context": list(context or []),
        "urgency": urgency,
        "at": at or now(),
    }
    if resumes_seq is not None:
        payload["resumes_seq"] = resumes_seq
    _validate(payload)
    return payload


def blocked(
    *, for_seq: int, needs: str, at: Optional[str] = None
) -> dict[str, Any]:
    """omega stopped to ask. A terminal record: it ends the turn.

    Stopping to ask is correct behaviour, not a failure (DL-011) — which is why
    this is its own kind rather than an ``outcome`` on ``turn.completed``.
    Folding it into failure would make the tray show a stall as an error, and
    would lose the distinction the whole initiative feature rests on.
    """
    payload = {
        "v": VERSION,
        "kind": TURN_BLOCKED,
        "for_seq": for_seq,
        "needs": needs,
        "at": at or now(),
    }
    _validate(payload)
    return payload


def completed(
    *,
    for_seq: int,
    outcome: str,
    reply: Optional[str] = None,
    tools: Optional[list[str]] = None,
    error: Optional[str] = None,
    at: Optional[str] = None,
) -> dict[str, Any]:
    """The turn ended. A terminal record.

    ``outcome="silent"`` with ``reply=None`` is the success case the loop exists
    to make possible, so it is spelled as a first-class outcome and not as an
    empty reply — ``""`` and "chose not to speak" must never be the same row.
    """
    payload = {
        "v": VERSION,
        "kind": TURN_COMPLETED,
        "for_seq": for_seq,
        "outcome": outcome,
        "reply": reply,
        "tools": list(tools or []),
        "error": error,
        "at": at or now(),
    }
    _validate(payload)
    return payload


def tool_called(
    *, for_seq: int, tool: str, args: dict[str, Any], at: Optional[str] = None
) -> dict[str, Any]:
    """A tool was invoked. Logged *before* the call, so a crash mid-tool leaves
    evidence that it started — which is the only way the restart report can tell
    "never ran" from "ran and we never heard back"."""
    payload = {
        "v": VERSION,
        "kind": TOOL_CALLED,
        "for_seq": for_seq,
        "tool": tool,
        "args": dict(args),
        "at": at or now(),
    }
    _validate(payload)
    return payload


def tool_returned(
    *,
    for_seq: int,
    tool: str,
    ok: bool,
    result: Optional[str] = None,
    error: Optional[str] = None,
    at: Optional[str] = None,
) -> dict[str, Any]:
    """A tool came back. ``ok=False`` carries ``error`` and is surfaced, never
    swallowed (§2.4) — a tool failure the loop hides is a lie in the log."""
    payload = {
        "v": VERSION,
        "kind": TOOL_RETURNED,
        "for_seq": for_seq,
        "tool": tool,
        "ok": ok,
        "result": result,
        "error": error,
        "at": at or now(),
    }
    _validate(payload)
    return payload


def work_finished(
    *, for_seq: int, summary: str, ok: bool = True, at: Optional[str] = None
) -> dict[str, Any]:
    """A detached sub-loop re-entered as an ordinary event (§1.5, DL-016).

    Long work does not hold the executor; it finishes and *arrives*, which is
    what keeps the single consumer draining while something slow is in progress.
    """
    payload = {
        "v": VERSION,
        "kind": WORK_FINISHED,
        "for_seq": for_seq,
        "summary": summary,
        "ok": ok,
        "at": at or now(),
    }
    _validate(payload)
    return payload


# --- validation -------------------------------------------------------------

_REQUIRED: dict[str, tuple[str, ...]] = {
    MESSAGE_INBOUND: ("text", "channel", "context", "urgency", "at"),
    TURN_BLOCKED: ("for_seq", "needs", "at"),
    TURN_COMPLETED: ("for_seq", "outcome", "reply", "tools", "error", "at"),
    TOOL_CALLED: ("for_seq", "tool", "args", "at"),
    TOOL_RETURNED: ("for_seq", "tool", "ok", "result", "error", "at"),
    WORK_FINISHED: ("for_seq", "summary", "ok", "at"),
}


def _validate(payload: dict[str, Any]) -> None:
    """The one gate. Both constructors and :func:`decode` run it, so a payload
    cannot be valid going in and invalid coming out."""
    if not isinstance(payload, dict):
        raise BadPayload(f"episode must be a dict, got {type(payload).__name__}")

    if payload.get("v") != VERSION:
        raise BadPayload(f"episode has version {payload.get('v')!r}, expected {VERSION}")

    kind = payload.get("kind")
    if kind not in KINDS:
        raise BadPayload(
            f"unknown episode kind {kind!r}; M1 kinds are {sorted(KINDS)}"
        )

    missing = [f for f in _REQUIRED[kind] if f not in payload]
    if missing:
        raise BadPayload(f"{kind} is missing {missing}")

    if kind == MESSAGE_INBOUND:
        _require_str(payload, "text")
        _require_str(payload, "channel", non_empty=True)
        if payload["urgency"] not in _URGENCIES:
            raise BadPayload(
                f"urgency {payload['urgency']!r} not in {sorted(_URGENCIES)}"
            )
        _validate_context(payload["context"])
        if "resumes_seq" in payload:
            _require_seq(payload, "resumes_seq")
    else:
        # Every non-inbound kind names the inbound turn it belongs to. Without
        # it a record cannot be matched to its turn, and §1.2's "exactly one
        # terminal record per inbound" stops being checkable.
        _require_seq(payload, "for_seq")

    if kind == TURN_COMPLETED:
        if payload["outcome"] not in OUTCOMES:
            raise BadPayload(
                f"outcome {payload['outcome']!r} not in {sorted(OUTCOMES)}"
            )
        if payload["outcome"] == "failed" and not payload["error"]:
            # A failure with no error is the shape that turns a real fault into
            # a silent one; the log would record that something went wrong and
            # lose the only field saying what.
            raise BadPayload("outcome 'failed' requires a non-empty error")
        if payload["outcome"] == "silent" and payload["reply"] is not None:
            raise BadPayload("outcome 'silent' cannot carry a reply")
        if not isinstance(payload["tools"], list):
            raise BadPayload("tools must be a list")

    if kind == TURN_BLOCKED:
        _require_str(payload, "needs", non_empty=True)

    if kind == TOOL_CALLED:
        _require_str(payload, "tool", non_empty=True)
        if not isinstance(payload["args"], dict):
            raise BadPayload("args must be an object")

    if kind == TOOL_RETURNED:
        _require_str(payload, "tool", non_empty=True)
        if not isinstance(payload["ok"], bool):
            raise BadPayload("ok must be a bool")
        if not payload["ok"] and not payload["error"]:
            raise BadPayload("a failed tool call requires a non-empty error")

    if kind == WORK_FINISHED:
        _require_str(payload, "summary", non_empty=True)
        if not isinstance(payload["ok"], bool):
            raise BadPayload("ok must be a bool")

    _require_str(payload, "at", non_empty=True)


def _validate_context(context: Any) -> None:
    if not isinstance(context, list):
        raise BadPayload("context must be a list")
    for i, item in enumerate(context):
        if not isinstance(item, dict):
            raise BadPayload(f"context[{i}] must be an object")
        for field in ("id", "kind", "title"):
            if field not in item:
                raise BadPayload(f"context[{i}] is missing {field!r}")
        if not isinstance(item["id"], str) or not item["id"]:
            raise BadPayload(f"context[{i}].id must be a non-empty string")
        if item["kind"] not in _CONTEXT_KINDS:
            raise BadPayload(
                f"context[{i}].kind {item['kind']!r} not in {sorted(_CONTEXT_KINDS)}"
            )


def _require_str(payload: dict[str, Any], field: str, *, non_empty: bool = False) -> None:
    value = payload[field]
    if not isinstance(value, str):
        raise BadPayload(f"{field} must be a string, got {type(value).__name__}")
    if non_empty and not value:
        raise BadPayload(f"{field} must not be empty")


def _require_seq(payload: dict[str, Any], field: str) -> None:
    value = payload[field]
    # `bool` is an `int` in Python, and True would silently become seq 1.
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise BadPayload(f"{field} must be a positive int, got {value!r}")
