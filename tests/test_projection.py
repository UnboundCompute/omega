"""The outward projection — M1_SPEC.md §Q10.

The wire format is a *filter* over the episode stream, so these cases are about
two things: that every kind lands in the right one of the tray's five work
states, and that the things policy withholds genuinely do not cross.
"""

from __future__ import annotations

import json

import pytest

from omega import episodes, projection
from omega.memory import MemoryStore
from omega.projection import (
    BLOCKED,
    COMPLETE,
    FAILED,
    NOT_PROJECTED,
    STATES,
    UNDERSTOOD,
    WORKING,
    Update,
    project,
    updates_since,
)
from omega.queue import EventQueue

AT = "2026-09-23T12:00:00+00:00"


@pytest.fixture
def q(store: MemoryStore) -> EventQueue:
    return EventQueue(store)


def wire(payload: dict, seq: int = 1) -> dict:
    update = project(payload, seq)
    assert update is not None, f"{payload['kind']} was withheld"
    out = update.wire()
    # Everything on this wire is JSON, or it is not a wire format.
    assert json.loads(json.dumps(out)) == out
    return out


# --- the five states --------------------------------------------------------


def test_there_are_exactly_five_work_states() -> None:
    """The tray's five durable states are the alphabet. A sixth is a design
    change, not an implementation detail, so it fails here first."""
    assert STATES == {UNDERSTOOD, WORKING, BLOCKED, FAILED, COMPLETE}
    assert len(STATES) == 5


def test_an_inbound_message_is_understood() -> None:
    out = wire(episodes.inbound("book the flight", channel="tray", at=AT), seq=7)
    assert out["state"] == UNDERSTOOD
    assert out["seq"] == 7
    assert out["for_seq"] == 7, "an inbound message is its own turn"
    assert out["text"] == "book the flight"
    assert out["urgency"] == "normal"
    assert out["at"] == AT
    assert out["kind"] == episodes.MESSAGE_INBOUND


def test_a_tool_call_is_working_and_a_return_is_working_too() -> None:
    called = wire(
        episodes.tool_called(for_seq=4, tool="shell", args={"cmd": "ls"}, at=AT), seq=5
    )
    returned = wire(
        episodes.tool_returned(
            for_seq=4, tool="shell", ok=True, result="a\nb", at=AT
        ),
        seq=6,
    )
    assert called["state"] == WORKING and returned["state"] == WORKING
    assert called["for_seq"] == returned["for_seq"] == 4
    assert called["tool"] == returned["tool"] == "shell"
    assert "ok" not in called, "a call has no outcome yet"
    assert returned["ok"] is True


def test_a_failed_tool_is_still_working_and_says_it_failed() -> None:
    """§2.4 — a tool error is surfaced, never swallowed. It is not a *turn*
    failure: the sub-loop sees it and may well recover."""
    out = wire(
        episodes.tool_returned(
            for_seq=4, tool="shell", ok=False, error="exit 1", at=AT
        )
    )
    assert out["state"] == WORKING
    assert out["ok"] is False


def test_detached_work_coming_back_is_working(q: EventQueue) -> None:
    out = wire(episodes.work_finished(for_seq=3, summary="index rebuilt", at=AT))
    assert out["state"] == WORKING
    assert out["text"] == "index rebuilt"
    assert out["ok"] is True


def test_a_blocked_turn_is_its_own_state_and_carries_the_question() -> None:
    """Blocked is a peer outcome, not a failure (DL-011). If it projected as
    failed, a stall would render as an error and the distinction the whole
    initiative feature rests on would be gone from the UI."""
    out = wire(episodes.blocked(for_seq=9, needs="which airport?", at=AT), seq=10)
    assert out["state"] == BLOCKED
    assert out["state"] != FAILED
    assert out["needs"] == "which airport?"
    assert out["for_seq"] == 9


def test_a_turn_that_spoke_is_verified_complete_and_carries_the_reply() -> None:
    out = wire(
        episodes.completed(for_seq=2, outcome="spoke", reply="booked", at=AT), seq=3
    )
    assert out["state"] == COMPLETE
    assert out["outcome"] == "spoke"
    assert out["reply"] == "booked"


def test_a_turn_that_failed_is_failed_and_carries_the_error() -> None:
    out = wire(
        episodes.completed(
            for_seq=2, outcome="failed", error="the model returned nothing", at=AT
        )
    )
    assert out["state"] == FAILED
    assert out["outcome"] == "failed"
    assert out["error"] == "the model returned nothing"
    assert out["reply"] is None


# --- DL-011 on the wire -----------------------------------------------------


def test_silence_and_failure_are_different_on_the_wire() -> None:
    """DL-011, at the one place a person actually sees it.

    A model that decided not to speak succeeded; a model that returned nothing
    failed. If the projection collapsed them, the loop's most important property
    would be invisible exactly where it matters.
    """
    silent = wire(episodes.completed(for_seq=1, outcome="silent", at=AT))
    failed = wire(episodes.completed(for_seq=1, outcome="failed", error="boom", at=AT))

    assert silent["state"] == COMPLETE and failed["state"] == FAILED
    assert silent["outcome"] == "silent" and failed["outcome"] == "failed"
    assert silent != failed


def test_a_silent_turn_sends_reply_null_rather_than_omitting_it() -> None:
    """``null`` and *absent* are different claims. A reply field that vanished
    would let a client read "no reply key" as "nothing happened", and an empty
    string would read as "said nothing" — both are the collapse DL-011
    forbids."""
    out = wire(episodes.completed(for_seq=1, outcome="silent", at=AT))
    assert "reply" in out
    assert out["reply"] is None
    assert out["reply"] != ""


def test_a_spoken_reply_is_never_confused_with_silence() -> None:
    spoke = wire(episodes.completed(for_seq=1, outcome="spoke", reply="hi", at=AT))
    silent = wire(episodes.completed(for_seq=1, outcome="silent", at=AT))
    assert spoke["reply"] == "hi" and silent["reply"] is None
    assert spoke["state"] == silent["state"] == COMPLETE
    assert spoke["outcome"] != silent["outcome"]


# --- what policy withholds --------------------------------------------------


def test_tool_arguments_never_cross_the_projection() -> None:
    """The tray has no business seeing tool arguments: they are the part of a
    turn most likely to carry a secret, and a UI has no use for them."""
    secret = "sk-not-a-real-key-0xdeadbeef"
    out = wire(
        episodes.tool_called(
            for_seq=1, tool="http", args={"headers": {"auth": secret}}, at=AT
        )
    )
    assert secret not in json.dumps(out)
    assert "args" not in out
    assert out["tool"] == "http", "the fact of the call still crosses"


def test_tool_results_never_cross_the_projection() -> None:
    """Tool output is untrusted input (DL-014). It informs the next pass of the
    sub-loop; it does not get piped to the UI unfiltered."""
    secret = "contents-of-the-private-file"
    out = wire(
        episodes.tool_returned(
            for_seq=1, tool="read_file", ok=True, result=secret, at=AT
        )
    )
    assert secret not in json.dumps(out)
    assert "result" not in out


def test_context_ids_cross_but_previews_do_not() -> None:
    """§2.1 splits this explicitly — the tray keeps the previews, the log keeps
    the identities. The id must survive a ``kill -9`` on either side, and it
    does, because it is in the payload rather than in the tray's memory."""
    out = wire(
        episodes.inbound(
            "look at this",
            channel="tray",
            context=[
                {"id": "ctx-1", "kind": "file", "title": "salary-review.pdf"},
                {"id": "ctx-2", "kind": "screen", "title": "a screenshot"},
            ],
            at=AT,
        )
    )
    assert out["context"] == [
        {"id": "ctx-1", "kind": "file"},
        {"id": "ctx-2", "kind": "screen"},
    ]
    assert "salary-review.pdf" not in json.dumps(out)


def test_a_blob_reference_does_not_go_back_out_to_the_client() -> None:
    """DL-027 changed what a context item *stores*, not what it projects.

    A digest is an identifier for bytes the tray already has — it uploaded them
    — so sending it back is text on the wire with no reader, which is the exact
    argument that strips titles. Keeping the projection at ``{id, kind}`` is
    also what keeps the split honest: the tray keeps the previews, the log keeps
    the identities, and now the store keeps the bytes.
    """
    digest = "sha256:" + "0123456789abcdef" * 4
    out = wire(
        episodes.inbound(
            "what is this",
            channel="tray",
            context=[
                {
                    "id": "ctx-1",
                    "kind": "image",
                    "title": "Area capture",
                    "blob": digest,
                    "mime": "image/png",
                    "bytes": 184320,
                }
            ],
            at=AT,
        )
    )
    assert out["context"] == [{"id": "ctx-1", "kind": "image"}]
    assert digest not in json.dumps(out)
    assert "image/png" not in json.dumps(out)


def test_an_unknown_kind_is_withheld_rather_than_guessed() -> None:
    """``None`` is a real answer. A future kind must not leak outwards with an
    invented state just because the projection did not recognise it."""
    assert project({"v": 1, "kind": "future.kind", "at": AT}, 1) is None
    assert project({"v": 1, "at": AT}, 1) is None


def test_every_m1_kind_is_accounted_for() -> None:
    """A whole missing category is a miss, not a non-result. If a kind is added
    to the codec and not to the projection, this fails rather than silently
    dropping it from every UI."""
    made = {
        episodes.MESSAGE_INBOUND: episodes.inbound("t", channel="tray", at=AT),
        episodes.TURN_BLOCKED: episodes.blocked(for_seq=1, needs="?", at=AT),
        episodes.TURN_COMPLETED: episodes.completed(
            for_seq=1, outcome="spoke", reply="r", at=AT
        ),
        episodes.TOOL_CALLED: episodes.tool_called(
            for_seq=1, tool="t", args={}, at=AT
        ),
        episodes.TOOL_RETURNED: episodes.tool_returned(
            for_seq=1, tool="t", ok=True, at=AT
        ),
        episodes.WORK_FINISHED: episodes.work_finished(for_seq=1, summary="s", at=AT),
        episodes.SCHEDULE_CREATED: episodes.schedule_created(
            id="s", instruction="i", cron="0 9 *", at=AT
        ),
        episodes.SCHEDULE_CANCELLED: episodes.schedule_cancelled(id="s", at=AT),
        episodes.CLAIM_EXTRACTED: episodes.claim_extracted(
            for_seq=1, text="c", source_seq=1, situation="s", explicit=True, at=AT
        ),
        episodes.CLAIM_RETRACTED: episodes.claim_retracted(
            for_seq=1, claim_seq=2, at=AT
        ),
    }
    assert set(made) == set(episodes.KINDS)
    for kind, payload in made.items():
        update = project(payload, 1)
        if kind in NOT_PROJECTED:
            # Withheld on purpose. Asserting the *decision* rather than
            # accepting a None keeps this guard able to tell a policy choice
            # apart from a kind nobody wired up.
            assert update is None, f"{kind} is listed as withheld but projected"
            continue
        assert update is not None, f"{kind} projects to nothing"
        assert update.state in STATES


def test_a_scheduled_fire_is_visible_to_the_tray() -> None:
    """omega acting on its own must not happen invisibly. A fire is an
    ordinary inbound, so it reaches the wire like any arriving message —
    that is the user-facing half of DL-035's 'ordinary event' rule."""
    fire = episodes.inbound(
        "morning brief",
        channel="schedule",
        schedule_id="brief",
        schedule_slot=AT,
        at=AT,
    )

    update = project(fire, 7)

    assert update is not None and update.state == "understood"


# --- reading a stream out of the log ---------------------------------------


def test_updates_since_replays_exactly_what_was_missed(q: EventQueue) -> None:
    first = q.append(episodes.inbound("one", channel="tray", at=AT))
    q.append(episodes.completed(for_seq=first, outcome="silent", at=AT))
    second = q.append(episodes.inbound("two", channel="tray", at=AT))

    everything = list(updates_since(q, 0))
    assert [u.seq for u in everything] == [1, 2, 3]
    assert [u.state for u in everything] == [UNDERSTOOD, COMPLETE, UNDERSTOOD]

    missed = list(updates_since(q, 2))
    assert [u.seq for u in missed] == [second]


def test_updates_since_head_is_empty_and_past_head_is_refused(q: EventQueue) -> None:
    q.append(episodes.inbound("one", channel="tray", at=AT))
    assert list(updates_since(q, q.head())) == []
    with pytest.raises(ValueError):
        list(updates_since(q, q.head() + 1))
    with pytest.raises(ValueError):
        list(updates_since(q, -1))


def test_updates_since_on_an_empty_log_yields_nothing(q: EventQueue) -> None:
    assert list(updates_since(q, 0)) == []


class GrowsMidRead:
    """A queue that gains episodes between the head read and the window read.

    Not a mock of the queue — it delegates everything to a real one over a real
    log. It only forces the interleaving that the drain thread produces by
    accident, at the one instant it matters, so the case is deterministic
    instead of a race that fails on someone else's machine.
    """

    def __init__(self, queue: EventQueue, grow) -> None:
        self._queue = queue
        self._grow = grow

    def head(self) -> int:
        return self._queue.head()

    def recent(self, n: int, *, before=None):
        self._grow()
        return self._queue.recent(n, before=before)


def test_updates_since_is_not_shortened_by_an_append_made_while_it_reads(
    q: EventQueue,
) -> None:
    """The window must be absolute, not relative.

    ``recent(n)`` means *the last n*, and the last n moves when the log grows.
    Since M1 step 7 the executor drains on another thread, so an append can and
    does land between the head this reads and the head ``recent`` reads — and a
    relative window would then slide forward by exactly that much, silently
    dropping the oldest episodes the caller asked for. Silently is the problem:
    the client's cursor would advance past updates it never received, so nothing
    would ever ask for them again.
    """
    first = q.append(episodes.inbound("one", channel="tray", at=AT))
    q.append(episodes.completed(for_seq=first, outcome="spoke", reply="hi", at=AT))

    def append_two_more() -> None:
        third = q.append(episodes.inbound("two", channel="tray", at=AT))
        q.append(episodes.completed(for_seq=third, outcome="silent", at=AT))

    growing = GrowsMidRead(q, append_two_more)
    seen = list(updates_since(growing, 0))  # type: ignore[arg-type]

    assert [u.seq for u in seen] == [1, 2], (
        "the caller asked for seqs 1..2 and must get seqs 1..2, whatever "
        "arrived while it was reading"
    )
    # And the episodes that landed mid-read are not lost either: they are simply
    # after the cursor, which is what the next call asks for.
    assert [u.seq for u in updates_since(q, 2)] == [3, 4]


def test_the_projection_holds_no_state_of_its_own(q: EventQueue) -> None:
    """DL-016 — zero authoritative state in RAM. Read it twice, get the same
    thing; there is no cursor inside the projection to get out of step."""
    q.append(episodes.inbound("one", channel="tray", at=AT))
    q.append(episodes.completed(for_seq=1, outcome="spoke", reply="hi", at=AT))
    once = [u.wire() for u in updates_since(q, 0)]
    twice = [u.wire() for u in updates_since(q, 0)]
    assert once == twice


def test_an_update_always_carries_its_envelope() -> None:
    for payload in (
        episodes.inbound("t", channel="tray", at=AT),
        episodes.blocked(for_seq=1, needs="?", at=AT),
        episodes.completed(for_seq=1, outcome="silent", at=AT),
        episodes.tool_called(for_seq=1, tool="t", args={}, at=AT),
    ):
        out = wire(payload, seq=12)
        assert out["v"] == projection.VERSION
        assert out["op"] == "update"
        assert out["seq"] == 12
        assert out["at"] == AT
        assert out["kind"] == payload["kind"]
        assert out["for_seq"] >= 1


def test_the_wire_version_is_independent_of_the_episode_version() -> None:
    """They change for different reasons — one is ours, one is shared with a
    client we do not deploy. Tying them together would force a client update
    for a storage-only change."""
    stored = dict(episodes.inbound("t", channel="tray", at=AT))
    stored["v"] = 99  # a future storage schema, as it would arrive from the log
    out = project(stored, 1)
    assert out is not None
    assert out.wire()["v"] == projection.VERSION, (
        "the wire version must be stated, not copied out of the payload"
    )
    assert out.kind == episodes.MESSAGE_INBOUND


def test_update_is_frozen() -> None:
    update = Update(seq=1, state=UNDERSTOOD, for_seq=1, at=AT, kind="x")
    with pytest.raises(Exception):
        update.seq = 2  # type: ignore[misc]
