"""M1 step 1 — the episode payload codec (`agent/M1_SPEC.md` §2.1).

Organised green / red / yellow, the same way the M0 suite is:

* **green** — the thing works for the cases it is for;
* **red** — it *refuses* what it must refuse, loudly and with a usable message;
* **yellow** — the awkward middle: values that are technically well-formed and
  semantically wrong, and the distinctions that are easy to collapse by
  accident.

The yellow section is where the real content is. Green tests mostly restate the
constructors, and a codec whose only tests are green is a codec that will accept
``{"kind": "turn.complete"}`` -- one letter short, valid JSON, permanent in an
append-only log.
"""

from __future__ import annotations

import json

import pytest

from omega import episodes as ep
from omega.memory import MAX_BODY, MemoryStore

#: A well-formed blob reference (DL-027). Written out rather than computed, so
#: these cases pin the *format the log accepts* instead of restating whatever
#: the blob store happens to produce today.
DIGEST = "sha256:" + "0123456789abcdef" * 4


# --- green ------------------------------------------------------------------


def test_every_kind_round_trips():
    """Each constructor produces something `decode(encode(x)) == x`.

    Covers every kind in one place so adding another without a round-trip is a
    failure rather than an omission nobody notices.
    """
    built = [
        ep.inbound("hello", channel="tray", at="2026-09-23T10:00:00+00:00"),
        ep.blocked(for_seq=41, needs="which repo?", at="2026-09-23T10:00:00+00:00"),
        ep.completed(
            for_seq=41, outcome="spoke", reply="hi", at="2026-09-23T10:00:00+00:00"
        ),
        ep.tool_called(
            for_seq=41, tool="grep", args={"q": "x"}, at="2026-09-23T10:00:00+00:00"
        ),
        ep.tool_returned(
            for_seq=41, tool="grep", ok=True, result="2 hits",
            at="2026-09-23T10:00:00+00:00",
        ),
        ep.work_finished(
            for_seq=41, summary="index rebuilt", at="2026-09-23T10:00:00+00:00"
        ),
        ep.schedule_created(
            id="brief",
            instruction="write the morning brief",
            cron="0 9 *",
            at="2026-09-23T10:00:00+00:00",
        ),
        ep.schedule_cancelled(id="brief", at="2026-09-23T10:00:00+00:00"),
        ep.reflection_done(
            through=41, filed=1, at="2026-09-23T10:00:00+00:00"
        ),
        ep.claim_extraction_failed(
            for_seq=41,
            reason="400 Unsupported parameter: 'temperature'",
            at="2026-09-23T10:00:00+00:00",
        ),
        ep.claim_extracted(
            for_seq=41,
            text="prefers short replies",
            source_seq=41,
            situation="said so while reviewing a long answer",
            explicit=True,
            trigger={"any": ["review"]},
            at="2026-09-23T10:00:00+00:00",
        ),
        ep.claim_retracted(
            for_seq=41, claim_seq=12, at="2026-09-23T10:00:00+00:00"
        ),
        ep.transcript_ingested(
            session="63815d72-59e0-4cf4-a771-e7e561b4bef8",
            source="claude-code",
            project="-Users-me-project",
            filed=2,
            at="2026-09-23T10:00:00+00:00",
        ),
    ]
    assert {p["kind"] for p in built} == ep.KINDS, "a kind has no round-trip test"
    for payload in built:
        assert ep.decode(ep.encode(payload)) == payload


def test_silence_is_a_first_class_outcome():
    """DL-011's load-bearing case: the turn succeeded and said nothing."""
    payload = ep.completed(for_seq=7, outcome="silent", at="2026-09-23T10:00:00+00:00")
    assert payload["outcome"] == "silent"
    assert payload["reply"] is None
    assert payload["error"] is None
    assert ep.decode(ep.encode(payload)) == payload


def test_inbound_carries_context_identity():
    """Tray requirement 2: ids live in the payload, not in the tray's memory."""
    payload = ep.inbound(
        "what is this",
        channel="tray",
        context=[{"id": "ctx-1", "kind": "screen", "title": "Area capture"}],
        at="2026-09-23T10:00:00+00:00",
    )
    assert ep.decode(ep.encode(payload))["context"][0]["id"] == "ctx-1"


def test_a_context_item_may_carry_a_blob_reference():
    """DL-027 — the episode names the bytes; the bytes live beside the log.

    The reference round-trips whole, because it is the only thing that can find
    the content again: the log carries no path and no copy, so a digest that did
    not survive encoding would be an attachment that no longer exists.
    """
    item = {
        "id": "ctx-1",
        "kind": "image",
        "title": "Area capture",
        "blob": DIGEST,
        "mime": "image/png",
        "bytes": 184320,
    }
    payload = ep.inbound(
        "what is this",
        channel="tray",
        context=[item],
        at="2026-09-23T10:00:00+00:00",
    )
    assert ep.decode(ep.encode(payload))["context"] == [item]


def test_a_context_item_without_a_blob_reference_is_still_ordinary():
    """The three fields are optional as a *set*. A link or a selection has
    identity and no stored bytes, and that is the common case — this is the one
    that would break every existing client if the fields became required."""
    item = {"id": "ctx-2", "kind": "link", "title": "the ticket"}
    payload = ep.inbound("see this", channel="tray", context=[item], at="t")
    assert ep.decode(ep.encode(payload))["context"] == [item]


def test_encoding_is_deterministic():
    """Same episode, same bytes -- so a re-derived log is byte-comparable
    against the original, which is how DL-017's rebuild gets to be *checked*
    rather than trusted."""
    a = ep.completed(for_seq=1, outcome="spoke", reply="x", at="2026-09-23T10:00:00+00:00")
    b = ep.completed(for_seq=1, outcome="spoke", reply="x", at="2026-09-23T10:00:00+00:00")
    assert ep.encode(a) == ep.encode(b)
    # ...and stable across a round trip, not merely equal between two builds.
    assert ep.encode(ep.decode(ep.encode(a))) == ep.encode(a)


def test_terminal_kinds_are_exactly_completed_and_blocked():
    assert ep.is_terminal(ep.completed(for_seq=1, outcome="silent"))
    assert ep.is_terminal(ep.blocked(for_seq=1, needs="?"))
    assert not ep.is_terminal(ep.inbound("x", channel="tray"))
    assert not ep.is_terminal(ep.tool_called(for_seq=1, tool="t", args={}))
    assert not ep.is_terminal(ep.tool_returned(for_seq=1, tool="t", ok=True))
    assert not ep.is_terminal(ep.work_finished(for_seq=1, summary="s"))


def test_write_key_is_the_one_spelling():
    assert ep.turn_write_key(41) == "turn:41"


def test_payloads_survive_the_real_store():
    """The codec's only real job is to survive M0. Asserted against the actual
    store, not a mock -- an in-memory round trip proves the JSON works, not that
    an episode comes back out of a log."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        store_dir = Path(d) / "store"
        written = [
            ep.inbound("hello", channel="tray", at="2026-09-23T10:00:00+00:00"),
            ep.completed(
                for_seq=1, outcome="spoke", reply="hi", at="2026-09-23T10:00:01+00:00"
            ),
        ]
        with MemoryStore.open(store_dir) as store:
            store.append_episode(ep.encode(written[0]), write_key="in:1")
            store.append_episode(
                ep.encode(written[1]), write_key=ep.turn_write_key(1)
            )
        # Reopened, because surviving a close is the point.
        with MemoryStore.open(store_dir) as store:
            read = [ep.decode(e.payload) for e in store.episodes_since(0)]
        assert read == written


def test_the_write_key_makes_a_double_terminal_a_log_level_rejection():
    """§2.1: M0's dedup becomes a live invariant check on the executor's most
    dangerous bug, rather than an unused feature."""
    import tempfile
    from pathlib import Path

    from omega.memory import WriteKeyConflict

    with tempfile.TemporaryDirectory() as d:
        with MemoryStore.open(Path(d) / "store") as store:
            store.append_episode(
                ep.encode(ep.completed(for_seq=1, outcome="silent")),
                write_key=ep.turn_write_key(1),
            )
            with pytest.raises(WriteKeyConflict):
                store.append_episode(
                    ep.encode(ep.completed(for_seq=1, outcome="spoke", reply="x")),
                    write_key=ep.turn_write_key(1),
                )


# --- red --------------------------------------------------------------------


def test_an_unknown_kind_is_refused():
    """The typo case, and the reason `KINDS` is closed. `turn.complete` is one
    letter short, valid JSON, and permanent once appended."""
    with pytest.raises(ep.BadPayload) as exc:
        ep.decode(json.dumps({"v": 1, "kind": "turn.complete", "for_seq": 1}).encode())
    assert "turn.complete" in str(exc.value)


def test_decode_refuses_non_json():
    with pytest.raises(ep.BadPayload):
        ep.decode(b"not json at all")


def test_decode_refuses_invalid_utf8():
    with pytest.raises(ep.BadPayload):
        ep.decode(b"\xff\xfe\x00")


def test_decode_refuses_json_that_is_not_an_object():
    for raw in (b"[]", b'"a string"', b"42", b"null"):
        with pytest.raises(ep.BadPayload):
            ep.decode(raw)


def test_decode_refuses_a_missing_required_field():
    with pytest.raises(ep.BadPayload) as exc:
        ep.decode(json.dumps({"v": 1, "kind": "turn.blocked", "for_seq": 1}).encode())
    assert "needs" in str(exc.value)


def test_a_failure_with_no_error_is_refused():
    """The shape that turns a real fault into a silent one."""
    with pytest.raises(ep.BadPayload):
        ep.completed(for_seq=1, outcome="failed")


def test_a_failed_tool_call_with_no_error_is_refused():
    with pytest.raises(ep.BadPayload):
        ep.tool_returned(for_seq=1, tool="t", ok=False)


def test_an_unknown_outcome_is_refused():
    with pytest.raises(ep.BadPayload):
        ep.completed(for_seq=1, outcome="maybe")


def test_an_unknown_urgency_is_refused():
    with pytest.raises(ep.BadPayload):
        ep.inbound("x", channel="tray", urgency="SCREAMING")


def test_an_oversized_episode_is_refused_before_the_append():
    """Rejected here, where the caller still has a stack to blame -- not at the
    append, whose error names a byte limit and not an episode."""
    with pytest.raises(ep.BadPayload) as exc:
        ep.encode(ep.inbound("x" * (MAX_BODY + 100), channel="tray"))
    assert str(MAX_BODY) in str(exc.value)


def test_context_entries_must_be_well_formed():
    for bad in (
        [{"kind": "file", "title": "t"}],                 # no id
        [{"id": "", "kind": "file", "title": "t"}],       # empty id
        [{"id": "a", "kind": "hologram", "title": "t"}],  # unknown kind
        ["just a string"],
        "not a list",
    ):
        with pytest.raises(ep.BadPayload):
            ep.inbound("x", channel="tray", context=bad)


def test_an_empty_channel_is_refused():
    with pytest.raises(ep.BadPayload):
        ep.inbound("x", channel="")


def test_a_partial_blob_reference_is_refused():
    """DL-027 — ``{blob, mime, bytes}`` is all three or none.

    Each partial set below would look valid on read and be useless: a digest
    with no size is a reference nothing can budget for, a size with no digest
    names nothing at all. In an append-only log a shape like that is permanent,
    so it is refused at the one gate rather than puzzled over later.
    """
    full = {
        "id": "ctx-1",
        "kind": "image",
        "title": "Area capture",
        "blob": DIGEST,
        "mime": "image/png",
        "bytes": 184320,
    }
    for drop in ("blob", "mime", "bytes"):
        partial = {k: v for k, v in full.items() if k != drop}
        with pytest.raises(ep.BadPayload) as exc:
            ep.inbound("look", channel="tray", context=[partial])
        assert drop in str(exc.value), "the message must name what is missing"

    for keep in ("blob", "mime", "bytes"):
        lonely = {k: v for k, v in full.items() if k in ("id", "kind", "title", keep)}
        with pytest.raises(ep.BadPayload):
            ep.inbound("look", channel="tray", context=[lonely])


def test_a_malformed_blob_digest_is_refused():
    """The same rule the store enforces, checked here so a reference cannot
    reach the log in a spelling the filesystem would never produce."""
    for bad in (
        "ab" * 32,                    # no algorithm prefix
        "sha256:" + "AB" * 32,        # uppercase: two spellings of one content
        "sha256:" + "ab" * 31,        # too short
        "sha256:deadbeef",            # plausible and wrong
        "sha1:" + "ab" * 32,          # a real algorithm, not this one
        "",
        None,
        184320,
    ):
        with pytest.raises(ep.BadPayload):
            ep.inbound(
                "look",
                channel="tray",
                context=[
                    {
                        "id": "ctx-1",
                        "kind": "image",
                        "title": "t",
                        "blob": bad,
                        "mime": "image/png",
                        "bytes": 1,
                    }
                ],
            )


def test_a_blob_reference_with_a_nonsense_size_or_mime_is_refused():
    for mime, size in (
        ("", 1),              # empty mime: describes nothing
        (None, 1),            # not a string
        ("image/png", -1),    # negative: no file has a negative length
        ("image/png", "184320"),  # a string that looks like a number
        ("image/png", 1.5),   # not an int
        ("image/png", None),
    ):
        with pytest.raises(ep.BadPayload):
            ep.inbound(
                "look",
                channel="tray",
                context=[
                    {
                        "id": "ctx-1",
                        "kind": "image",
                        "title": "t",
                        "blob": DIGEST,
                        "mime": mime,
                        "bytes": size,
                    }
                ],
            )


# --- yellow -----------------------------------------------------------------


def test_a_future_version_is_distinguishable_from_garbage():
    """Two failures that demand opposite responses: a future `v` means *this
    code* is old, while garbage means the *log* is wrong. Only the first is
    fixed by upgrading, so collapsing them into one error class would send a
    reader to the wrong place."""
    with pytest.raises(ep.UnsupportedPayloadVersion):
        ep.decode(json.dumps({"v": 99, "kind": "turn.blocked",
                              "for_seq": 1, "needs": "?", "at": "t"}).encode())
    with pytest.raises(ep.BadPayload):
        ep.decode(json.dumps({"v": "one", "kind": "turn.blocked"}).encode())
    # ...but the future-version error is still a BadPayload, so a caller that
    # only wants "this did not decode" does not have to name both.
    assert issubclass(ep.UnsupportedPayloadVersion, ep.BadPayload)


def test_silence_and_an_empty_reply_are_not_the_same_row():
    """`""` and "chose not to speak" must never collapse. If they do, the
    silence metric (DL-024) counts empty replies as judgement."""
    with pytest.raises(ep.BadPayload):
        ep.completed(for_seq=1, outcome="silent", reply="")
    with pytest.raises(ep.BadPayload):
        ep.completed(for_seq=1, outcome="silent", reply="something")
    spoke_empty = ep.completed(for_seq=1, outcome="spoke", reply="")
    silent = ep.completed(for_seq=1, outcome="silent")
    assert spoke_empty["reply"] == "" and silent["reply"] is None
    assert spoke_empty["outcome"] != silent["outcome"]


def test_true_is_not_sequence_one():
    """`bool` is an `int` in Python, so `for_seq=True` would silently become
    turn 1 -- and `turn_write_key(True)` would collide with turn 1's real key,
    making M0's dedup reject a legitimate write."""
    with pytest.raises(ep.BadPayload):
        ep.blocked(for_seq=True, needs="?")
    with pytest.raises(ep.BadPayload):
        ep.turn_write_key(True)


def blob_item(**overrides):
    item = {
        "id": "ctx-1",
        "kind": "image",
        "title": "Area capture",
        "blob": DIGEST,
        "mime": "image/png",
        "bytes": 184320,
    }
    item.update(overrides)
    return item


def test_true_is_not_one_byte():
    """The same `bool`-is-an-`int` trap on the attachment's size.

    ``bytes: True`` would pass an `isinstance(x, int)` check and record a
    184 KB screenshot as one byte long — a value that is wrong, plausible, and
    permanent. ``False`` is worse: it would read as a zero-length attachment,
    which is a legitimate value, so nothing downstream would ever question it.
    """
    for truthy in (True, False):
        with pytest.raises(ep.BadPayload):
            ep.inbound("x", channel="tray", context=[blob_item(bytes=truthy)])


def test_a_zero_byte_attachment_is_allowed_but_a_negative_one_is_not():
    """Zero bytes is content — an empty file has a digest like anything else —
    while a negative length is not a small mistake but an impossible fact, and
    the two must not be collapsed into one "non-positive" refusal."""
    ok = ep.inbound("x", channel="tray", context=[blob_item(bytes=0)], at="t")
    assert ep.decode(ep.encode(ok))["context"][0]["bytes"] == 0

    with pytest.raises(ep.BadPayload):
        ep.inbound("x", channel="tray", context=[blob_item(bytes=-1)])


def test_seq_zero_and_negative_are_refused():
    """M0 sequences start at 1, so 0 is never a real turn -- and it is exactly
    what an uninitialised counter produces."""
    for bad in (0, -1):
        with pytest.raises(ep.BadPayload):
            ep.blocked(for_seq=bad, needs="?")
        with pytest.raises(ep.BadPayload):
            ep.turn_write_key(bad)


def test_an_empty_inbound_text_is_allowed_but_an_empty_needs_is_not():
    """Not symmetric, on purpose. The tray can send context with no words, so
    empty `text` is real. A `turn.blocked` with no question is not a block --
    it is a turn that stopped and cannot say why, which the user can never
    answer."""
    assert ep.inbound("", channel="tray",
                      context=[{"id": "c", "kind": "image", "title": "shot"}])["text"] == ""
    with pytest.raises(ep.BadPayload):
        ep.blocked(for_seq=1, needs="")


def test_unicode_survives_and_is_measured_in_bytes():
    """The limit is bytes, not characters. A 4-byte emoji counts as 4, and a
    codec that checks `len(str)` passes a payload the log then refuses."""
    payload = ep.inbound("🙂 café", channel="tray", at="2026-09-23T10:00:00+00:00")
    raw = ep.encode(payload)
    assert ep.decode(raw)["text"] == "🙂 café"

    # The bytes outnumber the characters by exactly the multi-byte surplus:
    # the emoji is 4 bytes for 1 character, é is 2 for 1.
    assert len(raw) - len(raw.decode("utf-8")) == (4 - 1) + (2 - 1)

    # And the emoji is stored as itself rather than as `🙂`, which is
    # the point of `ensure_ascii=False` -- the readable-in-a-raw-log property
    # this format was chosen for. Escaped JSON would be *longer* and unreadable.
    assert "🙂".encode("utf-8") in raw
    assert b"\\ud83d" not in raw

    # Just over the byte limit while comfortably under it in characters.
    big = "🙂" * (MAX_BODY // 4)
    with pytest.raises(ep.BadPayload):
        ep.encode(ep.inbound(big, channel="tray"))


def test_an_unknown_extra_field_is_tolerated_on_read():
    """Forward compatibility within a version: a field we do not know is
    ignored, so adding one does not require a version bump and does not make
    older readers reject newer logs."""
    payload = ep.blocked(for_seq=1, needs="?", at="2026-09-23T10:00:00+00:00")
    payload["speculative_future_field"] = "hello"
    assert ep.decode(ep.encode(payload))["speculative_future_field"] == "hello"


def test_resumes_seq_is_optional_and_validated_when_present():
    """An answer to a block is an ordinary turn that *points at* the block."""
    plain = ep.inbound("x", channel="tray")
    assert "resumes_seq" not in plain, "absent, not None -- None is a value"
    answer = ep.inbound("the second one", channel="tray", resumes_seq=41)
    assert ep.decode(ep.encode(answer))["resumes_seq"] == 41
    with pytest.raises(ep.BadPayload):
        ep.inbound("x", channel="tray", resumes_seq=0)


def test_constructors_copy_their_mutable_arguments():
    """A caller who reuses a list must not retroactively edit an episode it
    already built -- and in the log's case, one it already wrote."""
    ctx = [{"id": "c1", "kind": "file", "title": "a.py"}]
    tools = ["grep"]
    payload = ep.inbound("x", channel="tray", context=ctx)
    done = ep.completed(for_seq=1, outcome="spoke", reply="y", tools=tools)
    ctx.append({"id": "c2", "kind": "file", "title": "b.py"})
    tools.append("write")
    assert len(payload["context"]) == 1
    assert done["tools"] == ["grep"]


def test_validation_is_the_same_gate_in_both_directions():
    """A payload cannot be valid going in and invalid coming out. If the two
    paths ever diverge, the log accumulates episodes that can be written and
    never read -- the worst possible failure in an append-only store."""
    bad = {"v": 1, "kind": "turn.completed", "for_seq": 1, "outcome": "failed",
           "reply": None, "tools": [], "error": None, "at": "t"}
    with pytest.raises(ep.BadPayload):
        ep.encode(bad)
    with pytest.raises(ep.BadPayload):
        ep.decode(json.dumps(bad).encode())


def test_the_required_table_covers_every_kind():
    """A new kind with no required-field row would validate vacuously -- it
    would pass because nothing was declared, not because nothing was missing."""
    assert set(ep._REQUIRED) == ep.KINDS


# --- learned claims (DL-042) -------------------------------------------------


def _claim(**over):
    """A valid `claim.extracted`, so each case below can vary one field."""
    fields = dict(
        for_seq=1,
        text="prefers short replies",
        source_seq=1,
        situation="said so while reviewing a long answer",
        explicit=True,
    )
    fields.update(over)
    return ep.claim_extracted(**fields)


def test_a_claim_round_trips_with_its_trigger():
    payload = _claim(trigger={"any": ["deploy"], "hours": [9, 18]})
    back = ep.decode(ep.encode(payload))
    assert back["trigger"] == {"any": ["deploy"], "hours": [9, 18]}
    assert back["explicit"] is True
    assert back["situation"]


def test_a_claim_is_a_record_and_never_an_event():
    """The load-bearing property of the kind. If `claim.extracted` were an
    event, the drain would pick up omega's own learning as new input and every
    filed claim would cost a turn -- and a turn that files claims would feed
    itself forever."""
    from omega import queue

    assert ep.CLAIM_EXTRACTED in queue.RECORD_KINDS
    assert ep.CLAIM_EXTRACTED not in queue.EVENT_KINDS


def test_a_claim_with_no_trigger_is_valid_and_means_always():
    """Tone and working style have no situation because they apply to all of
    them, so always-active has to be representable -- and `None` is the single
    spelling for it."""
    assert _claim(trigger=None)["trigger"] is None
    with pytest.raises(ep.BadPayload) as exc:
        _claim(trigger={})
    assert "null" in str(exc.value)


def test_an_unknown_trigger_field_is_refused():
    """The one place the tolerate-unknown-fields rule is deliberately reversed.
    An ignored trigger field makes a conditional-looking claim fire on
    *everything*; failing to decode is recoverable, a habit silently applying to
    every turn is not."""
    with pytest.raises(ep.BadPayload) as exc:
        _claim(trigger={"unless": ["x"]})
    assert "unknown field" in str(exc.value)


def test_a_claim_with_no_situation_is_refused():
    """DL-034's provenance requirement, enforced at the only moment it can be
    met. Nobody can re-decide a contradiction from a claim that never recorded
    what was going on."""
    with pytest.raises(ep.BadPayload):
        _claim(situation="")


def test_explicit_must_be_a_real_bool_because_it_decides_escalation():
    """`explicit` is the whole of DL-042's answer to "what is core memory". A
    truthy string would make every claim count as deliberately authored, and
    the escalation it drives would fire on omega's own inferences."""
    with pytest.raises(ep.BadPayload):
        _claim(explicit="yes")


def test_an_empty_phrase_list_is_refused():
    """`{"any": []}` matches nothing, so the claim can never fire -- but it
    reads at a glance as a claim that has a trigger."""
    with pytest.raises(ep.BadPayload):
        _claim(trigger={"any": []})
    with pytest.raises(ep.BadPayload):
        _claim(trigger={"any": ["  "]})


def test_an_empty_hour_window_is_refused():
    """The window is half-open, so start == end can never match. It reads as
    "all day" to everyone except the matcher."""
    with pytest.raises(ep.BadPayload):
        _claim(trigger={"hours": [9, 9]})
    assert _claim(trigger={"hours": [22, 6]})  # wrapping is a real window


def test_true_is_not_hour_one():
    """`bool` is an `int` in Python, and the log is permanent."""
    with pytest.raises(ep.BadPayload):
        _claim(trigger={"hours": [True, 6]})


def test_supersedes_is_optional_and_cannot_name_the_source():
    """Two seqs on one record is a shape that invites confusing them, and the
    confusion would typecheck: a claim superseding the episode it was learned
    from is not a fact anyone meant to record."""
    assert "supersedes" not in _claim()
    assert _claim(supersedes=4)["supersedes"] == 4
    with pytest.raises(ep.BadPayload):
        _claim(source_seq=7, supersedes=7)


def test_a_claim_copies_the_trigger_it_was_handed():
    """Same rule as every other constructor here, and it matters more: a caller
    editing the list afterwards would change what omega fires on, in a record
    that has already been written."""
    phrases = ["deploy"]
    payload = _claim(trigger={"any": phrases})
    phrases.append("everything")
    assert payload["trigger"]["any"] == ["deploy"]
