"""What the model is shown when something is attached — DL-031, DL-027.

The bug this file exists to keep dead: an image arrived, stored perfectly, and
reached the model as ``you: what is in this picture?`` and nothing else. The
model was never told a picture existed, so it answered confidently about
nothing. Every case here asserts on the *prompt*, because that is where the
drop was — ingestion and storage were already green while this was broken.

No network and no key: the prompt is built and inspected directly.
"""

from __future__ import annotations

import base64

import pytest

from omega import blobs, episodes, provider
from omega.act import _act_messages
from omega.memory import MemoryStore
from omega.queue import EventQueue
from omega.tools import ToolBox
from omega.turn import (
    MAX_IMAGE_BYTES,
    MAX_TURN_IMAGE_BYTES,
    ActResult,
    TurnContext,
    _judge_messages,
    _render_event,
    _reply_messages,
)

AT = "2026-09-23T12:00:00+00:00"

PNG = b"\x89PNG\r\n\x1a\n" + b"pretend this is pixels"


@pytest.fixture
def q(store: MemoryStore) -> EventQueue:
    return EventQueue(store)


def attach(q: EventQueue, raw: bytes = PNG, *, mime: str = "image/png") -> str:
    """Put bytes in the blob store the way the tray's ``attach`` op does."""
    tmp = q.store.root.parent / "incoming.bin"
    tmp.write_bytes(raw)
    return blobs.BlobStore.open(q.store.root).put(tmp).digest


def item(digest: str, *, kind: str = "image", title: str = "shot.png", **over):
    entry = {
        "id": "c1",
        "kind": kind,
        "title": title,
        "blob": digest,
        "mime": "image/png",
        "bytes": len(PNG),
    }
    entry.update(over)
    return entry


def ctx_for(q: EventQueue, *context, text: str = "what is in this picture?"):
    seq = q.append(
        episodes.inbound(text, channel="tray", context=list(context), at=AT)
    )
    q.claim(seq)
    pending = q.at(seq)
    return TurnContext(
        seq=pending.seq,
        event=pending.payload,
        recalled=(),
        queue=q,
        complete=lambda *a, **k: None,  # never called; the prompt is the subject
    )


def content_of(message: provider.Message):
    return message["content"]


def parts_of(messages):
    """The user turn's content, always as a list of parts."""
    content = content_of(messages[-1])
    return [{"type": "input_text", "text": content}] if isinstance(content, str) else content


def images_in(messages):
    return [p for p in parts_of(messages) if p["type"] == "input_image"]


def text_in(messages):
    return "\n".join(p["text"] for p in parts_of(messages) if p["type"] == "input_text")


# --- the drop itself --------------------------------------------------------


def test_an_attached_image_reaches_act_as_real_bytes(q: EventQueue) -> None:
    """The case that was silently broken. ``act`` gets pixels, not a paragraph
    about a picture it cannot see."""
    ctx = ctx_for(q, item(attach(q)))

    messages = _act_messages(ctx, ToolBox(store_root=q.store.root))
    images = images_in(messages)

    assert len(images) == 1
    assert images[0]["image_url"] == (
        "data:image/png;base64," + base64.b64encode(PNG).decode("ascii")
    )


def test_the_reply_role_sees_the_image_too(q: EventQueue) -> None:
    """``reply`` composes the answer when no tool was needed — "what is in this
    picture" is exactly that shape, so it cannot be the text-only role."""
    ctx = ctx_for(q, item(attach(q)))

    messages = _reply_messages(ctx, ActResult(tools=()))

    assert len(images_in(messages)) == 1


def test_the_judge_is_told_an_image_exists_but_never_shown_it(q: EventQueue) -> None:
    """DL-031's cost split. The judge fires on every event and only routes, so
    it gets the name and not the bytes — which also keeps the cheap model free
    to be text-only."""
    ctx = ctx_for(q, item(attach(q)))

    messages = _judge_messages(ctx)

    assert images_in(messages) == []
    assert "[image: shot.png]" in text_in(messages)
    assert isinstance(content_of(messages[-1]), str), "no parts, no multimodal cost"


def test_an_event_with_no_attachments_is_still_a_plain_string(q: EventQueue) -> None:
    """The control. Nothing about the ordinary turn changes shape — a text-only
    event must not start arriving as a one-element list of parts."""
    ctx = ctx_for(q, text="just a question")

    for messages in (_act_messages(ctx, ToolBox(store_root=q.store.root)),
                     _reply_messages(ctx, ActResult(tools=())),
                     _judge_messages(ctx)):
        assert isinstance(content_of(messages[-1]), str)


# --- what gets named rather than shown --------------------------------------


def test_a_non_image_attachment_is_named_and_not_sent(q: EventQueue) -> None:
    ctx = ctx_for(
        q,
        item(attach(q), kind="file", title="report.pdf", mime="application/pdf"),
    )

    messages = _act_messages(ctx, ToolBox(store_root=q.store.root))

    assert images_in(messages) == []
    assert "[file: report.pdf]" in text_in(messages)


def test_mime_decides_and_not_kind(q: EventQueue) -> None:
    """A client's word for a tray row is a label; the media type is a fact
    about the bytes. A ``file`` that is a PNG is shown; a ``screen`` capture
    recorded as text is not."""
    digest = attach(q)
    shown = ctx_for(q, item(digest, kind="file", title="pasted.png"))
    named = ctx_for(
        q, item(digest, kind="screen", title="log.txt", mime="text/plain")
    )
    box = ToolBox(store_root=q.store.root)

    assert len(images_in(_act_messages(shown, box))) == 1
    assert images_in(_act_messages(named, box)) == []


def test_an_oversized_image_degrades_to_a_line_that_says_so(q: EventQueue) -> None:
    """Fails open, and visibly. The turn survives and the log carries the
    reason — which the silent drop this replaces never did."""
    digest = attach(q, b"x" * (MAX_IMAGE_BYTES + 1))
    ctx = ctx_for(q, item(digest, title="huge.png"))

    messages = _act_messages(ctx, ToolBox(store_root=q.store.root))

    assert images_in(messages) == []
    assert "[image: huge.png — not shown: too large]" in text_in(messages)


def test_the_per_turn_budget_stops_at_the_cap_and_names_the_rest(q: EventQueue) -> None:
    """Each of these passes the per-image check; together they do not. Without
    a running total the budget is unenforced in exactly the case it is for."""
    big = b"y" * (MAX_IMAGE_BYTES - 1)
    digest = attach(q, big)
    count = (MAX_TURN_IMAGE_BYTES // len(big)) + 1
    ctx = ctx_for(q, *(item(digest, title=f"p{n}.png", id=f"c{n}") for n in range(count)))

    messages = _act_messages(ctx, ToolBox(store_root=q.store.root))
    text = text_in(messages)

    assert len(images_in(messages)) < count
    assert "not shown: too many images this turn" in text


def test_a_blob_missing_from_the_store_does_not_fail_the_turn(q: EventQueue) -> None:
    """A reference the store cannot resolve — a replay on another machine, or a
    pruned blob. The prompt says it cannot show it and the turn goes on."""
    ctx = ctx_for(q, item("sha256:" + "ab" * 32, title="gone.png"))

    messages = _act_messages(ctx, ToolBox(store_root=q.store.root))

    assert images_in(messages) == []
    assert "[image: gone.png — not shown: missing]" in text_in(messages)


# --- recall -----------------------------------------------------------------


def test_recalled_images_are_named_not_resent(q: EventQueue) -> None:
    """``RECALL_N`` is 40. Re-sending every picture every turn makes the cost of
    a conversation grow with its length, so history stays text."""
    payload = episodes.inbound(
        "look at this", channel="tray", context=[item(attach(q))], at=AT
    )

    rendered = _render_event(payload)

    assert rendered == "you: look at this [image: shot.png]"
    assert "base64" not in rendered
