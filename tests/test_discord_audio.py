"""Audio over Discord, heard by the adapter (DL-079).

No network, no ffmpeg and no discord.py: a message is a plain object with the
fields the adapter reads, an attachment's ``save`` writes fixed bytes, and the
transcriber is a function. The core is real — a channel over loopback on a
temp store — so what lands is checked in the log, not in the adapter's account
of it.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Optional

import pytest

from omega import discord as adapter
from omega import episodes, listen, provider, sarvam
from omega.blobs import BlobStore
from omega.channel import Channel, ChannelClient
from omega.executor import Executor
from omega.memory import MemoryStore
from omega.queue import EventQueue

KEY = "sk_discord_secret_4a5b6c"
HEARING = sarvam.Settings(key=KEY, mode="codemix")
AUDIO = b"pretend this is opus"


@pytest.fixture
def q(store: MemoryStore) -> EventQueue:
    return EventQueue(store)


@pytest.fixture
def channel(q: EventQueue, store_dir: Path) -> Iterator[Channel]:
    ch = Channel(q, BlobStore.open(store_dir), port=0, poll=0.005)
    ch.start()
    try:
        yield ch
    finally:
        ch.stop()


class Room:
    """The DM channel: records what the bot says back."""

    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send(self, text: str) -> None:
        self.sent.append(text)

    async def typing(self) -> None:
        return None


class Upload:
    def __init__(
        self, filename: str, content_type: Optional[str], data: bytes = AUDIO, id: int = 7
    ) -> None:
        self.id = id
        self.filename = filename
        self.content_type = content_type
        self.data = data
        self.saves = 0

    async def save(self, path: Path) -> None:
        self.saves += 1
        Path(path).write_bytes(self.data)


def message(*uploads: Upload, content: str = "", voice: bool = False, id: int = 5001) -> Any:
    return SimpleNamespace(
        id=id,
        content=content,
        attachments=list(uploads),
        flags=SimpleNamespace(voice=voice),
        channel=Room(),
    )


class Ear:
    """The transcriber: says ``text``, or raises ``NotHeard(fail)``."""

    def __init__(self, text: str = "kal standup ten baje hai", fail: Optional[str] = None) -> None:
        self.text = text
        self.fail = fail
        self.calls: list[dict[str, Any]] = []

    def __call__(self, path: Path, **kw: Any) -> listen.Heard:
        self.calls.append({"path": path, "bytes": Path(path).read_bytes(), **kw})
        if self.fail is not None:
            raise listen.NotHeard(self.fail)
        return listen.Heard(text=self.text, duration=12.5, model="sarvam:saaras:v4")


def dm(channel: Channel, ear: Ear, *, hearing: Optional[sarvam.Settings] = HEARING,
       ready: Optional[str] = None) -> adapter.Dm:
    inbound = adapter.Inbound(
        channel.address,
        connect=lambda a: ChannelClient(a, timeout=5.0),
        sleep=lambda s: None,
        log=lambda line: None,
    )
    return adapter.Dm(
        inbound, hearing=hearing, transcribe=ear, not_ready=lambda: ready,
        log=lambda line: None,
    )


def logged(q: EventQueue) -> list[dict[str, Any]]:
    return [q.at(seq).payload for seq in range(1, q.head() + 1)]


def keys(q: EventQueue) -> list[Optional[str]]:
    return [p.write_key for p in q.recent(q.head())]


def run(handler: adapter.Dm, msg: Any) -> None:
    asyncio.run(handler.handle(msg))


# --- which attachments are audio ---------------------------------------------


@pytest.mark.parametrize(
    "filename, content_type, audio",
    [
        ("voice-message.ogg", "audio/ogg", True),
        ("memo", "audio/mpeg; codecs=mp3", True),
        ("standup.m4a", None, True),
        ("standup.M4A", "application/octet-stream", True),
        ("clip.mp4", "video/mp4", False),
        ("photo.png", "image/png", False),
        ("notes.pdf", "application/pdf", False),
    ],
)
def test_audio_is_the_declared_type_else_the_suffix(
    filename: str, content_type: Optional[str], audio: bool
) -> None:
    assert adapter.is_audio(filename, content_type) is audio


# --- a voice note ------------------------------------------------------------


def test_a_voice_note_becomes_a_marked_say_under_the_message_key(
    channel: Channel, q: EventQueue
) -> None:
    ear = Ear()
    msg = message(Upload("voice-message.ogg", "audio/ogg"), voice=True)
    run(dm(channel, ear), msg)

    (inbound,) = logged(q)
    assert inbound["kind"] == episodes.MESSAGE_INBOUND
    assert inbound["channel"] == "discord"
    assert inbound["text"] == "(voice note) kal standup ten baje hai"
    assert keys(q) == ["discord:5001"]
    # The adapter's settings reached Sarvam, the downloaded bytes were heard.
    (call,) = ear.calls
    assert call["bytes"] == AUDIO
    assert (call["key"], call["model"], call["language"], call["mode"]) == (
        KEY, "saaras:v4", "unknown", "codemix",
    )
    assert msg.channel.sent == []


def test_a_voice_note_with_typed_words_keeps_both(channel: Channel, q: EventQueue) -> None:
    msg = message(Upload("voice-message.ogg", "audio/ogg"), content="for tomorrow:", voice=True)
    run(dm(channel, Ear()), msg)
    (inbound,) = logged(q)
    assert inbound["text"] == "for tomorrow:\n\n(voice note) kal standup ten baje hai"


def test_a_replayed_voice_note_is_heard_once_and_filed_once(
    channel: Channel, q: EventQueue
) -> None:
    ear = Ear()
    handler = dm(channel, ear)
    msg = message(Upload("voice-message.ogg", "audio/ogg"), voice=True)
    run(handler, msg)
    run(handler, msg)  # a gateway replay: not downloaded or transcribed again
    assert len(ear.calls) == 1

    # A restarted adapter has no memory of it; the write key is the guard.
    run(dm(channel, ear), msg)
    assert len(ear.calls) == 2
    assert [p["kind"] for p in logged(q)] == [episodes.MESSAGE_INBOUND]


# --- a recording -------------------------------------------------------------


def test_an_audio_file_becomes_a_recording_report_keyed_on_its_audio(
    channel: Channel, q: EventQueue
) -> None:
    upload = Upload("standup.m4a", "audio/mp4")
    msg = message(upload)
    run(dm(channel, Ear()), msg)

    (report,) = logged(q)
    digest = "sha256:" + hashlib.sha256(AUDIO).hexdigest()
    assert report["kind"] == episodes.SENSE_REPORTED
    assert report["source"] == "recording"
    assert report["unit"] == digest
    assert report["device"] == "discord"
    assert report["body"] == "kal standup ten baje hai"
    assert report["reason"] is None
    assert report["meta"] == {
        "source": "discord", "title": "standup.m4a", "mime": "audio/mp4",
        "bytes": len(AUDIO), "duration": 12.5,
    }
    assert keys(q) == [f"report:recording:{digest}"]
    (ack,) = msg.channel.sent
    assert "Got standup.m4a and transcribed it" in ack
    assert "review isn't sent here" in ack


def test_the_core_reviews_a_discord_recording_like_a_relayed_one(
    channel: Channel, q: EventQueue, store: MemoryStore
) -> None:
    run(dm(channel, Ear("We agreed to ship Friday.")), message(Upload("standup.m4a", "audio/mp4")))

    def answer(role, messages):
        asked = "".join(str(m["content"]) for m in messages)
        if "JSON only" in asked or "highlights" in asked:
            return json.dumps({"summary": "A standup.", "important": [], "actions": ["ship"]})
        return json.dumps({"claims": [{"text": "They ship on Fridays.", "situation": "work"}]})

    ex = Executor(q, complete=provider.FakeProvider({provider.LEARN: answer}).complete)
    ex.recover()
    assert ex.work_reports() == 1
    receipts = [p for p in logged(q) if p["kind"] == episodes.AUDIO_CAPTURED]
    assert len(receipts) == 1
    assert receipts[0]["source"] == "discord" and receipts[0]["review"] is not None
    assert receipts[0]["filed"] == 1


def test_the_same_audio_sent_twice_is_filed_once(channel: Channel, q: EventQueue) -> None:
    run(dm(channel, Ear()), message(Upload("standup.m4a", "audio/mp4"), id=1))
    again = message(Upload("standup copy.m4a", "audio/mp4"), id=2)
    run(dm(channel, Ear()), again)
    assert [p["kind"] for p in logged(q)] == [episodes.SENSE_REPORTED]
    assert "already has standup copy.m4a" in again.channel.sent[0]


def test_a_failed_recording_is_reported_with_its_reason_and_told_without_the_key(
    channel: Channel, q: EventQueue
) -> None:
    ear = Ear(fail=f"sarvam refused piece 1 of 2: HTTP 403: bad key {KEY}")
    msg = message(Upload("standup.m4a", "audio/mp4"))
    run(dm(channel, ear), msg)

    (report,) = logged(q)
    assert report["body"] == "" and report["reason"].startswith("sarvam refused piece 1 of 2")
    (said,) = msg.channel.sent
    assert "couldn't transcribe standup.m4a" in said and "HTTP 403" in said
    for text in [said, json.dumps(report)]:
        assert KEY not in text


def test_a_failed_voice_note_is_told_and_nothing_is_said(channel: Channel, q: EventQueue) -> None:
    msg = message(Upload("voice-message.ogg", "audio/ogg"), voice=True)
    run(dm(channel, Ear(fail=f"could not reach sarvam at piece 1 of 1: {KEY}")), msg)
    assert logged(q) == []
    (said,) = msg.channel.sent
    assert "couldn't transcribe that voice note" in said
    assert KEY not in said


# --- refusals ----------------------------------------------------------------


def test_a_non_audio_attachment_is_refused(channel: Channel, q: EventQueue) -> None:
    ear = Ear()
    upload = Upload("photo.png", "image/png")
    msg = message(upload)
    run(dm(channel, ear), msg)
    assert logged(q) == []
    assert msg.channel.sent == [adapter.ATTACHMENTS_REFUSED]
    assert "Only audio" in adapter.ATTACHMENTS_REFUSED
    assert upload.saves == 0 and ear.calls == []


def test_a_mixed_message_is_refused_whole(channel: Channel, q: EventQueue) -> None:
    ear = Ear()
    msg = message(Upload("standup.m4a", "audio/mp4"), Upload("photo.png", "image/png"),
                  content="see attached")
    run(dm(channel, ear), msg)
    assert logged(q) == [] and ear.calls == []
    assert msg.channel.sent == [adapter.ATTACHMENTS_REFUSED]


@pytest.mark.parametrize("voice", [True, False])
def test_with_no_key_audio_is_told_it_cannot_be_transcribed(
    channel: Channel, q: EventQueue, voice: bool
) -> None:
    ear = Ear()
    upload = Upload("voice-message.ogg", "audio/ogg")
    msg = message(upload, voice=voice)
    run(dm(channel, ear, hearing=None), msg)
    # About the machine, not the file: no report that would mark it heard.
    assert logged(q) == []
    assert msg.channel.sent == [adapter.NO_KEY]
    assert upload.saves == 0 and ear.calls == []


def test_without_ffmpeg_audio_is_told_and_not_reported(channel: Channel, q: EventQueue) -> None:
    msg = message(Upload("standup.m4a", "audio/mp4"))
    run(dm(channel, Ear(), ready="ffmpeg is not installed"), msg)
    assert logged(q) == []
    assert "ffmpeg is not installed" in msg.channel.sent[0]


def test_typed_words_still_go_in_as_typed(channel: Channel, q: EventQueue) -> None:
    run(dm(channel, Ear()), message(content="  hello there "))
    (inbound,) = logged(q)
    assert inbound["text"] == "  hello there "


# --- configuration -----------------------------------------------------------


def test_main_hands_the_adapter_the_sarvam_settings(tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def start(**kwargs: Any) -> int:
        seen.update(kwargs)
        return 0

    environ = {
        "DISCORD_TOKEN": "t", "DISCORD_OWNER_ID": "111",
        "SARVAM_API_KEY": KEY, "SARVAM_STT_MODE": "codemix",
    }
    adapter.main(["--state", str(tmp_path)], environ=environ, run=start, log=lambda l: None)
    assert seen["hearing"] == sarvam.Settings(key=KEY, mode="codemix")

    lines: list[str] = []
    adapter.main(["--state", str(tmp_path)], environ={"DISCORD_TOKEN": "t", "DISCORD_OWNER_ID": "111"},
                 run=start, log=lines.append)
    assert seen["hearing"] is None
    assert any("SARVAM_API_KEY is not set" in l for l in lines)
