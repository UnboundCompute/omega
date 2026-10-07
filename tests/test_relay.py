"""The sense relay and the ``report`` op — DL-072, DL-073 §3.

Real sockets for the channel cases, for test_channel's reason: a transport
tested through a fake of itself is testing the fake. Every claim about what
was learned is read back out of the log — *grade the world, not the words*.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

import pytest

from omega import derive, episodes, listen, machine, notice, provider, relay
from omega.blobs import BlobStore
from omega.channel import Channel, ChannelClient, ChannelError
from omega.executor import Executor
from omega.memory import MemoryStore
from omega.queue import EventQueue

AT = "2026-10-07T12:00:00+00:00"
NOW = datetime(2026, 10, 7, 12, 0, tzinfo=timezone.utc)
SESSION_META = {"source": "claude-code", "project": "-Users-me-work"}
DIGEST = "[the person] rewrite the parser so it streams\n[the person] run the tests"


@pytest.fixture
def q(store: MemoryStore) -> EventQueue:
    return EventQueue(store)


@pytest.fixture
def woken() -> list[int]:
    return []


@pytest.fixture
def channel(q: EventQueue, store_dir: Path, woken: list[int]) -> Iterator[Channel]:
    ch = Channel(q, BlobStore.open(store_dir), port=0, poll=0.005, on_append=woken.append)
    ch.start()
    try:
        yield ch
    finally:
        ch.stop()


@pytest.fixture
def client(channel: Channel) -> Iterator[ChannelClient]:
    c = ChannelClient(channel.address, timeout=5.0)
    try:
        yield c
    finally:
        c.close()


def _written(store: MemoryStore, kind: str) -> list[dict[str, Any]]:
    decoded = (episodes.decode(e.payload) for e in store.episodes_since(0))
    return [p for p in decoded if p["kind"] == kind]


def _kinds(store: MemoryStore) -> list[str]:
    return [episodes.decode(e.payload)["kind"] for e in store.episodes_since(0)]


def _learns(claim: str = "They test before they commit.") -> provider.FakeProvider:
    """Answers both LEARN calls: the reflection pass and a recording's review."""

    def answer(role, messages):
        asked = "".join(str(m["content"]) for m in messages)
        if "JSON only" in asked or "highlights" in asked:
            return json.dumps({"summary": "A standup.", "important": [], "actions": ["ship"]})
        return json.dumps({"claims": [{"text": claim, "situation": "while working"}]})

    return provider.FakeProvider({provider.LEARN: answer})


def _ready(q: EventQueue, fp: provider.FakeProvider | None = None):
    q.append(episodes.inbound("morning", channel="tray"))
    fp = fp or _learns()
    ex = Executor(q, complete=fp.complete)
    ex.recover()
    return ex, fp


def _report(q: EventQueue, **kw: Any) -> int:
    """Append a report exactly as the channel does: same constructor, same key."""
    kw.setdefault("device", "mac")
    payload = episodes.sense_reported(**kw)
    return q.append(payload, episodes.report_write_key(kw["source"], kw["unit"]))


# --- the report op -----------------------------------------------------------


def test_a_report_appends_one_record_under_its_unit_key(
    client: ChannelClient, store: MemoryStore, q: EventQueue
) -> None:
    ack = client.report("transcript", "sess-1", device="mac", body=DIGEST, meta=SESSION_META)
    assert ack["op"] == "ack" and ack["duplicate"] is False

    held = q.at(ack["seq"])
    assert held.write_key == "report:transcript:sess-1"
    assert held.payload["kind"] == episodes.SENSE_REPORTED
    assert held.payload["body"] == DIGEST
    assert held.payload["device"] == "mac"
    assert held.is_event is False


def test_a_resend_is_the_duplicate_ack_not_a_second_record(
    client: ChannelClient, store: MemoryStore, q: EventQueue
) -> None:
    first = client.report("transcript", "sess-1", device="mac", body=DIGEST, meta=SESSION_META)
    head = q.head()
    again = client.report("transcript", "sess-1", device="mac", body=DIGEST, meta=SESSION_META)

    assert again == {**again, "op": "ack", "seq": first["seq"], "duplicate": True}
    assert "conflict" not in again
    assert q.head() == head
    assert len(_written(store, episodes.SENSE_REPORTED)) == 1

    changed = client.report("transcript", "sess-1", device="mac", body="other", meta=SESSION_META)
    assert changed["duplicate"] is True and "conflict" in changed
    assert q.head() == head


def test_a_report_never_wakes_the_drain(
    client: ChannelClient, woken: list[int]
) -> None:
    client.report("transcript", "sess-1", device="mac", body=DIGEST, meta=SESSION_META)
    client.report("presence", "mac@1", device="mac", body={"idle": 3.0})
    assert woken == []
    # The control: the same hook does fire for a message, so the empty list
    # above is the report path declining it and not a hook that never runs.
    said = client.say("hello", id="m-1")
    assert woken == [said["seq"]]


def test_a_report_is_never_an_inbound_and_never_starts_a_turn(
    client: ChannelClient, store: MemoryStore, q: EventQueue
) -> None:
    def no_turn(role, messages):
        raise AssertionError(f"a report started a model call ({role})")

    client.report("transcript", "sess-1", device="mac", body=DIGEST, meta=SESSION_META)
    client.report("machine", "mac@1", device="mac", body=machine.as_body(machine.Reading(cpus=8)))
    ex = Executor(q, complete=provider.FakeProvider({}).complete)
    ex.recover()

    assert ex.drain() == []
    assert episodes.MESSAGE_INBOUND not in _kinds(store)
    assert q.claimed() == q.head()


@pytest.mark.parametrize(
    "request_",
    [
        {"source": "screen", "unit": "u", "device": "mac", "body": ""},
        {"source": "transcript", "unit": "u", "device": "mac", "body": DIGEST},
        {"source": "transcript", "unit": "u", "device": "", "body": DIGEST, "meta": SESSION_META},
        {
            "source": "transcript", "unit": "u", "device": "mac",
            "body": "x" * (episodes.MAX_REPORT_CHARS + 1), "meta": SESSION_META,
        },
        {"source": "usage", "unit": "not-a-day", "device": "mac", "body": "", "meta": {"source": "k"}},
        {"source": "machine", "unit": "mac@1", "device": "mac", "body": {"disk_free": 1}},
        {"source": "presence", "unit": "mac@1", "device": "mac", "body": {"idle": None}},
        {
            "source": "transcript", "unit": "u", "device": "mac", "body": DIGEST,
            "meta": SESSION_META, "reason": "both",
        },
    ],
)
def test_a_malformed_report_is_refused_and_nothing_lands(
    client: ChannelClient, q: EventQueue, request_: dict
) -> None:
    answer = client.request({"v": 1, "op": "report", **request_})
    assert answer["op"] == "error", answer
    assert q.head() == 0


# --- the reported query ------------------------------------------------------


def test_reported_answers_from_receipts_not_from_reports(
    client: ChannelClient, q: EventQueue
) -> None:
    assert client.reported("transcript")["units"] == []
    client.report("transcript", "sess-1", device="mac", body=DIGEST, meta=SESSION_META)
    # Sent but not yet learned from: still owed, so not reported back.
    assert client.reported("transcript")["units"] == []

    q.append(episodes.transcript_ingested(session="sess-1", **SESSION_META, filed=0))
    q.append(episodes.usage_digested(day="2026-10-06", source="knowledgec"))
    assert client.reported("transcript") == {
        "v": 1, "op": "reported", "source": "transcript", "units": ["sess-1"],
    }
    assert client.reported("usage")["units"] == ["2026-10-06"]
    assert client.reported("recording")["units"] == []
    assert client.reported("machine")["op"] == "error"


# --- the core working a report -----------------------------------------------


def test_a_report_becomes_claims_and_the_local_receipt(
    store: MemoryStore, q: EventQueue
) -> None:
    ex, fp = _ready(q)
    seq = _report(q, source="transcript", unit="sess-1", body=DIGEST, meta=SESSION_META)

    assert ex.work_reports() == 1

    (claim,) = _written(store, episodes.CLAIM_EXTRACTED)
    assert claim["text"] == "They test before they commit."
    assert claim["explicit"] is False
    # The claim names the report it came from.
    assert claim["for_seq"] == claim["source_seq"] == seq
    (receipt,) = _written(store, episodes.TRANSCRIPT_INGESTED)
    assert receipt == {**receipt, "session": "sess-1", **SESSION_META, "filed": 1, "reason": None}
    assert "sess-1" in derive.Learned.rebuild(store).ingested

    # The reflection was shown the digest the Mac sent, under the work lens.
    prompt = "\n".join(m["content"] for m in fp.calls_for(provider.LEARN)[0])
    assert "rewrite the parser so it streams" in prompt


def test_a_second_identical_report_is_a_no_op(store: MemoryStore, q: EventQueue) -> None:
    ex, fp = _ready(q)
    seq = _report(q, source="transcript", unit="sess-1", body=DIGEST, meta=SESSION_META, at=AT)
    assert ex.work_reports() == 1
    head = q.head()

    # Byte-identical under the same key: the log hands back the first seq.
    assert _report(q, source="transcript", unit="sess-1", body=DIGEST, meta=SESSION_META, at=AT) == seq
    assert ex.work_reports() == 0
    assert q.head() == head
    assert len(fp.calls_for(provider.LEARN)) == 1
    assert len(_written(store, episodes.TRANSCRIPT_INGESTED)) == 1


def test_a_unit_read_locally_is_not_worked_again_from_a_report(
    store: MemoryStore, q: EventQueue
) -> None:
    """One cursor whatever the origin: the receipt a local read wrote is what
    a report of the same session is checked against."""
    ex, fp = _ready(q)
    q.append(episodes.transcript_ingested(session="sess-1", **SESSION_META, filed=0))
    _report(q, source="transcript", unit="sess-1", body=DIGEST, meta=SESSION_META)

    assert ex.work_reports() == 0
    assert fp.calls_for(provider.LEARN) == []


def test_a_usage_report_ends_in_a_usage_receipt(store: MemoryStore, q: EventQueue) -> None:
    ex, _ = _ready(q)
    _report(q, source="usage", unit="2026-10-06", body="09:00-12:00 Xcode", meta={"source": "knowledgec"})
    assert ex.work_reports() == 1
    (receipt,) = _written(store, episodes.USAGE_DIGESTED)
    assert (receipt["day"], receipt["source"], receipt["filed"]) == ("2026-10-06", "knowledgec", 1)


def test_a_recording_report_is_reviewed_kept_and_receipted(
    store: MemoryStore, q: EventQueue
) -> None:
    ex, _ = _ready(q)
    digest = "sha256:" + "a" * 64
    meta = {"source": "folder", "title": "standup.m4a", "mime": "audio/mp4", "bytes": 1234, "duration": 8.5}
    _report(q, source="recording", unit=digest, body="We agreed to ship Friday.", meta=meta)

    assert ex.work_reports() == 1
    (receipt,) = _written(store, episodes.AUDIO_CAPTURED)
    assert receipt["recording"] == digest and receipt["filed"] == 1
    assert receipt["duration"] == 8.5 and receipt["title"] == "standup.m4a"
    held = BlobStore.open(store.root)
    assert held.path_for(receipt["transcript"]).read_text() == "We agreed to ship Friday."
    assert receipt["review"] is not None
    assert digest in derive.Learned.rebuild(store).heard


def test_a_failed_or_empty_report_is_receipted_without_a_model_call(
    store: MemoryStore, q: EventQueue
) -> None:
    ex, fp = _ready(q)
    _report(q, source="transcript", unit="short", body="", meta=SESSION_META)
    _report(q, source="transcript", unit="broken", body="", meta=SESSION_META, reason="unreadable")

    assert ex.work_reports() == 2
    receipts = {r["session"]: r for r in _written(store, episodes.TRANSCRIPT_INGESTED)}
    assert (receipts["short"]["filed"], receipts["short"]["reason"]) == (0, None)
    assert receipts["broken"]["reason"] == "unreadable"
    assert fp.calls_for(provider.LEARN) == []


def test_reports_are_worked_at_the_local_caps(store: MemoryStore, q: EventQueue) -> None:
    ex, _ = _ready(q)
    for n in range(5):
        _report(q, source="transcript", unit=f"s{n}", body="", meta=SESSION_META)
    assert ex.work_reports() == 3
    assert ex.work_reports() == 2
    assert ex.work_reports() == 0


# --- the relay ---------------------------------------------------------------


def _session(root: Path, id: str, lines: int = 6) -> None:
    directory = root / "-Users-me-work"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{id}.jsonl"
    records = [{"message": {"role": "user", "content": f"step {i} of the parser"}} for i in range(lines)]
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    old = time.time() - 7200
    os.utime(path, (old, old))


def _relay(channel: Channel, **kw: Any) -> relay.Relay:
    return relay.Relay(channel.address, **kw)


def test_the_relay_skips_units_the_core_already_holds_receipts_for(
    channel: Channel, store: MemoryStore, q: EventQueue, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "projects"
    _session(root, "sess-1")
    _session(root, "sess-2")
    q.append(episodes.transcript_ingested(session="sess-1", **SESSION_META, filed=0))
    digested: list[str] = []
    real = relay.transcripts.digest

    def watched(session, **kw):
        digested.append(session.id)
        return real(session, **kw)

    monkeypatch.setattr(relay.transcripts, "digest", watched)

    sent = _relay(channel, transcripts_root=root).pass_once()

    assert sent["transcript"] == 1
    assert digested == ["sess-2"], "an already-receipted session was digested"
    (report,) = _written(store, episodes.SENSE_REPORTED)
    assert report["unit"] == "sess-2"
    assert report["meta"] == SESSION_META
    assert "step 1 of the parser" in report["body"]


def test_a_recording_is_filed_away_only_after_the_ack(
    channel: Channel, store: MemoryStore, q: EventQueue, tmp_path: Path,
) -> None:
    folder = tmp_path / "recordings"
    folder.mkdir()
    audio = folder / "standup.m4a"
    audio.write_bytes(b"not really audio")
    old = time.time() - 3600
    os.utime(audio, (old, old))
    heard = lambda path: listen.Heard(text="We agreed to ship Friday.", duration=8.0)  # noqa: E731

    class Refusing:
        def __init__(self, address):
            self._real = ChannelClient(address, timeout=5.0)

        def request(self, obj):
            if obj["op"] == "report":
                return {"op": "error", "error": "the core is full"}
            return self._real.request(obj)

        def close(self):
            self._real.close()

    refused = relay.Relay(
        channel.address, recordings=str(folder), transcribe=heard,
        not_ready=lambda: None, connect=Refusing,
    ).pass_once()
    assert refused["recording"] == 0
    assert audio.exists(), "a recording left the folder before the core held it"

    sent = _relay(channel, recordings=str(folder), transcribe=heard, not_ready=lambda: None).pass_once()
    assert sent["recording"] == 1
    assert not audio.exists()
    (report,) = _written(store, episodes.SENSE_REPORTED)
    assert report["unit"] == "sha256:" + hashlib.sha256(b"not really audio").hexdigest()
    assert report["meta"]["bytes"] == len(b"not really audio")


def test_the_relay_does_not_transcribe_a_recording_already_heard(
    channel: Channel, q: EventQueue, tmp_path: Path,
) -> None:
    folder = tmp_path / "recordings"
    folder.mkdir()
    audio = folder / "standup.m4a"
    audio.write_bytes(b"heard before")
    old = time.time() - 3600
    os.utime(audio, (old, old))
    digest = "sha256:" + hashlib.sha256(b"heard before").hexdigest()
    q.append(
        episodes.audio_captured(
            recording=digest, source="folder", title="standup.m4a", mime="audio/mp4", bytes=12
        )
    )

    def never(path):
        raise AssertionError("whisper ran over a recording the core already heard")

    sent = _relay(channel, recordings=str(folder), transcribe=never, not_ready=lambda: None).pass_once()
    assert sent["recording"] == 0
    assert not audio.exists()


def test_the_relay_only_ever_reports(tmp_path: Path) -> None:
    """Enforced in the relay, not by the server: the one door out refuses
    anything that is not ``report`` or ``reported``."""
    ops: list[str] = []

    class Recording:
        def __init__(self, address):
            pass

        def request(self, obj):
            ops.append(obj["op"])
            if obj["op"] == "reported":
                return {"op": "reported", "units": []}
            return {"op": "ack", "seq": 1, "duplicate": False}

        def close(self):
            pass

    root = tmp_path / "projects"
    _session(root, "sess-1")
    r = relay.Relay(
        ("127.0.0.1", 1), transcripts_root=root, connect=Recording,
        read_machine=lambda: machine.Reading(cpus=8), read_idle=lambda: 4.0,
    )
    r.pass_once()
    assert ops and set(ops) <= relay.ALLOWED_OPS
    with pytest.raises(ChannelError):
        r._send(Recording(None), "say", text="hi")


def test_a_core_that_is_down_costs_the_relay_a_log_line(tmp_path: Path) -> None:
    lines: list[str] = []

    def down(address):
        raise ConnectionRefusedError("refused")

    r = relay.Relay(("127.0.0.1", 1), connect=down, log=lines.append, read_idle=lambda: 1.0)
    assert sum(r.pass_once().values()) == 0
    assert lines and "cannot reach" in lines[0]


# --- machine readings: on change, heartbeat, staleness -----------------------


def test_a_reading_is_resent_on_change_and_on_the_heartbeat_only() -> None:
    sent: list[dict] = []
    clock = [1_000_000.0]
    reading = [machine.Reading(battery_percent=80, charging=False, on_battery=True, cpus=8)]

    class Client:
        def __init__(self, address):
            pass

        def request(self, obj):
            sent.append(obj)
            return {"op": "ack", "seq": len(sent), "duplicate": False}

        def close(self):
            pass

    r = relay.Relay(("h", 1), connect=Client, clock=lambda: clock[0], read_machine=lambda: reading[0])
    r.pass_once()
    assert len(sent) == 1 and sent[0]["source"] == "machine" and sent[0]["device"] == "mac"
    clock[0] += 60
    reading[0] = machine.Reading(battery_percent=75, charging=False, on_battery=True, cpus=8)
    r.pass_once()
    assert len(sent) == 1, "a five-point move is not a change"
    reading[0] = machine.Reading(battery_percent=70, charging=False, on_battery=True, cpus=8)
    r.pass_once()
    assert len(sent) == 2, "a ten-point move is"
    reading[0] = machine.Reading(battery_percent=70, charging=True, on_battery=False, cpus=8)
    clock[0] += 60
    r.pass_once()
    assert len(sent) == 3, "the power source changing is"
    clock[0] += machine.HEARTBEAT_SECONDS - 1
    r.pass_once()
    assert len(sent) == 3
    clock[0] += 1
    r.pass_once()
    assert len(sent) == 4, "the heartbeat"


def test_a_fresh_reading_renders_labeled_and_a_stale_one_fails_closed() -> None:
    body = machine.as_body(
        machine.Reading(disk_free=5 * 1024**3, disk_total=228 * 1024**3, battery_percent=64,
                        charging=False, on_battery=True, cpus=8)
    )
    fresh = machine.reported_lines("mac", (NOW - timedelta(minutes=89)).isoformat(), body, now=NOW)
    assert any(line.startswith("mac: Disk: 5 GB free") and "LOW" in line for line in fresh)
    assert "mac: Battery: 64%, on battery" in fresh

    stale = machine.reported_lines("mac", (NOW - timedelta(minutes=91)).isoformat(), body, now=NOW)
    assert len(stale) == 1 and "couldn't determine" in stale[0]
    assert "64" not in stale[0] and "GB" not in stale[0], "a stale reading showed its last value"

    unreadable = machine.reported_lines("mac", "yesterday", body, now=NOW)
    assert len(unreadable) == 1 and "couldn't determine" in unreadable[0]


def test_the_look_shows_both_machines_labeled_and_a_stale_mac_as_unknown(
    q: EventQueue,
) -> None:
    q.append(episodes.inbound("hello", channel="tray", at=(NOW - timedelta(hours=2)).isoformat()))
    body = machine.as_body(machine.Reading(battery_percent=12, charging=False, on_battery=True))
    q.append(
        episodes.sense_reported(
            source="machine", unit="mac@a", device="mac", body=body,
            at=(NOW - timedelta(hours=3)).isoformat(),
        ),
        "report:machine:mac@a",
    )
    ex = Executor(
        q, complete=provider.FakeProvider({provider.JUDGE: "SILENT"}).complete,
        online=lambda: True, machine=lambda: ["Disk: 40 GB free of 100 GB (40%)"],
    )
    ex.recover()
    ex.drain()
    assert ex.notice(now=NOW) is True
    look = [episodes.decode(e.payload) for e in q.store.episodes_since(0)][-1]
    assert look["channel"] == notice.CHANNEL
    assert "where omega runs: Disk: 40 GB free of 100 GB (40%)" in look["text"]
    assert "mac: couldn't determine" in look["text"]
    assert "12%" not in look["text"]


def test_a_core_with_no_reports_renders_the_machine_exactly_as_before(q: EventQueue) -> None:
    ex = Executor(q, machine=lambda: ["Disk: 40 GB free"])
    assert ex._machines(("Disk: 40 GB free",), now=NOW) == ("Disk: 40 GB free",)


# --- presence (DL-073 §3) ----------------------------------------------------


def _presence(idle: float, minutes_ago: float) -> derive.Reported:
    return derive.Reported(
        seq=1, device="mac", at=(NOW - timedelta(minutes=minutes_ago)).isoformat(),
        body={"idle": idle},
    )


def test_fresh_and_active_is_at_the_mac() -> None:
    assert machine.at_the_mac(_presence(idle=30, minutes_ago=2), now=NOW) is True


def test_fresh_and_away_is_not() -> None:
    assert machine.at_the_mac(_presence(idle=10 * 60, minutes_ago=2), now=NOW) is False


def test_stale_and_active_is_not() -> None:
    assert machine.at_the_mac(_presence(idle=30, minutes_ago=10), now=NOW) is False


def test_never_reported_is_not() -> None:
    assert machine.at_the_mac(None, now=NOW) is False
    assert machine.at_the_mac(derive.Readings().latest("presence", "mac"), now=NOW) is False


def test_presence_folds_to_the_latest_report_per_device(q: EventQueue) -> None:
    for n, idle in enumerate((900.0, 5.0), start=1):
        q.append(
            episodes.sense_reported(
                source="presence", unit=f"mac@{n}", device="mac", body={"idle": idle},
                at=(NOW - timedelta(minutes=3 - n)).isoformat(),
            ),
            f"report:presence:mac@{n}",
        )
    latest = derive.Readings.rebuild(q.store).latest("presence", "mac")
    assert latest is not None and latest.body == {"idle": 5.0}
    assert machine.at_the_mac(latest, now=NOW) is True


def test_presence_is_sent_on_a_flip_and_every_five_minutes_while_active() -> None:
    sent: list[float] = []
    clock = [1_000_000.0]
    idle = [5.0]

    class Client:
        def __init__(self, address):
            pass

        def request(self, obj):
            sent.append(obj["body"]["idle"])
            return {"op": "ack", "seq": 1, "duplicate": False}

        def close(self):
            pass

    r = relay.Relay(("h", 1), connect=Client, clock=lambda: clock[0], read_idle=lambda: idle[0])
    r.pass_once()
    assert sent == [5.0]
    clock[0] += 60
    r.pass_once()
    assert sent == [5.0], "active and under five minutes: nothing new to say"
    clock[0] += machine.PRESENCE_EVERY_SECONDS
    r.pass_once()
    assert len(sent) == 2
    idle[0] = 600.0
    clock[0] += 60
    r.pass_once()
    assert sent[-1] == 600.0, "the flip to away"
    clock[0] += 3600
    r.pass_once()
    assert len(sent) == 3, "away sends nothing more until it flips back"
    idle[0] = None  # type: ignore[assignment]
    r.pass_once()
    assert len(sent) == 3, "an unreadable idle time sends nothing"


def test_hid_idle_time_parses_from_nanoseconds() -> None:
    assert machine.parse_idle('  |   "HIDIdleTime" = 2500000000\n') == pytest.approx(2.5)
    assert machine.parse_idle("no such key") is None


# --- reports never take a recall slot ----------------------------------------


def _conversation(q: EventQueue, exchanges: int) -> list[int]:
    seqs = []
    for n in range(exchanges):
        seq = q.append(episodes.inbound(f"message {n}", channel="tray"))
        seqs.append(seq)
        seqs.append(q.append(episodes.completed(for_seq=seq, outcome="silent"), episodes.turn_write_key(seq)))
    return seqs


def _presence_reports(q: EventQueue, count: int) -> list[int]:
    return [
        _report(q, source="presence", unit=f"mac@{n}", body={"idle": 4.0}, at=AT)
        for n in range(count)
    ]


def test_fifty_presence_reports_leave_the_conversation_in_recall(q: EventQueue) -> None:
    from omega.turn import RECALL_N, recall

    said = _conversation(q, RECALL_N // 2)
    _presence_reports(q, 50)

    recalled = recall(q, before=q.head() + 1)
    assert [p.seq for p in recalled] == said
    assert all(p.kind != episodes.SENSE_REPORTED for p in recalled)


def test_recall_still_takes_n_when_reports_are_interleaved(q: EventQueue) -> None:
    from omega.turn import recall

    said = []
    for n in range(30):
        said.append(q.append(episodes.inbound(f"message {n}", channel="tray")))
        _report(q, source="presence", unit=f"mac@{n}", body={"idle": 4.0}, at=AT)
    assert [p.seq for p in recall(q, before=q.head() + 1, n=10)] == said[-10:]


def test_a_log_of_only_reports_recalls_nothing_and_ends(q: EventQueue) -> None:
    from omega.turn import recall

    _presence_reports(q, 300)
    assert recall(q, before=q.head() + 1) == []


def test_the_reflection_window_skips_reports(store: MemoryStore, q: EventQueue) -> None:
    fp = _learns()
    ex = Executor(q, complete=fp.complete)
    _conversation(q, 20)
    _presence_reports(q, 50)
    ex.recover()
    assert ex.reflect() is True
    prompt = "\n".join(m["content"] for m in fp.calls_for(provider.LEARN)[0])
    assert "message 0" in prompt and "message 19" in prompt
    assert "idle" not in prompt
