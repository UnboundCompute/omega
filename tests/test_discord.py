"""The Discord adapter (DL-073) — everything but discord.py.

The owner filter, the ``say`` it sends, the routing rule, the chunking and the
cursor are tested here against a real channel over loopback; the discord.py shim
is the one part that is not, and it is kept thin so that is a small gap. Nothing
here imports discord.py, and the suite must pass without it installed.
"""

from __future__ import annotations

import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

import pytest

from omega import discord as adapter
from omega import episodes, notice, provider
from omega.blobs import BlobStore
from omega.channel import Channel, ChannelClient
from omega.executor import Executor
from omega.memory import MemoryStore
from omega.queue import EventQueue

OWNER = 111
BOT = 999


@pytest.fixture
def q(store: MemoryStore) -> EventQueue:
    return EventQueue(store)


@pytest.fixture
def clock() -> dict[str, datetime]:
    """The channel's clock, movable so a presence report can go stale."""
    return {"now": datetime.now(timezone.utc)}


@pytest.fixture
def channel(q: EventQueue, store_dir: Path, clock: dict[str, datetime]) -> Iterator[Channel]:
    ch = Channel(q, BlobStore.open(store_dir), port=0, poll=0.005, now=lambda: clock["now"])
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


@pytest.fixture
def cursor(tmp_path: Path) -> adapter.Cursor:
    state = tmp_path / "state"
    state.mkdir()
    return adapter.Cursor(state / adapter.CURSOR_FILE)


def until(condition, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.005)
    return False


def drain(q: EventQueue, reply: str) -> None:
    ex = Executor(
        q,
        complete=provider.FakeProvider(
            {
                provider.JUDGE: lambda role, messages: "SPEAK",
                provider.ACT: lambda role, messages: reply,
            }
        ).complete,
    )
    ex.recover()
    ex.drain()


def inbound(q: EventQueue, channel: str, text: str = "hi") -> int:
    return q.append(episodes.inbound(text, channel=channel))


def spoke(q: EventQueue, for_seq: int, reply: str) -> int:
    return q.append(episodes.completed(for_seq=for_seq, outcome="spoke", reply=reply))


def presence(client: ChannelClient, idle: float, unit: str = "mac@1") -> None:
    ack = client.report("presence", unit, device="mac", body={"idle": idle})
    assert ack["op"] == "ack"


def wire_of(q: EventQueue, seq: int) -> dict[str, Any]:
    from omega import projection

    (pending,) = [p for p in q.recent(q.head()) if p.seq == seq]
    update = projection.project_pending(pending)
    assert update is not None
    return update.wire()


class Sent:
    """The fake Discord: records each DM, or fails on demand."""

    def __init__(self) -> None:
        self.messages: list[str] = []
        self.fail = False

    def __call__(self, text: str) -> None:
        if self.fail:
            raise RuntimeError("discord is down")
        self.messages.append(text)


def outbound(channel: Channel, cursor: adapter.Cursor, sent: Sent) -> adapter.Outbound:
    return adapter.Outbound(
        channel.address,
        cursor,
        sent,
        connect=lambda a: ChannelClient(a, timeout=5.0),
        sleep=lambda s: None,
        log=lambda line: None,
    )


# --- the owner filter -------------------------------------------------------


def test_a_dm_from_the_owner_is_accepted() -> None:
    assert adapter.accepts(
        author_id=OWNER, owner_id=OWNER, self_id=BOT, in_guild=False, author_is_bot=False
    )


def test_a_dm_from_anyone_else_is_ignored() -> None:
    assert not adapter.accepts(
        author_id=222, owner_id=OWNER, self_id=BOT, in_guild=False, author_is_bot=False
    )


def test_the_owner_in_a_guild_is_ignored() -> None:
    assert not adapter.accepts(
        author_id=OWNER, owner_id=OWNER, self_id=BOT, in_guild=True, author_is_bot=False
    )


def test_the_bots_own_message_is_ignored() -> None:
    assert not adapter.accepts(
        author_id=BOT, owner_id=OWNER, self_id=BOT, in_guild=False, author_is_bot=True
    )
    # Even if the owner id were misconfigured to the bot's own.
    assert not adapter.accepts(
        author_id=BOT, owner_id=BOT, self_id=BOT, in_guild=False, author_is_bot=False
    )


def test_another_bot_is_ignored() -> None:
    assert not adapter.accepts(
        author_id=OWNER, owner_id=OWNER, self_id=BOT, in_guild=False, author_is_bot=True
    )


# --- in: the say ------------------------------------------------------------


def test_a_dm_becomes_a_say_on_the_discord_channel_keyed_by_message_id() -> None:
    assert adapter.say_request(1234, "book it") == {
        "v": 1,
        "op": "say",
        "text": "book it",
        "channel": "discord",
        "id": "discord:1234",
    }


def test_a_dm_lands_as_one_inbound_and_a_redelivery_is_a_duplicate(
    channel: Channel, q: EventQueue
) -> None:
    inbound_ = adapter.Inbound(channel.address, sleep=lambda s: None, log=lambda l: None)
    first = inbound_.say(1234, "book it")
    again = inbound_.say(1234, "book it")
    assert first is not None and first["op"] == "ack" and first["duplicate"] is False
    assert again is not None and again["duplicate"] is True and again["seq"] == first["seq"]
    (pending,) = q.recent(q.head())
    assert pending.payload["kind"] == episodes.MESSAGE_INBOUND
    assert pending.payload["channel"] == "discord"
    assert pending.write_key == "discord:1234"


def test_an_unreachable_core_is_none_after_bounded_retries() -> None:
    tries: list[int] = []
    slept: list[float] = []

    def refuse(address: Any) -> Any:
        tries.append(1)
        raise ConnectionRefusedError

    inbound_ = adapter.Inbound(
        ("127.0.0.1", 1), connect=refuse, sleep=slept.append, attempts=3, log=lambda l: None
    )
    assert inbound_.say(1, "x") is None
    assert len(tries) == 3 and slept == [1.0, 2.0]


def test_the_inbound_projects_its_channel(q: EventQueue) -> None:
    seq = inbound(q, "discord")
    assert wire_of(q, seq)["channel"] == "discord"


# --- out: the routing matrix ------------------------------------------------


def _spoke(reply: str = "here you go") -> dict[str, Any]:
    return {"op": "update", "seq": 9, "for_seq": 7, "state": "complete", "outcome": "spoke", "reply": reply}


def _never() -> bool:
    raise AssertionError("presence must not be asked")


def test_a_reply_to_a_discord_message_is_dmd() -> None:
    assert adapter.decide(_spoke(), origin=lambda: "discord", at_the_mac=_never) == "here you go"


def test_a_reply_to_a_discord_message_is_dmd_even_at_the_mac() -> None:
    assert adapter.decide(_spoke(), origin=lambda: "discord", at_the_mac=lambda: True) == "here you go"


def test_a_reply_to_a_tray_message_is_not_dmd() -> None:
    assert adapter.decide(_spoke(), origin=lambda: "tray", at_the_mac=lambda: False) is None


def test_an_unprompted_message_at_the_mac_is_not_dmd() -> None:
    assert adapter.decide(_spoke(), origin=lambda: None, at_the_mac=lambda: True) is None


def test_an_unprompted_message_away_from_the_mac_is_dmd() -> None:
    assert adapter.decide(_spoke(), origin=lambda: None, at_the_mac=lambda: False) == "here you go"


def test_silence_and_progress_are_never_dmd() -> None:
    silent = {**_spoke(), "outcome": "silent", "reply": None}
    working = {"op": "update", "seq": 8, "for_seq": 7, "state": "working", "tool": "x"}
    understood = {"op": "update", "seq": 7, "for_seq": 7, "state": "understood", "text": "hi"}
    for update in (silent, working, understood):
        assert adapter.decide(update, origin=lambda: "discord", at_the_mac=lambda: False) is None


def test_a_failed_turn_tells_discord_and_nobody_else() -> None:
    failed = {"op": "update", "seq": 9, "for_seq": 7, "state": "failed", "outcome": "failed", "reply": None}
    assert adapter.decide(failed, origin=lambda: "discord", at_the_mac=_never) == adapter.TURN_FAILED
    assert adapter.decide(failed, origin=lambda: "tray", at_the_mac=_never) is None
    assert adapter.decide(failed, origin=lambda: None, at_the_mac=lambda: False) is None


def test_a_blocked_turn_asks_on_discord() -> None:
    blocked = {"op": "update", "seq": 9, "for_seq": 7, "state": "blocked", "needs": "which day?"}
    assert adapter.decide(blocked, origin=lambda: "discord", at_the_mac=_never) == "which day?"


# --- out: the same matrix through a real core -------------------------------


def _handled(
    channel: Channel, q: EventQueue, cursor: adapter.Cursor, reply_seq: int
) -> list[str]:
    sent = Sent()
    with ChannelClient(channel.address, timeout=5.0) as c:
        outbound(channel, cursor, sent).handle(c, wire_of(q, reply_seq))
    return sent.messages


def test_live_a_discord_reply_is_dmd(channel: Channel, q: EventQueue, cursor: adapter.Cursor) -> None:
    asked = inbound(q, "discord")
    assert _handled(channel, q, cursor, spoke(q, asked, "done")) == ["done"]


def test_live_a_tray_reply_is_not(channel: Channel, q: EventQueue, cursor: adapter.Cursor) -> None:
    asked = inbound(q, "tray")
    assert _handled(channel, q, cursor, spoke(q, asked, "done")) == []


@pytest.mark.parametrize("unprompted", sorted(notice.UNPROMPTED_CHANNELS))
def test_live_unprompted_at_the_mac_is_not(
    channel: Channel, client: ChannelClient, q: EventQueue, cursor: adapter.Cursor, unprompted: str
) -> None:
    presence(client, idle=5.0)
    look = inbound(q, unprompted, "what is open")
    assert _handled(channel, q, cursor, spoke(q, look, "the branch is dirty")) == []


@pytest.mark.parametrize("unprompted", sorted(notice.UNPROMPTED_CHANNELS))
def test_live_unprompted_away_is_dmd(
    channel: Channel, client: ChannelClient, q: EventQueue, cursor: adapter.Cursor, unprompted: str
) -> None:
    presence(client, idle=20 * 60.0)
    look = inbound(q, unprompted, "what is open")
    assert _handled(channel, q, cursor, spoke(q, look, "nudge")) == ["nudge"]


def test_live_unprompted_with_no_presence_is_dmd(
    channel: Channel, q: EventQueue, cursor: adapter.Cursor
) -> None:
    look = inbound(q, notice.CHANNEL)
    assert _handled(channel, q, cursor, spoke(q, look, "nudge")) == ["nudge"]


def test_live_unprompted_with_stale_presence_is_dmd(
    channel: Channel, client: ChannelClient, q: EventQueue, cursor: adapter.Cursor,
    clock: dict[str, datetime],
) -> None:
    presence(client, idle=5.0)
    clock["now"] += timedelta(minutes=11)
    look = inbound(q, notice.CHANNEL)
    assert _handled(channel, q, cursor, spoke(q, look, "nudge")) == ["nudge"]


def test_a_long_reply_goes_out_as_several_dms(
    channel: Channel, q: EventQueue, cursor: adapter.Cursor
) -> None:
    asked = inbound(q, "discord")
    long = "\n\n".join("para %d " % i + "x" * 900 for i in range(5))
    parts = _handled(channel, q, cursor, spoke(q, asked, long))
    assert len(parts) == 3 and all(len(p) <= 2000 for p in parts)


# --- the presence op --------------------------------------------------------


def test_presence_with_no_report_is_not_at_the_mac(client: ChannelClient) -> None:
    assert client.presence() == {"v": 1, "op": "presence", "at_the_mac": False}


def test_presence_fresh_and_active_is_at_the_mac(client: ChannelClient) -> None:
    presence(client, idle=30.0)
    assert client.presence() == {"v": 1, "op": "presence", "at_the_mac": True}


def test_presence_reads_the_latest_report(client: ChannelClient) -> None:
    presence(client, idle=30.0, unit="mac@1")
    presence(client, idle=15 * 60.0, unit="mac@2")
    assert client.presence()["at_the_mac"] is False
    presence(client, idle=2.0, unit="mac@3")
    assert client.presence()["at_the_mac"] is True


def test_presence_goes_stale(client: ChannelClient, clock: dict[str, datetime]) -> None:
    presence(client, idle=30.0)
    clock["now"] += timedelta(minutes=10)
    assert client.presence()["at_the_mac"] is False


def test_presence_appends_nothing(client: ChannelClient, q: EventQueue) -> None:
    head = q.head()
    client.presence()
    assert q.head() == head


# --- chunking ---------------------------------------------------------------


def test_a_short_reply_is_one_message() -> None:
    assert adapter.chunks("hello") == ["hello"]
    assert adapter.chunks("x" * 2000) == ["x" * 2000]


def test_a_long_reply_splits_at_paragraphs_first() -> None:
    a, b = "a" * 1500, "b" * 1500
    assert adapter.chunks(f"{a}\n\n{b}") == [a, b]


def test_then_at_lines_then_at_words() -> None:
    a, b = "a" * 1500, "b" * 1500
    assert adapter.chunks(f"{a}\n{b}") == [a, b]
    words = " ".join(["word"] * 1000)  # 4999 chars
    parts = adapter.chunks(words)
    assert all(len(p) <= 2000 for p in parts)
    assert " ".join(parts) == words, "a word split at a space loses nothing"
    assert all(not p.startswith(" ") and not p.endswith(" ") for p in parts)


def test_text_with_no_boundary_is_cut_hard_at_2000() -> None:
    text = "z" * 4500
    assert [len(p) for p in adapter.chunks(text)] == [2000, 2000, 500]
    assert "".join(adapter.chunks(text)) == text


def test_every_chunk_is_within_the_limit_and_nothing_is_lost() -> None:
    text = ("line one\n" * 300) + ("\n\n" + "y" * 2500)
    parts = adapter.chunks(text)
    assert all(0 < len(p) <= 2000 for p in parts)
    assert "".join(parts).replace("\n", "") == text.replace("\n", "")


# --- the cursor -------------------------------------------------------------


def test_the_cursor_round_trips_and_starts_empty(cursor: adapter.Cursor) -> None:
    assert cursor.load() is None
    cursor.save(41)
    assert cursor.load() == 41


def test_a_first_start_begins_at_head_and_never_dms_history(
    channel: Channel, q: EventQueue, cursor: adapter.Cursor
) -> None:
    old = inbound(q, "discord", "from before")
    spoke(q, old, "an old reply")
    head = q.head()

    sent = Sent()
    stopping = threading.Event()
    out = outbound(channel, cursor, sent)
    thread = threading.Thread(target=out.run, args=(stopping,), daemon=True)
    thread.start()
    try:
        assert until(lambda: cursor.load() == head), "the cursor starts at head"
        assert until(lambda: out._subscribed)
        asked = inbound(q, "discord", "new")
        reply = spoke(q, asked, "a new reply")
        assert until(lambda: sent.messages == ["a new reply"])
        assert until(lambda: cursor.load() == reply), "the cursor moves past what was sent"
        time.sleep(0.05)
        assert sent.messages == ["a new reply"], "history was never sent"
    finally:
        stopping.set()
        channel.stop()
        thread.join(timeout=5)
    assert not thread.is_alive()


def test_a_restart_resumes_from_the_cursor(
    channel: Channel, q: EventQueue, cursor: adapter.Cursor
) -> None:
    asked = inbound(q, "discord")
    first = spoke(q, asked, "one")
    cursor.save(first)
    second = spoke(q, inbound(q, "discord"), "two")
    sent = Sent()
    with ChannelClient(channel.address, timeout=5.0) as c:
        out = outbound(channel, cursor, sent)
        stopping = threading.Event()
        def follow() -> None:
            try:
                out.session(c, stopping)
            except Exception:  # the connection closing under it ends the test
                pass

        thread = threading.Thread(target=follow, daemon=True)
        thread.start()
        assert until(lambda: sent.messages == ["two"])
        assert until(lambda: cursor.load() == second)
        stopping.set()
    thread.join(timeout=5)


def test_a_failed_dm_does_not_move_the_cursor(
    channel: Channel, q: EventQueue, cursor: adapter.Cursor
) -> None:
    asked = inbound(q, "discord")
    reply = spoke(q, asked, "done")
    cursor.save(asked)
    sent = Sent()
    sent.fail = True
    with ChannelClient(channel.address, timeout=5.0) as c:
        with pytest.raises(RuntimeError):
            outbound(channel, cursor, sent).handle(c, wire_of(q, reply))
    assert cursor.load() == asked, "a reply that did not land is sent again"


def test_an_update_not_sent_still_moves_the_cursor(
    channel: Channel, q: EventQueue, cursor: adapter.Cursor
) -> None:
    asked = inbound(q, "tray")
    reply = spoke(q, asked, "done")
    assert _handled(channel, q, cursor, reply) == []
    assert cursor.load() == reply


def test_a_lost_core_reconnects_with_backoff_and_does_not_raise(cursor: adapter.Cursor) -> None:
    slept: list[float] = []
    stopping = threading.Event()

    def sleep(seconds: float) -> None:
        slept.append(seconds)
        if len(slept) >= 8:
            stopping.set()

    def refuse(address: Any) -> Any:
        raise ConnectionRefusedError

    adapter.Outbound(
        ("127.0.0.1", 1), cursor, Sent(), connect=refuse, sleep=sleep, log=lambda l: None
    ).run(stopping)
    assert slept == [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 60.0, 60.0]


def test_a_real_turn_from_discord_comes_back_out(
    channel: Channel, q: EventQueue, cursor: adapter.Cursor
) -> None:
    """The say, the executor, the projection and the routing, end to end."""
    ack = adapter.Inbound(channel.address, log=lambda l: None).say(77, "how's it going")
    assert ack is not None
    drain(q, "all good")
    replies = [
        p.seq for p in q.recent(q.head()) if p.payload["kind"] == episodes.TURN_COMPLETED
    ]
    assert len(replies) == 1
    assert _handled(channel, q, cursor, replies[0]) == ["all good"]


# --- unconfigured -----------------------------------------------------------


def _never_run(**kwargs: Any) -> int:
    raise AssertionError("an unconfigured adapter must not start")


@pytest.mark.parametrize(
    "environ",
    [
        {},
        {"DISCORD_OWNER_ID": "111"},
        {"DISCORD_TOKEN": "t"},
        {"DISCORD_TOKEN": "t", "DISCORD_OWNER_ID": "not-a-number"},
        {"DISCORD_TOKEN": "  ", "DISCORD_OWNER_ID": "111"},
    ],
)
def test_unconfigured_logs_why_and_exits_zero(environ: dict[str, str], tmp_path: Path) -> None:
    lines: list[str] = []
    code = adapter.main(["--state", str(tmp_path)], environ=environ, run=_never_run, log=lines.append)
    assert code == 0
    assert lines and "off" in lines[0]


def test_configured_starts_with_the_owner_and_the_cursor(tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def run(**kwargs: Any) -> int:
        seen.update(kwargs)
        return 0

    code = adapter.main(
        ["--state", str(tmp_path), "--port", "17999"],
        environ={"DISCORD_TOKEN": "secret-token", "DISCORD_OWNER_ID": "111"},
        run=run,
        log=lambda l: None,
    )
    assert code == 0
    assert seen["owner_id"] == 111 and seen["address"] == ("127.0.0.1", 17999)
    assert seen["token"] == "secret-token"


def test_the_suite_never_imported_discord_py() -> None:
    assert "discord" not in sys.modules, "discord.py is imported only by the shim"
