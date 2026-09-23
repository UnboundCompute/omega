"""The duplex localhost channel — M1_SPEC.md §Q10, §1.1, §1.6, §Q12.

Real sockets, real threads, no mocks: a transport tested through a fake of
itself is testing the fake. Every case here connects over loopback TCP and
speaks the JSON-lines protocol a tray would speak.
"""

from __future__ import annotations

import json
import socket
import struct
import threading
import time
from typing import Iterator

import pytest

from omega import episodes, provider
from omega.channel import MAX_LINE, PROTOCOL, Channel, ChannelClient, ChannelError
from omega.executor import Executor
from omega.memory import MemoryStore
from omega.projection import COMPLETE, UNDERSTOOD, WORKING
from omega.queue import EventQueue

AT = "2026-09-23T12:00:00+00:00"


@pytest.fixture
def q(store: MemoryStore) -> EventQueue:
    return EventQueue(store)


@pytest.fixture
def channel(q: EventQueue) -> Iterator[Channel]:
    """Port 0: the OS picks a free one. A fixed port in a test suite is a
    collision waiting for a second test run."""
    ch = Channel(q, port=0, poll=0.005)
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


def speaking(reply: str = "done") -> provider.FakeProvider:
    return provider.FakeProvider(
        {
            provider.JUDGE: lambda role, messages: "SPEAK",
            provider.ACT: lambda role, messages: reply,
        }
    )


def drain(q: EventQueue, **kwargs) -> None:
    ex = Executor(q, complete=speaking(**kwargs).complete)
    ex.recover()
    ex.drain()


def until(condition, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.005)
    return False


# --- green: the two ops -----------------------------------------------------


def test_connecting_says_hello_with_the_current_head(
    channel: Channel, q: EventQueue
) -> None:
    q.append(episodes.inbound("earlier", channel="tray", at=AT))
    with ChannelClient(channel.address) as client:
        assert client.hello["op"] == "hello"
        assert client.hello["v"] == PROTOCOL
        assert client.hello["head"] == 1, "a client needs a cursor to resume from"


def test_saying_something_appends_one_episode_and_acks_its_seq(
    client: ChannelClient, q: EventQueue
) -> None:
    """§1.1 — the returned seq *is* the acknowledgement token. Nothing else in
    the protocol acknowledges, because nothing else is durable."""
    ack = client.say(
        "book the flight",
        context=[{"id": "ctx-9", "kind": "file", "title": "itinerary.pdf"}],
    )
    assert ack == {"v": PROTOCOL, "op": "ack", "seq": 1, "duplicate": False}

    assert q.head() == 1
    payload = q.at(1).payload
    assert payload["kind"] == episodes.MESSAGE_INBOUND
    assert payload["text"] == "book the flight"
    assert payload["channel"] == "tray"
    assert payload["urgency"] == "normal"
    assert payload["context"] == [
        {"id": "ctx-9", "kind": "file", "title": "itinerary.pdf"}
    ], "the log keeps the whole item; only the projection narrows it"


def test_the_channel_is_a_producer_not_a_second_entry_point(
    client: ChannelClient, q: EventQueue
) -> None:
    """§1.6 — the only way to cause a turn is to append an episode. If any other
    op could reach the loop, M5's clock would be a second engine and the two
    would drift."""
    client.subscribe(0)
    client.request({"v": PROTOCOL, "op": "ping"})
    client.request({"v": PROTOCOL, "op": "nonsense"})
    assert q.head() == 0, "no op but say may put anything in the log"


def test_a_subscriber_receives_the_whole_turn_in_order(
    client: ChannelClient, q: EventQueue
) -> None:
    client.subscribe(0)
    ack = client.say("hello omega")
    drain(q, reply="hi back")

    first, second = client.read(), client.read()
    assert first["state"] == UNDERSTOOD
    assert first["seq"] == ack["seq"]
    assert first["text"] == "hello omega"
    assert second["state"] == COMPLETE
    assert second["for_seq"] == ack["seq"]
    assert second["outcome"] == "spoke"
    assert second["reply"] == "hi back"


def test_a_silent_turn_reaches_the_client_as_a_success(
    client: ChannelClient, q: EventQueue
) -> None:
    """DL-011 end to end. The tray must be able to show "I read it and had
    nothing to add" — not an error, and not an empty message bubble."""
    client.subscribe(0)
    client.say("fyi, the meeting moved")
    ex = Executor(
        q,
        complete=provider.FakeProvider(
            {provider.JUDGE: lambda role, messages: "SILENT"}
        ).complete,
    )
    ex.recover()
    ex.drain()

    assert client.read()["state"] == UNDERSTOOD
    completed = client.read()
    assert completed["state"] == COMPLETE
    assert completed["outcome"] == "silent"
    assert completed["reply"] is None


def test_a_subscriber_that_reconnects_is_served_from_the_log(
    channel: Channel, q: EventQueue
) -> None:
    """DL-016 — zero authoritative state in RAM. The server keeps no copy of
    the stream, so "what did I miss" is answered by the log and survives a
    ``kill -9`` on either side for free."""
    with ChannelClient(channel.address) as first:
        first.subscribe(0)
        first.say("one")
        drain(q)
        seen = [first.read(), first.read()]
        assert [u["seq"] for u in seen] == [1, 2]

    # Away. Things keep happening.
    q.append(episodes.inbound("while you were out", channel="tray", at=AT))
    drain(q)

    with ChannelClient(channel.address) as second:
        answer = second.subscribe(2)
        assert answer["since"] == 2
        missed = [second.read(), second.read()]
        assert [u["seq"] for u in missed] == [3, 4]
        assert missed[0]["text"] == "while you were out"


def test_subscribing_from_head_receives_only_what_happens_next(
    client: ChannelClient, q: EventQueue
) -> None:
    q.append(episodes.inbound("before", channel="tray", at=AT))
    client.subscribe(q.head())
    client.say("after")
    update = client.read()
    assert update["text"] == "after"


def test_the_same_id_twice_is_one_episode_and_the_same_ack(
    client: ChannelClient, q: EventQueue
) -> None:
    """Tray requirement 6: at-most-once belongs to the core, not the UI. A
    client that lost its connection mid-send retries with the same id and gets
    the original seq back, so it never has to guess whether it already sent."""
    first = client.say("did this land?", id="tray-msg-1")
    second = client.say("did this land?", id="tray-msg-1")

    assert first["duplicate"] is False and second["duplicate"] is True
    assert first["seq"] == second["seq"]
    assert q.head() == 1, "a retry must not become a second message"


def test_a_retry_whose_timestamp_moved_on_still_dedups(
    client: ChannelClient, q: EventQueue
) -> None:
    """The real retry path: ``at`` is stamped on arrival, so a retry a second
    later is *not* byte-identical and the log raises rather than dedups. The
    channel resolves that to the original seq instead of filing a second copy.
    """
    seq = q.append(
        episodes.inbound("sent before the drop", channel="tray", at=AT),
        "tray-msg-7",
    )
    ack = client.say("sent before the drop", id="tray-msg-7")

    assert ack["seq"] == seq
    assert ack["duplicate"] is True
    assert "conflict" not in ack, "same message, so nothing to warn about"
    assert q.head() == seq


def test_reusing_an_id_for_a_different_message_is_named(
    client: ChannelClient, q: EventQueue
) -> None:
    """The first copy stands — rewriting it to match a retry is the one thing
    an append-only log must not do — but the client is told, because a reused
    id is a client bug that would otherwise silently drop a real message."""
    seq = q.append(
        episodes.inbound("the first one", channel="tray", at=AT), "tray-msg-3"
    )
    ack = client.say("a completely different message", id="tray-msg-3")

    assert ack["seq"] == seq and ack["duplicate"] is True
    assert "conflict" in ack
    assert q.head() == seq
    assert q.at(seq).payload["text"] == "the first one"


def test_the_append_callback_fires_once_per_real_message(q: EventQueue) -> None:
    """How the executor thread learns there is work without polling. It must
    not fire for a duplicate: a wake per retry would turn one dropped
    connection into a burst of empty drains."""
    woken: list[int] = []
    ch = Channel(q, port=0, poll=0.005, on_append=woken.append)
    ch.start()
    try:
        with ChannelClient(ch.address) as client:
            client.say("one", id="a")
            client.say("one", id="a")
            client.say("two", id="b")
    finally:
        ch.stop()
    assert woken == [1, 2]


# --- what does not cross ----------------------------------------------------


def test_tool_arguments_and_results_never_reach_a_client(
    client: ChannelClient, q: EventQueue
) -> None:
    """End to end, on the socket itself — the unit-level rule in
    test_projection.py is worth little if the channel sends raw payloads."""
    secret_args = "sk-live-0xdeadbeef"
    secret_result = "the contents of a private file"
    client.subscribe(0)
    seq = q.append(episodes.inbound("read it", channel="tray", at=AT))
    q.append(
        episodes.tool_called(
            for_seq=seq, tool="read_file", args={"token": secret_args}, at=AT
        )
    )
    q.append(
        episodes.tool_returned(
            for_seq=seq, tool="read_file", ok=True, result=secret_result, at=AT
        )
    )

    lines = [client.read() for _ in range(3)]
    blob = json.dumps(lines)
    assert secret_args not in blob
    assert secret_result not in blob
    assert [u["state"] for u in lines] == [UNDERSTOOD, WORKING, WORKING]
    assert [u.get("tool") for u in lines] == [None, "read_file", "read_file"]


def test_context_previews_do_not_reach_a_client(
    client: ChannelClient, q: EventQueue
) -> None:
    client.subscribe(0)
    client.say(
        "look",
        context=[{"id": "ctx-1", "kind": "file", "title": "salary-review.pdf"}],
    )
    update = client.read()
    assert update["context"] == [{"id": "ctx-1", "kind": "file"}]
    assert "salary-review.pdf" not in json.dumps(update)


# --- red: bad input on the one port an untrusted sender can reach -----------


def test_a_malformed_line_is_answered_and_the_connection_survives(
    client: ChannelClient, q: EventQueue
) -> None:
    """The listener is the only surface an untrusted sender can touch, so it has
    to be unexcitable: a bad line is a named error, not a closed channel and
    certainly not an exception on the executor thread."""
    client.send_raw("{not json at all")
    answer = client.read()
    assert answer["op"] == "error" and "JSON" in answer["error"]

    assert client.request({"v": PROTOCOL, "op": "ping"})["op"] == "pong"
    assert q.head() == 0


@pytest.mark.parametrize(
    "line",
    ['[1, 2, 3]', '"a string"', '42', 'null'],
)
def test_a_request_that_is_not_an_object_is_refused(
    client: ChannelClient, line: str
) -> None:
    client.send_raw(line)
    answer = client.read()
    assert answer["op"] == "error"
    assert client.request({"v": PROTOCOL, "op": "ping"})["op"] == "pong"


def test_an_unknown_op_is_named_back(client: ChannelClient) -> None:
    answer = client.request({"v": PROTOCOL, "op": "delete_everything"})
    assert answer["op"] == "error"
    assert "delete_everything" in answer["error"]


@pytest.mark.parametrize("text", ["", "   ", None, 42, {"a": 1}])
def test_saying_nothing_is_refused_rather_than_logged(
    client: ChannelClient, q: EventQueue, text
) -> None:
    """An empty message is a client bug. Appending it would put a turn in the
    queue that cannot mean anything, and the episode log is forever."""
    answer = client.request({"v": PROTOCOL, "op": "say", "text": text})
    assert answer["op"] == "error"
    assert q.head() == 0


def test_a_non_string_id_is_refused(client: ChannelClient, q: EventQueue) -> None:
    answer = client.request(
        {"v": PROTOCOL, "op": "say", "text": "hello", "id": 7}
    )
    assert answer["op"] == "error"
    assert q.head() == 0


@pytest.mark.parametrize("since", [-1, "0", 1.5, True, None])
def test_a_subscribe_cursor_that_is_not_a_sequence_number_is_refused(
    client: ChannelClient, since
) -> None:
    answer = client.request({"v": PROTOCOL, "op": "subscribe", "since": since})
    assert answer["op"] == "error"


def test_a_subscribe_cursor_past_the_end_is_refused(
    client: ChannelClient, q: EventQueue
) -> None:
    """Fail closed. A client whose cursor is ahead of the log has state we do
    not, and serving it an empty stream would hide that until something else
    broke."""
    answer = client.request({"v": PROTOCOL, "op": "subscribe", "since": 5})
    assert answer["op"] == "error" and "5" in answer["error"]


def test_an_oversized_line_is_refused_by_name(channel: Channel, q: EventQueue) -> None:
    """An unterminated line is an unbounded allocation. The limit is enforced
    on read, not after."""
    with ChannelClient(channel.address, timeout=10.0) as client:
        client.send_raw("x" * (MAX_LINE + 10))
        answer = client.read()
        assert answer["op"] == "error"
        assert str(MAX_LINE) in answer["error"]
    assert q.head() == 0


def test_an_invalid_episode_is_refused_without_killing_the_server(
    client: ChannelClient, q: EventQueue
) -> None:
    """The codec is the gate, and the channel lets it be. A context item with
    no id cannot keep identity across a restart, so it never gets stored."""
    answer = client.request(
        {
            "v": PROTOCOL,
            "op": "say",
            "text": "hello",
            "context": [{"kind": "file", "title": "no id here"}],
        }
    )
    assert answer["op"] == "error"
    assert q.head() == 0
    assert client.request({"v": PROTOCOL, "op": "ping"})["op"] == "pong"


# --- yellow: clients and servers going away ---------------------------------


def test_one_client_dying_does_not_disturb_another(
    channel: Channel, q: EventQueue
) -> None:
    """The server outlives every UI that attaches to it. A broken pipe is an
    ordinary event and must never reach the turn loop."""
    doomed = ChannelClient(channel.address)
    survivor = ChannelClient(channel.address)
    try:
        doomed.subscribe(0)
        survivor.subscribe(0)
        survivor.say("before the crash")
        assert doomed.read()["state"] == UNDERSTOOD
        assert survivor.read()["state"] == UNDERSTOOD

        # A client that vanishes rather than hanging up: SO_LINGER 0 makes the
        # close send an RST, which is what a killed tray looks like from here.
        doomed._sock.setsockopt(
            socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)
        )
        doomed.close()
        assert until(lambda: len(channel._conns) < 2), "the dead client was not reaped"

        survivor.say("after the crash")
        assert survivor.read()["text"] == "after the crash"
        assert q.head() == 2
    finally:
        survivor.close()
        doomed.close()


def test_a_client_hanging_up_mid_stream_leaves_the_log_untouched(
    channel: Channel, q: EventQueue
) -> None:
    client = ChannelClient(channel.address)
    client.subscribe(0)
    ack = client.say("it landed")
    client.close()
    drain(q)
    assert q.head() == 2
    assert q.at(ack["seq"]).payload["text"] == "it landed"


def test_stopping_the_channel_is_idempotent_and_closes_clients(
    q: EventQueue,
) -> None:
    ch = Channel(q, port=0, poll=0.005)
    address = ch.start()
    client = ChannelClient(address)
    ch.stop()
    ch.stop()
    with pytest.raises((ChannelError, OSError, ValueError, json.JSONDecodeError)):
        client.request({"v": PROTOCOL, "op": "ping"})
    with pytest.raises(OSError):
        ChannelClient(address, timeout=1.0)


def test_the_address_is_only_real_once_it_is_listening(q: EventQueue) -> None:
    ch = Channel(q, port=0)
    with pytest.raises(ChannelError):
        ch.address
    ch.start()
    try:
        with pytest.raises(ChannelError):
            ch.start()
    finally:
        ch.stop()


def test_the_channel_refuses_to_listen_off_loopback(q: EventQueue) -> None:
    """Not a warning and not a default that can be widened: binding the wrong
    address exposes the assistant's entire input surface to the network."""
    for host in ("0.0.0.0", "", "192.168.1.5", "example.com"):
        with pytest.raises(ValueError):
            Channel(q, host=host, port=0)


def test_a_bad_poll_interval_is_refused(q: EventQueue) -> None:
    for poll in (0, -1):
        with pytest.raises(ValueError):
            Channel(q, port=0, poll=poll)


def test_the_listener_is_bound_to_loopback_in_fact_not_only_in_policy(
    channel: Channel,
) -> None:
    host, port = channel.address
    assert host == "127.0.0.1"
    outward = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    outward.settimeout(0.5)
    try:
        hostname_ip = socket.gethostbyname(socket.gethostname())
    except OSError:  # pragma: no cover - no external interface configured
        pytest.skip("no external address on this machine")
    if hostname_ip.startswith("127."):  # pragma: no cover
        pytest.skip("this machine has no non-loopback address")
    with pytest.raises(OSError):
        outward.connect((hostname_ip, port))
    outward.close()


# --- Q12: the listener and the executor are different threads ---------------


def test_a_message_is_acknowledged_while_a_turn_is_in_flight(
    channel: Channel, q: EventQueue
) -> None:
    """§Q12 — one process, two threads. The whole reason for the split: a slow
    turn must not stop the tray from being acknowledged. This test deadlocks
    rather than fails if the listener ever waits on the executor.
    """
    inside = threading.Event()
    release = threading.Event()

    def judge(role: str, messages: list) -> str:
        if role == provider.JUDGE:
            inside.set()
            assert release.wait(timeout=10), "the turn was never released"
        return "SPEAK" if role == provider.JUDGE else "done"

    fake = provider.FakeProvider({provider.JUDGE: judge, provider.ACT: judge})
    executor = Executor(q, complete=fake.complete)
    executor.recover()

    with ChannelClient(channel.address) as client:
        client.say("the first one")
        turn = threading.Thread(target=executor.drain, daemon=True)
        turn.start()
        assert inside.wait(timeout=10), "the turn never started"

        # The executor is parked inside a model call. The listener is not.
        ack = client.say("and one more while you think")
        assert ack["op"] == "ack" and ack["seq"] == 2
        assert q.head() == 2
        assert q.in_flight() == 1, "the first turn is still the one in flight"

        release.set()
        turn.join(timeout=10)
        assert not turn.is_alive()

    assert q.claimed() == q.done() == q.head()


def test_updates_keep_flowing_while_a_turn_runs(
    channel: Channel, q: EventQueue
) -> None:
    """The pump is its own thread too, so a client watching does not have to
    wait for the turn it is watching to end."""
    inside = threading.Event()
    release = threading.Event()

    def judge(role: str, messages: list) -> str:
        if role == provider.JUDGE:
            inside.set()
            release.wait(timeout=10)
        return "SPEAK" if role == provider.JUDGE else "done"

    fake = provider.FakeProvider({provider.JUDGE: judge, provider.ACT: judge})
    executor = Executor(q, complete=fake.complete)
    executor.recover()

    with ChannelClient(channel.address) as client:
        client.subscribe(0)
        client.say("go")
        assert client.read()["state"] == UNDERSTOOD

        turn = threading.Thread(target=executor.drain, daemon=True)
        turn.start()
        assert inside.wait(timeout=10)

        # Something else lands in the log mid-turn and reaches the client now,
        # not after the turn finishes.
        q.append(episodes.tool_called(for_seq=1, tool="shell", args={}, at=AT))
        working = client.read()
        assert working["state"] == WORKING and working["tool"] == "shell"

        release.set()
        turn.join(timeout=10)


def test_many_clients_all_see_the_same_stream(
    channel: Channel, q: EventQueue
) -> None:
    clients = [ChannelClient(channel.address) for _ in range(4)]
    try:
        for c in clients:
            c.subscribe(0)
        clients[0].say("broadcast me")
        drain(q)
        for c in clients:
            assert [c.read()["state"], c.read()["state"]] == [UNDERSTOOD, COMPLETE]
    finally:
        for c in clients:
            c.close()


def test_the_pump_does_not_resend_what_it_has_already_sent(
    client: ChannelClient, q: EventQueue
) -> None:
    """The cursor advances past *withheld* episodes too. If it only advanced
    past projected ones, an unprojectable kind would be re-read on every tick
    forever and the client would see the next real update again and again."""
    client.subscribe(0)
    client.say("once")
    first = client.read()
    time.sleep(0.1)  # several pump ticks with nothing new
    client.say("twice")
    second = client.read()
    assert first["seq"] == 1 and second["seq"] == 2
