"""The duplex localhost channel — M1 step 5, spec §Q10, §1.1, §1.6.

JSON lines over a loopback TCP socket, one object per line, both directions on
one connection. Network-shaped from the first commit (DL-016) so that moving the
client off the machine later is a config change rather than a rewrite, and
loopback-only because nothing about omega should be reachable from off the box.

**Inbound is one op and it is an append.** ``say`` appends a ``message.inbound``
episode and answers with its ``seq``, which *is* the acknowledgement token
(§1.1, tray requirement 1). There is no second way to cause a turn: the channel
is a producer against the same queue M5's clock will use (§1.6), so a wake
arriving from a person and a wake arriving from a timer enter at the same point
and cannot drift apart.

**At-most-once belongs here, not to the UI** (tray requirement 6). A client that
loses its connection mid-send and retries with the same ``id`` gets the *same*
seq back, because the id becomes the episode's write key and M0's dedup refuses
the second copy. The client does not have to remember whether it already sent
something; the log already knows.

**Outbound is the projection and nothing else.** A subscriber gets
``projection.Update`` lines from a cursor it names, read out of the log on
demand. The server holds no copy of the stream: a client that was away comes
back with ``since`` and is served from the log (DL-016 — zero authoritative
state in RAM), which is also why a ``kill -9`` on either side costs nothing but
a reconnect.
"""

from __future__ import annotations

import json
import socket
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional

from omega import episodes, projection
from omega.memory import WriteKeyConflict
from omega.queue import EventQueue

__all__ = [
    "PROTOCOL",
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "MAX_LINE",
    "LOOPBACK",
    "ChannelError",
    "Channel",
    "ChannelClient",
]

#: The wire protocol version, announced in every server line.
PROTOCOL = 1

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7717

#: A single line is a single message, so an unterminated line is an unbounded
#: allocation. One mebibyte is far more than a message plus its context ids and
#: far less than a problem; a longer one is refused by name rather than read.
MAX_LINE = 1 << 20

#: The only addresses the server will bind. Not a default that can be widened by
#: passing a different host — a wrong value here exposes the assistant's entire
#: input surface to the network, so it is a refusal, not a warning.
LOOPBACK = frozenset({"127.0.0.1", "::1", "localhost"})


class ChannelError(RuntimeError):
    """The channel could not be used as asked."""


@dataclass
class _Conn:
    """One client. ``cursor`` is ``None`` until it subscribes — an unsubscribed
    connection can still ``say``, because sending and watching are independent
    halves of the duplex and a client that only speaks should not be forced to
    listen."""

    sock: socket.socket
    reader: Any
    writer: Any
    lock: threading.Lock = field(default_factory=threading.Lock)
    cursor: Optional[int] = None
    alive: bool = True

    def send(self, obj: dict[str, Any]) -> bool:
        """Write one line. Returns False once the client is gone.

        A dead client is an ordinary event, not an error: the server outlives
        every UI that attaches to it, and a broken pipe must never reach the
        turn loop.
        """
        line = json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
        with self.lock:
            if not self.alive:
                return False
            try:
                self.writer.write(line + "\n")
                self.writer.flush()
                return True
            except (OSError, ValueError):
                self.alive = False
                return False

    def close(self) -> None:
        """Hang up, from any thread.

        ``shutdown`` before ``close``, and the *socket* rather than the file
        objects, because this is normally called by a different thread than the
        one reading. Closing a buffered reader that another thread is parked
        inside waits for that thread's lock, which it will never release — it is
        parked on the read. Shutting the socket down wakes the read instead, and
        that thread then closes its own files in ``_read_loop``.
        """
        with self.lock:
            self.alive = False
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self.sock.close()
        except OSError:
            pass

    def dispose(self) -> None:
        """Close the file objects. Only the reading thread may call this."""
        self.close()
        for closeable in (self.reader, self.writer):
            try:
                closeable.close()
            except (OSError, ValueError):
                pass


class Channel:
    """The listener half of the resident process (§Q12).

    Two threads: one accepting and reading connections, one pumping the
    projection outwards. Neither of them runs a turn — the executor does that,
    on its own thread, by draining the same queue. The split is the whole point:
    a slow turn must not stop the tray from being acknowledged, and a chatty
    client must not stop a turn from finishing.
    """

    def __init__(
        self,
        queue: EventQueue,
        *,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        poll: float = 0.02,
        on_append: Optional[Callable[[int], None]] = None,
    ) -> None:
        if host not in LOOPBACK:
            raise ValueError(
                f"the channel binds loopback only; {host!r} is not one of "
                f"{sorted(LOOPBACK)}"
            )
        if poll <= 0:
            raise ValueError(f"poll must be positive, got {poll}")
        self._queue = queue
        self._host = host
        self._port = port
        self._poll = poll
        self._on_append = on_append
        self._server: Optional[socket.socket] = None
        self._threads: list[threading.Thread] = []
        self._conns: list[_Conn] = []
        self._conns_lock = threading.Lock()
        self._stopping = threading.Event()

    # --- lifecycle --------------------------------------------------------

    @property
    def address(self) -> tuple[str, int]:
        if self._server is None:
            raise ChannelError("the channel is not listening")
        host, port = self._server.getsockname()[:2]
        return str(host), int(port)

    def start(self) -> tuple[str, int]:
        if self._server is not None:
            raise ChannelError("the channel is already listening")
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self._host, self._port))
        server.listen(8)
        server.settimeout(self._poll)
        self._server = server
        self._spawn(self._accept_loop, "omega-channel-accept")
        self._spawn(self._pump_loop, "omega-channel-pump")
        return self.address

    def stop(self) -> None:
        """Idempotent, and safe to call from any thread."""
        self._stopping.set()
        if self._server is not None:
            try:
                self._server.close()
            except OSError:
                pass
        with self._conns_lock:
            conns, self._conns = list(self._conns), []
        for conn in conns:
            conn.close()
        for thread in self._threads:
            if thread is not threading.current_thread():
                thread.join(timeout=5)
        self._threads = []
        self._server = None

    def __enter__(self) -> "Channel":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    def _spawn(self, target: Callable[[], None], name: str) -> None:
        thread = threading.Thread(target=target, name=name, daemon=True)
        thread.start()
        self._threads.append(thread)

    # --- accepting and reading -------------------------------------------

    def _accept_loop(self) -> None:
        while not self._stopping.is_set():
            server = self._server
            if server is None:
                return
            try:
                sock, _ = server.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            conn = _Conn(
                sock=sock,
                reader=sock.makefile("r", encoding="utf-8", newline="\n"),
                writer=sock.makefile("w", encoding="utf-8", newline="\n"),
            )
            with self._conns_lock:
                self._conns.append(conn)
            conn.send({"v": PROTOCOL, "op": "hello", "head": self._queue.head()})
            self._spawn(
                lambda c=conn: self._read_loop(c), "omega-channel-conn"
            )

    def _read_loop(self, conn: _Conn) -> None:
        try:
            while not self._stopping.is_set():
                try:
                    line = conn.reader.readline(MAX_LINE + 1)
                except (OSError, ValueError):
                    return
                if not line:
                    return  # the client hung up; ordinary
                if len(line) > MAX_LINE:
                    conn.send(
                        {
                            "v": PROTOCOL,
                            "op": "error",
                            "error": f"line longer than {MAX_LINE} bytes",
                        }
                    )
                    return
                line = line.strip()
                if not line:
                    continue
                conn.send(self._handle(conn, line))
        finally:
            conn.alive = False
            with self._conns_lock:
                if conn in self._conns:
                    self._conns.remove(conn)
            conn.dispose()

    def _handle(self, conn: _Conn, line: str) -> dict[str, Any]:
        """One request to one response. **Never raises.**

        A malformed line is a client bug and gets a named error back on the same
        connection; it does not close the channel and it never reaches the
        executor. The listener is the only part of omega an untrusted sender can
        talk to, so it is the part that has to be unexcitable.
        """
        try:
            request = json.loads(line)
        except ValueError as exc:
            return self._error(f"not JSON: {exc}")
        if not isinstance(request, dict):
            return self._error("a request must be a JSON object")

        op = request.get("op")
        try:
            if op == "say":
                return self._say(request)
            if op == "subscribe":
                return self._subscribe(conn, request)
            if op == "ping":
                return {"v": PROTOCOL, "op": "pong", "head": self._queue.head()}
            return self._error(f"unknown op {op!r}")
        except (ValueError, TypeError, KeyError) as exc:
            return self._error(str(exc), op=op)
        except Exception as exc:  # pragma: no cover - the unexpected still answers
            return self._error(f"{type(exc).__name__}: {exc}", op=op)

    # --- the two ops ------------------------------------------------------

    def _say(self, request: dict[str, Any]) -> dict[str, Any]:
        """Append one inbound episode. The returned seq is the acknowledgement.

        ``id`` is optional and becomes the write key. With it, a retry after a
        dropped connection is free: the log returns the original seq and the
        answer says ``duplicate``, so the client learns that its first attempt
        landed rather than guessing. Without it, a retry is a second message —
        which is the honest outcome, because nothing identified it as the same
        one.
        """
        text = request.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("say needs a non-empty 'text'")
        write_key = request.get("id", "")
        if not isinstance(write_key, str):
            raise ValueError("'id' must be a string")

        payload = episodes.inbound(
            text,
            channel=str(request.get("channel", "tray")),
            context=list(request.get("context") or []),
            urgency=str(request.get("urgency", "normal")),
            resumes_seq=request.get("resumes_seq"),
        )
        head_before = self._queue.head()
        try:
            seq = self._queue.append(payload, write_key)
        except WriteKeyConflict:
            # The id is in the log under a payload that is not byte-identical —
            # which a plain retry is, because ``at`` is stamped on arrival and a
            # retry arrives later. So this is the *normal* retry path, not an
            # error path, and it is where the scan is paid for rather than on
            # every message.
            #
            # The first copy stands either way: the log records what arrived,
            # and rewriting it to match a retry is the one thing an append-only
            # log must not do. What differs is what we tell the client — if the
            # *text* also differs, the id was genuinely reused for a different
            # message and that is a client bug worth naming.
            existing = self._existing(write_key)
            answer = {
                "v": PROTOCOL,
                "op": "ack",
                "seq": existing.seq,
                "duplicate": True,
            }
            if existing.payload.get("text") != text:
                answer["conflict"] = (
                    "that id is already in the log carrying a different message"
                )
            return answer

        duplicate = seq <= head_before
        if not duplicate and self._on_append is not None:
            self._on_append(seq)
        return {"v": PROTOCOL, "op": "ack", "seq": seq, "duplicate": duplicate}

    def _subscribe(self, conn: _Conn, request: dict[str, Any]) -> dict[str, Any]:
        since = request.get("since", 0)
        if not isinstance(since, int) or isinstance(since, bool) or since < 0:
            raise ValueError("'since' must be a sequence number >= 0")
        head = self._queue.head()
        if since > head:
            raise ValueError(f"'since' {since} is ahead of head {head}")
        conn.cursor = since
        return {"v": PROTOCOL, "op": "subscribed", "since": since, "head": head}

    def _existing(self, write_key: str) -> Any:
        """The episode already filed under this write key.

        A backwards scan of the log. M0 exposes no index on write keys, and
        adding one for a path that only runs on a retry would be speculative —
        so this is honest about being linear rather than hiding it behind a
        cache that would then need invalidating.
        """
        for pending in reversed(self._queue.recent(self._queue.head())):
            if pending.write_key == write_key:
                return pending
        raise ChannelError(
            f"the log rejected write key {write_key!r} but does not hold it"
        )

    # --- pumping the projection outwards ----------------------------------

    def _pump_loop(self) -> None:
        """Poll the log and fan new updates out to every subscriber.

        Polling rather than being notified, deliberately: the source of truth is
        the log, so a subscriber catching up after a reconnect and a subscriber
        watching live take **the same path** through the same reader. A
        notification fast-path would be a second way for an update to reach a
        client, and the two would drift.
        """
        while not self._stopping.wait(self._poll):
            try:
                self._pump_once()
            except Exception:  # pragma: no cover - the pump outlives its clients
                continue

    def _pump_once(self) -> None:
        with self._conns_lock:
            conns = [c for c in self._conns if c.alive and c.cursor is not None]
        if not conns:
            return
        head = self._queue.head()
        for conn in conns:
            cursor = conn.cursor
            if cursor is None or cursor >= head:
                continue
            for update in projection.updates_since(self._queue, cursor):
                if update.seq > head:
                    break
                if not conn.send(update.wire()):
                    break
            # Advance past filtered-out episodes too, or a withheld kind would
            # be re-read on every tick forever.
            conn.cursor = head

    @staticmethod
    def _error(message: str, op: Any = None) -> dict[str, Any]:
        out = {"v": PROTOCOL, "op": "error", "error": message}
        if op is not None:
            out["request"] = op
        return out

    def __repr__(self) -> str:
        where = f"{self._host}:{self._port}" if self._server is None else self.address
        return f"<Channel {where} conns={len(self._conns)}>"


class ChannelClient:
    """A small blocking client. Used by the tests and by anything headless.

    It is deliberately not clever: no reconnect, no backoff, no queueing. Those
    belong to a real UI, which has to make them visible to a person; a helper
    that hid them would make the tests pass against behaviour the tray cannot
    rely on.
    """

    def __init__(self, address: tuple[str, int], *, timeout: float = 5.0) -> None:
        self._sock = socket.create_connection(address, timeout=timeout)
        self._sock.settimeout(timeout)
        self._reader = self._sock.makefile("r", encoding="utf-8", newline="\n")
        self._writer = self._sock.makefile("w", encoding="utf-8", newline="\n")
        #: Updates that arrived while a reply was being waited for. On one
        #: duplex line a push can land between a request and its answer, so a
        #: client that assumed the next line was its reply would silently read
        #: an update as one. Buffering is the smallest correct handling, and it
        #: keeps update order intact.
        self._pushed: deque[dict[str, Any]] = deque()
        self.hello = self.read()

    def send(self, obj: dict[str, Any]) -> None:
        self._writer.write(json.dumps(obj) + "\n")
        self._writer.flush()

    def send_raw(self, line: str) -> None:
        """For the malformed-input cases. A client that can only send valid
        JSON cannot test what happens when one does not."""
        self._writer.write(line + "\n")
        self._writer.flush()

    def read(self) -> dict[str, Any]:
        """The next line, buffered pushes first."""
        if self._pushed:
            return self._pushed.popleft()
        return self._read_line()

    def _read_line(self) -> dict[str, Any]:
        line = self._reader.readline()
        if not line:
            raise ChannelError("the channel closed the connection")
        return json.loads(line)

    def request(self, obj: dict[str, Any]) -> dict[str, Any]:
        """Send one op and read its reply, setting aside any pushes first.

        Both directions share one line, so an update can arrive between the
        request and its answer. That is duplex working correctly, not a race:
        the reply is the first line that is not a push.
        """
        self.send(obj)
        while True:
            message = self._read_line()
            if message.get("op") == "update":
                self._pushed.append(message)
                continue
            return message

    def say(self, text: str, **kwargs: Any) -> dict[str, Any]:
        return self.request({"v": PROTOCOL, "op": "say", "text": text, **kwargs})

    def subscribe(self, since: int = 0) -> dict[str, Any]:
        return self.request({"v": PROTOCOL, "op": "subscribe", "since": since})

    def updates(self, count: int) -> Iterator[dict[str, Any]]:
        for _ in range(count):
            yield self.read()

    def close(self) -> None:
        for closeable in (self._reader, self._writer):
            try:
                closeable.close()
            except OSError:
                pass
        try:
            self._sock.close()
        except OSError:
            pass

    def __enter__(self) -> "ChannelClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
