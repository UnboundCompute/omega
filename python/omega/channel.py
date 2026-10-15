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

**Attachment bytes arrive before the message does** (DL-027). ``attach`` streams
one file into the content-addressed store beside the log and answers with its
digest; the tray then names that digest in the ``say`` that follows. It is a
separate op rather than a field on ``say`` so the upload happens at *capture*
time — the cost and any failure surface while the person is still staging,
rather than in the moment they press send.

**Outbound is the projection and nothing else.** A subscriber gets
``projection.Update`` lines from a cursor it names, read out of the log on
demand. The server holds no copy of the stream: a client that was away comes
back with ``since`` and is served from the log (DL-016 — zero authoritative
state in RAM), which is also why a ``kill -9`` on either side costs nothing but
a reconnect.

**A sense relay is a second producer, and it can only report** (DL-072). Once
the core moved off the Mac its senses read paths that are not there, so the
Mac runs a relay that reads and digests and sends ``report``; the core appends
``sense.reported`` and learns from it later, at its own idle moment. A report
is a record, never an event: it wakes no turn, and the relay has no ``say``.
``reported`` lets the relay ask which units already have receipts, so it does
not digest — or run whisper over — something the core already learned from.

**``presence`` answers one question: is the person at the Mac?** (DL-073 §3.)
The Discord adapter asks it before it buzzes the phone with something nobody
asked for. Read-only, folded from the latest presence report, and it fails
toward the phone: no report, a stale one or an unreadable one is ``False``.

**``blob`` hands back one stored blob by digest** (DL-097), base64, so the
Discord adapter can send a document omega made as a file.

**``put`` is ``attach`` for a sender with no path on this machine** (DL-101):
the Discord adapter runs beside the core, not on its disk, so a file or a
screenshot sent in the DM arrives as base64 chunks, each at most
:data:`MAX_PUT_CHUNK` bytes and :data:`MAX_PUT_IN` in all, written at their
offset into one staging file in the blob root. A chunk sent again lands where
it did the first time, so a retry after a lost ack is harmless; the last one
stores the file and answers exactly as ``attach`` does.
"""

from __future__ import annotations

import base64
import json
import re
import socket
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterator, Optional

from omega import delegate, episodes, machine, projection
from omega.derive import Learned, Readings
from omega.blobs import BlobError, BlobStore, is_digest, mime_for
from omega.memory import WriteKeyConflict
from omega.queue import EventQueue

__all__ = [
    "PROTOCOL",
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "MAX_LINE",
    "MAX_BLOB_OUT",
    "MAX_PUT_CHUNK",
    "MAX_PUT_IN",
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

#: The largest stored blob the ``blob`` op hands back (DL-097): a document
#: omega made for them, which the Discord adapter delivers as a file. Far over
#: a handoff document (:data:`omega.handoff.MAX_DOCUMENT_BYTES`) and under
#: what a DM takes.
MAX_BLOB_OUT = 8 << 20

#: One ``put`` chunk's bytes, before base64: its line stays under
#: :data:`MAX_LINE`.
MAX_PUT_CHUNK = 512 << 10
#: The largest file ``put`` takes (DL-101): more than a phone screenshot or a
#: document, and what one Slack post of it reads into memory at most.
MAX_PUT_IN = 25 << 20
#: What an upload is called, which names its staging file.
_PUT_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

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
        blobs: BlobStore,
        *,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        poll: float = 0.02,
        on_append: Optional[Callable[[int], None]] = None,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        desk: Optional[delegate.Desk] = None,
    ) -> None:
        if host not in LOOPBACK:
            raise ValueError(
                f"the channel binds loopback only; {host!r} is not one of "
                f"{sorted(LOOPBACK)}"
            )
        if poll <= 0:
            raise ValueError(f"poll must be positive, got {poll}")
        self._queue = queue
        # Handed in, exactly like the queue: both are collaborators the channel
        # uses and neither is one it owns. A channel that opened its own store
        # would be a second opener of the same directory, and which of the two
        # you were looking at would depend on how the process was assembled.
        self._blobs = blobs
        self._host = host
        self._port = port
        self._poll = poll
        self._on_append = on_append
        self._server: Optional[socket.socket] = None
        self._threads: list[threading.Thread] = []
        self._conns: list[_Conn] = []
        self._conns_lock = threading.Lock()
        self._stopping = threading.Event()
        # The receipt cursor the `reported` query answers from. Its own fold,
        # behind its own lock, for the reason the executor's scheduler keeps its
        # own copy: it is derived from the log, so two folds agree by
        # construction, and sharing the executor's would mean locking a view
        # the drain thread mutates.
        self._learned = Learned()
        self._learned_lock = threading.Lock()
        # The presence fold behind ``presence``, kept apart from the receipts
        # for the same reason: its own view, its own lock, derived from the log.
        self._readings = Readings()
        self._readings_lock = threading.Lock()
        self._now = now
        # DL-104: the Claude Code jobs, folded from the log for the Mac's
        # worker, and the desk its polls mark it present on.
        self._jobs = delegate.Jobs()
        self._jobs_lock = threading.Lock()
        self._desk = desk if desk is not None else delegate.desk()

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

        ``BlobError`` is in the named set with the argument errors for the same
        reason they are: a file the client cannot attach is an ordinary answer
        to an ordinary request, and it must reach that client as a sentence
        rather than as a dropped connection it has to interpret.
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
            if op == "attach":
                return self._attach(request)
            if op == "history":
                return self._history(request)
            if op == "subscribe":
                return self._subscribe(conn, request)
            if op == "report":
                return self._report(request)
            if op == "reported":
                return self._reported(request)
            if op == "presence":
                return self._presence()
            if op == "blob":
                return self._blob(request)
            if op == "put":
                return self._put(request)
            if op == "delegate.poll":
                return self._delegate_poll(request)
            if op in ("delegate.update", "delegate.result"):
                return self._delegate_report(request, op)
            if op == "ping":
                return {"v": PROTOCOL, "op": "pong", "head": self._queue.head()}
            return self._error(f"unknown op {op!r}")
        except (ValueError, TypeError, KeyError, BlobError) as exc:
            return self._error(str(exc), op=op)
        except Exception as exc:  # pragma: no cover - the unexpected still answers
            return self._error(f"{type(exc).__name__}: {exc}", op=op)

    # --- the ops ----------------------------------------------------------

    def _say(self, request: dict[str, Any]) -> dict[str, Any]:
        """Append one inbound episode. The returned seq is the acknowledgement.

        ``id`` is optional and becomes the write key. With it, a retry after a
        dropped connection is free: the log returns the original seq and the
        answer says ``duplicate``, so the client learns that its first attempt
        landed rather than guessing. Without it, a retry is a second message —
        which is the honest outcome, because nothing identified it as the same
        one.
        """
        text = request.get("text", "")
        if not isinstance(text, str):
            raise ValueError("'text' must be a string")
        context = request.get("context") or []
        if not isinstance(context, list):
            raise ValueError("'context' must be a list")
        if not text.strip() and not context:
            # Empty text is only empty when nothing came with it. Dropping a
            # screenshot in and saying nothing is a real message — the thing
            # shown *is* the message — and the tray already treats it as one
            # (``canSend`` is text-or-context). The episode schema agrees:
            # ``message.inbound`` requires ``text`` to be a string but not a
            # non-empty one, where it demands ``channel`` be non-empty right
            # beside it. This check was the only place that disagreed, and it
            # made a submission the UI accepts fail at the socket.
            raise ValueError("say needs 'text', 'context', or both")
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

    def _report(self, request: dict[str, Any]) -> dict[str, Any]:
        """Append one ``sense.reported`` record. The seq is the acknowledgement.

        ``{source, unit, device, body}`` plus, for a learning sense, ``meta`` —
        the receipt's identity fields the core cannot see without the file —
        and ``reason`` when the Mac could not digest the unit. Validated by the
        episode constructor, so the channel cannot accept a report the log
        would refuse or the core could never turn into a receipt.

        **The write key is the unit, so a resend is the duplicate path** — the
        same one ``say`` takes, answered the same way. A relay that lost an ack
        sends again and is told its first copy landed, which is what lets it
        file a recording away only once the core holds it.

        **Never wakes the drain.** ``say`` calls ``on_append`` because an inbound
        is a turn waiting; a report is not, and the core works it at the next
        idle sense pass. Leaving the hook out here is what keeps "a report never
        wakes a turn" true by construction rather than by the drain happening to
        skip a record kind.
        """
        source = request.get("source")
        unit = request.get("unit")
        if not isinstance(source, str) or not isinstance(unit, str):
            raise ValueError("'source' and 'unit' must be strings")
        payload = episodes.sense_reported(
            source=source,
            unit=unit,
            device=request.get("device"),  # type: ignore[arg-type]
            body=request.get("body"),
            meta=request.get("meta"),
            reason=request.get("reason"),
            # Stamped on arrival by the core's clock, never taken from the
            # sender: staleness is judged against this, and a Mac whose clock
            # runs ahead must not be able to keep its own reading fresh. The
            # relay reads at send time and holds nothing back, so arrival is
            # the reading's time to within a round trip.
        )
        write_key = episodes.report_write_key(source, unit)
        head_before = self._queue.head()
        try:
            seq = self._queue.append(payload, write_key)
        except WriteKeyConflict:
            # The ordinary resend: `at` differs, so the bytes do, and the first
            # copy stands. Named as a conflict only when what was reported
            # differs too — a unit whose digest changed between sends.
            existing = self._existing(write_key)
            answer = {
                "v": PROTOCOL,
                "op": "ack",
                "seq": existing.seq,
                "duplicate": True,
            }
            if existing.payload.get("body") != payload["body"]:
                answer["conflict"] = (
                    "that unit is already in the log carrying a different report"
                )
            return answer
        return {
            "v": PROTOCOL,
            "op": "ack",
            "seq": seq,
            "duplicate": seq <= head_before,
        }

    def _reported(self, request: dict[str, Any]) -> dict[str, Any]:
        """The units of one learning sense the core holds a receipt for.

        Receipts, not reports: a unit the Mac sent but the core has not worked
        yet is still owed, and saying otherwise would let a relay file away a
        recording the core never learned from. A machine or presence reading
        has no receipt, so asking about one is an error rather than an empty
        answer that looks like "nothing yet".
        """
        source = request.get("source")
        if source not in episodes.LEARNING_SOURCES:
            raise ValueError(
                f"'source' must be one of {sorted(episodes.LEARNING_SOURCES)}"
            )
        with self._learned_lock:
            self._learned.advance(self._queue.store)
            units = sorted(self._learned.worked(source))
        return {"v": PROTOCOL, "op": "reported", "source": source, "units": units}

    def _presence(self) -> dict[str, Any]:
        """``{at_the_mac}`` from the Mac's latest presence report (DL-073 §3).

        The whole decision is :func:`omega.machine.at_the_mac`, against the
        core's clock; this only folds the log up to now and asks it. Nothing
        else crosses — not the idle seconds, not the report's age — because the
        one client that asks needs one bit, and a wider answer is a wider
        surface for no reader.
        """
        with self._readings_lock:
            self._readings.advance(self._queue.store)
            latest = self._readings.latest(episodes.REPORT_PRESENCE, machine.MAC)
        return {
            "v": PROTOCOL,
            "op": "presence",
            "at_the_mac": machine.at_the_mac(latest, now=self._now()),
        }

    def _attach(self, request: dict[str, Any]) -> dict[str, Any]:
        """Ingest one file into the blob store and answer with its reference.

        The answer — ``{blob, mime, bytes}`` — is exactly the three fields the
        client then puts on a context item in its ``say``. One shape, produced
        here and validated there, so a tray cannot assemble a reference omega
        would reject.

        **This reads a path the client names, and that is not an escalation.**
        The channel is loopback-only and enforced so (see ``LOOPBACK``), which
        means the sender is already a local process running as the user — and
        any such process can read what the user can read, directly, without
        asking omega. The op adds no reach; it adds a *copy* under a name that
        cannot be tampered with afterwards. Writing this down because it is the
        kind of line that gets rediscovered as a finding by whoever reads the op
        list next, and the reasoning is the same every time.

        There is no size cap (DL-027 defers it). What bounds this is the store's
        refusal of anything that is not a regular file: a size cap keeps a large
        file out, but only that check keeps an *endless* one out.
        """
        path = request.get("path")
        if not isinstance(path, str):
            raise ValueError("'path' must be a string")
        if not path.strip():
            raise ValueError("'path' must not be empty")

        ref = self._blobs.put(path)
        return {
            "v": PROTOCOL,
            "op": "attached",
            "blob": ref.digest,
            "mime": ref.mime,
            "bytes": ref.bytes,
        }

    def _blob(self, request: dict[str, Any]) -> dict[str, Any]:
        """One stored blob's bytes, base64 (DL-097): the document a reply's
        marker names, for a channel that delivers it as a file.

        The reverse of ``attach``, and no more of an escalation for the same
        reason: the sender is a local process running as the user, which can
        read the store directly. It names a blob by its digest only, so it
        reaches nothing outside the store.
        """
        digest = request.get("blob")
        if not is_digest(digest):
            raise ValueError("'blob' must be a blob reference (sha256:<hex>)")
        if not self._blobs.has(digest):
            raise ValueError(f"no blob {digest}")
        path = self._blobs.path_for(digest)
        if path.stat().st_size > MAX_BLOB_OUT:
            raise ValueError(f"blob {digest} is over {MAX_BLOB_OUT} bytes")
        data = path.read_bytes()
        return {
            "v": PROTOCOL,
            "op": "blob",
            "blob": digest,
            "bytes": len(data),
            "data": base64.b64encode(data).decode("ascii"),
        }

    def _put(self, request: dict[str, Any]) -> dict[str, Any]:
        """One chunk of a file sent as bytes (DL-101); the last one stores it.

        No more reach than ``attach``: a local process could write the store
        itself. It writes one staging file, named by the upload's id inside
        the blob root, and nothing outside it.
        """
        upload = request.get("upload")
        if not isinstance(upload, str) or not _PUT_ID.match(upload):
            raise ValueError("'upload' must be 1-64 letters, digits, '-' or '_'")
        name = request.get("name")
        if not isinstance(name, str) or not name.strip() or "/" in name or "\\" in name:
            raise ValueError("'name' must be a file name")
        offset = request.get("offset")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("'offset' must be a whole number of bytes")
        data = request.get("data")
        if not isinstance(data, str):
            raise ValueError("'data' must be base64")
        try:
            chunk = base64.b64decode(data, validate=True)
        except ValueError:
            raise ValueError("'data' must be base64") from None
        if len(chunk) > MAX_PUT_CHUNK:
            raise ValueError(f"a chunk is at most {MAX_PUT_CHUNK} bytes")
        if offset + len(chunk) > MAX_PUT_IN:
            raise ValueError(f"a file is at most {MAX_PUT_IN >> 20} MB")
        staged = self._blobs.root / f".ingest-put-{upload}"
        size = staged.stat().st_size if staged.exists() else 0
        if offset > size:
            raise ValueError(f"chunk at {offset} but {size} bytes arrived so far")
        with open(staged, "r+b" if staged.exists() else "wb") as out:
            out.seek(offset)
            out.write(chunk)
            out.truncate()
        if not request.get("last"):
            return {"v": PROTOCOL, "op": "put", "upload": upload, "bytes": offset + len(chunk)}
        try:
            ref = self._blobs.put(staged)
        finally:
            staged.unlink(missing_ok=True)
        return {
            "v": PROTOCOL,
            "op": "attached",
            "blob": ref.digest,
            "mime": mime_for(name),
            "bytes": ref.bytes,
        }

    def _subscribe(self, conn: _Conn, request: dict[str, Any]) -> dict[str, Any]:
        since = request.get("since", 0)
        if not isinstance(since, int) or isinstance(since, bool) or since < 0:
            raise ValueError("'since' must be a sequence number >= 0")
        head = self._queue.head()
        if since > head:
            raise ValueError(f"'since' {since} is ahead of head {head}")
        conn.cursor = since
        return {"v": PROTOCOL, "op": "subscribed", "since": since, "head": head}

    def _history(self, request: dict[str, Any]) -> dict[str, Any]:
        """One backward page of the outward projection, without moving a cursor."""
        head = self._queue.head()
        before = request.get("before", head + 1)
        limit = request.get("limit", 50)
        if not isinstance(before, int) or isinstance(before, bool):
            raise ValueError("'before' must be a sequence number")
        if before < 1 or before > head + 1:
            raise ValueError(f"'before' must be between 1 and {head + 1}")
        if (
            not isinstance(limit, int)
            or isinstance(limit, bool)
            or not 1 <= limit <= 100
        ):
            raise ValueError("'limit' must be between 1 and 100")

        pending = self._queue.recent(limit, before=before)
        updates = []
        for item in pending:
            update = projection.project_pending(item)
            if update is not None:
                updates.append(update.wire())
        next_before = pending[0].seq if pending else 1
        return {
            "v": PROTOCOL,
            "op": "history",
            "before": next_before,
            "more": next_before > 1,
            "updates": updates,
        }

    def _delegate_poll(self, request: dict[str, Any]) -> dict[str, Any]:
        """The Mac's worker asking what to do (DL-104): ``{folders}`` is what
        its allowlist holds. Answers every command it has not yet answered.

        Marks the worker present, in memory: that, and only that, is what puts
        ``delegate`` in a turn. Nothing is written, so a worker polling every
        few seconds adds nothing to the log.
        """
        folders = request.get("folders")
        if (
            not isinstance(folders, list) or not folders or len(folders) > 50
            or not all(isinstance(f, str) and f.strip() for f in folders)
        ):
            raise ValueError("'folders' must be the worker's allowed folders")
        self._desk.seen(self._now(), folders)
        with self._jobs_lock:
            self._jobs.advance(self._queue.store)
            commands = self._jobs.pending()
        return {"v": PROTOCOL, "op": "delegate.poll", "commands": commands}

    def _delegate_report(self, request: dict[str, Any], op: str) -> dict[str, Any]:
        """The Mac's worker reporting on one job (DL-104): what its session
        did (``delegate.update``) or what it produced (``delegate.result``).

        ``id`` is the write key, so a resend is the duplicate path, as for
        ``report``. The record lands first and its tell after it, under a key
        of its own: a resend after a crash between the two writes the tell it
        missed, and never a second one.
        """
        rid = request.get("id")
        if not isinstance(rid, str) or not _PUT_ID.match(rid):
            raise ValueError("'id' must be 1 to 64 letters, digits, '-' or '_'")
        name = request.get("job")
        with self._jobs_lock:
            self._jobs.advance(self._queue.store)
            if self._jobs.by_id.get(name) is None:
                raise ValueError(f"there is no job {name!r}")
        session = request.get("session") or None
        if op == "delegate.update":
            payload = episodes.delegate_updated(
                job=name, state=request.get("state"),  # type: ignore[arg-type]
                session=session, detail=request.get("detail") or None,
                cmd=request.get("cmd"), at=self._now().isoformat(),
            )
        else:
            files = request.get("files") or []
            if not isinstance(files, list):
                raise ValueError("'files' must be a list")
            for item in files:
                if not isinstance(item, dict) or not self._blobs.has(str(item.get("blob"))):
                    raise ValueError("every file must be one this channel was sent with put")
            payload = episodes.delegate_finished(
                job=name, text=request.get("text"),  # type: ignore[arg-type]
                files=files, session=session, at=self._now().isoformat(),
            )
        write_key = f"delegate:{rid}"
        head_before = self._queue.head()
        try:
            seq = self._queue.append(payload, write_key)
            duplicate = seq <= head_before
        except WriteKeyConflict:
            existing = self._existing(write_key)
            seq, duplicate, payload = existing.seq, True, dict(existing.payload)
        with self._jobs_lock:
            self._jobs.advance(self._queue.store)
            job = self._jobs.by_id[name]
        told = delegate.tell(job, payload, at=str(payload["at"]))
        if told is not None:
            head_before = self._queue.head()
            try:
                tseq = self._queue.append(told, f"delegate-tell:{rid}")
            except WriteKeyConflict:
                pass
            else:
                if tseq > head_before and self._on_append is not None:
                    self._on_append(tseq)
        return {"v": PROTOCOL, "op": "ack", "seq": seq, "duplicate": duplicate}

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

    def attach(self, path: Any) -> dict[str, Any]:
        return self.request({"v": PROTOCOL, "op": "attach", "path": path})

    def history(self, **kwargs: Any) -> dict[str, Any]:
        return self.request({"v": PROTOCOL, "op": "history", **kwargs})

    def subscribe(self, since: int = 0) -> dict[str, Any]:
        return self.request({"v": PROTOCOL, "op": "subscribe", "since": since})

    def report(self, source: str, unit: str, *, device: str, body: Any, **kwargs: Any) -> dict[str, Any]:
        return self.request(
            {
                "v": PROTOCOL,
                "op": "report",
                "source": source,
                "unit": unit,
                "device": device,
                "body": body,
                **kwargs,
            }
        )

    def reported(self, source: str) -> dict[str, Any]:
        return self.request({"v": PROTOCOL, "op": "reported", "source": source})

    def presence(self) -> dict[str, Any]:
        return self.request({"v": PROTOCOL, "op": "presence"})

    def blob(self, digest: str) -> dict[str, Any]:
        return self.request({"v": PROTOCOL, "op": "blob", "blob": digest})

    def put(self, upload: str, name: str, data: bytes) -> dict[str, Any]:
        """Send one file as bytes, in chunks (DL-101): the ``attached`` answer."""
        offset = 0
        while True:
            chunk = data[offset: offset + MAX_PUT_CHUNK]
            last = offset + len(chunk) >= len(data)
            answer = self.request({
                "v": PROTOCOL, "op": "put", "upload": upload, "name": name,
                "offset": offset, "data": base64.b64encode(chunk).decode("ascii"),
                "last": last,
            })
            if last or answer.get("op") != "put":
                return answer
            offset += len(chunk)

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
