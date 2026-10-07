"""The sense relay — the Mac reads, the core learns. DL-072, DL-073 §3.

Once the core moved to the VM (DL-069) its senses kept running and found
nothing: the transcripts, the usage database and the recordings folder are on
the Mac, and the VM has none of them. This is the half that has to stay where
the material is. It discovers, digests (and runs whisper for a recording), and
sends the digest as a ``report``; the core does the reflection, the filing and
the receipt, through the same code the local senses use.

**No store and no key.** The relay holds no log, no memory and no model
credential. Everything it knows about what has already been learned it asks the
core (``reported``), and everything it learns it hands to the core. A relay
that crashes loses at most the pass it was in, and the next pass asks again.

**It can only report.** The only ops this module will send are ``report`` and
``reported`` — checked here, in :meth:`Relay._send`, not left to the server to
refuse. A sense that could ``say`` would be a second, unattended way to start a
conversation with omega, and nothing about reading a disk needs one.

**A recording leaves the folder only after the core holds it.** Whisper takes
minutes, and a recording filed away before its report was acknowledged would be
a meeting lost to a dropped connection. The ack is the receipt of arrival; the
resend after a lost ack lands on the core's duplicate path, not a second copy.

**Never raises out of a pass.** A connection that is down, a file that will not
read, a whisper that fails: each is logged and the next pass tries again. The
relay is a background process nobody is watching.
"""

from __future__ import annotations

import hashlib
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from omega import blobs, episodes, habits, listen, machine, transcripts
from omega.channel import ChannelClient, ChannelError

__all__ = [
    "PASS_SECONDS",
    "DEVICE",
    "ALLOWED_OPS",
    "Relay",
]

#: How often a pass runs. The core senses once a minute (``SENSE_SECONDS``), so
#: reporting faster would only queue work it cannot take yet.
PASS_SECONDS = 60

#: How this machine names itself in a report. One word the look can print, and
#: the name DL-073 routes on; a hostname would put the person's machine name in
#: the log for no gain.
DEVICE = "mac"

#: The only ops a relay sends. See the module docstring.
ALLOWED_OPS = frozenset({"report", "reported"})


def _sha256(path: Path) -> tuple[str, int]:
    """The recording's digest as the blob store would name it, and its size.

    Computed here because the audio never leaves the Mac: the core keys the
    recording on this digest (``audio.captured.recording``), so it has to be
    the same function :class:`omega.blobs.BlobStore` would have applied.
    """
    hasher = hashlib.sha256()
    size = 0
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(blobs.CHUNK)
            if not chunk:
                break
            hasher.update(chunk)
            size += len(chunk)
    return blobs.DIGEST_PREFIX + hasher.hexdigest(), size


class Relay:
    """One relay, pointed at one core. Each :meth:`pass_once` is self-contained.

    Every outside effect is injectable — the connection, the clock, the
    readings, whisper — so a test drives a pass without a socket, pmset, ioreg
    or a model on disk.
    """

    def __init__(
        self,
        address: tuple[str, int],
        *,
        device: str = DEVICE,
        transcripts_root: Optional[Path] = None,
        usage_path: Optional[Path] = None,
        recordings: Optional[str] = None,
        read_machine: Optional[Callable[[], machine.Reading]] = None,
        read_idle: Optional[Callable[[], Optional[float]]] = None,
        transcribe: Callable[[Path], listen.Heard] = listen.transcribe,
        not_ready: Callable[[], Optional[str]] = listen.not_ready,
        connect: Callable[[tuple[str, int]], Any] = lambda a: ChannelClient(a, timeout=30.0),
        clock: Callable[[], float] = time.time,
        log: Callable[[str], None] = lambda line: None,
    ) -> None:
        self._address = address
        self._device = device
        self._transcripts = transcripts_root
        self._usage = usage_path
        self._recordings = recordings
        self._read_machine = read_machine
        self._read_idle = read_idle
        self._transcribe = transcribe
        self._not_ready = not_ready
        self._connect = connect
        self._clock = clock
        self._note = log
        # What was last *acknowledged*, never what was last read: a reading
        # that did not arrive must not suppress the next one as "unchanged".
        self._machine_sent: Optional[machine.Reading] = None
        self._machine_sent_at = 0.0
        self._away_sent: Optional[bool] = None
        self._presence_sent_at = 0.0

    # --- the wire ---------------------------------------------------------

    def _send(self, client: Any, op: str, **fields: Any) -> dict[str, Any]:
        """The one door out. Refuses anything that is not a report or a query."""
        if op not in ALLOWED_OPS:
            raise ChannelError(f"a relay does not send {op!r}; it only reports")
        return client.request({"v": 1, "op": op, **fields})

    def _report(
        self, client: Any, source: str, unit: str, body: Any, **fields: Any
    ) -> bool:
        """Send one report. True only on an ack — a duplicate ack included."""
        answer = self._send(
            client, "report", source=source, unit=unit, device=self._device,
            body=body, **fields,
        )
        if answer.get("op") != "ack":
            self._note(f"relay: {source} {unit} refused: {answer.get('error')}")
            return False
        if answer.get("conflict"):
            self._note(f"relay: {source} {unit}: {answer['conflict']}")
        return True

    def _worked(self, client: Any, source: str) -> frozenset[str]:
        answer = self._send(client, "reported", source=source)
        if answer.get("op") != "reported":
            raise ChannelError(f"reported {source}: {answer.get('error')}")
        return frozenset(str(u) for u in answer.get("units", []))

    def _unit(self) -> str:
        stamp = datetime.fromtimestamp(self._clock(), timezone.utc)
        return f"{self._device}@{stamp.isoformat(timespec='seconds')}"

    # --- one pass ---------------------------------------------------------

    def pass_once(self) -> dict[str, int]:
        """Read and report everything due. Returns counts sent, per sense.

        Readings first, because they are cheap and time-sensitive, and the
        recording last, because whisper is the slow part and the cheap senses
        should not wait behind a meeting being transcribed.
        """
        sent = {"machine": 0, "presence": 0, "transcript": 0, "usage": 0, "recording": 0}
        try:
            client = self._connect(self._address)
        except OSError as exc:
            self._note(f"relay: cannot reach the core at {self._address}: {exc}")
            return sent
        try:
            for name, step in (
                ("machine", self._machine_pass),
                ("presence", self._presence_pass),
                ("transcript", self._transcript_pass),
                ("usage", self._usage_pass),
                ("recording", self._recording_pass),
            ):
                try:
                    sent[name] = step(client)
                except (OSError, ChannelError, ValueError) as exc:
                    # One sense failing does not stop the others; a dropped
                    # connection fails them all, and the next pass reconnects.
                    self._note(f"relay: {name} pass failed: {exc}")
        finally:
            try:
                client.close()
            except Exception:  # noqa: BLE001 - closing a dead socket is not news
                pass
        return sent

    def _machine_pass(self, client: Any) -> int:
        """On change, plus a heartbeat (DL-072)."""
        if self._read_machine is None:
            return 0
        current = self._read_machine()
        now = self._clock()
        due = now - self._machine_sent_at >= machine.HEARTBEAT_SECONDS
        if not (machine.changed(self._machine_sent, current) or due):
            return 0
        if not self._report(client, episodes.REPORT_MACHINE, self._unit(), machine.as_body(current)):
            return 0
        self._machine_sent, self._machine_sent_at = current, now
        return 1

    def _presence_pass(self, client: Any) -> int:
        """On every active/away flip, and every five minutes while active.

        An unreadable idle time sends nothing. Nothing goes stale, and stale
        reads as "not at the Mac" — the direction DL-073 chose for unknown.
        """
        if self._read_idle is None:
            return 0
        idle = self._read_idle()
        if idle is None:
            return 0
        away = machine.is_away(idle)
        now = self._clock()
        flipped = self._away_sent is None or away != self._away_sent
        due = not away and now - self._presence_sent_at >= machine.PRESENCE_EVERY_SECONDS
        if not (flipped or due):
            return 0
        if not self._report(client, episodes.REPORT_PRESENCE, self._unit(), {"idle": float(idle)}):
            return 0
        self._away_sent, self._presence_sent_at = away, now
        return 1

    def _transcript_pass(self, client: Any) -> int:
        if self._transcripts is None:
            return 0
        worked = self._worked(client, episodes.REPORT_TRANSCRIPT)
        sessions = transcripts.discover(self._transcripts, seen=worked)
        sent = 0
        for session in sessions[: transcripts.MAX_SESSIONS_PER_PASS]:
            body, reason = self._digested(lambda: transcripts.digest(session))
            meta = {"source": session.source, "project": session.project}
            if self._report(
                client, episodes.REPORT_TRANSCRIPT, session.id, body,
                meta=meta, reason=reason,
            ):
                sent += 1
        return sent

    def _usage_pass(self, client: Any) -> int:
        if self._usage is None:
            return 0
        worked = self._worked(client, episodes.REPORT_USAGE)
        days = habits.discover(self._usage, seen=worked)
        sent = 0
        for day in days[: habits.MAX_DAYS_PER_PASS]:
            body, reason = self._digested(lambda: habits.digest(day))
            if self._report(
                client, episodes.REPORT_USAGE, day.id, body,
                meta={"source": day.source}, reason=reason,
            ):
                sent += 1
        return sent

    @staticmethod
    def _digested(make: Callable[[], Optional[str]]) -> tuple[str, Optional[str]]:
        """A digest as ``(body, reason)``: empty body for "nothing to show",
        a reason for "could not be read" — the two outcomes the core's receipt
        already tells apart."""
        try:
            return make() or "", None
        except Exception as exc:  # noqa: BLE001 - becomes the receipt's reason
            return "", str(exc) or type(exc).__name__

    def _recording_pass(self, client: Any) -> int:
        """One recording per pass, as the local sense takes one (DL-066)."""
        if self._recordings is None:
            return 0
        if self._not_ready() is not None:
            # Whisper missing on this Mac is about the machine, not the file,
            # so no report: one would mark the recording heard forever.
            return 0
        waiting = listen.waiting(self._recordings)
        if not waiting:
            return 0
        worked = self._worked(client, episodes.REPORT_RECORDING)
        sent = 0
        for path in waiting[: listen.MAX_PER_PASS]:
            digest, size = _sha256(path)
            if digest in worked:
                # Already heard, by this relay earlier or by a core on the Mac.
                listen.file_away(path, folder=self._recordings)
                continue
            duration: Optional[float] = None
            try:
                heard = self._transcribe(path)
                body, reason = heard.text[: listen.MAX_REVIEW_CHARS], None
                # Unknown rather than zero: the receipt refuses a zero
                # duration, and a report the core cannot receipt would be
                # re-transcribed every pass forever.
                if heard.duration and heard.duration > 0:
                    duration = heard.duration
            except Exception as exc:  # noqa: BLE001 - becomes the receipt's reason
                body, reason = "", str(exc) or type(exc).__name__
            meta = {
                "source": "folder",
                "title": path.name,
                "mime": blobs.mime_for(path),
                "bytes": size,
                "duration": duration,
            }
            if self._report(
                client, episodes.REPORT_RECORDING, digest, body,
                meta=meta, reason=reason,
            ):
                # Only now. See the module docstring.
                listen.file_away(path, folder=self._recordings)
                sent += 1
        return sent

    # --- the loop ---------------------------------------------------------

    def run(self, stopping: threading.Event, *, every: float = PASS_SECONDS) -> None:
        """Pass, wait, pass, until ``stopping`` is set. The wait is interruptible."""
        while not stopping.is_set():
            sent = self.pass_once()
            if any(sent.values()):
                self._note(
                    "relay: sent "
                    + ", ".join(f"{n} {k}" for k, n in sent.items() if n)
                )
            stopping.wait(every)
