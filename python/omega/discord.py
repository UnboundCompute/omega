"""The phone is a Discord DM (DL-073). ``python -m omega.discord``.

A channel *client*, like the tray: it runs beside the core, talks to it over the
loopback channel, and holds no memory. Everything it knows lives in the log; the
only thing it keeps is a delivery cursor, which is a cache of how far it has
read, not a record of anything.

**In.** A direct message from exactly one Discord user — ``DISCORD_OWNER_ID`` —
becomes a ``say`` on the ``discord`` channel with the write key
``discord:<message id>``, so a redelivery is the channel's duplicate ack, not a
second message. Guild messages, other users, other bots and this bot itself are
ignored without a word.

**Audio is heard here, on the VM, and handed in as text** (DL-079). The core
stays audio-free and holds no Sarvam key; the adapter downloads the attachment,
transcribes it with :func:`omega.sarvam.transcribe` off the event loop, and
sends words. A Discord *voice message* is the person talking to omega, so it is
an ordinary ``say`` under the same ``discord:<message id>`` key, marked
:data:`VOICE_MARK`. Any other audio file is a *recording*, filed as a
``report`` exactly as the Mac relay files one (DL-072): the unit is the digest
of the audio, so the same file sent twice is the channel's duplicate ack. There
is no whisper on the VM, so a failure is told to the person in the DM, never
dropped. Anything that is not audio is refused with a plain reply: the only
attach path the channel has reads a file *by path on the core's machine*, and
a Discord upload is not one (never attach-by-path).

**Out.** It subscribes from its cursor and decides per update:

* a reply to a message that came in on Discord is DM'd;
* a reply to a message that came in anywhere else a person types (the tray) is
  not — the tray already shows it;
* a message nobody asked for (omega's own look, a schedule fire) is DM'd only
  when the core says the person is **not** at the Mac. Unknown is "not at the
  Mac": a double buzz beats a missed one (DL-073 §3).

A reply's update names only its turn (``for_seq``); the turn's channel is on the
inbound's update, and an unprompted inbound is withheld from the wire entirely.
So the origin is read back with ``history`` at exactly ``for_seq``: an inbound
there names its channel, and nothing there means nobody asked.

**At most one message twice, never one lost.** The cursor is written after a
DM lands, so a crash between the send and the write re-sends that one message
on restart. A first start with no cursor begins at the head: history is never
DM'd.

**discord.py stays at the edge.** The owner filter, the routing decision, the
chunking and the cursor are plain functions and classes, tested without it; the
shim at the bottom is the only code that imports it, lazily.

Never logs a message's text or the token.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import tempfile
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Optional

from omega import blobs, episodes, listen, projection, sarvam
from omega.relay import _sha256
from omega.channel import DEFAULT_HOST, DEFAULT_PORT, PROTOCOL, ChannelClient, ChannelError

__all__ = [
    "CHANNEL",
    "LIMIT",
    "ENV_TOKEN",
    "ENV_OWNER",
    "accepts",
    "say_request",
    "report_request",
    "is_audio",
    "Dm",
    "chunks",
    "decide",
    "Cursor",
    "Inbound",
    "Outbound",
    "main",
]

#: The channel name a Discord message is filed under, and the one a reply is
#: routed back on.
CHANNEL = "discord"

#: Discord's limit on one message's content, in characters.
LIMIT = 2000

ENV_TOKEN = "DISCORD_TOKEN"
ENV_OWNER = "DISCORD_OWNER_ID"

#: Where the cursor lives: a host volume on the VM (``/var/lib/omega-discord``).
DEFAULT_STATE = "/state"
CURSOR_FILE = "cursor"

ATTACHMENTS_REFUSED = (
    "Only audio attachments are supported here, so I didn't pass that on. "
    "Send the words on their own, or a voice note."
)
#: What a transcribed voice note starts with, so the log and the model can tell
#: spoken words (and their mishearings) from typed ones.
VOICE_MARK = "(voice note) "
NO_KEY = (
    "I can't transcribe audio here: no Sarvam key is set up for the Discord "
    "side, so that wasn't passed on. Type it instead?"
)
#: How much of a failure's reason reaches the DM. The reason is Sarvam's or
#: ffmpeg's last line, already scrubbed of the key.
REASON_CHARS = 300
#: How many handled message ids are remembered, so a gateway replay of the
#: same message is not downloaded and transcribed a second time.
SEEN_MAX = 512
CORE_UNREACHABLE = "I couldn't reach omega just now, so that didn't land. Try again in a minute."
TURN_FAILED = "That one failed on my side, so there's no answer. Try again?"

#: Backoff for a lost core connection: doubling from the first to the last.
BACKOFF_FIRST = 1.0
BACKOFF_MAX = 60.0


def _stderr(line: str) -> None:
    print(f"omega-discord: {line}", file=sys.stderr, flush=True)


# --- in ---------------------------------------------------------------------


def accepts(
    *,
    author_id: int,
    owner_id: int,
    self_id: Optional[int],
    in_guild: bool,
    author_is_bot: bool,
) -> bool:
    """Only a DM, only from the owner, never from a bot or from ourselves."""
    if in_guild or author_is_bot:
        return False
    if self_id is not None and author_id == self_id:
        return False
    return author_id == owner_id


def is_audio(filename: str, content_type: Optional[str]) -> bool:
    """Audio by its declared type, else by the suffix the Mac relay hears.

    A file Discord labels video or image is not audio even with an ``.mp4`` or
    ``.webm`` name, which :data:`listen.AUDIO_SUFFIXES` would otherwise admit.
    """
    kind = (content_type or "").split(";")[0].strip().lower()
    if kind.startswith("audio/"):
        return True
    if kind.startswith(("video/", "image/")):
        return False
    return Path(filename or "").suffix.lower() in listen.AUDIO_SUFFIXES


def report_request(
    digest: str,
    *,
    title: str,
    mime: str,
    size: int,
    body: str,
    duration: Optional[float] = None,
    reason: Optional[str] = None,
) -> dict[str, Any]:
    """The ``report`` an audio file becomes: the relay's recording, from here.

    The unit is the digest of the audio because the core's receipt is keyed on
    it (``audio.captured.recording`` must be a blob digest), which is also what
    makes a second send of the same file a duplicate rather than a second
    review. ``meta.source`` says it came over Discord rather than a folder.
    """
    return {
        "v": PROTOCOL,
        "op": "report",
        "source": episodes.REPORT_RECORDING,
        "unit": digest,
        "device": CHANNEL,
        "body": body,
        "meta": {
            "source": CHANNEL,
            "title": title,
            "mime": mime,
            "bytes": size,
            "duration": duration,
        },
        "reason": reason,
    }


def say_request(message_id: Any, text: str) -> dict[str, Any]:
    """The ``say`` a DM becomes. The message id is the write key."""
    return {
        "v": PROTOCOL,
        "op": "say",
        "text": text,
        "channel": CHANNEL,
        "id": f"{CHANNEL}:{message_id}",
    }


class Inbound:
    """Hands one DM to the core, on a short connection of its own.

    Retries a lost connection a few times; the write key makes a retry after a
    lost ack land on the duplicate path rather than as a second message.
    Returns the ack, or ``None`` when the core could not be reached at all.
    """

    def __init__(
        self,
        address: tuple[str, int],
        *,
        connect: Callable[[tuple[str, int]], Any] = lambda a: ChannelClient(a, timeout=10.0),
        sleep: Callable[[float], None] = time.sleep,
        attempts: int = 4,
        log: Callable[[str], None] = _stderr,
    ) -> None:
        self._address = address
        self._connect = connect
        self._sleep = sleep
        self._attempts = attempts
        self._note = log

    def say(self, message_id: Any, text: str) -> Optional[dict[str, Any]]:
        return self.send(say_request(message_id, text))

    def send(self, request: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Any one request, with the same retries. Only ``say`` and ``report``
        are ever sent, and both carry a write key, so a retry is idempotent."""
        op = request.get("op")
        delay = BACKOFF_FIRST
        for attempt in range(1, self._attempts + 1):
            try:
                client = self._connect(self._address)
                try:
                    return client.request(request)
                finally:
                    client.close()
            except (OSError, ChannelError, ValueError) as exc:
                self._note(f"{op} attempt {attempt} failed: {type(exc).__name__}")
                if attempt < self._attempts:
                    self._sleep(delay)
                    delay = min(delay * 2, BACKOFF_MAX)
        return None


def _scrub(reason: str, key: Optional[str]) -> str:
    """One line, short, and never the key — a last guard, not the plan."""
    if key:
        reason = reason.replace(key, "[key]")
    reason = " ".join(reason.split())
    return reason[:REASON_CHARS] or "no reason given"


async def _save(attachment: Any, path: Path) -> None:
    """The real download: discord.py's own ``Attachment.save``."""
    await attachment.save(path)


class Dm:
    """One owner DM, start to finish: text, voice note, recording, refusal.

    Duck-typed over discord.py's ``Message`` (``id``, ``content``,
    ``attachments``, ``flags.voice``, ``channel.send``) so a test hands it a
    plain object. ``fetch`` (the download), ``transcribe`` and ``not_ready``
    (ffmpeg present?) are the seams; the blocking parts — the transcription and
    the channel requests — run in a worker thread so the gateway heartbeat
    keeps beating through a long recording.

    **Once per message.** The core's write keys make a resend land on the
    duplicate path; the ``seen`` memory is what keeps a gateway replay from
    paying for a second download and transcription before it gets there.
    """

    def __init__(
        self,
        inbound: Inbound,
        *,
        hearing: Optional[sarvam.Settings],
        fetch: Callable[[Any, Path], Awaitable[None]] = _save,
        transcribe: Callable[..., listen.Heard] = sarvam.transcribe,
        not_ready: Callable[[], Optional[str]] = sarvam.not_ready,
        log: Callable[[str], None] = _stderr,
    ) -> None:
        self._inbound = inbound
        self._hearing = hearing
        self._fetch = fetch
        self._transcribe = transcribe
        self._not_ready = not_ready
        self._note = log
        self._seen: OrderedDict[Any, None] = OrderedDict()

    async def handle(self, message: Any) -> None:
        if message.id in self._seen:
            self._note("skipped a DM already handled")
            return
        self._seen[message.id] = None
        while len(self._seen) > SEEN_MAX:
            self._seen.popitem(last=False)

        attachments = list(getattr(message, "attachments", None) or [])
        audio = [a for a in attachments if is_audio(a.filename, getattr(a, "content_type", None))]
        if len(audio) < len(attachments):
            # The whole message, as before audio was heard: passing on half of
            # it would leave the person guessing which half arrived.
            self._note(f"refused a DM with {len(attachments) - len(audio)} non-audio attachment(s)")
            await message.channel.send(ATTACHMENTS_REFUSED)
            return
        # Typed words go in as typed, exactly as before audio was heard.
        text = getattr(message, "content", "") or ""
        if not text.strip():
            text = ""
        flags = getattr(message, "flags", None)
        voice = bool(getattr(flags, "voice", False)) and bool(audio)

        if voice:
            spoken = await self._hear_voice(message, audio[0])
            if spoken is not None:
                text = f"{text}\n\n{spoken}" if text else spoken
            audio = audio[1:]
        if text:
            await self._say(message, text)
        for attachment in audio:
            await self._recording(message, attachment)

    async def _say(self, message: Any, text: str) -> None:
        ack = await asyncio.to_thread(self._inbound.say, message.id, text)
        if ack is None or ack.get("op") != "ack":
            self._note("a DM did not reach the core")
            await message.channel.send(CORE_UNREACHABLE)
            return
        self._note(f"DM filed as seq {ack['seq']}{' (duplicate)' if ack.get('duplicate') else ''}")
        try:
            await message.channel.typing()
        except Exception:  # noqa: BLE001 - a typing hint is decoration
            pass

    def _unready(self) -> Optional[str]:
        """Why nothing can be heard on this machine, as the DM says it."""
        if self._hearing is None:
            return NO_KEY
        why = self._not_ready()
        if why is not None:
            return f"I can't transcribe audio here right now ({why}), so that wasn't passed on."
        return None

    async def _heard(
        self, attachment: Any, folder: Path
    ) -> tuple[Path, Optional[listen.Heard], Optional[str]]:
        """Download and transcribe one attachment: ``(path, heard, reason)``."""
        name = Path(attachment.filename or "audio").name or "audio"
        path = folder / name
        try:
            await self._fetch(attachment, path)
        except Exception as exc:  # noqa: BLE001 - told to the person, not raised
            return path, None, f"the download from Discord failed ({type(exc).__name__})"
        assert self._hearing is not None
        chosen = self._hearing
        try:
            heard = await asyncio.to_thread(
                self._transcribe, path, key=chosen.key, model=chosen.model,
                language=chosen.language, mode=chosen.mode,
            )
        except Exception as exc:  # noqa: BLE001 - NotHeard, or anything else
            return path, None, _scrub(str(exc) or type(exc).__name__, chosen.key)
        return path, heard, None

    async def _hear_voice(self, message: Any, attachment: Any) -> Optional[str]:
        """The voice note's words, marked; ``None`` after telling why not."""
        unready = self._unready()
        if unready is not None:
            await message.channel.send(unready)
            return None
        with tempfile.TemporaryDirectory(prefix="omega-discord-") as tmp:
            _, heard, reason = await self._heard(attachment, Path(tmp))
        if heard is None:
            self._note("a voice note could not be transcribed")
            await message.channel.send(
                f"I couldn't transcribe that voice note ({reason}), so it wasn't "
                "passed on. Try again, or type it?"
            )
            return None
        return VOICE_MARK + heard.text.strip()

    async def _recording(self, message: Any, attachment: Any) -> None:
        """One audio file, filed as the relay files a recording."""
        title = Path(attachment.filename or "audio").name or "audio"
        unready = self._unready()
        if unready is not None:
            # About this machine, not the file, so no report: one would mark
            # the recording heard for good (the relay's rule, DL-072).
            await message.channel.send(unready)
            return
        try:
            await message.channel.typing()
        except Exception:  # noqa: BLE001
            pass
        with tempfile.TemporaryDirectory(prefix="omega-discord-") as tmp:
            path, heard, reason = await self._heard(attachment, Path(tmp))
            if not path.is_file() or path.stat().st_size == 0:
                # Nothing to key a report on: the digest *is* the identity.
                await message.channel.send(
                    f"I couldn't get {title} from Discord "
                    f"({reason or 'the file was empty'}), so it wasn't passed on."
                )
                return
            digest, size = await asyncio.to_thread(_sha256, path)
        mime = (getattr(attachment, "content_type", None) or "").split(";")[0].strip()
        mime = mime or blobs.mime_for(title)
        if heard is not None:
            body = heard.text[: listen.MAX_REVIEW_CHARS]
            duration = heard.duration if heard.duration and heard.duration > 0 else None
        else:
            body, duration = "", None
        request = report_request(
            digest, title=title, mime=mime, size=size, body=body,
            duration=duration, reason=reason,
        )
        ack = await asyncio.to_thread(self._inbound.send, request)
        if ack is None or ack.get("op") != "ack":
            self._note("a recording did not reach the core")
            await message.channel.send(CORE_UNREACHABLE)
            return
        self._note(
            f"recording filed as seq {ack['seq']}"
            f"{' (duplicate)' if ack.get('duplicate') else ''}"
        )
        if ack.get("duplicate"):
            await message.channel.send(
                f"omega already has {title} (the same audio came in before), "
                "so it wasn't filed again."
            )
        elif reason is not None:
            await message.channel.send(
                f"I couldn't transcribe {title} ({reason}). omega has a note "
                "that it arrived, but nothing from it."
            )
        else:
            await message.channel.send(
                f"Got {title} and transcribed it. omega reviews it in the "
                "background and keeps what matters; the review isn't sent "
                "here, so ask about it when you want it."
            )


# --- out --------------------------------------------------------------------


def chunks(text: str, limit: int = LIMIT) -> list[str]:
    """Split a reply into messages of at most ``limit`` characters.

    Cuts at the last paragraph break that fits, else the last line break, else
    the last space, else hard at the limit. Whitespace at a cut is dropped, so
    no message starts or ends blank; no non-blank text is lost.
    """
    if limit < 1:
        raise ValueError(f"limit must be positive, got {limit}")
    out: list[str] = []
    rest = text.strip()
    while len(rest) > limit:
        window = rest[: limit + 1]
        cut = -1
        for sep in ("\n\n", "\n", " "):
            at = window.rfind(sep)
            if at > 0:
                cut = at
                break
        if cut <= 0:
            cut = limit
        head = rest[:cut].rstrip()
        if head:
            out.append(head)
        rest = rest[cut:].lstrip()
    if rest:
        out.append(rest)
    return out


def _text(update: dict[str, Any], origin: Callable[[], Optional[str]]) -> Optional[str]:
    """What a terminal update would say to the person, or ``None``."""
    state = update.get("state")
    if state == projection.COMPLETE:
        reply = update.get("reply")
        if update.get("outcome") == "spoke" and isinstance(reply, str) and reply.strip():
            return reply
        return None
    if state == projection.BLOCKED:
        needs = update.get("needs")
        return needs if isinstance(needs, str) and needs.strip() else None
    if state == projection.FAILED:
        # Only for a turn someone asked for on Discord: they are waiting on the
        # phone and would otherwise hear nothing. The error itself stays in.
        return TURN_FAILED if origin() == CHANNEL else None
    return None


def decide(
    update: dict[str, Any],
    *,
    origin: Callable[[], Optional[str]],
    at_the_mac: Callable[[], bool],
) -> Optional[str]:
    """The text to DM for one update, or ``None`` to send nothing.

    ``origin`` is the channel of the inbound that started the update's turn,
    or ``None`` when nobody asked (an unprompted turn, whose inbound never
    reaches the wire). Both questions are callables so they are asked only
    about the updates that would say something.
    """
    text = _text(update, origin)
    if text is None:
        return None
    came_from = origin()
    if came_from == CHANNEL:
        return text
    if came_from is None:
        return None if at_the_mac() else text
    return None


class Cursor:
    """The last seq handled, in one small file. Written atomically."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)

    def load(self) -> Optional[int]:
        try:
            raw = self._path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return None
        try:
            value = int(raw)
        except ValueError:
            return None
        return value if value >= 0 else None

    def save(self, seq: int) -> None:
        tmp = self._path.with_name(self._path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(f"{seq}\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self._path)


class Outbound:
    """Follows the projection and DMs what routing says to.

    ``send`` delivers one message (one chunk) to the owner and raises if it
    could not; a raise ends the session without moving the cursor, so the next
    session sends it again. A lost core connection reconnects with backoff and
    never ends :meth:`run`.
    """

    def __init__(
        self,
        address: tuple[str, int],
        cursor: Cursor,
        send: Callable[[str], None],
        *,
        connect: Callable[[tuple[str, int]], Any] = lambda a: ChannelClient(a, timeout=None),
        sleep: Callable[[float], None] = time.sleep,
        log: Callable[[str], None] = _stderr,
    ) -> None:
        self._address = address
        self._cursor = cursor
        self._send = send
        self._connect = connect
        self._sleep = sleep
        self._note = log
        self._subscribed = False

    def run(self, stopping: threading.Event) -> None:
        delay = BACKOFF_FIRST
        while not stopping.is_set():
            try:
                client = self._connect(self._address)
            except OSError as exc:
                self._note(f"core unreachable ({type(exc).__name__}); retrying in {delay:.0f}s")
            else:
                self._subscribed = False
                try:
                    self.session(client, stopping)
                except (OSError, ChannelError, ValueError) as exc:
                    self._note(f"core connection lost ({type(exc).__name__}); retrying in {delay:.0f}s")
                except Exception as exc:  # noqa: BLE001 - a failed DM is retried, not fatal
                    self._note(f"delivery failed ({type(exc).__name__}); retrying in {delay:.0f}s")
                finally:
                    try:
                        client.close()
                    except Exception:  # noqa: BLE001
                        pass
            if stopping.is_set():
                return
            if self._subscribed:
                # The connection worked before it dropped: start the backoff
                # over rather than punishing a healthy link for an old outage.
                delay = BACKOFF_FIRST
                self._subscribed = False
            self._sleep(delay)
            delay = min(delay * 2, BACKOFF_MAX)

    def session(self, client: Any, stopping: threading.Event) -> None:
        """One connection: subscribe from the cursor, handle until it drops."""
        since = self._cursor.load()
        if since is None:
            since = int(client.hello["head"])
            self._cursor.save(since)
            self._note(f"first start: beginning at head {since}, no history sent")
        answer = client.request({"v": PROTOCOL, "op": "subscribe", "since": since})
        if answer.get("op") != "subscribed":
            raise ChannelError(f"subscribe refused: {answer.get('error')}")
        self._subscribed = True
        self._note(f"following the core from seq {since}")
        while not stopping.is_set():
            update = client.read()
            if update.get("op") != "update":
                continue
            self.handle(client, update)

    def handle(self, client: Any, update: dict[str, Any]) -> bool:
        """Route one update; True when it was DM'd. Moves the cursor past it."""
        seq = int(update["seq"])
        origin_cache: list[Optional[str]] = []

        def origin() -> Optional[str]:
            if not origin_cache:
                origin_cache.append(self._origin(client, int(update["for_seq"])))
            return origin_cache[0]

        text = decide(update, origin=origin, at_the_mac=lambda: self._at_the_mac(client))
        if text is not None:
            parts = chunks(text)
            for part in parts:
                self._send(part)
            self._note(f"sent seq {seq} as {len(parts)} message(s)")
        self._cursor.save(seq)
        return text is not None

    @staticmethod
    def _origin(client: Any, for_seq: int) -> Optional[str]:
        """The channel of the inbound at ``for_seq``, or ``None`` if nobody asked."""
        answer = client.request(
            {"v": PROTOCOL, "op": "history", "before": for_seq + 1, "limit": 1}
        )
        if answer.get("op") != "history":
            raise ChannelError(f"history refused: {answer.get('error')}")
        for item in answer.get("updates", []):
            if item.get("seq") == for_seq and item.get("state") == projection.UNDERSTOOD:
                channel = item.get("channel")
                return str(channel) if channel else ""
        return None

    @staticmethod
    def _at_the_mac(client: Any) -> bool:
        """The core's answer, and anything but a clear yes is no."""
        answer = client.request({"v": PROTOCOL, "op": "presence"})
        return answer.get("op") == "presence" and answer.get("at_the_mac") is True


# --- entry point ------------------------------------------------------------


def _owner(raw: str) -> Optional[int]:
    raw = raw.strip()
    if not raw.isdigit():
        return None
    value = int(raw)
    return value if value > 0 else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omega.discord",
        description="Carry omega's conversation to and from one person's Discord DMs.",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help=argparse.SUPPRESS)
    parser.add_argument("--port", default=DEFAULT_PORT, type=int, help="the core's channel port")
    parser.add_argument("--state", default=DEFAULT_STATE, help="where the delivery cursor lives")
    return parser


def main(
    argv: Optional[list[str]] = None,
    *,
    environ: Mapping[str, str] = os.environ,
    run: Optional[Callable[..., int]] = None,
    log: Callable[[str], None] = _stderr,
) -> int:
    """Unconfigured is not a failure: it logs why and exits 0."""
    args = build_parser().parse_args(argv)
    token = environ.get(ENV_TOKEN, "").strip()
    if not token:
        log(f"{ENV_TOKEN} is not set; the Discord adapter is off")
        return 0
    owner = _owner(environ.get(ENV_OWNER, ""))
    if owner is None:
        log(f"{ENV_OWNER} is not set to a Discord user id; the Discord adapter is off")
        return 0
    state = Path(args.state)
    if not state.is_dir():
        log(f"the state directory {state} does not exist; the Discord adapter is off")
        return 0
    hearing = sarvam.settings(None, environ=environ)
    if hearing is None:
        log("SARVAM_API_KEY is not set; audio sent over Discord will be refused")
    return (run or _run)(
        token=token,
        owner_id=owner,
        address=(args.host, args.port),
        cursor=Cursor(state / CURSOR_FILE),
        hearing=hearing,
        log=log,
    )


# --- the discord.py shim ----------------------------------------------------


def _run(
    *,
    token: str,
    owner_id: int,
    address: tuple[str, int],
    cursor: Cursor,
    hearing: Optional[sarvam.Settings] = None,
    log: Callable[[str], None],
) -> int:
    """The only code that touches discord.py. Kept thin on purpose."""
    import signal

    import discord  # the optional extra; imported here so nothing else needs it

    # DMs carry their content without the privileged message-content intent,
    # so the default intents are enough and the bot needs no portal toggle.
    client = discord.Client(intents=discord.Intents.default())
    inbound = Inbound(address, log=log)
    dms = Dm(inbound, hearing=hearing, log=log)
    stopping = threading.Event()
    started = threading.Event()

    async def dm(text: str) -> None:
        user = client.get_user(owner_id) or await client.fetch_user(owner_id)
        await user.send(text)

    def send(text: str) -> None:
        asyncio.run_coroutine_threadsafe(dm(text), client.loop).result(timeout=120)

    @client.event
    async def on_ready() -> None:
        if started.is_set():
            return  # a gateway reconnect, not a second start
        started.set()
        log("connected to Discord")
        outbound = Outbound(address, cursor, send, log=log)
        threading.Thread(
            target=outbound.run, args=(stopping,), name="omega-discord-out", daemon=True
        ).start()

    @client.event
    async def on_message(message: Any) -> None:
        if not accepts(
            author_id=message.author.id,
            owner_id=owner_id,
            self_id=client.user.id if client.user else None,
            in_guild=message.guild is not None,
            author_is_bot=bool(message.author.bot),
        ):
            return
        await dms.handle(message)

    async def runner() -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda: asyncio.ensure_future(client.close()))
        async with client:
            await client.start(token)

    try:
        asyncio.run(runner())
    except discord.LoginFailure:
        log("Discord refused the token; the Discord adapter is off")
        return 0
    finally:
        stopping.set()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
