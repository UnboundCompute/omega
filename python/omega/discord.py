"""The phone is a Discord DM (DL-073). ``python -m omega.discord``.

A channel *client*, like the tray: it runs beside the core, talks to it over the
loopback channel, and holds no memory. Everything it knows lives in the log; the
only thing it keeps is a delivery cursor, which is a cache of how far it has
read, not a record of anything.

**In.** A direct message from exactly one Discord user — ``DISCORD_OWNER_ID`` —
becomes a ``say`` on the ``discord`` channel with the write key
``discord:<message id>``, so a redelivery is the channel's duplicate ack, not a
second message. Guild messages, other users, other bots and this bot itself are
ignored without a word. An attachment is refused with a plain reply: the only
attach path the channel has reads a file *by path on the core's machine*, and a
Discord upload is not one (never attach-by-path).

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
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from omega import projection
from omega.channel import DEFAULT_HOST, DEFAULT_PORT, PROTOCOL, ChannelClient, ChannelError

__all__ = [
    "CHANNEL",
    "LIMIT",
    "ENV_TOKEN",
    "ENV_OWNER",
    "accepts",
    "say_request",
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
    "Attachments aren't supported here yet, so I didn't pass that on. "
    "Send the words on their own."
)
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
        request = say_request(message_id, text)
        delay = BACKOFF_FIRST
        for attempt in range(1, self._attempts + 1):
            try:
                client = self._connect(self._address)
                try:
                    return client.request(request)
                finally:
                    client.close()
            except (OSError, ChannelError, ValueError) as exc:
                self._note(f"say attempt {attempt} failed: {type(exc).__name__}")
                if attempt < self._attempts:
                    self._sleep(delay)
                    delay = min(delay * 2, BACKOFF_MAX)
        return None


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
    return (run or _run)(
        token=token,
        owner_id=owner,
        address=(args.host, args.port),
        cursor=Cursor(state / CURSOR_FILE),
        log=log,
    )


# --- the discord.py shim ----------------------------------------------------


def _run(
    *,
    token: str,
    owner_id: int,
    address: tuple[str, int],
    cursor: Cursor,
    log: Callable[[str], None],
) -> int:
    """The only code that touches discord.py. Kept thin on purpose."""
    import asyncio
    import signal

    import discord  # the optional extra; imported here so nothing else needs it

    # DMs carry their content without the privileged message-content intent,
    # so the default intents are enough and the bot needs no portal toggle.
    client = discord.Client(intents=discord.Intents.default())
    inbound = Inbound(address, log=log)
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
        if message.attachments:
            log(f"refused a DM with {len(message.attachments)} attachment(s)")
            await message.channel.send(ATTACHMENTS_REFUSED)
            return
        if not message.content.strip():
            return
        ack = await asyncio.to_thread(inbound.say, message.id, message.content)
        if ack is None or ack.get("op") != "ack":
            log("a DM did not reach the core")
            await message.channel.send(CORE_UNREACHABLE)
            return
        log(f"DM filed as seq {ack['seq']}{' (duplicate)' if ack.get('duplicate') else ''}")
        try:
            await message.channel.typing()
        except Exception:  # noqa: BLE001 - a typing hint is decoration
            pass

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
