"""Share to their work Slack, as them (DL-101).

The person's own user token (``xoxp-``, :data:`ENV_TOKEN`) posts in their
name, straight away: a link, a few lines, a file or a screenshot they sent,
or something omega wrote ("post that summary to #team"). There is no draft
and no "May I?": they asked, on their own message, and the gate in
:mod:`omega.tools` is what makes sure it was their own message. A turn that
has read someone else's words holds the post for their yes instead (DL-064).

**One HTTPS seam.** Every call to the Slack Web API goes through ``http``
(:func:`_request` in production), so the tests hand in a fake and nothing
here ever touches the wire under test. Methods used, and nothing else:

- ``auth.test`` — who the token is, for the startup line and the probe;
- ``conversations.list`` (public, private, im, mpim) and ``users.list`` —
  the directory "#channel", "@person" or a name resolves against, cached per
  process and fetched again once when a name is not in it;
- ``conversations.open`` — the DM with a person;
- ``chat.postMessage``, ``chat.getPermalink`` and ``chat.delete`` — text;
- ``files.getUploadURLExternal``, the upload POST, and
  ``files.completeUploadExternal`` (with ``channel_id`` and an optional
  ``initial_comment``) — a file; ``files.delete`` takes one back.

**Never a blind retry.** A call that failed is not sent again, with exactly
one exception: Slack said ``ratelimited`` (HTTP 429), which means it did
nothing, and the call is sent once more after its ``Retry-After``. A post
that timed out may have landed, so it is reported as failed and left alone.

**The token never leaves this module.** It goes in one header and nowhere
else: no record, no log line, no error string carries it, and
:func:`_scrub` runs over every message built from what Slack said as a last
guard.

Scopes the user token needs: ``chat:write``, ``channels:read``,
``groups:read``, ``im:read``, ``mpim:read``, ``users:read``, ``im:write``,
``mpim:write`` and ``files:write`` (:data:`SCOPES`).
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, Optional, Sequence

from omega import episodes

if TYPE_CHECKING:  # pragma: no cover
    from omega.memory import MemoryStore

__all__ = [
    "ENV_TOKEN",
    "ORIGIN",
    "SLACK_FIX",
    "SCOPES",
    "MAX_FILE_BYTES",
    "MAX_TEXT",
    "Reply",
    "SlackError",
    "Unclear",
    "Target",
    "Client",
    "client",
    "account_line",
    "probe",
    "Posts",
    "Seat",
    "attachments",
    "fold",
    "posts",
]

ENV_TOKEN = "OMEGA_SLACK_USER_TOKEN"
API = "https://slack.com/api/"
#: The :mod:`omega.access` origin a refused token is lost under.
ORIGIN = "slack"
#: The one fix for a refused token, in omega's words. Its presence in a failed
#: ``slack`` result is what :func:`omega.access.classify` reads.
SLACK_FIX = (
    "Slack refused omega's token: make a new user token (xoxp-) for your "
    f"work Slack and set {ENV_TOKEN}"
)
SCOPES = (
    "chat:write", "channels:read", "groups:read", "im:read", "mpim:read",
    "users:read", "im:write", "mpim:write", "files:write",
)
#: Slack's own ceiling on a message's text is 40k; a chat post is far less.
MAX_TEXT = 12_000
#: What omega will upload in one go. Slack takes more; a phone screenshot or
#: a document is far less, and this bounds what one call reads into memory.
MAX_FILE_BYTES = 25 << 20
#: How many candidates an unclear ``to`` names back.
MAX_CANDIDATES = 5
#: The longest omega waits on a ``Retry-After`` before its one retry.
MAX_RETRY_AFTER = 30.0
#: Directory pages read at most, per listing, at 200 a page.
MAX_PAGES = 25
TIMEOUT = 20.0

#: What each refusal Slack names means, in a sentence the model can repeat.
_SAID = {
    "not_in_channel": "you are not in that channel, so it cannot be posted there as you; join it first",
    "channel_not_found": "Slack has no channel by that name that you can see",
    "is_archived": "that channel is archived",
    "missing_scope": "omega's Slack token lacks a permission this needs; add the scope to the token and set it again",
    "message_not_found": "that message is no longer in Slack",
    "cant_delete_message": "Slack would not let that message be deleted",
    "msg_too_long": "that is too long for one Slack message",
    "no_text": "there was nothing to post",
    "file_not_found": "that file is no longer in Slack",
    "user_not_found": "Slack has no such person",
    "ratelimited": "Slack is rate limiting omega; try again in a minute",
}
#: The refusals that mean the token itself is no good: :data:`SLACK_FIX`.
_LOST = frozenset({"invalid_auth", "token_revoked", "not_authed", "account_inactive", "token_expired"})


@dataclass
class Reply:
    """What one HTTPS request came back with."""

    status: int
    headers: dict[str, str]
    body: bytes


class _NoRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        return None


def _request(url: str, headers: dict[str, str], body: bytes, timeout: float) -> Reply:
    """One HTTPS POST: the seam every test replaces."""
    opener = urllib.request.build_opener(_NoRedirects())
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")  # noqa: S310
    try:
        with opener.open(request, timeout=timeout) as response:
            return Reply(getattr(response, "status", 200),
                         {k.lower(): v for k, v in response.headers.items()},
                         response.read(1 << 22))
    except urllib.error.HTTPError as exc:
        return Reply(exc.code, {k.lower(): v for k, v in exc.headers.items()}, exc.read(1 << 16))


Http = Callable[[str, dict[str, str], bytes, float], Reply]


class SlackError(RuntimeError):
    """A Slack call that did not do what it was asked. ``code`` is Slack's
    own error name, or ``unreachable`` / ``bad_reply`` for omega's."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code

    @property
    def lost(self) -> bool:
        return self.code in _LOST


class Unclear(RuntimeError):
    """``to`` named nobody, or more than one: never a guess, a question."""

    def __init__(self, to: str, candidates: Sequence[str]) -> None:
        self.to = to
        self.candidates = tuple(candidates[:MAX_CANDIDATES])
        if self.candidates:
            named = ", ".join(self.candidates)
            text = f"{to!r} could be {named}; which one? Nothing was posted."
        else:
            text = f"no channel or person in Slack matches {to!r}; who did you mean? Nothing was posted."
        super().__init__(text)


@dataclass(frozen=True)
class Target:
    """Where a post goes: a conversation id and how it is called."""

    id: str
    name: str  # "#team", "@sam", or a group DM's name
    user: str = ""  # for a person, their user id: the DM is opened at post time


def _scrub(text: str, token: str) -> str:
    return text.replace(token, "[token]") if token else text


def _said(code: str) -> str:
    if code in _LOST:
        return SLACK_FIX
    return _SAID.get(code, f"Slack said {code}")


def _retry_after(headers: Mapping[str, str]) -> float:
    """Seconds Slack asked omega to wait, from ``Retry-After``: 1 when it did
    not say, never more than :data:`MAX_RETRY_AFTER`."""
    said = next((v for k, v in headers.items() if k.lower() == "retry-after"), "")
    try:
        wait = float(str(said).strip())
    except ValueError:
        wait = 1.0
    if wait != wait or wait < 0:  # NaN or negative
        wait = 1.0
    return min(wait, MAX_RETRY_AFTER)


class Client:
    """One token's session. Thread-safe for the directory cache only; a turn
    makes its calls one at a time."""

    def __init__(
        self,
        token: str,
        *,
        http: Http = _request,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._token = token
        self._http = http
        self._sleep = sleep
        self._lock = threading.Lock()
        self._channels: Optional[list[dict[str, Any]]] = None
        self._people: Optional[list[dict[str, Any]]] = None

    # --- the wire ---------------------------------------------------------

    def call(self, method: str, params: Mapping[str, Any]) -> dict[str, Any]:
        """One Web API method, form-encoded. Retried once on ``ratelimited``
        and on nothing else."""
        body = urllib.parse.urlencode(
            {k: (json.dumps(v) if isinstance(v, (list, dict)) else str(v))
             for k, v in params.items() if v is not None}
        ).encode()
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
        }
        return self._send(API + method, headers, body, method)

    def _send(self, url: str, headers: dict[str, str], body: bytes, what: str, *, raw: bool = False) -> dict[str, Any]:
        for attempt in (1, 2):
            try:
                reply = self._http(url, headers, body, TIMEOUT)
            except OSError as exc:
                raise SlackError(
                    "unreachable",
                    f"couldn't reach Slack ({type(exc).__name__}); it may or may "
                    "not have gone through, so check before sending it again",
                ) from None
            limited = reply.status == 429
            data: dict[str, Any] = {}
            if not raw or reply.status != 200:
                try:
                    loaded = json.loads(reply.body.decode("utf-8", errors="replace") or "{}")
                    data = loaded if isinstance(loaded, dict) else {}
                except ValueError:
                    data = {}
            if raw and reply.status == 200:
                return {"ok": True}
            if data.get("error") == "ratelimited":
                limited = True
            if limited and attempt == 1:
                self._sleep(_retry_after(reply.headers))
                continue
            if limited:
                raise SlackError("ratelimited", _said("ratelimited"))
            if reply.status in (401, 403) and not data and not raw:
                raise SlackError("invalid_auth", SLACK_FIX)
            if data.get("ok") is True:
                return data
            code = str(data.get("error") or (f"http_{reply.status}" if reply.status != 200 else "bad_reply"))
            raise SlackError(code, _scrub(_said(code), self._token))
        raise AssertionError("unreachable")  # pragma: no cover

    # --- who and where ----------------------------------------------------

    def whoami(self) -> tuple[str, str]:
        """``(workspace, user)`` from ``auth.test``."""
        data = self.call("auth.test", {})
        return str(data.get("team") or "Slack"), str(data.get("user") or "you")

    def _listing(self, method: str, key: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        cursor = ""
        for _ in range(MAX_PAGES):
            data = self.call(method, {**params, "limit": 200, **({"cursor": cursor} if cursor else {})})
            out.extend(item for item in data.get(key) or [] if isinstance(item, dict))
            cursor = str((data.get("response_metadata") or {}).get("next_cursor") or "")
            if not cursor:
                break
        return out

    def _directory(self, refresh: bool = False) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        with self._lock:
            if refresh or self._channels is None or self._people is None:
                self._channels = self._listing(
                    "conversations.list", "channels",
                    {"types": "public_channel,private_channel,im,mpim", "exclude_archived": "true"},
                )
                self._people = [
                    u for u in self._listing("users.list", "members", {})
                    if not u.get("deleted") and not u.get("is_bot") and u.get("id") != "USLACKBOT"
                ]
            return self._channels, self._people

    def channels(self) -> list[str]:
        """The channels they can post in, by name: ones they are in."""
        rooms, _ = self._directory()
        return sorted(
            f"#{c['name']}" for c in rooms
            if c.get("name") and c.get("is_member") and not c.get("is_im") and not c.get("is_mpim")
        )

    def resolve(self, to: str) -> Target:
        """``#channel``, ``@person`` or a person's name to one conversation.
        Fetches the directory again once when nothing matches. Raises
        :class:`Unclear` for no match or more than one."""
        found, maybe = self._match(to, refresh=False)
        if not found and not maybe:
            found, maybe = self._match(to, refresh=True)
        if len(found) == 1:
            return found[0]
        names = [t.name for t in found] if found else maybe
        raise Unclear(to.strip(), sorted(dict.fromkeys(names)))

    def _match(self, to: str, *, refresh: bool) -> tuple[list[Target], list[str]]:
        rooms, people = self._directory(refresh)
        want = to.strip()
        sigil = want[:1] if want[:1] in "#@" else ""
        bare = want[1:].strip() if sigil else want
        key = bare.lower()
        if not key:
            return [], []
        found: list[Target] = []
        maybe: list[str] = []
        if sigil != "@":
            for c in rooms:
                if c.get("is_im"):
                    continue
                name = str(c.get("name") or "")
                if c.get("id") == bare or name.lower() == key:
                    found.append(Target(str(c["id"]), f"#{name}"))
                elif key in name.lower():
                    maybe.append(f"#{name}")
        if sigil != "#":
            exact: list[Target] = []
            loose: list[Target] = []
            for u in people:
                profile = u.get("profile") or {}
                handle = str(u.get("name") or "")
                names = [handle, str(profile.get("display_name") or ""),
                         str(profile.get("real_name") or u.get("real_name") or "")]
                person = Target("", f"@{handle}", user=str(u.get("id") or ""))
                if u.get("id") == bare or key in (n.lower() for n in names if n):
                    exact.append(person)
                elif any(key in n.lower().split() or n.lower().startswith(key) for n in names if n):
                    loose.append(person)
            found.extend(exact)
            if not exact:
                if len(loose) == 1 and not found:
                    found.extend(loose)
                else:
                    maybe.extend(t.name for t in loose)
        return found, maybe

    # --- the acts -----------------------------------------------------------

    def _room(self, target: Target) -> str:
        if target.id:
            return target.id
        data = self.call("conversations.open", {"users": target.user})
        room = str((data.get("channel") or {}).get("id") or "")
        if not room:
            raise SlackError("bad_reply", "Slack did not open the DM")
        return room

    def permalink(self, room: str, ts: str) -> str:
        try:
            data = self.call("chat.getPermalink", {"channel": room, "message_ts": ts})
        except SlackError:
            return ""
        return str(data.get("permalink") or "")

    def post_text(self, target: Target, text: str) -> dict[str, str]:
        room = self._room(target)
        data = self.call("chat.postMessage", {"channel": room, "text": text})
        ts = str(data.get("ts") or "")
        room = str(data.get("channel") or room)
        return {"channel": room, "ts": ts, "permalink": self.permalink(room, ts) if ts else ""}

    def post_file(self, target: Target, name: str, data: bytes, comment: str = "") -> dict[str, str]:
        if len(data) > MAX_FILE_BYTES:
            raise SlackError("too_large", f"{name} is over {MAX_FILE_BYTES >> 20} MB")
        room = self._room(target)
        slot = self.call("files.getUploadURLExternal", {"filename": name, "length": len(data)})
        url = str(slot.get("upload_url") or "")
        file_id = str(slot.get("file_id") or "")
        if not url.startswith("https://") or not file_id:
            raise SlackError("bad_reply", "Slack gave no upload address")
        self._send(url, {"Content-Type": "application/octet-stream"}, data, "upload", raw=True)
        done = self.call("files.completeUploadExternal", {
            "files": [{"id": file_id, "title": name}],
            "channel_id": room,
            **({"initial_comment": comment} if comment else {}),
        })
        files = done.get("files") or [{}]
        first = files[0] if isinstance(files, list) and files and isinstance(files[0], dict) else {}
        return {"channel": room, "ts": "", "file": file_id, "permalink": str(first.get("permalink") or "")}

    def delete(self, post: Mapping[str, Any]) -> None:
        if post.get("file"):
            self.call("files.delete", {"file": post["file"]})
        else:
            self.call("chat.delete", {"channel": post["channel"], "ts": post["ts"]})


# --- the process's one client -------------------------------------------------

_CLIENT: dict[str, Client] = {}
_CLIENT_LOCK = threading.Lock()


def token(environ: Optional[Mapping[str, str]] = None) -> str:
    env = os.environ if environ is None else environ
    return str(env.get(ENV_TOKEN, "") or "").strip()


def client(environ: Optional[Mapping[str, str]] = None) -> Optional[Client]:
    """The process's client for the configured token, or ``None`` when no
    token is set: then the tool is not offered and nothing changes."""
    held = token(environ)
    if not held:
        return None
    with _CLIENT_LOCK:
        if held not in _CLIENT:
            _CLIENT.clear()
            _CLIENT[held] = Client(held)
        return _CLIENT[held]


def account_line(environ: Optional[Mapping[str, str]] = None, *, http: Http = _request) -> Optional[str]:
    """The startup line, or ``None`` with no token, so a machine without it
    says exactly what it said before. Names the workspace and the person
    from ``auth.test``; never the token."""
    held = token(environ)
    if not held:
        return None
    try:
        team, user = Client(held, http=http, sleep=lambda _s: None).whoami()
    except SlackError as exc:
        return f"slack: on, but Slack refused the check: {_scrub(str(exc), held)}"
    return f"slack: on, posting to {team} as {user}"


def probe(environ: Optional[Mapping[str, str]] = None, *, http: Http = _request) -> Optional[bool]:
    """Does the token work again? ``auth.test``, never a post."""
    held = token(environ)
    if not held:
        return None
    try:
        Client(held, http=http, sleep=lambda _s: None).whoami()
    except SlackError as exc:
        return False if exc.lost else None
    return True


# --- what omega posted ----------------------------------------------------------


@dataclass(frozen=True)
class Posts:
    """omega's posts still standing, oldest first: ``slack.posted`` less
    ``slack.deleted``, folded from the log."""

    standing: tuple[dict[str, Any], ...] = ()

    def find(self, which: str) -> Optional[dict[str, Any]]:
        """By its ts, its file id or its permalink; empty is the latest."""
        which = (which or "").strip()
        if not which or which == "last":
            return self.standing[-1] if self.standing else None
        for post in reversed(self.standing):
            if which in (post.get("ts"), post.get("file"), post.get("permalink")):
                return post
        return None


def fold(decoded: Iterable[tuple[int, dict[str, Any]]]) -> Posts:
    standing: dict[tuple[str, str], dict[str, Any]] = {}
    for seq, payload in decoded:
        kind = payload.get("kind")
        if kind == episodes.SLACK_POSTED:
            key = (str(payload["channel"]), str(payload.get("ts") or payload.get("file") or ""))
            standing[key] = {**payload, "seq": seq}
        elif kind == episodes.SLACK_DELETED:
            standing.pop((str(payload["channel"]), str(payload.get("ts") or payload.get("file") or "")), None)
    return Posts(tuple(standing.values()))


@dataclass(frozen=True)
class Seat:
    """What a turn's box holds for ``slack`` (DL-101): the client, the files
    that came with their recent messages (context items with a blob, newest
    first), and the fold of what omega posted, read only when asked."""

    client: Client
    attached: tuple[dict[str, Any], ...] = ()
    posted: Callable[[], Posts] = Posts


def attachments(event: Mapping[str, Any], recalled: Iterable[Any] = ()) -> tuple[dict[str, Any], ...]:
    """The stored files on this message and the ones before it, newest
    first: the context items that carry a blob."""
    out: list[dict[str, Any]] = []
    payloads = [event] + [getattr(p, "payload", p) for p in reversed(list(recalled))]
    for payload in payloads:
        if not isinstance(payload, Mapping) or payload.get("kind", episodes.MESSAGE_INBOUND) != episodes.MESSAGE_INBOUND:
            continue
        for item in payload.get("context") or []:
            if isinstance(item, dict) and item.get("blob"):
                out.append(item)
    return tuple(out)


def posts(store: "MemoryStore") -> Posts:
    """The whole log, folded."""
    return fold(
        (e.seq, episodes.decode(e.payload)) for e in store.episodes_since(0)
    )
