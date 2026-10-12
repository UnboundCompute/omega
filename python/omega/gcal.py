"""Their work calendar through Google sign-in — DL-081.

DL-080 reads a calendar's *secret iCal address*. A Google Workspace admin can
switch that address off, and a work calendar is exactly where they do. This is
the other way in: the person signs in once, grants omega read-only access to
their events (``calendar.events.readonly``), and omega keeps a refresh token on
the host to mint short-lived access tokens from.

**Same surface as the iCal reader.** What comes back is DL-080's
:class:`~omega.calendar.Occurrence`, so the look, the ``calendar`` tool and the
brief do not know or care which way the calendar was read. Google expands
recurring events itself (``singleEvents=true``), so this path guesses nothing.

**What it reads.** Title, location, start, end, status and whether *they*
declined — named in ``fields=`` so nothing else is even sent. Never the
description: invite text is strangers' writing and the injection surface a web
page is. Cancelled events and ones they declined are not on their day.

**The tokens are credentials.** The refresh token, the client secret and every
access token stay out of every exception, line and log: failures become
:class:`~omega.calendar.CalendarUnavailable` with a reason built here, raised
``from None``. Access tokens live in memory only.

**Signing in** (:func:`authorize`, ``python -m omega calendar-auth``) runs where
a browser is — the person's own machine — using the installed-app loopback
flow with PKCE, and prints the three lines to put in the host's env file. It
is the only code path that ever shows the refresh token, and only to the
person who just signed in.
"""

from __future__ import annotations

import base64
import hashlib
import http.client
import http.server
import json
import os
import secrets
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime, time as dtime, timedelta
from typing import Callable, Mapping, Optional

from omega.calendar import (
    CACHE_SECONDS,
    MAX_LOCATION,
    MAX_SUMMARY,
    CalendarUnavailable,
    Occurrence,
    _LOCAL,
    _clean,
    _overlaps,
)

__all__ = [
    "ENV_CLIENT_ID",
    "ENV_CLIENT_SECRET",
    "ENV_REFRESH_TOKEN",
    "SCOPE",
    "Credentials",
    "credentials",
    "read",
    "probe",
    "authorize",
]

ENV_CLIENT_ID = "OMEGA_GCAL_CLIENT_ID"
ENV_CLIENT_SECRET = "OMEGA_GCAL_CLIENT_SECRET"
ENV_REFRESH_TOKEN = "OMEGA_GCAL_REFRESH_TOKEN"

#: The narrowest scope that lists events: read-only by grant, not only by code.
SCOPE = "https://www.googleapis.com/auth/calendar.events.readonly"

AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
EVENTS_URL = "https://www.googleapis.com/calendar/v3/calendars/primary/events"

#: Only what a line needs. ``attendees(self,responseStatus)`` is enough to see
#: their own answer without reading anyone else's address.
FIELDS = (
    "nextPageToken,items(status,summary,location,start,end,"
    "attendees(self,responseStatus))"
)

#: A window is asked for on these boundaries, so looks a minute apart share
#: one cached answer instead of each calling Google.
_GRAIN = timedelta(minutes=5)
MAX_PAGES = 5
MAX_BYTES = 5_000_000
#: An access token is renewed this long before Google says it expires.
_EARLY = 60

SIGN_IN_AGAIN = "Google sign-in expired or was revoked: run `python -m omega calendar-auth` again"


@dataclass(frozen=True)
class Credentials:
    client_id: str
    client_secret: str = field(repr=False)
    refresh_token: str = field(repr=False)


def credentials(environ: Mapping[str, str] = os.environ) -> Optional[Credentials]:
    """The sign-in on this host, or ``None`` when any of the three is unset."""
    found = [environ.get(k, "").strip() for k in (ENV_CLIENT_ID, ENV_CLIENT_SECRET, ENV_REFRESH_TOKEN)]
    return Credentials(*found) if all(found) else None


# --- transport -----------------------------------------------------------------------

#: ``send(method, url, form, headers) -> (status, body)``. Every network call
#: goes through one, so tests inject a fake and never touch the network.
Send = Callable[[str, str, Optional[Mapping[str, str]], Mapping[str, str]], tuple[int, bytes]]


class _NoRedirects(urllib.request.HTTPRedirectHandler):
    """Google's API and token endpoint answer directly; a redirect would carry
    a bearer token somewhere it was not meant to go."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        return None


def send(
    method: str,
    url: str,
    form: Optional[Mapping[str, str]] = None,
    headers: Mapping[str, str] = {},
    *,
    timeout: float = 10.0,
) -> tuple[int, bytes]:
    """One https call. An HTTP error status is returned, not raised — the
    caller decides what it means; a failure to talk at all is raised as
    :class:`CalendarUnavailable` with a reason that names no secret."""
    data = urllib.parse.urlencode(form).encode() if form is not None else None
    request = urllib.request.Request(url, data=data, method=method, headers={"User-Agent": "omega-calendar", **headers})
    reason: str
    try:
        opener = urllib.request.build_opener(_NoRedirects)
        with opener.open(request, timeout=timeout) as resp:
            return resp.status, resp.read(MAX_BYTES + 1)
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read(64_000)
        except Exception:
            body = b""
        return exc.code, body
    except urllib.error.URLError as exc:
        why = exc.reason
        reason = "timed out" if isinstance(why, TimeoutError) else f"couldn't connect to Google ({type(why).__name__ if isinstance(why, BaseException) else 'network'})"
    except TimeoutError:
        reason = "timed out"
    except (OSError, http.client.HTTPException, ValueError) as exc:
        reason = f"connection to Google failed ({type(exc).__name__})"
    raise CalendarUnavailable(reason) from None


def _json(body: bytes) -> dict:
    if len(body) > MAX_BYTES:
        raise CalendarUnavailable("Google's answer was too large")
    try:
        value = json.loads(body.decode("utf-8", errors="replace"))
    except ValueError:
        raise CalendarUnavailable("Google's answer couldn't be read") from None
    return value if isinstance(value, dict) else {}


# --- access tokens -------------------------------------------------------------------

#: refresh-token digest -> (monotonic expiry, access token). Keyed by digest so
#: the dict itself never holds the refresh token.
_TOKENS: dict[str, tuple[float, str]] = {}


def _key(creds: Credentials) -> str:
    return hashlib.sha256(f"{creds.client_id}\0{creds.refresh_token}".encode()).hexdigest()


def _access_token(creds: Credentials, send: Send) -> str:
    key = _key(creds)
    hit = _TOKENS.get(key)
    now = time.monotonic()
    if hit is not None and now < hit[0]:
        return hit[1]
    status, body = send(
        "POST",
        TOKEN_URL,
        {
            "client_id": creds.client_id,
            "client_secret": creds.client_secret,
            "refresh_token": creds.refresh_token,
            "grant_type": "refresh_token",
        },
        {},
    )
    answer = _json(body) if body else {}
    if status != 200:
        if answer.get("error") in ("invalid_grant", "unauthorized_client"):
            raise CalendarUnavailable(SIGN_IN_AGAIN)
        if answer.get("error") == "invalid_client":
            raise CalendarUnavailable("Google doesn't recognise the OAuth client (check OMEGA_GCAL_CLIENT_ID/SECRET)")
        raise CalendarUnavailable(f"Google sign-in failed (status {status})")
    token = answer.get("access_token")
    if not isinstance(token, str) or not token:
        raise CalendarUnavailable("Google sign-in gave no access token")
    try:
        life = max(0, int(answer.get("expires_in", 3600)) - _EARLY)
    except (TypeError, ValueError):
        life = 0
    _TOKENS[key] = (now + life, token)
    return token


# --- events --------------------------------------------------------------------------


def _floor(t: datetime) -> datetime:
    t = t.astimezone(_LOCAL).replace(second=0, microsecond=0)
    return t - timedelta(minutes=t.minute % 5)


def _ceil(t: datetime) -> datetime:
    f = _floor(t)
    return f if f == t.astimezone(_LOCAL) else f + _GRAIN


def _instant(value: object) -> Optional[tuple[datetime, bool]]:
    """``{"dateTime": ...}`` or ``{"date": ...}`` → (aware time, all-day)."""
    if not isinstance(value, dict):
        return None
    try:
        if isinstance(value.get("dateTime"), str):
            t = datetime.fromisoformat(value["dateTime"])
            return (t if t.tzinfo else t.replace(tzinfo=_LOCAL)), False
        if isinstance(value.get("date"), str):
            return datetime.combine(date.fromisoformat(value["date"]), dtime(), _LOCAL), True
    except ValueError:
        return None
    return None


def _declined(item: dict) -> bool:
    for who in item.get("attendees") or ():
        if isinstance(who, dict) and who.get("self") and who.get("responseStatus") == "declined":
            return True
    return False


def _occurrence(item: object) -> Optional[Occurrence]:
    if not isinstance(item, dict) or item.get("status") == "cancelled" or _declined(item):
        return None
    start, end = _instant(item.get("start")), _instant(item.get("end"))
    if start is None:
        return None
    if end is None or end[0] < start[0]:
        end = start
    summary = item.get("summary")
    location = item.get("location")
    return Occurrence(
        start=start[0],
        end=end[0],
        all_day=start[1],
        summary=_clean(summary, MAX_SUMMARY) if isinstance(summary, str) and summary.strip() else "(no title)",
        location=_clean(location, MAX_LOCATION) if isinstance(location, str) else "",
    )


_EVENTS: dict[tuple[str, datetime, datetime], tuple[float, tuple[Occurrence, ...]]] = {}


def _list(creds: Credentials, start: datetime, end: datetime, send: Send) -> tuple[Occurrence, ...]:
    key = (_key(creds), start, end)
    hit = _EVENTS.get(key)
    now = time.monotonic()
    if hit is not None and now - hit[0] < CACHE_SECONDS:
        return hit[1]
    found: list[Occurrence] = []
    page = ""
    retried = False
    for _ in range(MAX_PAGES):
        query = {
            "timeMin": start.isoformat(),
            "timeMax": end.isoformat(),
            "singleEvents": "true",
            "orderBy": "startTime",
            "maxResults": "250",
            "fields": FIELDS,
        }
        if page:
            query["pageToken"] = page
        token = _access_token(creds, send)
        status, body = send("GET", f"{EVENTS_URL}?{urllib.parse.urlencode(query)}", None, {"Authorization": f"Bearer {token}"})
        if status == 401 and not retried:  # revoked early or rotated: mint once more
            _TOKENS.pop(_key(creds), None)
            retried = True
            continue
        if status != 200:
            raise CalendarUnavailable(_refusal(status, body))
        answer = _json(body)
        for item in answer.get("items") or ():
            o = _occurrence(item)
            if o is not None:
                found.append(o)
        page = answer.get("nextPageToken") or ""
        if not isinstance(page, str) or not page:
            break
    result = tuple(found)
    _EVENTS[key] = (now, result)
    return result


def _refusal(status: int, body: bytes) -> str:
    """A plain reason for a refused list. Google's own message is never echoed
    — only its machine-readable reason is matched."""
    try:
        reasons = {
            str(e.get("reason"))
            for e in (json.loads(body.decode("utf-8", errors="replace")).get("error", {}).get("errors") or ())
            if isinstance(e, dict)
        }
    except Exception:
        reasons = set()
    if status == 403 and reasons & {"accessNotConfigured", "SERVICE_DISABLED"}:
        return "the Calendar API is not enabled in the Google Cloud project"
    if status == 403 and reasons & {"rateLimitExceeded", "userRateLimitExceeded", "quotaExceeded"}:
        return "Google is rate-limiting calendar reads; try again shortly"
    if status in (401, 403):
        return f"Google refused access (status {status})"
    if status == 404:
        return "Google found no primary calendar for this account"
    return f"Google calendar answered with status {status}"


def read(
    start: datetime,
    end: datetime,
    *,
    environ: Mapping[str, str] = os.environ,
    send: Send = send,
) -> list[Occurrence]:
    """Their events overlapping ``[start, end)``, in start order.

    Raises :class:`CalendarUnavailable` when unconfigured or unreadable; never
    returns an empty list to mean anything but "nothing on".
    """
    creds = credentials(environ)
    if creds is None:
        raise CalendarUnavailable("Google sign-in is not set up on this host")
    found = _list(creds, _floor(start), _ceil(end), send)
    return [o for o in found if _overlaps(o, start, end)]


def probe(
    environ: Mapping[str, str] = os.environ, *, send: Send = send
) -> Optional[bool]:
    """Does the sign-in work again (DL-087 #3)? The ordinary refresh and
    nothing more: ``True`` it gave an access token, ``False`` Google still
    says the sign-in expired or was revoked, ``None`` anything else."""
    creds = credentials(environ)
    if creds is None:
        return None
    try:
        _access_token(creds, send)
    except CalendarUnavailable as exc:
        return False if str(exc) == SIGN_IN_AGAIN else None
    return True


# --- signing in, once, where a browser is ---------------------------------------------


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


class _Catch(http.server.BaseHTTPRequestHandler):
    """Takes Google's one redirect back and nothing else."""

    caught: dict[str, str]

    def do_GET(self) -> None:  # noqa: N802 - http.server's name
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
        if "code" in query or "error" in query:
            self.server.caught = {k: v[0] for k, v in query.items()}  # type: ignore[attr-defined]
            text = "omega can read your calendar now. You can close this tab."
        else:
            text = "waiting for Google"
        page = text.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(page)))
        self.end_headers()
        self.wfile.write(page)

    def log_message(self, *args: object) -> None:  # the code is a credential too
        pass


def authorize(
    client_id: str,
    client_secret: str,
    *,
    open_browser: Callable[[str], object],
    send: Send = send,
    say: Callable[[str], object] = print,
    timeout: float = 300.0,
) -> str:
    """The one-time sign-in. Returns the refresh token.

    Listens on 127.0.0.1 only, on a port the OS picks; checks ``state``; uses
    PKCE so an intercepted code is useless alone; asks for offline access with
    ``prompt=consent`` so Google issues a refresh token even on a second run.
    """
    verifier = _b64(secrets.token_bytes(48))
    state = _b64(secrets.token_bytes(16))
    server = http.server.HTTPServer(("127.0.0.1", 0), _Catch)
    server.caught = {}  # type: ignore[attr-defined]
    server.timeout = 1.0
    redirect = f"http://127.0.0.1:{server.server_address[1]}"
    url = AUTH_URL + "?" + urllib.parse.urlencode(
        {
            "client_id": client_id,
            "redirect_uri": redirect,
            "response_type": "code",
            "scope": SCOPE,
            "code_challenge": _b64(hashlib.sha256(verifier.encode()).digest()),
            "code_challenge_method": "S256",
            "state": state,
            "access_type": "offline",
            "prompt": "consent",
        }
    )
    try:
        say("Opening Google sign-in in your browser. If it doesn't open, visit:")
        say(url)
        open_browser(url)
        deadline = time.monotonic() + timeout
        while not server.caught and time.monotonic() < deadline:  # type: ignore[attr-defined]
            server.handle_request()
        caught: dict[str, str] = server.caught  # type: ignore[attr-defined]
    finally:
        server.server_close()
    if not caught:
        raise CalendarUnavailable("no answer from Google sign-in in time")
    if caught.get("state") != state:
        raise CalendarUnavailable("sign-in answer didn't match this request (state); try again")
    if "error" in caught:
        raise CalendarUnavailable(f"Google sign-in was not completed ({_clean(caught['error'], 40)})")
    status, body = send(
        "POST",
        TOKEN_URL,
        {
            "client_id": client_id,
            "client_secret": client_secret,
            "code": caught.get("code", ""),
            "code_verifier": verifier,
            "redirect_uri": redirect,
            "grant_type": "authorization_code",
        },
        {},
    )
    answer = _json(body) if body else {}
    if status != 200:
        raise CalendarUnavailable(f"Google wouldn't exchange the sign-in (status {status}, {_clean(str(answer.get('error', '')), 40)})")
    granted = str(answer.get("scope", SCOPE)).split()
    if SCOPE not in granted:
        raise CalendarUnavailable("calendar access was not granted (tick the calendar box on Google's consent page)")
    token = answer.get("refresh_token")
    if not isinstance(token, str) or not token:
        raise CalendarUnavailable("Google gave no refresh token; remove omega's access at myaccount.google.com/permissions and run this again")
    return token


def env_lines(client_id: str, client_secret: str, refresh_token: str) -> list[str]:
    """What to paste into the host's env file."""
    return [
        f"{ENV_CLIENT_ID}={client_id}",
        f"{ENV_CLIENT_SECRET}={client_secret}",
        f"{ENV_REFRESH_TOKEN}={refresh_token}",
    ]
