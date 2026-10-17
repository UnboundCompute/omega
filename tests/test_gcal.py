"""Their work calendar through Google sign-in — DL-081.

Offline throughout: every Google call goes through an injected ``send``, and
the sign-in test talks only to its own 127.0.0.1 listener.

What is pinned: the events read map onto DL-080's occurrences (cancelled and
declined dropped, all-day and timed both right, description never asked for);
access tokens are minted once and reused; every failure is a plain reason that
never carries the refresh token, client secret or access token; and the look,
the tool and startup treat a Google sign-in as a connected calendar.
"""

from __future__ import annotations

import json
import threading
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from omega import calendar as cal, gcal
from omega.__main__ import _account_lines, calendar_auth
from omega.calendar import CalendarUnavailable

IST = ZoneInfo("Asia/Kolkata")
NOW = datetime(2026, 10, 12, 8, 0, tzinfo=IST)
REFRESH = "1//refresh-SECRET-zz"
CLIENT_SECRET = "GOCSPX-client-SECRET"
ACCESS = "ya29.access-SECRET"
ENV = {
    gcal.ENV_CLIENT_ID: "123.apps.googleusercontent.com",
    gcal.ENV_CLIENT_SECRET: CLIENT_SECRET,
    gcal.ENV_REFRESH_TOKEN: REFRESH,
}
SECRETS = (REFRESH, CLIENT_SECRET, ACCESS)


@pytest.fixture(autouse=True)
def _fresh() -> None:
    gcal._TOKENS.clear()
    gcal._EVENTS.clear()
    cal._CACHE.clear()


def _event(summary="Standup", start="2026-10-12T09:30:00+05:30", end="2026-10-12T09:45:00+05:30", **extra):
    return {"status": "confirmed", "summary": summary, "start": {"dateTime": start}, "end": {"dateTime": end}, **extra}


class Google:
    """A fake Google: one token endpoint, one events endpoint."""

    def __init__(self, items=(), *, token_status=200, token_body=None, events_status=200, events_body=None, pages=None):
        self.items = list(items)
        self.token_status, self.token_body = token_status, token_body
        self.events_status, self.events_body = events_status, events_body
        self.pages = pages
        self.calls: list[tuple[str, str, dict | None, dict]] = []

    def __call__(self, method, url, form, headers):
        self.calls.append((method, url, dict(form) if form else None, dict(headers)))
        if url == gcal.TOKEN_URL:
            body = self.token_body if self.token_body is not None else {"access_token": ACCESS, "expires_in": 3599}
            return self.token_status, json.dumps(body).encode()
        assert url.startswith(gcal.EVENTS_URL + "?")
        assert headers["Authorization"] == f"Bearer {ACCESS}"
        if self.events_status != 200:
            return self.events_status, json.dumps(self.events_body or {}).encode()
        if self.pages is not None:
            query = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
            index = int(query.get("pageToken", ["0"])[0])
            body = {"items": self.pages[index]}
            if index + 1 < len(self.pages):
                body["nextPageToken"] = str(index + 1)
            return 200, json.dumps(body).encode()
        return 200, json.dumps({"items": self.items}).encode()

    def queries(self):
        return [urllib.parse.parse_qs(urllib.parse.urlsplit(u).query) for m, u, f, h in self.calls if u != gcal.TOKEN_URL]


def _read(google, start=NOW, hours=12):
    return gcal.read(start, start + timedelta(hours=hours), environ=ENV, send=google)


# --- what is read --------------------------------------------------------------------


def test_events_become_occurrences_and_the_ask_is_narrow() -> None:
    google = Google([_event(location="Room 4"), _event("Lunch", "2026-10-12T13:00:00+05:30", "2026-10-12T14:00:00+05:30")])
    found = _read(google)
    assert [(o.summary, o.location, o.start.hour, o.start.minute) for o in found] == [
        ("Standup", "Room 4", 9, 30),
        ("Lunch", "", 13, 0),
    ]
    (query,) = google.queries()
    assert query["singleEvents"] == ["true"] and query["orderBy"] == ["startTime"]
    assert "description" not in query["fields"][0]
    assert gcal.SCOPE.endswith("calendar.events.readonly")


def test_cancelled_and_declined_are_not_on_their_day() -> None:
    google = Google(
        [
            _event("Gone", status="cancelled"),
            _event("Said no", attendees=[{"self": True, "responseStatus": "declined"}]),
            _event("Others said no", attendees=[{"responseStatus": "declined"}, {"self": True, "responseStatus": "accepted"}]),
        ]
    )
    assert [o.summary for o in _read(google)] == ["Others said no"]


def test_all_day_and_untitled_and_hostile_titles() -> None:
    google = Google(
        [
            {"status": "confirmed", "start": {"date": "2026-10-12"}, "end": {"date": "2026-10-13"}},
            _event("\x1b[31mred‮ title " + "x" * 300),
        ]
    )
    # an all-day date is the host's own day (as in DL-080), and the timed one is
    # fixed in IST, so look from the host's 11 Oct through its 12 Oct
    day = datetime(2026, 10, 11).astimezone()
    allday, timed = sorted(_read(google, day, hours=48), key=lambda o: not o.all_day)
    assert allday.all_day and allday.summary == "(no title)"
    assert "\x1b" not in timed.summary and "‮" not in timed.summary
    assert len(timed.summary) <= cal.MAX_SUMMARY


def test_the_window_is_honoured_after_rounding() -> None:
    google = Google([_event("Before", "2026-10-12T07:00:00+05:30", "2026-10-12T07:59:00+05:30"), _event()])
    assert [o.summary for o in _read(google, NOW + timedelta(minutes=2))] == ["Standup"]
    (query,) = google.queries()
    # floored to the 5-minute grain; compared as an instant, since the host zone renders it
    assert datetime.fromisoformat(query["timeMin"][0]) == NOW


def test_pages_are_followed() -> None:
    google = Google(pages=[[_event("One")], [_event("Two", "2026-10-12T10:00:00+05:30", "2026-10-12T10:30:00+05:30")]])
    assert [o.summary for o in _read(google)] == ["One", "Two"]


def test_token_minted_once_and_answers_cached() -> None:
    google = Google([_event()])
    _read(google)
    _read(google)
    assert [u for m, u, f, h in google.calls].count(gcal.TOKEN_URL) == 1
    assert len(google.queries()) == 1  # second look inside CACHE_SECONDS reuses the answer
    gcal._EVENTS.clear()
    _read(google)
    assert [u for m, u, f, h in google.calls].count(gcal.TOKEN_URL) == 1  # token still good


def test_a_401_mints_a_fresh_token_once() -> None:
    google = Google([_event()])
    statuses = iter([401, 200])
    real = google.__call__

    def flaky(method, url, form, headers):
        if url != gcal.TOKEN_URL and next(statuses) == 401:
            google.calls.append((method, url, None, dict(headers)))
            return 401, b"{}"
        return real(method, url, form, headers)

    assert [o.summary for o in gcal.read(NOW, NOW + timedelta(hours=12), environ=ENV, send=flaky)] == ["Standup"]
    assert [u for m, u, f, h in google.calls].count(gcal.TOKEN_URL) == 2


# --- failures say why, and never say a secret ----------------------------------------


@pytest.mark.parametrize(
    "google, reason",
    [
        (Google(token_status=400, token_body={"error": "invalid_grant", "error_description": REFRESH}), "calendar-auth"),
        (Google(token_status=401, token_body={"error": "invalid_client"}), "OAuth client"),
        (Google(token_status=500, token_body={}), "status 500"),
        (Google(events_status=403, events_body={"error": {"errors": [{"reason": "accessNotConfigured"}], "message": ACCESS}}), "not enabled"),
        (Google(events_status=403, events_body={"error": {"errors": [{"reason": "forbidden"}], "message": REFRESH}}), "refused access (status 403)"),
        (Google(events_status=404), "no primary calendar"),
    ],
)
def test_failures_are_plain_reasons_without_secrets(google, reason) -> None:
    with pytest.raises(CalendarUnavailable) as caught:
        _read(google)
    text = str(caught.value)
    assert reason in text
    assert not any(s in text for s in SECRETS)
    assert caught.value.__cause__ is None


def test_failure_to_connect_names_no_secret(monkeypatch) -> None:
    import urllib.error

    def boom(*a, **k):
        raise urllib.error.URLError(OSError(f"no route {REFRESH}"))

    monkeypatch.setattr(urllib.request.OpenerDirector, "open", boom)
    with pytest.raises(CalendarUnavailable) as caught:
        _read(gcal.send)
    assert not any(s in str(caught.value) for s in SECRETS)


def test_credentials_hide_secrets_in_repr() -> None:
    text = repr(gcal.credentials(ENV))
    assert REFRESH not in text and CLIENT_SECRET not in text
    assert gcal.credentials({gcal.ENV_CLIENT_ID: "x", gcal.ENV_REFRESH_TOKEN: "y"}) is None


# --- one calendar, whichever way it was read -----------------------------------------


def test_read_window_merges_google_with_a_feed_and_dedupes() -> None:
    google = lambda a, b: [cal.Occurrence(NOW + timedelta(hours=1), NOW + timedelta(hours=2), False, "Same", "")]
    ics = (
        "BEGIN:VCALENDAR\nBEGIN:VEVENT\nUID:1\nDTSTART;TZID=Asia/Kolkata:20261012T090000\n"
        "DTEND;TZID=Asia/Kolkata:20261012T100000\nSUMMARY:Same\nEND:VEVENT\nEND:VCALENDAR\n"
    )
    window = cal.read_window(
        NOW, NOW + timedelta(hours=12), environ={cal.ENV: "https://x.test/a.ics"}, get=lambda u: ics, google=google
    )
    assert [o.summary for o in window.occurrences] == ["Same"]


def test_google_failure_beside_a_working_feed_is_an_error_not_a_crash() -> None:
    def google(a, b):
        raise CalendarUnavailable("Google refused access (status 403)")

    window = cal.read_window(
        NOW, NOW + timedelta(hours=1), environ={cal.ENV: "https://x.test/a.ics"}, get=lambda u: "BEGIN:VCALENDAR\nEND:VCALENDAR\n", google=google
    )
    assert window.errors == ("Google calendar: Google refused access (status 403)",)


def test_look_and_tool_treat_a_sign_in_as_connected(monkeypatch) -> None:
    google = Google([_event()])
    real = gcal.read
    monkeypatch.setattr(gcal, "read", lambda a, b, *, environ: real(a, b, environ=environ, send=google))
    look = cal.look_lines(NOW, environ=ENV)
    assert len(look) == 1 and "Standup" in look[0]
    assert "Standup" in cal.tool_text(0, NOW, environ=ENV)
    assert cal.connected(ENV) and not cal.connected({})
    assert cal.look_lines(NOW, environ={}) == []


def test_startup_line_names_the_sign_in_not_its_secrets() -> None:
    (calendar_line, _) = _account_lines(ENV)
    assert calendar_line == "calendar: Google sign-in (read-only)"
    both = _account_lines({**ENV, cal.ENV: "https://x.test/a.ics"})[0]
    assert both == "calendar: 1 feed + Google sign-in (read-only)"
    assert not any(s in calendar_line for s in SECRETS)


# --- signing in ----------------------------------------------------------------------


def _browser(answer):
    """Plays the person's browser: follows the auth URL straight back to the
    loopback redirect, as Google would after consent."""

    def open_browser(url):
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
        params = answer(query)
        target = query["redirect_uri"][0] + "/?" + urllib.parse.urlencode(params)
        assert target.startswith("http://127.0.0.1:")
        threading.Thread(target=lambda: urllib.request.urlopen(target, timeout=5).read(), daemon=True).start()

    return open_browser


def test_sign_in_round_trip_with_pkce_and_state() -> None:
    seen = {}

    def answer(query):
        seen.update(query)
        return {"code": "the-code", "state": query["state"][0]}

    def send(method, url, form, headers):
        assert url == gcal.TOKEN_URL
        seen["form"] = form
        return 200, json.dumps({"refresh_token": REFRESH, "scope": gcal.SCOPE, "access_token": ACCESS}).encode()

    token = gcal.authorize("cid", CLIENT_SECRET, open_browser=_browser(answer), send=send, say=lambda s: None, timeout=10)
    assert token == REFRESH
    assert seen["scope"] == [gcal.SCOPE] and seen["access_type"] == ["offline"]
    assert seen["code_challenge_method"] == ["S256"]
    import base64, hashlib

    verifier = seen["form"]["code_verifier"]
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    assert seen["code_challenge"] == [challenge]
    assert seen["form"]["code"] == "the-code"


def test_sign_in_refuses_a_mismatched_state() -> None:
    browser = _browser(lambda q: {"code": "c", "state": "forged"})
    with pytest.raises(CalendarUnavailable, match="state"):
        gcal.authorize("cid", "s", open_browser=browser, send=lambda *a: pytest.fail("no exchange"), say=lambda s: None, timeout=10)


def test_sign_in_refuses_when_calendar_box_was_unticked() -> None:
    browser = _browser(lambda q: {"code": "c", "state": q["state"][0]})
    send = lambda *a: (200, json.dumps({"refresh_token": REFRESH, "scope": "openid"}).encode())
    with pytest.raises(CalendarUnavailable, match="not granted"):
        gcal.authorize("cid", "s", open_browser=browser, send=send, say=lambda s: None, timeout=10)


def test_calendar_auth_command_prints_the_env_lines() -> None:
    out: list[str] = []
    asked: list[str] = []
    code = calendar_auth(
        [],
        environ={},
        ask=lambda p: asked.append(p) or "cid",
        ask_secret=lambda p: asked.append(p) or CLIENT_SECRET,
        say=out.append,
        authorize=lambda cid, secret, **k: REFRESH,
    )
    assert code == 0 and len(asked) == 2
    assert out[-3:] == [f"{gcal.ENV_CLIENT_ID}=cid", f"{gcal.ENV_CLIENT_SECRET}={CLIENT_SECRET}", f"{gcal.ENV_REFRESH_TOKEN}={REFRESH}"]


def test_calendar_auth_command_reports_failure_plainly() -> None:
    def fail(*a, **k):
        raise CalendarUnavailable("no answer from Google sign-in in time")

    out: list[str] = []
    assert calendar_auth([], environ=ENV, say=out.append, authorize=fail) == 1
    assert out == ["calendar-auth: no answer from Google sign-in in time"]


def test_occurrence_carries_the_uid_and_whether_it_is_a_series() -> None:
    one = gcal._occurrence(_event(iCalUID="a@example.test", id="x1"))
    inst = gcal._occurrence(_event(id="x2_20261012", recurringEventId="x2"))
    assert (one.uid, one.recurring) == ("a@example.test", False)
    assert (inst.uid, inst.recurring) == ("x2_20261012", True)


def test_a_moved_instance_says_when_it_was_due() -> None:
    """C2: ``originalStartTime`` is asked for, and a moved one is marked."""
    assert "originalStartTime" in gcal.FIELDS and "recurringEventId" in gcal.FIELDS
    moved = gcal._occurrence(_event(
        recurringEventId="x2", originalStartTime={"dateTime": "2026-10-12T09:00:00+05:30"}))
    kept = gcal._occurrence(_event(
        recurringEventId="x2", originalStartTime={"dateTime": "2026-10-12T09:30:00+05:30"}))
    assert moved is not None and moved.moved_from is not None
    assert moved.moved_from.isoformat() == "2026-10-12T09:00:00+05:30"
    assert kept is not None and kept.moved_from is None
