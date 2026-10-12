"""DL-091 — where the person is: what they said, their phone's Shortcut, and
the places they saved.

Fakes only. The one socket is the Shortcut listener's own, bound to
127.0.0.1 on port 0 and spoken to over loopback; no key, no provider.
"""

from __future__ import annotations

import http.client
import json
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from omega import episodes, location, projection, provider, tools, turn
from omega.act import act_loop
from omega.location import Listener, Locations
from omega.memory import MemoryStore
from omega.provider import ToolCall
from omega.queue import EventQueue
from omega.runtime import Runtime
from omega.turn import run_turn

NOW = datetime(2026, 10, 12, 9, 30, tzinfo=timezone.utc)
AT = NOW.isoformat()
TOKEN = "t0ken-for-tests-only-0123456789"


@pytest.fixture
def q(store: MemoryStore) -> EventQueue:
    return EventQueue(store)


def payloads(q: EventQueue) -> list[dict]:
    return [episodes.decode(e.payload) for e in q.store.episodes_since(0)]


def kinds(q: EventQueue, kind: str) -> list[dict]:
    return [p for p in payloads(q) if p.get("kind") == kind]


def call(tool: str, **arguments: Any) -> ToolCall:
    return ToolCall(id="c1", name=tool, arguments=dict(arguments))


def fix_at(minutes_ago: float, **fields: Any) -> dict[str, Any]:
    fields.setdefault("source", "shortcut")
    if "label" not in fields and "city" not in fields and "lat" not in fields:
        fields.update(lat=12.97159, lon=77.59456, accuracy_m=25.0)
    return episodes.location_set(at=(NOW - timedelta(minutes=minutes_ago)).isoformat(), **fields)


def hhmm(minutes_ago: float) -> str:
    return f"{(NOW - timedelta(minutes=minutes_ago)).astimezone():%H:%M}"


def turn_with(q: EventQueue, text: str, act: list[Any], *, judge: str = "ACT",
              channel: str = "tray", with_location: bool = True, **extra: Any):
    seq = q.append(episodes.inbound(text, channel=channel, at=AT, **extra))
    q.claim(seq)
    fp = provider.FakeProvider({provider.JUDGE: [judge], provider.ACT: list(act)})
    state = Locations().advance(q.store, upto=seq).state() if with_location else None
    result = run_turn(q, q.at(seq), complete=fp.complete, act=act_loop, location=state, at=AT)
    q.finish(seq)
    return result, fp


def offered(fp: provider.FakeProvider, n: int = 0) -> list[str]:
    return [t["function"]["name"] for t in fp.offers_for(provider.ACT)[n] or ()
            if "function" in t]


def results(fp: provider.FakeProvider) -> str:
    """What the tools answered, as the model was handed it."""
    return "\n".join(str(m["content"]) for call in fp.calls_for(provider.ACT)
                     for m in call if m.get("role") == "tool")


def judge_prompt(fp: provider.FakeProvider) -> str:
    return str(fp.calls_for(provider.JUDGE)[0][-1]["content"])


# --- the records -------------------------------------------------------------------


@pytest.mark.parametrize("fields", [
    {"source": "said"},  # nothing to say where
    {"source": "said", "lat": 91.0, "lon": 0.0},
    {"source": "said", "lat": 1.0, "lon": 181.0},
    {"source": "said", "lat": 1.0},  # half a coordinate
    {"source": "said", "lat": True, "lon": 1.0},
    {"source": "said", "lat": float("nan"), "lon": 1.0},
    {"source": "said", "label": "  "},
    {"source": "said", "label": "x" * 201},
    {"source": "said", "label": "a", "accuracy_m": -1},
    {"source": "elsewhere", "label": "a"},
    {"source": "saved", "label": "a"},  # a saved fix names its place
    {"source": "said", "label": "a", "place": "home"},
    {"source": "saved", "label": "a", "place": "Home"},  # not case-folded
])
def test_a_bad_fix_is_refused(fields: dict) -> None:
    with pytest.raises(episodes.BadPayload):
        episodes.location_set(at=AT, **fields)


def test_a_fix_needs_a_zoned_time_and_a_place_needs_somewhere() -> None:
    with pytest.raises(episodes.BadPayload):
        episodes.location_set(source="said", label="a", at="2026-10-12T09:00:00")
    with pytest.raises(episodes.BadPayload):
        episodes.place_saved(name="home", at=AT)
    with pytest.raises(episodes.BadPayload):
        episodes.place_forgotten(name=" home", at=AT)
    # A city alone is a rough fix, and is enough.
    rough = episodes.location_set(source="shortcut", city="Townsville", at=AT)
    assert episodes.decode(episodes.encode(rough)) == rough


def test_every_location_record_is_kept_out_of_recall_and_off_every_surface() -> None:
    assert episodes.LOCATION_KINDS <= turn.NOT_RECALLED
    assert episodes.LOCATION_KINDS <= projection.NOT_PROJECTED
    for payload in (fix_at(0), episodes.place_saved(name="home", label="x", at=AT),
                    episodes.place_forgotten(name="home", at=AT)):
        assert projection.project(payload, 1) is None


# --- the fold ----------------------------------------------------------------------


def test_the_current_location_is_the_latest_fix_and_history_is_kept(q: EventQueue) -> None:
    q.append(fix_at(300, source="said", label="the office"))
    q.append(fix_at(10))
    s = location.fold(q.store)
    assert s.current.source == "shortcut" and s.current.lat == 12.97159
    assert [f.label for f in s.fixes] == ["the office", None]
    assert s.at_or_before(NOW - timedelta(minutes=60)).label == "the office"
    assert s.at_or_before(NOW - timedelta(days=1)) is None


def test_places_are_saved_overwritten_and_forgotten(q: EventQueue) -> None:
    q.append(episodes.place_saved(name="home", label="12 Example Street", at=AT))
    q.append(episodes.place_saved(name="gym", lat=1.5, lon=2.5, at=AT))
    q.append(episodes.place_saved(name="home", label="34 Sample Road", at=AT))
    q.append(episodes.place_forgotten(name="gym", at=AT))
    s = location.fold(q.store)
    assert set(s.places) == {"home"} and s.places["home"].label == "34 Sample Road"


def test_it_survives_a_restart(store_dir) -> None:
    with MemoryStore.open(store_dir) as store:
        q = EventQueue(store)
        q.append(fix_at(5, source="said", label="the library"))
        q.append(episodes.place_saved(name="home", label="12 Example Street", at=AT))
        before = location.fold(store)
    with MemoryStore.open(store_dir) as store:
        after = location.fold(store)
    assert after == before and after.current.label == "the library"


def test_the_fold_is_in_order_and_incremental(q: EventQueue) -> None:
    a = q.append(fix_at(30, source="said", label="a"))
    b = q.append(fix_at(20, source="said", label="b"))
    view = Locations().advance(q.store, upto=a)
    assert view.state().current.label == "a"
    assert view.advance(q.store).state().current.label == "b" and view.through == b
    with pytest.raises(ValueError):
        view.apply(a, fix_at(0))


# --- the line a prompted turn sees ---------------------------------------------------


def test_a_fresh_fix_is_their_location_as_of_its_time() -> None:
    s = Locations.fold([(1, fix_at(40))]).state()
    assert location.context_line(s, AT) == (
        f"Their location: 12.97159,77.59456 ±25 m (as of {hhmm(40)}, via their phone).\n"
    )
    said = Locations.fold([(1, fix_at(5, source="said", label="the office", city="Townsville"))]).state()
    assert location.context_line(said, AT) == (
        f"Their location: the office, Townsville (as of {hhmm(5)}, via what they told you).\n"
    )


def test_a_stale_fix_is_only_ever_their_last_known_location() -> None:
    s = Locations.fold([(1, fix_at(3 * 60 + 1, source="said", label="the office"))]).state()
    line = location.context_line(s, AT)
    assert line.startswith("Their location now is unknown: last known the office at ")
    assert f"at {hhmm(181)}" in line and "3 hours ago" in line
    assert "Do not treat it as where they are now" in line
    assert "Their location:" not in line
    # Exactly three hours is still fresh; a minute later is not.
    edge = Locations.fold([(1, fix_at(180, source="said", label="x"))]).state()
    assert location.context_line(edge, AT).startswith("Their location: x")


def test_a_stale_fix_from_another_day_names_the_day() -> None:
    s = Locations.fold([(1, fix_at(60 * 30, source="said", label="the office"))]).state()
    local = (NOW - timedelta(hours=30)).astimezone()
    assert f"last known the office at {local:%H:%M} on {local:%a %d %b}" in location.context_line(s, AT)
    assert "30 hours ago" in location.context_line(s, AT)


def test_a_saved_place_reads_by_its_name() -> None:
    s = Locations.fold([(1, fix_at(1, source="saved", place="home", label="12 Example Street",
                                    lat=1.0, lon=2.0))]).state()
    assert location.context_line(s, AT).startswith(
        "Their location: home (12 Example Street) [1.00000,2.00000] (as of "
    )
    assert location.context_line(s, AT).endswith("via a place they saved).\n")


def test_their_own_message_sees_the_line_and_nothing_unprompted_does(q: EventQueue) -> None:
    q.append(fix_at(20, source="said", label="the office"))
    _, fp = turn_with(q, "find coffee near me", ["Try the corner cafe."])
    assert f"Their location: the office (as of {hhmm(20)}, via what they told you).\n" in judge_prompt(fp)
    act = str(fp.calls_for(provider.ACT)[0][-1]["content"])
    assert "Their location: the office" in act
    assert tools.LOCATION in offered(fp)
    # A brief, a watch, a look: none of them sees it or may call for it.
    for channel, extra in (("schedule", {"schedule_id": "brief"}), ("self", {})):
        _, fp = turn_with(q, "Daily brief you promised them: the day", ["Nothing."],
                          channel=channel, **extra)
        assert "the office" not in str(fp.calls)
        assert tools.LOCATION not in offered(fp)


def test_the_tool_refuses_on_an_unprompted_turn_even_if_named(q: EventQueue) -> None:
    q.append(fix_at(20, source="said", label="the office"))
    _, fp = turn_with(q, "Daily brief you promised them: the day",
                      [call(tools.LOCATION, action="where"), "Nothing."],
                      channel="schedule", schedule_id="brief")
    assert "the office" not in str(fp.calls)
    assert not kinds(q, episodes.TOOL_RETURNED) or all(
        "the office" not in str(p) for p in kinds(q, episodes.TOOL_RETURNED)
    )


def test_location_records_never_reach_recall(q: EventQueue) -> None:
    q.append(fix_at(20, source="said", label="Hidden Courtyard Cafe"))
    q.append(episodes.place_saved(name="secret spot", label="Hidden Courtyard Cafe", at=AT))
    _, fp = turn_with(q, "hello", ["Hi."], judge="SPEAK", with_location=False)
    assert "Hidden Courtyard" not in str(fp.calls)


# --- equivalence -------------------------------------------------------------------


def test_no_record_renders_and_offers_exactly_as_no_location_at_all(tmp_path) -> None:
    """Byte for byte: one turn with ``location=None`` and one with the empty
    fold render identical judge and act prompts and offers. The only change a
    log with no fix sees is the ``location`` tool offered on their own turn."""
    rendered = []
    for i, arg in enumerate((None, location.State(), "no kwarg")):
        d = tmp_path / f"s{i}"
        d.mkdir()
        with MemoryStore.open(d) as store:
            q = EventQueue(store)
            seq = q.append(episodes.inbound("what's on tuesday", channel="tray", at=AT))
            q.claim(seq)
            fp = provider.FakeProvider({provider.JUDGE: ["ACT"], provider.ACT: ["Nothing."]})
            kwargs = {} if arg == "no kwarg" else {"location": arg}
            run_turn(q, q.at(seq), complete=fp.complete, act=act_loop, at=AT, **kwargs)
            rendered.append(repr((fp.calls, fp.offers)).replace(str(d.resolve()), "<store>"))
    assert rendered[0] == rendered[1] == rendered[2]


def test_with_nothing_recorded_the_tool_is_offered_on_every_turn_of_theirs(q: EventQueue) -> None:
    from omega.tools import ToolBox, hidden, hosted, schemas

    # Whether a question needs where they are is the model's to judge: no
    # word list decides the offer, so any phrasing of theirs is offered it.
    _, plain = turn_with(q, "what's on tuesday", ["Nothing."])
    assert tools.LOCATION in offered(plain)
    # DL-094: every act turn is now also offered `look_back`, past location.
    assert [s for s in plain.offers_for(provider.ACT)[0]
            if s.get("function", {}).get("name") not in (tools.LOCATION, tools.TOPICS)] == schemas(
        hidden(ToolBox(store_root=q.store.root, prompted=True))) + [
        ToolBox(store_root=q.store.root, history=list).schemas()[-1]] + hosted()
    for words in ("find coffee near me", "wher am i?", "whre am i", "am I near the office?",
                  "which city is this", "location?", "hello"):
        _, fp = turn_with(q, words, ["Hmm."])
        assert tools.LOCATION in offered(fp), words
    q.append(fix_at(1))
    _, after = turn_with(q, "what's on tuesday", ["Nothing."])
    assert tools.LOCATION in offered(after)
    # Still never on a turn they did not start.
    _, brief = turn_with(q, "Daily brief you promised them: the day", ["Nothing."],
                         channel="schedule", schedule_id="brief")
    assert tools.LOCATION not in offered(brief)


def test_where_with_no_fix_says_how_to_give_one_and_dates_a_recalled_place(q: EventQueue) -> None:
    _, fp = turn_with(q, "wher am i?", [call(tools.LOCATION, action="where"), "Unsure."])
    answer = results(fp)
    assert "no location recorded" in answer
    assert "I'm at <place>" in answer and "home is <address>" in answer
    assert "iPhone Shortcut" in answer and "as of" in answer and "not where they are now" in answer
    q.append(fix_at(600, source="said", label="the office"))
    _, fp = turn_with(q, "where am I", [call(tools.LOCATION, action="where"), "Unsure."])
    stale = results(fp)
    assert "last known the office" in stale and "iPhone Shortcut" in stale
    assert "give it only with its date" in stale
    q.append(fix_at(5, source="said", label="the station"))
    _, fp = turn_with(q, "where am I", [call(tools.LOCATION, action="where"), "The station."])
    assert "their location: the station" in results(fp) and "iPhone Shortcut" not in results(fp)


def test_the_tool_tells_the_model_to_look_before_answering_where() -> None:
    box = tools.ToolBox(store_root=".", prompted=True, append=lambda p: 1,
                        locations=lambda: location.State())
    words = next(s for s in box.schemas()
                 if s["function"]["name"] == tools.LOCATION)["function"]["description"]
    assert "Call 'where' before answering" in words
    assert "never give a place recalled from conversation as current" in words


def test_the_section_is_empty_without_a_fix() -> None:
    ctx = turn.TurnContext(seq=1, event=episodes.inbound("hi", channel="tray", at=AT),
                           recalled=(), queue=None, complete=None)  # type: ignore[arg-type]
    for state in (None, location.State(),
                  Locations.fold([(1, episodes.place_saved(name="home", label="x", at=AT))]).state()):
        assert turn._location_section(turn.TurnContext(**{**ctx.__dict__, "location": state})) == ""


def test_no_token_means_no_listener_and_no_startup_line(store_dir) -> None:
    from omega.__main__ import _location_args, _location_lines

    assert _location_args({}) == {} and _location_lines(None, {}) == []  # type: ignore[arg-type]
    assert _location_args({location.ENV_PORT: "7718"}) == {}
    rt = Runtime(store_dir, complete=provider.FakeProvider({}).complete,
                 listen=False, clock=False)
    rt.start()
    try:
        assert rt.location_address is None and rt.location_error == ""
    finally:
        rt.stop()
    with pytest.raises(ValueError):
        Listener("", lambda p: 1)


# --- the tool ----------------------------------------------------------------------


def test_here_notes_what_they_said(q: EventQueue) -> None:
    turn_with(q, "I'm at the Example Library",
              [call(tools.LOCATION, action="here", text="the Example Library"), "Noted."])
    [fix] = kinds(q, episodes.LOCATION_SET)
    assert (fix["source"], fix["label"], fix["lat"], fix["place"]) == ("said", "the Example Library", None, None)
    returned = kinds(q, episodes.TOOL_RETURNED)[-1]
    assert "their location: the Example Library" in str(returned)


def test_here_names_a_saved_place_or_takes_coordinates(q: EventQueue) -> None:
    q.append(episodes.place_saved(name="home", label="12 Example Street", lat=1.0, lon=2.0, at=AT))
    turn_with(q, "I'm home", [call(tools.LOCATION, action="here", text="Home"), "Noted."])
    turn_with(q, "I'm at 12.5, 77.25", [call(tools.LOCATION, action="here", text="12.5, 77.25"), "Noted."])
    saved, coords = kinds(q, episodes.LOCATION_SET)
    assert (saved["source"], saved["place"], saved["lat"], saved["label"]) == ("saved", "home", 1.0, "12 Example Street")
    assert (coords["source"], coords["lat"], coords["lon"], coords["label"]) == ("said", 12.5, 77.25, None)


def test_save_names_where_they_are_or_what_they_said_and_forget_drops_it(q: EventQueue) -> None:
    q.append(fix_at(5, lat=10.0, lon=20.0))
    turn_with(q, "save this as the gym", [call(tools.LOCATION, action="save", name="The Gym"), "Saved."])
    turn_with(q, "home is 12 Example Street",
              [call(tools.LOCATION, action="save", name="home", text="12 Example Street"), "Saved."])
    gym, home = kinds(q, episodes.PLACE_SAVED)
    assert (gym["name"], gym["lat"], gym["lon"]) == ("the gym", 10.0, 20.0)
    assert (home["name"], home["label"]) == ("home", "12 Example Street")
    _, fp = turn_with(q, "what places do you know", [call(tools.LOCATION, action="places"), "Two."])
    listed = str(fp.calls_for(provider.ACT)[-1])
    assert "home: 12 Example Street" in listed and "the gym: 10.00000,20.00000" in listed
    turn_with(q, "forget the gym", [call(tools.LOCATION, action="forget", name="the gym"), "Done."])
    assert set(location.fold(q.store).places) == {"home"}


def test_a_refusal_never_echoes_where_or_which_place(q: EventQueue) -> None:
    _, fp = turn_with(q, "forget my saved place called secret spot",
                      [call(tools.LOCATION, action="forget", name="Secret Spot Seven"), "Hmm."])
    assert "no saved place has that name" in results(fp)
    assert not kinds(q, episodes.PLACE_FORGOTTEN)
    for p in kinds(q, episodes.TOOL_RETURNED) + kinds(q, episodes.TOOL_CALLED):
        assert "Secret Spot Seven" not in str(p.get("error", "")) + str(p.get("why", ""))
    assert "Secret Spot Seven" not in results(fp)
    _, fp = turn_with(q, "where am I", [call(tools.LOCATION, action="teleport", text="Narnia Gate"), "Hmm."])
    assert "'action' is one of" in results(fp) and "Narnia Gate" not in results(fp)
    _, fp = turn_with(q, "where was I", [call(tools.LOCATION, action="where", when="Narnia Gate"), "Hmm."])
    assert "could not read 'when'" in results(fp) and "Narnia Gate" not in results(fp)


def test_where_reads_now_and_as_of_a_past_time(q: EventQueue) -> None:
    q.append(fix_at(300, source="said", label="the office"))
    q.append(fix_at(30, source="said", label="the station"))
    _, fp = turn_with(q, "where am I", [call(tools.LOCATION, action="where"), "The station."])
    assert f"their location: the station, as of {hhmm(30)}" in str(fp.calls_for(provider.ACT)[-1])
    _, fp = turn_with(q, "where was I 2 hours ago",
                      [call(tools.LOCATION, action="where", when="2 hours ago"), "The office."])
    answer = str(fp.calls_for(provider.ACT)[-1])
    assert "the latest location recorded was the office" in answer and "3 hours before then" in answer
    _, fp = turn_with(q, "where was I last month",
                      [call(tools.LOCATION, action="where", when="whenever-ish"), "?"])
    assert "could not read 'when'" in str(fp.calls_for(provider.ACT)[-1])


def test_where_says_a_stale_fix_is_not_now(q: EventQueue) -> None:
    q.append(fix_at(600, source="said", label="the office"))
    _, fp = turn_with(q, "where am I", [call(tools.LOCATION, action="where"), "Unsure."])
    answer = str(fp.calls_for(provider.ACT)[-1])
    assert "not known now; last known the office" in answer and "not where they are now" in answer


def test_history_lists_the_last_days_newest_first(q: EventQueue) -> None:
    q.append(fix_at(60 * 50, source="said", label="two days back"))
    q.append(fix_at(60 * 5, source="said", label="this morning"))
    q.append(fix_at(10, source="said", label="just now"))
    _, fp = turn_with(q, "where have I been", [call(tools.LOCATION, action="history"), "Around."])
    answer = str(fp.calls_for(provider.ACT)[-1])
    assert answer.index("just now") < answer.index("this morning") and "two days back" not in answer
    s = location.fold(q.store)
    assert "two days back" in location.history_text(s, NOW, 3)
    assert "no location recorded in the last 1 day" == location.history_text(location.State(), NOW, 1)


def test_location_is_outside_ring_one_and_owns_its_writes() -> None:
    from omega import grants, lessons

    assert tools.LOCATION not in tools.TOOL_NAMES
    assert grants.REVERSIBILITY[tools.LOCATION] == grants.OWN and not grants.is_money(tools.LOCATION)
    assert tools.LOCATION not in {s["function"]["name"] for s in tools.schemas()}
    box = tools.ToolBox(store_root=".", prompted=True, append=lambda p: 1)
    assert tools.LOCATION not in {s["function"]["name"] for s in box.schemas()}
    with pytest.raises(tools.ToolRejected):
        box.classify(tools.LOCATION, {"action": "where"})
    del lessons


@pytest.mark.parametrize("words,expected", [
    ("2 hours ago", NOW - timedelta(hours=2)),
    ("45 minutes ago", NOW - timedelta(minutes=45)),
    ("3 days ago", NOW - timedelta(days=3)),
    ("2026-10-12T08:00:00+00:00", datetime(2026, 10, 12, 8, 0, tzinfo=timezone.utc)),
])
def test_when_reads_the_usual_ways_of_saying_it(words: str, expected: datetime) -> None:
    assert location.parse_when(words, NOW) == expected


def test_when_reads_clock_times_in_their_zone() -> None:
    local = NOW.astimezone()
    earlier = (local - timedelta(minutes=30))
    got = location.parse_when(f"{earlier:%H:%M}", NOW)
    assert got.astimezone().date() == earlier.date() and f"{got.astimezone():%H:%M}" == f"{earlier:%H:%M}"
    later = (local + timedelta(minutes=30))
    ahead = location.parse_when(f"{later:%H:%M}", NOW)
    assert ahead < NOW  # a time still ahead today means yesterday's
    yday = location.parse_when("yesterday 18:00", NOW).astimezone()
    assert yday.date() == local.date() - timedelta(days=1) and (yday.hour, yday.minute) == (18, 0)
    assert location.parse_when("yesterday 25:00", NOW) is None
    assert location.parse_when("soonish", NOW) is None


# --- the Shortcut's door -----------------------------------------------------------


class Door:
    """A real listener on 127.0.0.1:0 with a fake clock and a recording append."""

    def __init__(self, token: str = TOKEN) -> None:
        self.appended: list[dict] = []
        self.now = 0.0
        self.listener = Listener(token, self.appended.append, port=0, clock=lambda: self.now).start()

    def post(self, body: Any, *, auth: str | None = f"Bearer {TOKEN}", path: str = "/location",
             method: str = "POST", raw: bytes | None = None) -> tuple[int, dict]:
        host, port = self.listener.address
        conn = http.client.HTTPConnection(host, port, timeout=5)
        data = raw if raw is not None else json.dumps(body).encode()
        headers = {"Content-Type": "application/json"}
        if auth is not None:
            headers["Authorization"] = auth
        conn.request(method, path, body=data if method == "POST" else None, headers=headers)
        response = conn.getresponse()
        reply = response.read()
        conn.close()
        return response.status, (json.loads(reply) if reply else {})


@pytest.fixture
def door():
    d = Door()
    yield d
    d.listener.stop()


def test_a_good_fix_is_appended_as_from_their_phone(door: Door) -> None:
    status, reply = door.post({"lat": 12.5, "lon": "77.25", "accuracy": 30, "label": "Work"})
    assert (status, reply) == (200, {"ok": True})
    [fix] = door.appended
    assert (fix["kind"], fix["source"], fix["lat"], fix["lon"], fix["accuracy_m"], fix["label"]) == (
        episodes.LOCATION_SET, "shortcut", 12.5, 77.25, 30.0, "Work")
    assert door.listener.address[0] == "127.0.0.1"


def test_a_city_alone_is_a_rough_fix(door: Door) -> None:
    assert door.post({"city": "Townsville", "lat": "", "lon": ""})[0] == 200
    [fix] = door.appended
    assert (fix["city"], fix["lat"], fix["lon"]) == ("Townsville", None, None)


@pytest.mark.parametrize("auth", [None, "", f"Bearer {TOKEN}x", f"bearer {TOKEN}", TOKEN,
                                  f"Basic {TOKEN}", "Bearer " + TOKEN[:-1]])
def test_without_the_token_nothing_is_written(door: Door, auth: str | None) -> None:
    status, reply = door.post({"lat": 1, "lon": 2}, auth=auth)
    assert status == 401 and reply == {"error": "unauthorized"}
    assert door.appended == []


@pytest.mark.parametrize("body", [
    {"lat": 91, "lon": 0}, {"lat": 0, "lon": -181}, {"lat": 1}, {"label": "just a label"},
    {"lat": 1, "lon": 2, "extra": 1}, {"lat": "north", "lon": 2}, {"lat": True, "lon": 2},
    {"lat": 1, "lon": 2, "accuracy": -5}, {"city": "x" * 101}, [1, 2], "hello",
])
def test_a_bad_body_is_a_400_and_writes_nothing(door: Door, body: Any) -> None:
    status, reply = door.post(body)
    assert status == 400 and "error" in reply
    assert door.appended == []


def test_bad_json_an_oversized_body_and_wrong_routes_are_refused(door: Door) -> None:
    assert door.post(None, raw=b"{not json")[0] == 400
    assert door.post(None, raw=b'{"lat": 1, "lon": 2, "label": "' + b"x" * 5000 + b'"}')[0] == 413
    assert door.post({"lat": 1, "lon": 2}, path="/other")[0] == 404
    assert door.post({"lat": 1, "lon": 2}, method="GET")[0] == 405
    assert door.appended == []


def test_a_body_without_its_length_is_refused(door: Door) -> None:
    host, port = door.listener.address
    conn = http.client.HTTPConnection(host, port, timeout=5)
    conn.putrequest("POST", "/location")
    conn.putheader("Authorization", f"Bearer {TOKEN}")
    conn.endheaders()
    assert conn.getresponse().status == 411
    conn.close()
    assert door.appended == []


def test_it_is_rate_limited_and_the_window_slides(door: Door) -> None:
    for _ in range(location.RATE_LIMIT):
        assert door.post({"lat": 1, "lon": 2}, auth="Bearer wrong")[0] == 401
    assert door.post({"lat": 1, "lon": 2})[0] == 429  # even with the right token
    door.now += location.RATE_WINDOW
    assert door.post({"lat": 1, "lon": 2})[0] == 200
    assert len(door.appended) == 1


def test_nothing_echoes_the_token_or_the_fix(door: Door, capsys) -> None:
    status, reply = door.post({"lat": 12.5, "lon": 77.25, "label": "Hidden Courtyard"})
    bad = door.post({"lat": 12.5, "lon": 77.25, "label": "Hidden Courtyard", "x": TOKEN})
    out = capsys.readouterr()
    assert TOKEN not in json.dumps([reply, bad[1]]) + out.out + out.err
    assert "Hidden Courtyard" not in json.dumps([reply, bad[1]]) + out.out + out.err


def test_settings_read_the_token_and_port() -> None:
    assert not location.settings({}).wanted
    assert not location.settings({location.ENV_TOKEN: "   "}).wanted
    short = location.settings({location.ENV_TOKEN: "abc123"})
    assert short.wanted and not short.startable and "shorter than" in short.problem
    assert "abc123" not in short.problem
    assert not location.settings({location.ENV_TOKEN: TOKEN, location.ENV_PORT: "0"}).startable
    bad = location.settings({location.ENV_TOKEN: TOKEN, location.ENV_PORT: "port"})
    assert not bad.startable and location.ENV_PORT in bad.problem
    good = location.settings({location.ENV_TOKEN: TOKEN, location.ENV_PORT: "7799"})
    assert good.startable and good.port == 7799
    assert location.settings({location.ENV_TOKEN: TOKEN}).port == location.DEFAULT_PORT == 7718


def test_the_runtime_takes_a_fix_through_its_own_writer_and_says_where(store_dir) -> None:
    from omega.__main__ import _location_args, _location_lines

    env = {location.ENV_TOKEN: TOKEN, location.ENV_PORT: "7799"}
    assert _location_args(env) == {"location_token": TOKEN, "location_port": 7799}
    rt = Runtime(store_dir, complete=provider.FakeProvider({}).complete,
                 listen=False, clock=False, location_token=TOKEN, location_port=0)
    rt.start()
    try:
        host, port = rt.location_address
        [line] = _location_lines(rt, env)
        assert line == f"location: listening for the phone on 127.0.0.1:{port} (POST /location)"
        assert TOKEN not in line
        conn = http.client.HTTPConnection(host, port, timeout=5)
        conn.request("POST", "/location", body=json.dumps({"lat": 1.5, "lon": 2.5}),
                     headers={"Authorization": f"Bearer {TOKEN}"})
        assert conn.getresponse().status == 200
        conn.close()
        assert location.fold(rt.queue.store).current.lat == 1.5
    finally:
        rt.stop()
    assert rt.location_address is None
    with MemoryStore.open(store_dir) as store:
        assert location.fold(store).current.source == "shortcut"
    assert _location_lines(None, {location.ENV_TOKEN: "short"}) == [  # type: ignore[arg-type]
        f"location: not listening ({location.ENV_TOKEN} is shorter than {location.MIN_TOKEN} characters)"
    ]


def test_the_executor_hands_each_turn_the_fold_as_of_its_event(q: EventQueue) -> None:
    from omega.executor import Executor

    q.append(fix_at(15, source="said", label="the office"))
    first = q.append(episodes.inbound("is there a pharmacy nearby", channel="tray", at=AT))
    # Arrived after their message: not where they were when they asked.
    q.append(fix_at(-1, source="said", label="the airport"))
    q.append(episodes.inbound("thanks", channel="tray", at=AT))
    fp = provider.FakeProvider({provider.JUDGE: ["SPEAK", "SPEAK"], provider.ACT: ["One.", "Sure."]})
    ex = Executor(q, complete=fp.complete, act=act_loop)
    ex.recover()
    ex.drain()
    one, two = (str(c[-1]["content"]) for c in fp.calls_for(provider.JUDGE))
    assert "Their location: the office" in one and "the airport" not in one
    assert "Their location: the airport" in two
    assert first
