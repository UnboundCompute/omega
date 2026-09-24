"""Reading what the Mac already wrote down — DL-059.

Four groups.

The **discovery** cases are refusals, and they are the ones that protect
history. Today is never offered, a day already digested is never offered twice,
and — the one that matters most — a database that cannot be read produces *no
days at all* rather than a run of failed ones. Without Full Disk Access the
second-best behaviour would be to write a failure receipt per day, and every
receipt is permanent, so the whole lookback would be marked read with nothing
read and granting the permission afterwards would recover none of it.

The **digest** cases pin the arithmetic and the floor. Overlapping intervals
are the trap: macOS records an app as in use while another is in focus, so
summing gives a twenty-six-hour day, and the model would then be reasoning from
a false premise rather than from a rounding error.

The **privacy** cases are this file's version of the transcript suite's
attacker group. ``knowledgeC`` carries window titles and document names, and a
window title is attacker-controlled the moment a page sets one. The check is
that none of it can reach a digest, because the query names its columns.

The **end-to-end** cases are the pair `CLAUDE.md` asks for. The capability: a
working day files what it showed. The violation, which must not regress: an
ordinary quiet day files nothing and still leaves a receipt — because a day
that taught nothing and left no receipt is a day the next pass reads again.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from omega import episodes, habits, provider
from omega.executor import Executor
from omega.queue import EventQueue

#: Local noon on a day whose *previous* day is the one the fixtures fill in.
#: Naive and local on purpose: the unit under test is a local calendar date, so
#: a UTC constant here would make the suite pass or fail by time zone.
NOW = datetime(2026, 9, 24, 12, 0, 0)
YESTERDAY = date(2026, 9, 23)

EDITOR = "com.microsoft.VSCode"
TERMINAL = "com.googlecode.iterm2"
BROWSER = "com.apple.Safari"
CHAT = "com.tinyspeck.slackmacgap"


# --- building a knowledgeC-shaped database -----------------------------------


def _cocoa(day: date, hour: float) -> float:
    """A local wall-clock hour on ``day``, in the Cocoa seconds the real
    database stores."""
    base = datetime(day.year, day.month, day.day).timestamp()
    return base + hour * 3600 - habits.APPLE_EPOCH


def _every_day() -> list[date]:
    """The whole discoverable window.

    The end-to-end cases need this and it is not padding: a pass digests the
    *oldest* undigested days, so a fixture holding only yesterday would run
    every one of them over a day with no rows — and they would pass, having
    exercised nothing.
    """
    today = NOW.date()
    return [today - timedelta(days=n) for n in range(habits.LOOKBACK_DAYS, 0, -1)]


def _db(
    path: Path,
    rows: list[tuple[str, float, float]],
    *,
    days: list[date] = [YESTERDAY],
    stream: str = habits.STREAM,
    extra_columns: bool = True,
) -> Path:
    """A database with ``ZOBJECT`` shaped the way the real one is.

    ``extra_columns`` puts the free-text columns in by default — the title and
    document fields this reader must never touch. A fixture without them could
    not fail the privacy cases, which would make them pass for the wrong reason.
    """
    conn = sqlite3.connect(path)
    columns = [
        "Z_PK INTEGER PRIMARY KEY",
        "ZSTREAMNAME TEXT",
        "ZVALUESTRING TEXT",
        "ZSTARTDATE REAL",
        "ZENDDATE REAL",
    ]
    if extra_columns:
        columns += ["ZTITLE TEXT", "ZDOCUMENTNAME TEXT"]
    conn.execute(f"CREATE TABLE ZOBJECT ({', '.join(columns)})")
    index = 0
    for day in days:
        for bundle, start, end in rows:
            index += 1
            values = [index, stream, bundle, _cocoa(day, start), _cocoa(day, end)]
            if extra_columns:
                values += ["Re: Q3 budget — Mail", "/Users/me/Documents/salary.pdf"]
            conn.execute(
                "INSERT INTO ZOBJECT VALUES (" + ",".join("?" * len(values)) + ")",
                values,
            )
    conn.commit()
    conn.close()
    return path


def _working_day() -> list[tuple[str, float, float]]:
    """Nine to six, with the shape a real day has: overlapping records, a
    morning in the editor, an afternoon that drifts to the browser."""
    return [
        (EDITOR, 9.0, 12.5),
        (TERMINAL, 9.25, 12.0),  # overlaps the editor throughout
        (BROWSER, 10.0, 10.5),
        (CHAT, 12.5, 13.0),
        (EDITOR, 13.0, 16.0),
        (BROWSER, 16.0, 18.0),
    ]


def _day(path: Path, day: date = YESTERDAY) -> habits.Day:
    return habits.Day(date=day, path=path)


# --- discovery ----------------------------------------------------------------


def test_today_is_never_offered_because_it_is_still_being_written(
    tmp_path: Path,
) -> None:
    """The reason a session must be idle, in its calendar form. A day in
    progress is a window still growing, and digesting it would file a claim
    that the rest of the day contradicts."""
    path = _db(tmp_path / "k.db", _working_day())

    found = habits.discover(path, now=NOW, seen=())

    assert found, "a month of finished days should be discoverable"
    assert NOW.date().isoformat() not in {d.id for d in found}
    assert found[-1].id == YESTERDAY.isoformat()


def test_days_come_back_oldest_first(tmp_path: Path) -> None:
    """So a later day can supersede an earlier claim rather than contradict it
    — the same ordering argument the transcript reader makes."""
    path = _db(tmp_path / "k.db", _working_day())

    ids = [d.id for d in habits.discover(path, now=NOW, seen=())]

    assert ids == sorted(ids)
    assert len(ids) == habits.LOOKBACK_DAYS


def test_a_day_with_a_receipt_is_not_offered_again(tmp_path: Path) -> None:
    path = _db(tmp_path / "k.db", _working_day())

    found = habits.discover(path, now=NOW, seen={YESTERDAY.isoformat()})

    assert YESTERDAY.isoformat() not in {d.id for d in found}


def test_an_unreadable_database_yields_no_days_rather_than_failed_ones(
    tmp_path: Path,
) -> None:
    """The case that protects the backlog, and the reason the permission check
    is in `discover` rather than in `digest`.

    Without Full Disk Access every read fails. If that produced days, each one
    would digest to a failure, each failure would write a receipt, and receipts
    are permanent — so thirty days would be marked read having never been read,
    and granting the permission afterwards would recover none of them. A failure
    to *look* must not be recorded as having looked.
    """
    missing = tmp_path / "nowhere" / "knowledgeC.db"

    assert habits.discover(missing, now=NOW, seen=()) == []

    not_a_database = tmp_path / "k.db"
    not_a_database.write_text("permission denied, probably")
    assert habits.discover(not_a_database, now=NOW, seen=()) == []


# --- the digest ---------------------------------------------------------------


def test_a_working_day_renders_its_shape(tmp_path: Path) -> None:
    path = _db(tmp_path / "k.db", _working_day())

    rendered = habits.digest(_day(path))

    assert rendered is not None
    assert "2026-09-23 (Wednesday)" in rendered
    assert "from 09:00 to 18:00" in rendered
    # Bundle ids, and the busiest one first.
    assert EDITOR in rendered and TERMINAL in rendered
    assert rendered.index(EDITOR) < rendered.index(CHAT)
    # The sequence, not only the totals: an editor morning and a browser
    # evening have the same totals as the reverse and mean different things.
    assert "09:00  " + EDITOR in rendered
    assert "17:00  " + BROWSER in rendered


def test_overlapping_records_do_not_add_up_to_a_twenty_six_hour_day(
    tmp_path: Path,
) -> None:
    """The arithmetic trap. macOS records an app as in use while another is in
    focus, so a sum reports more hours than the day has — and the model is then
    reasoning from a false premise, not from a rounding error."""
    rows = [(EDITOR, 9.0, 17.0), (TERMINAL, 9.0, 17.0), (BROWSER, 9.0, 17.0)]
    path = _db(tmp_path / "k.db", rows)

    rendered = habits.digest(_day(path))

    assert rendered is not None
    assert "Recorded app use: 8h00m" in rendered
    assert "24h" not in rendered


def test_a_quiet_day_digests_to_nothing(tmp_path: Path) -> None:
    """Fail closed on empty, and the violation metric this sense ships with.

    Ten minutes of three apps is a laptop that was opened and shut. Handing it
    to a model that was asked what it shows about someone's habits is asking it
    to invent one, which is the firehose arriving through a new door.
    """
    rows = [(EDITOR, 9.0, 9.1), (TERMINAL, 9.1, 9.2), (BROWSER, 9.2, 9.3)]
    path = _db(tmp_path / "k.db", rows)

    assert habits.digest(_day(path)) is None


def test_a_long_day_of_one_app_is_still_nothing(tmp_path: Path) -> None:
    """The other shape of an empty day: four hours of one screensaver is a long
    time and no information."""
    path = _db(tmp_path / "k.db", [("com.apple.ScreenSaver.Engine", 9.0, 13.0)])

    assert habits.digest(_day(path)) is None


def test_a_day_with_no_rows_at_all_digests_to_nothing(tmp_path: Path) -> None:
    path = _db(tmp_path / "k.db", _working_day(), days=[YESTERDAY - timedelta(days=5)])

    assert habits.digest(_day(path)) is None


def test_the_other_streams_are_not_read(tmp_path: Path) -> None:
    """`ZOBJECT` carries notifications, media playback and Safari history in
    the same table. App usage is the narrowest stream that answers *when do
    they work*, and it is the only one this asks for."""
    path = _db(tmp_path / "k.db", _working_day(), stream="/safari/history")

    assert habits.digest(_day(path)) is None


# --- what must not reach a model ----------------------------------------------


def test_window_titles_and_document_names_cannot_reach_a_digest(
    tmp_path: Path,
) -> None:
    """The structural defence, and the reason the column list in `_rows` is a
    security boundary rather than a style choice.

    A window title is attacker-controlled the moment a web page sets one, and a
    document name is the person's private business. Neither is filtered out of
    the digest — neither is ever read.
    """
    path = _db(tmp_path / "k.db", _working_day())

    rendered = habits.digest(_day(path))

    assert rendered is not None
    assert "budget" not in rendered
    assert "salary" not in rendered
    assert "Documents" not in rendered


def test_the_live_database_is_never_opened_in_place(tmp_path: Path) -> None:
    """Read from a copy. The real file is open and journalling under a process
    we do not control, so this removes the whole class of *the file moved under
    us* — and means omega never holds a handle on a file the system is writing.
    """
    path = _db(tmp_path / "k.db", _working_day())
    opened: list[str] = []
    real = sqlite3.connect

    def watched(target, *args, **kwargs):
        opened.append(str(target))
        return real(target, *args, **kwargs)

    sqlite3.connect = watched
    try:
        assert habits.digest(_day(path)) is not None
    finally:
        sqlite3.connect = real

    assert opened, "nothing was opened at all"
    assert str(path) not in opened


# --- end to end ---------------------------------------------------------------


def _claims(text: str) -> str:
    return json.dumps(
        {"claims": [{"text": text, "situation": "when planning their day"}]}
    )


def _counting_reader() -> provider.FakeProvider:
    """A reader whose answer names which call it is.

    One claim per day is the thing worth asserting — a pass that read two days
    and made one model call, or made two and filed one, both look like "it
    worked" against a constant answer.
    """
    calls = [0]

    def answer(role, messages):
        calls[0] += 1
        return _claims(f"They start around nine (day {calls[0]}).")

    return provider.FakeProvider({provider.LEARN: answer})


def _ready(store) -> tuple[EventQueue, Executor, provider.FakeProvider]:
    """An executor over a log omega has already been spoken through.

    Not scenery: a claim names the episode it came from, so the pass refuses to
    run against an empty log — the first thing in it must not be a belief about
    a person omega has not met.
    """
    q = EventQueue(store)
    q.append(episodes.inbound("morning", channel="tray"))
    fp = _counting_reader()
    ex = Executor(q, complete=fp.complete)
    ex.recover()
    return q, ex, fp


def _written(store, kind: str) -> list[dict]:
    """Every episode of ``kind``, read back from the bytes — *grade the world,
    not the words*."""
    decoded = (episodes.decode(e.payload) for e in store.episodes_since(0))
    return [p for p in decoded if p["kind"] == kind]


def test_a_working_day_files_what_it_showed(store, tmp_path: Path) -> None:
    path = _db(tmp_path / "k.db", _working_day(), days=_every_day())
    q, ex, fp = _ready(store)

    assert ex.digest_usage(path=path, now=NOW) == habits.MAX_DAYS_PER_PASS

    filed = _written(store, episodes.CLAIM_EXTRACTED)
    # One model call per day, and each one's claim filed.
    assert [c["text"] for c in filed] == [
        "They start around nine (day 1).",
        "They start around nine (day 2).",
    ]
    # Inferred, never typed. DL-042: a claim omega concluded from a table of
    # app names must not carry the weight of a sentence the person wrote.
    assert all(c["explicit"] is False for c in filed)


def test_a_day_that_taught_nothing_still_leaves_a_receipt(
    store, tmp_path: Path
) -> None:
    """The violation metric. The receipt *is* the cursor (DL-036), and a quiet
    day is the common outcome — so if silence left no receipt, the next pass
    would read the same day again, and the one after that."""
    quiet = [(EDITOR, 9.0, 9.1), (TERMINAL, 9.1, 9.2), (BROWSER, 9.2, 9.3)]
    path = _db(tmp_path / "k.db", quiet, days=_every_day())
    q, ex, fp = _ready(store)

    ex.digest_usage(path=path, now=NOW)

    assert _written(store, episodes.CLAIM_EXTRACTED) == []
    receipts = _written(store, episodes.USAGE_DIGESTED)
    assert len(receipts) == habits.MAX_DAYS_PER_PASS
    assert all(r["filed"] == 0 and r["reason"] is None for r in receipts)


def test_a_digested_day_is_never_read_twice(store, tmp_path: Path) -> None:
    path = _db(tmp_path / "k.db", _working_day(), days=_every_day())
    q, ex, fp = _ready(store)

    ex.digest_usage(path=path, now=NOW)
    first = {r["day"] for r in _written(store, episodes.USAGE_DIGESTED)}
    ex.digest_usage(path=path, now=NOW)
    second = {r["day"] for r in _written(store, episodes.USAGE_DIGESTED)}

    assert first < second, "the second pass should have moved on to new days"
    assert len(second) == 2 * habits.MAX_DAYS_PER_PASS


def test_a_pass_reads_at_most_its_cap(store, tmp_path: Path) -> None:
    """The cost bound. Each day is one model call, and a laptop opened after a
    month away would otherwise pay for thirty in the first idle moment."""
    path = _db(tmp_path / "k.db", _working_day())
    q, ex, fp = _ready(store)

    assert ex.digest_usage(path=path, now=NOW) == habits.MAX_DAYS_PER_PASS


def test_a_broken_database_is_a_recorded_reason_not_a_dead_drain(
    store, tmp_path: Path
) -> None:
    """The failure this sense is most likely to have, and the one it cannot
    prevent: `knowledgeC`'s schema is undocumented and has moved between macOS
    releases. Failing to a reason keeps an OS update from stopping every future
    turn omega would take."""
    path = tmp_path / "k.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE ZSOMETHINGELSE (Z_PK INTEGER)")
    conn.commit()
    conn.close()
    q, ex, fp = _ready(store)

    assert ex.digest_usage(path=path, now=NOW) == habits.MAX_DAYS_PER_PASS

    receipts = _written(store, episodes.USAGE_DIGESTED)
    assert receipts and all(r["filed"] == 0 for r in receipts)
    assert all("ZOBJECT" in (r["reason"] or "") for r in receipts)


def test_nothing_about_a_day_is_ever_pushed_outward(store, tmp_path: Path) -> None:
    """DL-059's whole consent story in one assertion.

    This is the *noticed* category, from a source the person did not hand over
    but the operating system already had. "I see you were in Slack all
    afternoon" is the sentence that would end the project, and the way it stays
    impossible is that the receipt has no outward surface at all.
    """
    from omega import projection

    # The decision, asserted as a decision. `project` returns ``None`` for a
    # kind nobody wired up as readily as for one deliberately withheld, so a
    # test that only checked the ``None`` would pass on a fall-through — which
    # is a check that passes on empty.
    assert episodes.USAGE_DIGESTED in projection.NOT_PROJECTED
    assert episodes.CLAIM_EXTRACTED in projection.NOT_PROJECTED

    path = _db(tmp_path / "k.db", _working_day(), days=_every_day())
    q, ex, fp = _ready(store)
    ex.digest_usage(path=path, now=NOW)

    seen = set()
    for seq, payload in (
        (e.seq, episodes.decode(e.payload)) for e in store.episodes_since(0)
    ):
        if payload["kind"] in (episodes.USAGE_DIGESTED, episodes.CLAIM_EXTRACTED):
            seen.add(payload["kind"])
            assert projection.project(payload, seq) is None
    assert seen == {episodes.USAGE_DIGESTED, episodes.CLAIM_EXTRACTED}, (
        "the pass wrote neither kind, so nothing was actually checked"
    )


def test_the_pass_refuses_to_run_against_an_empty_log(store, tmp_path: Path) -> None:
    """The first thing in an empty log must not be a belief about a person
    omega has not met, inferred from a database it found on disk."""
    path = _db(tmp_path / "k.db", _working_day())
    q = EventQueue(store)
    ex = Executor(q, complete=_counting_reader().complete)
    ex.recover()

    assert ex.digest_usage(path=path, now=NOW) == 0
    assert _written(store, episodes.USAGE_DIGESTED) == []


def test_the_pass_runs_on_an_executor_built_the_way_production_builds_one(
    store, tmp_path: Path
) -> None:
    """DL-058's lesson, applied to the sense that came after it.

    `Runtime` builds the executor with no completer and lets it fall back to
    the process-wide provider. A test that injects one is constructing a
    different object than the one that ships, and that difference is exactly
    how the previous two idle passes shipped unreachable.
    """
    path = _db(tmp_path / "k.db", _working_day(), days=_every_day())
    q = EventQueue(store)
    q.append(episodes.inbound("morning", channel="tray"))
    fp = _counting_reader()
    ex = Executor(q)  # no `complete=`: exactly what `Runtime` passes
    ex.recover()

    previous = provider.set_provider(fp)
    try:
        assert ex.digest_usage(path=path, now=NOW) == habits.MAX_DAYS_PER_PASS
    finally:
        provider.set_provider(previous)

    assert [c["text"] for c in _written(store, episodes.CLAIM_EXTRACTED)] == [
        "They start around nine (day 1).",
        "They start around nine (day 2).",
    ]
