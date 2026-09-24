"""``python -m omega`` — the surface a person actually touches.

The CLI is a *client* of the runtime, so the cases here are about the two things
only it can get wrong: how a turn is rendered (DL-011 — silence and failure must
never look the same), and what an unconfigured omega says on the first run.

Nothing here reaches the network or wants a key. The one test that drives
``main`` all the way through substitutes a ``FakeProvider`` for the builder, and
every case that touches configuration pins ``--env`` at a path inside ``tmp_path``
so the developer's own `.env` can never be read into the test process.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import invariants
from omega import __main__ as cli
from omega import episodes, projection, provider, runtime
from omega.memory import EPISODES_FILENAME, MemoryStore
from omega.queue import EventQueue

AT = "2026-09-23T12:00:00+00:00"


@pytest.fixture(autouse=True)
def no_ambient_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test may inherit a real key from the shell that ran pytest.

    Autouse and not optional: a suite whose result depends on whether the
    developer exported a key is a suite that passes for one person.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def speaking(reply: str = "ok") -> provider.FakeProvider:
    return provider.FakeProvider(
        {
            provider.JUDGE: lambda role, messages: "SPEAK",
            provider.ACT: lambda role, messages: reply,
        }
    )


def said(**kwargs) -> runtime.Said:
    base = dict(seq=1, record_seq=2, state=projection.COMPLETE)
    base.update(kwargs)
    return runtime.Said(**base)  # type: ignore[arg-type]


def lines_of(capsys: pytest.CaptureFixture) -> list[str]:
    return capsys.readouterr().out.splitlines()


def typing(script: list[str], then=EOFError):
    """Stand in for ``input``: the scripted lines, then how the session ends.

    ``then`` is a parameter because the two ways out — EOF and Ctrl-C — take
    different paths through ``main`` and must both land on the same clean stop.
    """
    lines = iter(script)

    def read(*args) -> str:
        try:
            return next(lines)
        except StopIteration:
            raise then
    return read


# --- green: the three outcomes are three different lines --------------------


def test_the_three_outcomes_render_as_three_different_things() -> None:
    """DL-011's whole point, at the only place a person sees it.

    A UI that printed nothing for silence and nothing for a failure would erase
    the distinction the milestone is built to keep — and would do it while every
    test below the UI still passed.
    """
    spoke = cli.render(said(outcome="spoke", reply="the parcel is here"))
    quiet = cli.render(said(outcome="silent", reply=None))
    broke = cli.render(
        said(state=projection.FAILED, outcome="failed", error="ProviderError: down")
    )
    stuck = cli.render(said(state=projection.BLOCKED, needs="the door code"))

    assert spoke == "omega: the parcel is here"
    assert quiet == cli.SILENCE
    assert "ProviderError: down" in broke
    assert "the door code" in stuck
    assert len({spoke, quiet, broke, stuck}) == 4, "no two outcomes may collide"


def test_silence_is_a_sentence_and_not_an_empty_line() -> None:
    """A blank line and a crash look identical in a terminal. Silence is a
    *success* (DL-011), so it has to say so out loud or the person reading the
    scrollback cannot tell it apart from omega having died."""
    rendered = cli.render(said(outcome="silent"))
    assert rendered.strip(), "silence must print something"
    assert "error" not in rendered.lower() and "fail" not in rendered.lower()


def test_the_repl_reads_lines_appends_them_and_prints_how_each_ended(
    store_dir: Path,
) -> None:
    typed = iter(["hello", "   ", "", "again"])
    written: list[str] = []

    def read_line() -> str:
        try:
            return next(typed)
        except StopIteration:
            raise EOFError

    with runtime.Runtime(store_dir, complete=speaking("hi").complete, listen=False) as rt:
        code = cli.repl(rt, read_line=read_line, write=written.append)

    assert code == 0, "reaching EOF is leaving, not failing"
    # Two lines typed, two answers -- the blank ones are not turns, because an
    # empty episode would be a turn omega has to judge for no reason.
    assert written == ["omega: hi", "omega: hi", ""]

    with MemoryStore.open(store_dir) as store:
        q = EventQueue(store)
        assert len([p for p in q.recent(q.head()) if p.kind == episodes.MESSAGE_INBOUND]) == 2


def test_main_runs_a_whole_session_and_leaves_the_log_clean(
    store_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    """The command end to end, with the model seam and nothing else faked."""
    fake = speaking("end to end")
    monkeypatch.setattr(provider, "provider_from_env", lambda *, env_path=None: fake)
    monkeypatch.setattr("builtins.input", typing(["are you there?"]))

    code = cli.main(
        ["--store", str(store_dir), "--env", str(tmp_path / "absent.env"), "--no-listen"]
    )

    out = lines_of(capsys)
    assert code == 0
    assert any(str(store_dir) in line for line in out), "it says which store it opened"
    assert "socket listener disabled" in out
    assert "omega: end to end" in out

    with MemoryStore.open(store_dir) as store:
        check = invariants.check(EventQueue(store))
    assert check, str(check)


def test_service_mode_starts_without_reading_a_terminal(
    store_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    fake = speaking("unused")
    entered: list[tuple[str, int]] = []
    monkeypatch.setattr(provider, "provider_from_env", lambda *, env_path=None: fake)
    monkeypatch.setattr(
        cli,
        "serve",
        lambda rt, *, write: entered.append(rt.address) or 0,
    )
    monkeypatch.setattr(
        "builtins.input",
        lambda *_args: pytest.fail("service mode must not read stdin"),
    )

    code = cli.main([
        "--store", str(store_dir),
        "--env", str(tmp_path / "absent.env"),
        "--port", "0",
        "--serve",
    ])

    assert code == 0
    assert len(entered) == 1 and entered[0][1] > 0
    assert "omega agent is running" not in "\n".join(lines_of(capsys))


def test_ctrl_c_leaves_the_same_way_eof_does(
    store_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    """Ctrl-C is not a crash (DL-016). It must stop between turns and leave the
    log in the state a clean stop leaves it — not the state a ``kill -9`` does,
    which is the one the startup report exists to describe."""
    fake = speaking("before the interrupt")
    monkeypatch.setattr(provider, "provider_from_env", lambda *, env_path=None: fake)
    monkeypatch.setattr(
        "builtins.input", typing(["one question"], then=KeyboardInterrupt)
    )

    code = cli.main(
        ["--store", str(store_dir), "--env", str(tmp_path / "absent.env"), "--no-listen"]
    )
    assert code == 0
    capsys.readouterr()

    # Reopening is the assertion: a clean report means no turn was abandoned.
    with runtime.Runtime(store_dir, complete=fake.complete, listen=False) as rt:
        assert rt.report.clean is True
        assert rt.report.lines() == []


def test_the_startup_report_is_printed_before_anything_else(
    store_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    """§1.2 — omega *says so*. An interrupted turn is surfaced to the person, and
    a report nobody prints is a report nobody gets."""
    with MemoryStore.open(store_dir) as store:
        q = EventQueue(store)
        seq = q.append(episodes.inbound("what was I doing?", channel="cli", at=AT))
        q.claim(seq)  # claimed, never finished: exactly what a kill -9 leaves

    fake = speaking("carrying on")
    monkeypatch.setattr(provider, "provider_from_env", lambda *, env_path=None: fake)
    monkeypatch.setattr("builtins.input", typing([]))

    code = cli.main(
        ["--store", str(store_dir), "--env", str(tmp_path / "absent.env"), "--no-listen"]
    )

    out = lines_of(capsys)
    assert code == 0
    assert any("what was I doing?" in line for line in out)
    assert not any("nothing was left in flight" in line for line in out)


# --- red: no key, and no traceback ------------------------------------------


def test_a_missing_key_is_a_sentence_naming_a_file_not_a_traceback(
    store_dir: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    """The most likely first run. A stack trace tells a person nothing about
    where their key goes, so this path must name the exact file."""
    env = tmp_path / "nowhere" / ".env"

    code = cli.main(["--store", str(store_dir), "--env", str(env), "--no-listen"])

    out = capsys.readouterr().out
    assert code == 2
    assert "Traceback" not in out and "ProviderNotConfigured" not in out
    assert "OPENAI_API_KEY" in out
    assert str(env) in out, "it must name the file, not just the variable"
    # And it stopped *before* taking M0's exclusive lock: an omega that cannot
    # think has no business holding the log open against the next attempt.
    assert not (store_dir / EPISODES_FILENAME).exists()


def test_a_key_that_is_present_but_unusable_is_not_reported_as_a_missing_key(
    store_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    """``ProviderNotConfigured`` covers both "no key" and "no client library".
    Sending someone to edit a `.env` that is already correct is the worst kind of
    wrong answer: it is confident, specific, and it wastes the afternoon."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-present")

    def unusable(*, env_path=None):
        raise provider.ProviderNotConfigured("the openai package is not installed")

    monkeypatch.setattr(provider, "provider_from_env", unusable)

    code = cli.main(
        ["--store", str(store_dir), "--env", str(tmp_path / "absent.env"), "--no-listen"]
    )

    out = capsys.readouterr().out
    assert code == 2
    assert "openai package is not installed" in out
    assert "put one in" not in out, "do not send them to fix a file that is right"


# --- yellow: where `.env` comes from ----------------------------------------


def test_an_explicit_env_wins_and_does_not_fall_back(tmp_path: Path) -> None:
    """Reading a *different* file than the one named is worse than reading
    none: the person has no way to see it happened."""
    store = tmp_path / "store"
    store.mkdir()
    (store / ".env").write_text("OPENAI_API_KEY=beside\n")
    named = tmp_path / "named.env"
    named.write_text("OPENAI_API_KEY=named\n")

    found, where = cli.resolve_env(str(named), store)
    assert found == named and where == named

    missing = tmp_path / "gone.env"
    found, where = cli.resolve_env(str(missing), store)
    assert found is None, "an explicit file that is absent must not fall back"
    assert where == missing, "and the message must still name what was asked for"


def test_the_store_wins_over_the_repo_root(tmp_path: Path) -> None:
    store = tmp_path / "store"
    store.mkdir()
    beside = store / ".env"
    beside.write_text("OPENAI_API_KEY=beside\n")

    found, where = cli.resolve_env(None, store)
    assert found == beside and where == beside


def test_with_nothing_anywhere_it_still_names_a_place_to_put_the_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = tmp_path / "store"
    store.mkdir()
    monkeypatch.setattr(cli, "_REPO_ROOT", tmp_path / "no-such-repo")

    found, where = cli.resolve_env(None, store)
    assert found is None
    assert where == store / ".env", '"put it somewhere" is not help'


def test_the_working_directory_is_never_consulted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``provider.load_env`` defaults to ``cwd()/.env``, which would make omega
    work from the repo root and nowhere else — a rule nobody can see and
    everybody trips over. The CLI resolves the path itself for exactly that
    reason, so the cwd must not creep back in."""
    cwd = tmp_path / "somewhere-else"
    cwd.mkdir()
    (cwd / ".env").write_text("OPENAI_API_KEY=cwd\n")
    store = tmp_path / "store"
    store.mkdir()
    monkeypatch.setattr(cli, "_REPO_ROOT", tmp_path / "no-such-repo")
    monkeypatch.chdir(cwd)

    found, where = cli.resolve_env(None, store)
    assert found is None
    assert where != cwd / ".env"


def test_main_never_passes_none_to_load_env(
    store_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same rule one layer down. ``provider_from_env(env_path=None)`` is the
    cwd fallback wearing a different name, so the CLI must always hand it a real
    path — even one that does not exist."""
    seen: list[object] = []

    def record(*, env_path=None):
        seen.append(env_path)
        raise provider.ProviderNotConfigured("no key")

    monkeypatch.setattr(provider, "provider_from_env", record)
    monkeypatch.setattr(cli, "_REPO_ROOT", tmp_path / "no-such-repo")

    cli.main(["--store", str(store_dir), "--no-listen"])

    assert seen and seen[0] is not None
    assert Path(seen[0]) == store_dir / ".env"  # type: ignore[arg-type]


# --- the flags --------------------------------------------------------------


def test_the_defaults_are_the_documented_ones() -> None:
    args = cli.build_parser().parse_args([])
    assert args.store == cli.DEFAULT_STORE == "~/.omega"
    assert args.env is None
    assert args.listen is True


def test_the_listener_can_be_turned_off() -> None:
    assert cli.build_parser().parse_args(["--no-listen"]).listen is False


def test_the_store_path_expands_a_tilde(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``~/.omega`` is the default, so a literal ``~`` directory in the cwd would
    be a quietly wrong store — and the person would find an empty omega with no
    idea why."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(cli, "_REPO_ROOT", tmp_path / "no-such-repo")
    _found, where = cli.resolve_env(None, Path(cli.DEFAULT_STORE).expanduser())
    assert where == tmp_path / ".omega" / ".env"


# --- the authorities the assembled process grants ----------------------------


def test_the_real_process_grants_both_senses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one place that assembles omega is the one place that hands it an
    authority to read outside its own store — and until now nothing checked
    that it still does.

    This is DL-058's lesson at the layer above it. Both senses default to
    ``None`` on ``Runtime`` deliberately, so that no test and no embedder reads
    the person's home directory by accident; the cost of that default is that
    deleting the grant in `main` breaks nothing any other test can see. The
    sense would simply be off in production, silently, which is exactly how
    reflection and ingestion each shipped unreachable.

    Asserted as *the real paths*, not merely as "not None": a grant pointing
    somewhere plausible but wrong is the same silent failure with more steps.
    """
    from omega import habits, transcripts

    seen: dict[str, object] = {}

    class Recorded:
        def __init__(self, store_dir, **kwargs):
            seen.update(kwargs)
            seen["store_dir"] = store_dir

        def start(self):
            raise SystemExit(0)

    fake = speaking()
    monkeypatch.setattr(provider, "provider_from_env", lambda *, env_path=None: fake)
    monkeypatch.setattr(runtime, "Runtime", Recorded)

    with pytest.raises(SystemExit):
        cli.main(["--store", str(tmp_path / "store"), "--no-listen"])

    assert seen["transcripts_root"] == transcripts.default_root()
    assert seen["usage_path"] == habits.default_path()
