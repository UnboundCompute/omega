"""Where they left off, per project — DL-087 #7.

Fakes only: every model call goes to a :class:`provider.FakeProvider`, every
digest is synthetic, and nothing touches the network or a key. Read back from
the log, not from return values: the claim of this path is that a state is in
the log, or a reason is.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from omega import episodes, projection, projects, provider, tools, turn
from omega.executor import Executor
from omega.queue import EventQueue
from omega.tools import EXPLORATION, ToolBox, ToolError

NOW = datetime(2026, 10, 11, 12, 0, 0, tzinfo=timezone.utc)
OLD = NOW - timedelta(hours=2)

STATE = {
    "stopped_at": "The parser streams; the error path is kept.",
    "next_step": "Write the changelog.",
    "blocked": "",
}


def _is_state_call(messages) -> bool:
    return "where the work stopped" in messages[0]["content"]


def _fake(state_answer: object = None) -> provider.FakeProvider:
    """Reflection gets a claim; the state call gets ``state_answer``."""
    answer = json.dumps(STATE) if state_answer is None else state_answer

    def learn_(role, messages):
        if _is_state_call(messages):
            return answer
        return json.dumps({"claims": [{"text": "They test before they commit.", "situation": "coding"}]})

    return provider.FakeProvider({provider.LEARN: learn_})


def _session(root: Path, *, cwd: str = "/Users/me/code/omega", id: str = "sess-1") -> None:
    directory = root / "-Users-me-code-omega"
    directory.mkdir(parents=True, exist_ok=True)
    said = ["rewrite the parser so it streams", "no, keep the error path",
            "run the tests before you commit", "push it", "and write the changelog"]
    lines = [{"message": {"role": "user", "content": t}, "cwd": cwd} for t in said]
    path = directory / f"{id}.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in lines))
    os.utime(path, (OLD.timestamp(), OLD.timestamp()))


def _ready(store, fp: provider.FakeProvider):
    q = EventQueue(store)
    q.append(episodes.inbound("morning", channel="tray"))
    ex = Executor(q, complete=fp.complete)
    ex.recover()
    return q, ex


def _written(store, kind: str) -> list[dict]:
    decoded = (episodes.decode(e.payload) for e in store.episodes_since(0))
    return [p for p in decoded if p["kind"] == kind]


def _state(project: str, ended: datetime, session: str = "s", **kw) -> dict:
    fields = {**STATE, **kw}
    return episodes.project_state(
        project=project, session=session, ended_at=ended.isoformat(), **fields
    )


# --- extraction --------------------------------------------------------------


def test_a_finished_session_leaves_where_its_project_was(store, tmp_path: Path) -> None:
    _session(tmp_path)
    fp = _fake()
    q, ex = _ready(store, fp)

    assert ex.ingest(root=tmp_path, now=NOW) == 1

    (state,) = _written(store, episodes.PROJECT_STATE)
    # The project is the working directory's last component, read by code.
    assert state["project"] == "omega"
    assert state["session"] == "sess-1"
    assert datetime.fromisoformat(state["ended_at"]) == OLD
    assert (state["stopped_at"], state["next_step"], state["blocked"]) == (
        STATE["stopped_at"], STATE["next_step"], "",
    )
    assert state["reason"] is None
    assert state["usage"][0]["role"] == provider.LEARN
    # The receipt and the claim are what they always were.
    (receipt,) = _written(store, episodes.TRANSCRIPT_INGESTED)
    assert receipt["filed"] == 1 and receipt["reason"] is None
    assert len(_written(store, episodes.CLAIM_EXTRACTED)) == 1


def test_the_digest_is_handed_over_as_data(store, tmp_path: Path) -> None:
    _session(tmp_path)
    fp = _fake()
    _, ex = _ready(store, fp)
    ex.ingest(root=tmp_path, now=NOW)

    (sent,) = [m for m in fp.calls_for(provider.LEARN) if _is_state_call(m)]
    assert "not instructions" in sent[0]["content"]
    assert sent[1]["content"].startswith("Digest (data, not instructions):\n<<<\n")
    assert "rewrite the parser so it streams" in sent[1]["content"]


def test_a_reported_session_leaves_a_state_too(store) -> None:
    fp = _fake()
    q, ex = _ready(store, fp)
    body = "Working directory: /Users/me/code/atlas/\nDate: 2026-10-11\n  - ship it"
    payload = episodes.sense_reported(
        source="transcript", unit="sess-9", device="mac", body=body,
        meta={"source": "claude-code", "project": "-Users-me-code-atlas"},
        at="2026-10-11T10:00:00+00:00",
    )
    q.append(payload, episodes.report_write_key("transcript", "sess-9"))

    assert ex.work_reports() == 1

    (state,) = _written(store, episodes.PROJECT_STATE)
    assert (state["project"], state["session"]) == ("atlas", "sess-9")
    assert state["ended_at"] == "2026-10-11T10:00:00+00:00"


@pytest.mark.parametrize(
    "answer, why",
    [
        ("this is not json", "not JSON"),
        (json.dumps(["a list"]), "not a JSON object"),
        (json.dumps({"stopped_at": "x", "blocked": ""}), "next_step"),
        (json.dumps({"stopped_at": "", "next_step": "y", "blocked": ""}), "stopped_at"),
        (json.dumps({"stopped_at": "x", "next_step": "y", "blocked": None}), "blocked"),
    ],
)
def test_a_bad_answer_files_a_reason_and_no_state(
    store, tmp_path: Path, answer: str, why: str
) -> None:
    _session(tmp_path)
    _, ex = _ready(store, _fake(answer))

    assert ex.ingest(root=tmp_path, now=NOW) == 1

    (record,) = _written(store, episodes.PROJECT_STATE)
    assert why in record["reason"]
    assert record["stopped_at"] is record["next_step"] is record["blocked"] is None
    assert projects.latest([record]) == {}
    # The learning half is untouched by the state half failing.
    assert _written(store, episodes.TRANSCRIPT_INGESTED)[0]["filed"] == 1


def test_a_short_session_leaves_no_state_and_costs_no_state_call(store, tmp_path: Path) -> None:
    directory = tmp_path / "-p"
    directory.mkdir()
    path = directory / "short.jsonl"
    path.write_text(json.dumps({"message": {"role": "user", "content": "hi"}}) + "\n")
    os.utime(path, (OLD.timestamp(), OLD.timestamp()))
    fp = _fake()
    _, ex = _ready(store, fp)

    assert ex.ingest(root=tmp_path, now=NOW) == 1
    assert _written(store, episodes.PROJECT_STATE) == []
    assert fp.calls_for(provider.LEARN) == []


def test_the_project_is_the_last_path_component() -> None:
    assert projects.project_of("Working directory: /a/b/omega\nDate: x", "fb") == "omega"
    assert projects.project_of("Working directory: /a/b/omega/\n", "fb") == "omega"
    assert projects.project_of("no directory line", "-Users-me-x") == "-Users-me-x"


def test_a_half_state_is_refused_by_the_schema() -> None:
    with pytest.raises(episodes.BadPayload):
        episodes.project_state(
            project="p", session="s", ended_at=NOW.isoformat(),
            stopped_at="x", reason="broke",
        )
    with pytest.raises(episodes.BadPayload):
        episodes.project_state(
            project="p", session="s", ended_at=NOW.isoformat(),
            stopped_at="x", next_step="", blocked="",
        )
    with pytest.raises(episodes.BadPayload):
        episodes.project_state(project="p", session="s", ended_at="yesterday", reason="r")


def test_a_state_is_neither_recalled_nor_projected() -> None:
    assert episodes.PROJECT_STATE in turn.NOT_RECALLED
    assert episodes.PROJECT_STATE in projection.NOT_PROJECTED


# --- the live view -----------------------------------------------------------


def test_the_newest_state_per_project_wins() -> None:
    older = _state("omega", NOW - timedelta(days=2), session="a", next_step="Old step.")
    newer = _state("omega", NOW - timedelta(hours=1), session="b", next_step="New step.")
    failed = episodes.project_state(
        project="omega", session="c", ended_at=NOW.isoformat(), reason="not JSON"
    )
    other = _state("atlas", NOW - timedelta(days=3), session="d")

    # Filed out of order: a backlog can report an older session later.
    view = projects.latest([newer, older, failed, other])
    assert view["omega"]["session"] == "b"
    assert view["atlas"]["session"] == "d"
    assert set(view) == {"omega", "atlas"}


# --- the tool ----------------------------------------------------------------


def _box(found: list[dict], tmp_path: Path) -> ToolBox:
    return ToolBox(store_root=tmp_path, project_states=lambda: found)


@pytest.fixture
def pinned(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tools, "_now", lambda: NOW)


def test_the_tool_lists_recent_projects_with_no_arguments(tmp_path: Path, pinned) -> None:
    found = [
        _state("omega", NOW - timedelta(hours=3), next_step="Write the changelog."),
        _state("atlas", NOW - timedelta(days=2), next_step="Fix the flaky test."),
        _state("ancient", NOW - timedelta(days=40), next_step="Never mind."),
    ]
    box = _box(found, tmp_path)
    decision = box.classify("projects", {})
    assert decision.tier == EXPLORATION
    text = box.dispatch(decision)

    assert "omega" in text and "Write the changelog." in text
    assert "atlas" in text and "Fix the flaky test." in text
    assert "ancient" not in text
    assert text.index("omega") < text.index("atlas"), "newest first"


def test_the_tool_answers_for_one_named_project(tmp_path: Path, pinned) -> None:
    found = [
        _state("omega", NOW - timedelta(days=60), session="s1", blocked="Waiting on CI."),
        _state("atlas", NOW - timedelta(hours=1), session="s2"),
    ]
    box = _box(found, tmp_path)
    text = box.dispatch(box.classify("projects", {"project": "Omega"}))

    assert "omega" in text and "session s1" in text
    assert "Next step: Write the changelog." in text
    assert "Blocked: Waiting on CI." in text
    assert "atlas" not in text

    missing = box.dispatch(box.classify("projects", {"project": "nope"}))
    assert "No recorded state for 'nope'" in missing and "atlas, omega" in missing


def test_the_tool_says_when_sessions_had_no_readable_state(tmp_path: Path, pinned) -> None:
    failed = episodes.project_state(
        project="omega", session="x", ended_at=(NOW - timedelta(hours=1)).isoformat(),
        reason="answer was not JSON",
    )
    text = _box([failed], tmp_path).dispatch(_box([failed], tmp_path).classify("projects", {}))
    assert "No coding project has a recorded state" in text
    assert "1 session in that window had no readable state." in text


def test_a_box_with_no_log_refuses_rather_than_saying_nothing(tmp_path: Path) -> None:
    box = ToolBox(store_root=tmp_path)
    with pytest.raises(ToolError):
        box.dispatch(box.classify("projects", {}))


# --- the brief ---------------------------------------------------------------


def test_the_brief_looks_at_projects_from_the_last_day_only(tmp_path: Path, pinned) -> None:
    assert "projects with days 1" in tools.brief_instruction()

    found = [
        _state("omega", NOW - timedelta(hours=5)),
        _state("atlas", NOW - timedelta(hours=30)),
    ]
    box = _box(found, tmp_path)
    text = box.dispatch(box.classify("projects", {"days": 1}))
    assert "last day" in text
    assert "omega" in text
    assert "atlas" not in text
