"""DL-104 — hand work to a live Claude Code session on their Mac.

**No ``claude``, no model, no network beyond loopback.** Every ``claude``
command goes to :class:`FakeClaude`, which records the argv and the folder it
ran in and answers as ``claude --bg`` does; every hook event is a line the test
writes into the job's events file, exactly as the hook script would; every
model answer is scripted on ``provider.FakeProvider``. The clock is pinned.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import pytest

from omega import act as act_mod
from omega import delegate, discord, episodes, projection, provider, tools, turn as turn_mod
from omega.act import act_loop
from omega.blobs import BlobStore
from omega.channel import Channel, ChannelClient
from omega.delegate import RULES, Desk, Worker
from omega.derive import OpenWork
from omega.grants import Grants
from omega.memory import MemoryStore
from omega.provider import ToolCall
from omega.queue import EventQueue
from omega.tools import ToolBox, ToolError
from omega.turn import run_turn

AT = "2026-10-15T09:00:00+00:00"
NOW = datetime.fromisoformat(AT)
SESSION = "d21bfd32"
FULL = "d21bfd32-5544-4fbb-b4b4-603d2251aa7f"


@pytest.fixture(autouse=True)
def clock(monkeypatch: pytest.MonkeyPatch) -> Desk:
    """A pinned clock and a fresh desk nobody has polled."""
    monkeypatch.setattr(episodes, "now", lambda: AT)
    monkeypatch.setattr(delegate, "_now", lambda: NOW)
    fresh = Desk()
    monkeypatch.setattr(delegate, "_DESK", fresh)
    monkeypatch.delenv(delegate.ENV_DIRS, raising=False)
    return fresh


@pytest.fixture
def q(store: MemoryStore) -> EventQueue:
    return EventQueue(store)


@pytest.fixture
def live(clock: Desk) -> Desk:
    """A worker on their Mac polled just now."""
    clock.seen(NOW, ["/Users/someone/work"])
    return clock


class FakeClaude:
    """``claude``, in memory: records each command, answers each as the CLI
    does, and refuses a folder it was told is untrusted."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], Path]] = []
        self.untrusted: set[Path] = set()
        self.ids = iter([SESSION, "e31cfe40", "f41dff50", "a51e0060"])
        self.agents: Optional[list[dict]] = None  # ``claude agents --json --all``; None fails

    def __call__(self, argv: Any, cwd: Path) -> tuple[int, str]:
        argv = list(argv)
        self.calls.append((argv, Path(cwd)))
        if argv[1] == "stop":
            return 0, f"stopped {argv[2]}\n"
        if argv[1] == "agents":
            return (1, "") if self.agents is None else (0, json.dumps(self.agents))
        if Path(cwd) in self.untrusted:
            return 1, f"Workspace not trusted. Run `claude` in {cwd} once and accept the trust prompt, then retry.\n"
        name = argv[argv.index("-n") + 1]
        return 0, f"backgrounded · {next(self.ids)} · {name}\n  claude attach {SESSION}\n"

    def starts(self) -> list[list[str]]:
        return [a for a, _ in self.calls if "--bg" in a]


def call(name: str = tools.DELEGATE, id: str = "c1", **arguments: Any) -> ToolCall:
    return ToolCall(id=id, name=name, arguments=dict(arguments))


def start(folder: str = "site", worktree: bool = False, **more: Any) -> ToolCall:
    return call(do="start", title="write the launch doc", brief="Write a one-page launch doc.",
                folder=folder, worktree=worktree, **more)


def say(q: EventQueue, text: str, channel: str = "discord", **more: Any) -> int:
    return q.append(episodes.inbound(text, channel=channel, at=AT, **more))


def turn(q: EventQueue, seq: int, *, act: list[Any]):
    q.claim(seq)
    fp = provider.FakeProvider({provider.JUDGE: ["ACT"], provider.ACT: list(act)})
    result = run_turn(
        q, q.at(seq), complete=fp.complete, act=act_loop,
        open_work=OpenWork().advance(q.store, upto=seq).blocks(),
        grants=Grants().advance(q.store, upto=seq).state(),
        at=AT,
    )
    q.finish(seq)
    return result, fp


def payloads(q: EventQueue) -> list[dict]:
    return [episodes.decode(e.payload) for e in q.store.episodes_since(0)]


def kinds(q: EventQueue, kind: str) -> list[dict]:
    return [p for p in payloads(q) if p.get("kind") == kind]


def offered(fp: provider.FakeProvider) -> list[str]:
    return [
        s["function"]["name"] for o in fp.offers_for(provider.ACT) for s in (o or []) if "function" in s
    ]


def proposal(q: EventQueue) -> int:
    return [e.seq for e in q.store.episodes_since(0)
            if episodes.decode(e.payload).get("kind") == episodes.GRANT_PROPOSED][-1]


def requested(q: EventQueue, live: Desk, folder: str = "site", worktree: bool = False) -> dict:
    """A start they said yes to: the whole path through the hold."""
    seq = say(q, "ask claude to write the launch doc")
    turn(q, seq, act=[start(folder=folder, worktree=worktree), "Asked."])
    yes = say(q, f"yes p{proposal(q)}")
    turn(q, yes, act=["Started."])
    [rec] = kinds(q, episodes.DELEGATE_REQUESTED)[-1:]
    return rec


# --- the allowlist, on the Mac ----------------------------------------------------


def test_no_allowlist_is_no_feature(tmp_path: Path) -> None:
    assert delegate.roots({}) == ()
    assert delegate.roots({delegate.ENV_DIRS: ""}) == ()
    # Relative paths and folders that do not exist are dropped, never made.
    assert delegate.roots({delegate.ENV_DIRS: f"work{os.pathsep}{tmp_path / 'nope'}"}) == ()
    assert not (tmp_path / "nope").exists()
    stopping = __import__("threading").Event()
    assert delegate.start_worker(("127.0.0.1", 1), environ={}, stopping=stopping, log=print) is None
    with pytest.raises(ValueError):
        Worker(("127.0.0.1", 1), allowed=(), home=tmp_path)


def test_the_environment_wins_over_the_env_file(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    env_file = tmp_path / ".env"
    env_file.write_text(f"{delegate.ENV_DIRS}={a}\n")
    assert delegate.roots({}, env_file) == (Path(os.path.realpath(a)),)
    assert delegate.roots({delegate.ENV_DIRS: str(b)}, env_file) == (Path(os.path.realpath(b)),)
    assert delegate.roots({delegate.ENV_DIRS: f"{a}{os.pathsep}{b}"}) == (
        Path(os.path.realpath(a)), Path(os.path.realpath(b)))


def test_a_folder_outside_the_list_or_escaping_by_a_symlink_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "work"
    (root / "site").mkdir(parents=True)
    outside = tmp_path / "private"
    outside.mkdir()
    (root / "escape").symlink_to(outside)
    allowed = delegate.roots({delegate.ENV_DIRS: str(root)})
    real_site = Path(os.path.realpath(root / "site"))
    assert delegate.resolve_folder("site", allowed) == real_site
    assert delegate.resolve_folder(str(root / "site"), allowed) == real_site
    assert delegate.resolve_folder("work", allowed) == Path(os.path.realpath(root))
    for refused in (str(outside), "escape", str(root / "escape"), "..", "site/../..",
                    str(root / "site" / ".." / ".."), "", "nope", "/"):
        assert delegate.resolve_folder(refused, allowed) is None, refused
    assert delegate.resolve_folder("site", ()) is None


# --- the tool, on the core -----------------------------------------------------------


def test_the_tool_is_not_offered_unprompted_or_with_the_mac_offline(
    q: EventQueue, clock: Desk,
) -> None:
    seq = say(q, "ask claude to write the launch doc")
    _, fp = turn(q, seq, act=["I can't."])
    assert tools.DELEGATE not in offered(fp)
    # Polled, but longer ago than LIVE_SECONDS: offline.
    clock.seen(NOW - timedelta(seconds=delegate.LIVE_SECONDS + 1), ["/w"])
    seq = say(q, "ask claude again")
    _, fp = turn(q, seq, act=["I can't."])
    assert tools.DELEGATE not in offered(fp)
    # Live, but nobody asked: a scheduled turn never holds it.
    clock.seen(NOW, ["/w"])
    seq = say(q, "check in", channel="schedule")
    _, fp = turn(q, seq, act=["ok"])
    assert tools.DELEGATE not in offered(fp)
    seq = say(q, "ask claude to write the launch doc")
    _, fp = turn(q, seq, act=["ok"])
    assert tools.DELEGATE in offered(fp)


def test_a_call_while_offline_is_refused_by_the_gate(q: EventQueue) -> None:
    box = ToolBox(store_root=q.store.root, append=q.append, prompted=True)
    with pytest.raises(tools.ToolRejected):
        box.classify(tools.DELEGATE, {"do": "status"})


def test_every_start_is_held_even_on_a_clean_turn(q: EventQueue, live: Desk) -> None:
    seq = say(q, "ask claude to write the launch doc")
    result, _ = turn(q, seq, act=[start(), "Asked."])
    assert kinds(q, episodes.DELEGATE_REQUESTED) == []
    [held] = kinds(q, episodes.GRANT_PROPOSED)
    assert held["proposes"] == "hold" and held["reason"] == act_mod._ALWAYS_HELD
    assert held["actions"][0]["tool"] == tools.DELEGATE
    # They read the brief before they say yes to it.
    assert "Write a one-page launch doc." in result.reply
    assert f"yes p{proposal(q)}" in result.reply
    assert result.reply.startswith("Should I start Claude Code on your Mac to write the launch doc, in site?")


def test_their_yes_files_the_start_with_the_rules_and_their_channel(q: EventQueue, live: Desk) -> None:
    rec = requested(q, live)
    assert rec["brief"] == "Write a one-page launch doc.\n\n" + RULES
    assert rec["reply_to"] == "discord" and rec["folder"] == "site" and rec["worktree"] is False
    assert rec["job"] == f"j{rec['for_seq']}-1"
    assert "never pay, order, book or check out" in RULES and "never force-push" in RULES
    assert "Never delete anything outside" in RULES and "outbox folder" in RULES
    # Their no files nothing.
    seq = say(q, "ask claude to write it again")
    turn(q, seq, act=[start(), "Asked."])
    no = say(q, f"no p{proposal(q)}")
    turn(q, no, act=["Okay."])
    assert len(kinds(q, episodes.DELEGATE_REQUESTED)) == 1


def test_a_tainted_turn_holds_a_follow_up_too(q: EventQueue, live: Desk) -> None:
    assert act_mod._outward(tools.Decision(tool=tools.DELEGATE, tier=tools.LOCAL, why="x",
                                           args={"do": "followup", "job": "j1-1", "text": "more"}))
    assert not act_mod._outward(tools.Decision(tool=tools.DELEGATE, tier=tools.LOCAL, why="x",
                                               args={"do": "stop", "job": "j1-1"}))
    assert tools.untrusted_source(tools.DELEGATE, {"do": "status"}) == delegate.SOURCE


def test_offline_at_dispatch_says_so_and_queues_nothing(q: EventQueue, live: Desk) -> None:
    seat = delegate.Seat(desk=live, jobs=lambda: delegate.jobs(q.store), reply_to="tray")
    box = ToolBox(store_root=q.store.root, append=q.append, prompted=True, delegate=seat, for_seq=3)
    decision = box.classify(tools.DELEGATE, {"do": "start", "title": "t", "brief": "b", "folder": "f"})
    live.last = NOW - timedelta(minutes=5)
    with pytest.raises(ToolError, match="your Mac is not connected.*nothing was queued"):
        box.dispatch(decision)
    assert kinds(q, episodes.DELEGATE_REQUESTED) == []


# --- the worker, end to end over the channel ------------------------------------------


@pytest.fixture
def mac(tmp_path: Path) -> dict[str, Path]:
    root = tmp_path / "work"
    (root / "site").mkdir(parents=True)
    (root / "repo" / ".git").mkdir(parents=True)
    return {"root": root, "home": tmp_path / "jobs"}


@pytest.fixture
def channel(q: EventQueue, store_dir: Path, live: Desk):
    ch = Channel(q, BlobStore.open(store_dir), port=0, poll=0.005, now=lambda: NOW, desk=live)
    ch.start()
    try:
        yield ch
    finally:
        ch.stop()


def worker(channel: Channel, mac: dict[str, Path], fake: FakeClaude, **kw: Any) -> Worker:
    return Worker(
        channel.address, allowed=delegate.roots({delegate.ENV_DIRS: str(mac["root"])}),
        home=mac["home"], run=fake, connect=lambda a: ChannelClient(a, timeout=5.0),
        python="/usr/bin/python3", log=print, **kw,
    )


def hook(mac: dict[str, Path], job: str, **event: Any) -> None:
    with (mac["home"] / job / "events.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def tells(q: EventQueue) -> list[dict]:
    return [p for p in kinds(q, episodes.MESSAGE_INBOUND) if tools.is_tell(p)]


def test_started_waiting_done_carries_the_result_and_its_files(
    q: EventQueue, live: Desk, channel: Channel, mac: dict[str, Path],
) -> None:
    rec = requested(q, live)
    job = rec["job"]
    fake = FakeClaude()
    w = worker(channel, mac, fake)
    assert w.pass_once() == 1
    [argv] = fake.starts()
    assert fake.calls[0][1] == Path(os.path.realpath(mac["root"] / "site"))
    assert argv[:4] == ["claude", "--bg", "-n", "omega: write the launch doc"]
    assert argv[argv.index("--permission-mode") + 1] == "auto"
    assert "--worktree" not in argv
    joined = " ".join(argv)
    assert "bypassPermissions" not in joined and "dangerously" not in joined
    outbox = mac["home"] / job / "outbox"
    assert argv[argv.index("--add-dir") + 1] == str(outbox)
    assert argv[-1] == rec["brief"] + f"\n\nOutbox folder: {outbox}"
    hooks = json.loads(argv[argv.index("--settings") + 1])["hooks"]
    assert set(hooks) == {"Stop", "Notification"}
    [up] = kinds(q, episodes.DELEGATE_UPDATED)
    assert up["state"] == "started" and up["session"] == SESSION and up["cmd"] == rec_seq(q, job)
    [told] = tells(q)
    assert told["reply_to"] == "discord" and told["untrusted"] == delegate.SOURCE
    assert f"`claude attach {SESSION}`" in told["text"]
    # The command was answered: a second pass starts nothing.
    assert w.pass_once() == 0 and len(fake.starts()) == 1

    hook(mac, job, event="notification", session_id=FULL, notification_type="idle_prompt", message="idle")
    hook(mac, job, event="notification", session_id=FULL, notification_type="permission_prompt",
         message="Claude needs your permission to use Bash")
    w.pass_once()
    assert [u["state"] for u in kinds(q, episodes.DELEGATE_UPDATED)] == ["started", "waiting"]
    assert "needs your permission" in tells(q)[-1]["text"]

    (outbox / "launch.md").write_text("# Launch\n")
    (outbox / ".hidden").write_text("no")
    (outbox / "link.md").symlink_to(mac["root"] / "site")
    hook(mac, job, event="stop", session_id=FULL, last_assistant_message="Wrote launch.md.")
    w.pass_once()
    [done] = kinds(q, episodes.DELEGATE_FINISHED)
    assert done["text"] == "Wrote launch.md." and [f["name"] for f in done["files"]] == ["launch.md"]
    last = tells(q)[-1]
    assert "Wrote launch.md." in last["text"]
    assert f"[attached: launch.md {done['files'][0]['blob']}]" in last["text"]
    # Nothing is sent twice.
    w.pass_once()
    assert len(kinds(q, episodes.DELEGATE_FINISHED)) == 1

    # The tell's reply goes back to Discord, and says where.
    seq = [e.seq for e in q.store.episodes_since(0)
           if episodes.decode(e.payload).get("text") == last["text"]][0]
    result, _ = turn(q, seq, act=[])
    assert result.reply and "Wrote launch.md." in result.reply
    [rec_done] = [p for p in kinds(q, episodes.TURN_COMPLETED) if p["for_seq"] == seq]
    assert rec_done["reply_to"] == "discord"
    wire = [u.wire() for u in projection.updates_since(q, seq - 1) if u.for_seq == seq and u.kind == episodes.TURN_COMPLETED]
    assert wire and wire[-1]["channel"] == "discord"
    assert discord.decide(wire[-1], origin=lambda: None, at_the_mac=lambda: True) == result.reply
    assert discord.decide({**wire[-1], "channel": "tray"}, origin=lambda: None, at_the_mac=lambda: False) is None


def rec_seq(q: EventQueue, job: str) -> int:
    return [e.seq for e in q.store.episodes_since(0)
            if episodes.decode(e.payload).get("kind") == episodes.DELEGATE_REQUESTED
            and episodes.decode(e.payload)["job"] == job][0]


def test_a_git_folder_gets_its_own_worktree(
    q: EventQueue, live: Desk, channel: Channel, mac: dict[str, Path],
) -> None:
    rec = requested(q, live, folder="repo", worktree=True)
    fake = FakeClaude()
    worker(channel, mac, fake).pass_once()
    [argv] = fake.starts()
    assert argv[argv.index("--worktree") + 1] == f"omega-{rec['job']}"
    # Worktree asked for, but not a git folder: no flag.
    requested(q, live, folder="site", worktree=True)
    worker(channel, mac, fake).pass_once()
    assert "--worktree" not in fake.starts()[-1]


def test_a_folder_the_mac_does_not_allow_starts_nothing(
    q: EventQueue, live: Desk, channel: Channel, mac: dict[str, Path], tmp_path: Path,
) -> None:
    (tmp_path / "secret").mkdir()
    (mac["root"] / "sneaky").symlink_to(tmp_path / "secret")
    for folder in (str(tmp_path / "secret"), "sneaky"):
        requested(q, live, folder=folder)
    fake = FakeClaude()
    worker(channel, mac, fake).pass_once()
    assert fake.calls == []
    ups = kinds(q, episodes.DELEGATE_UPDATED)
    assert [u["state"] for u in ups] == ["failed", "failed"]
    assert all("not inside a folder this Mac allows" in u["detail"] for u in ups)
    assert "could not do" in tells(q)[-1]["text"]


def test_a_brief_without_the_rules_is_refused_on_the_mac(
    q: EventQueue, store_dir: Path, mac: dict[str, Path], channel: Channel,
) -> None:
    q.append(episodes.delegate_requested(for_seq=1, job="j1-1", title="t", brief="do it, no rules",
                                         folder="site", worktree=False))
    fake = FakeClaude()
    worker(channel, mac, fake).pass_once()
    assert fake.calls == []
    [up] = kinds(q, episodes.DELEGATE_UPDATED)
    assert up["state"] == "failed" and "fixed rules" in up["detail"]


def test_an_untrusted_folder_says_how_to_trust_it(
    q: EventQueue, live: Desk, channel: Channel, mac: dict[str, Path],
) -> None:
    requested(q, live)
    fake = FakeClaude()
    fake.untrusted.add(Path(os.path.realpath(mac["root"] / "site")))
    worker(channel, mac, fake).pass_once()
    [up] = kinds(q, episodes.DELEGATE_UPDATED)
    assert up["state"] == "failed" and "accept the trust prompt" in up["detail"]


def test_at_most_two_run_at_once(
    q: EventQueue, live: Desk, channel: Channel, mac: dict[str, Path],
) -> None:
    for _ in range(3):
        requested(q, live)
    fake = FakeClaude()
    worker(channel, mac, fake).pass_once()
    assert len(fake.starts()) == delegate.MAX_RUNNING == 2
    states = [u["state"] for u in kinds(q, episodes.DELEGATE_UPDATED)]
    assert states == ["started", "started", "failed"]
    assert "already running" in kinds(q, episodes.DELEGATE_UPDATED)[-1]["detail"]


def test_a_session_that_ended_unseen_frees_its_slot_only_when_listed_so(
    q: EventQueue, live: Desk, channel: Channel, mac: dict[str, Path],
) -> None:
    for _ in range(3):
        requested(q, live)
    fake = FakeClaude()
    fake.agents = [{"id": SESSION, "state": "running"}, {"id": "e31cfe40", "state": "stopped"}]
    worker(channel, mac, fake).pass_once()
    # The second was stopped by hand: reported so, and the third takes its slot.
    assert len(fake.starts()) == 3
    states = [u["state"] for u in kinds(q, episodes.DELEGATE_UPDATED)]
    assert states == ["started", "started", "stopped", "started"]
    assert "no longer running" in kinds(q, episodes.DELEGATE_UPDATED)[2]["detail"]


def test_a_command_for_a_job_the_mac_never_had_is_answered_once(
    q: EventQueue, live: Desk, channel: Channel, mac: dict[str, Path],
) -> None:
    rec = requested(q, live)
    job = rec["job"]
    fake = FakeClaude()
    w = worker(channel, mac, fake)
    w.pass_once()
    shutil.rmtree(mac["home"] / job)
    seq = say(q, "stop that claude session")
    turn(q, seq, act=[call(do="stop", job=job), "Stopping."])
    w.pass_once()
    w.pass_once()
    last = kinds(q, episodes.DELEGATE_UPDATED)
    assert [u["state"] for u in last] == ["started", "failed"]
    assert "no record of that job" in last[-1]["detail"]


def test_stop_and_follow_up(
    q: EventQueue, live: Desk, channel: Channel, mac: dict[str, Path],
) -> None:
    rec = requested(q, live)
    job = rec["job"]
    fake = FakeClaude()
    w = worker(channel, mac, fake)
    w.pass_once()
    # While it works, a follow-up is not sent: they steer it themselves.
    seq = say(q, "tell claude to add a FAQ")
    held_before = len(kinds(q, episodes.GRANT_PROPOSED))
    _, fp = turn(q, seq, act=[call(do="followup", job=job, text="Add a FAQ."), "ok"])
    said = "\n".join(str(m.get("content", "")) for msgs in fp.calls_for(provider.ACT) for m in msgs)
    assert f"it's still working; open it with `claude attach {SESSION}` to steer" in said
    assert kinds(q, episodes.DELEGATE_ASKED) == []
    assert len(kinds(q, episodes.GRANT_PROPOSED)) == held_before
    # Done, then followed up: the idle session stops, and resumes in the background.
    tree = mac["root"] / "site" / ".claude" / "worktrees" / "omega"
    tree.mkdir(parents=True)
    hook(mac, job, event="stop", session_id=FULL, last_assistant_message="Done.", cwd=str(tree))
    w.pass_once()
    seq = say(q, "tell claude to add a FAQ")
    turn(q, seq, act=[call(do="followup", job=job, text="Add a FAQ."), "Sent."])
    yes = say(q, f"yes p{proposal(q)}")
    turn(q, yes, act=["Sent."])
    [asked] = kinds(q, episodes.DELEGATE_ASKED)
    assert asked["do"] == "followup" and asked["text"] == "Add a FAQ."
    w.pass_once()
    assert fake.calls[-2][0] == ["claude", "stop", SESSION]
    argv = fake.calls[-1][0]
    assert argv[:4] == ["claude", "--resume", FULL, "--bg"] and argv[-1] == "Add a FAQ."
    assert fake.calls[-1][1] == Path(os.path.realpath(tree))  # where the session ran
    assert "bypassPermissions" not in " ".join(argv)
    assert kinds(q, episodes.DELEGATE_UPDATED)[-1]["state"] == "started"
    # Stop: not held, sent straight away, and the Mac stops it.
    seq = say(q, "stop that claude session")
    turn(q, seq, act=[call(do="stop", job=job), "Stopping."])
    w.pass_once()
    assert fake.calls[-1][0][:2] == ["claude", "stop"]
    assert kinds(q, episodes.DELEGATE_UPDATED)[-1]["state"] == "stopped"
    assert "stopped" in tells(q)[-1]["text"]


def test_a_command_the_mac_lost_the_receipt_for_is_not_run_twice(
    q: EventQueue, live: Desk, channel: Channel, mac: dict[str, Path],
) -> None:
    rec = requested(q, live)
    job, cmd = rec["job"], rec_seq(q, rec["job"])
    fake = FakeClaude()
    worker(channel, mac, fake).pass_once()
    # The same receipt again, by its id: the duplicate path, one record, one tell.
    with ChannelClient(channel.address) as c:
        again = c.request({"v": 1, "op": "delegate.update", "id": f"{job}-c{cmd}", "job": job,
                           "state": "started", "session": SESSION, "cmd": cmd})
    assert again["op"] == "ack" and again["duplicate"] is True
    assert len(kinds(q, episodes.DELEGATE_UPDATED)) == 1 and len(tells(q)) == 1
    # A worker restarted with the command still unanswered on the core finds
    # the job's file and resends the receipt, rather than starting it again.
    (mac["home"] / job / "job.json").write_text(json.dumps(
        {**json.loads((mac["home"] / job / "job.json").read_text()), "state": "starting"}))
    q.append(episodes.delegate_asked(for_seq=cmd, job=job, do="stop"))
    worker(channel, mac, fake).pass_once()
    assert len(fake.starts()) == 1


def test_the_channel_refuses_a_report_for_no_job_or_an_unknown_file(
    q: EventQueue, channel: Channel,
) -> None:
    with ChannelClient(channel.address) as c:
        no_job = c.request({"v": 1, "op": "delegate.update", "id": "x1", "job": "j9-1", "state": "started"})
        assert no_job["op"] == "error" and "no job" in no_job["error"]
        q.append(episodes.delegate_requested(for_seq=1, job="j1-1", title="t", brief=RULES,
                                             folder="site", worktree=False))
        bad = c.request({"v": 1, "op": "delegate.result", "id": "x2", "job": "j1-1", "text": "t",
                         "files": [{"name": "a", "blob": "sha256:" + "0" * 64, "mime": "text/plain", "bytes": 1}]})
        assert bad["op"] == "error" and "sent with put" in bad["error"]
        assert c.request({"v": 1, "op": "delegate.poll", "folders": []})["op"] == "error"


def test_the_worker_sends_only_its_own_ops(tmp_path: Path) -> None:
    w = Worker(("127.0.0.1", 1), allowed=(tmp_path,), home=tmp_path / "h")
    with pytest.raises(RuntimeError):
        w._send(object(), "say", text="hi")
    assert delegate.WORKER_OPS == {"delegate.poll", "delegate.update", "delegate.result", "put"}


# --- reading the result -----------------------------------------------------------------


def test_the_hook_script_appends_one_line(tmp_path: Path) -> None:
    events = tmp_path / "events.jsonl"
    stdin = json.dumps({"session_id": FULL, "last_assistant_message": "smoke ok",
                        "transcript_path": "/t.jsonl", "cwd": "/x", "hook_event_name": "Stop"})
    subprocess.run([sys.executable, "-I", "-c", delegate._HOOK, str(events), "stop"],
                   input=stdin, text=True, check=True)
    assert json.loads(events.read_text()) == {
        "event": "stop", "session_id": FULL, "transcript_path": "/t.jsonl",
        "last_assistant_message": "smoke ok", "cwd": "/x",
    }


def test_the_transcript_is_only_the_fallback(tmp_path: Path) -> None:
    t = tmp_path / "t.jsonl"
    t.write_text("\n".join([
        json.dumps({"type": "user", "message": {"content": "hi"}}),
        json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "first"}]}}),
        "not json",
        json.dumps({"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "x"}]}}),
        json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "smoke ok"}]}}),
    ]))
    assert delegate.last_text(t) == "smoke ok"
    assert delegate.last_text(tmp_path / "missing") == ""


# --- with no worker, nothing changes ---------------------------------------------------------


def test_with_no_worker_nothing_changes(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def run() -> str:
        s = MemoryStore.open(tmp_path_factory.mktemp("s"))
        try:
            q = EventQueue(s)
            seq = say(q, "ask claude to write the launch doc")
            _, fp = turn(q, seq, act=[start(), "I can't."])
            log = [p.payload for p in q.recent(1000)]
            return repr((log, fp.calls, fp.offers)).replace(str(s.root), "<store>")
        finally:
            s.close()

    unset = run()
    assert "delegate." not in unset and '"name": "delegate"' not in unset
    monkeypatch.setattr(act_mod, "_delegate_seat", lambda ctx: None)
    monkeypatch.setattr(delegate, "available", lambda: False)
    assert run() == unset


def test_with_no_worker_the_prompts_schemas_and_wire_are_byte_identical(
    q: EventQueue, live: Desk,
) -> None:
    seq = say(q, "hi")
    plain = ToolBox(store_root=q.store.root, append=q.append, prompted=True)
    unset = ToolBox(store_root=q.store.root, append=q.append, prompted=True, delegate=None)
    assert json.dumps(plain.schemas(), sort_keys=True) == json.dumps(unset.schemas(), sort_keys=True)
    assert tools.DELEGATE not in json.dumps(plain.schemas())
    # A turn record with no reply_to puts no channel on the wire.
    done = episodes.completed(for_seq=seq, outcome="spoke", reply="hi", at=AT)
    assert "reply_to" not in done
    q.append(done, episodes.turn_write_key(seq))
    [wire] = [u.wire() for u in projection.updates_since(q, seq) if u.kind == episodes.TURN_COMPLETED]
    assert "channel" not in wire
    assert discord.decide(wire, origin=lambda: None, at_the_mac=lambda: False) == "hi"


def test_the_judge_and_reply_name_the_hand_only_when_it_is_there(
    q: EventQueue, clock: Desk,
) -> None:
    seq = say(q, "ask claude to write the launch doc")
    _, off = turn(q, seq, act=["no"])
    judge_off = off.calls_for(provider.JUDGE)[0][0]["content"]
    assert "Claude Code" not in judge_off
    assert judge_off == turn_mod._JUDGE_SYSTEM
    clock.seen(NOW, ["/w"])
    seq = say(q, "ask claude to write the launch doc")
    _, on = turn(q, seq, act=["no"])
    judge_on = on.calls_for(provider.JUDGE)[0][0]["content"]
    assert "hand work to a Claude Code session on their Mac" in judge_on
    assert "ask Claude or Claude Code to make, write or fix something" in judge_on


def test_the_kinds_never_reach_the_wire_or_recall() -> None:
    assert episodes.DELEGATE_KINDS <= projection.NOT_PROJECTED
    assert episodes.DELEGATE_KINDS <= turn_mod.NOT_RECALLED



def test_the_relay_runs_the_worker_only_with_an_allowlist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from omega import __main__ as cli

    class FakeRelay:
        def __init__(self, address: Any, **kw: Any) -> None:
            pass

    started: list[Any] = []
    monkeypatch.setattr(cli.relaying, "Relay", FakeRelay)
    monkeypatch.setattr(delegate.Worker, "run", lambda self, stopping, every=0: started.append(self))
    env_file = tmp_path / ".env"
    env_file.write_text("")
    lines: list[str] = []
    assert cli.relay("127.0.0.1", 1, write=lines.append, wait=lambda e: e.set(), env_path=env_file) == 0
    assert not any("Claude Code" in line for line in lines)
    env_file.write_text(f"{delegate.ENV_DIRS}={tmp_path}\n")
    assert cli.relay("127.0.0.1", 1, write=lines.append, wait=lambda e: e.set(), env_path=env_file) == 0
    assert any(line == f"omega relay runs Claude Code sessions in {os.path.realpath(tmp_path)}" for line in lines)
    for _ in range(100):
        if started:
            break
        __import__("time").sleep(0.01)
    assert len(started) == 1
