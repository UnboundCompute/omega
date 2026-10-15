"""Hand work to a live Claude Code session on their Mac (DL-104).

"Ask Claude to make a doc about X", "get Claude to fix the flaky test in repo
Y": omega writes a brief, they say yes, and a background Claude Code session
starts on their Mac, in a folder they allowed. They can watch it
(``claude logs <id>``) and steer it (``claude attach <id>``) the whole time.
When it finishes, its final message and any files it left in the job's outbox
come back to the channel they asked on, where omega's other tools (Slack, a
handoff) can take them further.

**Two halves, one log.** The core holds the decisions and the records: a start
they said yes to is ``delegate.requested``, a follow-up or a stop is
``delegate.asked``, and each is a *command* until the Mac's receipt for it
lands. The Mac's worker (:class:`Worker`, a thread in the relay) polls for
those commands, because the channel only ever answers: it never calls out. It
runs ``claude --bg`` and reports back what the session did
(``delegate.updated``) and what it produced (``delegate.finished``). Every one
of those reaches them as a tell, delivered as written with no model call.

**The trust is on the Mac.** The worker runs only in a folder inside one of
``OMEGA_DELEGATE_DIRS``, read from the Mac's own environment or ``.env``,
checked after resolving symlinks; no list, no worker, and the feature does not
exist. It never passes ``bypassPermissions`` or
``--dangerously-skip-permissions``: the session runs in ``auto`` permission
mode, so Claude Code's own classifier still stands between it and anything
risky. It refuses a brief that does not carry omega's fixed rules (:data:`RULES`:
never pay, never push to main, never delete outside the job, deliverables in
the outbox), and runs at most :data:`MAX_RUNNING` at once.

**Offered by structure, never by words.** The ``delegate`` tool exists in a
turn only when it is their own message and a worker with delegation on has
polled within :data:`LIVE_SECONDS`. Every start is held for their yes, tainted
turn or not: it is a session that acts on their machine, so it is always theirs
to approve. With no worker, every prompt, schema, record and wire line is what
it was.

**Where the result comes from.** Claude Code's ``Stop`` hook hands the hook its
``last_assistant_message``, the final text of the turn, which the hooks
reference names as the way to read it. The transcript is the fallback only
(its format is not a documented interface), and ``claude logs`` is never read:
it is a terminal's redraws, not text.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from omega import blobs, episodes

__all__ = [
    "ENV_DIRS",
    "MAX_RUNNING",
    "POLL_SECONDS",
    "LIVE_SECONDS",
    "SOURCE",
    "RULES",
    "WORKER_OPS",
    "Desk",
    "desk",
    "available",
    "Seat",
    "Job",
    "Jobs",
    "jobs",
    "brief",
    "attach_hint",
    "roots",
    "inside",
    "resolve_folder",
    "Worker",
]

#: The Mac's allowlist: folders a session may work in, ``os.pathsep``-separated.
ENV_DIRS = "OMEGA_DELEGATE_DIRS"
#: How many sessions run at once. A third start is refused, plainly.
MAX_RUNNING = 2
#: How often the worker asks for commands and reads its hooks' events.
POLL_SECONDS = 5.0
#: A worker not heard from in this long is offline: four polls missed.
LIVE_SECONDS = 20.0
#: Who wrote what a session says: not omega and not them (DL-064).
SOURCE = "a Claude Code session on your Mac"
#: The largest file sent back from an outbox, and how many of them.
MAX_FILE_BYTES = 25 << 20
MAX_FILES = episodes.MAX_DELEGATE_FILES

#: Appended to every brief, in code. The Mac refuses a brief without it.
RULES = (
    "Rules from omega, which always apply: never pay, order, book or check out "
    "anything; if that is the next step, stop and give the link instead. Never "
    "push to main or master and never force-push. Never delete anything outside "
    "this job's own work. Put every file you make for them in the outbox folder "
    "named at the end of this brief. When you are done, end with a short summary "
    "of what you did."
)

#: The only ops the worker sends; checked in :meth:`Worker._send`.
WORKER_OPS = frozenset({"delegate.poll", "delegate.update", "delegate.result", "put"})

#: Notification types that mean the session is waiting on them.
_WAITING = frozenset({
    "permission_prompt", "elicitation_dialog", "elicitation_url_dialog", "agent_needs_input",
})
_BG_ID = re.compile(r"backgrounded\W+([0-9a-f]{6,})")
_UNTRUSTED_FOLDER = "Workspace not trusted"


def _now() -> datetime:
    """The clock seam: tests pin it."""
    return datetime.now(timezone.utc)


def attach_hint(session: Optional[str]) -> str:
    return f"Watch or steer it with `claude attach {session}`." if session else ""


# --- the core's view of the worker ---------------------------------------------


@dataclass
class Desk:
    """Is a worker with delegation on connected right now? In memory only:
    a fact about this process's last poll, never a record."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    last: Optional[datetime] = None
    folders: tuple[str, ...] = ()

    def seen(self, at: datetime, folders: Sequence[str]) -> None:
        with self.lock:
            self.last = at
            self.folders = tuple(str(f) for f in folders)

    def available(self, now: Optional[datetime] = None) -> bool:
        with self.lock:
            if self.last is None or not self.folders:
                return False
            age = ((now or _now()) - self.last).total_seconds()
            return 0 <= age <= LIVE_SECONDS


_DESK = Desk()


def desk() -> Desk:
    return _DESK


def available() -> bool:
    return _DESK.available()


@dataclass(frozen=True)
class Seat:
    """What a turn's box holds for ``delegate``: the desk, the jobs folded
    from the log when asked, and the channel this turn came in on."""

    desk: Desk
    jobs: Callable[[], "Jobs"]
    reply_to: str = ""


def brief(task: str) -> str:
    """The brief a session is started with: their task, then the rules."""
    return f"{task.strip()}\n\n{RULES}"


# --- the jobs, folded from the log ---------------------------------------------


@dataclass
class Job:
    job: str
    title: str
    folder: str
    worktree: bool
    reply_to: str = ""
    state: str = "requested"
    session: str = ""
    detail: str = ""
    text: str = ""
    files: tuple[dict[str, Any], ...] = ()


class Jobs:
    """Every job and every command still waiting for the Mac's receipt."""

    def __init__(self) -> None:
        self.by_id: dict[str, Job] = {}
        self._commands: dict[int, dict[str, Any]] = {}
        self._through = 0

    def apply(self, seq: int, payload: Mapping[str, Any]) -> None:
        kind = payload.get("kind")
        if kind not in episodes.DELEGATE_KINDS:
            return
        name = str(payload["job"])
        if kind == episodes.DELEGATE_REQUESTED:
            self.by_id[name] = Job(
                job=name, title=str(payload["title"]), folder=str(payload["folder"]),
                worktree=bool(payload["worktree"]), reply_to=str(payload.get("reply_to") or ""),
            )
            self._commands[seq] = {
                "op": "delegate.start", "cmd": seq, "job": name, "title": payload["title"],
                "brief": payload["brief"], "folder": payload["folder"],
                "worktree": payload["worktree"],
            }
            return
        job = self.by_id.get(name)
        if job is None:
            return
        if kind == episodes.DELEGATE_ASKED:
            self._commands[seq] = {"op": f"delegate.{payload['do']}", "cmd": seq, "job": name}
            if payload.get("text"):
                self._commands[seq]["text"] = payload["text"]
            return
        if kind == episodes.DELEGATE_UPDATED:
            job.state = str(payload["state"])
            job.session = str(payload.get("session") or job.session)
            job.detail = str(payload.get("detail") or "")
            self._commands.pop(int(payload.get("cmd") or 0), None)
            return
        job.state = "done"
        job.session = str(payload.get("session") or job.session)
        job.text = str(payload["text"])
        job.files = tuple(payload["files"])

    def advance(self, store: Any) -> "Jobs":
        for episode in store.episodes_since(self._through):
            self.apply(episode.seq, episodes.decode(episode.payload))
            self._through = episode.seq
        return self

    def pending(self) -> list[dict[str, Any]]:
        """The commands the Mac has not answered, oldest first."""
        return [dict(self._commands[seq]) for seq in sorted(self._commands)]

    def started_by(self, for_seq: int) -> int:
        return sum(1 for name in self.by_id if name.startswith(f"j{for_seq}-"))


def jobs(store: Any) -> Jobs:
    """The whole log, folded."""
    return Jobs().advance(store)


def tell_words(job: Job, payload: Mapping[str, Any]) -> Optional[str]:
    """What a Mac report says to them, or ``None`` when it says nothing."""
    title = job.title
    kind = payload.get("kind")
    hint = attach_hint(job.session)
    if kind == episodes.DELEGATE_FINISHED:
        said = str(payload["text"]).strip() or "(it finished without a final message)"
        files = "".join(
            f"\n[attached: {f['name']} {f['blob']}]" for f in payload["files"]
        )
        return f"Claude Code finished {title}:\n\n{said}{files}\n\n{hint}".rstrip()
    state = payload.get("state")
    detail = str(payload.get("detail") or "")
    if state == "started":
        return f"Claude Code started on your Mac: {title}. {hint}".rstrip()
    if state == "waiting":
        why = f": {detail}" if detail else ""
        return f"Claude Code is waiting on you for {title}{why}. {hint}".rstrip()
    if state == "failed":
        return f"Claude Code could not do {title}: {detail or 'it failed'}."
    if state == "stopped":
        return f"Claude Code stopped: {title}."
    if detail and payload.get("cmd"):
        # A command the Mac answered without changing anything, such as a
        # follow-up while the session is still working.
        return f"About {title}: {detail}"
    return None


def tell(job: Job, payload: Mapping[str, Any], at: str) -> Optional[dict[str, Any]]:
    """The tell inbound for a Mac report, sent back where they asked."""
    from omega import tools  # tools imports this module

    words = tell_words(job, payload)
    if words is None:
        return None
    return episodes.inbound(
        tools.tell_instruction(words), channel="schedule", at=at,
        untrusted=SOURCE, reply_to=job.reply_to or None,
    )


# --- the Mac's trust: the allowlist ----------------------------------------------


def roots(
    environ: Optional[Mapping[str, str]] = None, env_path: Optional[Path] = None,
) -> tuple[Path, ...]:
    """The allowed folders, resolved. Empty when unset: the feature is off.

    The environment wins over the file, which is parsed, never loaded. A
    folder that does not exist is dropped, never created.
    """
    from omega import provider

    env = os.environ if environ is None else environ
    raw = env.get(ENV_DIRS)
    if raw is None and env_path is not None:
        raw = provider.read_env(env_path).get(ENV_DIRS)
    out: list[Path] = []
    for part in (raw or "").split(os.pathsep):
        part = part.strip()
        if not part or not os.path.isabs(os.path.expanduser(part)):
            continue
        real = Path(os.path.realpath(os.path.expanduser(part)))
        if real.is_dir() and real not in out:
            out.append(real)
    return tuple(out)


def inside(path: Path, allowed: Sequence[Path]) -> bool:
    """Is ``path``, symlinks resolved, one of the allowed folders or under one?"""
    real = Path(os.path.realpath(path))
    return any(real == root or root in real.parents for root in allowed)


def resolve_folder(requested: str, allowed: Sequence[Path]) -> Optional[Path]:
    """The real folder a request names, or ``None`` when it is not allowed.

    An absolute path is taken as it is; a bare name is a folder of that name
    directly inside an allowed one, or an allowed one by its own name. Either
    way the answer is resolved and checked against the list: a symlink that
    points out of it is refused.
    """
    requested = (requested or "").strip()
    if not requested or not allowed:
        return None
    expanded = os.path.expanduser(requested)
    if os.path.isabs(expanded):
        candidates = [Path(expanded)]
    else:
        if "/" in requested or requested in (".", ".."):
            return None
        candidates = [root / requested for root in allowed]
        candidates += [root for root in allowed if root.name == requested]
    for candidate in candidates:
        if candidate.is_dir() and inside(candidate, allowed):
            return Path(os.path.realpath(candidate))
    return None


def _is_git(folder: Path) -> bool:
    return any((p / ".git").exists() for p in (folder, *folder.parents))


# --- the Mac's worker ------------------------------------------------------------

#: What the hooks run: append one line, ``{event, ...}``, to the job's events
#: file. Only the fields the worker reads are kept.
_HOOK = (
    "import json,sys\n"
    "try: d=json.load(sys.stdin)\n"
    "except Exception: d={}\n"
    "keep=('session_id','transcript_path','last_assistant_message','message',"
    "'notification_type','cwd')\n"
    "line={'event':sys.argv[2],**{k:d[k] for k in keep if k in d}}\n"
    "open(sys.argv[1],'a',encoding='utf-8').write(json.dumps(line)+'\\n')\n"
)

Runner = Callable[[Sequence[str], Path], tuple[int, str]]


def _run(argv: Sequence[str], cwd: Path) -> tuple[int, str]:
    """Run one ``claude`` command: its exit code and what it printed."""
    try:
        done = subprocess.run(
            list(argv), cwd=str(cwd), capture_output=True, text=True, timeout=120,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, str(exc)
    return done.returncode, (done.stdout or "") + (done.stderr or "")


def last_text(transcript: Path) -> str:
    """The last assistant text in a transcript: the fallback for a ``Stop``
    event that carried none. Its format is not a documented interface, so
    anything unreadable is empty rather than an error."""
    found = ""
    try:
        lines = transcript.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    for line in lines:
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if not isinstance(entry, dict) or entry.get("type") != "assistant":
            continue
        content = (entry.get("message") or {}).get("content") or []
        texts = [c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"]
        if any(t.strip() for t in texts):
            found = "\n".join(t for t in texts if t.strip())
    return found


class Worker:
    """The Mac half. Each :meth:`pass_once` asks for commands, carries them
    out, and reports what each session's hooks wrote since the last pass.

    Every outside effect is injectable: the connection, the ``claude``
    runner, the clock and the log line. Its state per job is one small JSON
    file in the job's folder, written before anything is started, so a
    restarted worker never starts a job twice.
    """

    def __init__(
        self,
        address: tuple[str, int],
        *,
        allowed: Sequence[Path],
        home: Path,
        run: Runner = _run,
        connect: Optional[Callable[[tuple[str, int]], Any]] = None,
        python: str = sys.executable,
        cap: int = MAX_RUNNING,
        log: Callable[[str], None] = lambda line: None,
    ) -> None:
        if not allowed:
            raise ValueError(f"no {ENV_DIRS}: delegation is off")
        if connect is None:
            from omega.channel import ChannelClient

            connect = lambda a: ChannelClient(a, timeout=30.0)  # noqa: E731
        self._address = address
        self._allowed = tuple(allowed)
        self._home = Path(home)
        self._run = run
        self._connect = connect
        self._python = python
        self._cap = cap
        self._note = log

    # --- the wire ---------------------------------------------------------

    def _send(self, client: Any, op: str, **fields: Any) -> dict[str, Any]:
        if op not in WORKER_OPS:
            raise RuntimeError(f"the delegate worker does not send {op!r}")
        if op == "put":
            return client.put(fields["upload"], fields["name"], fields["data"])
        return client.request({"v": 1, "op": op, **fields})

    def _update(self, client: Any, state: dict[str, Any], status: str, *, key: str,
                detail: str = "", cmd: Optional[int] = None) -> bool:
        """One report, under an id made of what it answers (a command, or a
        line of the events file), so a resend is always the duplicate path."""
        answer = self._send(
            client, "delegate.update", id=f"{state['job']}-{key}",
            job=state["job"], state=status, session=state.get("session") or None,
            detail=detail[: episodes.MAX_DELEGATE_DETAIL] or None, cmd=cmd,
        )
        ok = answer.get("op") == "ack"
        if not ok:
            self._note(f"delegate: update for {state['job']} refused: {answer.get('error')}")
        return ok

    # --- the per-job state file --------------------------------------------

    def _dir(self, job: str) -> Path:
        return self._home / job

    def _load(self, job: str) -> Optional[dict[str, Any]]:
        try:
            return json.loads((self._dir(job) / "job.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _save(self, state: dict[str, Any]) -> None:
        path = self._dir(state["job"]) / "job.json"
        tmp = path.with_name("job.json.tmp")
        tmp.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)

    def _states(self) -> list[dict[str, Any]]:
        if not self._home.is_dir():
            return []
        out = []
        for child in sorted(self._home.iterdir()):
            state = self._load(child.name) if child.is_dir() else None
            if state is not None:
                out.append(state)
        return out

    def _running(self) -> int:
        return sum(1 for s in self._states() if s.get("state") in ("starting", "started", "waiting"))

    # --- one pass ---------------------------------------------------------

    def pass_once(self) -> int:
        """Poll, act, report. Returns how many commands were carried out.
        Never raises: the next pass tries again."""
        try:
            client = self._connect(self._address)
        except OSError as exc:
            self._note(f"delegate: cannot reach the core at {self._address}: {exc}")
            return 0
        done = 0
        try:
            answer = self._send(
                client, "delegate.poll", folders=[str(p) for p in self._allowed], cap=self._cap,
            )
            for command in answer.get("commands") or []:
                try:
                    done += self._command(client, command)
                except (OSError, ValueError, KeyError, RuntimeError) as exc:
                    self._note(f"delegate: command {command.get('cmd')} failed: {exc}")
            for state in self._states():
                if state.get("state") in ("started", "waiting", "done"):
                    self._events(client, state)
        except Exception as exc:  # noqa: BLE001 - a background thread nobody watches
            self._note(f"delegate: pass failed: {exc}")
        finally:
            try:
                client.close()
            except Exception:  # noqa: BLE001
                pass
        return done

    def _command(self, client: Any, command: dict[str, Any]) -> int:
        cmd = int(command["cmd"])
        job = str(command["job"])
        if not re.match(r"^j[0-9]{1,12}-[0-9]{1,3}$", job):
            raise ValueError(f"not a job id: {job!r}")
        state = self._load(job)
        if state is not None and cmd in state.get("cmds", []):
            # Carried out already and its receipt was lost: send it again. A
            # start cut off mid-launch is not tried twice: it is reported lost.
            detail = ""
            if state["state"] == "starting":
                state["state"] = "failed"
                detail = "omega lost track of this start; `claude agents` shows what is running"
                self._save(state)
            return int(self._update(client, state, state["state"], key=f"c{cmd}", detail=detail, cmd=cmd))
        op = command.get("op")
        if op == "delegate.start":
            return self._start(client, command, cmd)
        if state is None:
            # A job this Mac has no file for: answered, so it is not asked again.
            gone = {"job": job, "session": ""}
            return int(self._update(client, gone, "failed", key=f"c{cmd}",
                                    detail="this Mac has no record of that job", cmd=cmd))
        if op == "delegate.stop":
            return self._stop(client, state, cmd)
        if op == "delegate.followup":
            return self._followup(client, state, cmd, str(command.get("text") or ""))
        raise ValueError(f"unknown command {op!r}")

    def _where(self, state: dict[str, Any]) -> Path:
        """Where a session ran, as its hooks said (its worktree, when it has
        one), so it is resumed where Claude Code filed it; only ever a folder
        the allowlist holds, else the job's own folder."""
        cwd = state.get("cwd")
        if cwd and Path(cwd).is_dir() and inside(Path(cwd), self._allowed):
            return Path(os.path.realpath(cwd))
        return Path(state["folder"])

    def _launch(self, state: dict[str, Any], prompt: str, *, resume: Optional[str] = None,
                cwd: Optional[Path] = None) -> tuple[bool, str]:
        """Start (or resume) the session in the background: ok and its id, or
        not and why."""
        folder = cwd or Path(state["folder"])
        jobdir = self._dir(state["job"])
        events = jobdir / "events.jsonl"
        hook = lambda event: shlex.join([self._python, "-I", "-c", _HOOK, str(events), event])  # noqa: E731
        settings = json.dumps({"hooks": {
            "Stop": [{"hooks": [{"type": "command", "command": hook("stop")}]}],
            "Notification": [{"hooks": [{"type": "command", "command": hook("notification")}]}],
        }})
        argv = ["claude"]
        if resume:
            argv += ["--resume", resume]
        argv += ["--bg", "-n", f"omega: {state['title']}"]
        if state.get("worktree") and not resume:
            argv += ["--worktree", f"omega-{state['job']}"]
        argv += [
            "--add-dir", str(jobdir / "outbox"),
            "--permission-mode", "auto",
            "--settings", settings,
            prompt,
        ]
        code, out = self._run(argv, folder)
        if _UNTRUSTED_FOLDER in out:
            return False, (
                f"Claude Code does not trust {folder} yet: run `claude` there once "
                "and accept the trust prompt, then ask again"
            )
        found = _BG_ID.search(out)
        if code != 0 or found is None:
            return False, (out.strip().splitlines() or [f"claude exited {code}"])[-1][:300]
        return True, found.group(1)

    def _start(self, client: Any, command: dict[str, Any], cmd: int) -> int:
        job = str(command["job"])
        title = str(command.get("title") or job)[: episodes.MAX_DELEGATE_TITLE]
        state: dict[str, Any] = {
            "job": job, "title": title, "folder": "", "worktree": False, "state": "starting",
            "cmds": [cmd], "session": "", "session_id": "", "read": 0, "shipped": [],
        }

        def refuse(why: str) -> int:
            state["state"] = "failed"
            self._dir(job).mkdir(parents=True, exist_ok=True)
            self._save(state)
            return int(self._update(client, state, "failed", key=f"c{cmd}", detail=why, cmd=cmd))

        folder = resolve_folder(str(command.get("folder") or ""), self._allowed)
        if folder is None:
            return refuse(
                f"{command.get('folder')!r} is not inside a folder this Mac allows "
                f"({ENV_DIRS}), so nothing started"
            )
        text = str(command.get("brief") or "")
        if RULES not in text:
            return refuse("the brief is missing omega's fixed rules, so nothing started")
        if self._running() >= self._cap:
            self._reap(client)
        if self._running() >= self._cap:
            return refuse(
                f"{self._cap} Claude Code sessions are already running; stop one or wait, then ask again"
            )
        state["folder"] = str(folder)
        state["worktree"] = bool(command.get("worktree")) and _is_git(folder)
        jobdir = self._dir(job)
        (jobdir / "outbox").mkdir(parents=True, exist_ok=True)
        # Written before the launch: a worker that dies mid-start finds this
        # and reports it lost, rather than starting the job a second time.
        self._save(state)
        ok, said = self._launch(state, f"{text}\n\nOutbox folder: {jobdir / 'outbox'}")
        if not ok:
            return refuse(said)
        state["state"], state["session"] = "started", said
        self._save(state)
        return int(self._update(client, state, "started", key=f"c{cmd}", cmd=cmd))

    def _reap(self, client: Any) -> None:
        """Free the slots of sessions that ended without a ``Stop`` hook (one
        stopped by hand, say), read from ``claude agents --json --all``. Asked
        only when the cap is reached; an answer it cannot read frees none."""
        code, out = self._run(["claude", "agents", "--json", "--all"], self._home)
        try:
            listed = json.loads(out) if code == 0 else None
        except ValueError:
            listed = None
        if not isinstance(listed, list):
            return
        # Listed and not ended counts as alive; ``done`` stays its hook's call.
        alive = {
            str(a.get("id")) for a in listed
            if isinstance(a, dict) and a.get("state") not in ("stopped", "failed", "exited", "killed")
        }
        for state in self._states():
            if state.get("state") in ("started", "waiting") and state.get("session") not in alive:
                state["state"] = "stopped"
                self._save(state)
                self._update(client, state, "stopped", key=f"gone-{state.get('session', '')}",
                             detail="the session is no longer running")

    def _stop(self, client: Any, state: dict[str, Any], cmd: int) -> int:
        if state.get("session"):
            self._run(["claude", "stop", state["session"]], Path(state["folder"] or self._home))
        state["state"] = "stopped"
        state["cmds"] = [*state.get("cmds", []), cmd]
        self._save(state)
        return int(self._update(client, state, "stopped", key=f"c{cmd}", cmd=cmd))

    def _followup(self, client: Any, state: dict[str, Any], cmd: int, text: str) -> int:
        state["cmds"] = [*state.get("cmds", []), cmd]
        if state.get("state") != "done" or not state.get("session"):
            self._save(state)
            how = (
                f"it's still working; open it with `claude attach {state['session']}` to steer"
                if state.get("state") in ("started", "waiting") and state.get("session")
                else "that session is not running any more, so nothing was sent"
            )
            return int(self._update(client, state, state.get("state") or "failed", key=f"c{cmd}",
                                    detail=how, cmd=cmd))
        # Idle and done: stop the waiting process, then continue the same
        # session in the background under the same id.
        self._run(["claude", "stop", state["session"]], Path(state["folder"]))
        ok, said = self._launch(state, text, resume=state.get("session_id") or state["session"],
                                cwd=self._where(state))
        if not ok:
            self._save(state)
            return int(self._update(client, state, "done", key=f"c{cmd}",
                                    detail=f"the follow-up did not start: {said}", cmd=cmd))
        state["state"], state["session"] = "started", said
        self._save(state)
        return int(self._update(client, state, "started", key=f"c{cmd}", cmd=cmd))

    # --- what the hooks wrote ---------------------------------------------

    def _events(self, client: Any, state: dict[str, Any]) -> None:
        path = self._dir(state["job"]) / "events.jsonl"
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return
        start = int(state.get("read", 0))
        for n, line in enumerate(lines[start:], start=start):
            try:
                event = json.loads(line)
            except ValueError:
                event = {}
            if event.get("session_id"):
                state["session_id"] = str(event["session_id"])
            if event.get("cwd"):
                state["cwd"] = str(event["cwd"])
            if event.get("event") == "stop":
                if not self._finished(client, state, event, key=f"r{n}"):
                    return  # unsent: read again next pass
            elif (
                event.get("event") == "notification"
                and event.get("notification_type") in _WAITING
                and state.get("state") == "started"
            ):
                if not self._update(client, state, "waiting", key=f"e{n}",
                                    detail=str(event.get("message") or "")):
                    return
                state["state"] = "waiting"
            state["read"] = n + 1
            self._save(state)

    def _finished(self, client: Any, state: dict[str, Any], event: dict[str, Any], *, key: str) -> bool:
        text = str(event.get("last_assistant_message") or "")
        if not text.strip() and event.get("transcript_path"):
            text = last_text(Path(str(event["transcript_path"])))
        files = self._ship(client, state)
        answer = self._send(
            client, "delegate.result", id=f"{state['job']}-{key}",
            job=state["job"], text=text[: episodes.MAX_DELEGATE_TEXT], files=files,
            session=state.get("session") or None,
        )
        if answer.get("op") != "ack":
            self._note(f"delegate: result for {state['job']} refused: {answer.get('error')}")
            return False
        state["state"] = "done"
        return True

    def _ship(self, client: Any, state: dict[str, Any]) -> list[dict[str, Any]]:
        """Upload the outbox's new files: regular files only, never a link,
        each under the size cap, at most :data:`MAX_FILES` per result."""
        outbox = self._dir(state["job"]) / "outbox"
        shipped = set(state.get("shipped", []))
        out: list[dict[str, Any]] = []
        try:
            entries = sorted(outbox.iterdir())
        except OSError:
            return out
        for entry in entries:
            if len(out) >= MAX_FILES:
                break
            if entry.name.startswith(".") or entry.is_symlink() or not entry.is_file():
                continue
            size = entry.stat().st_size
            mark = f"{entry.name}:{size}:{int(entry.stat().st_mtime)}"
            if mark in shipped or size > MAX_FILE_BYTES:
                continue
            upload = re.sub(r"[^A-Za-z0-9_-]", "-", f"{state['job']}-{len(shipped) + len(out)}")[:64]
            # The name a document marker can carry (:data:`omega.handoff.MARKER`),
            # so a channel that delivers files can swap the marker for this one.
            name = re.sub(r"[^A-Za-z0-9._ -]", "-", entry.name)[:80].strip() or "file"
            answer = self._send(client, "put", upload=upload, name=name, data=entry.read_bytes())
            if answer.get("op") != "attached":
                continue
            out.append({"name": name, "blob": answer["blob"], "mime": blobs.mime_for(entry),
                        "bytes": int(answer["bytes"])})
            shipped.add(mark)
        state["shipped"] = sorted(shipped)
        return out

    # --- the loop ---------------------------------------------------------

    def run(self, stopping: threading.Event, *, every: float = POLL_SECONDS) -> None:
        while not stopping.is_set():
            self.pass_once()
            stopping.wait(every)


def start_worker(
    address: tuple[str, int], *, environ: Optional[Mapping[str, str]] = None,
    env_path: Optional[Path] = None, home: Optional[Path] = None,
    stopping: threading.Event, log: Callable[[str], None],
) -> Optional[threading.Thread]:
    """The worker's thread in the relay, or ``None`` with no allowlist."""
    allowed = roots(environ, env_path)
    if not allowed:
        return None
    worker = Worker(
        address, allowed=allowed, home=home or Path.home() / ".omega" / "delegate", log=log,
    )
    thread = threading.Thread(target=worker.run, args=(stopping,), name="omega-delegate", daemon=True)
    thread.start()
    return thread

