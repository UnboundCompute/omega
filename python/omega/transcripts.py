"""Reading other agents' transcripts — DL-057.

omega has a mouth and a clock and, until this file, no senses. The heartbeat
ticks correctly every thirty seconds over an empty table and fires nothing,
forever, because a scheduler is not a source of things worth saying. This is
where something to say comes from.

**The cheapest sense with the highest signal is already on disk.** The person
works in coding agents all day and those write a full JSONL transcript per
session, locally, with no permission needed and no protocol to speak. That is a
record of what they work on, what they ask for, what they correct and what they
keep coming back to — which is exactly the object DL-054 says a belief about a
person is a conclusion from.

**A digest, not the transcript.** Sessions run to tens of megabytes; nothing
here is an optimisation of a version that fed the raw file to a model, because
that version cannot exist. :func:`digest` is mechanical, streams the file, and
renders the few hundred lines that carry signal about the *person*.

**What the digest deliberately leaves out is the whole security story.** A
transcript contains tool results: file contents, fetched pages, command output.
That is attacker-influenced text, and it is the first untrusted input to reach
omega's learning path (`docs/reference-implementations.md` §3). So the digest
reads what the person wrote and the *names* of what was done, and never the
bytes a tool returned. Not reading them is cheaper and stronger than reading
them carefully.

**The unit is a finished session.** Sessions are append-only files, so a live
one is a window that is still growing and re-reflecting over it would file the
same claim twice. :func:`discover` therefore skips anything touched recently,
and the caller records a receipt per session so each is read exactly once.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Optional

__all__ = [
    "Session",
    "SOURCE",
    "DEFAULT_ROOT",
    "IDLE_SECONDS",
    "MAX_SESSIONS_PER_PASS",
    "MAX_DIGEST_CHARS",
    "MAX_MESSAGES",
    "MAX_MESSAGE_CHARS",
    "MIN_MESSAGES",
    "default_root",
    "discover",
    "digest",
]

#: Which agent wrote these. Recorded on the receipt so a second reader can be
#: added later without the first one's receipts becoming ambiguous.
SOURCE = "claude-code"

#: Where that agent keeps them. One directory per project, one file per session.
DEFAULT_ROOT = Path(".claude") / "projects"

#: How long a file must sit untouched before it counts as a finished session.
#: Thirty minutes is long enough that a coffee break does not split one session
#: into two digests, and short enough that a morning's work is learned from
#: before the evening.
IDLE_SECONDS = 30 * 60

#: How many sessions one pass will read. The cap is the cost bound: each
#: session is one model call, and a laptop opened after a week away would
#: otherwise pay for fifty of them in the first idle moment. Unread sessions do
#: not expire, so a low cap delays learning rather than losing it.
MAX_SESSIONS_PER_PASS = 3

#: Ceiling on a rendered digest. Generous for a normal session and a hard stop
#: for a pathological one.
MAX_DIGEST_CHARS = 12_000

#: How many of the person's messages a digest carries. A session with more than
#: this many is summarised by its first and last, because the shape of a long
#: session is in how it started and where it ended up.
MAX_MESSAGES = 60

#: Per-message truncation. Long pastes are context for the coding agent, not
#: evidence about the person, and they are where the digest budget goes if
#: nothing stops them.
MAX_MESSAGE_CHARS = 400

#: Below this many messages a session is not evidence of anything. Filing a
#: habit from a two-message session is precisely the invention DL-054 was
#: written to prevent, and skipping it costs a model call rather than a belief.
MIN_MESSAGES = 4

#: Wrappers the harness injects into the user role. None of them are the person
#: talking, and a digest that carried them would teach omega about its own
#: tooling.
_NOT_THE_PERSON = (
    "<system-reminder>",
    "<command-name>",
    "<local-command-stdout>",
    "<command-message>",
    "Caveat: The messages below",
)


def default_root() -> Path:
    """Where transcripts live for this user."""
    return Path.home() / DEFAULT_ROOT


@dataclass(frozen=True)
class Session:
    """One finished transcript, as the filesystem describes it."""

    id: str
    project: str
    path: Path
    modified: datetime

    @property
    def source(self) -> str:
        return SOURCE


def discover(
    root: Optional[Path] = None,
    *,
    now: Optional[datetime] = None,
    idle_seconds: int = IDLE_SECONDS,
    seen: Iterable[str] = (),
) -> list[Session]:
    """Finished sessions not yet read, oldest first.

    Oldest first because learning in the order the work happened is the only
    order in which a later session can supersede an earlier claim rather than
    contradict it.

    ``seen`` is the set of session ids that already have a receipt. It is passed
    in rather than read here because the answer lives in omega's log and this
    module deliberately knows nothing about the log.
    """
    root = root or default_root()
    moment = now or datetime.now(timezone.utc)
    already = set(seen)
    out: list[Session] = []
    try:
        projects = sorted(p for p in root.iterdir() if p.is_dir())
    except OSError:
        # A missing root is the ordinary state on a machine where the agent has
        # never run. It is not an error and must not stop the pass that called
        # us — there may be a second source later.
        return []
    for project in projects:
        try:
            files = sorted(project.glob("*.jsonl"))
        except OSError:
            continue
        for path in files:
            if path.stem in already:
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            modified = datetime.fromtimestamp(stat.st_mtime, timezone.utc)
            if (moment - modified).total_seconds() < idle_seconds:
                continue  # still being written to
            if stat.st_size == 0:
                continue
            out.append(
                Session(
                    id=path.stem,
                    project=project.name,
                    path=path,
                    modified=modified,
                )
            )
    out.sort(key=lambda s: s.modified)
    return out


def _records(path: Path) -> Iterator[dict]:
    """Every JSON object in the file, skipping anything unreadable.

    Streamed, because these files reach tens of megabytes. Tolerant, because a
    transcript written by a process that was killed mid-write ends in a partial
    line, and that is a normal way for a session to end rather than a reason to
    learn nothing from the part that was written.
    """
    try:
        with path.open(errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                if isinstance(obj, dict):
                    yield obj
    except OSError:
        return


def _person_said(content: object) -> Optional[str]:
    """What the person typed in this message, or ``None`` if it was not them.

    The user role carries three different things: the person talking, tool
    results the harness fed back, and injected reminders. Only the first is
    evidence about the person, and a message carrying a tool result is not
    partially the person — the whole message is machinery.
    """
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_result":
                return None  # machinery, and untrusted: see the module docstring
            if block.get("type") == "text":
                parts.append(block.get("text") or "")
        text = "\n".join(parts)
    else:
        return None
    text = text.strip()
    if not text:
        return None
    if any(marker in text for marker in _NOT_THE_PERSON):
        return None
    return text


def _tools_used(content: object) -> Iterator[str]:
    """The names of the tools an assistant turn called. Names only — never the
    arguments, which carry file paths and command lines, and never the results.
    """
    if not isinstance(content, list):
        return
    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            name = block.get("name")
            if isinstance(name, str) and name:
                yield name


def _clip(text: str, limit: int = MAX_MESSAGE_CHARS) -> str:
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 1].rstrip() + "…"


def digest(session: Session, *, max_chars: int = MAX_DIGEST_CHARS) -> Optional[str]:
    """Render one session as the few hundred lines that say something about the
    person, or ``None`` if it says nothing.

    Returning ``None`` rather than an empty digest is *fail closed on empty*:
    a session too short to show a pattern must not reach a model that was asked
    to find one, because it will find one anyway.
    """
    said: list[str] = []
    tools: dict[str, int] = {}
    cwd = ""
    for record in _records(session.path):
        if not cwd and isinstance(record.get("cwd"), str):
            cwd = record["cwd"]
        message = record.get("message")
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if role == "user":
            text = _person_said(content)
            if text is not None:
                said.append(text)
        elif role == "assistant":
            for name in _tools_used(content):
                tools[name] = tools.get(name, 0) + 1

    if len(said) < MIN_MESSAGES:
        return None

    kept = said
    elided = 0
    if len(kept) > MAX_MESSAGES:
        half = MAX_MESSAGES // 2
        elided = len(kept) - MAX_MESSAGES
        kept = kept[:half] + kept[-half:]

    lines = [
        f"Working directory: {cwd or session.project}",
        f"Date: {session.modified.date().isoformat()}",
        f"The person wrote {len(said)} messages.",
    ]
    if tools:
        busiest = sorted(tools.items(), key=lambda kv: (-kv[1], kv[0]))[:8]
        lines.append(
            "Work done: " + ", ".join(f"{name} ×{count}" for name, count in busiest)
        )
    lines += ["", "What they asked for, in order:"]
    for index, text in enumerate(kept):
        if elided and index == MAX_MESSAGES // 2:
            lines.append(f"  … {elided} messages omitted …")
        lines.append(f"  - {_clip(text)}")

    rendered = "\n".join(lines)
    if len(rendered) > max_chars:
        rendered = rendered[:max_chars].rstrip() + "\n  … digest truncated …"
    return rendered
