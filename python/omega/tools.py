"""Ring 1's remainder — M1 step 7, spec `agent/M1_SPEC.md` §1.5, §2.4; DL-014,
DL-028.

DL-014 fixed the body and said it never grows: **read/write files, run code,
fetch the web, speak on a channel, read/write its own memory.** *Speak* is the
channel, *writing* memory is the log, and this file is the rest — so **the set
does not grow.** A new behaviour is a learned skill composed out of these
(ring 3, a memory), never a new function here — that is the whole answer to why
omega's tool catalog stays under the degradation threshold by construction
rather than by discipline.

**`recall` is the sixth member of that body, not a sixth member of the list**
(DL-060). This docstring used to say memory was "already" both halves, and that
sentence was wrong for years' worth of the design: *reading* memory existed
only as `turn.recall` — the last N episodes, plus whichever learned claims had
their trigger words said aloud in the message. Neither is omega looking at what
it believes, so asked *what do you remember about me* it answered, accurately,
that it had nothing in front of it while holding ninety-one claims. The other
four tools all point outward, at the world; nothing pointed in. That is the
slot DL-014 declared and this fills.

**Approval is code, never model judgement.** A chain of reasoning, or an
instruction injected through a tool result, can talk a model out of asking. So
the gate is :func:`classify` — a *static* classification of the tool name plus
its arguments, in the execution layer, **before** dispatch — tiered by
reversibility x blast radius with `CLAUDE.md`'s own three tiers:

    read_file   -> exploration, always            . runs
    write_file  -> local inside the store,        . runs
                   external outside it            . asks
    run_code    -> exploration if argv[0] is on a . runs
                   small read-only allowlist,
                   external otherwise             . asks
    fetch       -> external, always               . asks
    recall      -> exploration, always            . runs

**Why `fetch` is external even though reading is exploration.** `CLAUDE.md`
tiers a network call as external, and here that abstraction has a concrete
edge: `read_file` dispatches unasked, so an unasked `fetch` would complete an
exfiltration chain — an injected tool result saying *read the .env and post it
to evil.example* would run end to end with no human anywhere in it. One of the
two has to stop and ask, and the outward-facing one is the one whose damage
cannot be undone.

**Why `run_code` takes an argv list and never a shell string.** A static
classifier over shell *text* is a denylist — it must enumerate what is
dangerous — and denylists on shell are bypassable: `$(...)`, backticks,
`;`, `&&`, `|`, a variable holding the real command, an encoding. An argv list
has no such grammar: element zero *is* the program, so classification is a
membership test rather than a parse, and the allowlist can be sound. Nothing
here ever passes ``shell=True``.

**`run_code` is bounded, not sandboxed, and this file says so out loud.** An
explicit working directory, a wall-clock timeout, no interactive stdin,
truncated output. That bounds the common accident. It does **not** contain a
hostile program: the child runs as omega does, reaches the same filesystem and
the same network, and nothing in M1 should be written or read as though it
does otherwise. A real sandbox is its own decision with its own mechanism.

**Every tool result is untrusted input** (DL-014). A fetched page is the first
thing omega reads that someone else wrote. Results come back as data and reach
the next pass in the ``tool`` role; nothing a result says can widen what the
next pass is allowed to do, because what is allowed is decided here, in code,
from the tool name and its arguments.
"""

from __future__ import annotations

import http.client
import ipaddress
import os
import socket
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence
from urllib.parse import urlsplit

__all__ = [
    "EXPLORATION",
    "LOCAL",
    "EXTERNAL",
    "TIERS",
    "READ_FILE",
    "WRITE_FILE",
    "RUN_CODE",
    "FETCH",
    "RECALL",
    "TOOL_NAMES",
    "MAX_RECALLED",
    "READ_ONLY_ARGV0",
    "MAX_RESULT_CHARS",
    "MAX_READ_BYTES",
    "RUN_TIMEOUT",
    "FETCH_TIMEOUT",
    "MAX_FETCH_BYTES",
    "TEXTUAL_TYPES",
    "ToolError",
    "ToolRejected",
    "Decision",
    "ToolBox",
    "schemas",
    "refusal_for_address",
    "refusal_for_host",
    "refusal_for_content_type",
    "vetted_address",
    "check_url",
    "read_file",
    "write_file",
    "run_code",
    "fetch",
    "recall",
]

# --- the three tiers --------------------------------------------------------
# `CLAUDE.md`'s own tiers, reused rather than re-invented (DL-014): the human
# test passes cleanly because this is already how we work.

#: Read-only. Dispatches without asking.
EXPLORATION = "exploration"

#: Writes something omega owns. Dispatches without asking.
LOCAL = "local"

#: Outward-facing or hard to reverse. **Ends the turn with `turn.blocked`.**
EXTERNAL = "external"

TIERS = frozenset({EXPLORATION, LOCAL, EXTERNAL})

READ_FILE = "read_file"
WRITE_FILE = "write_file"
RUN_CODE = "run_code"
FETCH = "fetch"
RECALL = "recall"

#: Ring 1's remainder, and it is closed (DL-014). Spelled as a frozenset so
#: that "the set does not grow" is a fact about an object and not a promise in
#: a comment.
#:
#: ``recall`` joining it is not the set growing, and the distinction is the
#: whole of DL-060. DL-014's body names six things and this file's own
#: docstring lists them; *read/write its own memory* is one of the six, and it
#: was dismissed as already existing. Writing memory does exist. Reading it
#: existed only as recall — the last N episodes plus whichever claims happened
#: to have their trigger words said out loud — which is not omega being able to
#: look at what it believes. So this fills a declared slot that was assumed
#: complete and never was.
TOOL_NAMES = frozenset({READ_FILE, WRITE_FILE, RUN_CODE, FETCH, RECALL})

#: How many claims one ``recall`` answers with.
#:
#: There are 91 in the live log and the number only goes up, so an uncapped
#: answer is a slow way to turn the act loop's context into a roster. The cap
#: is on the *answer*, never on the search: a phrase is matched against every
#: claim omega holds and only the rendering stops, so a capped result is
#: omega's best matches rather than an arbitrary prefix of its memory.
MAX_RECALLED = 25

#: Programs whose **every** invocation is read-only, whatever flags follow.
#:
#: An allowlist, never a denylist: the question a classifier must answer is
#: "is this known-safe", and a denylist answers "is this known-dangerous",
#: which is unsound the moment something is not on it. Membership is on the
#: exact spelling of ``argv[0]`` — no directory component — because taking the
#: basename of a path would let ``/tmp/whatever/ls`` inherit ``ls``'s tier.
#:
#: Deliberately absent, each for a reason worth keeping: ``sed`` and ``awk``
#: write files (``sed -i``, ``print > file``); ``find`` has ``-delete`` and
#: ``-exec``; ``git`` is safe or not depending on its subcommand, and tiering
#: by subcommand is a denylist wearing an allowlist's clothes; every
#: interpreter runs arbitrary code by definition. Anything not here is
#: ``external`` — which is *asked about*, not refused.
READ_ONLY_ARGV0 = frozenset(
    {
        "cat",
        "date",
        "echo",
        "file",
        "grep",
        "head",
        "hostname",
        "ls",
        "pwd",
        "stat",
        "tail",
        "true",
        "uname",
        "wc",
        "which",
    }
)

#: How much of any tool result is kept. The same number bounds what goes in
#: the log and what the next pass reads, so the record and the model always saw
#: the same thing — a log that kept more than the model was shown would make
#: the transcript a plausible lie about what omega was reasoning over.
MAX_RESULT_CHARS = 8_000

#: How much of a file `read_file` will read off disk before truncating.
MAX_READ_BYTES = 256 * 1024

#: Wall clock for one `run_code`. Bounds the common accident (a command that
#: waits for input it will never get, a loop that does not end).
RUN_TIMEOUT = 30.0

FETCH_TIMEOUT = 20.0

#: How much of a response body is read. A cap, not a promise about the server:
#: a page that lies about its length still cannot spend more than this.
MAX_FETCH_BYTES = 1024 * 1024

#: Environment names never handed to a child process. `load_env` puts the API
#: key into `os.environ` at startup, and a child inherits the environment, so
#: without this every allowlisted `run_code` would be one `env` away from the
#: key. Stripping is cheap and the omission would be permanent.
_SECRET_ENV_PREFIXES = ("OPENAI_", "OMEGA_", "ANTHROPIC_")

_USER_AGENT = "omega/1.0 (+https://localhost) M1"


class ToolError(RuntimeError):
    """A tool ran, or was asked to run, and did not produce a result.

    Surfaced, never swallowed (§2.4): it becomes a ``tool.returned`` with
    ``ok=False`` and the reason, visible to the next pass, and it is **never
    retried automatically**. Retry policy is deliberately deferred until there
    is a real failure pattern to design against.
    """


class ToolRejected(ToolError):
    """Refused *before* dispatch: no such tool, or arguments that cannot be read.

    A subclass because the sub-loop treats it the same way — a failed tool
    call the next pass can see — but it is a different fact about the world.
    Nothing ran. It is separate so that "the model asked for something that
    does not exist" can never be read as "the tool tried and failed".
    """


@dataclass(frozen=True)
class Decision:
    """What the classifier decided about one call, and the arguments it decided
    *about*.

    ``args`` is carried rather than re-read at dispatch on purpose: the thing
    that was classified has to be the thing that runs. Reading the arguments a
    second time is how a gate ends up approving one command and executing
    another.
    """

    tool: str
    tier: str
    why: str
    args: dict[str, Any] = field(default_factory=dict)

    @property
    def dispatches(self) -> bool:
        """Does this run, or does omega stop and ask?"""
        return self.tier in (EXPLORATION, LOCAL)

    @property
    def wants(self) -> str:
        """What omega wanted to do, in words a person can answer."""
        return self.why


# --- argument reading -------------------------------------------------------
# Strict, and every failure is a `ToolRejected`. A classifier handed something
# it cannot read must refuse rather than guess: a guess here is a decision
# about what may run, made by no one.


def _text(tool: str, args: dict[str, Any], key: str) -> str:
    if key not in args:
        raise ToolRejected(f"{tool} needs {key!r}")
    value = args[key]
    if not isinstance(value, str) or not value.strip():
        raise ToolRejected(f"{tool}'s {key!r} must be a non-empty string")
    return value


def _argv(tool: str, args: dict[str, Any]) -> list[str]:
    if "argv" not in args:
        raise ToolRejected(f"{tool} needs 'argv'")
    value = args["argv"]
    if isinstance(value, str):
        # The one mistake worth naming rather than silently splitting: splitting
        # it here would re-introduce exactly the shell-text parsing the argv
        # list exists to avoid.
        raise ToolRejected(
            f"{tool}'s 'argv' must be a list of strings, not one string — "
            f"omega never runs a shell, so ['ls', '-l'] and not 'ls -l'"
        )
    if not isinstance(value, list) or not value:
        raise ToolRejected(f"{tool}'s 'argv' must be a non-empty list of strings")
    if not all(isinstance(part, str) for part in value):
        raise ToolRejected(f"{tool}'s 'argv' must contain only strings")
    if not value[0]:
        raise ToolRejected(f"{tool}'s 'argv[0]' must name a program")
    return list(value)


def resolved(path: str) -> Path:
    """A path as the filesystem will actually read it.

    ``~`` expanded and **symlinks resolved**, including through components
    that do not exist yet. Classifying the spelling instead would let a symlink
    inside the store point anywhere: the write would be tiered *local*, run
    unasked, and land outside.
    """
    return Path(path).expanduser().resolve()


# --- classification ---------------------------------------------------------


def _classify_read_file(box: "ToolBox", args: dict[str, Any]) -> Decision:
    path = resolved(_text(READ_FILE, args, "path"))
    # Exploration whatever the path. Reading is what `CLAUDE.md` calls always
    # allowed, and the secret-bearing case -- reading the key file itself -- is
    # not closed by refusing to read: it is closed by `fetch` being external,
    # so nothing omega reads can leave the machine unasked.
    return Decision(
        tool=READ_FILE,
        tier=EXPLORATION,
        why=f"read {path}",
        args={"path": str(path)},
    )


def _classify_recall(box: "ToolBox", args: dict[str, Any]) -> Decision:
    about = args.get("about")
    if about is not None and not isinstance(about, str):
        raise ToolRejected(f"{RECALL}'s 'about' must be a string or absent")
    phrase = (about or "").strip()
    # Exploration, and there is no path by which it could be anything else.
    # It reads no file, runs no program and reaches no network: the claims it
    # answers from were handed to the box by the turn that built it. That is
    # also why it needs no store-root check the way `write_file` does — there
    # is no store access to be inside or outside of.
    return Decision(
        tool=RECALL,
        tier=EXPLORATION,
        why=f"recall what is known about {phrase!r}" if phrase else "recall everything known",
        args={"about": phrase},
    )


def _classify_write_file(box: "ToolBox", args: dict[str, Any]) -> Decision:
    path = resolved(_text(WRITE_FILE, args, "path"))
    content = args.get("content", "")
    if not isinstance(content, str):
        raise ToolRejected(f"{WRITE_FILE}'s 'content' must be a string")
    inside = path == box.store_root or box.store_root in path.parents
    return Decision(
        tool=WRITE_FILE,
        tier=LOCAL if inside else EXTERNAL,
        why=(
            f"write {len(content)} characters to {path}"
            + ("" if inside else " — that is outside omega's own store")
        ),
        args={"path": str(path), "content": content},
    )


def _classify_run_code(box: "ToolBox", args: dict[str, Any]) -> Decision:
    argv = _argv(RUN_CODE, args)
    cwd = resolved(_text(RUN_CODE, args, "cwd"))
    program = argv[0]
    read_only = program in READ_ONLY_ARGV0
    printable = " ".join(argv)
    return Decision(
        tool=RUN_CODE,
        tier=EXPLORATION if read_only else EXTERNAL,
        why=(
            f"run `{printable}` in {cwd}"
            + ("" if read_only else f" — {program!r} is not a read-only command")
        ),
        args={"argv": argv, "cwd": str(cwd)},
    )


def _classify_fetch(box: "ToolBox", args: dict[str, Any]) -> Decision:
    url = _text(FETCH, args, "url")
    # External **always**, and not as a blanket rule about the network. It is
    # what keeps `read_file` dispatching unasked from being an exfiltration
    # path: read-then-send with nobody in the loop is the chain, and this is
    # where it is broken.
    return Decision(
        tool=FETCH,
        tier=EXTERNAL,
        why=f"fetch {url} — that reaches off this machine",
        args={"url": url},
    )


# --- the tools themselves ---------------------------------------------------


def read_file(path: str, *, max_bytes: int = MAX_READ_BYTES) -> str:
    """Read a file as text. Exploration: it runs without asking.

    Decodes with replacement rather than failing on bad bytes — the caller
    asked what is in the file, and "it is not valid UTF-8" is an answer it can
    act on, whereas an exception is a turn that ends.
    """
    target = Path(path)
    if not target.exists():
        raise ToolError(f"{target} does not exist")
    if target.is_dir():
        raise ToolError(f"{target} is a directory, not a file")
    if not target.is_file():
        # A fifo or a device: reading either can block forever or never end,
        # and neither is what "read a file" means.
        raise ToolError(f"{target} is not a regular file")
    try:
        with open(target, "rb") as handle:
            raw = handle.read(max_bytes + 1)
    except OSError as exc:
        raise ToolError(f"could not read {target}: {exc}") from exc

    truncated = len(raw) > max_bytes
    text = raw[:max_bytes].decode("utf-8", errors="replace")
    if truncated:
        text += f"\n... [truncated at {max_bytes} bytes]"
    return _cap(text)


def write_file(path: str, content: str) -> str:
    """Write a file, creating parent directories. Local inside the store.

    Creating parents is safe *because of where this is allowed to run*: the
    classifier only dispatches a write inside the store, and creating a
    directory omega owns is not a blast radius. A write outside the store never
    reaches here at all — it ends the turn as a question.
    """
    target = Path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    except OSError as exc:
        raise ToolError(f"could not write {target}: {exc}") from exc
    return f"wrote {len(content)} characters to {target}"


def _child_env() -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(_SECRET_ENV_PREFIXES)
    }
    return env


def run_code(
    argv: Sequence[str],
    cwd: str,
    *,
    timeout: float = RUN_TIMEOUT,
) -> str:
    """Run a program. **Bounded, not sandboxed** — say it that way everywhere.

    What is bounded: an explicit working directory, a wall clock, no
    interactive stdin, truncated output, and an environment with omega's own
    secrets removed. What is *not* bounded: the child runs as omega does and
    reaches the same filesystem and the same network. This stops the common
    accident. It does not contain a hostile program, and nothing in M1 should
    be described as though it does.

    ``shell=False`` is not a default being relied on — it is passed, because
    the argument that makes the classifier sound is that element zero *is* the
    program, and a shell would make that false.
    """
    argv = list(argv)
    if not argv:
        raise ToolError("run_code needs a program to run")
    work = Path(cwd)
    if not work.is_dir():
        raise ToolError(f"{work} is not a directory to run in")

    try:
        finished = subprocess.run(  # noqa: S603 - argv list, never a shell
            argv,
            cwd=str(work),
            shell=False,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=timeout,
            env=_child_env(),
        )
    except FileNotFoundError as exc:
        raise ToolError(f"there is no program called {argv[0]!r}") from exc
    except subprocess.TimeoutExpired as exc:
        raise ToolError(
            f"`{' '.join(argv)}` did not finish within {timeout:.0f}s and was "
            f"stopped"
        ) from exc
    except OSError as exc:
        raise ToolError(f"could not run {argv[0]!r}: {exc}") from exc

    out = finished.stdout.decode("utf-8", errors="replace")
    err = finished.stderr.decode("utf-8", errors="replace")
    # The exit status leads, because it is the part a model most often narrates
    # wrongly: `CLAUDE.md`'s grade-the-world rule applied to a subprocess.
    parts = [f"exit status {finished.returncode}"]
    if out.strip():
        parts.append(f"stdout:\n{out.rstrip()}")
    if err.strip():
        parts.append(f"stderr:\n{err.rstrip()}")
    if not out.strip() and not err.strip():
        parts.append("(no output)")
    return _cap("\n".join(parts))


# --- fetch, and the address check that makes it safe to have ----------------


def refusal_for_address(address: str) -> Optional[str]:
    """Why this **resolved address** may not be fetched, or ``None``.

    By address and never by hostname string: the string is attacker-controlled
    and DNS is the rebinding seam, so ``totally-fine.example`` resolving to
    ``127.0.0.1`` has to be refused by what it *is*, not by what it is called.

    The concrete local reason, which is why this is not a theoretical control:
    omega's own channel listens on ``127.0.0.1:7717`` and cloud metadata
    answers at ``169.254.169.254``. A fetch tool without this is not "read a
    web page" — it is a reach into every unauthenticated service on the
    machine, arriving through the one tool whose inputs come from outside.

    **Fails closed.** After the named ranges there is a catch-all on
    ``is_global``, so a range nobody thought to name — carrier-grade NAT at
    ``100.64.0.0/10``, say — is refused rather than allowed by omission.
    """
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return f"{address!r} is not an address omega can check"

    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None:
        why = refusal_for_address(str(mapped))
        return None if why is None else f"it maps to IPv4 {mapped}, and {why}"

    if ip.is_unspecified:
        return "it is the unspecified address, which means 'this host'"
    if ip.is_loopback:
        return (
            "it is a loopback address, where omega's own channel listens "
            "(127.0.0.1:7717)"
        )
    if ip.is_link_local:
        return (
            "it is link-local, where cloud instance metadata answers "
            "(169.254.169.254)"
        )
    if ip.is_private:
        return "it is a private address on this network, not the public web"
    if ip.is_multicast:
        return "it is a multicast address, which is not a web server"
    if ip.is_reserved:
        return "it is in a reserved range"
    if not ip.is_global:
        # The catch-all, and it is the important one: an unnamed special range
        # must be refused by default, not allowed by omission.
        return "it is not a globally routable address"
    return None


def _resolve_host(host: str, port: int) -> list[str]:
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ToolError(f"could not resolve {host!r}: {exc}") from exc
    return [info[4][0] for info in infos]


def _refusal_for_addresses(host: str, addresses: Sequence[str]) -> Optional[str]:
    if not addresses:
        # Fail closed: "resolved to nothing" is not "resolved to something
        # safe", and a check that passes on empty is not a check.
        return f"{host!r} resolved to no address at all"
    for address in addresses:
        why = refusal_for_address(address)
        if why is not None:
            return f"{host} resolves to {address} and {why}"
    return None


def refusal_for_host(
    host: str,
    port: int = 443,
    *,
    resolve: Callable[[str, int], list[str]] = _resolve_host,
) -> Optional[str]:
    """Why this host may not be fetched, or ``None``. Checks **every** address
    it resolves to, not the first.

    A name with both a public and a loopback record would otherwise pass on the
    ordering of a DNS answer, which is the attacker's to choose. ``resolve`` is
    injectable so the rule is testable without a network or a real name.

    This is the *predicate*, and on its own it is still a check-then-use: it
    answers about one resolution, and whoever connects afterwards resolves
    again. :func:`vetted_address` is the half that closes that, and `fetch`
    goes through that one — see its docstring for why the gap was real.
    """
    return _refusal_for_addresses(host, resolve(host, port))


def vetted_address(
    host: str,
    port: int = 443,
    *,
    resolve: Callable[[str, int], list[str]] = _resolve_host,
) -> str:
    """The address `fetch` will connect to, refusing if any resolution is bad.

    Returns an address rather than a verdict, and that is the entire point.
    :func:`refusal_for_host` could only ever say *this name looked fine a
    moment ago*; the connection that followed did its own DNS lookup, so a name
    whose answer changed in between — one public record to pass the check, a
    loopback or `169.254.169.254` record to serve the connection — was refused
    by nothing. The check and the use were about two different addresses.

    Handing back the vetted address makes them one address: the caller connects
    to *this*, not to whatever the name says next, and the hostname is carried
    separately for `Host` and for TLS so certificate validation is unweakened.
    """
    addresses = resolve(host, port)
    why = _refusal_for_addresses(host, addresses)
    if why is not None:
        raise ToolError(f"refusing to fetch {host}: {why}")
    return addresses[0]


def check_url(
    url: str,
    *,
    resolve: Callable[[str, int], list[str]] = _resolve_host,
) -> str:
    """Refuse a URL omega must not fetch, and return its host. Raises
    :class:`ToolError`.

    ``http`` and ``https`` only. Every other scheme a URL library will happily
    open is a different capability wearing ``fetch``'s name — ``file:`` is
    ``read_file`` with no path classification, ``ftp:`` and ``gopher:`` are
    unauthenticated reaches the address check was never written against.
    """
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise ToolError(
            f"refusing to fetch {url!r}: omega fetches http and https only, "
            f"not {parts.scheme or 'a URL with no scheme'!r}"
        )
    host = parts.hostname
    if not host:
        raise ToolError(f"refusing to fetch {url!r}: it names no host")
    try:
        port = parts.port or (443 if parts.scheme == "https" else 80)
    except ValueError as exc:
        raise ToolError(f"refusing to fetch {url!r}: {exc}") from exc

    why = refusal_for_host(host, port, resolve=resolve)
    if why is not None:
        raise ToolError(f"refusing to fetch {url}: {why}")
    return host


TEXTUAL_TYPES = frozenset(
    {
        "application/json",
        "application/xml",
        "application/javascript",
        "application/ecmascript",
        "application/x-ndjson",
        "application/yaml",
        "application/x-yaml",
        "application/graphql",
    }
)


def refusal_for_content_type(content_type: str) -> Optional[str]:
    """Why this body may not be read as a page, or ``None``.

    `fetch` returns text, and until this existed it returned text *whatever
    came back* — a PDF, a font, a tarball, a video, all run through
    ``decode(errors="replace")`` and handed to the model as several thousand
    replacement characters. That is not a security hole so much as a quiet
    waste with a sharp edge: the bytes are spent, the context is spent, and
    what the model reads is indistinguishable from a page that happened to be
    gibberish, so it cannot even report the problem accurately.

    The rule is an allowlist because the failure is open-ended: there is no
    enumerating the binary types, but the textual ones are a short list plus
    two structured-suffix conventions. A missing header is ``text/plain`` by
    the HTTP default, which the header parser already applies.
    """
    kind = (content_type or "").strip().lower()
    if kind.startswith("text/") or kind in TEXTUAL_TYPES:
        return None
    subtype = kind.partition("/")[2]
    if subtype.endswith("+json") or subtype.endswith("+xml"):
        return None
    return (
        f"it answered with {content_type or 'no type at all'}, which is not "
        "text — omega reads pages, not binaries"
    )


class _PinnedHTTPConnection(http.client.HTTPConnection):
    """An HTTP connection that dials a given address instead of resolving.

    ``host`` stays the hostname, so the ``Host`` header is the one the server
    expects; only the socket's destination is replaced.
    """

    def __init__(self, host: str, *, address: str, **kw: Any) -> None:
        super().__init__(host, **kw)
        self._address = address

    def connect(self) -> None:
        self.sock = self._create_connection(
            (self._address, self.port), self.timeout, self.source_address
        )
        try:
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            # Not every platform implements it; stock http.client tolerates
            # this too, and it is a performance option, not a correctness one.
            pass
        if self._tunnel_host:
            self._tunnel()


class _PinnedHTTPSConnection(_PinnedHTTPConnection, http.client.HTTPSConnection):
    """The same, with TLS still validated against the *name*.

    The certificate check is what would quietly rot if the address were carried
    in ``host``: the handshake would be validated against `93.184.216.34` and
    fail, and the tempting fix for that failure is to stop verifying. Passing
    ``server_hostname`` separately means pinning costs nothing in TLS strength.
    """

    def connect(self) -> None:
        _PinnedHTTPConnection.connect(self)
        server_hostname = self._tunnel_host or self.host
        self.sock = self._context.wrap_socket(self.sock, server_hostname=server_hostname)


class _PinningHandler:
    """Resolve, refuse, and pin — once per hop, at the moment of connecting."""

    def __init__(self, resolve: Callable[[str, int], list[str]]) -> None:
        self._resolve = resolve

    def _address_for(self, req: urllib.request.Request) -> str:
        parts = urlsplit(req.full_url)
        host = parts.hostname
        if not host:
            raise ToolError(f"refusing to fetch {req.full_url!r}: it names no host")
        port = parts.port or (443 if parts.scheme == "https" else 80)
        return vetted_address(host, port, resolve=self._resolve)


class _PinnedHTTPHandler(_PinningHandler, urllib.request.HTTPHandler):
    def __init__(self, resolve: Callable[[str, int], list[str]]) -> None:
        _PinningHandler.__init__(self, resolve)
        urllib.request.HTTPHandler.__init__(self)

    def http_open(self, req):  # type: ignore[override]
        address = self._address_for(req)
        return self.do_open(
            lambda host, **kw: _PinnedHTTPConnection(host, address=address, **kw), req
        )


class _PinnedHTTPSHandler(_PinningHandler, urllib.request.HTTPSHandler):
    def __init__(self, resolve: Callable[[str, int], list[str]]) -> None:
        _PinningHandler.__init__(self, resolve)
        urllib.request.HTTPSHandler.__init__(self)

    def https_open(self, req):  # type: ignore[override]
        address = self._address_for(req)
        # context and check_hostname forwarded exactly as the stock handler
        # forwards them, so pinning changes the destination and nothing else
        # about how the connection is secured.
        return self.do_open(
            lambda host, **kw: _PinnedHTTPSConnection(host, address=address, **kw),
            req,
            context=self._context,
            check_hostname=self._check_hostname,
        )


class _CheckedRedirects(urllib.request.HTTPRedirectHandler):
    """Re-run the address check on **every** hop.

    Checking only the first URL is the same mistake as checking the hostname
    string: a public page that answers ``302 -> http://169.254.169.254/`` walks
    straight past a first-hop-only check, and it is the ordinary shape of this
    attack rather than an exotic one.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        check_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch(
    url: str,
    *,
    timeout: float = FETCH_TIMEOUT,
    max_bytes: int = MAX_FETCH_BYTES,
    resolve: Callable[[str, int], list[str]] = _resolve_host,
) -> str:
    """Fetch a page. **External always** — the sub-loop stops and asks first.

    What comes back is untrusted input (DL-014) and the most untrusted thing
    omega reads: it is the first content someone else wrote. It is returned as
    data, and nothing in it can change what the next pass may run, because that
    is decided by :func:`classify` from the tool name and its arguments.

    Three things guard the reach itself, and each closes a way the address
    check could have been true and useless: every hop is **pinned** to an
    address that was vetted rather than re-resolved (:func:`vetted_address`),
    **no proxy** is consulted, and the body must be **text**
    (:func:`refusal_for_content_type`). The redirect handler still re-checks
    each hop's scheme, and each hop opens through the pinning handlers, so a
    `302` is judged exactly as the first request was.
    """
    check_url(url, resolve=resolve)
    opener = urllib.request.build_opener(
        # No proxy, deliberately. An `http_proxy` in the environment would send
        # every fetch to a host the address check never saw and the pin never
        # covered — the guard would still pass and mean nothing.
        urllib.request.ProxyHandler({}),
        _PinnedHTTPHandler(resolve),
        _PinnedHTTPSHandler(resolve),
        _CheckedRedirects(),
    )
    request = urllib.request.Request(  # noqa: S310 - scheme checked above
        url, headers={"User-Agent": _USER_AGENT}, method="GET"
    )
    try:
        with opener.open(request, timeout=timeout) as response:
            content_type = response.headers.get_content_type()
            why = refusal_for_content_type(content_type)
            if why is not None:
                # Before the body is read, not after: the point is not to spend
                # the bytes, and `max_bytes` of a video is still a download.
                raise ToolError(f"refusing to read {url}: {why}")
            charset = response.headers.get_content_charset() or "utf-8"
            body = response.read(max_bytes + 1)
            status = getattr(response, "status", None)
            final = response.geturl()
    except ToolError:
        # A refused redirect hop, a refused address, or a refused type.
        # Propagated as itself so the reason names what was refused rather than
        # flattening into "could not fetch".
        raise
    except urllib.error.HTTPError as exc:
        raise ToolError(f"{url} answered {exc.code} {exc.reason}") from exc
    except Exception as exc:  # noqa: BLE001 - one failure class, per the seam
        raise ToolError(f"could not fetch {url}: {exc}") from exc

    truncated = len(body) > max_bytes
    try:
        text = body[:max_bytes].decode(charset, errors="replace")
    except LookupError:
        # A charset nobody has heard of is the server's problem, not a reason
        # to fail the fetch; utf-8 with replacement is what we did before.
        text = body[:max_bytes].decode("utf-8", errors="replace")
    head = f"{status} {final}" if final != url else f"{status} {url}"
    if truncated:
        text += f"\n... [truncated at {max_bytes} bytes]"
    return _cap(f"{head}\n\n{text}")


def recall(claims: Sequence[Any], about: str = "") -> str:
    """What omega has written down, optionally narrowed to a phrase (DL-060).

    ``claims`` is passed in and never fetched. The running process holds an
    exclusive lock on the log, so a tool that opened the store to answer this
    would deadlock against the agent that called it — and beyond the lock, the
    turn was *already handed* the whole set (``run_turn``'s ``known``), so
    reading it again would be a second retrieval path that could disagree with
    the one the prompt rendered from.

    **Matching is looser here than a trigger, deliberately.** ``fires_on`` is
    substring over the phrases a claim names about itself, and it is kept dumb
    because it decides whether omega *acts* on something unasked. This decides
    what omega *shows* when asked, where the cost of a near-miss is a line the
    person skims and the cost of a strict miss is the answer that started
    DL-060. So a search matches against the claim's own text as well as its
    trigger words, and an empty search returns everything.

    The answer is deliberately flat text, not JSON: the model reads it, the
    person never sees it, and a structured shape would invite parsing it back
    into something that decides. This only ever describes.
    """
    from omega.learn import when_phrase  # local: Ring 1 must not import policy

    needle = about.strip().lower()
    matched = []
    for claim in claims:
        if not needle:
            matched.append(claim)
            continue
        trigger = getattr(claim, "trigger", None) or {}
        words = [str(p) for p in (trigger.get("any") or [])]
        haystack = " ".join([str(getattr(claim, "text", "")), *words]).lower()
        if needle in haystack:
            matched.append(claim)

    if not matched:
        if not claims:
            # Distinct from "nothing matched", and it has to be: an empty
            # memory and a failed search read identically to a model that is
            # about to tell the person one or the other.
            return "You have not written anything down about this person yet."
        return (
            f"Nothing you have written down mentions {about.strip()!r}. "
            f"You hold {len(claims)} claims in total."
        )

    shown = matched[:MAX_RECALLED]
    lines = [f"- {getattr(c, 'text', '')} ({when_phrase(getattr(c, 'trigger', None))})" for c in shown]
    header = (
        f"You have written down {len(claims)} things about this person."
        if not needle
        else f"{len(matched)} of {len(claims)} mention {about.strip()!r}."
    )
    if len(matched) > len(shown):
        lines.append(f"... and {len(matched) - len(shown)} more; search for something narrower.")
    return _cap("\n".join([header, *lines]))


def _cap(text: str) -> str:
    if len(text) <= MAX_RESULT_CHARS:
        return text
    return text[:MAX_RESULT_CHARS] + f"\n... [truncated at {MAX_RESULT_CHARS} characters]"


# --- the box ----------------------------------------------------------------


@dataclass(frozen=True)
class _Tool:
    classify: Callable[["ToolBox", dict[str, Any]], Decision]
    dispatch: Callable[["ToolBox", dict[str, Any]], str]
    schema: dict[str, Any]


def _schema(name: str, description: str, properties: dict, required: list) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


_TOOLS: dict[str, _Tool] = {
    READ_FILE: _Tool(
        classify=_classify_read_file,
        dispatch=lambda box, args: read_file(args["path"]),
        schema=_schema(
            READ_FILE,
            "Read a text file from this machine.",
            {"path": {"type": "string", "description": "Absolute path to read."}},
            ["path"],
        ),
    ),
    WRITE_FILE: _Tool(
        classify=_classify_write_file,
        dispatch=lambda box, args: write_file(args["path"], args["content"]),
        schema=_schema(
            WRITE_FILE,
            "Write a text file. Writing outside omega's own store stops to ask.",
            {
                "path": {"type": "string", "description": "Absolute path to write."},
                "content": {"type": "string", "description": "The whole new contents."},
            },
            ["path", "content"],
        ),
    ),
    RUN_CODE: _Tool(
        classify=_classify_run_code,
        dispatch=lambda box, args: run_code(args["argv"], args["cwd"]),
        schema=_schema(
            RUN_CODE,
            "Run a program. Give argv as a list; there is no shell, so "
            "pipes, redirection and $(...) do not work.",
            {
                "argv": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "The program and its arguments, e.g. ['ls', '-l'].",
                },
                "cwd": {"type": "string", "description": "Directory to run in."},
            },
            ["argv", "cwd"],
        ),
    ),
    FETCH: _Tool(
        classify=_classify_fetch,
        dispatch=lambda box, args: fetch(args["url"]),
        schema=_schema(
            FETCH,
            "Fetch a web page over http or https. Always stops to ask first.",
            {"url": {"type": "string", "description": "The http(s) URL to fetch."}},
            ["url"],
        ),
    ),
    RECALL: _Tool(
        classify=_classify_recall,
        dispatch=lambda box, args: recall(box.remembered, args.get("about") or ""),
        schema=_schema(
            RECALL,
            "Look at what you have written down about this person. Use it "
            "when you are asked what you know or remember about them, or "
            "before answering something your notes might already cover.",
            {
                "about": {
                    "type": "string",
                    "description": "Narrow to claims mentioning this. Omit for all of them.",
                }
            },
            [],
        ),
    ),
}

assert set(_TOOLS) == TOOL_NAMES, "the tool set and its registry disagree"


def schemas() -> list[dict[str, Any]]:
    """What is offered to the model, in the order the tools are declared."""
    return [_TOOLS[name].schema for name in (READ_FILE, WRITE_FILE, RUN_CODE, FETCH, RECALL)]


@dataclass(frozen=True)
class ToolBox:
    """Ring 1 bound to one store.

    The only state is ``store_root``, and it is here because the *tier* of a
    write depends on it: inside omega's own store a write is local and runs,
    outside it is external and asks. A box with no store could not tell those
    apart and would have to pick one, which means picking wrong in one
    direction every time.
    """

    store_root: Path

    #: Every claim omega holds, handed over by the turn (DL-060). A tuple and
    #: not a store handle: the process that builds this box is the one holding
    #: the log's exclusive lock, so a box that opened the store to answer
    #: `recall` would deadlock against its own caller.
    #:
    #: Defaulted to empty so every existing construction keeps working, and an
    #: empty box answers "you have not written anything down" — which is true
    #: of a box nobody gave any claims to, and is the only honest thing a tool
    #: that was handed nothing can say.
    remembered: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "store_root", Path(self.store_root).resolve())
        object.__setattr__(self, "remembered", tuple(self.remembered))

    def classify(self, name: str, arguments: dict[str, Any]) -> Decision:
        """The gate. Static, code-only, and it runs **before** dispatch.

        Never consults a model and never reads a tool result: DL-014 settles
        that an instruction injected through a result can talk a model out of
        asking, so what may run is decided from the tool name and its arguments
        and from nothing else.
        """
        tool = _TOOLS.get(name)
        if tool is None:
            raise ToolRejected(
                f"there is no tool called {name!r}; omega has "
                f"{sorted(TOOL_NAMES)} and that set does not grow"
            )
        if not isinstance(arguments, dict):
            raise ToolRejected(
                f"{name} was called with {type(arguments).__name__} arguments, "
                f"not an object"
            )
        return tool.classify(self, arguments)

    def dispatch(self, decision: Decision) -> str:
        """Run what was classified. Refuses anything the classifier did not clear.

        The check is not belt and braces — it is the invariant. A dispatch path
        that could be reached with an ``external`` decision would make the gate
        advisory, and an advisory gate is one refactor away from no gate.
        """
        if not decision.dispatches:
            raise ToolRejected(
                f"{decision.tool} was classified {decision.tier}; it must be "
                f"asked about, not dispatched"
            )
        return _TOOLS[decision.tool].dispatch(self, decision.args)
