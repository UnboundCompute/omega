"""``python -m omega`` — the way a person talks to omega.

A terminal REPL over :class:`omega.runtime.Runtime`. It is a *client* of the
resident process rather than a second way into it: every line typed here becomes
an episode and the answer is read back out of the projection, so what the
terminal shows and what the tray would show are the same filter over the same
log (§Q10, §1.6).

**It renders the three outcomes apart, and that is not cosmetic.** DL-011 makes
staying silent a first-class *success*; a UI that printed nothing for a silent
turn and nothing for a failure would erase the distinction the whole milestone
is built to keep, and the person reading the terminal is the metric.

**No key is the most likely first run.** The one thing that will greet most
people the first time is a missing ``OPENAI_API_KEY``, so that path gets a short
sentence naming the file to put it in — never a traceback. An unconfigured
assistant is a configuration problem, and a stack trace tells a person nothing
about where their key goes.

**Where `.env` is looked for, and why not just the working directory.**
``provider.load_env`` defaults to ``cwd()/.env``, which would make omega work
from the repo root and nowhere else — a rule nobody can see and everybody trips
over. So the file is resolved explicitly: ``--env`` if given, else beside the
store (the store is the thing that *is* this omega), else the repo root's, for
the source checkout where the key most likely already lives.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
from pathlib import Path
from typing import Callable, Optional

from omega import derive, learn, memory, provider, runtime, schedule as scheduling
from omega.channel import DEFAULT_HOST, DEFAULT_PORT
from omega.queue import EventQueue

__all__ = [
    "DEFAULT_STORE",
    "SILENCE",
    "resolve_env",
    "render",
    "review",
    "repl",
    "serve",
    "main",
]

#: Where omega keeps its log when nothing says otherwise. Nothing in M0 or M1
#: names a default — the store path was always an argument — so this is chosen
#: here and documented in the README rather than left to whatever directory the
#: command happened to be run from. A dotdir in ``$HOME`` because the store is
#: the user's, not the checkout's: two clones must not be two omegas.
DEFAULT_STORE = "~/.omega"

#: Rendered for a silent turn. A sentence, not an empty line: the person has to
#: be able to tell "omega decided this deserved nothing" from "omega broke"
#: (DL-011), and from the terminal those look identical unless one of them says
#: so out loud.
SILENCE = "(omega chose not to speak)"

_PROMPT = "you> "

#: The tree this file was installed from. In a source checkout it is the repo
#: root — the directory holding ``.env.example`` — which is where a key put down
#: while following the README already is.
_REPO_ROOT = Path(__file__).resolve().parents[2]


# --- configuration ----------------------------------------------------------


def resolve_env(explicit: Optional[str], store_dir: Path) -> tuple[Optional[Path], Path]:
    """Find the ``.env`` to read, and say where a key should go if there is none.

    Returns ``(found, where)``. ``found`` is the file that exists and will be
    read, or ``None``; ``where`` is the path to name in the no-key message, which
    is a real answer even when nothing was found — "put it somewhere" is not
    help.

    The order is deliberate. An explicit ``--env`` wins and does **not** fall
    back, because silently reading a different file than the one named is worse
    than reading none. Then the store's own directory, because that is the thing
    that identifies this omega. Then the repo root, for the checkout where
    `.env.example` was copied. The working directory is **not** in the list: it
    is the one input that changes for reasons that have nothing to do with omega.
    """
    if explicit is not None:
        named = Path(explicit).expanduser()
        return (named if named.is_file() else None), named

    beside = store_dir / ".env"
    if beside.is_file():
        return beside, beside

    repo = _REPO_ROOT / ".env"
    if repo.is_file():
        return repo, repo

    return None, beside


# --- rendering --------------------------------------------------------------


def render(said: runtime.Said) -> str:
    """One turn as one line of terminal output.

    Four outcomes, four shapes, and none of them is the absence of another.
    ``blocked`` is checked before ``failed`` because a turn that stopped to ask
    is correct behaviour (§2.1), and ``silent`` is its own sentence for the
    reason in :data:`SILENCE`.
    """
    if said.blocked:
        return f"omega needs something first: {said.needs}"
    if said.failed:
        return f"omega hit an error, and logged it: {said.error}"
    if said.silent:
        return SILENCE
    return f"omega: {said.reply}"


def review(store_dir: Path, *, write: Callable[[str], None]) -> int:
    """Print what omega has been taught, without starting it (DL-048).

    Offline by construction: it opens the log, folds the two derived views and
    renders them. No provider, no key, no network, no turn — so a person can
    audit omega's memory on a machine that could not run it, and reading the
    record can never itself change the record.

    **It cannot run while omega is running, and that is the store's design
    rather than a gap here.** The log holds an exclusive advisory lock for the
    lifetime of the open handle (``src/log.rs``), which is what makes DL-016's
    single-writer rule true instead of aspirational. So this is the audit you
    sit down to do, not the question you ask in passing — the in-conversation
    form has to go through the resident process and is a separate decision
    (DL-049). The lock message says which state the person is in, because
    "omega is already running" and "something is wrong with your store" want
    opposite next actions and the raw error distinguishes neither.
    """
    try:
        store = memory.MemoryStore.open(store_dir)
    except memory.AlreadyLocked:
        write(f"omega is already running on {store_dir}.")
        write("")
        write("    Only one process may hold the log, so this cannot read it")
        write("    while that one is up. Stop omega and run this again.")
        return 2
    except Exception as exc:  # noqa: BLE001 - a person's problem, not a traceback
        write(f"omega could not open {store_dir}: {type(exc).__name__}: {exc}")
        return 2

    with store:
        queue = EventQueue(store)
        learned = derive.Learned.rebuild(store)
        standing = scheduling.Scheduler(queue)
        standing.refresh()
        # Read after both folds, never before: a head taken first could only be
        # stale in the direction that claims the views are behind when they are
        # not. Nothing else writes to this log — the lock above is what
        # guarantees that — so equality here really does mean *whole log seen*.
        write(
            learn.review(
                learned.claims(),
                running=standing.schedules,
                broken=standing.broken,
                current=learned.through >= queue.head(),
            )
        )
    return 0


def _startup_lines(rt: runtime.Runtime, *, interactive: bool = True) -> list[str]:
    """What omega says when it opens.

    The startup report is printed **first and always**, because it is where an
    interrupted turn and a rebuilt cursor are surfaced (§1.2) — omega *says so*,
    and a report nobody prints is a report nobody gets.
    """
    out = [f"omega — store {rt.store_path}, log head {rt.queue.head()}"]
    if rt.address is not None:
        host, port = rt.address
        out.append(f"listening on {host}:{port}")
    else:
        out.append("socket listener disabled")
    out.extend(f"omega: {line}" for line in rt.report.lines())
    if rt.report.clean:
        out.append("(nothing was left in flight last time)")
    if interactive:
        out.append("type a line to talk; ctrl-d or ctrl-c to leave")
    return out


# --- the loop ---------------------------------------------------------------


def repl(
    rt: runtime.Runtime,
    *,
    read_line: Callable[[], str],
    write: Callable[[str], None],
    timeout: Optional[float] = None,
) -> int:
    """Read a line, append it, print how the turn ended. Until EOF.

    ``read_line`` and ``write`` are parameters rather than ``input``/``print`` so
    the loop can be driven by a test without a terminal — the rendering of the
    three outcomes is the part most worth asserting, and a loop that could only
    be exercised by hand would be exactly the part nobody checks.

    EOF returns 0: leaving is not a failure. A dead drain returns 1, because the
    loop is no longer running and pretending otherwise would let a person keep
    typing into something that has stopped listening.
    """
    while True:
        try:
            line = read_line()
        except EOFError:
            write("")
            return 0
        text = line.strip()
        if not text:
            continue
        try:
            write(render(rt.say(text, timeout=timeout)))
        except runtime.TurnTimeout as exc:
            # Not a failure and not an answer: the turn is still running and
            # will still be logged, so say that rather than inventing either.
            write(f"omega: (still working — {exc})")
        except runtime.DrainFailed as exc:
            write(f"omega stopped draining: {exc}")
            return 1


def serve(
    rt: runtime.Runtime,
    *,
    write: Callable[[str], None],
    wait: Optional[Callable[[], None]] = None,
) -> int:
    """Keep the resident runtime alive without reading a terminal.

    The tray owns this mode. SIGTERM asks for the same clean, between-turn stop
    as Ctrl-C in the REPL; the outer ``main`` finally block performs that stop.
    ``wait`` is injectable so the lifecycle can be proven without parking a
    test process forever.
    """
    stopping = threading.Event()

    def request_stop(_signum: int, _frame: object) -> None:
        stopping.set()

    previous_term = signal.signal(signal.SIGTERM, request_stop)
    previous_int = signal.signal(signal.SIGINT, request_stop)
    try:
        write("omega agent is running")
        (wait or stopping.wait)()
        return 0
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGINT, previous_int)


# --- entry point ------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omega",
        description="Talk to omega. Every line becomes an episode in the log.",
    )
    parser.add_argument(
        "--store",
        default=DEFAULT_STORE,
        metavar="PATH",
        help=f"the store directory holding the episode log (default: {DEFAULT_STORE})",
    )
    parser.add_argument(
        "--env",
        default=None,
        metavar="PATH",
        help=(
            "the .env holding OPENAI_API_KEY; default: beside the store, then "
            "the repo root. Never the working directory."
        ),
    )
    parser.add_argument(
        "--learned",
        action="store_true",
        help=(
            "print what omega has been taught and exit; reads the log, starts "
            "nothing, and needs no API key"
        ),
    )
    parser.add_argument(
        "--no-listen",
        dest="listen",
        action="store_false",
        help="do not open the localhost socket; this terminal is the only client",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="run the resident listener without opening an interactive terminal",
    )
    parser.add_argument(
        "--host", default=DEFAULT_HOST, metavar="HOST", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--port",
        default=DEFAULT_PORT,
        type=int,
        metavar="PORT",
        help=f"the listener port (default: {DEFAULT_PORT})",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """Open one omega, talk to it, close it. Ctrl-C and EOF both leave cleanly."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.serve and not args.listen:
        parser.error("--serve requires the localhost listener")
    if args.learned and args.serve:
        parser.error("--learned reads the log and exits; it cannot also serve")
    write = _writer()

    store_dir = Path(args.store).expanduser()
    if args.learned:
        # Before the provider is touched on purpose. Reading what omega already
        # knows must not depend on omega being able to think — a key that has
        # expired is exactly when a person wants to check the record, and
        # failing here for a missing key would be answering a question about
        # the log with a question about the network.
        return review(store_dir, write=write)
    env_path, env_hint = resolve_env(args.env, store_dir)
    if args.env is not None and env_path is None:
        # A named file that is not there is worth a word. Silently ignoring an
        # explicit flag is how a person spends ten minutes on the wrong file.
        write(f"note: {env_hint} does not exist; reading configuration from the "
              f"environment only")

    try:
        # Always an explicit path, even one that does not exist: passing None
        # would let provider.load_env fall back to the working directory, which
        # is the cwd-dependence this file exists to remove.
        provider.set_provider(provider.provider_from_env(env_path=env_path or env_hint))
    except provider.ProviderNotConfigured as exc:
        if os.environ.get("OPENAI_API_KEY", "").strip():
            # A key *is* there, so this is the other unconfigured thing — the
            # client package. Pointing at .env here would send a person to fix a
            # file that is already right.
            write(f"omega cannot reach a model: {exc}")
            return 2
        # The most likely first run. A sentence and a path, never a traceback.
        write("omega has no API key, so it cannot think yet.")
        write("")
        write(f"    put one in {env_hint} :")
        write("")
        write("        OPENAI_API_KEY=sk-...")
        write("")
        write("    or export OPENAI_API_KEY in this shell.")
        return 2

    rt = runtime.Runtime(
        store_dir, listen=args.listen, host=args.host, port=args.port
    )
    try:
        rt.start()
    except Exception as exc:  # noqa: BLE001 - startup failures are a person's problem
        write(f"omega could not open {store_dir}: {type(exc).__name__}: {exc}")
        return 2

    code = 0
    try:
        for line in _startup_lines(rt, interactive=not args.serve):
            write(line)
        if args.serve:
            code = serve(rt, write=write)
        else:
            code = repl(rt, read_line=lambda: input(_PROMPT), write=write)
    except KeyboardInterrupt:
        # Ctrl-C lands on this thread; the drain is untouched and finishes the
        # turn it is holding while stop() waits for it. Not a crash, so not a
        # crash exit — and the log must be left the way a clean stop leaves it.
        write("")
    finally:
        try:
            rt.stop()
        except runtime.DrainFailed as exc:
            write(f"omega stopped draining: {exc}")
            code = code or 1
        except runtime.DrainStuck as exc:
            write(f"omega is still finishing a turn: {exc}")
            code = code or 1
    return code


def _writer() -> Callable[[str], None]:
    def write(line: str) -> None:
        print(line, flush=True)

    return write


if __name__ == "__main__":  # pragma: no cover - exercised by `python -m omega`
    sys.exit(main())
