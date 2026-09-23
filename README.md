# omega

omega is a personal second brain: someone to delegate to and think alongside, with
identity, continuity, initiative, opinions, and a growing understanding of how its user
works.

```text
omega/
├── src/            # Rust: the append-only episode log ("what exists")
├── python/omega/   # Python: the memory seam, the queue, the turn loop ("what comes back")
├── tests/          # The M0 suite — every case in agent/M0_SPEC.md
├── apps/mac-tray/  # Native macOS desk surface
├── agent/          # The milestone specs (M0_SPEC.md, M1_SPEC.md)
├── docs/           # Offline references: harness practices, systems we've read
├── AGENT.md        # Private, gitignored design north star
└── CLAUDE.md       # Project working rules
```

Rust and Python are **one artifact**, not two: the Rust crate is a PyO3 extension module
imported in-process as `omega._log`. That is why there is one CI job and not two — a green
Python run against a stale `.so` is worse than no run at all.

## What works today

**M0 — the spine — is built.** An append-only log with a durable frame format, crash
recovery that will not brick itself on a half-landed append, checkpoints that survive
their own corruption, and a memory seam (`python/omega/memory/`) that is the only module
in the tree allowed to know frames and offsets exist. Its contract is
[`agent/M0_SPEC.md`](agent/M0_SPEC.md) and every case in it has a test.

**M1 — one turn, end to end — runs.** The episode codec, the provider seam, the queue, the
turn loop, the outward projection, the socket channel and the resident process are all in.
The build order and what each step owes is in [`agent/M1_SPEC.md`](agent/M1_SPEC.md). The
Swift tray (step 8) is the part still outstanding.

## Talking to it

```sh
.venv/bin/python -m omega
```

That opens the store, says what the last stop left behind, and gives you a prompt. Every
line you type becomes an episode in the log; the loop reads it back off the log and
answers. Ctrl-D or Ctrl-C leaves — between turns, never inside one, so the next start has
nothing to report.

Three answers are possible and they read differently on purpose: a reply, `(omega chose
not to speak)`, and a logged error. **Staying silent is a success**, not a failure to
answer, and a UI that showed them the same way would erase the distinction the whole
milestone is built to measure.

| flag | what it does |
| --- | --- |
| `--store PATH` | the store directory. Default `~/.omega` — a dotdir in `$HOME`, so two checkouts are not two omegas |
| `--env PATH` | the `.env` holding the key. Default: beside the store, then the repo root; **never** the working directory |
| `--no-listen` | do not open the localhost socket; this terminal is the only client |
| `--port PORT` | where the tray connects |

One process holds the log, because the log is opened exclusively by design — so the
terminal, the socket and the loop are threads inside that one process rather than several
of them.

## Building and testing it

The build is pinned to a virtualenv at `.venv` (see `.cargo/config.toml`), so that comes
first — `cargo test` links libpython and `maturin develop` refuses to run outside a venv.

```sh
python3.11 -m venv .venv                # 3.11 specifically: the pin names python3.11
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install "maturin>=1.7,<2.0" pytest

cargo test                          # the Rust half
.venv/bin/maturin develop --release # build the extension into the venv
.venv/bin/pytest                    # the Python half, against what you just built
```

`--release` is not optional in practice: the 10,000-append and 10 MiB cases take long
enough in a debug build to look like a hang.

## Configuring it

One secret:

```sh
cp .env.example .env    # then put your key in it
```

`.env` is gitignored and `.env.example` documents every variable omega reads. A value
already exported in your shell wins over the file. **The test suite never reads the key and
never touches the network** — if a test ever needs one, that is a bug in the test.

`python -m omega` looks for that file beside the store first (`~/.omega/.env`), then at the
repo root — and deliberately not in whatever directory you happened to run it from, because
"it works from the repo and nowhere else" is a rule nobody can see. Without a key it prints
one sentence naming the exact file to put it in, and exits.

## Boundary rule

`apps/mac-tray` is a channel, not the agent. It may render conversation, collect context,
and display durable task state, but it must not own memory, personality, initiative
judgement, or agent orchestration. Those belong to the Python/Rust core.

The same rule cuts the other way inside the core: nothing outside `python/omega/memory/`
touches the store, and the seam speaks episodes — never frames, offsets or CRCs. That one
is enforced by a test, not by discipline.
