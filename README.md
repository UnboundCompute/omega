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

**M1 — one turn, end to end — is partly built.** The episode codec and the provider seam
are in; the queue, the turn loop and the outward projection are being written now. The
build order is in [`agent/M1_SPEC.md`](agent/M1_SPEC.md).

**There is no command that runs omega yet.** The first version you can actually talk to
arrives when M1 closes. Until then the honest answer to "can I try it" is: you can run the
test suite, and you can read what it proves.

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

One secret, and it is only needed once there is a turn loop to spend it:

```sh
cp .env.example .env    # then put your key in it
```

`.env` is gitignored and `.env.example` documents every variable omega reads. A value
already exported in your shell wins over the file. **The test suite never reads the key and
never touches the network** — if a test ever needs one, that is a bug in the test.

## Boundary rule

`apps/mac-tray` is a channel, not the agent. It may render conversation, collect context,
and display durable task state, but it must not own memory, personality, initiative
judgement, or agent orchestration. Those belong to the Python/Rust core.

The same rule cuts the other way inside the core: nothing outside `python/omega/memory/`
touches the store, and the seam speaks episodes — never frames, offsets or CRCs. That one
is enforced by a test, not by discipline.
