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
| `--learned` | print what omega has been taught, then exit. Reads the log, starts nothing, needs no key |

One process holds the log, because the log is opened exclusively by design — so the
terminal, the socket and the loop are threads inside that one process rather than several
of them.

That exclusivity is why `--learned` is the audit you sit down to do rather than a question
you ask in passing: it cannot read the log while omega is up, and says so plainly instead
of failing with a lock error. It prints what omega currently believes, what it will wake up
and do, and — the part visible nowhere else — any schedule that was written down but whose
expression will not parse, so it has been silently never firing.

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

### Other MCP servers as tools (opt-in)

omega can call tools on remote MCP servers (Streamable HTTP, `https://` only). Nothing is
on by default and no server is built in: with `OMEGA_MCP_SERVERS` unset, omega offers
exactly the tools it always did. Each server you name is configured from the environment:

```sh
OMEGA_MCP_SERVERS=github,exa           # short lowercase names, comma separated
OMEGA_MCP_GITHUB_URL=https://...       # required, https only
OMEGA_MCP_GITHUB_TOOLS=a,b             # the allowlist: only these are offered
OMEGA_MCP_GITHUB_TOKEN=...             # optional; sent as "Authorization: Bearer ..."
OMEGA_MCP_GITHUB_HEADER=...            # optional; send the token raw in this header instead
OMEGA_MCP_GITHUB_WRITE_TOOLS=c         # tools that change things (see below)
```

The model sees them as `<server>__<tool>`. Tools are read-only: a tool listed under
`_WRITE_TOOLS`, or one the server marks as not read-only, is refused until the approval
gate exists. **Money never goes through omega.** A tool whose name pays, checks out,
orders, books, buys, sells or trades is refused even when you allowlist it, and no setting
lifts that. What a server returns is capped and handed to the model as outside data,
never as instructions.

Some servers it has been shaped around (check each one's own docs for its tool names):

- **GitHub remote MCP**: `https://api.githubcopilot.com/mcp/` with a fine-grained PAT as
  `_TOKEN`. Allowlist read tools only (listing and reading issues, PRs and files).
- **Firecrawl**: `https://mcp.firecrawl.dev/v2/mcp`. Keyless, and rate-limited.
- **Exa**: `https://mcp.exa.ai/mcp`. Keyless.
- **Google Maps Grounding Lite**: `https://mapstools.googleapis.com/mcp`, with the API key
  as `_TOKEN` and `_HEADER=X-Goog-Api-Key`. Set a daily quota cap on the key in the Cloud
  console, so a looping model cannot run up a bill.
- **Expedia**: `https://www.expedia.com/mcp`. No login. Search only: booking is refused
  by the money rule, so omega hands you the link.

### Location from your phone (opt-in)

Tell omega where you are ("I'm at the office", "I'm home") and it notes it; name places
with "save this as home". On your own messages it then sees one line, *their location*
as of a time and how it knows, so a maps tool can answer "near me" and route questions.
After three hours that line becomes your *last known* location and is never presented as
current. Location never appears in briefs, watches or tells, is never learned as a fact
about you, and omega sends it nowhere except as an argument the model passes a tool for
the question you asked. History is kept in the log; ask "where was I yesterday at 6".

Your phone can keep it current. Set a token and omega listens for `POST /location` on
`127.0.0.1` only; unset, nothing listens:

```sh
OMEGA_LOCATION_TOKEN=<long random string>   # e.g. openssl rand -hex 32
OMEGA_LOCATION_PORT=7718                    # optional; the default
```

On the host, publish that port to your tailnet only (MagicDNS and HTTPS certificates
enabled), never with `tailscale funnel`:

```sh
sudo tailscale serve --bg --https=7718 http://127.0.0.1:7718
```

Then in the Shortcuts app on the phone (with Tailscale connected), make a shortcut:

1. **Get Current Location**.
2. **Get Contents of URL**: `https://<vm>.<tailnet>.ts.net:7718/location`, Method
   **POST**, Header `Authorization` = `Bearer <token>`, Request Body **JSON** with `lat`
   (Number: the location's Latitude) and `lon` (Number: its Longitude). Optional fields:
   `accuracy` (metres), `label`, `city`. A `city` alone is accepted as a rough location.
3. Optionally, an Automation to run it (when you arrive or leave somewhere, or at set
   times).

A wrong token is a 401 and a bad body a 400; nothing is written for either, and requests
are rate-limited. The listener logs nothing, the token included.

## Boundary rule

`apps/mac-tray` is a channel, not the agent. It may render conversation, collect context,
and display durable task state, but it must not own memory, personality, initiative
judgement, or agent orchestration. Those belong to the Python/Rust core.

The same rule cuts the other way inside the core: nothing outside `python/omega/memory/`
touches the store, and the seam speaks episodes — never frames, offsets or CRCs. That one
is enforced by a test, not by discipline.

## License

omega is licensed under the GNU Affero General Public License v3.0 or later. See LICENSE.
