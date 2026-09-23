# Reference implementations — what we take, and what we avoid

> **What this is.** A running register of the agent harnesses we've read, what each does
> well, what each does badly, and — the point of the document — **which specific ideas omega
> takes and which it deliberately rejects.** It is the companion to `harness-practices.md`:
> that file is what the *literature* says, this file is what working systems actually do.
>
> **How to read it.** Scan the register first; it is the whole document in one table. The
> per-system sections are the evidence behind each row. Every substantive claim is traceable
> to a file in the source tree as read on the date given — these are living codebases, so
> treat a line reference as "where it was," not a permanent address.
>
> **Verdicts.** `TAKE` — adopt the idea more or less as-is. `ADAPT` — the idea is right, the
> implementation is shaped by something that isn't true for us. `AVOID` — actively don't do
> this, reason given. `N/A` — it isn't there at all, which is sometimes the finding.
>
> **Standing caution.** Both systems are built for a different job than omega. opencode is a
> *coding* agent: bounded sessions over a versioned file tree. Hermes is the closest analogue
> to omega and is the system omega replaces, so its failures are the more instructive half.
> Neither is a target to match feature-for-feature.

---

## The register

| # | Idea | Source | Verdict | Lands in |
|---|---|---|---|---|
| 1 | Single permission service; wildcard last-match rules; default *ask* | opencode | **TAKE** | DL-014 (approval) |
| 2 | "Always" approval cascades to clear identical pending asks | opencode | **TAKE** | DL-014 (approval volume) |
| 3 | Tool = id + description + schema + execute, nothing else | opencode | **TAKE** | DL-014 (ring 1/2) |
| 4 | Shared wrapper gives every tool validation, truncation, tracing for free | opencode | **TAKE** | DL-014 |
| 5 | Project-local tool added by **file drop**, zero core edits | opencode | **TAKE** | DL-014 (the seam) |
| 6 | Small built-in surface (~14), filtered per turn by capability | opencode | **TAKE** | DL-014 (ring 1) |
| 7 | Universal output truncation with spillover file + bounded preview + path | opencode | **TAKE** | DL-014, DL-008 |
| 8 | Errors written as model-readable instructions ("rewrite the input") | opencode | **TAKE** | DL-011 (tool errors) |
| 9 | Doom-loop guard: same tool + same input N times triggers a gate | opencode | **TAKE** | DL-011 (stall) |
| 10 | Skills as progressive disclosure: names+descriptions pinned, body on demand | opencode | **ADAPT** | DL-014 (ring 3) |
| 11 | Cacheable baseline context + appended diffs instead of per-turn re-render | opencode (V2, unshipped) | **TAKE** | DL-008 (resident set) |
| 12 | Instructions follow the file you touch (nearest ancestor, attached lazily) | opencode | **ADAPT** | DL-008 (associative pull) |
| 13 | Sub-agent = child session with derived, scoped permissions | opencode | **ADAPT** | DL-011 (isolate mode) |
| 14 | Compaction replaces old tool results with a "cleared" marker | opencode | **AVOID** | DL-013 (raw must survive) |
| 15 | Snapshot/diff as the only notion of "did it work" | opencode | **AVOID** | DL-011 (verify-on-resume) |
| 16 | Cross-session memory of any kind | opencode | **N/A** | — (the gap) |
| 17 | Verification that an action achieved its goal | opencode | **N/A** | — (the gap) |
| 18 | "Footprint ladder": a new core tool is the **last** resort, after five cheaper seams | Hermes | **TAKE** | DL-014 (the fence) |
| 19 | Approval floor is deterministic code (static classifier), not model judgement | Hermes | **TAKE** | DL-014 (approval) |
| 20 | Bypass flag **frozen at process start** so no running tool can flip it mid-session | Hermes | **TAKE** | DL-014 (injection) |
| 21 | Progressive disclosure: long-tail tools replaced by 3 bridge tools, query-capped | Hermes | **TAKE** | DL-014 (ring 2) |
| 22 | Skill bodies injected as a **user** message, not the system prompt (cache validity) | Hermes | **TAKE** | DL-008 |
| 23 | Agent-created skills + a background curator that pins/archives/consolidates them | Hermes | **ADAPT** | DL-014 (ring 3) |
| 24 | Growth redirected out-of-tree to a reviewed, pinned manifest catalog | Hermes | **ADAPT** | DL-014 (ring 2) |
| 25 | Cron as a real proactive wake path, independent of the user | Hermes | **TAKE** | DL-011 (time-wake) |
| 26 | Tool failure returned to the model as a result; model decides whether to retry | Hermes | **ADAPT** | DL-011 |
| 27 | ~58 tools sent on **every** call, endpoint-shaped (browser = 10 verbs) | Hermes | **AVOID** | DL-014 (ring 1) |
| 28 | Verification opt-in and **defaulting off**; guard is "policy-only, never checks" | Hermes | **AVOID** | DL-011 |
| 29 | Plugins importing internal symbols → 1,148 re-exports of compat scaffolding | Hermes | **AVOID** | DL-014 (the seam) |
| 30 | Splitting god-files mechanically (one concern now spans 5 sibling modules) | Hermes | **AVOID** | *stay small and legible* |
| 31 | Three different "the default is X" values for one setting in one codebase | Hermes | **AVOID** | — (scale smell) |
| — | *Hermes memory & context rows pending — reader in flight* | Hermes | — | — |

---

## 1. opencode

*Read 2026-09-23, shallow clone of `github.com/sst/opencode` at HEAD. Public, MIT-adjacent
OSS, actively developed, widely regarded. ~2,700 TypeScript files.*

**What it is.** A terminal-first coding agent: CLI, TUI and server over a shared core. The
unit of work is a *session* bound to a project directory. Note when reading it that the repo
is **mid-rewrite** — a live V1 under `packages/opencode/src/session/` and an in-progress V2
under `packages/core/src/session/`, with V1 still driving the actual prompt loop. That
duality is itself a finding; see *Bad* below.

### Loop and session

One loop, in `packages/opencode/src/session/prompt.ts` (`runLoop`), guarded so two loops
can't run on one session concurrently. A cycle reloads uncompacted history, checks whether
the last assistant turn finished with no pending tool calls (the exit condition), resolves
the agent, model and this turn's tool set, assembles system context, and makes exactly one
streamed model call. It `continue`s until a terminal finish reason with nothing pending, or
a per-agent step ceiling is hit — at which point a "max steps" prompt is injected rather
than the loop dying silently.

This is the same shape as DL-011's single cycle, and the confirmation is useful: the inner
tool-call repetition is not a second architecture, it's the `while(true)` that every
tool-using agent has.

Tool errors are first-class and model-readable rather than exceptions that end the turn —
an invalid-arguments error carries prose telling the model to rewrite its input, and the
error text becomes the tool result the model sees. Provider errors get exponential backoff
with jitter, honouring `Retry-After`. There's a **doom-loop guard**: the same tool called
with the same input past a threshold triggers a permission gate, which is a cheap and
well-placed version of the stall detection DL-011 wants.

Sub-agents are child sessions created by a `task` tool, with a *derived, narrowed*
permission set — foreground by default, background behind an experimental flag. The scoping
primitive is filesystem paths, which is exactly the part that doesn't transfer.

### Context

Per-turn system context is environment + instructions + MCP instructions + skills,
concatenated with projected message history. Instructions are pinned **verbatim every
turn**, unsummarized — which is the pinning discipline `harness-practices.md` argues for,
arrived at independently.

Budget is managed at two layers, neither of which is "drop context sources":

- **Tool output is capped mechanically and universally** — a line and byte ceiling applied
  in the shared tool wrapper, so an individual tool author cannot forget to bound output.
  Oversized output spills to a file with a retention window; the model gets a bounded
  preview *plus the path*, so it can re-read or grep selectively. This is the single
  cheapest idea in the codebase and it belongs in omega's ring 1.
- **Compaction** triggers on cumulative token usage crossing the model's usable context
  (limit minus a reserved output buffer), checked both after a step and before starting one.
  It is LLM summarization, not sliding-window truncation, and it protects a recent tail and
  a list of protected tools from pruning.

Files are kept out of context by reference rather than inlined: the read tool records which
paths it loaded into message metadata, which both dedupes instruction attachment and lets
content be replayed by path.

The unshipped **V2** design is the most interesting thing here. A registry of stable-keyed
context *producers* renders one immutable baseline per "context epoch"; when any single
source later changes, that becomes a small durable appended message rather than a re-render
of the whole system prompt — explicitly to preserve the provider's cache prefix. V1 just
rebuilds the full array every turn. For omega's resident working set this is the better
shape, and it is much cheaper to adopt at the start than to retrofit.

### Memory

**There isn't any, across sessions.** Storage is SQLite holding session, message and part
transcripts, plus derived artifacts (todo lists, compaction summaries). There is no
learned-facts table, no preference store, no retrieval over past sessions, no embeddings.
Permission "always" grants are in-memory per process and reset on restart.

Its answer to "remember this" is: *write it in `AGENTS.md` by hand.* That file is loaded
explicitly — a global one plus the nearest project one found by walking up from the working
directory, first match wins — and concatenated raw into the system prompt every turn. There
is one genuinely clever wrinkle: when the read tool opens a file in a subdirectory, the
system walks upward from *that file* and lazily attaches any nearer instruction file not
already loaded, once per assistant message. Context follows the file you touch. That is a
real associative-pull mechanism, just keyed on directory structure rather than meaning.

For omega this is the decisive gap. On the hardest part of the design — durable memory
across sessions — the best-regarded open agent offers nothing to build on. DL-007/008/009
are unpaved road, not a wheel being reinvented.

### Tools, skills, plugins, permissions

A tool is an object: id, description, a schema for parameters, and an execute function. A
shared wrapper supplies argument validation with model-legible error text, output
truncation, and tracing — the author writes none of it. Adding a *built-in* tool is a
one-line entry in a single registry array. Adding a *project-local* tool requires **zero**
core-file changes: drop a file exporting the right shape into a `tool/` directory on the
config search path and it's discovered by glob. Plugins are a single exported function
returning a hook object, resolved from config or npm, merged at runtime; MCP servers are
merged into the same flat tool record. None of these require touching core.

Built-ins number roughly fourteen, and the set is *filtered per turn* by capability and
feature flags rather than exposed wholesale.

**Skills are the progressive-disclosure mechanism.** Only skill names and descriptions sit
in context for the selected agent; the body is loaded on demand by a `skill` tool call that
asks permission, reads a `SKILL.md`, and returns its content plus a listing of the skill
directory. Cheap index, expensive detail on demand, gated.

**Permissions** are centralized: one evaluation function does last-match wildcard resolution
over merged rulesets (agent config plus session overrides), defaulting to *ask*. The ask
blocks the calling tool's fibre on a deferred until a UI replies. An "always" reply appends
session-scoped rules **and cascades** to resolve every other pending request that now
matches — one approval clears a batch, without weakening the per-call gate.

### Good — worth taking

- The permission architecture, entire. It is the shape DL-014 reasons to, already built, and
  the cascade is a real confirmation-fatigue mitigation rather than a loosened gate.
- The tool contract and its two extension seams (file drop, plugin function). This is the
  concrete existence proof that "adding a tool touches one file or none" is achievable.
- The free cross-cutting wrapper. Making truncation impossible to forget is better than
  documenting that authors should remember.
- Output spillover to a file with a path the model can follow — bounded context without
  information loss.
- Errors written as instructions to the model, and the doom-loop guard.
- Keeping the live surface small and filtering it per turn.
- The V2 cacheable-baseline-plus-diffs context design, despite being unshipped.

### Bad — or simply doesn't transfer

- **Compaction makes the summary authoritative**: once compacted, old tool results are
  literally replaced with a "cleared" marker. This is precisely the failure DL-013 names —
  derived text replacing raw text, with the raw unreachable afterwards. omega's L2/L3 must
  reach real episodes.
- **No verification primitive.** The filesystem snapshot-and-diff exists for undo and UI,
  not to check that an action achieved its goal. For a coding agent that's defensible — "run
  the tests" is a normal next tool call. For an assistant whose actions are emails sent and
  events created, most effects aren't diffable and many aren't reversible.
- **No cross-session memory** (above). The gap, not a criticism of a coding agent.
- Sub-agent permission scoping is path-shaped. omega's equivalent has to scope by data
  domain, not directory.
- The snapshot/patch/revert system generalizes only to versioned file trees.
- **The dual V1/V2 rewrite is a caution, not a lesson.** A live rewrite running beside the
  shipped system for an extended period is expensive to even *read* — settling omega's
  session and context model before building is worth more than shipping a v1 that a v2
  later has to fight.

---

## 2. Hermes

*Read 2026-09-23 at `github.com/NousResearch/hermes-agent`. Public. ~6,800 Python files,
282MB. This is the system omega replaces, and the stated reason for replacing it is that it
"grew too big and bloated, which made adding and customizing things hard" — so the useful
output here is a **diagnosis**, not just an inventory.*

**Scale.** ~809,000 lines of Python excluding tests. `agent/` alone is 240 files and ~107,000
lines. The repository root holds roughly thirty `hermes_state_*.py` modules side by side —
schema, compression, fts, search, sessions, timeline, rewind, maintenance, wal, guard,
repair. The state layer is not knowable from one file.

### Loop

Three nested loops. An outer driver (CLI repl, gateway event dispatch, or cron) calls one
`run_conversation`; inside it a turn loop bounded by an iteration budget; inside *that* a
per-API-call retry loop. Each turn iteration runs a fixed phase sequence — begin, prepare,
assemble request, preflight gate, announce, retry loop, normalize response, then either a
tool round or a final text response. Normal termination is the model returning no tool
calls; forced termination is budget exhaustion falling through to a finalizer.

That confirms DL-011's shape from a third angle: one cycle, with the inner tool-call
repetition as a given rather than a second architecture.

**Cron is a genuine proactive wake path** — a supervised ticker thread on a schedule
(durations, "every" phrases, 5-field cron, one-shots), building a job prompt and running it
through the same turn machinery, with an inactivity watchdog and catch-up windows. A second
autonomous path auto-claims tasks from a shared board and spawns unattended workers. So the
time-wake in DL-011 is well-trodden, not exotic.

**Tool errors** are caught broadly and returned to the model as an error result; the model,
not the framework, decides whether to retry, and there is no automatic re-execution of a
logically-failed call. Truncated tool-call JSON has its own separate retry path, and
API-transport recovery is heavy machinery (~1,800 lines classifying 401s, rate limits,
format errors, overflow). Note the asymmetry: enormous effort on *transport* failure,
almost none on *semantic* failure.

**Verification barely exists and defaults off.** The turn-end guard is explicitly
policy-only and states it never runs checks itself; the evidence ledger explicitly never
blocks completion; the whole mechanism is opt-in behind an environment flag. The one real
check found anywhere in the tree is re-reading a file's sha256 after writing it and hard-
erroring on mismatch. Everything else — terminal, browser, delegation, home automation —
trusts the handler's own return value.

### Tools

Declaration is one call into a registry at import time, auto-discovered. Adding a core tool
is genuinely two touchpoints (the tool file, plus naming it in a toolset); a fully custom
tool needs no core files at all. That part is good.

The problem is what's exposed. The shared core bundle is **about 58 tools sent on every API
call**, on every platform, and they are **endpoint-shaped**: the browser registers ten
separate one-verb tools, the task board about fourteen. Their own docs concede the
consequence — "every model tool is sent on every API call... the bar for a new core tool is
high." A handful of umbrella tools (delegate, execute code, skill management) are the
counterexample and are visibly the better design.

Two mitigations are worth taking. **Availability gating** evaluates a per-tool check before
the model ever sees it, TTL-cached. And a real **progressive-disclosure bridge** replaces the
long tail (MCP servers, plugin tools) in the model-visible array with three bridge tools —
search, describe, call — capped at a few queries per call. Core tools never defer; only the
tail does.

**Approval** is the strongest part of the codebase. The never-allow floor is deterministic
code: a ~1,500-line static classifier of dangerous shell patterns, then a gate with pluggable
human-decision transports (CLI prompt, gateway round-trip, protocol elicitation), with an
optional second-opinion model only for borderline cases. And the bypass flag is **frozen at
process start**, explicitly so that no running tool or skill can flip it mid-session — a
deliberate prompt-injection defence, and exactly the hole DL-014 warns about.

### Skills and plugins

A skill is a directory with a `SKILL.md` (frontmatter plus markdown), optionally scripts,
references and templates. 62 built-in across 14 categories, 150 more shipped-but-inactive
and installed on demand. Adding one is one or two files, filesystem-scanned, no registry
edit — close to zero ceremony, and the best seam in the system.

Two details worth stealing. Invoking a skill injects its body as a **user message rather than
into the system prompt**, deliberately, to keep the prompt cache valid. And skills are *also*
exposed as ordinary tools, so **the model can read and rewrite its own skills** — with a
background curator that forks a model pass to pin, archive and consolidate the skills the
agent wrote for itself.

That last mechanism is the closest prior art anywhere to DL-014's ring 3. It is built as
files rather than as memory, which makes it the second independent implementation to choose
files — see the assessment below.

**Plugins** are Python implementing a hook contract across five kinds, each with its own
discovery. And there is a **policy freeze**: no new in-tree memory providers since May 2026,
no new third-party plugins since June 2026, with new work redirected to an out-of-tree
catalog of ~196 reviewed, SHA-pinned manifests holding no code in the tree. That freeze is a
documented admission that the in-tree plugin surface became unmanageable.

### The bloat diagnosis

Size is the symptom. The mechanism is that **there was never a narrow, stable extension
surface**, so plugins imported whatever internal symbol happened to be reachable. When the
internals were finally refactored, that refactor could not proceed without months of
load-bearing compatibility scaffolding: a 290KB compat manifest plus 160KB of documentation
enumerating 1,148 lazy re-exports, 592 restored imports, 290 restored deletions and 34 names
that could not be restored at all, with 332 files carrying a compat block, CI-enforced and
sunset-dated.

Two second-order lessons:

- **Mechanical file-splitting redistributes bloat rather than removing it.** Their stated rule
  is that a file over ~2,000 lines should be split into topic siblings. It isn't holding —
  an 8,138-line client, a 7,343-line adapter, a 4,835-line CLI that already has fourteen
  mixin siblings. And where splitting *did* happen, the turn loop became thirty sibling files
  totalling ~12,000 lines in which one concern ("how are errors retried") now spans five of
  them. The boundary needed redrawing, not the file.
- **Conflicting defaults are a reliable smell of scale.** One iteration setting has three
  different documented defaults in one codebase — unlimited in the constructor, 500 in a
  docstring, 250 in the shipped config.

### Good — worth taking

- The **footprint ladder**: their own docs rank adding a core tool last, after extending an
  existing one, a skill, a gated tool, a plugin, or a catalog entry. This is DL-014's "the
  tool surface never grows; skills do," reached independently by people who earned it.
- The **approval floor in deterministic code**, and especially freezing the bypass flag at
  process start.
- The **progressive-disclosure bridge** for the long tail, and pre-model availability gating.
- **Skills as a near-zero-ceremony filesystem convention**, injected as a user message to
  preserve cache validity.
- **Agent-authored skills with a curator** — the real prior art for learned skills.
- **Cron as a first-class wake path.**
- Redirecting growth **out of tree** to a reviewed, pinned catalog once in-tree growth stops
  being manageable. Better still would be not needing the freeze.

### Bad — avoid

- **~58 endpoint-shaped tools on every call.** Four times opencode's surface, well past where
  selection degrades, and the reason a disclosure bridge had to be bolted on later.
- **Verification opt-in and defaulting off**, with a guard that by its own docstring never
  checks anything.
- **No stable extension ABI** — the single root cause of the compat tax, and the thing omega
  must get right on day one rather than retrofit.
- **Splitting files without redrawing boundaries.**
- Enormous investment in transport-failure recovery beside almost none in semantic-failure
  recovery.

### Memory and context

> **Pending** — reader still in flight. Given ~30 state modules and an explicit compression
> layer, this is the section most likely to contain something we actually want.

---

## Cross-cutting findings

*Things true of every implementation read so far. These are the ones that most affect
omega's design, because a gap common to all of them is unlikely to be an accident of one
codebase.*

1. **Nobody verifies that an action achieved its goal.** opencode's snapshot-diff serves undo,
   not success. Hermes ships a verification mechanism that is opt-in, defaults off, and whose
   guard states in its own docstring that it never runs checks. Between them the only real
   postcondition check found anywhere is re-reading a file hash after a write. The literature
   calls unverified completion the dominant source of false success, and two mature systems
   confirm nobody has built the check. **"Done = a verified state change" is therefore not a
   principle we restated — it is the differentiator, and there is no prior art to copy.**

2. **Both invest heavily in transport failure and barely at all in semantic failure.**
   Retries, backoff, `Retry-After` honouring, error classification — thousands of lines. A
   tool that ran, returned cleanly, and did the wrong thing gets nothing. That asymmetry is
   the same blind spot as (1), seen from the error-handling side.

3. **Both chose skills as files on disk, and one added a curator.** opencode pins names and
   descriptions and loads bodies on demand; Hermes scans a directory, injects bodies as user
   messages, and lets the model rewrite its own skills under a background consolidator. Two
   independent implementations converging is evidence, and it suggests **authored** skills
   (files, versioned, editable by hand) and **learned** skills (derived, in memory, written by
   omega about you) are two different things rather than one — DL-014 chose the second, and
   should probably carry both, distinguished by who wrote it.

4. **Both keep the model's live tool surface deliberately small, and the one that didn't had
   to retrofit a bridge.** opencode ships ~14 built-ins filtered per turn. Hermes ships ~58 on
   every call and subsequently built a three-tool search/describe/call bridge to hide the long
   tail. Nobody who has run one of these at scale believes a big flat catalog works.

5. **Growth pressure eventually gets pushed out of the tree.** Hermes froze in-tree plugins
   and redirected to a reviewed pinned catalog; opencode's equivalent is a file-drop seam and
   plugin functions that need no core edits. The difference is that opencode had the seam from
   the start and Hermes had to declare a freeze — which is the whole argument for fixing the
   extension contract before the first external consumer exists.
