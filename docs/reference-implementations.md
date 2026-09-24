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
| 32 | Compaction publishes a **child** session; the parent's raw messages survive, linked | Hermes | **TAKE** | DL-013 (raw survives) |
| 33 | System prompt ordered **by volatility** — stable → context → volatile — for prefix caching | Hermes | **TAKE** | DL-008 (resident set) |
| 34 | Protect a head *and* a tail (first 3, last 20); summarize only the middle | Hermes | **TAKE** | DL-013 (`head_tail`) |
| 35 | Derived memory **hard-capped in characters**, forcing consolidation instead of growth | Hermes | **ADAPT** | DL-008, DL-009 |
| 36 | A background pass forks the agent every ~10 turns to ask "should memory change?" | Hermes | **ADAPT** | DL-009 (mutation points) |
| 37 | Keyword/BM25 recall, plus a second tokenizer for scripts the default one silently breaks on | Hermes | **ADAPT** | DL-008 (recall) |
| 38 | Two unrelated stores both named "memory", plus a third pluggable one | Hermes | **AVOID** | *the singleton rule* |
| 39 | Transcript is mutable — hard `DELETE`, and past message content editable in place | Hermes | **AVOID** | DL-007 (lossless raw) |
| 40 | Derived memory **frozen at session start**; what you say today lands tomorrow | Hermes | **AVOID** | Continuity |
| 41 | No entity identity anywhere — `user_id` is plain TEXT, no persons table | Hermes | **AVOID** | DL-009 (knowing-you) |
| 42 | Retention = binary archive + hard delete at 90d, off by default, no tiering | Hermes | **AVOID** | DL-008 (deletion ≠ demotion) |
| 43 | Five separate "shrink the context now" trigger paths converging on one 5,299-line class | Hermes | **AVOID** | DL-013 |
| 44 | Facade bypassed by raw SQL from ≥5 outside files; one table declared twice | Hermes | **AVOID** | DL-014 (the seam) |
| 45 | The compressor **feeds its own prior summary back in** and updates it, session after session | Hermes | **AVOID** | *never re-summarize a summary* |
| 46 | Two unrelated "survive compaction" paths — marker-tagged user message *and* prompt pinning | Hermes | **ADAPT** | DL-008 (one pinning rule) |
| 47 | Instruction files merged **down the whole directory chain**, with per-directory provenance | Hermes | **ADAPT** | DL-008 (associative pull) |
| 48 | Thresholds, caps and cooldowns scattered as constants across ≥4 files; no tunables module | Hermes | **AVOID** | *stay small and legible* |
| 49 | Automatic associative recall — anything pulled in because it *became* relevant | both | **N/A** | — (the gap) |
| 50 | Proactive output **batched into one daily delivery**, never streamed as events fire | ChatGPT Pulse | **TAKE** | DL-011 (firehose) |
| 51 | The delivery is **deliberately finite** and closes with an explicit terminator | ChatGPT Pulse | **TAKE** | DL-011 |
| 52 | Volume capped at 5–10 items; skim the card, tap for the whole thing | ChatGPT Pulse | **ADAPT** | tray panel |
| 53 | Data connectors **off by default**; each one opted in separately | ChatGPT Pulse | **TAKE** | DL-011 |
| 54 | Per-item feedback + topic curation + a feedback history you can read *and delete* | ChatGPT Pulse | **ADAPT** | DL-009 |
| 55 | Notify only on a **terminal state** *and* only when the user **appears away** | Claude Code | **TAKE** | DL-011 (when to speak) |
| 56 | Quiet channel by default; the louder channel is opt-in per environment | Claude Code | **TAKE** | tray |
| 57 | The trigger is "the agent stopped and needs you", not "something happened" | Aider | **TAKE** | DL-011 |
| 58 | Asking and declaring-done are **typed tool calls**, not free prose | Cline | **TAKE** | DL-042, DL-055 |
| 59 | "Do not indicate that you will perform an action without actually doing it" | Cline | **TAKE** | DL-055 (corroboration) |
| 60 | Recurring scheduled research that reports back on a fixed cadence | Tasks, Perplexity | **ADAPT** | DL-035 |
| 61 | Hard per-tier ceilings on event-triggered runs (30/hour, 720/day) | ChatGPT Tasks | **TAKE** | DL-036 (unattended cost) |
| 62 | An agent that can't reliably tell the time, scheduling its own reminders | ChatGPT Tasks | **AVOID** | DL-035 |
| 63 | Proactive cards inferred from passive behaviour logging; wound down to a feed | Google Now | **AVOID** | DL-011 |
| 64 | Acting irreversibly for the user **without asking first** | Instinct | **AVOID** | DL-014 (approval) |
| 65 | A proactive channel that **outlives its own consent** | Instinct | **AVOID** | DL-011, DL-048 |
| 66 | Retaining user data with **no deletion path** until publicly embarrassed | Instinct | **AVOID** | DL-048 (retraction) |
| 67 | Perpetual irrevocable licence over the user's material, for training | Instinct | **AVOID** | *local-first* |
| 68 | Inbox-reading plus autonomous action = a standing injection target | Instinct | **AVOID** | DL-014 (rings) |
| 69 | Local-first, model-agnostic, reaching the user through chat apps they already run | OpenClaw | **ADAPT** | DL-004 (channels) |
| 70 | Reactive-only assistants died of latency and reliability — proactivity was never the issue | Humane, Rabbit | **N/A** | — (honest negative) |

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
- **The system prompt ordered by volatility** so the cacheable prefix stays stable.
- **Compaction that publishes a child session and keeps the parent's raw messages**, linked by
  lineage — the one place a read system got raw-survives-derivation right.
- **A head *and* a tail protected**, middle summarized — independent corroboration of
  `head_tail`.
- **A character cap on curated memory** that forces consolidation rather than accretion.
- **Merging instruction files down the whole directory chain with provenance**, rather than
  letting the nearest file shadow the rest.

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
- **Three separate things called "memory."** Adding a new kind of remembered thing starts with
  choosing which of three unrelated systems it belongs to.
- **A mutable transcript** — hard deletes, and past messages editable in place.
- **Derived memory frozen at session start**, so what you tell it today lands tomorrow.
- **No entity identity at all** — strings remembered, never who or what they are about.
- **Re-summarizing its own summaries** in an iterative-update loop.
- **Two unrelated mechanisms for "must survive compaction"** with no shared abstraction.
- **Tunables scattered as constants across four-plus files**, producing several unrelated
  `0.85`s that even their own docs conflate.

### Memory

**There are two unrelated systems here, both called "memory," plus a third that's pluggable.**
That is the headline finding, and it is the clearest single illustration of the bloat
complaint in the whole codebase.

*System A — the raw transcript, in SQLite.* One schema definition (`hermes_state_common.py`),
sessions and messages tables plus operational tables for usage, routing, locks, leases and
delegations. Sessions carry a `parent_session_id` self-foreign-key used for compaction and
fork lineage. Crucially, **storage is mutable, not append-only**: the default delete path is a
soft `active=0` flag, but hard `DELETE` is used routinely by pruning and clearing, and a past
message's *content* can be updated in place after the fact. A transcript you can rewrite is
not an audit trail, and DL-007's lossless raw log is a deliberate departure from this.

*System B — curated facts, in flat files.* `MEMORY.md` and `USER.md` under `~/.hermes/`, not
in the database at all, holding free-text entries joined by a separator. There is no entity
schema — no person key, no project key, just strings. Entries get there two ways: the model
calling a `memory` tool, or **a background reviewer that forks the live agent roughly every
ten turns** and asks a model whether memory or skills should change. That's a real extra LLM
call, self-documented at around 30K tokens an event.

*System C — pluggable third-party vector/semantic memory* behind a provider interface,
separate from both.

Two things here are genuinely good. First, **A and B coexist rather than one overwriting the
other** — the curated layer never touches the raw transcript, which is exactly the
raw-survives-derivation property DL-013 insists on and the thing opencode gets wrong. Second,
**both curated stores are hard-capped in characters** (about 2,200 and 1,375). A cap that small
forces the model to *consolidate* — to decide what earns a slot — instead of appending
forever. That is a much more interesting pressure than a token budget, and it is worth
considering for omega's resident set independently of whether the numbers transfer.

**Recall is keyword search, not embeddings.** There is no vector index in the core at all:
recall is SQLite FTS5 with BM25 ranking, plus a second virtual table with a different
tokenizer added specifically because the default one needs three-character terms and silently
degrades to a full scan on CJK text. That second table is a good lesson in itself — a
retrieval default that fails *quietly* on a whole class of input is worse than one that fails
loudly.

But recall is **explicit-tool-invoked only**. The model must decide to search. The one thing
injected automatically is System B's `MEMORY.md`/`USER.md`, **frozen into the system prompt at
session start** to protect the prompt cache and not refreshed mid-conversation except after a
compaction event. The consequence is worth stating plainly, because it's the exact failure
omega exists to avoid: *tell it something about yourself today and it does not take effect
until tomorrow.* Continuity is traded away for a cache hit.

**There is no entity identity.** `user_id`, `session_key` and `chat_id` are plain TEXT columns;
there is no persons table, no projects table. Workspace identity is recomputed per query from
the working directory rather than stored as a key. The only real relationship in the schema is
session lineage. So the "knowing-you" half of omega has no prior art here either — Hermes
remembers *strings*, not *who or what they are about*.

**Eviction doesn't exist as a concept.** There's no hot/warm/cold tiering. Archiving and
pruning both default to off; when enabled it's a binary archived flag plus a hard delete after
a retention window, throttled by a stored timestamp rather than a scheduler. Everything is
kept forever, or deleted — which is precisely the deletion-vs-demotion collapse DL-013 names.

### Context

Prompt assembly is the best-designed part of the memory/context half. `build_system_prompt_parts`
composes **three explicitly ordered tiers, ordered by volatility** so the longest common prefix
stays stable: *stable* (identity, guidance) → *context* (project files, workspace snapshot) →
*volatile* (skills index, memory files, timestamp, runtime hints). The result is cached and
rebuilt only at session start or after compaction, and a separate layer places the provider's
cache breakpoints — four of them, with per-provider carve-outs for envelopes that relocate or
reject part-level markers. Cache preservation is a stated project invariant.

That ordering rule is worth taking directly. It is the same insight as opencode's unshipped V2
epoch design, reached by a different route: *don't rebuild the prompt, and lay it out so the
parts that change are at the end.*

Compaction is two files totalling ~9,600 lines. Defaults protect the first 3 and last 20
messages and summarize the middle at 50% occupancy — independent corroboration of the
`head_tail` result in `harness-practices.md`, which is a useful confirmation since that finding
was the basis for reversing DL-008's uncapped clause. And compaction **publishes a child
session** rather than rewriting rows: the summary becomes the head of a `parent_session_id`
chain while the original messages remain soft-archived in the parent. Compare opencode, which
replaces old tool results with a "cleared" marker. Hermes keeps the raw. That is the right
shape and we should take it.

Three cautions sit against that, though, and the second is serious:

- **Five distinct "shrink now" trigger paths** converge on the 5,299-line compressor — idle
  wall-clock, preflight threshold, reactive provider-error, plus two off-by-default
  sub-mechanisms living inside the same class. An earlier reading of their own docs suggested
  a clean "gateway at 85%, agent at 50%" split; the code doesn't support it. There are at
  least four unrelated `0.85` constants in the tree — a degenerate-window cap, a third-party
  compaction threshold, a local-runtime window-growth policy — and they mean different things.
  Their own documentation had flattened that into one tidy sentence that isn't true.
- **The compressor re-summarizes its own prior output.** In iterative-update mode it feeds the
  previous summary back in verbatim alongside new turns and asks the model to preserve, update
  or retire items. Over a long session that is summary-of-summary compounding — the precise
  failure our working method names as *never re-summarize a summary*. Finding it shipped in a
  mature system is the strongest argument yet for regenerating derived views from raw instead
  of patching them. It also explains why raw-survives matters so much: without the parent
  chain, this would be unrecoverable drift.
- **Thresholds, caps and cooldown seconds are scattered as module constants across at least
  four files**, with no central tunables module. That is how you end up with four `0.85`s that
  nobody can reconcile.

One more structural finding worth carrying over: Hermes has **two unrelated mechanisms for
"this must survive compaction."** Identity, memory and skills survive by being pinned in a
system-prompt tier. Open to-do items survive by an entirely different route — in-memory state
re-emitted as a synthetic *user* message behind a stable header marker that the compressor
itself recognizes and strips. Same underlying need, two code paths, no shared abstraction.
omega should have exactly one answer to "what survives," and both cases should use it.

### Findability, as a bloat symptom

Three concrete data points, all of them about names not matching jobs:

- `hermes_state_schema.py` does not define the schema; it holds migration and column
  reconciliation. The schema is in `hermes_state_common.py`.
- `prompt_builder.py` (1,767 lines) is not the prompt orchestrator; `system_prompt.py` is.
- Of ~30 `hermes_state_*.py` siblings, six hold nearly all the write paths; the other 24 are
  SQLite operational hardening — justified by genuine multi-process access, but it means
  answering "how does recall work" requires reading four files plus a fifth for the SQL.

Their own contributor docs concede it: *"reading the facade first is the expensive way."* And
the facade isn't even sole owner — at least five files outside the state family run raw SQL
against `sessions` directly, and two tables are declared a second time elsewhere in the tree,
flagged in a comment as a known drift source.

Documentation drift is the independent corroboration that this is a maintenance-burden problem
rather than an aesthetic one: the storage doc states a schema version seven behind its own
migration table, and the prompt-assembly doc describes instruction-file discovery as
current-directory-only when the code actually walks the whole chain from the repository root
down and merges every level with per-directory provenance. That last one is a *feature* the
documentation loses — which is its own argument for keeping the system small enough that the
docs can stay true.

(That chain-merge, incidentally, is a better idea than opencode's nearest-ancestor-wins rule,
and it's row 47: merge the whole chain, keep provenance, rather than letting the closest file
silently shadow everything above it.)

---

## 3. The proactive-messaging survey — how shipping assistants speak, and when

**Read 2026-09-24. A different evidence standard, stated up front.**

Everything above this line is a source-tree read: a claim points at a file and a line, and
you can go and check it. This section cannot do that, because the systems it covers are
mostly closed. What it rests on instead is vendor documentation, product announcements and
press reporting, and where a claim is only a press report it is labelled as one. Three items
*are* source- or doc-verified and are marked **[verified]** with where; treat the rest as
what a company says about itself, which is a weaker thing.

Two gaps are admitted rather than papered over. The corpus of criticism about assistant
tone — the "You're absolutely right!" genre — was not collected: the search path for it was
blocked. And Apple's notification interruption levels, which would have been the natural
external taxonomy for "how loud is this", could not be fetched from Apple's own
documentation and so are **not** cited here. Neither absence is evidence of anything.

The question this section exists to answer is not "how should an assistant word a reply" —
DL-055 settled that, and §6 of `harness-practices.md` holds the research layer. It is the
narrower and harder one: **when does a system decide to speak first, and why.** That is
DL-011's territory, and DL-011 names the terminal failure — the notification firehose.

### The batch, not the stream

**ChatGPT Pulse** (launched 25 Sep 2025, Pro on mobile first) is the most considered answer
to proactivity currently shipping, and its shape is worth copying before its content is. It
works overnight and delivers **once**, in the morning, as five to ten cards. It is
deliberately finite: the run ends, and the last card says so — *"Great, that's it for
today."* OpenAI's stated reason for the cap is that people should be able to **get back to
what matters** rather than scroll, which is an unusual thing for an engagement-funded
product to say out loud and is the part omega should take seriously.

The rest of Pulse's controls read like a list of the things DL-011 will need anyway.
Connectors — Gmail, Calendar — are **off by default** and opted into one at a time. You can
curate the topics it works on. Each card takes a thumbs up or down, and the feedback history
is itself viewable and deletable, so the model of you is legible and revocable rather than
accumulated silently. The cards are swipeable and skimmable; the depth is behind a tap.

The lesson generalises past the product. **A batch is governable and a stream is not.** A
batch has a size you can cap, an end you can announce, and a schedule the user can move. An
event stream has none of those, which is why every system that streams eventually grows a
mute button and then dies of it.

*Not verified: the card count, the terminator wording and the connector defaults are from
OpenAI's launch material and contemporaneous coverage, not from a build.*

### The schedule, and its ceilings

**ChatGPT Tasks** is the explicit-schedule half: one-time or recurring, free tier limited to
roughly one a day inside coarse windows (morning / afternoon / night), paid tiers getting
hourly and exact times. The number to carry across is the one on **event-triggered** runs —
connector events from Gmail, Slack, GitHub are capped at **30 an hour and 720 a day**. That
is a vendor with effectively unlimited compute deciding that an agent reacting to inbound
events needs a hard ceiling, which is the same conclusion DL-036 reached from the cost side.

Tasks also supplies a clean **AVOID**: reports of tasks overwriting one another, and of the
model being unable to reliably tell what time it is while scheduling something for later.
An agent that schedules its own future work needs the clock to be a tool result, never a
thing it believes. That is already how DL-035 is built; this is the failure that justifies it.

**Perplexity's Scheduled Searches** are the same pattern with less surface: daily, weekly or
monthly, notify with the result. Nothing to take beyond confirming the shape is standard.

### The terminal state, and the absent user

The two coding harnesses converge on a trigger rule that is sharper than anything in the
consumer products, and both are checkable.

**Claude Code** **[verified — `code.claude.com/docs/en/terminal-config`]**: it fires a
notification when Claude *"finishes a task or pauses for a permission prompt, **and you
appear to be away from the terminal**."* Two conditions, joined by an `and`. The first is a
**terminal state** — the work stopped, either done or blocked on the user. The second is an
**attention check** — the user is not already looking. Nothing fires because something
merely *happened*. It also defaults quiet: real desktop notifications only where the
terminal supports them natively, otherwise you opt in to a bell or a hook.

**Aider** **[verified — `aider.chat/docs/usage/notifications.html`]** states the trigger in
one sentence: notify when *"the LLM has finished generating a response and is waiting for
your input."* Opt-in behind `--notifications`, with `--notifications-command` to route it
anywhere — Slack, Discord, Pushbullet.

Both say the same thing in different words, and it is the single most useful rule in this
survey: **the message is "I have stopped and I need you", not "something occurred."** A
terminal state is a bounded, countable, non-repeating event. "Something occurred" is a
firehose with extra steps. Pair that with the absent-user gate and you get a proactivity
policy that is two predicates long and hard to abuse.

### Asking and finishing as typed acts

**Cline** **[verified — `sdk/packages/shared/src/prompt/system/act.ts` and
`apps/vscode/src/shared/tools.ts`, read via the GitHub API]** contributes something
structural rather than temporal. `ask_followup_question` and `attempt_completion` are
entries in a tool enum — **asking the user a question and declaring the work done are typed
tool calls, not sentences the model happens to emit.** That means both are logged, both are
countable, and neither can be faked by prose that merely sounds like a question or a
completion. It is the same instinct as DL-042's claim-plus-receipt, arrived at from the
other direction.

Its prompt also contains an independent restatement of DL-055, which is worth quoting
because it was written by people who had no idea omega existed:

> Do not indicate that you will perform an action without actually doing it. Always provide
> the final result in your response. Always validate your answer with checking the code and
> running it if possible.

That is *grade the world, not the words*, written as a prompt rule. Two harnesses reaching
it separately is the closest thing to external confirmation DL-055 is going to get.

*One negative to record honestly: Cline's well-known tone rules — the "never open with
'Great' or 'Certainly'" family — could not be found at current HEAD. The prompt has been
refactored. They are therefore **not** cited here as verified, despite being widely quoted.*

### The old failure, and the new one

**Google Now** is the ten-year-old version of this idea and the instructive corpse. Cards
inferred from repeated actions, location, calendar and search history, surfaced without
being asked for. Google began winding it down in 2015 and folded "Now cards" into the
undifferentiated "Feed" in October 2016. The criticism that stuck was not that the cards
were wrong — it was that they were *right*, and being right revealed how much Google knew.
The cards were an accidental disclosure interface for the surveillance underneath them.
**Proactivity is a confession.** Every unprompted message says "here is what I have been
watching", and a system whose watching is not something the user chose will be experienced
as creepy exactly in proportion to how good it is.

**Instinct** is the 2026 version, and it is the anti-case this whole section is worth
writing for. Reported by TechCrunch on 2026-08-24 — after this document's other sections
were written, and after the assistant's own knowledge cutoff, so everything here is press
reporting and nothing is verified. A stealth SF company, Spear Street Technology, with a
research team out of Sierra. It reaches users by **text, WhatsApp and phone calls**. It
connects email, messaging, calendar, device audio, location and screen. It books
appointments, transport and flights, manages an inbox, shops. Reception was ecstatic — *"like
magic"*, one of the most exciting launches since OpenClaw — and it raised a $250M Series B
at a $2.5B valuation from Index and Benchmark after publication.

The reported failures are, one for one, the architecture omega chose against:

- It enters binding *"agreements, commitments, or transactions"* on a user's behalf,
  **sometimes without prior approval**. One user found it had *"sent an email on her behalf
  without checking with me first."* That is DL-014's approval ring, absent.
- It indexed and retained emails **without permission and refused deletion requests**; a
  deletion tool was added later, after complaints. That is DL-048's retraction, absent.
- A user disconnected their Gmail and **still received email summaries at 2 PM**, stored
  *"in plain text for later searches."* This is the one to name: **the proactive channel
  outlived its own consent.** The user revoked the input and the output kept arriving. Any
  scheduled speech omega emits has to be derived from a live permission at send time, not
  from a schedule created when the permission existed.
- Its terms grant a *"perpetual and irrevocable license"* to *"access, use, host, cache,
  store, reproduce, transmit, display, publish, distribute, and modify"* user material, for
  training. It receives screen captures, cursor movements and keyboard input. That is the
  case for local-first, made by the opposition.
- A founder deleted his account after finding it phishable; it autonomously pulled
  verification codes out of inboxes. **An agent that reads your mail and can act without
  asking is a standing prompt-injection target with your credentials attached.**

Instinct is not a strawman — it is the most capable product in this survey, and people love
it. That is the point. The failures are not incompetence; they are what you get when
capability ships ahead of the consent machinery, and they are why the boring parts of omega's
design (an append-only log on the user's own disk, approval rings, retraction) are the
product rather than overhead.

**OpenClaw** is the counterweight: open source, **runs locally on the user's own machine**,
model-agnostic across Claude and DeepSeek, reaching the user through Discord, Telegram,
WhatsApp, Google Chat, iMessage or Matrix on macOS, Windows and Linux. Nothing in what was
read indicates it is proactive, so it contributes channel strategy rather than timing
strategy — meet people in a chat app they already run rather than asking them to adopt a new
surface, which is the live question behind DL-004. Its security posture was **not
established in this session**; that is an absence of findings, not a clean bill.

### One honest negative

**Humane's AI Pin** and the **Rabbit R1** are the obvious things to reach for when writing
about ambient assistants failing, and the reach would be wrong. Both were primarily
*reactive* — you spoke to them and they answered, badly, slowly. Their deaths are latency and
reliability stories. Proactive notification UX was not established as their failure mode, and
forcing them into this narrative would be inventing evidence for a conclusion already held.
Recorded as row 70 with a **N/A** verdict for that reason.

### What omega should take from this

Synthesised, the shipping consensus is four rules, and the first two are nearly free:

1. **Speak on a terminal state or a batched schedule — never on an event.** (Claude Code,
   Aider, Pulse.)
2. **Gate on attention: don't speak to someone already looking at you.** (Claude Code.)
3. **Cap the volume and end explicitly.** (Pulse's 5–10 and its terminator; Tasks' 30/hour.)
4. **Never let the channel outlive its consent.** (Instinct's 2 PM digests, inverted.)

These are a *design decision*, not an implementation detail, so per this project's order of
work they belong in the ledger before they belong in code. Flagged for a DL-011 amendment
rather than written into the tray unilaterally.

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

   **But note what Hermes's curator actually is:** *one* background pass that asks a single
   question — "should memory **or skills** be updated?" — and writes to both. Storage is files;
   the *decision* is one decision. That is real support for DL-014's "skills are memory, not
   code," from the system that stores them as code. The split to make is by **author**, not by
   substrate.

4. **Both keep the model's live tool surface deliberately small, and the one that didn't had
   to retrofit a bridge.** opencode ships ~14 built-ins filtered per turn. Hermes ships ~58 on
   every call and subsequently built a three-tool search/describe/call bridge to hide the long
   tail. Nobody who has run one of these at scale believes a big flat catalog works.

5. **Growth pressure eventually gets pushed out of the tree.** Hermes froze in-tree plugins
   and redirected to a reviewed pinned catalog; opencode's equivalent is a file-drop seam and
   plugin functions that need no core edits. The difference is that opencode had the seam from
   the start and Hermes had to declare a freeze — which is the whole argument for fixing the
   extension contract before the first external consumer exists.

6. **Neither system models entity identity.** opencode has no cross-session memory to have
   identity in. Hermes has memory, and its identity columns are plain TEXT with no persons or
   projects table — it remembers *strings*, never who or what they are about. The only real
   relationship either schema expresses is session lineage. So the "knowing-you" half of omega
   — facts attached to people, projects and commitments that can be reconciled and superseded —
   is, like verification, **unpaved road**. Two reads, zero prior art.

7. **Neither gets the raw/derived relationship fully right, but they fail at opposite ends.**
   opencode *replaces* old tool results with a "cleared" marker — derived text becomes
   authoritative and the raw is gone. Hermes keeps the raw (soft-archived in the parent session,
   reachable by lineage) but then **re-summarizes its own summaries**, compounding drift on top
   of a raw log it no longer consults. The correct combination is neither: **keep raw
   losslessly, and regenerate derived views from it rather than patching them.** DL-007 and
   DL-013 already say exactly this; what's new is that both halves are now observed failing
   separately, in production, in mature systems. That moves the rule from a principle we
   reasoned to into one we've watched two codebases pay for.

8. **In a large harness, the documentation stops being true — and the failure is silent.**
   Hermes's storage doc is seven schema versions behind its own migration table; its
   prompt-assembly doc describes a feature as narrower than the code actually implements; its
   compaction docs flatten several unrelated constants into one tidy claim that direct reading
   doesn't support. opencode's version of this is a live V1/V2 rewrite where neither is labelled
   as the one that runs. In both cases *the code was fine and the map was wrong* — which is the
   sharpest argument for `CLAUDE.md`'s "stay small and legible" that either read produced. Small
   isn't an aesthetic preference; it's the condition under which the description of the system
   can stay accurate.
