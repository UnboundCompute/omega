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
| — | *Hermes rows pending — readers in flight* | Hermes | — | — |

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

> **Pending.** Two readers are in flight — one on memory and context, one on the loop, tools
> and skills. This section and the register's Hermes rows fill in when they land.

One observation available without reading any code: the repository root contains roughly
thirty `hermes_state_*.py` modules side by side — schema, compression, fts, search,
sessions, timeline, rewind, maintenance, wal, guard, repair, and more. Whatever else is
true, the state layer is not knowable from one file.

---

## Cross-cutting findings

*Things true of every implementation read so far. These are the ones that most affect
omega's design, because a gap common to all of them is unlikely to be an accident of one
codebase.*

1. **Nobody verifies that an action achieved its goal.** Both systems trust the tool's
   return. The literature says this is the dominant source of false success. omega's "done =
   a verified state change" is therefore not a truism restated — it is the differentiator,
   and there is no prior art to copy.
2. **Durable cross-session memory is the unbuilt part.** (Pending confirmation from the
   Hermes read, where the answer may well differ — it has a large state layer.)
