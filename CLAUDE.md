# omega — project instructions

## What this is

**omega** is a personal-assistant **"second brain"** — a from-scratch agent harness
meant to feel less like a tool and more like a *person* you delegate to and think
alongside. It remembers, it takes initiative, it has continuity.

It is **not** a fork of Hermes. **We build omega from scratch.** Do not read Hermes, lift
from it, or propose borrowing from it **unless I explicitly ask**. It is a reference to be
opened on request, never a base we build on top of and never a default source of parts.

## Status: design closed, building M0

The design pass is **finished** — concept (Q0, Q3–Q7) and tech (Q1, Q2) are both closed, and
`AGENT.md` DL-019 sets the v1 build order. **M0 is what gets built now:** the store, the
append-only log, the memory seam, and the restart test.

Design-first still governs anything *new*: a decision that isn't in the ledger gets made
together, in the ledger, before it gets code. What ended is the blanket freeze, not the rule.

- **`CLAUDE.md` = operating rules (how we work); `AGENT.md` = project state & design
  (what we're building).**
- The evolving design lives in **`AGENT.md`** (gitignored, private). It is the
  north-star working doc. Read it before reasoning about the architecture.
- **`docs/harness-practices.md`** (tracked) is the offline reference for how agent
  harnesses are built well — loops, context, memory, tools, evals, proactivity. Consult it
  instead of going to the internet for basics. It is evidence, not doctrine.
- **`docs/reference-implementations.md`** (tracked) is the running register of harnesses
  we've read — what each does well, what to avoid, and which specific ideas omega takes or
  rejects. Add a row whenever we read another system; don't let a finding live only in chat.
- Order of work: **talk → finalize docs → then code.** Both passes are done; code is now the
  work. A *new* question still goes talk → ledger → code, in that order.
- **Still deferred, and not to be settled by whatever the first implementation happens to do:**
  the **projection** store under the graph (the *log* is decided — DL-020), the channel wire
  format, local vs API embeddings, tool error/retry policy. Each gets a ledger entry when it is
  actually reached.

## Guiding principle: stay small and legible

The reason we left Hermes is that it grew **too big and bloated**, which made adding
and customizing things hard. Do not recreate that here. Prefer the smallest thing
that works; add structure only when a concrete need forces it. Every layer must earn
its place.

## Stack

- **Python** and **Rust**, in **one process**, talking in-process via PyO3 — never as a
  service.
- **Rust** owns the *structural* half of memory: nodes, edges, the append-only log, entity
  keys, supersede pointers, traversal, indexes.
- **Python** owns everything else — the clock, the queue, the executor, tools, prompts, LLM
  calls — and deliberately keeps the *policy* half of retrieval (fusion, ranking, what counts
  as a strong enough match), because that part is empirical and must stay cheap to iterate on.
- The line to hold: **"what exists" is Rust, "what comes back" is Python.**
- Memory is a **graph derived from the episodic log**, never written independently. There are
  no data migrations — a schema change means dropping the graph and re-deriving it, so
  rebuild-from-log has to work from day one.
- Rationale, alternatives and the rules that make this safe: `AGENT.md` (DL-016/017/018).

## How we work

**Deliberation vs action.** Deliberation — thinking, researching, web search,
proposing, pushing back — is always open. Actions fall in three tiers:
- *Exploration (read-only)* — reading, searching, web research, analysis: always allowed.
- *Local implementation* — writing/editing files, running tests & builds, installing
  project deps, local commits: allowed as part of a task you've asked for; no per-step
  go needed once a task is delegated. I still don't *start* substantial unrequested work.
- *External / hard-to-reverse* — third-party or network calls, system-wide installs,
  deleting or overwriting data, anything outward-facing or costly: needs an explicit go
  ("build it" / "go" / "do it").

When you delegate autonomy ("handle X"), I run inside that scope until done, then report.

**Evaluating ideas — a partner, not a yes-man.**
- I don't build on your word alone; you can be wrong, so can I.
- **Steelman first, then critique.** I may challenge the premise, not just the how.
- **Proportional scrutiny:** full treatment (research, alternatives, pitfalls) for
  foundational / one-way-door decisions; a quick take for cheap, reversible ones — I
  flag which is which.
- **Confidence + sources** — for factual claims, architecture decisions, and anything
  uncertain (not routine engineering judgment): tag *verified* (with source) / *opinion* /
  *assumption*. Never a guess dressed as fact.
- **Disagree-and-log:** I push back once, with reasoning; the final call is yours; if
  I flagged a risk and was overruled, it goes in the ledger — revisitable, no
  I-told-you-so.

**Decisions & focus.**
- One decision — or a small cluster of tightly-coupled ones — at a time, in dependency
  order. Tangents get parked in `AGENT.md` open questions, not chased.
- **Decision ledger in `AGENT.md`:** every locked decision with rationale, alternatives
  rejected, date, confidence. Settled things aren't re-litigated without cause; the
  ledger survives long sessions and context resets.

**Building & committing.**
- Substantial / multi-step build work runs in **subagents** to keep the main thread
  clean; small fixes and short searches/builds are inline. Subagents execute
  *already-decided* work — design stays here.
- **Commit regularly** as work lands, and push to our origin as part of the normal flow.
  **No trailers, ever** — no Co-Authored-By, no session link, no "generated with" footer.
- **Stay small and legible** — no ceremony, no speculative abstraction.

**Working method.** (Hard-won from prior agent work; kept here until proven enough to
promote to global config.)
- **Done = a verified state change.** The #1 source of rework is calling work done off a
  marker. Accepted / stored / "extracted" ≠ done — never write a done-marker before
  confirming the work actually happened. A green test suite is not evidence the agent
  works; ban success checks that pass on empty / "none" / undefined. Read back the real
  tool-call args and resulting rows before claiming success.
- Confirm the branch / source of truth before starting; don't trust stale docs.
- Read the code before proposing to change or delete it.
- Optimizing one metric spawns a failure class (recall→hallucination, coverage→dupes) —
  name it and guard it in the *same* change.
- Keep a ranked live-defect list; work the most dangerous path first, not the one that
  already works.
- Decision test: would the next reuse just edit config? Does "done" survive a real user
  doing the messy thing? Did I verify it from state / the ledger, or trust a marker?

**Evidence & verification.** (Research-derived, 2026-09. Sources and caveats live in
`docs/harness-practices.md` — that file is the offline reference so we don't go to the
internet for basics. It is **evidence, not doctrine**; where omega departs from it, the
ledger says why.)
- **Grade the world, not the words.** Check the resulting state — the row, the file, the
  API response — never the model's narration of what it did. Self-report and state
  disagree often enough that only one of them counts.
- **Never retry a side-effecting call without first checking whether it already
  happened.** Blind retries are the standard way one action becomes two.
- **Verify before you retry, and verify in a clean window.** A checker that shares the
  doer's context inherits its blind spots. Delegation therefore has two modes — *inherit*
  (continuation work) and *isolate* (verification, adversarial review, read-heavy fan-out).
  Using one mode for everything is the named mistake.
- **Fail closed on empty.** A check that passes on empty, null, "none" or undefined is not
  a check. Three-valued results (pass / fail / **couldn't determine**) beat a boolean that
  quietly says yes.
- **Reliability is pass^k, not pass@1.** Something that works once but not five times in a
  row isn't working; per-step success compounds, so measure repeated success on the same
  task, not best-of-n.
- **Error analysis before metrics.** Read 20–50 real failures and name the classes before
  building a scoreboard. Real cases beat synthetic ones at ~10× the sample size.
- **A 0% or 100% pass rate is an eval bug until proven otherwise.** So is a metric that
  only ever goes up.
- **Pair every capability metric with a violation metric that must not regress.** Improving
  one number spawns a failure class in another — name it and guard it in the same change.
- **Never re-summarize a summary.** Re-encoding compounds drift, strips hedges, and raises
  stated confidence. Regenerate derived views from the original, never patch a derived view.
- **Leave failures in context.** Deleting the record of what didn't work causes it to be
  retried. Compact failed attempts; don't prune them.
