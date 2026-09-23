# M1 — one turn, end to end, logged

The build contract for M1 (`AGENT.md` DL-011, DL-016, DL-019). **This milestone *is* the core
loop.** Nothing else in M0–M6 builds a second one.

M0's spec could open by saying everything in it was decided. **This one cannot, and saying so
is the point.** M1 sits on three things the ledger deliberately left open — the queue
mechanism (inside Q1), tool errors and the stop threshold (inside Q3), and the channel wire
format (named in `CLAUDE.md` as *not to be settled by whatever the first implementation
happens to do*). So this file is in three parts: **Decided** (derived from the ledger, build
it), **Proposed** (my call, reversible, flagged), and **Needs your call** (material enough
that guessing wrong means rework, not a patch).

## What M1 is

A single resident process that drains one durable queue serially, running the full turn —
**perceive → recall → judge → act → reply → write memory** — and logging every exchange.

**Done-bar (DL-019):** daily use starts here; every exchange lands in the log.

**Not in M1:** the graph, entity resolution, real retrieval policy, derived what's-open,
initiative, the clock. Recall is deliberately dumb and that is the design, not a shortcut —
M1's job is to *start the corpus*, and retrieval quality is tuned afterwards on real episodes
(DL-019).

---

# Part 1 — Decided

## 1.1 The queue is the log, and the cursor is a checkpoint

This is the load-bearing result of M1 and it is *derived*, not chosen.

DL-016 fixes **zero authoritative state in RAM**. An in-memory queue holds authoritative
state: events accepted but not yet processed. `kill -9` with three events queued loses three
events, and the restart test says omega loses **only the turn that was mid-flight**. So the
queue must be durable. The only durable thing M0 built is the log.

Therefore:

```
enqueue(event)   ≡  store.append_episode(payload, write_key=...)
pending()        ≡  store.episodes_since(store.checkpoint(CLAIMED))
claim(episode)   ≡  store.set_checkpoint(CLAIMED, episode.seq)
finish(episode)  ≡  store.set_checkpoint(DONE,    episode.seq)
```

**Two named checkpoints, not one** — `CLAIMED = "executor.claimed"` and `DONE =
"executor.done"`. M0 supports named checkpoints and `checkpoint_names()` precisely so there
can be several; this is the first real use. One consumer, one drain, serial by construction.
DL-016 predicted "at-most-once and ordering fall out free" — they fall out of the checkpoints
being monotonic and the log being totally ordered by `seq`. No queue library, no second store,
no new infrastructure. **M0's checkpoint API stops being speculative and becomes
load-bearing.**

**The drain advances past records it does not process.** `turn.completed` and `tool.*` are
appended to the same log, so they appear in `pending()` too. The drain either *processes* an
episode (event kinds) or *skips* it (record kinds), advancing both cursors either way. Without
this the executor would treat its own output as a new event and loop forever — the one
genuinely sharp edge in reusing the log as the queue, and it is a three-line guard, not a
second store.

*Consequence worth stating plainly:* this makes `episodes_since` and `set_checkpoint` the two
hottest calls in the system, so M0's `CheckpointAhead` behaviour is now a liveness property
of the loop, not a corner case.

## 1.2 Claim-before-act, and the mid-flight turn is visible, not replayed

**The checkpoint advances when a turn is claimed, not when it completes.**

The alternative — checkpoint on completion — gives at-*least*-once: a crash mid-turn
re-delivers the event on restart and the turn runs again. For a loop whose `act` step calls
tools with side effects, that is the standard way one action becomes two, which `CLAUDE.md`
names directly: *never retry a side-effecting call without first checking whether it already
happened.*

So M1 is at-most-once, and the lost turn is **recovered as information rather than as a
replay**. The two cursors make the detection an integer comparison rather than a search:

```
claimed > done   →  the episode at seq `claimed` was interrupted
claimed == done  →  nothing was in flight; a clean stop
```

- every claimed turn ends by appending a terminal record — `turn.completed`, or
  `turn.blocked` when it stops to ask you something (§2.1) — **and then** advancing `DONE`;
- on startup, `claimed > done` is the whole check — no scan, no scan window to get wrong, and
  it cannot be fooled by a detached `work.finished` landing between the two;
- omega **says so** — "you asked me X, I was cut off partway, want me to pick it up?" — and
  the decision to resume is a judgement surfaced to you, never an automatic re-run.

*Why the order matters:* `turn.completed` is appended **before** `DONE` advances. Crashing
between them leaves `claimed > done` with a completed record present — omega reports an
interrupted turn that had in fact finished. That is the safe direction of the error: it
over-reports a cut-off turn rather than silently dropping one, and the report is a question to
you, not an action. The reverse order would lose turns silently.

This is exactly DL-016's restart test made concrete: *lost nothing except the turn that was
mid-flight — which it can tell you about, because the episode was logged before the action
fired.*

## 1.3 Silence must be logged, and now there is a mechanical reason

DL-011 makes silence a **first-class successful outcome**. A silent turn writes
`turn.completed` with `reply: null` and `outcome: "silent"`, and advances `DONE` exactly like
a speaking turn.

*An earlier draft of this spec claimed a mechanical forcing argument here — that skipping the
record would make silence indistinguishable from a crash. Under the two-cursor design in §1.2
that is no longer true: the cursors detect the crash, not the record's absence. The honest
reasons silence is logged are the ledger one and the corpus one — DL-011 makes it an outcome,
and the done-bar is "**every** exchange lands in the log", of which a deliberate silence is
one. Stated plainly so nobody later removes the record on the strength of an argument that
does not hold.*

What the design does still enforce mechanically is the scoring: `outcome` is an explicit
field with `silent` as a peer of `spoke`, not an absence to be inferred. Any later metric that
counts turns, replies or completions can therefore see the difference between *said nothing*
and *failed*, and must not score the first as the second — DL-011 calls this the single most
load-bearing consequence of the time-wake, and M5 becomes a notification firehose if it is
violated.

## 1.4 The turn, step by step

One function, six named steps, in this order. Each is separately testable.

| Step | What it does at M1 | What it becomes later |
|---|---|---|
| **perceive** | Read the claimed episode; decode its payload. | Multi-channel envelope (Track B). |
| **recall** | Last N episodes from the log, newest-last. Nothing else. | M3 graph + policy retrieval. |
| **judge** | **Its own step.** Decide: `speak` · `act_then_speak` · `stay_silent`. | M5 initiative sources feed it. |
| **act** | The inner sub-loop (§1.5). Skipped unless `act_then_speak`. | M4/M5 tool rings. |
| **reply** | Emit outbound, or nothing on `stay_silent`. | Real transport. |
| **write memory** | Append `turn.completed`. | Graph derivation hangs off the same episodes. |

**`judge` is not folded into `act`.** DL-011 is explicit: it is where deciding to say nothing
lives. Merging them puts silence inside an action path, which is how silence becomes a
failure mode instead of an outcome.

## 1.5 `act` is a sub-loop, never a sub-agent

Inside one `act` step: tool → result → tool → result. Each pass re-asks DL-011's two
questions:

- **are we done, verified?** — the mechanism behind "done = a verified state change". A tool
  reporting success is not the answer; the resulting state is.
- **is this moving?** — stall → stop, re-plan, or come back and ask. Never flail silently.

**Sub-loop, never sub-agent.** A sub-agent implies a second context with its own judgement,
which fractures identity and leaves its work outside the episodic log. Sub-agents stay
legitimate only as a context-isolation tool that returns a result and holds no memory.

Long work detaches (DL-016) and its completion **re-enters as an ordinary enqueued event** —
i.e. it is appended to the log like any other wake. It is not a second entry path.

## 1.6 One entry point, even though only one wake exists

Only the you-wake exists at M1. The time-wake is M5. **The queue must already be the single
entry point anyway**, so that M5's clock is a *producer* against the same `append_episode`,
never a second path into the loop. DL-011: two engines would drift, and initiative would
quietly become a different agent wearing omega's name.

Concretely: nothing in M1 may call the turn function directly. The only way to cause a turn
is to append an episode.

---

# Part 2 — Proposed (my call, reversible, flagged)

## 2.1 Episode payload schema — JSON, versioned, one `kind` field

M0 made `payload` **opaque bytes** on purpose, which means this is a Python-side decision
(DL-018: meaning lives in Python) and cheap to change — DL-017's rebuild-from-log replaces
migration, so a schema change is a re-derive, not a migration.

```json
{"v": 1, "kind": "message.inbound", "text": "...", "channel": "tray",
 "context": [{"id": "…", "kind": "file|image|text|link|screen", "title": "…"}],
 "urgency": "normal", "at": "..."}
{"v": 1, "kind": "turn.blocked",   "for_seq": 41, "needs": "…what omega needs from you…"}
{"v": 1, "kind": "turn.completed", "for_seq": 41, "outcome": "spoke|silent|failed",
 "reply": "..." , "tools": [...], "error": null}
```

M1 kinds, and that is all of them: `message.inbound`, `turn.blocked`, `turn.completed`,
`tool.called`, `tool.returned`, `work.finished` (a detached sub-loop re-entering per §1.5).

Three fields exist because the tray requires them (§Q10), not because they were anticipated:

- **`context[].id`** — tray requirement 2, stable identity through delivery *and failure
  recovery*. Ids must be in the payload, not held in the tray's memory, or a `kill -9` on
  either side strands the staged items. The tray keeps the previews; the log keeps the
  identities.
- **`urgency`** — tray requirement 4, normal vs genuinely time-sensitive. Always `"normal"`
  at M1; reserved so M5 does not have to version the schema to add one enum value.
- **`turn.blocked`** — tray requirement 5 lists *blocked* as a peer outcome. It is where the
  sub-loop's "is this moving?" answering **no** surfaces. Without it, a stall has to be
  encoded as a failure, and DL-011 is explicit that stopping to ask is the correct behaviour,
  not a failure.

**`turn.blocked` is terminal for the cursor, and this is not a detail.** A blocked turn is
waiting on a human, which is unbounded; if it stayed claimed, `DONE` would not advance and the
single consumer would stop draining — one question to you would freeze every other event,
including M5's ticks. DL-016 already answers this: long work detaches and its completion
re-enters as an ordinary event. Waiting on a person is the longest work there is.

So blocking **ends** the turn: append `turn.blocked`, advance `DONE`, release the executor.
The block is durable because it is in the log, not because a thread is parked on it — a
`turn.blocked` with no later resolution *is* the pending question, and it survives `kill -9`
for free. Your answer arrives as a new `message.inbound` carrying `resumes_seq`, and is an
ordinary turn with the blocked one in its recall.

*Consequence for §1.2:* a claimed turn ends with `turn.completed` **or** `turn.blocked`.
Both advance `DONE`; only their absence means interrupted.

**Write keys used as a real guard, not decoration.** `turn.completed` for inbound seq N
carries `write_key = "turn:N"`. A double-write is then rejected *by the log itself* rather
than by a convention in the loop — M0's dedup becomes an invariant check instead of an
unused feature.

**Why JSON and not a binary encoding:** it is human-readable during exactly the phase where
we will be reading raw logs to debug the loop, and the log's opacity means we pay nothing
structurally for changing it later. Confidence: medium-high. *Reversible in one commit while
no real log exists — which is true today and stops being true the day daily use starts.*

**This is not the channel wire format.** That stays deferred. This is the *storage* shape of
an episode. Keeping them separate is the whole lesson of DL-006: the stub transport's shape
nearly became the contract by default.

## 2.2 Recall = last N episodes, N = 40, no scoring

Deliberately dumb, per DL-019 — "making recall good first trades a date that cannot be
recovered for a quality that can." N is a labelled guess, not a measured value, and it is
tuned in M3 against the real corpus M1 produces. It reads through `episodes_since`, never
through diagnostics.

Honest limitation: DL-019 describes M1 recall as "recent N + **what's open**". *What's open*
is derived from the log and lands in **M2**. M1 therefore ships the recent-N half only, and
this file does not pretend otherwise.

## 2.3 Stop threshold = a fixed pass cap

Q3 explicitly defers "where the stop threshold actually sits" to the implementation pass.
M1 uses a fixed cap on `act` passes (proposed: 8) plus the is-this-moving check, and **logs
every stop with its reason** so that M2+ tunes the threshold on real stalls rather than on a
guess. A cap is not the answer to the question; it is a floor that makes the question
answerable with data.

## 2.4 Tool errors: surface, never swallow

Also deferred inside Q3. M1's rule: a failed tool call is a `tool.returned` episode with the
error recorded, it is visible to the next pass of the sub-loop, and it is **never retried
automatically**. Tool results are untrusted input (DL-014). Retry policy proper gets a ledger
entry when a real failure pattern exists to design against — per `CLAUDE.md`, error analysis
before metrics.

---

# Part 3 — Needs your call

## Q10 — does M1 include the real transport? **Recommendation revised: yes, integrate.**

*An earlier draft of this section recommended landing the loop headless first and letting a
few days of real turns shape the envelope. That recommendation rested on one premise — that
nothing yet exists to shape the contract with, so fixing it now would be the speculative
abstraction DL-006 rejected. **The premise is false, and reading the tray is what showed it.**
Retracted in place rather than quietly replaced.*

### What the tray already fixes about the contract

`apps/mac-tray/docs/V1_IMPLEMENTATION.md` §"Deliberately waiting at the agent boundary" lists
six guarantees the replacement must preserve. They were written against this same ledger, so
they do not merely constrain M1 — they **converge** with it:

| Tray requirement | What in M1 satisfies it |
|---|---|
| 1. Acknowledged once, routed to the single omega queue | §1.1 — `append_episode` is the only entry; **the returned `seq` is the acknowledgement token** |
| 2. Context items keep stable identity through delivery and failure recovery | §2.1 — context ids live in the episode payload, so they survive `kill -9` |
| 3. Replies and task states never steal focus | Tray-side only; no brain consequence |
| 4. Proactive events declare normal vs genuinely time-sensitive | M5, but the payload must carry urgency — reserved now (§2.1) |
| 5. Outcomes distinguish accepted · failed · blocked · verified complete | §2.1 `turn.completed{outcome}` + `turn.blocked` |
| 6. At-most-once belongs to the core, not the UI | §1.2 — claim-before-act, exactly |

Requirement 6 is the striking one: the tray explicitly *declines* to own at-most-once and
hands it to the core. §1.2 owns it. Neither was written with the other in view.

### The result: the wire format is a projection of the episode stream

The tray's five durable work states — **understood · working · blocked · failed · verified
complete** — and M1's episode kinds are **the same alphabet**, because both were derived from
the same human test:

```
understood         ≡  inbound episode appended and claimed   (a durable fact, not a UI guess)
working            ≡  tool.called / tool.returned            (real passes of the act sub-loop)
blocked            ≡  turn.blocked                           ("is this moving?" answered no)
verified complete  ≡  turn.completed{outcome:"spoke"}        ("are we done, verified?" passed)
failed             ≡  turn.completed{outcome:"failed"}
```

So the transport is **not a second protocol to design.** It is an outward *projection* of the
episode stream the loop already writes — filtered by policy on the Python side (DL-018), since
the tray has no business seeing every internal tool call. Three things fall out:

- **"Progress describes real state" becomes structural.** The tray's own spec demands this,
  and today it cannot deliver: `TrayViewModel.swift:149-153` sets `.working("Working locally")`
  and `.complete("Demo response received")` around a single `await`, because `send() -> String`
  has nothing else to tell it. Projecting the log means the UI can only display what actually
  happened. The guarantee stops depending on discipline.
- **Initiative needs no second channel.** M5 pushes by appending an episode; the same
  projection carries it. DL-011's "both wake conditions enter at the same point" becomes
  visible in the UI rather than merely true internally.
- **DL-006 is satisfied, not bypassed.** Its fear was a *stub's accidental shape* becoming the
  contract. We are deleting the stub **because** its shape cannot carry what the tray spec and
  DL-016 both require, and the replacement is derived from the brain's actual output. That is
  precisely what DL-006 asked for.

### What this costs, honestly

`send(_:) async throws -> String` becomes a duplex, streaming, push-capable connection —
JSON-lines over a localhost socket, network-shaped from the first commit per DL-016. Swift
side: `TrayTransport` is replaced by a connection yielding an `AsyncStream` of turn events,
and `deliver(...)` stops synthesising `workState` and starts consuming it. That is real work
and more than the stub, and `TrayViewModel`'s failure-recovery path (draft and context
restoration) has to be re-proven against a stream rather than a single `await`.

**Recommendation: integrate in M1, and build the loop against the projection from step 3
rather than bolting it on at step 8.** The remaining open piece is not *whether* but *what the
projection filter admits* — a policy question, Python-side, cheap to change, and it needs real
turns to tune. That is the part worth deferring; the envelope is not.

## Q11 — is `judge` a model call at M1, or a stub?

This decides whether M1 pulls in an LLM client, prompts, and cost, or whether M1 is purely
the skeleton and the first model call lands in M2.

**Recommendation: a real model call, smallest possible prompt.** A loop whose `judge` always
returns `speak` never exercises the one thing DL-011 calls load-bearing — deciding to say
nothing — and M1 would ship with its most important property untested. A stubbed judge also
cannot start a *useful* corpus, and the corpus is the point.

## Q12 — one process or two at M1?

The resident process owns the clock, the listeners and the executor (DL-016). At M1 there is
no clock. Does the driver run in-process with the executor, or as a separate process talking
over the socket in Q10?

**Recommendation: one process, two threads** — listener and executor — with the socket
between the *client* and that process. Two processes would mean two openers of the log, and
M0 makes that a hard failure by design (the singleton rule is physical, not advisory).

---

## Build order inside M1

1. Episode payload codec + the `kind` set (§2.1). Pure Python, no store.
2. The queue as a checkpoint cursor (§1.1) — including claim-before-act (§1.2).
3. The turn skeleton with all six steps (§1.4), `judge` stubbed, `act` empty.
4. Interrupted-turn detection at startup (§1.2) and its test.
5. **The outward projection of the episode stream** (§Q10) and the duplex localhost socket.
   Placed here, not last: the tray's five work states are already the turn's own alphabet, so
   the projection is a filter over what steps 1–4 emit rather than a protocol bolted on after.
6. `judge` as a real call (§Q11, pending your answer).
7. The `act` sub-loop and its two questions (§1.5) — which is what makes *working*, *blocked*
   and *verified complete* real states rather than synthesised ones.
8. Swift side: replace `TrayTransport` with a streaming connection, delete
   `LocalDemoTransport`, and re-prove `TrayViewModel`'s draft/context restoration against a
   stream instead of a single `await`.

Steps 1–4 and 7 depend on **nothing that is open**, and touch no file M0 is currently
editing. They are what can start in parallel today. Step 5 needs only the projection-filter
policy, which is Python-side and cheap to change.

## The M1 violation metric

`CLAUDE.md` requires a capability metric to be paired with a violation metric that must not
regress. M1's capability is *turns complete and land in the log*. Its violation metric:

**No acknowledged episode is ever processed twice, and no claimed turn ever vanishes without
a record.** Concretely, after any number of `kill -9`/restart cycles:

- every inbound episode at seq ≤ `DONE` has **exactly one** terminal record naming it —
  `turn.completed` or `turn.blocked`, never both and never two;
- `claimed - done` is never greater than 1 — more than one turn in flight means the queue
  stopped being single-consumer;
- an inbound episode at seq ≤ `claimed` with no `turn.completed` appears in the startup
  interrupted-turn report, and **nowhere else** — it is never silently re-run.

Asserted against a real `kill -9`, not a mock, and **it fails closed**: a trial set in which
zero episodes were processed, or in which the kill landed before the first append, does not
pass — it reports *couldn't determine*. M0's crash suite already hit exactly that bug (2 of 20
trials killed the child during interpreter startup), and a pass count would have read 20/20.
