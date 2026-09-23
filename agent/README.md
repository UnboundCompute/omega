# omega agent core

Reserved for omega's agent runtime, memory, initiative, tools, and reliability machinery.
Still empty — design is complete and **M0 is the next thing built**.

The Mac tray under `apps/mac-tray` must not place agent logic here by accident or grow its
own competing implementation.

## Build order

Not layer-cake. Two properties of the design set the order: the append-only log is the only
component whose value is **time-dependent** (a day not logging is evidence lost for good), and
three of the five features — recall quality, knowing-you, initiative — **cannot be evaluated
without a corpus**. So: reach honest daily use early, with deliberately dumb recall, and spend
the accumulated history on the hard features afterwards.

| | Milestone | Done-bar |
|---|---|---|
| **M0** | Store, log, the memory seam, restart test | `kill -9` mid-turn → lose only the in-flight turn, green in CI |
| **M1** | **The core loop** — queue, executor, one full turn | **Daily use starts**; every exchange lands in the log |
| **M2** | Identity, continuity, opinions | Reopen after 3 days and it resumes cold; voice holds under pressure |
| **M3** | The derived graph + retrieval policy | Drop the graph, re-derive, identical projection |
| **M4** | Knowing you | States something true about how you work that you were never told |
| **M5** | Initiative | ≥1 unprompted item worth seeing; most ticks produce nothing |
| **M6** | Reach: second surface + one integration | A real external task, verified end to end |

Rust lands at **M0**, on the log — the simplest component to carry it, and it makes the seam a
real cross-language boundary from the first commit. A seam that isn't crossed isn't tested.

## The loop (M1)

There is exactly **one** loop and everything goes through it. A single-consumer queue, one
executor draining it serially, one turn:

```
perceive → recall → judge → act → reply → write memory
```

- **`judge` is a real step.** Deciding to say nothing is a *successful* turn and must be logged
  as one. If silence reads as failure anywhere — in the loop, in logs, in any later metric —
  initiative degenerates into a notification firehose.
- **The repetition lives inside `act`.** Tool → result → tool → result, re-asking each pass:
  *are we done, verified?* and *is this moving?* A stall means stop, re-plan or come back and
  ask — never flail silently. This is a sub-**loop**; a sub-agent would be a second context with
  its own judgement, which fractures identity and puts its work outside the episodic log.
- **Every wake enters at the same point.** M1 only has the you-wake, but the queue is already the
  single entry, so the clock's time-wake later becomes just another producer rather than a second
  path. Two engines would drift into two omegas.

**M1 is where `apps/mac-tray` converges.** It is the point at which the real transport replaces
`LocalDemoTransport`, under the three constraints below. Nothing before M1 should depend on the
stub's shape, and the stub is deleted — not adapted — when M1 lands.

Rationale, alternatives rejected, and the named risk in this sequence: `AGENT.md` (DL-019).

## Memory (decided)

Memory is a **graph** — episodes are nodes linked to the entities they mention, facts hang off
entities, and supersede pointers are edges. It is not a row store with entity columns.

The graph is **derived from an append-only episodic log**, which stays the only write path.
Nothing writes to the graph directly, including entity resolution: a resolution decision is
itself logged as an event, so re-deriving the graph from the log is deterministic.

That gives the migration strategy, and it is the unusual part worth stating plainly: **there are
no data migrations.** A schema change means dropping the graph and rebuilding it from the log.
Rebuild-from-log therefore has to work from the first version, not from the first time it is
needed.

**The language split** follows the churn rate, not the call graph. Rust holds the structural half
— nodes, edges, the log, entity keys, supersede pointers, traversal, indexes — behind a PyO3
boundary, in the same process. Python holds the loop and the policy half of retrieval: what to
pull for a query, how to fuse signals, how to rank. Retrieval quality is empirical and must stay
cheap to iterate on, so it must not sit behind a recompile. The line: *what exists* is Rust,
*what comes back* is Python.

## The store seam (hard rule)

Nothing outside the memory engine touches the store directly. No module opens the database,
writes a query, or reaches for a table. Everything goes through the named operations — append an
episode, resolve an entity, pull for a query, supersede a fact, excise a subtree.

The interface also speaks **domain language, not graph language**: it returns episodes, facts and
entities, never nodes, edges or cursors. If graph vocabulary leaks across it, the seam is
decorative and the engine underneath stops being replaceable.

Occasionally this means writing an interface method for a query that could have been inlined.
That friction is the mechanism, not a side effect.

## Process shape (decided)

omega runs as **one resident process** that is authoritative for **nothing**. It owns the
clock, the channel listeners, and the turn executor; every fact it acts on lives in the store.

The invariant is testable and every future change has to keep it: **`kill -9` the process at
any moment, restart it, and nothing is lost but the turn that was in flight.** If some state
only exists in the running process, it is in the wrong place.

What follows from that:

- **One loop, literally.** All wakes — a message on any channel, or the clock — enqueue onto a
  single queue drained by one executor. There is no second engine and no per-channel path.
- **Long work detaches.** A long-running job runs as a context-isolated sub-loop with no memory
  and no identity; its result is enqueued like any other wake, so the main loop stays responsive.
- **Idle time is for preparing, not speaking.** Most clock ticks should produce nothing the user
  sees, and anything prepared is queued in the store rather than held in the process.

For v1 the process runs locally on the Mac. It is expected to move to an always-on host later,
so nothing may assume the brain is on the same machine as a channel client.

## The channel boundary

Still undefined, and deliberately so — it will be derived from agent behavior, never inferred
from a temporary UI stub. `LocalDemoTransport` in the tray is a throwaway and is marked as one.

Three constraints on it are already fixed, because decisions elsewhere force them:

1. **Bidirectional.** omega initiates. Proactive messages arrive at a surface without the user
   having asked for anything, so a request/response transport cannot carry the product's core
   behavior. The current stub's `send(...) -> String` shape is request/response and is therefore
   not a candidate contract.
2. **Structured, not a flat string.** Rendering must not silently drop what the model produced;
   fixed slots that lose content are a bug.
3. **Network-shaped.** Localhost today, a real address later. Relocating the brain must be a
   config change, not a redesign of this boundary.

A channel is transport, never scope: the same omega, one memory and one identity, on every
surface. Delivery routes back to whichever channel asked, and at-most-once holds *across*
channels rather than per channel.
