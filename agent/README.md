# omega agent core

Reserved for omega's agent runtime, memory, initiative, tools, and reliability machinery.
Still empty — the memory and loop designs are settled at the concept level but not yet built.

The Mac tray under `apps/mac-tray` must not place agent logic here by accident or grow its
own competing implementation.

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
