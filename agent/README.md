# omega agent core

Reserved for omega's agent runtime, memory, initiative, tools, and reliability machinery.
Still empty — the memory and loop designs are settled at the concept level but not yet built.

The Mac tray under `apps/mac-tray` must not place agent logic here by accident or grow its
own competing implementation.

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
