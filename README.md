# omega

omega is a personal second brain: someone to delegate to and think alongside, with
identity, continuity, initiative, opinions, and a growing understanding of how its user
works.

The repository keeps user-facing surfaces separate from the agent core:

```text
omega/
├── apps/
│   └── mac-tray/   # Native macOS desk surface
├── agent/          # Agent runtime and memory (reserved; not implemented yet)
├── AGENT.md        # Private, gitignored design north star
└── CLAUDE.md       # Project working rules
```

The first implementation target is the Mac tray. Its confirmed v1 product and interface
contract is in [`apps/mac-tray/docs/V1_SPEC.md`](apps/mac-tray/docs/V1_SPEC.md).

## Boundary rule

`apps/mac-tray` is a channel, not the agent. It may render conversation, collect context,
and display durable task state, but it must not own memory, personality, initiative
judgement, or agent orchestration. Those capabilities belong under `agent/` when their
design is ready.

During the tray prototype, any local response mechanism is explicitly a throwaway seam,
not the future agent protocol.
