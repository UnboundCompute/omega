# omega agent core

Reserved for omega's agent runtime, memory, initiative, tools, and reliability machinery.
It is intentionally empty while those designs remain unresolved.

The Mac tray under `apps/mac-tray` must not place agent logic here by accident or grow its
own competing implementation. The future connection between channels and this core will be
defined from agent behavior, not inferred from a temporary UI stub.
