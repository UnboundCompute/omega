# omega — project instructions

## What this is

**omega** is a personal-assistant **"second brain"** — a from-scratch agent harness
meant to feel less like a tool and more like a *person* you delegate to and think
alongside. It remembers, it takes initiative, it has continuity.

It is **not** a fork of Hermes. Hermes is a reference we borrow specific, proven
pieces from *when a real need arises* — never a base we build on top of.

## Status: design-first (no code yet)

We are defining the design in docs **before** writing any code. Do **not** scaffold
code, choose frameworks, or lock in architecture until the design is finalized
together.

- The evolving design lives in **`AGENT.md`** (gitignored, private). It is the
  north-star working doc. Read it before reasoning about the architecture.
- Order of work: **talk → finalize docs → then code.** Code comes last.

## Guiding principle: stay small and legible

The reason we left Hermes is that it grew **too big and bloated**, which made adding
and customizing things hard. Do not recreate that here. Prefer the smallest thing
that works; add structure only when a concrete need forces it. Every layer must earn
its place.

## Stack

- **Python** and **Rust.** The exact boundary (what is Rust vs Python) is still being
  decided — see `AGENT.md`.

## Working style

- Discuss design decisions before implementing them.
- Match the surrounding code's idiom, naming, and comment density.
- **Commits only on explicit request.** Do not add attribution/footer trailers to
  commit messages or PR bodies.
