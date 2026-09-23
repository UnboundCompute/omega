# M0 — the log, the seam, and the durability invariant

The build contract for M0 (`AGENT.md` DL-016, DL-017, DL-018, DL-019, DL-020). Everything here
is decided; nothing in it is a suggestion. If something needed is *not* here, it is an open
decision and goes to the ledger before it gets code — it does not get invented in an
implementation.

## What M0 is

A durable, append-only episodic log with a hard seam over it, and a test suite that proves the
half of the restart test that is assertable without a loop.

**Not in M0:** the graph, entity resolution, retrieval, the loop, the queue, any channel, any
projection store. M0 ends at "episodes go in durably and come back out in order."

## Layout

```
Cargo.toml
pyproject.toml              maturin backend, python-source = "python"
src/lib.rs                  PyO3 module, exported as omega._log
src/frame.rs                encode / decode / validate one frame
src/log.rs                  open, recover, append, scan, offset index, dedup index
src/checkpoint.rs           named consumer checkpoints, atomic replace
python/omega/__init__.py
python/omega/memory/__init__.py    THE SEAM — the only module that may import omega._log
tests/                      pytest suite
```

## File format

All integers little-endian. One file, `episodes.log`.

```
Header, 32 bytes, written once at creation:
  magic     8 bytes   b"OMEGALOG"
  version   u32       = 1
  reserved  20 bytes  zero

Record frame, repeated:
  body_len  u32       byte length of body, 1..=MAX_BODY
  crc32     u32       CRC-32/ISO-HDLC over the body bytes
  body:
    seq        u64    starts at 1, strictly +1 per record, no gaps ever
    ts_micros  i64    unix microseconds UTC
    key_len    u16    0 means "no dedup key", else 1..=MAX_KEY
    key        key_len bytes, UTF-8
    payload    body_len - (8+8+2+key_len) bytes, OPAQUE
```

`MAX_BODY` = 64 MiB. `MAX_KEY` = 512 bytes. `seq` 0 is reserved and means "nothing consumed" —
it is never a record's sequence number.

The payload is **opaque bytes** to Rust. What an episode *means* is the seam's business
(DL-018: Rust owns structure, Python owns meaning). The log does not parse payloads.

## Durability rules

- **`fsync` after every append**, before the append returns. Non-negotiable: the invariant is
  that an append which returned survives `kill -9`. Cost is ~0.1–1ms against a turn dominated by
  a 0.5–3s model call.
- **`fsync` the parent directory** once after creating the file, or the file itself can vanish
  on crash.
- **Exclusive advisory lock** (`flock`) held for the lifetime of the open log. A second opener
  fails with `AlreadyLocked`. This is the singleton rule (DL-016) made physical: two processes
  writing one log is two sources of truth.

## Recovery — run on every open

Distinguishing a **torn tail** (survivable, truncate) from **corruption** (not survivable, fail
closed) is the core of this milestone. The rule is:

> A frame that is **incomplete** at EOF is a torn tail — truncate it and continue.
> A frame that is **complete** but invalid, with data after it, is corruption — refuse to open.

Data following a frame proves that frame was fully durable at some point, so a bad CRC there is
real corruption and truncating it would silently destroy real episodes.

Procedure:

1. File missing → create, write header, `fsync` file, `fsync` dir. Empty log, `head` = 0.
2. File is 0 bytes → treat as fresh; write the header. (Creation crashed before the header.)
3. File is 1..31 bytes → torn header. No record can exist yet, so truncate to 0 and rewrite the
   header.
4. Bad magic → `NotAnOmegaLog`. Unknown version → `UnsupportedVersion`. Both refuse to open.
5. Scan frames from offset 32, tracking `expected_seq` starting at 1:
   - fewer than 8 bytes remain → **torn tail**, truncate here.
   - `body_len` is 0 or > `MAX_BODY` → `CorruptFrame`. (We wrote that field; it cannot be
     legitimately out of range.)
   - fewer than `body_len` bytes remain → **torn tail**, truncate to the frame start.
   - CRC mismatch → **torn tail** if the frame ends exactly at EOF, else `CorruptFrame`.
   - `seq != expected_seq` → **torn tail** if the frame ends exactly at EOF, else `SequenceBreak`.
   - otherwise accept; record its offset; `expected_seq += 1`.
6. If anything was truncated, `set_len` to the last good offset and `fsync`.
7. Build the in-memory **offset index** (`seq → file offset`) and **dedup index**
   (`key → seq`) from the scan.

Recovery is **idempotent**: recovering an already-recovered file changes nothing. Both indexes
are caches derived from the file, never authoritative (DL-016).

## Append

```
append(payload, key, ts_micros) -> seq
```

- `key` empty → always writes a new record. Empty keys are **not** deduplicated against each
  other; "no key" is not a key.
- `key` non-empty and already present:
  - stored payload **identical** → return the existing seq, write nothing. This is DL-007's
    dedup guard: re-extraction must not duplicate.
  - stored payload **different** → `WriteKeyConflict`, write nothing. Silently returning the old
    seq would hide a caller bug, and "fail closed" beats a quiet wrong answer.
- Any rejected append leaves the file **byte-identical** and still openable.
- `ts_micros` defaults to now. It is metadata; ordering is by `seq`, never by timestamp.

## Read

```
head() -> seq of the last record, or 0 if empty
episodes_since(seq) -> iterator over records with sequence > seq
```

`episodes_since(n)` where `n > head()` → `CheckpointAhead`. It means a consumer is ahead of the
log, which cannot legitimately happen and must not be reported as "no new episodes." A check
that passes on empty is not a check.

## Checkpoints

Named consumer positions, so every later derived view (what's-open at M2, the graph at M3, the
user-model at M4) can resume. Stored beside the log, written by **write-temp, fsync, rename,
fsync-dir** so a crash never leaves a half-written checkpoint.

```
checkpoint(name) -> seq       unset reads as 0, never null
set_checkpoint(name, seq)     seq > head() is an error
```

Checkpoints are **not** episodes. They are mutable derived state and must never enter the log.

## The seam

`python/omega/memory/__init__.py` is the only module in the tree permitted to import
`omega._log`. This is DL-018's hard rule, and it is enforced by a **test**, not by discipline —
in the reference system a facade with five bypassing callers turned one refactor into 1,148
compatibility re-exports across 332 files.

The interface speaks **episodes**, never frames, offsets or rows:

```python
store = MemoryStore.open(path)
seq   = store.append_episode(payload: bytes, write_key: str = "", ts_micros: int | None = None)
store.head() -> int
store.episodes_since(seq: int) -> Iterator[Episode]    # Episode(seq, ts_micros, write_key, payload)
store.checkpoint(name: str) -> int
store.set_checkpoint(name: str, seq: int) -> None
store.close()                                          # also usable as a context manager
```

## What "done" means here

Per `CLAUDE.md`, done is a **verified state change**, never a marker, and a green suite is not by
itself evidence. M0 is done when the cases below pass, the crash case passes **repeatedly**
(reliability is pass^k, not pass@1), and the seam test fails when deliberately violated.

**The restart test is half-assertable at M0.** There is no loop, so "lose nothing but the
in-flight turn" waits for M1. M0 proves the durability half: *an append that returned survives
`kill -9`, and a torn tail truncates cleanly instead of corrupting read-back.*

### Green — it does what it should

1. Append one episode, read it back byte-identical, including the timestamp and key.
2. Append N; sequences are exactly 1..N with no gaps.
3. `episodes_since(0)` yields all; `episodes_since(k)` yields exactly k+1..N.
4. `head()` equals N; `head()` on a fresh log is 0.
5. Close and reopen: same episodes, same order, same `head()`.
6. Same write-key twice with identical payload → one record, same seq returned twice.
7. Two different write-keys → two records, different seqs.
8. Two appends with an **empty** key → two records. Empty is not a key.
9. Checkpoint round-trips; an unset checkpoint reads 0.
10. Payload edge values round-trip: empty payload, embedded NUL bytes, invalid UTF-8, 10 MiB.
11. 10,000 appends: all readable, contiguous, correct order.
12. The dedup index survives restart — the same write-key after reopening still deduplicates.

### Red — it must fail, loudly and without damage

13. CRC corrupted in a **middle** frame → open fails `CorruptFrame`. The file is **not**
    truncated and later episodes are not silently discarded.
14. Bad magic → `NotAnOmegaLog`. Unknown version → `UnsupportedVersion`.
15. `body_len` > `MAX_BODY` → `CorruptFrame`.
16. A hand-broken sequence in a middle frame → `SequenceBreak`.
17. Payload over `MAX_BODY` → append fails, file byte-identical, log still openable.
18. Key over `MAX_KEY` → append fails, file byte-identical.
19. Same write-key with a **different** payload → `WriteKeyConflict`, nothing written.
20. `episodes_since(head() + 1)` → `CheckpointAhead`, **not** an empty iterator.
21. `set_checkpoint(name, head() + 1)` → error.
22. Second process opening the same log → `AlreadyLocked`.
23. Unwritable path → a clear error, not a silent no-op.
24. **Seam violation is a test failure:** any module outside `omega.memory` importing
    `omega._log` fails the suite. The test must be shown to fail when a violation is introduced,
    or it is not a check.

### Yellow — the edges, where this actually gets decided

25. Torn tail mid-payload → recovers, truncates, `head()` is the last good seq.
26. Torn tail at an exact frame boundary (length written, body not).
27. Torn tail of 1 byte, and of 7 bytes (shorter than the frame header).
28. Torn header: file of 0 bytes, and of 1..31 bytes → recovers as an empty log, no error.
29. **After a torn-tail recovery, the next append continues the sequence with no gap.**
30. Recover, crash, recover again → identical state. Recovery is idempotent.
31. `kill -9` immediately after append returns → the episode is present after reopening.
32. `kill -9` *during* an append → the episode is either fully present or fully absent, never
    partial, and the log always reopens cleanly. Torn tails are expected here; corruption is not.
33. Cases 31 and 32 run **20 times each**. Passing once is not passing.
34. After any recovery, the rebuilt offset index matches a fresh full scan.
35. Both indexes are caches: deleting them in memory and rescanning yields identical results.
36. Concurrent readers in one process see a consistent view while appends happen.

### The violation metric

Capability here is "episodes go in and come back." The paired violation that must never regress
(`CLAUDE.md`: every capability metric gets one) is: **after any crash, at any point, the log
reopens and every episode that was acknowledged is present.** Not "usually." Any single failure
of that across the repeated runs fails M0.
