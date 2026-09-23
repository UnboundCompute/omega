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
  version   u32       = 2
  reserved  20 bytes  zero

Record frame, repeated:
  body_len  u32       byte length of body, 18..=MAX_BODY (18 = seq+ts+key_len, the real floor)
  len_crc   u32       CRC-32/ISO-HDLC over the four bytes of body_len
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

**Why `body_len` carries its own checksum** (version 2; version 1 did not, and that was a
data-loss bug found by audit and reproduced). `crc32` covers the body, so validating it requires
already knowing how long the body is. That makes `body_len` the one field nothing can vouch for,
and a single flipped bit in it that lands anywhere inside `18..=MAX_BODY` used to read as a frame
claiming more bytes than the file holds — which recovery classified as a torn tail and **silently
truncated**. Measured on a healthy three-episode log: one bit flipped in the first frame's length
field destroyed all three acknowledged episodes and cut the file from 143 bytes to 32, with no
error raised.

The circularity is the whole problem, so `len_crc` breaks it: the length field is validated
*before* it is trusted, without needing the body. A damaged length now fails its own checksum and
is classified on the evidence, instead of being believed and acted on. Version 1 is not readable
by this build and is not migrated — no log of it exists outside tests, and DL-017's
rebuild-from-log replaces migration anyway.

The payload is **opaque bytes** to Rust. What an episode *means* is the seam's business
(DL-018: Rust owns structure, Python owns meaning). The log does not parse payloads.

## Durability rules

- **`fsync` after every append**, before the append returns. Non-negotiable: the invariant is
  that an append which returned survives `kill -9`. **Measured cost on this machine is ~4ms per
  append**, not the ~0.1–1ms this spec first claimed — `File::sync_all()` on macOS issues
  `fcntl(F_FULLFSYNC)`, which is real durability and roughly 100× a plain `fsync(2)`. That is the
  right trade against a turn dominated by a 0.5–3s model call, but anything sizing a write path
  off the old figure would be out by an order of magnitude. It also caps append throughput at
  ~250/s, which is far above the ~100–300 episodes per *day* this is built for.
- **`kill -9` cannot test this.** The kernel completes an in-flight write and the page cache
  survives the process, so every crash test passes whether or not `fsync` is called at all —
  confirmed by deleting the `fsync` and watching all 115 tests stay green, 94× faster. The rule
  must therefore be held by a test that observes the *call*, not the outcome.
- **`fsync` the parent directory** once after creating the file, or the file itself can vanish
  on crash.
- **Exclusive advisory lock** (`flock`) held for the lifetime of the open log. A second opener
  fails with `AlreadyLocked`. This is the singleton rule (DL-016) made physical: two processes
  writing one log is two sources of truth.

## Recovery — run on every open

Distinguishing a **torn tail** (survivable, truncate) from **corruption** (not survivable, fail
closed) is the core of this milestone. The rule is:

> A frame that is **incomplete** at EOF is a torn tail — truncate it and continue.
> A run of **zeros** to EOF is a torn tail — truncate it and continue.
> A frame that is **complete** but invalid, with data after it, is corruption — refuse to open.

Data following a frame proves that frame was fully durable at some point, so a bad CRC there is
real corruption and truncating it would silently destroy real episodes.

**The zero-fill clause is not hypothetical and was added after it bit us.** A crash that is *not*
a process kill — real power loss, a kernel panic — can leave a filesystem having persisted the
*size extension* of a write without its data pages, so the tail reads back as zeros. An earlier
version of this spec classified a zero `body_len` as corruption; the result was that a log which
had survived exactly the crash this milestone promises to survive **refused to open, making every
acknowledged episode unreachable.** That is a direct violation of this spec's own violation
metric. A zero-filled tail is a crash artifact, not damage, and is truncated.

Procedure:

1. File missing → create, write header, `fsync` file, `fsync` dir. Empty log, `head` = 0.
2. File is 0 bytes → treat as fresh; write the header. (Creation crashed before the header.)
3. File is 1..31 bytes → **check the magic first, before touching anything.** If the bytes
   present are not a prefix of `b"OMEGALOG"` → `NotAnOmegaLog`, refuse to open, change nothing.
   Otherwise it is a torn header: no record can exist yet, so truncate to 0 and rewrite it.
   *(Order matters and did not always: truncating first meant any short file at this path — a
   note, a stray text file — was silently overwritten with a log header, because the magic check
   sat behind the destructive step and never ran for files under 32 bytes.)*
4. Bad magic → `NotAnOmegaLog`. Unknown version → `UnsupportedVersion`. Both refuse to open.
5. Scan frames from offset 32, tracking `expected_seq` starting at 1:
   - fewer than 12 bytes remain → **torn tail**, truncate here.
   - `len_crc` does not match `body_len` → the length field itself is damaged, so nothing it
     says may be acted on. Decide on the bytes' own evidence, in this order:
     - every byte from this frame's start to EOF is **zero** → **zero-filled tail**, truncate
       here. (A frame prefix we wrote is never all zeros, so this is a crash artifact.)
     - the frame is **whole apart from this one field** — some candidate length is plausible,
       the body it names fits in the file, that body matches the body CRC, and the `seq` inside
       it is the one expected → the frame was fully durable and a later bit-flip hit its length
       field or that field's checksum. Two candidates are tried, in order: the stored `body_len`,
       then the length implied by EOF. A half-landed write cannot satisfy all four for either,
       because its body is the part that did not arrive.

       **This repairs; it does not refuse.** The four facts hold *independently* of the broken
       field, so they identify the frame's true length, and a log whose damage is fully
       identified has lost nothing. Recovery rewrites the four length bytes and their checksum
       at the frame's offset and `fsync`s, before the truncation step. The repair must reach
       the disk: the EOF-implied candidate only identifies the **last** frame, so an
       in-memory-only fix would open today and refuse forever once the next append pushed that
       frame into the middle. `Log::repaired_lengths()` reports how many were rewritten on this
       open; a repaired file is byte-identical to one written healthy, so the count is the only
       trace. *(This rule formerly returned `CorruptFrame` here. Refusing is permanent, so it
       protected a provably intact episode by making every episode in the log unreachable.)*
     - otherwise the frame's extent is unknown, so every byte to EOF is **unexplained**. Scan
       that region for any frame that **verifies end to end and chains, by sequence number, all
       the way to EOF**. One found → `CorruptFrame`. More unexplained bytes than a single
       maximum frame → `CorruptFrame`, since an append writes one frame and `fsync`s before
       returning, so at most one is ever in flight. Nothing found → **torn tail**, truncate here.

       **Why chaining, and not one frame that verifies.** A 4-byte CRC match is not proof here.
       The one-in-four-billion reading of it assumes uniformly random bytes, and this region
       holds neither: payloads are opaque by design, and recycled blocks are usually older
       generations of this same log. A payload carrying frame-shaped bytes therefore made the
       log refuse, permanently, on every subsequent open. A match now only earns the candidate a
       look at whether the frames after it run unbroken to EOF, which a stray embedded frame
       cannot do.

     **Why the rule is "a frame that verifies", not "a non-zero byte".** See case 49. The rule
     above says *data following a frame proves that frame was durable*; an earlier version read
     "data" as "any non-zero byte", which counts a write's own wreckage as proof of its own
     durability. A crash during an append can persist a write's size extension and only some of
     its pages — pages are independent units of writeback and nothing makes two of them land
     together — so a frame straddling a page boundary lands **in part**, leaving real, non-zero
     bytes that were never acknowledged. Reading those as durable data made the log refuse, and
     refusal is permanent, so the log never opened again. The only honest form of "data" is a
     frame that verifies end to end. Zeros are not one; a torn prefix is not one either.
   - `body_len` is **implausible** — 0, below the 18-byte minimum body, or above `MAX_BODY` —
     while `len_crc` matches → `CorruptFrame`. A checksum-valid length we could never have
     written means the file was built by something other than this code.

     Note the 18-byte minimum is the real floor: `seq` + `ts_micros` + `key_len` alone is 18
     bytes, so a body below that cannot be one we wrote.
   - fewer than `body_len` bytes remain, `len_crc` valid → **torn tail**, truncate to the frame
     start. This is now safe to act on, and previously was not: the length is checksum-verified
     before it is believed, so a frame that claims to run past EOF really is a write that was
     interrupted, not a corrupted number pointing past the end of intact data.
   - CRC mismatch → **torn tail** if the frame ends exactly at EOF, else `CorruptFrame`.
   - `seq != expected_seq` → **torn tail** if the frame ends exactly at EOF, else `SequenceBreak`.
   - otherwise accept; record its offset; `expected_seq += 1`.
6. If anything was truncated, `set_len` to the last good offset and `fsync`.
7. Build the in-memory **offset index** (`seq → file offset`) and **dedup index**
   (`key → seq`) from the scan. When two records share a dedup key, the **lowest** seq wins:
   the first write of a key is the one it refers to.
8. **Load checkpoints, then check them against `head`.** A checkpoint ahead of `head` cannot
   legitimately happen and is *proof* that acknowledged episodes were lost — it is the one
   moment that evidence exists, so it is consulted here rather than lazily on the next read.
   Raise `CheckpointAhead` at open. Checking it only on the next `episodes_since` leaves the log
   opening "cleanly" and the consumer wedged: unable to read, and unable to reset its own
   position, because both paths raise.

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

**A damaged sidecar must never stop the log from opening.** It loaded during `open` behind a `?`,
so any problem with it — zero bytes, all zeros, a truncation, a flipped bit, bad magic — made
every acknowledged episode unreachable. That is the violation metric, reached through state that
is not the source of truth. Worse, the two most likely artifacts are the same ones the zero-fill
clause exists for: a `rename` visible before its directory `fsync`, or a size extension persisted
without its data pages, yields a zero-byte or all-zeros sidecar. The log's own recovery calls that
survivable; the sidecar called it fatal.

So: a sidecar that cannot be read is **renamed aside** to `<name>.damaged` and every checkpoint
reads 0. The log opens. This is deliberately the same outcome as a *missing* sidecar, which was
already the accepted, tested behaviour — so it adds no new failure mode, only a second road to a
place we already went. The evidence is preserved rather than deleted, and the fact that it
happened is reported through the seam, so it is loud without being fatal.

The one thing that still refuses to open is a **well-formed** sidecar whose position is ahead of
`head` (recovery step 8). That is not damage to derived state; it is proof that acknowledged
episodes were lost, and it is the only moment that proof exists. The documented way out is to
remove the sidecar by hand: a missing one reads as 0, consumers replay, and DL-007's write-key
dedup makes replay idempotent by construction.

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
27. Torn tail of 1, 7 and `PREFIX_LEN - 1` bytes, of **non-zero** bytes — every length
    shorter than the frame prefix, up to the last one. The prefix is **12** bytes in version 2
    (`body_len`, `len_crc`, `crc32`), so a parametrization that stops at 7 stops at version 1's
    boundary and never reaches the byte before a length becomes readable.
    The all-NUL versions of the same lengths are case 39; they take a different branch, so the
    two must be spelled out separately rather than left to the reader.
28. Torn header: file of 0 bytes, and of 1..31 bytes → recovers as an empty log, no error.
29. **After a torn-tail recovery, the next append continues the sequence with no gap.**
30. Recover, crash, recover again → identical state. Recovery is idempotent.
31. `kill -9` immediately after append returns → the episode is present after reopening.
32. `kill -9` *during* an append → the episode is either fully present or fully absent, never
    partial, and the log always reopens cleanly.
    **Read this case honestly, and do not let it be mistaken for evidence it does not provide.**
    Killing a process does **not** tear a write: the kernel completes an in-flight `pwrite` even
    though the process is gone. This case therefore proves *acknowledged appends survive* and
    proves **nothing** about torn-tail recovery. A green result here is not coverage of cases
    25–28. Those paths are only reachable by mutating the file deliberately, which is what they
    do. A test that passes without exercising what it claims is an eval bug.
33. Cases 31 and 32 run **20 times each**. Passing once is not passing.
34. After any recovery, the rebuilt offset index matches a fresh full scan.
35. Both indexes are caches: deleting them in memory and rescanning yields identical results.
36. Concurrent readers in one process see a consistent view while appends happen.
37. **Zero-filled tail** — append N episodes, append a run of NUL bytes (what real power loss
    leaves), reopen: the log opens, `head()` is N, all N episodes are readable, and the zeros are
    truncated. This case exists because the spec originally got it wrong and the log refused to
    open, losing everything.
38. Zero-filled tail *inside* an otherwise valid frame region, i.e. zeros followed by a **whole
    frame that verifies** → `CorruptFrame`. Zeros only mean "crash artifact" when nothing durable
    follows them. (Zeros followed by bytes that verify as no frame are covered by case 49: the
    log opens and the tail is preserved.)
39. A single NUL byte appended, and a run shorter than the 12-byte frame prefix → both recover
    as a torn tail with no loss.

### Added after the adversarial audit — each one is a bug that shipped

Every case below corresponds to a defect that existed and that the first 39 cases missed. They
are listed with what they caught, because a case whose origin is forgotten is a case someone
later deletes as redundant.

40. **A corrupted `body_len` never silently truncates.** Flip one bit in a frame's length field
    so it stays inside `18..=MAX_BODY` but claims more bytes than the file holds → `CorruptFrame`,
    the file is byte-identical afterwards, and every episode is still readable. Test the first
    frame, a middle frame, and a value stretched to end exactly at EOF. *This is the audit's
    finding 1: previously all three opened "cleanly" with `head()` reset and the file truncated
    on disk — three acknowledged episodes destroyed with no error.* The old suite missed it
    because both of its length-mutation tests used *implausible* values, which take a different
    branch; the plausible-but-too-large case was never constructed.
41. **`len_crc` is checked before `body_len` is used.** A length field that fails its own
    checksum, with the body still intact → the frame is **repaired**: the log opens, `head()` is
    unchanged, every episode reads back, `recovered_bytes` is 0, `repaired_lengths` is 1, and the
    file is byte-identical to one written healthy — so a second open repairs nothing. Test the
    first frame, a middle frame, and the last frame at EOF. With zeros to EOF → recovers as a
    zero-filled tail. *Revised: this case previously asserted `CorruptFrame`. Four facts that
    hold independently of the broken field identify the true length, and refusing a log whose
    damage is fully identified loses every episode in it to protect one that was never at risk.*
42. **A damaged sidecar never blocks the log.** For each of: zero bytes, all zeros, truncated by
    one byte, one flipped bit, bad magic, absurd count, absurd `name_len` — the log **opens**,
    every episode is readable, checkpoints read 0, the damaged file is preserved as
    `<name>.damaged`, and the seam reports that it happened. *Audit finding 2: all seven refused
    to open, over state that is not the source of truth.*
43. **A checkpoint ahead of `head` fails at open**, not lazily on the next read. *Audit finding
    3: the log opened "cleanly" and the consumer was then wedged — unable to read and unable to
    reset its position, since both raised.* Removing the sidecar by hand must recover it.
44. **`fsync` is observed, not assumed.** A test must fail if the `fsync` in `append` is deleted.
    *Audit finding 4: the entire suite — all 115 tests including 40 `kill -9` trials — passed
    with durability removed, 94× faster. Every durability call in the crate could be deleted and
    nothing noticed.* `kill -9` cannot catch this, so the test observes the call itself. The
    same must hold for the directory `fsync` and both sidecar `fsync`s.
45. **A small non-log file at the log path is refused, not overwritten.** A 26-byte text file
    → `NotAnOmegaLog` and the file is byte-identical afterwards. *Audit finding 5: it was
    silently replaced with a log header, because the magic check sat behind the truncation step
    and never ran for files under 32 bytes.*
46. **The seam's path rule is order-independent.** `MemoryStore.open(p)` puts the log in the same
    place whether or not `p` already exists. *Audit finding 6: the same argument produced either
    `p` as a file or `p/episodes.log` depending on what was on disk first, and the file form then
    permanently blocked the directory form.*
47. **There is no frame-boundary discontinuity.** Non-zero garbage at a frame start, in runs from
    1 byte up to and past the prefix length, is classified the *same way* throughout: it holds no
    frame that verifies, so it was never acknowledged, so it is truncated and preserved. *Audit
    finding 7 first caught that the old suite tested 1, 2 and 7 bytes but never 8, so the point
    where behaviour changed was untested. Case 49 then removed the change itself — the step at
    `PREFIX_LEN` was the bug, because that is exactly where a torn prefix falls. This case now
    sweeps the same lengths asserting the opposite: that no step exists anywhere.*
48. **First key wins in the dedup index.** Two records sharing a dedup key resolve to the lower
    seq, and a rebuild agrees. *Audit: unpinned — reversing it broke no test.*
49. **A half-landed append never makes the log unopenable.** Build a log whose final frame
    straddles a 4096-byte page boundary. For **both** writeback orders — only the earlier page
    lands, only the later page lands — and for every split point across the frame, the log
    reopens and every acknowledged episode is readable. Frames spanning three or more pages are
    covered with an arbitrary subset of pages landed.

    *This is the second time one clause produced the failure this spec's violation metric names,
    and it is the most expensive defect found in M0.* A crash during an append can persist the
    write's size extension while only some of its pages are written back. The recovery rule asked
    "are the bytes from here to EOF all zero", the partially landed bytes are not zero, so the
    tail was classified as corruption and the log **refused to open permanently, with every
    acknowledged episode intact on disk and unreachable.** Measured before the fix on a healthy
    27-episode log: landing only the earlier page destroyed 7 of 154 split points — every one a
    tear falling inside the 12-byte prefix, where the length checksum lives; landing only the
    later page, which no rule forbids, destroyed **152 of 154**.

    *Why the first 48 cases all missed it, which is the part worth remembering:* both controls
    pass. A wholly lost frame is the zero-fill case and recovers; a plain truncation recovers.
    Only their **combination** fails, and every zero-fill test in the suite appends onto a clean
    frame boundary, so the torn prefix each one constructs is entirely zeros — the one shape that
    works. `kill -9` cannot reach it either (case 32 says so explicitly): it is a page-writeback
    artifact, not a process-death artifact. A test that only ever tears at offset 0 of a frame is
    not testing tearing.

    *Extended after the third audit.* A frame whose prefix page never landed while a later page
    did is the same shape from the other side: the length field is gone, the body is on disk. The
    log must **open and repair** it — not refuse — and it must not file a `.discarded-tail`,
    because nothing is being discarded. Measured before the fix: 5 of 5 constructed variants
    refused permanently, and the reverse writeback order alone produced 8 violations of this
    case's own rule.

50. **Truncation never destroys the bytes it discards.** A truncated tail that is not all zeros
    is copied to `<log>.discarded-tail` beside the log before `set_len`, uniquified so a second
    recovery cannot clobber the first, and surfaced as `diagnostics.discarded_tail_path`. A
    failure to write it never stops the log opening — filing the evidence must not recreate the
    failure the evidence is about. Zeros are skipped, having nothing in them to read.

    *Case 49 turned several refusals into truncations. Refusing had one real virtue: it surfaced
    an anomaly instead of papering over it. This keeps that virtue without the cost, because the
    judgement being made is about what the bytes* are *— no frame verifies here — and never about
    what put them there. The bytes themselves remain, for whoever wants to look.*

### What the checksums are for, and what they are not for

M0's threat model is **crashes and bad disks, not tampering**, and that is a decision rather than
an oversight, so it is written down here next to the rules it governs.

`len_crc` closes the accidental-corruption hole. It does not close the adversarial one. A
**forged** `body_len` — one whose `len_crc` has been recomputed to match, sized so the frame ends
exactly at EOF — passes the length check, passes plausibility, has enough bytes behind it, and
fails only the body CRC. Recovery then classifies it as a torn tail and truncates, which is the
right call under this threat model: a matching `len_crc` is strong evidence the length is real,
and something shaped exactly like a half-written final frame should be treated as one. But it
means anyone who can write the file can still make the tail disappear.

That is accepted. CRC-32 is a bit-rot defence, not a MAC; making it a tamper defence needs a keyed
MAC and somewhere to keep the key, and M0 provides neither. The log is a local file under the
user's own account, and an attacker who can write it can also delete it — so integrity checking is
not the control that would save us. **Revisit if the log ever syncs, leaves the machine, or is
shared between users**, each of which changes who can write it.

The same reasoning covers the rest of the format: `write_key` dedup and `seq` continuity are
**consistency** mechanisms, not **authenticity** ones. Nothing in M0 may be cited as evidence that
an episode is *genuine* — only that it is *intact*.

### The violation metric

Capability here is "episodes go in and come back." The paired violation that must never regress
(`CLAUDE.md`: every capability metric gets one) is: **after any crash, at any point, the log
reopens and every episode that was acknowledged is present.** Not "usually." Any single failure
of that across the repeated runs fails M0.
