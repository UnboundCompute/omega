# Tray integration — M1 step 8

The handoff brief for the Swift side. Everything the tray talks to already exists and is
tested; nothing in `python/` needs to change for this work.

**The job:** delete `LocalDemoTransport`, replace the `TrayTransport` protocol with a
streaming connection to omega's localhost channel, and re-prove `TrayViewModel` against a
stream instead of a single `await`.

> **Status, 2026-09-24: the tray is running against live omega**, so §§1–4 are now a
> description of something that works rather than a brief for something to build. What
> changed underneath since this was written is **learning** — omega now files what it is
> taught, turns a taught time into a schedule, and infers from conversation on its own. That
> added six episode kinds, **none of which reach the wire**, which makes the stream properly
> gappy for the first time. Image understanding has landed too: bounded image attachments on
> the new event now reach the capable model as multimodal input. §4's ordering rule and the new
> §5 are the two parts to re-read.

**The boundary rule, which this work must not erode.** The tray is a *channel*, not the
agent. It may render conversation, collect context, and display durable work state. It must
not own memory, personality, initiative judgement, or agent orchestration. If a change starts
to look like the tray deciding something, it belongs in Python instead.

---

## 1. Why the protocol has to change, not just its implementation

```swift
protocol TrayTransport {
    func send(_ submission: TraySubmission) async throws -> String
}
```

One `String`, once, per submission. Three separate things make that shape wrong:

1. **A turn emits a stream, not a value.** `understood → working… → complete` are distinct
   events arriving over time, and the tray already has `WorkState` cases for all of them. The
   current signature can only report the last one.
2. **Silence returns no string at all.** A turn can legitimately end with omega choosing not
   to speak. That is a *success*. `-> String` has nowhere to put it, and `-> String?` would
   merge it with the empty reply.
3. **Updates outlive the request.** The connection is duplex: omega pushes updates for turns
   this client never asked about (later: the clock waking up on its own). A request/response
   function cannot receive those.

So the protocol becomes a long-lived connection that yields updates, plus a `send` that only
acknowledges receipt.

---

## 2. The wire, exactly

TCP on **127.0.0.1:7717** by default. **JSON Lines**: one UTF-8 JSON object per line,
`\n`-terminated, both directions on the one connection. A line the tray sends must be under
**1 MiB** or the server answers with an error and hangs up.

Both sides announce a version as `v`, and **the two `v`s are different numbers that happen to
both be 1 today**: responses carry the channel protocol version, `update` lines carry the
projection version. They are versioned separately on purpose — the storage shape is omega's,
the wire shape is shared with a client omega does not deploy. Do not assume one implies the
other.

### Tray → omega

```jsonc
{"op":"say","text":"…","id":"<uuid>","channel":"tray","context":[…],"urgency":"normal","resumes_seq":41}
{"op":"attach","path":"/abs/path/to/file.png"}
{"op":"subscribe","since":0}
{"op":"ping"}
```

`say` needs **`text`, `context`, or both** — a screenshot dropped in with no words is a real
message, and the tray's `canSend` already offers it. Only a request with neither is refused.
Fields:

| field | meaning |
|---|---|
| `id` | becomes the log's **write key**. Send a UUID. See §4 on retries. |
| `channel` | defaults to `"tray"`; send it anyway. |
| `context` | array of `{"id","kind","title"}` — **all three required**; plus `{"blob","mime","bytes"}` if the item has bytes behind it, all three or none. See below. |
| `urgency` | `"normal"` or `"timely"`. |
| `resumes_seq` | the `for_seq` of a `blocked` update this message answers. |

#### Attachments: upload first, then reference the digest

A context item carries identity. If it also has bytes behind it — a screenshot, a dropped
file — those bytes go into omega **before** the message that mentions them, through a
separate op, and the item then carries the *digest* of what was stored.

**Upload.** One file per call, with an absolute path:

```jsonc
→ {"op":"attach","path":"/abs/path/to/file.png"}
← {"v":1,"op":"attached","blob":"sha256:9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08","mime":"image/png","bytes":184320}
```

Omega copies the file into a content-addressed store beside the log and answers with the
reference. `blob` is `sha256:` followed by exactly 64 lowercase hex characters — the prefix
is part of the value, on the wire and in the log. `mime` is derived from the filename's
extension, falling back to `application/octet-stream`. `bytes` is the size omega actually
read and hashed, not what the filesystem claimed beforehand.

Attaching the same content twice — the same file, or two files with identical contents — is
stored once and answers the same digest both times. That is not a feature with a switch; it
is what content addressing does. It also means `attach` is safe to retry: a second call for
the same file costs a re-read and changes nothing.

**Reference.** The `say` that follows puts the three fields on the context item alongside
the identity fields:

```jsonc
{"op":"say","id":"<uuid>","channel":"tray","context":[
  {"id":"ctx-1","kind":"image","title":"Area capture",
   "blob":"sha256:9f86d081…","mime":"image/png","bytes":184320}
]}
```

`blob`, `mime` and `bytes` are **all three or none**. A partial set is refused with a named
error, not accepted with the rest inferred — a digest with no size and a size with no digest
are both half a fact, and the log keeps what it is given forever. Items with no bytes behind
them (selected text, a link) simply omit all three and are as ordinary as before.

**Why two ops rather than one.** The tray captures long before the person hits send: the
upload belongs at capture time, where its cost and its failures are visible and the person
can still drop the item. Folding the bytes into `say` would move both onto the send path.
And a bare path would not survive — the file can move, change or vanish between the message
and the moment memory is re-derived from it, whereas a digest names content that cannot
change under it. `attach` failing is an ordinary error envelope on the same connection; the
connection lives and the draft is untouched.

**Update the tray's staged-context model to carry the reference.** `StagedContext` needs the
digest returned by `attach` so `say` can cite it; the response echoed back on an `update`
still contains only `{id, kind}` (§6), which is unchanged and deliberate — the tray already
has the previews, and a digest is not something the UI needs back.

> ### Current understanding boundary
>
> Image understanding now exists (DL-031). Image attachments on the new event—including screen
> captures and image files—reach the capable role as bounded multimodal input. The limits are
> 5 MiB per image and 10 MiB across one turn; a missing, unreadable, or oversized image degrades
> to an explicit text note instead of failing the turn.
>
> Non-image file contents, links, and selected text remain reference-only at the model boundary.
> Storage and transport still make those items durable, but durability is not document
> understanding. The tray may present screen capture and image inspection as working while
> continuing to label non-image context honestly.

`context[].kind` must be one of **`file` `image` `text` `link` `screen`** — lowercase.
`StagedContext.Kind`'s raw values are capitalised (`"File"`, `"Image"`, …), so **map the case,
do not send `rawValue`.** A wrong kind is rejected with a named error, not silently accepted.

### omega → tray

**On connect, before the tray has said anything**, omega sends an unsolicited greeting:

```jsonc
{"v":1,"op":"hello","head":41}
```

A client that assumes its first read is the answer to its first request will misparse this.
It is also useful: `head` is the current end of the log, which is the cursor a first-launch
tray should subscribe from. There is no handshake to perform — reading `hello` is enough.

Responses:

```jsonc
{"v":1,"op":"ack","seq":42,"duplicate":false}
{"v":1,"op":"ack","seq":42,"duplicate":true,"conflict":"that id is already in the log carrying a different message"}
{"v":1,"op":"attached","blob":"sha256:9f86d081…","mime":"image/png","bytes":184320}
{"v":1,"op":"subscribed","since":0,"head":41}
{"v":1,"op":"pong","head":41}
{"v":1,"op":"error","error":"…","request":"say"}
```

`error` carries `request` only when the op was identifiable — a line that is not JSON, or is
JSON but not an object, gets an error without it. The over-length error (`"line longer than
1048576 bytes"`) also has no `request`, and the server closes the connection immediately
after sending it.

Pushed updates, after `subscribe`:

```jsonc
{"v":1,"op":"update","seq":42,"state":"understood","for_seq":42,"at":"…","kind":"message.inbound",
 "text":"…","urgency":"normal","context":[{"id":"…","kind":"file"}]}

{"v":1,"op":"update","seq":43,"state":"working","for_seq":42,"at":"…","kind":"tool.called","tool":"read_file"}
{"v":1,"op":"update","seq":44,"state":"working","for_seq":42,"at":"…","kind":"tool.returned","tool":"read_file","ok":true}

{"v":1,"op":"update","seq":45,"state":"blocked","for_seq":42,"at":"…","kind":"turn.blocked","needs":"which file did you mean?"}

{"v":1,"op":"update","seq":45,"state":"complete","for_seq":42,"at":"…","kind":"turn.completed","outcome":"spoke","reply":"…"}
{"v":1,"op":"update","seq":45,"state":"complete","for_seq":42,"at":"…","kind":"turn.completed","outcome":"silent","reply":null}
{"v":1,"op":"update","seq":45,"state":"failed","for_seq":42,"at":"…","kind":"turn.completed","outcome":"failed","reply":null,"error":"…"}
```

Five states and no more: **`understood` `working` `blocked` `complete` `failed`**. A sixth is
a design change, not an implementation detail — if the tray seems to need one, stop and say so.

`seq` is the episode that produced the line and doubles as the **resume cursor**. `for_seq` is
the turn it belongs to: group by `for_seq` and you never need a mapping table across a restart
of either side.

Optional fields are **omitted** rather than sent as null — except `reply`, which is
meaningfully null and is always present on `complete`/`failed`.

---

## 3. The one thing to get right: silence is a success

`outcome` is `spoke`, `silent`, or `failed`.

- **`spoke`** — append an omega message bubble with `reply`.
- **`silent`** — omega decided a reply would add nothing. **Append no bubble.** Do not render
  an empty message, and do not render it as an error. `TrayMessage.Role.status` already exists
  and is the right home for a quiet line like *"omega stayed quiet."* — or show nothing at all,
  but the work state must settle back to a non-busy state either way.
- **`failed`** — the turn broke. Use `error` for the detail and keep the retry affordance.

Merging `silent` into either of the others is the single worst outcome of this work. An agent
with initiative that cannot stay quiet becomes an agent that interrupts you constantly, and
that is the failure mode that kills the product. There is a test on the Python side asserting
these render differently; the Swift side needs its own.

---

## 4. Connection semantics you cannot guess from the shapes

**There is no request correlation id.** Responses and pushed updates share one connection, and
the pump writes from a different thread than the one answering requests, so an `ack` and an
`update` can interleave. Demultiplex on `op`. Keep **at most one request in flight per
connection** — with two `say`s outstanding there is no way to tell which `ack` is which. That
is cheap here (a human types one message at a time) and should be enforced in the client, not
assumed.

**`duplicate: true` is the good outcome of a retry**, not an error. Reuse the same `id` when
retrying after a dropped connection: the log returns the original `seq` and tells you the first
attempt landed. Without an `id` a retry is honestly a *second message*, because nothing
identified it as the same one. `conflict` additionally means that id is already in the log
carrying different text — that is a client bug worth surfacing in a log line.

**Resuming.** `subscribe` with `since: N` replays everything after `N`. Persist the last `seq`
you processed (`AppSettings` is the natural home) so a relaunch resumes rather than replaying
the whole history or missing an in-flight turn. On a first launch with nothing persisted, use
the `head` from the `hello` greeting.

**`since` ahead of `head` is an error**, which is what you get if the store was reset while the
tray kept its cursor. Catch it and fall back to `hello`'s `head`.

**Ordering: `seq` is monotonic, never contiguous.** Updates arrive in ascending `seq` order,
and that is the whole guarantee. **A gap is not data loss and must never trigger a reconnect.**
`project()` returns `None` for any episode the projection withholds, and `_pump_once` then
advances the cursor past it on purpose — the comment there says why: otherwise a withheld kind
would be re-read on every tick forever. A client that reconnects on a gap would ask for the
same range, receive the same gap, and reconnect again, forever.

**This is no longer theoretical, and the earlier draft of this document was wrong about it.**
It used to say the stream "happens to be contiguous — all six M1 episode kinds project". That
was true when there were six kinds. There are now **twelve, and six of them are withheld** —
the whole learning and scheduling half of the log (§5). A busy teaching session produces long
runs of `seq` the tray will never be handed. If any code was written against the contiguous
reading, that is the first thing to go looking for.

The `CLAIMED`/`DONE` checkpoints remain named checkpoints rather than episodes, so they still
consume no `seq`. Treat gaps as normal — and now expect them.

**The server outlives the tray.** Disconnecting is ordinary; omega keeps running and keeps
turning. Reconnect with backoff and resume from the cursor. Never let a broken pipe surface as
a modal.

**The recovery threshold — when to restore the draft, stated exactly.** A dropped connection
and a failed turn are different events and must not share a code path. The rule:

1. **Before the `ack`** — nothing is known to be in the log. Retry the same `say` with the
   same `id`. A `duplicate: true` answer means the first attempt landed; carry on as if it
   had been the first.
2. **After the `ack`, connection intact** — drive state from updates until a terminal one.
3. **After the `ack`, connection dropped** — **do not restore the draft.** The episode is
   durable and the turn will run whether or not the tray is watching; restoring the draft here
   is how one message becomes two. Reconnect, `subscribe` from the persisted cursor, and read
   the turn's outcome out of the replay.
4. **Only if that resume cannot establish an outcome** — the reconnect succeeds but no
   terminal update for that `for_seq` ever arrives, or the `ack` was never received in the
   first place — restore the draft and context and offer the retry.

The short version: **the draft comes back when the turn's fate is unknowable, not when the
socket hiccups.** The `ack` is the line between the two, which is what makes it worth holding
onto rather than treating as a formality.

---

## 5. The learning path — mostly a story about what the tray does *not* get

Omega learns now, on two paths, and the interesting thing from the Swift side is how little
of it reaches the wire. That is deliberate, and it is the part most likely to be mistaken for
a missing feature and "fixed".

Six of the twelve episode kinds are withheld by `projection.NOT_PROJECTED`:

| withheld kind | what it records |
|---|---|
| `claim.extracted` | omega wrote down something it was taught |
| `claim.retracted` | omega dropped something it had been told |
| `claim.extraction_failed` | omega tried to write something down and could not |
| `schedule.created` | a taught time became a standing schedule |
| `schedule.cancelled` | a standing schedule was stopped |
| `reflection.done` | a background pass read a window of conversation |

**The one rule behind all six: one turn, one outward voice.** A turn that learns four things
says so once, in its reply. Giving claims their own wire update would be a second way for one
turn to reach the person — which `act.py` §1.6 already refuses for the agent's own output, and
which is no better arriving through the projection. If the tray ever grows a "learning" feed,
that decision gets made in the ledger first, not in Swift.

### Teaching is not a mode, and there is no teach op

DL-034: a teach drop is deliberately indistinguishable **in kind** from an ordinary inbound
message. *"Remember that I take my medication at 6:40"* is a `say` like any other — same op,
same fields, same `understood → … → complete`. Omega decides it was a teach drop; the tray
does not, and must not send a flag saying so.

This settles the **Teach Omega composer mode** in `V1_SPEC.md`. It is legitimate as a *framing
affordance* — a composer that prompts the person to phrase a durable fact well — but it must
compile to a plain `say`. It is not a different submission type and it does not get a different
endpoint. The tray shaping how the person writes is fine; the tray telling omega how to file it
is the boundary this document opens by drawing.

### The receipt rides the reply, and a teach drop is never `silent`

What omega learned comes back inside the ordinary `turn.completed` reply, appended after any
conversational text with a blank line between them:

```
I wrote this down:
- I take my medication at 6:40 on weekdays (when you mention medication)
  replaces what you told me before: I take my medication in the mornings

I will wake up and do this:
- remind me to take my medication before I leave the house (at 6:40 on weekdays) [s42-0]
```

That block is the real output of `learn.receipt`, not a sketch of it. Note the two different
parentheticals, because they are easy to conflate: a **claim**'s is its *conversational*
trigger — when omega will bring it up — while a **schedule**'s is a clock time. *"when you
mention medication"* and *"at 6:40 on weekdays"* are answers to different questions, and the
receipt deliberately shows both so the person can disagree with either. The bracketed `s42-0`
is the schedule id; it is the only id the receipt shows, and it is there because stopping a
schedule needs a handle.

Three consequences for rendering:

1. **It is already Markdown-shaped** — lists, and a two-space-indented continuation line under
   a superseded claim. The spec's native-Markdown rendering handles it; nothing special is
   needed, but do not strip leading `-` or collapse the indent, because the indent is what
   makes "replaces what you told me before" belong to its claim rather than float free.
2. **The receipt is built from what was actually appended**, never from what the model claimed
   it wrote. A line in that block is a durable fact about the log. This is the person's
   precision check on omega's memory and the only place a *wrong* thing omega learned becomes
   visible — so it must render in full, never truncated behind a "show more".
3. **A teach drop always completes as `spoke`, even when the judge chose silence.** When
   there is no conversational reply, the receipt *becomes* the reply, because a teach omega
   says nothing about is indistinguishable from one it dropped. So §3's silent path is real
   for ordinary turns and unreachable for teaching ones. Do not special-case it; just know
   that a teach drop with no visible bubble is a bug, not a quiet success.

A failed extraction is the same shape: *"I could not write that down — …. Nothing was
recorded, so tell me again if it matters."* arrives as an ordinary `spoke` reply, **not** as
`failed`. The turn genuinely succeeded; the learning inside it did not. Rendering that as a
broken turn would be wrong, and adding a retry affordance to it would be worse — the person
re-teaching in their own words is the retry.

### Omega now learns from conversation nobody meant as teaching

DL-054: when the queue goes idle and enough has been said since the last pass, omega reflects
over a window of recent conversation and may file claims the person never typed. Those are
filed as **not explicit** — the distinction is what makes a contradiction worth interrupting
for or not.

**The tray sees none of this.** No update, no indicator, no "omega is thinking". That is the
`reflection.done` row in the table above and it is the strongest of the six: pushing it would
be omega announcing it had been thinking about you, unprompted, roughly once a session, which
is the notification firehose starting small — the named terminal failure in DL-011. A tray
that surfaces reflection activity is a tray that has reintroduced the thing this whole design
exists to avoid.

It does mean something slightly odd is true and worth knowing before it is filed as a bug:
**omega can turn out to believe something the person never sees it learn.** It will say so
when it becomes relevant, or on request. That is the intended shape.

### A schedule firing is the live proactivity path

DL-035: when a schedule fires, omega appends an ordinary `message.inbound` and runs an
ordinary turn. On the wire it is `understood` with a `for_seq` the tray never sent, followed
by the usual states.

§6's last bullet on `TrayViewModel` already says updates can arrive for a `for_seq` this
client never sent and the message list must tolerate an omega bubble with no user bubble
before it. **That is no longer a future concern — it is how every reminder arrives**, and with
the tray live it is exercisable today: teach omega a time a couple of minutes out and watch
the turn appear unprompted. It is also the cheapest end-to-end proof that the clock, the
schedule store and the projection agree.

### The honest gap: the tray cannot read what omega has learned

There is no op for it. `say`, `attach`, `subscribe`, `ping` — that is the whole vocabulary,
and none of them answers *what do you currently believe about me*. The standing set is
reachable only offline, through `python -m omega --learned` against a store omega is not
running on (DL-016: the log holds an exclusive lock for the lifetime of the open handle, so
that command cannot be run against the live process).

So a "what omega knows" panel is **not buildable today**, and the missing piece is a read op
on the channel, not anything in Swift. It is a real gap rather than an oversight — DL-049's
ring-1 read-only tools are where it gets decided — and it is named here so the tray side does
not spend a day looking for an endpoint that does not exist. The available answer in the
meantime is the conversational one: ask omega.

---

## 6. Swift changes, file by file

### `TrayTransport.swift` — rewrite
Replace the protocol with a connection abstraction. Shape it however suits the codebase, but
it must express: connect, `subscribe(since:)`, `say(…) -> ack`, and an `AsyncStream` (or
equivalent) of decoded updates. **Delete `LocalDemoTransport` entirely** — it is explicitly
marked for deletion once the real boundary exists, and leaving it as a fallback would make it
the thing that silently runs when the connection fails.

### New — the client itself
JSON-lines socket client: `Network.framework` or `URLSessionStreamTask`, line framing with the
1 MiB cap, `Codable` types for every message above, reconnect with backoff, cursor persistence.

`ChannelClient` in `python/omega/channel.py` (~line 440) is a working reference for the two
non-obvious bits: it reads the greeting in its initialiser (`self.hello = self.read()`), and
its `request()` buffers any `update` that lands between a request and its reply instead of
mistaking it for the reply. It is deliberately *not* a model for reconnect, backoff or
queueing — it has none, because those have to be visible to a person and that is the tray's
job, not a helper's.

Decoding must tolerate **unknown fields and unknown `state`/`kind` values** — omega gains kinds
before the tray does, and a strict decoder turns a Python-side addition into a crash on your
users' machines.

### `TrayModels.swift`
- `TraySubmission` currently carries `contextDescriptions: [String]`. It needs **identity**:
  the wire wants `{id, kind, title}` per item, plus `{blob, mime, bytes}` for an item with
  bytes behind it (§2). Carry the `StagedContext` ids and the digest `attach` returned (or a
  small value type), not display strings.
- `WorkState` maps from the five wire states. Decide deliberately where `silent` lands (§3).
- Note the asymmetry: the tray **sends** `{id, kind, title}` and **receives back** only
  `{id, kind}`. Titles are deliberately stripped — they are user-visible text with no reader on
  the wire. The tray keeps the previews; the log keeps the identities. So on a replay after a
  restart you cannot reconstruct titles from the stream, and must key your own staged-context
  store by `id`.

### `TrayViewModel.swift`
`deliver(_:messageID:originalDraft:originalContext:)` at line 149 is where the single `await`
lives (line 158). It becomes: send → hold the returned `seq` → drive state from updates whose
`for_seq` matches, until a terminal one (`complete`/`failed`/`blocked`) arrives.

**`workState` already has a second writer, and this is the subtle one.** The screen-capture
flow sets `.working`, `.blocked` and `.ready` locally at lines 273–343 — permission prompts
and capture progress that are genuinely the tray's own business and have no episode behind
them. Once updates also drive `workState`, the two can stomp each other: a capture permission
prompt can be overwritten by an incoming `working`, or a stale capture `.ready` can clear a
real `blocked`. Decide the arbitration explicitly rather than letting call order decide it.
The cleanest split is to keep them as **two** stored properties — local UI state and
turn state — and derive the displayed one, with local capture state winning while a capture
is actually in flight. Whatever you choose, write the test that proves it.

- Keep the existing **draft-and-context restoration on failure** working. It is genuinely good
  behaviour and it is currently proven against a thrown error from one `await`; it now has to
  be proven against a `failed` update *and* against the connection dropping mid-turn, which are
  different paths.
- `workState = .working("Working locally")` is currently set unconditionally and is a lie the
  moment it is real. Drive `working` from actual `working` updates, using `tool` for the detail.
  Nothing may claim "working" without an episode saying so — that is the whole reason the
  projection exists rather than a status API.
- A turn can reach `blocked` and stop. That is terminal for the turn; the answer is a **new**
  `say` carrying `resumes_seq`, not a resumption of the old one.
- Updates can arrive for a `for_seq` this client never sent (another client, or later the
  clock). Render them rather than dropping them — but they have no local `messageID`, so the
  message list needs to tolerate an omega bubble with no preceding user bubble of its own.

### `AppDelegate.swift`
Line 6 constructs `TrayViewModel(transport: LocalDemoTransport())`. It becomes the real client,
plus connection lifecycle on launch/teardown.

### Tests
`TrayModelTests.swift` has `ImmediateTransport` and `FailingTransport`; `TraySnapshotTests.swift`
has `SnapshotTransport`. All three implement the old protocol and must be rebuilt as scripted
*stream* fakes. Cover at least: a spoke turn; **a silent turn rendering differently from both a
spoke turn and a failure**; a blocked turn and its `resumes_seq` answer; a connection dropping
mid-turn with draft and context restored; a reconnect resuming from the cursor with no
duplicate bubbles; an unknown `kind` being ignored rather than crashing.

---

## 7. Running omega to develop against

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install "maturin>=1.7,<2.0" pytest
.venv/bin/maturin develop --release
.venv/bin/python -m omega            # REPL + listener on 127.0.0.1:7717
```

Flags: `--store PATH`, `--env PATH`, `--port PORT`, `--no-listen`. The REPL and the tray can be
attached at the same time and should see the same stream — that is the cheapest way to prove
the projection is a projection and not a second path.

`OPENAI_API_KEY` lives in `.env` and is never needed by any test on either side. **If a tray
test needs a key or a network, that is a bug in the test.**

One hard constraint from `M1_SPEC.md` §1.6: **the only way to cause a turn is to append an
episode.** The tray causes turns by sending `say` and by nothing else. There is no side door,
and adding one would make initiative a second engine wearing omega's name.
