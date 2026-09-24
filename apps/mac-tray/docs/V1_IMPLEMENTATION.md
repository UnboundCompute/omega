# Mac tray v1 — implementation status

Status: native surface, M1 streaming transport, image understanding, and durable attachment
ingestion complete; non-image attachment understanding unresolved.

This file maps the confirmed contract in `V1_SPEC.md` to the implementation. It separates
tray work from the agent-core work under `agent/` so the first transport cannot accidentally
become a second brain or an undeclared wire protocol.

## Implemented in the tray

- Top-center resting capsule, expanded panel, and non-focus-stealing proactive peek.
- Persistent `NSPanel` behavior across app switches, Spaces, full-screen apps, and displays.
- Configurable global shortcuts for panel toggle and direct area capture, with independent
  registration-conflict recovery.
- Focus enters the composer on open and returns to the prior app on explicit close.
- Area, window, and active-display capture through the macOS capture tool.
- The panel is removed before capture begins and restored after success, cancellation, or launch
  failure, preventing omega from covering or appearing inside the selected content.
- Screen Recording permission request, denied state, and direct System Settings recovery.
- File, image, URL, selected-text, and clipboard staging through native pasteboard/drop APIs.
- A resting drop target that expands before drop and states that the item will not be sent.
- Explicit submission only; capture and drop never imply send.
- Quick Look for staged file-backed context.
- Sent-context receipts, durable plain-language work state, and recoverable failed delivery.
- Long-lived JSON-lines connection to omega on `127.0.0.1:7717`, with unsolicited greeting,
  subscription replay, persisted cursor, bounded reconnect backoff, and a 1 MiB line limit.
- Stream-driven understood, working, blocked, complete, failed, spoken, and silent outcomes;
  no locally invented agent progress.
- Stable submission ids and duplicate acknowledgements for at-most-once retry behavior.
- Capture, file, and image ingestion through the channel's `attach` operation before submission,
  with visible storing/ready/failure states and retry without losing staged context.
- Content-addressed attachment references (`blob`, `mime`, and `bytes`) carried atomically into the
  eventual message episode; Send remains disabled until every file-backed item is durable.
- Post-ack disconnect recovery that resumes from the projection before restoring a draft.
- Forward-compatible decoding of unknown update states and kinds without losing the cursor.
- Exact draft and context restoration after a failed send; retry does not create a duplicate
  visible instruction.
- One active in-memory conversation that survives close/reopen during the app process.
- Proactive peek, unread collapse, no duplicate while open, and time-sensitive Notification
  Center fallback.
- A non-focus-stealing response peek when a submitted turn finishes after the panel has been
  closed.
- Preview redaction by default, screen-lock redaction, and a manual Privacy Veil for presenting
  or screen sharing.
- Launch-at-login support in the packaged app through `SMAppService`.
- Menu-bar fallback for open, capture, privacy, settings, sample proactivity, and quit.
- Keyboard semantics, VoiceOver labels, native type and controls, Reduced Motion, Reduced
  Transparency, inactive appearance, and state labels that do not rely on color.
- A reproducible `.app` packaging script that uses a stable local signing identity when one is
  available, preserving macOS privacy grants across rebuilds, with an explicit ad-hoc fallback.
- Render agent replies as native structured Markdown while preserving selection and links.
- Start a non-interactive omega runtime when the packaged tray launches, reuse an already-running
  resident process, stop the owned process on app quit, and surface startup failures in the tray.
- Add a lossless New Chat surface reset and a one-shot Teach Omega composer mode. Teach Omega is
  only a writing affordance: it sends the person's exact note through the ordinary `say` path,
  and omega's normal reply provides the durable claim or schedule receipt. A dedicated composer
  control enters the mode in one click; the overflow menu is not part of the learning path.
- Preserve indented continuation lines in learning receipts so replacement details stay attached
  to the claim they qualify, while rendering the complete receipt without truncation.
- Accept monotonic projection sequence gaps as normal; withheld learning and scheduling episodes
  never cause a reconnect or false data-loss state.

## Deliberately waiting at the model-understanding boundary

`LocalDemoTransport` is gone. `OmegaChannelClient` speaks the M1 channel and the tray remains a
projection client: it does not persist a second transcript or implement memory, initiative
judgement, task execution, or verification.

File-backed attachment ingestion is complete. The tray uploads each capture, dropped file, or
pasted image while it is staged; omega copies the bytes into `<store>/blobs`, returns their
content digest, and the tray sends that immutable reference with the message. Upload errors are
visible and retryable, and cannot silently degrade into metadata-only delivery.

Image attachments, including screen captures, reach the capable model as bounded multimodal input.
Non-image file contents are durable and re-readable by omega but do not reach the model yet. URL
values and selected text also remain display context rather than model-readable content. The UI
discloses this only when reference-only context is staged; durable ingestion must not be described
as document understanding.

The transport preserves these tray-side guarantees:

1. A user submission is acknowledged once and routed to the single omega queue.
2. Context items retain stable identity through delivery and failure recovery.
3. Inbound replies and task states never steal focus.
4. Proactive events declare whether they are normal or genuinely time-sensitive.
5. Delivery outcomes distinguish accepted, failed, blocked, and verified complete.
6. At-most-once behavior belongs to the cross-channel core, not this UI.

## Verification

Automated:

```sh
swift test
./scripts/package-app.sh
```

The test suite covers framing, attachment and message wire codecs, interleaved request/update demultiplexing,
unknown updates, spoken and silent turns, blocked-turn resume, cursor persistence, duplicate
replay suppression, post-ack reconnect recovery, context identity, capture/turn state
arbitration, attachment upload gating and retry, closed-panel response nudges, both shortcut
preferences, and proactive presentation. A live opt-in test uploads a real file through the
Python listener and validates the returned reference. Opt-in review renders cover
resting, peek, empty, staged-context disclosure, delivery recovery, drop target, and privacy.

Manual acceptance on a signed app bundle:

- Grant and deny Screen Recording once each; verify the recovery path.
- Exercise area, window, and each connected display capture.
- Drop files, images, links, and text while closed and open.
- Navigate every control with Full Keyboard Access and VoiceOver.
- Toggle Reduce Motion, Reduce Transparency, Increase Contrast, and Differentiate Without Color.
- Open over a notched laptop, an external display, a Space, and a full-screen application.
- Switch applications while open; verify omega persists without retaining keyboard focus.
- Enable launch at login, log out/in, and verify the packaged app returns.
- Turn on Privacy Veil during screen sharing and verify no conversation content is visible.
- Run repeated text-only delivery, disconnect, restart, silent, blocked/resume, and proactive
  tests against the real omega listener.
- Verify capture/image understanding with real pixels, and keep non-image files labeled
  reference-only until the provider can interpret their contents.
