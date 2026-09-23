# Mac tray v1 — implementation status

Status: native surface complete; agent transport intentionally waiting for omega M1.

This file maps the confirmed contract in `V1_SPEC.md` to the implementation. It separates
tray work from the agent-core work under `agent/` so the first transport cannot accidentally
become a second brain or an undeclared wire protocol.

## Implemented in the tray

- Top-center resting capsule, expanded panel, and non-focus-stealing proactive peek.
- Persistent `NSPanel` behavior across app switches, Spaces, full-screen apps, and displays.
- Configurable global shortcut with registration-conflict recovery.
- Focus enters the composer on open and returns to the prior app on explicit close.
- Area, window, and active-display capture through the macOS capture tool.
- Screen Recording permission request, denied state, and direct System Settings recovery.
- File, image, URL, selected-text, and clipboard staging through native pasteboard/drop APIs.
- A resting drop target that expands before drop and states that the item will not be sent.
- Explicit submission only; capture and drop never imply send.
- Quick Look for staged file-backed context.
- Sent-context receipts, durable plain-language work state, and recoverable failed delivery.
- Exact draft and context restoration after a failed send; retry does not create a duplicate
  visible instruction.
- One active in-memory conversation that survives close/reopen during the app process.
- Proactive peek, unread collapse, no duplicate while open, and time-sensitive Notification
  Center fallback.
- Preview redaction by default, screen-lock redaction, and a manual Privacy Veil for presenting
  or screen sharing.
- Launch-at-login support in the packaged app through `SMAppService`.
- Menu-bar fallback for open, capture, privacy, settings, sample proactivity, and quit.
- Keyboard semantics, VoiceOver labels, native type and controls, Reduced Motion, Reduced
  Transparency, inactive appearance, and state labels that do not rely on color.
- A reproducible `.app` packaging script with an ad-hoc signature for local use.

## Deliberately waiting at the agent boundary

`LocalDemoTransport` remains visibly named and honest. It will be replaced at omega M1, when
the core loop exists and DL-016's channel envelope is designed. The tray does not invent that
format early, persist its own transcript as a second source of truth, or implement memory,
initiative judgement, task execution, or verification.

The transport replacement must preserve these tray-side guarantees:

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

The test suite covers empty submission, shortcut persistence, capture-mode arguments, sent
context receipts, failed-send restoration, and proactive-message consumption. Opt-in review
renders cover resting, peek, empty, staged-context, delivery-recovery, drop-target, and privacy
states.

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
- Once M1 lands, replace the demo transport and run repeated end-to-end delivery and restart
  tests against real logged turns.
