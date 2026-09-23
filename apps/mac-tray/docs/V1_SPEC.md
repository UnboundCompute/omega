# Mac tray v1 — product and interface contract

Status: confirmed on 2026-09-23; native surface implemented. See `V1_IMPLEMENTATION.md` for
verification and the intentionally deferred M1 transport handoff.

This document records the agreed v1 experience before code. Changes to the product shape
should update this document and the private decision ledger in `AGENT.md`.

## Product thesis

The Mac tray is omega's desk presence: a persistent, screen-aware companion attached to
the top-center camera area. It is not a miniature full app, menu-bar dashboard, command
palette, memory browser, or generic chat popover.

The defining loop is:

> See something → summon omega → attach context → ask or delegate → keep working → receive
> an honest result.

## v1 experience

### Presence and placement

- omega rests at the top-center of the active display.
- On a notched MacBook it visually belongs to the camera housing.
- On a display without a notch it becomes a small top-center capsule.
- It adapts to camera safe areas, menu-bar geometry, display scale, Spaces, full-screen
  applications, and multiple displays.
- The idle state occupies almost no additional screen area.

### Opening, focus, and closing

- A configurable global hotkey toggles the panel.
- Opening focuses the composer immediately.
- The open panel stays above other windows and remains visible across app switches.
- Clicking outside gives focus back to the underlying app but never closes omega.
- Incoming content never steals keyboard focus.
- The hotkey or an explicit close control closes the panel.
- Closing and reopening preserves the active conversation, staged context, and scroll
  position.

### Capture and staging

- The user can capture an area, window, or display through the native macOS selection
  experience.
- omega temporarily leaves the screen while capture is active, then restores the exact prior
  tray state after selection or cancellation, so content behind it remains selectable.
- Files, images, screenshots, selected text, URLs, and clipboard content can be dragged or
  pasted into omega.
- Dragging toward the camera widens omega into an obvious drop target before drop.
- Captured items show a real preview, source, type, removable state, and whether they have
  been sent.
- **Capture is not send.** Every item is staged until the user deliberately submits it.
- One active capture bundle is sufficient for v1; additional items join that bundle.
- Staged items are never silently discarded.
- Continuous screen recording or background screen inspection is not part of v1.

### Conversation and delegation

- v1 supports one active conversation well rather than exposing a history browser.
- User instructions are compact; omega responses are text-led rather than a wall of chat
  bubbles.
- The panel grows with the exchange to a safe maximum height, then scrolls internally.
- The composer remains available at the bottom.
- Delegated work has durable plain-language states: understood, working, blocked, failed,
  and verified complete.
- Progress describes real state. It never invents percentages or uses indefinite typing
  dots as proof of work.
- Completion remains available after temporary UI feedback disappears.

### Proactivity

- omega can initiate a message from the top-center camera area.
- If the main panel is open, the message joins the active conversation without a duplicate
  notification.
- If closed, omega shows a small non-focus-stealing peek containing one or two useful
  lines.
- Ignoring a peek collapses it into a quiet unread indicator and never repeatedly animates
  the same thought.
- Clicking a peek opens the persistent panel.
- Sensitive content is replaced with neutral copy while locked, screen sharing, or
  presenting.
- System notifications are a fallback for meaningfully time-sensitive information, not
  the default delivery mechanism.

### Native Mac behavior

v1 uses the relevant macOS conventions and capabilities:

- camera-housing safe areas and active-display placement;
- multi-monitor, Spaces, and full-screen behavior;
- persistent floating-panel active/inactive appearance;
- configurable global shortcuts and focus restoration;
- native text editing, drag and drop, pasteboard, and Quick Look;
- native screen capture selection and permission flow;
- system typography and semantic status colors;
- VoiceOver, Full Keyboard Access, Reduce Motion, Reduce Transparency, Increase Contrast,
  and Differentiate Without Color;
- Notification Center and Focus awareness;
- launch at login and a small menu-bar fallback for reopen, settings, and quit.

Widgets, Spotlight, Siri, Shortcuts, Finder and Share extensions, Handoff, iCloud sync, and
Dock presence are not included merely to claim platform integration. They can earn their
place later.

## Visual system: Quiet Instrument

omega should feel like a small machined object unfolding from the camera housing.

### Structure

- The black camera region appears to stretch and unfold downward.
- The panel has no conventional title bar; the camera housing is its visual title.
- The surface is compact, approximately 420–480 points wide, with a maximum height around
  half the available display.
- Corners are concentric and physically connected to the top shell.
- Depth comes from one soft ambient shadow and a subtle inner highlight, not stacked borders
  or decorative glass.
- The inactive panel becomes quieter without becoming unreadable.

### Palette

The surface remains dark in both macOS appearances because it is visually continuous with
the physical camera housing. Contrast, material, and shadow adapt to the environment.

| Token | Value | Purpose |
| --- | --- | --- |
| Notch shell | `#050506` | Physical outer shape |
| Main surface | `#111214` | Conversation panel |
| Raised surface | `#1A1C1F` | Attachments and task states |
| User instruction | `#24262B` | Compact user-message surface |
| Primary text | `#F5F5F7` | Responses and important information |
| Secondary text | `#A8ABB2` | Metadata and explanations |
| Tertiary text | `#74777E` | Inactive labels and timestamps |
| Omega signal | `#FFB45C` | Identity, unread state, active seam |
| Inner highlight | white at 8–10% | Edge separation |

Amber is used sparingly for omega's identity, unread/active seam, selected context, and a
single primary emphasis. Success, warning, and failure use semantic macOS colors. State is
always paired with text, shape, or motion and never relies on color alone.

### Typography and icons

- Use native macOS system typography and text rendering.
- Use monospaced text only for code, data, or measurement.
- Use SF Symbols for familiar system actions.
- omega's identity mark is a custom two-arc glyph between an omega, an eye, and a camera
  aperture; it is not a mascot or robot.

### Motion

- Opening unfolds from the camera; closing folds back into it.
- Structural transitions are controlled and quick, roughly 200–300 ms, without bounce.
- Dragging context causes the resting surface to widen before it deepens.
- New response content settles without shifting the whole panel.
- Working state animates only the bottom signal seam.
- Reduced Motion replaces shape morphs with short fades and immediate final layout.

## Required states

- First launch and permissions
- Idle
- Closed with unread proactive message
- Proactive peek
- Open and empty
- Open with staged context
- Drag target
- Capturing
- Sending
- Responding
- Working in background
- Blocked for user input
- Verified completion
- Recoverable failure
- Offline or agent unavailable
- Inactive while another app has focus
- Reduced-motion, reduced-transparency, and increased-contrast variants

## Explicitly deferred to v2

- Full application shell
- Conversation archive and search
- Memory browser, graph, profile, or correction dashboard
- Multiple visible projects and threads
- Integration management
- Rich task history
- Advanced proactive feed
- Realtime voice conversation
- Continuous screen understanding
- Custom themes

## Architecture boundary for the first implementation

The v1 UI is a native macOS application because the defining experience depends on panel,
focus, display, capture, drag-and-drop, and accessibility behavior that should feel native.
The app lives entirely under `apps/mac-tray`.

The initial implementation may include a local demo responder to verify the interaction
loop. It must be named and documented as a demo transport, remain replaceable behind one
small seam, and never become the inferred cross-channel or agent contract.

## v1 acceptance bar

A real user can invoke omega without leaving the current application, stage visible screen
context without accidentally sending it, delegate a request, return focus to other work
while keeping omega visible, observe a response arrive without focus theft, close and reopen
without losing state, and receive one proactive message through the same surface.

The experience must remain usable with keyboard only and VoiceOver, on a notched laptop and
an external display, in light and dark environments, and with reduced motion enabled.
