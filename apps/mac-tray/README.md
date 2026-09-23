# omega Mac tray

The Mac tray is omega's v1 desk channel. It is a native macOS surface anchored to the
top-center of the active display and visually connected to the camera housing when one is
present.

Read [`docs/V1_SPEC.md`](docs/V1_SPEC.md) before changing behavior or appearance.

## Ownership boundary

This package owns:

- the top-center panel and its visual states;
- keyboard, focus, display, drag-and-drop, and accessibility behavior;
- explicit screen/file/text/URL capture and staging;
- rendering inbound messages and delegated-task state;
- forwarding deliberate user submissions through omega’s localhost channel.

It does not own:

- agent reasoning or personality;
- memory extraction, recall, or persistence;
- initiative judgement;
- tool execution and verification;
- cross-channel thread semantics.

The tray connects to omega on `127.0.0.1:7717` and renders the projected episode stream. It
does not run a second agent or fall back to a demo responder when omega is offline. The complete
surface status and current attachment limitation are recorded in
[`docs/V1_IMPLEMENTATION.md`](docs/V1_IMPLEMENTATION.md).

## Run during development

Requirements: macOS 14 or newer and the Swift toolchain included with Xcode.

Start omega from the repository root, then run the tray in another terminal:

```sh
.venv/bin/python -m omega
```

```sh
cd apps/mac-tray
swift run OmegaTray
```

The app starts as a top-center resting capsule. Press `Control–Option–Space` to toggle the
persistent panel, or `Control–Option–C` to select and stage a screen area directly. Both
shortcuts can be changed in Settings. The menu-bar fallback can capture, toggle the Privacy
Veil, show a sample proactive peek, open Settings, or quit.

Current M1 limitation: staged captures and files send stable identity, kind, and title, but not
their contents. The tray labels this honestly; omega cannot inspect an attachment until the
attachment-ingestion design is settled.

For visual inspection during development, launch directly into a state with
`OMEGA_TRAY_PREVIEW_STATE=expanded swift run OmegaTray` or use `peek`.

## Verify

```sh
cd apps/mac-tray
swift test
```

To write offscreen review renders when display capture is unavailable:

```sh
OMEGA_SNAPSHOT_DIR="$PWD/../../.impeccable/review" swift test \
  --filter TraySnapshotTests
```

## Package the local app

```sh
./scripts/package-app.sh
open .build/app/omega.app
```

The script builds release, creates the application bundle, uses the first available macOS
code-signing identity, and verifies both the bundle metadata and signature. Stable signing keeps
privacy permissions valid across local rebuilds. If no identity exists, it falls back to an
ad-hoc signature and warns that Screen Recording permission will need to be granted after each
rebuild. Distribution outside the local machine still requires Developer ID signing and
notarization.

To build, replace `/Applications/omega.app`, and open the installed app in one command:

```sh
./scripts/install-app.sh
```

The installer asks a running omega instance to quit cleanly and refuses to overwrite the app if
it does not exit. It only removes the exact `/Applications/omega.app` destination.
