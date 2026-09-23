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
- forwarding deliberate user submissions through a replaceable transport seam.

It does not own:

- agent reasoning or personality;
- memory extraction, recall, or persistence;
- initiative judgement;
- tool execution and verification;
- cross-channel thread semantics.

Until the real channel contract is designed, the app uses a visibly labelled local demo
transport. Nothing in that demo interface is a stable agent protocol.

## Run the prototype

Requirements: macOS 14 or newer and the Swift toolchain included with Xcode.

```sh
cd apps/mac-tray
swift run OmegaTray
```

The prototype starts as a top-center resting capsule. Press `Control–Option–Space` to
toggle the persistent panel. The menu-bar fallback can trigger an area capture or a sample
proactive peek.

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
