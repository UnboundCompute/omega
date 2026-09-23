import AppKit
import SwiftUI
import XCTest
@testable import OmegaTray

final class TraySnapshotTests: XCTestCase {
    @MainActor
    func testWriteReviewSnapshotsWhenRequested() throws {
        guard let outputDirectory = ProcessInfo.processInfo.environment["OMEGA_SNAPSHOT_DIR"] else {
            throw XCTSkip("Set OMEGA_SNAPSHOT_DIR to write review snapshots.")
        }

        let directory = URL(fileURLWithPath: outputDirectory, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        let model = TrayViewModel(transport: SnapshotTransport())
        try write(
            TrayRootView(viewModel: model, presentation: .resting, close: {}, open: {})
                .frame(width: 190, height: 38),
            to: directory.appendingPathComponent("mac-resting.png")
        )

        model.proactivePeek = "I found something worth your attention while you were working."
        try write(
            TrayRootView(viewModel: model, presentation: .peek, close: {}, open: {})
                .frame(width: 390, height: 108),
            to: directory.appendingPathComponent("mac-peek.png")
        )

        model.proactivePeek = nil
        try write(
            TrayRootView(viewModel: model, presentation: .expanded, close: {}, open: {})
                .frame(width: 460, height: 380),
            to: directory.appendingPathComponent("mac-expanded.png")
        )

        model.stagedContext = [
            StagedContext(kind: .text, title: "Roadmap notes from the planning call", detail: "Text · Not sent"),
            StagedContext(kind: .link, title: "developer.apple.com", detail: "Link · Not sent")
        ]
        try write(
            TrayRootView(viewModel: model, presentation: .expanded, close: {}, open: {})
                .frame(width: 460, height: 456),
            to: directory.appendingPathComponent("mac-staged-context.png")
        )

        model.workState = .failed("Request not delivered. Your draft and context were restored.")
        try write(
            TrayRootView(viewModel: model, presentation: .expanded, close: {}, open: {})
                .frame(width: 460, height: 520),
            to: directory.appendingPathComponent("mac-recovery.png")
        )

        model.workState = .blocked("Screen Recording permission is needed. Open System Settings to allow it.")
        model.capturePermission = .denied
        try write(
            TrayRootView(viewModel: model, presentation: .expanded, close: {}, open: {})
                .frame(width: 460, height: 520),
            to: directory.appendingPathComponent("mac-permission-recovery.png")
        )

        model.isDropTargeted = true
        try write(
            TrayRootView(viewModel: model, presentation: .resting, close: {}, open: {})
                .frame(width: 320, height: 72),
            to: directory.appendingPathComponent("mac-drop-target.png")
        )

        model.isDropTargeted = false
        model.manualPrivacyMode = true
        try write(
            TrayRootView(viewModel: model, presentation: .expanded, close: {}, open: {})
                .frame(width: 460, height: 380),
            to: directory.appendingPathComponent("mac-privacy-veil.png")
        )
    }

    @MainActor
    private func write<Content: View>(_ content: Content, to destination: URL) throws {
        let hostingView = NSHostingView(rootView: content)
        hostingView.frame = NSRect(origin: .zero, size: hostingView.fittingSize)
        hostingView.layoutSubtreeIfNeeded()

        guard let bitmap = hostingView.bitmapImageRepForCachingDisplay(in: hostingView.bounds) else {
            XCTFail("Could not render \(destination.lastPathComponent)")
            return
        }

        hostingView.cacheDisplay(in: hostingView.bounds, to: bitmap)

        guard let data = bitmap.representation(using: .png, properties: [:]) else {
            XCTFail("Could not encode \(destination.lastPathComponent)")
            return
        }

        try data.write(to: destination, options: .atomic)
    }
}

private struct SnapshotTransport: TrayTransport {
    func send(_ submission: TraySubmission) async throws -> String {
        "Snapshot"
    }
}
