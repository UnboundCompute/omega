import XCTest
@testable import OmegaTray

final class TrayModelTests: XCTestCase {
    @MainActor
    func testCannotSendAnEmptySubmission() {
        let model = TrayViewModel(transport: ScriptedTransport(), loadCursor: { nil }, persistCursor: { _ in })
        XCTAssertFalse(model.canSend)
    }

    @MainActor
    func testTextMakesSubmissionSendable() {
        let model = TrayViewModel(transport: ScriptedTransport(), loadCursor: { nil }, persistCursor: { _ in })
        model.draft = "Handle this"
        XCTAssertTrue(model.canSend)
    }

    @MainActor
    func testFileIsAttachedBeforeItsContextCanBeSent() async throws {
        let (model, transport) = connectedModel()
        let file = FileManager.default.temporaryDirectory
            .appendingPathComponent("omega-attachment-\(UUID().uuidString).txt")
        try Data("attachment body".utf8).write(to: file)
        defer { try? FileManager.default.removeItem(at: file) }

        model.stageFile(file)
        XCTAssertFalse(model.canSubmit)
        await settleTasks()

        let context = try XCTUnwrap(model.stagedContext.first)
        XCTAssertEqual(transport.attachments, [file])
        XCTAssertEqual(context.attachment, transport.attachmentReference)
        XCTAssertTrue(model.canSubmit)

        model.send()
        await settleTasks()
        let sent = try XCTUnwrap(transport.submissions.first?.context.first)
        XCTAssertEqual(sent.blob, transport.attachmentReference.blob)
        XCTAssertEqual(sent.mime, transport.attachmentReference.mime)
        XCTAssertEqual(sent.bytes, transport.attachmentReference.bytes)
    }

    @MainActor
    func testFailedAttachmentBlocksSendAndCanBeRetried() async throws {
        let (model, transport) = connectedModel()
        let file = FileManager.default.temporaryDirectory
            .appendingPathComponent("omega-attachment-\(UUID().uuidString).bin")
        try Data([0x01]).write(to: file)
        defer { try? FileManager.default.removeItem(at: file) }
        transport.attachmentError = AttachmentTestError.refused

        model.stageFile(file)
        await settleTasks()

        XCTAssertNotNil(model.attachmentFailure)
        XCTAssertFalse(model.canSubmit)

        transport.attachmentError = nil
        model.retryFailedAttachments()
        await settleTasks()

        XCTAssertNil(model.attachmentFailure)
        XCTAssertTrue(model.canSubmit)
    }

    @MainActor
    func testInMemoryImageIsMaterializedAndAttachedAsPNG() async throws {
        let (model, transport) = connectedModel()
        let image = NSImage(size: NSSize(width: 2, height: 2))
        image.lockFocus()
        NSColor.systemOrange.setFill()
        NSRect(x: 0, y: 0, width: 2, height: 2).fill()
        image.unlockFocus()

        model.stageImage(image, title: "Clipboard image", kind: .image)
        await settleTasks()

        let uploaded = try XCTUnwrap(transport.attachments.first)
        defer { try? FileManager.default.removeItem(at: uploaded) }
        XCTAssertEqual(uploaded.pathExtension, "png")
        XCTAssertTrue(FileManager.default.fileExists(atPath: uploaded.path))
        XCTAssertEqual(model.stagedContext.first?.attachment, transport.attachmentReference)
    }

    @MainActor
    func testSpokenTurnIsDrivenByStreamUpdates() async {
        let (model, transport) = connectedModel()
        model.draft = "Handle this"

        model.send()
        await settleTasks()
        transport.emit(.update(.init(seq: 1, forSeq: 1, state: "understood", kind: "message.inbound")))
        transport.emit(.update(.init(
            seq: 2,
            forSeq: 1,
            state: "complete",
            kind: "turn.completed",
            outcome: "spoke",
            reply: "Done"
        )))
        await settleTasks()

        XCTAssertEqual(model.draft, "")
        XCTAssertEqual(model.messages.map(\.role), [.user, .omega])
        XCTAssertEqual(model.messages.last?.text, "Done")
        XCTAssertEqual(model.messages.first?.delivery, .sent)
        XCTAssertEqual(model.turnState, .complete("Complete"))
    }

    @MainActor
    func testReplayBufferedUntilDuplicateAckMapsToLocalMessage() async {
        let (model, transport) = connectedModel()
        transport.bufferedUpdatesOnNextAck = [
            .init(seq: 1, forSeq: 1, state: "understood", kind: "message.inbound"),
            .init(
                seq: 2,
                forSeq: 1,
                state: "complete",
                kind: "turn.completed",
                outcome: "spoke",
                reply: "Recovered"
            ),
        ]
        model.draft = "Handle this once"

        model.send()
        await settleTasks()

        XCTAssertEqual(model.messages.map(\.role), [.user, .omega])
        XCTAssertEqual(model.messages.last?.text, "Recovered")
        XCTAssertEqual(model.messages.first?.delivery, .sent)
        XCTAssertEqual(model.turnState, .complete("Complete"))
    }

    @MainActor
    func testSilentTurnIsNeitherSpokenNorFailed() async {
        let (model, transport) = connectedModel()
        model.draft = "Only answer if useful"

        model.send()
        await settleTasks()
        transport.emit(.update(.init(
            seq: 2,
            forSeq: 1,
            state: "complete",
            kind: "turn.completed",
            outcome: "silent"
        )))
        await settleTasks()

        XCTAssertFalse(model.messages.contains { $0.role == .omega })
        XCTAssertEqual(model.messages.last?.role, .status)
        XCTAssertEqual(model.messages.last?.text, "omega stayed quiet.")
        XCTAssertEqual(model.turnState, .complete("Complete — no reply needed"))
        XCTAssertFalse(model.canRetryLastSend)
    }

    @MainActor
    func testFailedUpdateRestoresDraftAndContext() async {
        let (model, transport) = connectedModel()
        let context = StagedContext(kind: .text, title: "Do not lose me", detail: "Text · Not sent")
        model.draft = "Try this"
        model.stagedContext = [context]

        model.send()
        await settleTasks()
        XCTAssertEqual(transport.submissions.first?.context.first?.id, context.id)
        XCTAssertEqual(transport.submissions.first?.context.first?.kind, "text")
        transport.emit(.update(.init(
            seq: 2,
            forSeq: 1,
            state: "failed",
            kind: "turn.completed",
            outcome: "failed",
            error: "Provider unavailable"
        )))
        await settleTasks()

        XCTAssertEqual(model.draft, "Try this")
        XCTAssertEqual(model.stagedContext.map(\.id), [context.id])
        XCTAssertEqual(model.messages.first?.delivery, .failed)
        XCTAssertEqual(model.turnState, .failed("Provider unavailable"))
    }

    @MainActor
    func testBlockedAnswerCarriesResumesSequence() async {
        let (model, transport) = connectedModel()
        model.draft = "Start"
        model.send()
        await settleTasks()
        transport.emit(.update(.init(
            seq: 2,
            forSeq: 1,
            state: "blocked",
            kind: "turn.blocked",
            needs: "Which file?"
        )))
        await settleTasks()

        model.draft = "The second one"
        model.send()
        await settleTasks()

        XCTAssertEqual(transport.submissions.last?.resumesSeq, 1)
        XCTAssertEqual(model.messages.last?.text, "The second one")
    }

    @MainActor
    func testDisconnectAfterAckWaitsForResumeBeforeRestoring() async throws {
        let (model, transport) = connectedModel(terminalTimeout: .milliseconds(10))
        let context = StagedContext(kind: .text, title: "Keep me", detail: "Text · Not sent")
        model.draft = "Continue"
        model.stagedContext = [context]
        model.send()
        await settleTasks()

        transport.emit(.disconnected)
        try await Task.sleep(for: .milliseconds(20))
        XCTAssertEqual(model.draft, "", "A socket hiccup after ack must not restore the draft")

        transport.emit(.connected(head: 1))
        try await Task.sleep(for: .milliseconds(20))
        XCTAssertEqual(model.draft, "Continue")
        XCTAssertEqual(model.stagedContext.map(\.id), [context.id])
        XCTAssertTrue(model.canRetryLastSend)
    }

    @MainActor
    func testReplayDoesNotCreateDuplicateBubbleAndPersistsCursor() async {
        var persisted: [Int] = []
        let transport = ScriptedTransport()
        let model = TrayViewModel(
            transport: transport,
            loadCursor: { nil },
            persistCursor: { persisted.append($0) }
        )
        model.start()
        model.connectionState = .connected
        model.isExpanded = true
        transport.emit(.connected(head: 0))
        let update = TrayUpdate(
            seq: 4,
            forSeq: 1,
            state: "complete",
            kind: "turn.completed",
            outcome: "spoke",
            reply: "Arrived"
        )
        transport.emit(.update(update))
        transport.emit(.update(update))
        await settleTasks()

        XCTAssertEqual(model.messages.filter { $0.role == .omega }.count, 1)
        XCTAssertEqual(persisted, [4])
        XCTAssertEqual(transport.resumeCursors, [4])
    }

    @MainActor
    func testUnknownUpdateIsIgnoredButAdvancesCursor() async {
        var cursor: Int?
        let transport = ScriptedTransport()
        let model = TrayViewModel(
            transport: transport,
            loadCursor: { nil },
            persistCursor: { cursor = $0 }
        )
        model.start()
        transport.emit(.update(.init(seq: 9, forSeq: 4, state: "future", kind: "future.kind")))
        await settleTasks()

        XCTAssertTrue(model.messages.isEmpty)
        XCTAssertEqual(cursor, 9)
    }

    @MainActor
    func testUnsolicitedReplyUsesProactivePresentation() async {
        let (model, transport) = connectedModel()
        model.isExpanded = false
        var presentedMessage: String?
        var wasTimeSensitive = false
        model.proactivePresentation = { message, urgency in
            presentedMessage = message
            if case .timeSensitive = urgency { wasTimeSensitive = true }
        }

        transport.emit(.update(.init(
            seq: 1,
            forSeq: 1,
            state: "understood",
            kind: "message.inbound",
            text: "A wake from elsewhere",
            urgency: "timely"
        )))
        transport.emit(.update(.init(
            seq: 2,
            forSeq: 1,
            state: "complete",
            kind: "turn.completed",
            outcome: "spoke",
            reply: "This needs your attention"
        )))
        await settleTasks()

        XCTAssertEqual(presentedMessage, "This needs your attention")
        XCTAssertTrue(wasTimeSensitive)
        XCTAssertFalse(model.messages.contains { $0.role == .omega })
    }

    @MainActor
    func testCaptureStateWinsWhileRemoteWorkContinues() async {
        let (model, transport) = connectedModel()
        model.localWorkState = .working("Choose a region")
        transport.emit(.update(.init(
            seq: 2,
            forSeq: 1,
            state: "working",
            kind: "tool.called",
            tool: "read_file"
        )))
        await settleTasks()

        XCTAssertEqual(model.turnState, .working("Using read_file"))
        XCTAssertEqual(model.workState, .working("Choose a region"))
    }

    @MainActor
    func testCaptureModesKeepAreaAndWindowSelectionExplicit() {
        XCTAssertEqual(TrayViewModel.ScreenCaptureMode.area.arguments, ["-i", "-s", "-x"])
        XCTAssertEqual(TrayViewModel.ScreenCaptureMode.window.arguments, ["-i", "-w", "-x"])
        XCTAssertTrue(TrayViewModel.ScreenCaptureMode.display.arguments.first?.hasPrefix("-D") == true)
    }

    @MainActor
    func testProactiveMessageClearsUnreadStateWhenConsumed() {
        let model = TrayViewModel(transport: ScriptedTransport(), loadCursor: { nil }, persistCursor: { _ in })
        model.hasUnread = true
        model.proactivePeek = "Look at this"

        model.receiveProactiveMessage("Look at this")

        XCTAssertFalse(model.hasUnread)
        XCTAssertNil(model.proactivePeek)
        XCTAssertEqual(model.messages.last?.text, "Look at this")
    }

    @MainActor
    func testPermissionRefreshClearsAResolvedBlock() {
        let permission = PermissionState(granted: false)
        let model = TrayViewModel(
            transport: ScriptedTransport(),
            preflightScreenCaptureAccess: { permission.granted },
            requestScreenCaptureAccess: { permission.granted },
            loadCursor: { nil },
            persistCursor: { _ in }
        )
        model.capturePermission = .denied
        model.localWorkState = .blocked("Screen Recording permission is needed. Open System Settings to allow it.")

        permission.granted = true
        model.refreshScreenCapturePermission()

        XCTAssertEqual(model.capturePermission, .granted)
        XCTAssertNil(model.localWorkState)
    }

    @MainActor
    private func connectedModel(
        terminalTimeout: Duration = .seconds(120)
    ) -> (TrayViewModel, ScriptedTransport) {
        let transport = ScriptedTransport()
        let model = TrayViewModel(
            transport: transport,
            loadCursor: { nil },
            persistCursor: { _ in },
            terminalTimeout: terminalTimeout
        )
        model.start()
        model.connectionState = .connected
        model.isExpanded = true
        transport.emit(.connected(head: 0))
        return (model, transport)
    }

    @MainActor
    private func settleTasks() async {
        for _ in 0..<20 { await Task.yield() }
    }
}

private final class PermissionState {
    var granted: Bool

    init(granted: Bool) {
        self.granted = granted
    }
}

@MainActor
final class ScriptedTransport: TrayTransport {
    private(set) var submissions: [TraySubmission] = []
    private(set) var attachments: [URL] = []
    private(set) var resumeCursors: [Int] = []
    private var nextSequence = 1
    var bufferedUpdatesOnNextAck: [TrayUpdate] = []
    var attachmentError: Error?
    let attachmentReference = TrayAttachmentReference(
        blob: "sha256:" + String(repeating: "a", count: 64),
        mime: "application/octet-stream",
        bytes: 15
    )
    private let continuation: AsyncStream<TrayTransportEvent>.Continuation
    let events: AsyncStream<TrayTransportEvent>

    init() {
        var captured: AsyncStream<TrayTransportEvent>.Continuation!
        events = AsyncStream { captured = $0 }
        continuation = captured
    }

    func start(since: Int?) {}

    func attach(fileAt url: URL) async throws -> TrayAttachmentReference {
        attachments.append(url)
        if let attachmentError { throw attachmentError }
        return attachmentReference
    }

    func send(_ submission: TraySubmission) async throws -> TrayAcknowledgement {
        submissions.append(submission)
        defer { nextSequence += 1 }
        defer { bufferedUpdatesOnNextAck = [] }
        return TrayAcknowledgement(
            seq: nextSequence,
            duplicate: !bufferedUpdatesOnNextAck.isEmpty,
            bufferedUpdates: bufferedUpdatesOnNextAck
        )
    }

    func setResumeCursor(_ seq: Int) {
        resumeCursors.append(seq)
    }

    func stop() {
        continuation.finish()
    }

    func emit(_ event: TrayTransportEvent) {
        continuation.yield(event)
    }
}

private enum AttachmentTestError: Error {
    case refused
}
