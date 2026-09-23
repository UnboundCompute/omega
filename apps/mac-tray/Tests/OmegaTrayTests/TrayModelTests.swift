import XCTest
@testable import OmegaTray

final class TrayModelTests: XCTestCase {
    @MainActor
    func testCannotSendAnEmptySubmission() {
        let model = TrayViewModel(transport: ImmediateTransport())
        XCTAssertFalse(model.canSend)
    }

    @MainActor
    func testTextMakesSubmissionSendable() {
        let model = TrayViewModel(transport: ImmediateTransport())
        model.draft = "Handle this"
        XCTAssertTrue(model.canSend)
    }

    @MainActor
    func testSendClearsDraftAndAddsUserMessage() async {
        let model = TrayViewModel(transport: ImmediateTransport())
        model.draft = "Handle this"

        model.send()
        await Task.yield()

        XCTAssertEqual(model.draft, "")
        XCTAssertEqual(model.messages.first?.role, .user)
        XCTAssertEqual(model.messages.first?.text, "Handle this")
    }

    @MainActor
    func testCaptureModesKeepAreaAndWindowSelectionExplicit() {
        XCTAssertEqual(
            TrayViewModel.ScreenCaptureMode.area.arguments,
            ["-i", "-s", "-x"]
        )
        XCTAssertEqual(
            TrayViewModel.ScreenCaptureMode.window.arguments,
            ["-i", "-w", "-x"]
        )
        XCTAssertTrue(
            TrayViewModel.ScreenCaptureMode.display.arguments.first?.hasPrefix("-D") == true
        )
    }

    @MainActor
    func testSentMessageKeepsItsContextReceipt() async {
        let model = TrayViewModel(transport: ImmediateTransport())
        model.draft = "Use this"
        model.stagedContext = [
            StagedContext(kind: .text, title: "Meeting notes", detail: "Text · Not sent")
        ]

        model.send()
        await settleTasks()

        XCTAssertEqual(model.messages.first?.contextDescriptions, ["Meeting notes"])
        XCTAssertEqual(model.messages.first?.delivery, .sent)
        XCTAssertTrue(model.stagedContext.isEmpty)
    }

    @MainActor
    func testFailedSendRestoresDraftAndContext() async {
        let model = TrayViewModel(transport: FailingTransport())
        let context = StagedContext(kind: .text, title: "Do not lose me", detail: "Text · Not sent")
        model.draft = "Try this"
        model.stagedContext = [context]

        model.send()
        await settleTasks()

        XCTAssertEqual(model.draft, "Try this")
        XCTAssertEqual(model.stagedContext.map(\.id), [context.id])
        XCTAssertEqual(model.messages.first?.delivery, .failed)
        guard case .failed = model.workState else {
            return XCTFail("Expected a recoverable failure state")
        }
    }

    @MainActor
    func testProactiveMessageClearsUnreadStateWhenConsumed() {
        let model = TrayViewModel(transport: ImmediateTransport())
        model.hasUnread = true
        model.proactivePeek = "Look at this"

        model.receiveProactiveMessage("Look at this")

        XCTAssertFalse(model.hasUnread)
        XCTAssertNil(model.proactivePeek)
        XCTAssertEqual(model.messages.last?.text, "Look at this")
    }

    @MainActor
    private func settleTasks() async {
        for _ in 0..<12 { await Task.yield() }
    }
}

private struct ImmediateTransport: TrayTransport {
    func send(_ submission: TraySubmission) async throws -> String {
        "Done"
    }
}

private struct FailingTransport: TrayTransport {
    struct DeliveryError: Error {}

    func send(_ submission: TraySubmission) async throws -> String {
        throw DeliveryError()
    }
}
