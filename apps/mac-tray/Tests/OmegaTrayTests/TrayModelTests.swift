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
}

private struct ImmediateTransport: TrayTransport {
    func send(_ submission: TraySubmission) async throws -> String {
        "Done"
    }
}
