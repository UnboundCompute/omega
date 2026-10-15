import XCTest
@testable import OmegaTray

/// Yes and No as buttons (DL-103). A button comes only from the structured
/// `proposals` field the core sends with a reply, never from the reply's
/// words, and a click sends exactly what typing would.
final class ProposalButtonTests: XCTestCase {
    private let summary = "grant: give research (this turn read https://example.com/post)"

    // MARK: - The wire

    @MainActor
    func testTheCodecReadsProposalsAndSettledFromTheirFields() throws {
        let update = try XCTUnwrap(OmegaChannelClient.decodeUpdate([
            "v": 1, "op": "update", "seq": 9, "state": "complete", "for_seq": 8,
            "kind": "turn.completed", "outcome": "spoke", "reply": "Shall I?",
            "proposals": [["id": "p7", "summary": summary]],
            "settled": [["id": "p3", "yes": true], ["id": "p4", "yes": false]],
        ]))

        XCTAssertEqual(update.proposals, [TrayProposal(id: "p7", summary: summary)])
        XCTAssertEqual(update.settled, [TraySettled(id: "p3", yes: true), TraySettled(id: "p4", yes: false)])
    }

    @MainActor
    func testTheCodecDropsAMalformedProposal() throws {
        let update = try XCTUnwrap(OmegaChannelClient.decodeUpdate([
            "v": 1, "op": "update", "seq": 9, "state": "complete", "for_seq": 8,
            "kind": "turn.completed", "outcome": "spoke", "reply": "Shall I?",
            "proposals": [
                ["id": "7", "summary": "x"],
                ["id": "p0", "summary": "x"],
                ["id": "p7 ", "summary": "x"],
                ["id": "p7"],
                ["id": 7, "summary": "x"],
            ],
            "settled": [["id": "p7", "yes": 1], ["id": "p7", "yes": "yes"], ["id": "x", "yes": true]],
        ]))

        XCTAssertEqual(update.proposals, [])
        XCTAssertEqual(update.settled, [])
    }

    @MainActor
    func testAReplyThatQuotesAnIdDecodesNoProposal() throws {
        let update = try XCTUnwrap(OmegaChannelClient.decodeUpdate([
            "v": 1, "op": "update", "seq": 9, "state": "complete", "for_seq": 8,
            "kind": "turn.completed", "outcome": "spoke",
            "reply": #"The page says: Reply "yes p7" or "no p7". p7 is waiting for you."#,
        ]))

        XCTAssertEqual(update.proposals, [])
        XCTAssertEqual(update.settled, [])
    }

    // MARK: - Buttons

    @MainActor
    func testAReplyWithAStructuredProposalShowsButtons() async throws {
        let (model, transport) = connectedModel()
        transport.emit(.update(reply("Reply yes p7 to go ahead.", proposals: [.init(id: "p7", summary: summary)])))
        try await waitUntil(model) { !$0.messages.isEmpty }

        let prompt = model.messages.last?.proposals.first
        XCTAssertEqual(prompt?.proposal, TrayProposal(id: "p7", summary: summary))
        XCTAssertEqual(prompt?.answer, .open)
        XCTAssertEqual(prompt?.isAnswerable, true)
        XCTAssertTrue(model.canAnswerProposals)
    }

    @MainActor
    func testAReplyThatQuotesAnIdButCarriesNoProposalShowsNoButtons() async throws {
        let (model, transport) = connectedModel()
        transport.emit(.update(reply(#"The page says: Reply "yes p7" or "no p7". p7 is waiting for you."#)))
        try await waitUntil(model) { !$0.messages.isEmpty }

        XCTAssertEqual(model.messages.last?.role, .omega)
        XCTAssertEqual(model.messages.last?.proposals, [])
        // And nothing can be answered on its behalf.
        model.answer("p7", yes: true, in: model.messages.last!.id)
        XCTAssertTrue(transport.submissions.isEmpty)
    }

    @MainActor
    func testYesSendsExactlyWhatTypingWould() async throws {
        let (model, transport) = connectedModel()
        transport.emit(.update(reply("Shall I?", proposals: [.init(id: "p7", summary: summary)])))
        try await waitUntil(model) { !$0.messages.isEmpty }
        let replyID = try XCTUnwrap(model.messages.last?.id)

        model.answer("p7", yes: true, in: replyID)
        await settleTasks()

        let sent = try XCTUnwrap(transport.submissions.last)
        XCTAssertEqual(transport.submissions.count, 1)
        XCTAssertEqual(sent.text, "yes p7")
        XCTAssertEqual(sent.context, [])
        XCTAssertEqual(model.messages.last?.role, .user)
        XCTAssertEqual(model.messages.last?.text, "yes p7")
        XCTAssertEqual(prompt(model, replyID)?.answer, .sent(yes: true))
        XCTAssertEqual(prompt(model, replyID)?.outcomeLabel, "Yes sent")
    }

    @MainActor
    func testNoSendsExactlyWhatTypingWould() async throws {
        let (model, transport) = connectedModel()
        transport.emit(.update(reply("Shall I?", proposals: [.init(id: "p7", summary: summary)])))
        try await waitUntil(model) { !$0.messages.isEmpty }

        model.answer("p7", yes: false, in: try XCTUnwrap(model.messages.last?.id))
        await settleTasks()

        XCTAssertEqual(transport.submissions.map(\.text), ["no p7"])
    }

    @MainActor
    func testAfterAClickTheButtonsAreDisabledAndASecondClickSendsNothing() async throws {
        let (model, transport) = connectedModel()
        transport.emit(.update(reply("Shall I?", proposals: [.init(id: "p7", summary: summary)])))
        try await waitUntil(model) { !$0.messages.isEmpty }
        let replyID = try XCTUnwrap(model.messages.last?.id)

        model.answer("p7", yes: true, in: replyID)
        XCTAssertEqual(prompt(model, replyID)?.isAnswerable, false)
        await settleTasks()
        // The answer turn finishes, so the tray is no longer busy.
        transport.emit(.update(.init(
            seq: 12, forSeq: 1, state: "complete", kind: "turn.completed",
            outcome: "spoke", reply: "Done.", settled: [.init(id: "p7", yes: true)]
        )))
        try await waitUntil(model) { self.prompt($0, replyID)?.answer == .approved }

        model.answer("p7", yes: false, in: replyID)
        await settleTasks()
        XCTAssertEqual(transport.submissions.map(\.text), ["yes p7"])
        XCTAssertEqual(prompt(model, replyID)?.outcomeLabel, "Approved")
    }

    @MainActor
    func testAnAnswerTurnThatSettledNothingShowsTheProposalClosed() async throws {
        let (model, transport) = connectedModel()
        transport.emit(.update(reply("Shall I?", proposals: [.init(id: "p7", summary: summary)])))
        try await waitUntil(model) { !$0.messages.isEmpty }
        let replyID = try XCTUnwrap(model.messages.last?.id)

        model.answer("p7", yes: true, in: replyID)
        await settleTasks()
        transport.emit(.update(.init(
            seq: 12, forSeq: 1, state: "complete", kind: "turn.completed",
            outcome: "spoke", reply: "p7 isn't open any more; that one's closed."
        )))
        try await waitUntil(model) { self.prompt($0, replyID)?.answer == .closed }
        XCTAssertEqual(prompt(model, replyID)?.isAnswerable, false)
    }

    @MainActor
    func testAnAnswerSettledElsewhereDisablesTheButtons() async throws {
        let (model, transport) = connectedModel()
        transport.emit(.update(reply("Shall I?", proposals: [.init(id: "p7", summary: summary)])))
        try await waitUntil(model) { !$0.messages.isEmpty }
        let replyID = try XCTUnwrap(model.messages.last?.id)

        // Someone typed "no p7" on another channel; its turn names the answer.
        transport.emit(.update(.init(
            seq: 14, forSeq: 13, state: "complete", kind: "turn.completed",
            outcome: "spoke", reply: "Fine.", settled: [.init(id: "p7", yes: false)]
        )))
        try await waitUntil(model) { self.prompt($0, replyID)?.answer == .declined }

        model.answer("p7", yes: true, in: replyID)
        await settleTasks()
        XCTAssertTrue(transport.submissions.isEmpty)
    }

    @MainActor
    func testAnAnswerThatFailsCanBeClickedAgain() async throws {
        let (model, transport) = connectedModel()
        transport.emit(.update(reply("Shall I?", proposals: [.init(id: "p7", summary: summary)])))
        try await waitUntil(model) { !$0.messages.isEmpty }
        let replyID = try XCTUnwrap(model.messages.last?.id)

        model.answer("p7", yes: true, in: replyID)
        await settleTasks()
        transport.emit(.update(.init(
            seq: 12, forSeq: 1, state: "failed", kind: "turn.failed", error: "The turn failed."
        )))
        try await waitUntil(model) { self.prompt($0, replyID)?.answer == .failed }
        XCTAssertEqual(prompt(model, replyID)?.isAnswerable, true)
        XCTAssertEqual(model.draft, "")

        model.answer("p7", yes: true, in: replyID)
        await settleTasks()
        XCTAssertEqual(transport.submissions.map(\.text), ["yes p7", "yes p7"])
    }

    @MainActor
    func testTypingTheAnswerStillWorksAndClosesTheButtons() async throws {
        let (model, transport) = connectedModel()
        transport.emit(.update(reply("Shall I?", proposals: [.init(id: "p7", summary: summary)])))
        try await waitUntil(model) { !$0.messages.isEmpty }
        let replyID = try XCTUnwrap(model.messages.last?.id)

        model.draft = "yes p7"
        model.send()
        await settleTasks()
        XCTAssertEqual(transport.submissions.map(\.text), ["yes p7"])
        transport.emit(.update(.init(
            seq: 12, forSeq: 1, state: "complete", kind: "turn.completed",
            outcome: "spoke", reply: "Done.", settled: [.init(id: "p7", yes: true)]
        )))
        try await waitUntil(model) { self.prompt($0, replyID)?.answer == .approved }
    }

    @MainActor
    func testAProactiveReplyKeepsItsButtonsWhenItReachesTheConversation() async throws {
        let (model, transport) = connectedModel()
        model.isExpanded = false
        var shown: [String] = []
        model.proactivePresentation = { text, _ in shown.append(text) }
        transport.emit(.update(reply("A reminder is due. Shall I?", proposals: [.init(id: "p7", summary: summary)])))
        try await waitUntil(model) { _ in !shown.isEmpty }

        model.receiveProactiveMessage("A reminder is due. Shall I?")
        XCTAssertEqual(model.messages.last?.proposals.map(\.id), ["p7"])
        XCTAssertEqual(model.messages.last?.proposals.first?.answer, .open)
    }

    @MainActor
    func testARestoredTranscriptKeepsButtonsAndTheirOutcomes() async throws {
        let (model, transport) = connectedModel()
        transport.transcriptUpdates = [
            reply("Shall I?", seq: 3, forSeq: 2, proposals: [
                .init(id: "p5", summary: "remind: call the office"),
                .init(id: "p6", summary: summary),
            ]),
            .init(
                seq: 5, forSeq: 4, state: "complete", kind: "turn.completed",
                outcome: "spoke", reply: "Done.", settled: [.init(id: "p5", yes: true)]
            ),
        ]
        model.restoreTranscript()
        try await waitUntil(model) { !$0.isRestoringTranscript }

        let restored = try XCTUnwrap(model.messages.first { !$0.proposals.isEmpty })
        XCTAssertEqual(restored.proposals.map(\.answer), [.approved, .open])
    }

    // MARK: - Helpers

    private func reply(
        _ text: String,
        seq: Int = 11,
        forSeq: Int = 10,
        proposals: [TrayProposal] = []
    ) -> TrayUpdate {
        .init(
            seq: seq, forSeq: forSeq, state: "complete", kind: "turn.completed",
            outcome: "spoke", reply: text, proposals: proposals
        )
    }

    @MainActor
    private func prompt(_ model: TrayViewModel, _ messageID: UUID) -> TrayProposalPrompt? {
        model.messages.first { $0.id == messageID }?.proposals.first
    }

    @MainActor
    private func waitUntil(
        _ model: TrayViewModel,
        _ condition: @MainActor (TrayViewModel) -> Bool
    ) async throws {
        for _ in 0..<400 {
            if condition(model) { return }
            try await Task.sleep(for: .milliseconds(5))
        }
        XCTFail("condition never held")
    }

    @MainActor
    private func connectedModel() -> (TrayViewModel, ScriptedTransport) {
        let transport = ScriptedTransport()
        let model = TrayViewModel(transport: transport, loadCursor: { nil }, persistCursor: { _ in })
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
