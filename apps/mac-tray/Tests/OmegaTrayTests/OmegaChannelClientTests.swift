import Foundation
import Network
import XCTest
@testable import OmegaTray

final class OmegaChannelClientTests: XCTestCase {
    @MainActor
    func testConnectsAndAttachesToRunningOmegaWhenRequested() async throws {
        guard ProcessInfo.processInfo.environment["OMEGA_CHANNEL_INTEGRATION"] == "1" else {
            throw XCTSkip("Set OMEGA_CHANNEL_INTEGRATION=1 with omega listening on port 7717.")
        }

        let suiteName = "OmegaChannelClientTests.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!
        defer { defaults.removePersistentDomain(forName: suiteName) }
        let port = UInt16(ProcessInfo.processInfo.environment["OMEGA_CHANNEL_PORT"] ?? "7717")!
        let client = OmegaChannelClient(port: port, defaults: defaults)
        let connected = expectation(description: "Received hello and subscribed")
        let eventTask = Task {
            for await event in client.events {
                if case .connected = event {
                    connected.fulfill()
                    return
                }
            }
        }

        client.start(since: nil)
        await fulfillment(of: [connected], timeout: 3)

        let file = FileManager.default.temporaryDirectory
            .appendingPathComponent("omega-live-attachment-\(UUID().uuidString).txt")
        let body = Data("live attachment".utf8)
        try body.write(to: file)
        defer { try? FileManager.default.removeItem(at: file) }
        let reference = try await client.attach(fileAt: file)
        XCTAssertTrue(reference.blob.hasPrefix("sha256:"))
        XCTAssertEqual(reference.mime, "text/plain")
        XCTAssertEqual(reference.bytes, body.count)

        client.stop()
        eventTask.cancel()
    }

    func testFramerBuffersPartialLinesAndReturnsCompleteLinesInOrder() throws {
        var framer = OmegaJSONLineFramer()

        XCTAssertEqual(try framer.append(Data("{\"op\":\"hel".utf8)), [])
        let lines = try framer.append(Data("lo\"}\n{\"op\":\"pong\"}\ntail".utf8))

        XCTAssertEqual(lines.map { String(decoding: $0, as: UTF8.self) }, [
            #"{"op":"hello"}"#,
            #"{"op":"pong"}"#,
        ])
        XCTAssertEqual(String(decoding: framer.buffer, as: UTF8.self), "tail")
    }

    func testFramerRejectsAnOversizedUnterminatedLine() {
        var framer = OmegaJSONLineFramer()
        let data = Data(repeating: 0x61, count: OmegaJSONLineFramer.maximumLineBytes + 1)

        XCTAssertThrowsError(try framer.append(data)) { error in
            XCTAssertEqual(error as? OmegaChannelError, .lineTooLong)
        }
    }

    func testFramerRejectsALineWhoseTerminatorCrossesTheCap() {
        var framer = OmegaJSONLineFramer()
        var data = Data(repeating: 0x61, count: OmegaJSONLineFramer.maximumLineBytes)
        data.append(0x0A)

        XCTAssertThrowsError(try framer.append(data)) { error in
            XCTAssertEqual(error as? OmegaChannelError, .lineTooLong)
        }
    }

    func testDemultiplexerHoldsUpdatesUntilRequestCompletes() {
        let update = TrayUpdate(
            seq: 42,
            forSeq: 42,
            state: "understood",
            kind: "message.inbound"
        )
        var demultiplexer = OmegaUpdateDemultiplexer()

        XCTAssertEqual(
            demultiplexer.receive(update, whileRequestInFlight: true),
            []
        )
        XCTAssertEqual(demultiplexer.buffered, [update])
        XCTAssertEqual(demultiplexer.drain(), [update])
        XCTAssertTrue(demultiplexer.buffered.isEmpty)
    }

    func testDemultiplexerRetainsReplayAcrossSubscribeUntilSayAck() {
        let original = TrayUpdate(
            seq: 42,
            forSeq: 42,
            state: "understood",
            kind: "message.inbound"
        )
        let replayedTerminal = TrayUpdate(
            seq: 45,
            forSeq: 42,
            state: "complete",
            kind: "turn.completed",
            outcome: "silent"
        )
        var demultiplexer = OmegaUpdateDemultiplexer()

        demultiplexer.beginSay()
        XCTAssertEqual(
            demultiplexer.receive(original, whileRequestInFlight: true),
            []
        )

        let subscribeBuffer = demultiplexer.drain()
        demultiplexer.retain(subscribeBuffer)
        XCTAssertEqual(
            demultiplexer.receive(replayedTerminal, whileRequestInFlight: false),
            []
        )
        XCTAssertTrue(demultiplexer.isSayPending)
        XCTAssertEqual(demultiplexer.finishSay(), [original, replayedTerminal])
        XCTAssertFalse(demultiplexer.isSayPending)
    }

    func testReconnectBackoffGrowsForRapidDropsAndRemainsBounded() {
        var backoff = OmegaReconnectBackoff()

        XCTAssertEqual(
            (0..<7).map { _ in backoff.nextDelay(afterStableSession: false) },
            [.milliseconds(250), .milliseconds(500), .seconds(1), .seconds(2),
             .seconds(4), .seconds(8), .seconds(8)]
        )
        XCTAssertEqual(backoff.nextDelay(afterStableSession: true), .milliseconds(250))
    }

    @MainActor
    func testWaitingConnectionFailsTheAttemptInsteadOfHanging() {
        let refused = NWError.posix(.ECONNREFUSED)

        XCTAssertNotNil(OmegaChannelClient.readinessFailure(for: .waiting(refused)))
        XCTAssertNil(OmegaChannelClient.readinessFailure(for: .preparing))
        XCTAssertNil(OmegaChannelClient.readinessFailure(for: .ready))
    }

    @MainActor
    func testSayCodecCarriesIdentityAndMapsContextKindToLowercase() throws {
        let submissionID = UUID()
        let contextID = UUID()
        let submission = TraySubmission(
            id: submissionID,
            text: "look here",
            context: [.init(id: contextID, kind: "Screen", title: "Area capture")],
            urgency: "timely",
            resumesSeq: 41
        )

        let line = try OmegaChannelClient.encodeSay(submission)
        XCTAssertEqual(line.last, 0x0A)
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: line.dropLast()) as? [String: Any]
        )
        XCTAssertEqual(object["op"] as? String, "say")
        XCTAssertEqual(object["id"] as? String, submissionID.uuidString)
        XCTAssertEqual(object["channel"] as? String, "tray")
        XCTAssertEqual(object["resumes_seq"] as? Int, 41)
        let context = try XCTUnwrap((object["context"] as? [[String: Any]])?.first)
        XCTAssertEqual(context["id"] as? String, contextID.uuidString)
        XCTAssertEqual(context["kind"] as? String, "screen")
        XCTAssertEqual(context["title"] as? String, "Area capture")
    }

    @MainActor
    func testAttachmentCodecUploadsAPathAndCitesTheReturnedReference() throws {
        let file = URL(fileURLWithPath: "/tmp/omega attachment.png")
        let attachLine = try OmegaChannelClient.encodeAttach(fileAt: file)
        let attach = try XCTUnwrap(
            JSONSerialization.jsonObject(with: attachLine.dropLast()) as? [String: Any]
        )
        XCTAssertEqual(attach["op"] as? String, "attach")
        XCTAssertEqual(attach["path"] as? String, file.path)

        let reference = TrayAttachmentReference(
            blob: "sha256:" + String(repeating: "b", count: 64),
            mime: "image/png",
            bytes: 184_320
        )
        let context = TrayContextReference(
            id: UUID(),
            kind: "screen",
            title: "Area capture",
            attachment: reference
        )
        let sayLine = try OmegaChannelClient.encodeSay(
            TraySubmission(text: "look here", context: [context])
        )
        let say = try XCTUnwrap(
            JSONSerialization.jsonObject(with: sayLine.dropLast()) as? [String: Any]
        )
        let sentContext = try XCTUnwrap((say["context"] as? [[String: Any]])?.first)
        XCTAssertEqual(sentContext["blob"] as? String, reference.blob)
        XCTAssertEqual(sentContext["mime"] as? String, reference.mime)
        XCTAssertEqual(sentContext["bytes"] as? Int, reference.bytes)
    }

    @MainActor
    func testUpdateCodecPreservesSilentOutcome() throws {
        let update = try XCTUnwrap(OmegaChannelClient.decodeUpdate([
            "v": 1,
            "op": "update",
            "seq": 45,
            "state": "complete",
            "for_seq": 42,
            "kind": "turn.completed",
            "outcome": "silent",
            "reply": NSNull(),
        ]))

        XCTAssertEqual(update.outcome, "silent")
        XCTAssertNil(update.reply)
    }

    @MainActor
    func testUpdateCodecCarriesUnknownStateAndKindForCursorAdvancement() throws {
        let base: [String: Any] = [
            "v": 1,
            "op": "update",
            "seq": 45,
            "for_seq": 42,
        ]

        let unknownState = try XCTUnwrap(OmegaChannelClient.decodeUpdate(base.merging([
            "state": "paused",
            "kind": "turn.completed",
        ]) { _, new in new }))
        XCTAssertEqual(unknownState.seq, 45)
        XCTAssertEqual(unknownState.state, "paused")

        let unknownKind = try XCTUnwrap(OmegaChannelClient.decodeUpdate(base.merging([
            "state": "working",
            "kind": "future.kind",
        ]) { _, new in new }))
        XCTAssertEqual(unknownKind.seq, 45)
        XCTAssertEqual(unknownKind.kind, "future.kind")
    }
}
