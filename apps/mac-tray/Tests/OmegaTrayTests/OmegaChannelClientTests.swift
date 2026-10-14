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
    func testPutCodecSendsBase64ChunksAtTheirOffsets() throws {
        let chunk = OmegaPutUpload.maximumChunkBytes
        XCTAssertEqual(chunk, 512 * 1024)
        let data = Data((0..<(2 * chunk + 100)).map { UInt8(truncatingIfNeeded: $0 &* 31) })
        let lines = try OmegaPutUpload.lines(upload: "up-1", name: "omega attachment.png", data: data)

        XCTAssertEqual(lines.count, 3)
        var rebuilt = Data()
        for (index, line) in lines.enumerated() {
            XCTAssertEqual(line.last, 0x0A)
            XCTAssertLessThan(line.count, OmegaJSONLineFramer.maximumLineBytes)
            let object = try XCTUnwrap(
                JSONSerialization.jsonObject(with: line.dropLast()) as? [String: Any]
            )
            XCTAssertEqual(object["op"] as? String, "put")
            XCTAssertNil(object["path"])
            XCTAssertEqual(object["upload"] as? String, "up-1")
            XCTAssertEqual(object["name"] as? String, "omega attachment.png")
            XCTAssertEqual(object["offset"] as? Int, index * chunk)
            XCTAssertEqual(object["last"] as? Bool, index == 2)
            let bytes = try XCTUnwrap(Data(base64Encoded: try XCTUnwrap(object["data"] as? String)))
            XCTAssertEqual(bytes.count, index == 2 ? 100 : chunk)
            rebuilt.append(bytes)
        }
        XCTAssertEqual(rebuilt, data)

        let empty = try OmegaPutUpload.lines(upload: "up-2", name: "empty.txt", data: Data())
        XCTAssertEqual(empty.count, 1)
        let only = try XCTUnwrap(
            JSONSerialization.jsonObject(with: empty[0].dropLast()) as? [String: Any]
        )
        XCTAssertEqual(only["offset"] as? Int, 0)
        XCTAssertEqual(only["data"] as? String, "")
        XCTAssertEqual(only["last"] as? Bool, true)
        XCTAssertEqual(OmegaPutUpload.fileName(for: URL(fileURLWithPath: "/tmp/a b.pdf")), "a b.pdf")
    }

    @MainActor
    func testOverCapFileIsRefusedBeforeAnythingIsSent() throws {
        XCTAssertEqual(OmegaPutUpload.maximumFileBytes, 25 * 1024 * 1024)
        let file = FileManager.default.temporaryDirectory
            .appendingPathComponent("omega-over-cap-\(UUID().uuidString).bin")
        XCTAssertTrue(FileManager.default.createFile(atPath: file.path, contents: nil))
        defer { try? FileManager.default.removeItem(at: file) }
        let handle = try FileHandle(forWritingTo: file)
        try handle.truncate(atOffset: UInt64(OmegaPutUpload.maximumFileBytes + 1))
        try handle.close()

        XCTAssertThrowsError(try OmegaPutUpload.read(fileAt: file)) { error in
            XCTAssertEqual(error as? OmegaChannelError, .overCap)
            XCTAssertEqual(error.localizedDescription, "Over 25 MB, not sent")
        }
        XCTAssertThrowsError(try OmegaPutUpload.lines(
            upload: "up", name: "big.bin",
            data: Data(count: OmegaPutUpload.maximumFileBytes + 1)
        )) { error in
            XCTAssertEqual(error as? OmegaChannelError, .overCap)
        }

        let handle2 = try FileHandle(forWritingTo: file)
        try handle2.truncate(atOffset: UInt64(OmegaPutUpload.maximumFileBytes))
        try handle2.close()
        XCTAssertEqual(try OmegaPutUpload.read(fileAt: file).count, OmegaPutUpload.maximumFileBytes)
    }

    @MainActor
    func testRetryAfterADroppedChunkResendsWithoutCorruptingTheFile() async throws {
        let chunk = OmegaPutUpload.maximumChunkBytes
        let data = Data((0..<(2 * chunk + 7)).map { UInt8(truncatingIfNeeded: $0 &* 13 &+ 5) })
        let lines = try OmegaPutUpload.lines(upload: "up-retry", name: "notes.txt", data: data)
        let core = FakePutCore(dropAfterWritingRequest: 2)

        let reference = try await OmegaPutUpload.send(
            lines,
            isRunning: { true },
            request: { try core.handle($0) }
        )

        XCTAssertEqual(core.requests, 5)
        XCTAssertEqual(core.stored, data)
        XCTAssertEqual(reference.blob, FakePutCore.digest)
        XCTAssertEqual(reference.mime, "text/plain")
        XCTAssertEqual(reference.bytes, data.count)
    }

    @MainActor
    func testPutRefusalFromTheCoreReachesTheTrayAsItsSentence() async throws {
        let lines = try OmegaPutUpload.lines(upload: "up", name: "a.txt", data: Data("x".utf8))
        do {
            _ = try await OmegaPutUpload.send(lines, isRunning: { true }, request: { _ in
                ["v": 1, "op": "error", "error": "a file is at most 25 MB"]
            })
            XCTFail("expected a refusal")
        } catch {
            XCTAssertEqual(error as? OmegaChannelError, .server("a file is at most 25 MB"))
        }
    }

    @MainActor
    func testPutReferenceLandsOnTheContextItemUnchanged() async throws {
        let core = FakePutCore(dropAfterWritingRequest: nil)
        let data = Data(repeating: 0x42, count: 184_320)
        let reference = try await OmegaPutUpload.send(
            try OmegaPutUpload.lines(upload: "up-ref", name: "omega attachment.png", data: data),
            isRunning: { true },
            request: { try core.handle($0) }
        )
        XCTAssertEqual(reference, TrayAttachmentReference(
            blob: FakePutCore.digest, mime: "image/png", bytes: 184_320
        ))
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
    func testUpdateCodecCarriesTheInboundChannel() throws {
        let update = try XCTUnwrap(OmegaChannelClient.decodeUpdate([
            "v": 1,
            "op": "update",
            "seq": 42,
            "state": "understood",
            "for_seq": 42,
            "kind": "message.inbound",
            "text": "hello from Discord",
            "channel": "discord",
        ]))

        XCTAssertEqual(update.channel, "discord")
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

/// The core's `put` contract, in memory: each chunk is written at its offset and the file
/// truncated after it, an offset past what arrived is refused, and the last chunk answers
/// `attached`. It can drop the connection after writing one request, as a lost ack would.
@MainActor
private final class FakePutCore {
    static let digest = "sha256:" + String(repeating: "c", count: 64)

    private(set) var requests = 0
    private(set) var stored = Data()
    private var staged: [String: Data] = [:]
    private var dropAfterWritingRequest: Int?

    init(dropAfterWritingRequest: Int?) {
        self.dropAfterWritingRequest = dropAfterWritingRequest
    }

    func handle(_ line: Data) throws -> [String: Any] {
        requests += 1
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: line.dropLast()) as? [String: Any]
        )
        XCTAssertEqual(object["op"] as? String, "put")
        let upload = try XCTUnwrap(object["upload"] as? String)
        let offset = try XCTUnwrap(object["offset"] as? Int)
        let chunk = try XCTUnwrap(Data(base64Encoded: try XCTUnwrap(object["data"] as? String)))
        XCTAssertLessThanOrEqual(chunk.count, OmegaPutUpload.maximumChunkBytes)
        var file = staged[upload] ?? Data()
        guard offset <= file.count else {
            return ["v": 1, "op": "error", "error": "chunk at \(offset) but \(file.count) bytes arrived so far"]
        }
        file = file.prefix(offset) + chunk
        staged[upload] = file
        if requests == dropAfterWritingRequest {
            dropAfterWritingRequest = nil
            throw OmegaChannelError.disconnected
        }
        guard object["last"] as? Bool == true else {
            return ["v": 1, "op": "put", "upload": upload, "bytes": offset + chunk.count]
        }
        stored = file
        staged[upload] = nil
        let name = object["name"] as? String ?? ""
        let mime = name.hasSuffix(".png") ? "image/png" : name.hasSuffix(".txt") ? "text/plain"
            : "application/octet-stream"
        return ["v": 1, "op": "attached", "blob": Self.digest, "mime": mime, "bytes": file.count]
    }
}
