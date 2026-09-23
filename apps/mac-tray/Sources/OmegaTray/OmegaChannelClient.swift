import Foundation
import Network

enum OmegaChannelError: Error, Equatable, LocalizedError {
    case stopped
    case disconnected
    case invalidGreeting
    case invalidMessage(String)
    case lineTooLong
    case server(String)
    case conflict(String)

    var errorDescription: String? {
        switch self {
        case .stopped: "Omega channel stopped"
        case .disconnected: "Omega channel disconnected"
        case .invalidGreeting: "Omega channel sent an invalid greeting"
        case .invalidMessage(let detail): "Invalid Omega channel message: \(detail)"
        case .lineTooLong: "Omega channel line exceeded 1 MiB"
        case .server(let detail): detail
        case .conflict(let detail): detail
        }
    }
}

struct OmegaJSONLineFramer {
    static let maximumLineBytes = 1 << 20

    private(set) var buffer = Data()

    mutating func append(_ data: Data) throws -> [Data] {
        buffer.append(data)
        var lines: [Data] = []

        while let newline = buffer.firstIndex(of: 0x0A) {
            let line = buffer[..<newline]
            guard line.count < Self.maximumLineBytes else {
                throw OmegaChannelError.lineTooLong
            }
            lines.append(Data(line))
            buffer.removeSubrange(...newline)
        }

        guard buffer.count <= Self.maximumLineBytes else {
            throw OmegaChannelError.lineTooLong
        }
        return lines
    }
}

struct OmegaUpdateDemultiplexer {
    private(set) var buffered: [TrayUpdate] = []
    private(set) var isSayPending = false

    mutating func beginSay() {
        precondition(!isSayPending)
        isSayPending = true
    }

    mutating func receive(_ update: TrayUpdate, whileRequestInFlight: Bool) -> [TrayUpdate] {
        guard isSayPending || whileRequestInFlight else { return [update] }
        buffered.append(update)
        return []
    }

    mutating func retain(_ updates: [TrayUpdate]) {
        buffered.append(contentsOf: updates)
    }

    mutating func finishSay() -> [TrayUpdate] {
        isSayPending = false
        return drain()
    }

    mutating func drain() -> [TrayUpdate] {
        defer { buffered.removeAll(keepingCapacity: true) }
        return buffered
    }

    mutating func discard() {
        buffered.removeAll(keepingCapacity: true)
    }
}

struct OmegaReconnectBackoff {
    private(set) var failures = 0

    mutating func nextDelay(afterStableSession stable: Bool) -> Duration {
        if stable { failures = 0 }
        let exponent = min(failures, 5)
        failures = min(failures + 1, 5)
        return .milliseconds(min(250 * (1 << exponent), 8_000))
    }
}

@MainActor
final class OmegaChannelClient: TrayTransport {
    let events: AsyncStream<TrayTransportEvent>

    private let host: NWEndpoint.Host
    private let port: NWEndpoint.Port
    private let defaults: UserDefaults
    private let cursorKey: String
    private let eventContinuation: AsyncStream<TrayTransportEvent>.Continuation
    private let clock = ContinuousClock()

    private var connection: NWConnection?
    private var connectionID: UUID?
    private var connectionTask: Task<Void, Never>?
    private var receiveTask: Task<Void, Never>?
    private var framer = OmegaJSONLineFramer()
    private var updateDemultiplexer = OmegaUpdateDemultiplexer()
    private var isRunning = false
    private var isSubscribed = false
    private var sessionConnectedAt: ContinuousClock.Instant?
    private var emittedDisconnected = false
    private var requestedCursor: Int?

    private var connectionWaiters: [UUID: CheckedContinuation<Void, Error>] = [:]
    private var readyWaiter: (id: UUID, continuation: CheckedContinuation<Void, Error>)?
    private var sessionWaiter: CheckedContinuation<Void, Error>?
    private var requestWaiters: [CheckedContinuation<Void, Never>] = []
    private var requestInFlight = false
    private var sayWaiters: [CheckedContinuation<Void, Never>] = []
    private var sayInFlight = false
    private var pendingResponse: CheckedContinuation<[String: Any], Error>?

    init(
        host: String = "127.0.0.1",
        port: UInt16 = 7717,
        defaults: UserDefaults = .standard,
        cursorKey: String = "projectionCursor"
    ) {
        self.host = NWEndpoint.Host(host)
        self.port = NWEndpoint.Port(rawValue: port)!
        self.defaults = defaults
        self.cursorKey = cursorKey

        var continuation: AsyncStream<TrayTransportEvent>.Continuation!
        events = AsyncStream(bufferingPolicy: .unbounded) { continuation = $0 }
        eventContinuation = continuation
    }

    deinit {
        connectionTask?.cancel()
        receiveTask?.cancel()
        connection?.cancel()
        eventContinuation.finish()
    }

    func start(since: Int?) {
        guard !isRunning else { return }

        requestedCursor = since ?? persistedCursor
        isRunning = true
        connectionTask = Task { [weak self] in
            await self?.runConnectionLoop()
        }
    }

    func send(_ submission: TraySubmission) async throws -> TrayAcknowledgement {
        let line = try Self.encodeSay(submission)
        await acquireSaySlot()
        defer { releaseSaySlot() }
        try Task.checkCancellation()
        updateDemultiplexer.beginSay()

        do {
            while isRunning {
                try Task.checkCancellation()
                try await waitUntilSubscribed()

                do {
                    let (response, requestUpdates) = try await request(line)
                    var bufferedUpdates = requestUpdates
                    bufferedUpdates.append(contentsOf: updateDemultiplexer.finishSay())
                    guard Self.integer(response["v"]) == 1,
                          response["op"] as? String == "ack",
                          let seq = Self.integer(response["seq"]),
                          let duplicate = response["duplicate"] as? Bool
                    else {
                        yieldUpdates(bufferedUpdates)
                        throw Self.responseError(response, expected: "ack")
                    }
                    if let conflict = response["conflict"] as? String {
                        yieldUpdates(bufferedUpdates)
                        throw OmegaChannelError.conflict(conflict)
                    }
                    return TrayAcknowledgement(
                        seq: seq,
                        duplicate: duplicate,
                        bufferedUpdates: bufferedUpdates
                    )
                } catch OmegaChannelError.disconnected {
                    // The server may have appended before the socket dropped. Reusing the
                    // encoded line preserves the submission id and makes the retry idempotent.
                    continue
                }
            }
            throw OmegaChannelError.stopped
        } catch {
            if updateDemultiplexer.isSayPending {
                yieldUpdates(updateDemultiplexer.finishSay())
            }
            throw error
        }
    }

    func setResumeCursor(_ seq: Int) {
        guard seq >= 0 else { return }
        if let current = requestedCursor, seq < current { return }
        storeCursor(seq)
    }

    func stop() {
        guard isRunning else { return }
        isRunning = false
        connectionTask?.cancel()
        connectionTask = nil
        receiveTask?.cancel()
        receiveTask = nil
        connection?.cancel()
        connection = nil
        connectionID = nil
        isSubscribed = false

        failPending(OmegaChannelError.stopped)
        failReadyWaiter(OmegaChannelError.stopped)
        resumeConnectionWaiters(throwing: OmegaChannelError.stopped)
        sessionWaiter?.resume(throwing: OmegaChannelError.stopped)
        sessionWaiter = nil
        eventContinuation.finish()
    }

    private var persistedCursor: Int? {
        guard defaults.object(forKey: cursorKey) != nil else { return nil }
        return defaults.integer(forKey: cursorKey)
    }

    private func storeCursor(_ seq: Int) {
        requestedCursor = seq
        defaults.set(seq, forKey: cursorKey)
    }

    private func runConnectionLoop() async {
        var backoff = OmegaReconnectBackoff()

        while isRunning && !Task.isCancelled {
            do {
                try await runSession()
            } catch is CancellationError {
                break
            } catch OmegaChannelError.stopped {
                break
            } catch {
                let wasStable = sessionConnectedAt.map {
                    clock.now - $0 >= .seconds(10)
                } ?? false
                disconnectCurrentConnection(error)
                guard isRunning && !Task.isCancelled else { break }

                try? await Task.sleep(for: backoff.nextDelay(afterStableSession: wasStable))
            }
        }
    }

    private func runSession() async throws {
        sessionConnectedAt = nil
        let id = UUID()
        let newConnection = NWConnection(host: host, port: port, using: .tcp)
        connection = newConnection
        connectionID = id
        framer = OmegaJSONLineFramer()

        try await waitForReady(newConnection, id: id)
        let greeting = try await receiveGreeting(on: newConnection, id: id)
        guard greeting["op"] as? String == "hello",
              Self.integer(greeting["v"]) == 1,
              let head = Self.integer(greeting["head"])
        else {
            throw OmegaChannelError.invalidGreeting
        }

        receiveTask = Task { [weak self] in
            await self?.receiveLoop(on: newConnection, id: id)
        }

        var cursor = requestedCursor ?? head
        if cursor > head {
            cursor = head
            storeCursor(head)
        }
        requestedCursor = cursor

        do {
            try await subscribe(since: cursor)
        } catch let OmegaChannelError.server(message) where message.contains("ahead of head") {
            cursor = head
            storeCursor(head)
            try await subscribe(since: head)
        }

        guard connectionID == id else { throw OmegaChannelError.disconnected }
        isSubscribed = true
        sessionConnectedAt = clock.now
        emittedDisconnected = false
        eventContinuation.yield(.connected(head: head))
        resumeConnectionWaiters()

        try await withCheckedThrowingContinuation { continuation in
            sessionWaiter = continuation
        }
    }

    private func subscribe(since: Int) async throws {
        let line = try Self.encodeObject(["op": "subscribe", "since": since])
        let (response, bufferedUpdates) = try await request(line, allowsUnsubscribed: true)
        guard Self.integer(response["v"]) == 1,
              response["op"] as? String == "subscribed"
        else {
            throw Self.responseError(response, expected: "subscribed")
        }
        if updateDemultiplexer.isSayPending {
            updateDemultiplexer.retain(bufferedUpdates)
        } else {
            yieldUpdates(bufferedUpdates)
        }
    }

    private func waitForReady(_ connection: NWConnection, id: UUID) async throws {
        try await withCheckedThrowingContinuation { continuation in
            readyWaiter = (id, continuation)
            connection.stateUpdateHandler = { [weak self] state in
                Task { @MainActor [weak self] in
                    guard let self, self.connectionID == id else { return }
                    switch state {
                    case .ready:
                        self.completeReadyWaiter(id: id)
                        connection.stateUpdateHandler = nil
                    default:
                        if let error = Self.readinessFailure(for: state) {
                            self.completeReadyWaiter(id: id, throwing: error)
                            connection.stateUpdateHandler = nil
                        }
                    }
                }
            }
            connection.start(queue: .global(qos: .userInitiated))
        }
    }

    static func readinessFailure(for state: NWConnection.State) -> Error? {
        switch state {
        case .waiting(let error), .failed(let error):
            error
        case .cancelled:
            OmegaChannelError.disconnected
        default:
            nil
        }
    }

    private func receiveGreeting(on connection: NWConnection, id: UUID) async throws -> [String: Any] {
        while connectionID == id {
            let (data, complete) = try await receiveChunk(on: connection)
            for line in try framer.append(data) where !line.isEmpty {
                return try Self.decodeObject(line)
            }
            if complete { throw OmegaChannelError.disconnected }
        }
        throw OmegaChannelError.disconnected
    }

    private func receiveLoop(on connection: NWConnection, id: UUID) async {
        do {
            while isRunning && connectionID == id && !Task.isCancelled {
                let (data, complete) = try await receiveChunk(on: connection)
                for line in try framer.append(data) where !line.isEmpty {
                    try handle(try Self.decodeObject(line))
                }
                if complete { throw OmegaChannelError.disconnected }
            }
        } catch {
            guard connectionID == id else { return }
            disconnectCurrentConnection(error)
        }
    }

    private func receiveChunk(on connection: NWConnection) async throws -> (Data, Bool) {
        try await withCheckedThrowingContinuation { continuation in
            connection.receive(minimumIncompleteLength: 1, maximumLength: 64 * 1024) {
                data, _, complete, error in
                if let error {
                    continuation.resume(throwing: error)
                } else {
                    continuation.resume(returning: (data ?? Data(), complete))
                }
            }
        }
    }

    private func handle(_ message: [String: Any]) throws {
        switch message["op"] as? String {
        case "update":
            guard let update = Self.decodeUpdate(message) else {
                if let seq = Self.integer(message["seq"]) {
                    setResumeCursor(seq)
                }
                return
            }
            let ready = updateDemultiplexer.receive(
                update,
                whileRequestInFlight: requestInFlight || pendingResponse != nil
            )
            yieldUpdates(ready)

        case "ack", "subscribed", "pong", "error":
            guard let pendingResponse else {
                throw OmegaChannelError.invalidMessage("unsolicited response")
            }
            self.pendingResponse = nil
            pendingResponse.resume(returning: message)

        default:
            throw OmegaChannelError.invalidMessage("unknown op")
        }
    }

    private func request(
        _ line: Data,
        allowsUnsubscribed: Bool = false
    ) async throws -> ([String: Any], [TrayUpdate]) {
        await acquireRequestSlot()
        defer { releaseRequestSlot() }
        try Task.checkCancellation()

        guard isRunning, allowsUnsubscribed || isSubscribed, let connection else {
            throw OmegaChannelError.disconnected
        }

        let response: [String: Any] = try await withCheckedThrowingContinuation { continuation in
            pendingResponse = continuation
            connection.send(content: line, completion: .contentProcessed { [weak self] error in
                guard let error else { return }
                Task { @MainActor [weak self] in
                    self?.disconnectCurrentConnection(error)
                }
            })
        }
        return (response, updateDemultiplexer.drain())
    }

    private func acquireRequestSlot() async {
        if !requestInFlight {
            requestInFlight = true
            return
        }
        await withCheckedContinuation { requestWaiters.append($0) }
    }

    private func releaseRequestSlot() {
        if requestWaiters.isEmpty {
            requestInFlight = false
        } else {
            requestWaiters.removeFirst().resume()
        }
    }

    private func acquireSaySlot() async {
        if !sayInFlight {
            sayInFlight = true
            return
        }
        await withCheckedContinuation { sayWaiters.append($0) }
    }

    private func releaseSaySlot() {
        if sayWaiters.isEmpty {
            sayInFlight = false
        } else {
            sayWaiters.removeFirst().resume()
        }
    }

    private func waitUntilSubscribed() async throws {
        if isSubscribed { return }
        guard isRunning else { throw OmegaChannelError.stopped }
        let id = UUID()
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
                if Task.isCancelled {
                    continuation.resume(throwing: CancellationError())
                } else {
                    connectionWaiters[id] = continuation
                }
            }
        } onCancel: {
            Task { @MainActor [weak self] in
                self?.cancelConnectionWaiter(id)
            }
        }
    }

    private func resumeConnectionWaiters(throwing error: Error? = nil) {
        let waiters = connectionWaiters.values
        connectionWaiters.removeAll()
        for waiter in waiters {
            if let error { waiter.resume(throwing: error) }
            else { waiter.resume() }
        }
    }

    private func cancelConnectionWaiter(_ id: UUID) {
        connectionWaiters.removeValue(forKey: id)?.resume(throwing: CancellationError())
    }

    private func completeReadyWaiter(id: UUID, throwing error: Error? = nil) {
        guard let waiter = readyWaiter, waiter.id == id else { return }
        readyWaiter = nil
        if let error { waiter.continuation.resume(throwing: error) }
        else { waiter.continuation.resume() }
    }

    private func failReadyWaiter(_ error: Error) {
        guard let waiter = readyWaiter else { return }
        readyWaiter = nil
        waiter.continuation.resume(throwing: error)
    }

    private func disconnectCurrentConnection(_ error: Error) {
        connection?.cancel()
        connection = nil
        connectionID = nil
        receiveTask?.cancel()
        receiveTask = nil
        isSubscribed = false
        failPending(OmegaChannelError.disconnected)
        failReadyWaiter(error)
        if !updateDemultiplexer.isSayPending {
            updateDemultiplexer.discard()
        }

        if !emittedDisconnected {
            emittedDisconnected = true
            eventContinuation.yield(.disconnected)
        }

        sessionWaiter?.resume(throwing: error)
        sessionWaiter = nil
    }

    private func failPending(_ error: Error) {
        pendingResponse?.resume(throwing: error)
        pendingResponse = nil
    }

    private func yieldUpdates(_ updates: [TrayUpdate]) {
        for update in updates {
            eventContinuation.yield(.update(update))
        }
    }

    static func encodeSay(_ submission: TraySubmission) throws -> Data {
        var object: [String: Any] = [
            "op": "say",
            "id": submission.id.uuidString,
            "text": submission.text,
            "channel": "tray",
            "context": submission.context.map {
                ["id": $0.id.uuidString, "kind": $0.kind.lowercased(), "title": $0.title]
            },
            "urgency": submission.urgency,
        ]
        if let resumesSeq = submission.resumesSeq {
            object["resumes_seq"] = resumesSeq
        }
        return try encodeObject(object)
    }

    private static func encodeObject(_ object: [String: Any]) throws -> Data {
        var data = try JSONSerialization.data(withJSONObject: object)
        data.append(0x0A)
        guard data.count <= OmegaJSONLineFramer.maximumLineBytes else {
            throw OmegaChannelError.lineTooLong
        }
        return data
    }

    private static func decodeObject(_ data: Data) throws -> [String: Any] {
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw OmegaChannelError.invalidMessage("expected a JSON object")
        }
        return object
    }

    static func decodeUpdate(_ object: [String: Any]) -> TrayUpdate? {
        guard integer(object["v"]) == 1,
              let seq = integer(object["seq"]),
              let forSeq = integer(object["for_seq"]),
              let state = object["state"] as? String,
              let kind = object["kind"] as? String
        else { return nil }

        let context = (object["context"] as? [[String: Any]] ?? []).compactMap { item -> TrayUpdateContext? in
            guard let idString = item["id"] as? String,
                  let id = UUID(uuidString: idString),
                  let kind = item["kind"] as? String
            else { return nil }
            return TrayUpdateContext(id: id, kind: kind)
        }

        return TrayUpdate(
            seq: seq,
            forSeq: forSeq,
            state: state,
            kind: kind,
            text: object["text"] as? String,
            urgency: object["urgency"] as? String,
            context: context,
            tool: object["tool"] as? String,
            ok: object["ok"] as? Bool,
            needs: object["needs"] as? String,
            outcome: object["outcome"] as? String,
            reply: object["reply"] as? String,
            error: object["error"] as? String
        )
    }

    private static func responseError(_ object: [String: Any], expected: String) -> OmegaChannelError {
        if object["op"] as? String == "error" {
            return .server(object["error"] as? String ?? "Omega channel request failed")
        }
        return .invalidMessage("expected \(expected)")
    }

    private static func integer(_ value: Any?) -> Int? {
        guard let number = value as? NSNumber,
              CFGetTypeID(number) != CFBooleanGetTypeID(),
              number.doubleValue.rounded(.towardZero) == number.doubleValue
        else { return nil }
        return number.intValue
    }
}
