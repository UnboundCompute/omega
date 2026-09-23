import AppKit
import Foundation

struct TrayMessage: Identifiable, Equatable {
    enum Role: Equatable {
        case user
        case omega
        case status
    }

    let id: UUID
    let role: Role
    let text: String
    let contextDescriptions: [String]
    var delivery: Delivery

    enum Delivery: Equatable {
        case none
        case sending
        case sent
        case failed
    }

    init(
        id: UUID = UUID(),
        role: Role,
        text: String,
        contextDescriptions: [String] = [],
        delivery: Delivery = .none
    ) {
        self.id = id
        self.role = role
        self.text = text
        self.contextDescriptions = contextDescriptions
        self.delivery = delivery
    }
}

struct StagedContext: Identifiable {
    enum Kind: String {
        case file = "File"
        case image = "Image"
        case text = "Text"
        case link = "Link"
        case screen = "Screen"

        var wireValue: String { rawValue.lowercased() }
    }

    let id: UUID
    let kind: Kind
    let title: String
    let detail: String
    let preview: NSImage?
    let fileURL: URL?
    let text: String?

    init(
        id: UUID = UUID(),
        kind: Kind,
        title: String,
        detail: String,
        preview: NSImage? = nil,
        fileURL: URL? = nil,
        text: String? = nil
    ) {
        self.id = id
        self.kind = kind
        self.title = title
        self.detail = detail
        self.preview = preview
        self.fileURL = fileURL
        self.text = text
    }
}

enum WorkState: Equatable {
    case ready
    case sending
    case understood
    case working(String)
    case blocked(String)
    case complete(String)
    case failed(String)

    var isBusy: Bool {
        switch self {
        case .sending, .understood, .working: true
        default: false
        }
    }

    var label: String {
        switch self {
        case .ready: "Ready"
        case .sending: "Sending"
        case .understood: "Understood"
        case .working(let detail): detail
        case .blocked: "Action needed"
        case .complete: "Verified complete"
        case .failed: "Not delivered"
        }
    }
}

enum ScreenCapturePermission: Equatable {
    case unknown
    case granted
    case denied
}

struct TraySubmission: Equatable {
    let id: UUID
    let text: String
    let context: [TrayContextReference]
    let urgency: String
    let resumesSeq: Int?

    init(
        id: UUID = UUID(),
        text: String,
        context: [TrayContextReference] = [],
        urgency: String = "normal",
        resumesSeq: Int? = nil
    ) {
        self.id = id
        self.text = text
        self.context = context
        self.urgency = urgency
        self.resumesSeq = resumesSeq
    }
}

struct TrayContextReference: Codable, Equatable, Sendable {
    let id: UUID
    let kind: String
    let title: String
}

struct TrayAcknowledgement: Equatable, Sendable {
    let seq: Int
    let duplicate: Bool
    let bufferedUpdates: [TrayUpdate]

    init(seq: Int, duplicate: Bool, bufferedUpdates: [TrayUpdate] = []) {
        self.seq = seq
        self.duplicate = duplicate
        self.bufferedUpdates = bufferedUpdates
    }
}

struct TrayUpdateContext: Codable, Equatable, Sendable {
    let id: UUID
    let kind: String
}

struct TrayUpdate: Equatable, Sendable {
    let seq: Int
    let forSeq: Int
    let state: String
    let kind: String
    let text: String?
    let urgency: String?
    let context: [TrayUpdateContext]
    let tool: String?
    let ok: Bool?
    let needs: String?
    let outcome: String?
    let reply: String?
    let error: String?

    init(
        seq: Int,
        forSeq: Int,
        state: String,
        kind: String,
        text: String? = nil,
        urgency: String? = nil,
        context: [TrayUpdateContext] = [],
        tool: String? = nil,
        ok: Bool? = nil,
        needs: String? = nil,
        outcome: String? = nil,
        reply: String? = nil,
        error: String? = nil
    ) {
        self.seq = seq
        self.forSeq = forSeq
        self.state = state
        self.kind = kind
        self.text = text
        self.urgency = urgency
        self.context = context
        self.tool = tool
        self.ok = ok
        self.needs = needs
        self.outcome = outcome
        self.reply = reply
        self.error = error
    }
}

enum TrayTransportEvent: Equatable, Sendable {
    case connected(head: Int)
    case disconnected
    case update(TrayUpdate)
}

enum TrayConnectionState: Equatable {
    case connecting
    case connected
    case disconnected

    var label: String {
        switch self {
        case .connecting: "Connecting to agent"
        case .connected: "Ready"
        case .disconnected: "Agent offline"
        }
    }
}
