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
    case working(String)
    case blocked(String)
    case complete(String)
    case failed(String)

    var isBusy: Bool {
        switch self {
        case .sending, .working: true
        default: false
        }
    }

    var label: String {
        switch self {
        case .ready: "Ready"
        case .sending: "Understood"
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
    let text: String
    let contextDescriptions: [String]
}
