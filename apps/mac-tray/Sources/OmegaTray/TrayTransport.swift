import Foundation

protocol TrayTransport {
    func send(_ submission: TraySubmission) async throws -> String
}

/// Interaction-only responder for the Mac tray prototype.
///
/// This is deliberately not an agent protocol and must be deleted when the real channel
/// boundary is defined from omega's agent behavior.
struct LocalDemoTransport: TrayTransport {
    func send(_ submission: TraySubmission) async throws -> String {
        try await Task.sleep(for: .milliseconds(850))

        let context = submission.contextDescriptions.isEmpty
            ? "No screen or file context was attached."
            : "I received \(submission.contextDescriptions.count) staged context item\(submission.contextDescriptions.count == 1 ? "" : "s"): \(submission.contextDescriptions.joined(separator: ", "))."

        return "\(context) The tray interaction is working locally; no agent action has been performed yet."
    }
}
