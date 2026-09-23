import Foundation

@MainActor
protocol TrayTransport: AnyObject {
    var events: AsyncStream<TrayTransportEvent> { get }

    func start(since: Int?)
    func send(_ submission: TraySubmission) async throws -> TrayAcknowledgement
    func setResumeCursor(_ seq: Int)
    func stop()
}
