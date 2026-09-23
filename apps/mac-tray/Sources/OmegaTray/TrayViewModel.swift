import AppKit
import CoreGraphics
import Foundation
import UniformTypeIdentifiers

@MainActor
final class TrayViewModel: ObservableObject {
    enum ScreenCaptureMode {
        case area
        case window
        case display

        var arguments: [String] {
            switch self {
            case .area: ["-i", "-s", "-x"]
            case .window: ["-i", "-w", "-x"]
            case .display: ["-D\(activeDisplayIndex())", "-x"]
            }
        }

        var title: String {
            switch self {
            case .area: "Area capture"
            case .window: "Window capture"
            case .display: "Display capture"
            }
        }

        private func activeDisplayIndex() -> Int {
            let pointer = NSEvent.mouseLocation
            let index = NSScreen.screens.firstIndex { NSMouseInRect(pointer, $0.frame, false) }
            return (index ?? 0) + 1
        }
    }

    @Published var draft = ""
    @Published var messages: [TrayMessage] = []
    @Published var stagedContext: [StagedContext] = []
    @Published var workState: WorkState = .ready
    @Published var isDropTargeted = false
    @Published var isExpanded = false
    @Published var hasUnread = false
    @Published var proactivePeek: String?
    @Published var composerFocusRequest = 0
    @Published var capturePermission: ScreenCapturePermission
    @Published var hotKeyRegistrationFailed = false
    @Published var isPrivacyRestricted = false

    private let transport: TrayTransport
    private var failedSend: FailedSend?

    private struct FailedSend {
        let messageID: UUID
        let submission: TraySubmission
        let draft: String
        let context: [StagedContext]
    }

    init(transport: TrayTransport) {
        self.transport = transport
        capturePermission = CGPreflightScreenCaptureAccess() ? .granted : .unknown
    }

    var canSend: Bool {
        !draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            || !stagedContext.isEmpty
    }

    var canSubmit: Bool {
        canSend && !workState.isBusy
    }

    var displayedProactivePeek: String {
        guard !isPrivacyRestricted, !AppSettings.shared.hideProactivePreviews else {
            return "omega has something for you"
        }
        return proactivePeek ?? "omega has something for you"
    }

    func send() {
        guard canSubmit else { return }

        let text = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        let sentContext = stagedContext
        let submission = TraySubmission(
            text: text,
            contextDescriptions: sentContext.map { $0.title }
        )

        let visibleInstruction = text.isEmpty ? "Use the attached context." : text
        let messageID = UUID()
        messages.append(
            .init(
                id: messageID,
                role: .user,
                text: visibleInstruction,
                contextDescriptions: sentContext.map { $0.title },
                delivery: .sending
            )
        )
        draft = ""
        stagedContext = []
        workState = .sending
        failedSend = nil

        deliver(
            submission,
            messageID: messageID,
            originalDraft: text,
            originalContext: sentContext
        )
    }

    func retryLastSend() {
        guard let failedSend, !workState.isBusy else { return }
        stagedContext.removeAll { context in failedSend.context.contains { $0.id == context.id } }
        if draft == failedSend.draft { draft = "" }
        updateMessage(failedSend.messageID) { $0.delivery = .sending }
        workState = .sending
        self.failedSend = nil
        deliver(
            failedSend.submission,
            messageID: failedSend.messageID,
            originalDraft: failedSend.draft,
            originalContext: failedSend.context
        )
    }

    private func deliver(
        _ submission: TraySubmission,
        messageID: UUID,
        originalDraft: String,
        originalContext: [StagedContext]
    ) {
        Task {
            do {
                workState = .working("Working locally")
                let response = try await transport.send(submission)
                updateMessage(messageID) { $0.delivery = .sent }
                messages.append(.init(role: .omega, text: response))
                workState = .complete("Demo response received")
            } catch {
                updateMessage(messageID) { $0.delivery = .failed }
                if draft.isEmpty { draft = originalDraft }
                let stagedIDs = Set(stagedContext.map(\.id))
                stagedContext.append(contentsOf: originalContext.filter { !stagedIDs.contains($0.id) })
                failedSend = .init(
                    messageID: messageID,
                    submission: submission,
                    draft: originalDraft,
                    context: originalContext
                )
                workState = .failed("Request not delivered. Your draft and context were restored.")
            }
        }
    }

    func removeContext(id: UUID) {
        stagedContext.removeAll { $0.id == id }
    }

    func receiveProactiveMessage(_ text: String) {
        messages.append(.init(role: .omega, text: text))
        hasUnread = false
        proactivePeek = nil
    }

    func importProviders(_ providers: [NSItemProvider]) -> Bool {
        var accepted = false

        for provider in providers {
            if provider.hasItemConformingToTypeIdentifier(UTType.fileURL.identifier) {
                accepted = true
                provider.loadDataRepresentation(forTypeIdentifier: UTType.fileURL.identifier) { [weak self] data, _ in
                    guard
                        let data,
                        let url = URL(dataRepresentation: data, relativeTo: nil)
                    else { return }
                    Task { @MainActor in self?.stageFile(url) }
                }
            } else if provider.hasItemConformingToTypeIdentifier(UTType.url.identifier) {
                accepted = true
                provider.loadItem(forTypeIdentifier: UTType.url.identifier, options: nil) { [weak self] item, _ in
                    let url = (item as? URL) ?? (item as? String).flatMap(URL.init(string:))
                    guard let url else { return }
                    Task { @MainActor in self?.stageURL(url) }
                }
            } else if provider.canLoadObject(ofClass: NSImage.self) {
                accepted = true
                _ = provider.loadObject(ofClass: NSImage.self) { [weak self] object, _ in
                    guard let image = object as? NSImage else { return }
                    Task { @MainActor in self?.stageImage(image, title: "Dropped image", kind: .image) }
                }
            } else if provider.canLoadObject(ofClass: NSString.self) {
                accepted = true
                _ = provider.loadObject(ofClass: NSString.self) { [weak self] object, _ in
                    guard let text = object as? String else { return }
                    Task { @MainActor in self?.stageText(text) }
                }
            }
        }

        return accepted
    }

    func pasteFromClipboard() {
        let pasteboard = NSPasteboard.general

        if let urls = pasteboard.readObjects(forClasses: [NSURL.self]) as? [URL], !urls.isEmpty {
            urls.forEach { $0.isFileURL ? stageFile($0) : stageURL($0) }
            return
        }

        if let image = NSImage(pasteboard: pasteboard) {
            stageImage(image, title: "Clipboard image", kind: .image)
            return
        }

        if let text = pasteboard.string(forType: .string), !text.isEmpty {
            if let url = URL(string: text), url.scheme != nil {
                stageURL(url)
            } else {
                stageText(text)
            }
        }
    }

    func captureArea() {
        captureScreen(.area)
    }

    func captureScreen(_ mode: ScreenCaptureMode) {
        guard ensureScreenCapturePermission() else { return }

        let destination = FileManager.default.temporaryDirectory
            .appendingPathComponent("omega-capture-\(UUID().uuidString).png")
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/sbin/screencapture")
        process.arguments = mode.arguments + [destination.path]
        workState = .working(mode == .display ? "Capturing active display" : "Choose a \(mode == .area ? "region" : "window")")

        process.terminationHandler = { [weak self] process in
            Task { @MainActor in
                guard process.terminationStatus == 0,
                      FileManager.default.fileExists(atPath: destination.path),
                      let image = NSImage(contentsOf: destination)
                else {
                    self?.workState = .ready
                    return
                }

                self?.stagedContext.append(
                    StagedContext(
                        kind: .screen,
                        title: mode.title,
                        detail: "Screen · Not sent",
                        preview: image,
                        fileURL: destination
                    )
                )
                self?.workState = .ready
            }
        }

        do {
            try process.run()
        } catch {
            workState = .failed("Screen capture could not start.")
        }
    }

    func openScreenCaptureSettings() {
        guard let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture") else { return }
        NSWorkspace.shared.open(url)
    }

    func previewContext(_ selected: StagedContext) {
        guard let selectedURL = selected.fileURL else { return }
        let urls = stagedContext.compactMap(\.fileURL)
        QuickLookController.shared.preview(urls, startingWith: selectedURL)
    }

    private func ensureScreenCapturePermission() -> Bool {
        if CGPreflightScreenCaptureAccess() {
            capturePermission = .granted
            return true
        }

        let granted = CGRequestScreenCaptureAccess()
        capturePermission = granted ? .granted : .denied
        if !granted {
            workState = .blocked("Screen Recording permission is needed. Open System Settings to allow it.")
        }
        return granted
    }

    private func updateMessage(_ id: UUID, change: (inout TrayMessage) -> Void) {
        guard let index = messages.firstIndex(where: { $0.id == id }) else { return }
        change(&messages[index])
    }

    private func stageFile(_ url: URL) {
        if let image = NSImage(contentsOf: url) {
            stageImage(image, title: url.lastPathComponent, kind: .image, fileURL: url)
            return
        }

        let icon = NSWorkspace.shared.icon(forFile: url.path)
        stagedContext.append(
            StagedContext(
                kind: .file,
                title: url.lastPathComponent,
                detail: "File · Not sent",
                preview: icon,
                fileURL: url
            )
        )
    }

    private func stageURL(_ url: URL) {
        stagedContext.append(
            StagedContext(
                kind: .link,
                title: url.host ?? url.absoluteString,
                detail: "Link · Not sent",
                text: url.absoluteString
            )
        )
    }

    private func stageText(_ text: String) {
        let normalized = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !normalized.isEmpty else { return }
        stagedContext.append(
            StagedContext(
                kind: .text,
                title: String(normalized.prefix(48)),
                detail: "Text · Not sent",
                text: normalized
            )
        )
    }

    private func stageImage(
        _ image: NSImage,
        title: String,
        kind: StagedContext.Kind,
        fileURL: URL? = nil
    ) {
        stagedContext.append(
            StagedContext(
                kind: kind,
                title: title,
                detail: "Image · Not sent",
                preview: image,
                fileURL: fileURL
            )
        )
    }
}
