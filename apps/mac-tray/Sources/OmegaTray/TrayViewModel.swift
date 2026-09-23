import AppKit
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

    private let transport: TrayTransport

    init(transport: TrayTransport) {
        self.transport = transport
    }

    var canSend: Bool {
        !draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            || !stagedContext.isEmpty
    }

    func send() {
        guard canSend, workState != .sending else { return }

        let text = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        let sentContext = stagedContext
        let submission = TraySubmission(
            text: text,
            contextDescriptions: sentContext.map { $0.title }
        )

        let visibleInstruction = text.isEmpty ? "Use the attached context." : text
        messages.append(.init(role: .user, text: visibleInstruction))
        draft = ""
        stagedContext = []
        workState = .sending

        Task {
            do {
                workState = .working("Working locally")
                let response = try await transport.send(submission)
                messages.append(.init(role: .omega, text: response))
                workState = .complete("Demo response received")
            } catch {
                workState = .failed("Could not deliver the request. Try again.")
            }
        }
    }

    func removeContext(id: UUID) {
        stagedContext.removeAll { $0.id == id }
    }

    func receiveProactiveMessage(_ text: String) {
        messages.append(.init(role: .omega, text: text))
        hasUnread = false
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
            urls.forEach(stageFile)
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
