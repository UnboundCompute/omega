import AppKit
import CoreGraphics
import Foundation
import UniformTypeIdentifiers

@MainActor
protocol CapturePresentationControlling: AnyObject {
    func hideForCapture(then start: @escaping @MainActor () -> Void)
    func restoreAfterCapture()
}

@MainActor
final class TrayViewModel: ObservableObject {
    enum ComposerMode: Equatable {
        case ask
        case teach
    }

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
    @Published var turnState: WorkState = .ready
    @Published var localWorkState: WorkState?
    @Published var connectionState: TrayConnectionState = .connecting
    @Published var agentStartupFailure: String?
    @Published var isDropTargeted = false
    @Published var isExpanded = false
    @Published var hasUnread = false
    @Published var proactivePeek: String?
    @Published var composerFocusRequest = 0
    @Published var composerMode: ComposerMode = .ask
    @Published var capturePermission: ScreenCapturePermission
    @Published var hotKeyRegistrationFailure: String?
    @Published var isScreenLocked = false
    @Published var manualPrivacyMode = false

    private let transport: TrayTransport
    private let preflightScreenCaptureAccess: () -> Bool
    private let requestScreenCaptureAccess: () -> Bool
    private let loadCursor: @MainActor () -> Int?
    private let persistCursor: @MainActor (Int) -> Void
    private let terminalTimeout: Duration
    private var failedSend: FailedSend?
    private var activeTurn: ActiveTurn?
    private var eventTask: Task<Void, Never>?
    private var terminalTask: Task<Void, Never>?
    private var attachmentTasks: [UUID: Task<Void, Never>] = [:]
    private var lastProcessedSeq: Int?
    private var pendingResumeSeq: Int?
    private var urgencyByTurn: [Int: String] = [:]
    private var didStart = false
    weak var capturePresentation: CapturePresentationControlling?
    var proactivePresentation: ((String, ProactiveUrgency) -> Void)?

    private struct FailedSend {
        let messageID: UUID
        let submission: TraySubmission
        let draft: String
        let context: [StagedContext]
        let composerMode: ComposerMode
    }

    private struct ActiveTurn {
        let seq: Int
        let messageID: UUID
        let submission: TraySubmission
        let draft: String
        let context: [StagedContext]
        let composerMode: ComposerMode
        var awaitingResume = false
    }

    init(
        transport: TrayTransport,
        preflightScreenCaptureAccess: @escaping () -> Bool = CGPreflightScreenCaptureAccess,
        requestScreenCaptureAccess: @escaping () -> Bool = CGRequestScreenCaptureAccess,
        loadCursor: @escaping @MainActor () -> Int? = { AppSettings.shared.projectionCursor },
        persistCursor: @escaping @MainActor (Int) -> Void = { AppSettings.shared.projectionCursor = $0 },
        terminalTimeout: Duration = .seconds(120)
    ) {
        self.transport = transport
        self.preflightScreenCaptureAccess = preflightScreenCaptureAccess
        self.requestScreenCaptureAccess = requestScreenCaptureAccess
        self.loadCursor = loadCursor
        self.persistCursor = persistCursor
        self.terminalTimeout = terminalTimeout
        lastProcessedSeq = loadCursor()
        capturePermission = preflightScreenCaptureAccess() ? .granted : .unknown
    }

    deinit {
        eventTask?.cancel()
        terminalTask?.cancel()
    }

    var workState: WorkState {
        localWorkState ?? turnState
    }

    var canSend: Bool {
        let hasText = !draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        return switch composerMode {
        case .ask: hasText || !stagedContext.isEmpty
        case .teach: hasText && stagedContext.isEmpty
        }
    }

    var canSubmit: Bool {
        canSend
            && !stagedContext.contains(where: \.blocksSending)
            && !workState.isBusy
            && connectionState == .connected
    }

    var canRetryLastSend: Bool { failedSend != nil }

    var canStartNewChat: Bool {
        !messages.isEmpty
            && draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && stagedContext.isEmpty
            && isConversationSettled
            && failedSend == nil
    }

    var canBeginTeaching: Bool {
        stagedContext.isEmpty && isConversationSettled && failedSend == nil
    }

    var composerPlaceholder: String {
        composerMode == .teach ? "What should omega learn?" : "Ask or delegate…"
    }

    private var isConversationSettled: Bool {
        switch turnState {
        case .ready, .complete, .failed: true
        case .sending, .understood, .working, .blocked: false
        }
    }

    var hasStagedContentWithoutModelUnderstanding: Bool {
        stagedContext.contains { context in
            if context.kind == .image || context.kind == .screen { return false }
            return context.attachment?.mime.lowercased().hasPrefix("image/") != true
        }
    }

    var attachmentFailure: String? {
        for context in stagedContext {
            if case .failed(let detail) = context.attachmentState { return detail }
        }
        return nil
    }

    var statusLabel: String {
        if workState != .ready { return workState.label }
        return connectionState.label
    }

    var displayedProactivePeek: String {
        guard !isPrivacyRestricted, !AppSettings.shared.hideProactivePreviews else {
            return "omega has something for you"
        }
        return proactivePeek ?? "omega has something for you"
    }

    var isPrivacyRestricted: Bool {
        isScreenLocked || manualPrivacyMode
    }

    func start() {
        guard !didStart else { return }
        didStart = true
        connectionState = .connecting
        eventTask = Task { [weak self, transport] in
            for await event in transport.events {
                guard !Task.isCancelled else { return }
                self?.handleTransportEvent(event)
            }
        }
        transport.start(since: lastProcessedSeq)
    }

    func stop() {
        eventTask?.cancel()
        eventTask = nil
        terminalTask?.cancel()
        terminalTask = nil
        attachmentTasks.values.forEach { $0.cancel() }
        attachmentTasks.removeAll()
        transport.stop()
        didStart = false
    }

    func send() {
        guard canSubmit else { return }

        let text = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        let sentContext = stagedContext
        let sentMode = composerMode
        let submission = TraySubmission(
            text: sentMode == .teach ? teachingInstruction(for: text) : text,
            context: sentContext.map {
                TrayContextReference(
                    id: $0.id,
                    kind: $0.kind.wireValue,
                    title: $0.title,
                    attachment: $0.attachment
                )
            },
            resumesSeq: pendingResumeSeq
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
        composerMode = .ask
        turnState = .sending
        failedSend = nil
        pendingResumeSeq = nil

        deliver(
            submission,
            messageID: messageID,
            originalDraft: text,
            originalContext: sentContext,
            originalComposerMode: sentMode
        )
    }

    func startNewChat() {
        guard canStartNewChat else { return }
        messages.removeAll()
        turnState = .ready
        pendingResumeSeq = nil
        urgencyByTurn.removeAll()
        proactivePeek = nil
        hasUnread = false
        composerMode = .ask
        composerFocusRequest += 1
    }

    func beginTeaching() {
        guard canBeginTeaching else { return }
        composerMode = .teach
        composerFocusRequest += 1
    }

    func cancelTeaching() {
        composerMode = .ask
        composerFocusRequest += 1
    }

    func retryLastSend() {
        guard let failedSend, !workState.isBusy, connectionState == .connected else { return }
        stagedContext.removeAll { context in failedSend.context.contains { $0.id == context.id } }
        if draft == failedSend.draft {
            draft = ""
            composerMode = .ask
        }
        updateMessage(failedSend.messageID) { $0.delivery = .sending }
        turnState = .sending
        self.failedSend = nil
        deliver(
            failedSend.submission,
            messageID: failedSend.messageID,
            originalDraft: failedSend.draft,
            originalContext: failedSend.context,
            originalComposerMode: failedSend.composerMode
        )
    }

    private func deliver(
        _ submission: TraySubmission,
        messageID: UUID,
        originalDraft: String,
        originalContext: [StagedContext],
        originalComposerMode: ComposerMode
    ) {
        Task {
            do {
                let acknowledgement = try await transport.send(submission)
                updateMessage(messageID) { $0.delivery = .sent }
                activeTurn = ActiveTurn(
                    seq: acknowledgement.seq,
                    messageID: messageID,
                    submission: submission,
                    draft: originalDraft,
                    context: originalContext,
                    composerMode: originalComposerMode
                )
                scheduleTerminalTimeout(for: acknowledgement.seq)
                for update in acknowledgement.bufferedUpdates {
                    handleTransportEvent(.update(update))
                }
            } catch {
                restoreFailedSend(
                    messageID: messageID,
                    submission: submission,
                    draft: originalDraft,
                    context: originalContext,
                    composerMode: originalComposerMode,
                    detail: "Request not delivered. Your draft and context were restored."
                )
            }
        }
    }

    private func handleTransportEvent(_ event: TrayTransportEvent) {
        switch event {
        case .connected:
            connectionState = .connected
            agentStartupFailure = nil
            if var activeTurn, activeTurn.awaitingResume {
                activeTurn.awaitingResume = false
                self.activeTurn = activeTurn
                scheduleTerminalTimeout(for: activeTurn.seq)
            }
        case .disconnected:
            connectionState = .disconnected
            terminalTask?.cancel()
            terminalTask = nil
            if var activeTurn {
                activeTurn.awaitingResume = true
                self.activeTurn = activeTurn
            }
        case .update(let update):
            guard lastProcessedSeq.map({ update.seq > $0 }) ?? true else { return }
            handleUpdate(update)
            lastProcessedSeq = update.seq
            persistCursor(update.seq)
            transport.setResumeCursor(update.seq)
        }
    }

    private func handleUpdate(_ update: TrayUpdate) {
        let belongsToActiveTurn = activeTurn?.seq == update.forSeq

        switch update.state {
        case "understood":
            urgencyByTurn[update.forSeq] = update.urgency ?? "normal"
            if belongsToActiveTurn {
                turnState = .understood
                if let messageID = activeTurn?.messageID {
                    updateMessage(messageID) { $0.delivery = .sent }
                }
            } else if update.kind == "message.inbound" {
                let visibleText = update.text?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
                messages.append(
                    .init(
                        role: .user,
                        text: visibleText.isEmpty ? "Context shared with omega" : visibleText,
                        contextDescriptions: update.context.map { "\($0.kind.capitalized) context" },
                        delivery: .sent
                    )
                )
            }
        case "working":
            let detail: String
            if let tool = update.tool, !tool.isEmpty {
                detail = update.kind == "tool.returned" ? "Finished \(tool)" : "Using \(tool)"
            } else {
                detail = update.text.flatMap { $0.isEmpty ? nil : $0 } ?? "Working"
            }
            turnState = .working(detail)
        case "blocked":
            finishActiveTurn(for: update.forSeq)
            pendingResumeSeq = update.forSeq
            let needs = update.needs?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            let detail = needs.isEmpty ? "omega needs more information." : needs
            turnState = .blocked(detail)
            if !belongsToActiveTurn {
                if isExpanded {
                    messages.append(.init(role: .status, text: detail))
                } else {
                    proactivePresentation?(detail, urgency(for: update.forSeq))
                }
            }
            urgencyByTurn.removeValue(forKey: update.forSeq)
        case "complete":
            finishActiveTurn(for: update.forSeq)
            switch update.outcome {
            case "spoke":
                if let reply = update.reply, !reply.isEmpty {
                    if isExpanded {
                        messages.append(.init(role: .omega, text: reply))
                    } else if let proactivePresentation {
                        proactivePresentation(reply, urgency(for: update.forSeq))
                    } else {
                        proactivePeek = reply
                        hasUnread = true
                    }
                }
                turnState = .complete("Complete")
            case "silent":
                messages.append(.init(role: .status, text: "omega stayed quiet."))
                turnState = .complete("Complete — no reply needed")
            default:
                turnState = .complete("Complete")
            }
            urgencyByTurn.removeValue(forKey: update.forSeq)
        case "failed":
            let detail = update.error.flatMap { $0.isEmpty ? nil : $0 } ?? "The turn failed."
            if let activeTurn, activeTurn.seq == update.forSeq {
                restoreFailedSend(
                    messageID: activeTurn.messageID,
                    submission: activeTurn.submission,
                    draft: activeTurn.draft,
                    context: activeTurn.context,
                    composerMode: activeTurn.composerMode,
                    detail: detail
                )
            } else {
                turnState = .failed(detail)
                messages.append(.init(role: .status, text: detail))
            }
        default:
            break
        }
    }

    private func urgency(for seq: Int) -> ProactiveUrgency {
        urgencyByTurn[seq] == "timely" ? .timeSensitive : .normal
    }

    private func finishActiveTurn(for seq: Int) {
        guard activeTurn?.seq == seq else { return }
        terminalTask?.cancel()
        terminalTask = nil
        activeTurn = nil
        failedSend = nil
    }

    private func scheduleTerminalTimeout(for seq: Int) {
        terminalTask?.cancel()
        terminalTask = Task { [weak self, terminalTimeout] in
            try? await Task.sleep(for: terminalTimeout)
            guard !Task.isCancelled else { return }
            self?.terminalOutcomeWasNotRecovered(for: seq)
        }
    }

    private func terminalOutcomeWasNotRecovered(for seq: Int) {
        guard connectionState == .connected,
              let activeTurn,
              activeTurn.seq == seq
        else { return }
        restoreFailedSend(
            messageID: activeTurn.messageID,
            submission: activeTurn.submission,
            draft: activeTurn.draft,
            context: activeTurn.context,
            composerMode: activeTurn.composerMode,
            detail: "omega did not record an outcome. Your draft and context were restored."
        )
    }

    private func restoreFailedSend(
        messageID: UUID,
        submission: TraySubmission,
        draft originalDraft: String,
        context originalContext: [StagedContext],
        composerMode originalComposerMode: ComposerMode,
        detail: String
    ) {
        terminalTask?.cancel()
        terminalTask = nil
        activeTurn = nil
        updateMessage(messageID) { $0.delivery = .failed }
        if draft.isEmpty {
            draft = originalDraft
            composerMode = originalComposerMode
        }
        let stagedIDs = Set(stagedContext.map(\.id))
        stagedContext.append(contentsOf: originalContext.filter { !stagedIDs.contains($0.id) })
        failedSend = .init(
            messageID: messageID,
            submission: submission,
            draft: originalDraft,
            context: originalContext,
            composerMode: originalComposerMode
        )
        turnState = .failed(detail)
    }

    private func teachingInstruction(for note: String) -> String {
        """
        Teaching note from me. Treat this as something to remember and apply in future conversations, not as a task to execute. Briefly confirm what you learned.

        \(note)
        """
    }

    func removeContext(id: UUID) {
        attachmentTasks.removeValue(forKey: id)?.cancel()
        stagedContext.removeAll { $0.id == id }
    }

    func retryFailedAttachments() {
        for context in stagedContext {
            guard case .failed = context.attachmentState,
                  let fileURL = context.fileURL
            else { continue }
            uploadAttachment(id: context.id, fileAt: fileURL)
        }
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
        localWorkState = nil
        guard ensureScreenCapturePermission() else { return }

        let start: @MainActor () -> Void = { [weak self] in
            guard let self else { return }
            self.startCapture(mode)
        }

        if let capturePresentation {
            capturePresentation.hideForCapture(then: start)
        } else {
            start()
        }
    }

    private func startCapture(_ mode: ScreenCaptureMode) {
        let destination = FileManager.default.temporaryDirectory
            .appendingPathComponent("omega-capture-\(UUID().uuidString).png")
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/sbin/screencapture")
        process.arguments = mode.arguments + [destination.path]
        localWorkState = .working(mode == .display ? "Capturing active display" : "Choose a \(mode == .area ? "region" : "window")")

        process.terminationHandler = { [weak self] process in
            Task { @MainActor in
                defer { self?.capturePresentation?.restoreAfterCapture() }
                guard process.terminationStatus == 0,
                      FileManager.default.fileExists(atPath: destination.path),
                      let image = NSImage(contentsOf: destination)
                else {
                    self?.localWorkState = nil
                    return
                }

                self?.stage(
                    StagedContext(
                        kind: .screen,
                        title: mode.title,
                        detail: "Screen · Not sent",
                        preview: image,
                        fileURL: destination
                    )
                )
                self?.localWorkState = nil
            }
        }

        do {
            try process.run()
        } catch {
            localWorkState = .failed("Screen capture could not start.")
            capturePresentation?.restoreAfterCapture()
        }
    }

    func openScreenCaptureSettings() {
        guard let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture") else { return }
        NSWorkspace.shared.open(url)
    }

    func refreshScreenCapturePermission() {
        let granted = preflightScreenCaptureAccess()

        if granted {
            let wasDenied = capturePermission == .denied
            capturePermission = .granted
            if wasDenied,
               case .blocked("Screen Recording permission is needed. Open System Settings to allow it.") = localWorkState {
                localWorkState = nil
            }
        } else if capturePermission == .granted {
            capturePermission = .denied
            localWorkState = .blocked("Screen Recording permission is needed. Open System Settings to allow it.")
        }
    }

    func previewContext(_ selected: StagedContext) {
        guard let selectedURL = selected.fileURL else { return }
        let urls = stagedContext.compactMap(\.fileURL)
        QuickLookController.shared.preview(urls, startingWith: selectedURL)
    }

    private func ensureScreenCapturePermission() -> Bool {
        if preflightScreenCaptureAccess() {
            capturePermission = .granted
            return true
        }

        let granted = requestScreenCaptureAccess()
        capturePermission = granted ? .granted : .denied
        if !granted {
            localWorkState = .blocked("Screen Recording permission is needed. Open System Settings to allow it.")
        }
        return granted
    }

    private func updateMessage(_ id: UUID, change: (inout TrayMessage) -> Void) {
        guard let index = messages.firstIndex(where: { $0.id == id }) else { return }
        change(&messages[index])
    }

    func stageFile(_ url: URL) {
        if let image = NSImage(contentsOf: url) {
            stageImage(image, title: url.lastPathComponent, kind: .image, fileURL: url)
            return
        }

        let icon = NSWorkspace.shared.icon(forFile: url.path)
        stage(
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
        stage(
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
        stage(
            StagedContext(
                kind: .text,
                title: String(normalized.prefix(48)),
                detail: "Text · Not sent",
                text: normalized
            )
        )
    }

    func stageImage(
        _ image: NSImage,
        title: String,
        kind: StagedContext.Kind,
        fileURL: URL? = nil
    ) {
        if let fileURL {
            stage(
                StagedContext(
                    kind: kind,
                    title: title,
                    detail: "Image · Not sent",
                    preview: image,
                    fileURL: fileURL
                )
            )
            return
        }

        do {
            let temporaryURL = try writeTemporaryPNG(image)
            stage(
                StagedContext(
                    kind: kind,
                    title: title,
                    detail: "Image · Not sent",
                    preview: image,
                    fileURL: temporaryURL
                )
            )
        } catch {
            stagedContext.append(
                StagedContext(
                    kind: kind,
                    title: title,
                    detail: "Image · Not sent",
                    preview: image,
                    attachmentState: .failed("The image could not be prepared for storage.")
                )
            )
        }
    }

    private func stage(_ context: StagedContext) {
        stagedContext.append(context)
        guard let fileURL = context.fileURL else { return }
        uploadAttachment(id: context.id, fileAt: fileURL)
    }

    private func uploadAttachment(id: UUID, fileAt url: URL) {
        attachmentTasks.removeValue(forKey: id)?.cancel()
        updateContext(id: id) { $0.attachmentState = .uploading }

        attachmentTasks[id] = Task { [weak self, transport] in
            do {
                let reference = try await transport.attach(fileAt: url)
                try Task.checkCancellation()
                self?.updateContext(id: id) { $0.attachmentState = .ready(reference) }
            } catch is CancellationError {
                return
            } catch {
                self?.updateContext(id: id) {
                    $0.attachmentState = .failed(error.localizedDescription)
                }
            }
            self?.attachmentTasks[id] = nil
        }
    }

    private func updateContext(id: UUID, change: (inout StagedContext) -> Void) {
        guard let index = stagedContext.firstIndex(where: { $0.id == id }) else { return }
        change(&stagedContext[index])
    }

    private func writeTemporaryPNG(_ image: NSImage) throws -> URL {
        guard let tiff = image.tiffRepresentation,
              let bitmap = NSBitmapImageRep(data: tiff),
              let png = bitmap.representation(using: .png, properties: [:])
        else {
            throw CocoaError(.fileWriteUnknown)
        }
        let destination = FileManager.default.temporaryDirectory
            .appendingPathComponent("omega-image-\(UUID().uuidString).png")
        try png.write(to: destination, options: .atomic)
        return destination
    }
}
