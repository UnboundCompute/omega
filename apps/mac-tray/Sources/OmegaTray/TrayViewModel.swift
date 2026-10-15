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
    @Published var conversationScrollRequest = 0
    @Published var composerMode: ComposerMode = .ask
    @Published var capturePermission: ScreenCapturePermission
    @Published var hotKeyRegistrationFailure: String?
    @Published var isRestoringTranscript = false
    @Published var transcriptRestoreFailure: String?
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
    /// Send pressed while files named in the message were still being stored.
    private var sendWhenStored = false
    private var transcriptRestoreTask: Task<Void, Never>?
    private var updatesDuringTranscriptRestore: [TrayUpdate] = []
    private var lastProcessedSeq: Int?
    private var pendingResumeSeq: Int?
    private var urgencyByTurn: [Int: String] = [:]
    /// Answers the core recorded (`settled`), by proposal id, from any channel.
    private var settledAnswers: [String: Bool] = [:]
    /// Replies shown as a notification before they reach the conversation,
    /// with the proposals the core sent beside them.
    private var proactiveProposals: [(text: String, proposals: [TrayProposal])] = []
    /// A button's own "yes pN" / "no pN" message, by the user message it shows as.
    private var answersByMessage: [UUID: (proposalID: String, yes: Bool)] = [:]
    /// The turn a button's answer became, by its seq.
    private var answersByTurn: [Int: String] = [:]
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
        transcriptRestoreTask?.cancel()
    }

    var workState: WorkState {
        localWorkState ?? turnState
    }

    var canSend: Bool {
        let hasText = !draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        return switch composerMode {
        case .ask: hasText || !stagedContext.isEmpty
        case .teach: hasText
        }
    }

    var canSubmit: Bool {
        canSend
            && !stagedContext.contains(where: \.blocksSending)
            && !workState.isBusy
            && connectionState == .connected
    }

    var canRetryLastSend: Bool { failedSend != nil }

    var canAnswerProposals: Bool {
        connectionState == .connected && !workState.isBusy
    }

    var canStartNewChat: Bool {
        !messages.isEmpty
            && draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && stagedContext.isEmpty
            && isConversationSettled
            && failedSend == nil
    }

    var canBeginTeaching: Bool {
        isConversationSettled && failedSend == nil
    }

    var canRestoreTranscript: Bool {
        !isRestoringTranscript
            && connectionState == .connected
            && draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && stagedContext.isEmpty
            && isConversationSettled
            && failedSend == nil
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
        if stageFilesNamedInDraft() {
            // Sent once those files are stored, by `uploadAttachment`.
            sendWhenStored = true
            return
        }
        sendWhenStored = false

        let text = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        let sentContext = stagedContext
        let sentMode = composerMode
        let submission = TraySubmission(
            text: sentMode == .teach ? teachingInstruction(for: text) : text,
            context: sentContext.map {
                TrayContextReference(
                    id: $0.id,
                    kind: $0.kind.wireValue,
                    title: $0.submissionTitle,
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

    /// A Yes or No button. It sends exactly what typing would, "yes pN" or
    /// "no pN", down the same path as `send`, so the core answers it with the
    /// same approval rule, records and taint; nothing here decides an outcome.
    func answer(_ proposalID: String, yes: Bool, in messageID: UUID) {
        guard canAnswerProposals,
              let message = messages.first(where: { $0.id == messageID }),
              let prompt = message.proposals.first(where: { $0.id == proposalID }),
              prompt.isAnswerable
        else { return }

        let text = "\(yes ? "yes" : "no") \(proposalID)"
        let submission = TraySubmission(text: text, resumesSeq: pendingResumeSeq)
        let userMessageID = UUID()
        messages.append(.init(id: userMessageID, role: .user, text: text, delivery: .sending))
        answersByMessage[userMessageID] = (proposalID, yes)
        updatePrompts(proposalID) { $0.answer = .sending(yes: yes) }
        turnState = .sending
        failedSend = nil
        pendingResumeSeq = nil

        deliver(
            submission,
            messageID: userMessageID,
            originalDraft: "",
            originalContext: [],
            originalComposerMode: .ask
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

    func restoreTranscript() {
        guard canRestoreTranscript else { return }
        transcriptRestoreTask?.cancel()
        isRestoringTranscript = true
        transcriptRestoreFailure = nil
        updatesDuringTranscriptRestore = []

        transcriptRestoreTask = Task { [weak self] in
            guard let self else { return }
            do {
                let history = try await transport.transcript()
                try Task.checkCancellation()
                let combined = history + updatesDuringTranscriptRestore
                let unique = Dictionary(grouping: combined, by: \.seq)
                    .compactMap { $0.value.last }
                    .sorted { $0.seq < $1.seq }
                for update in unique { recordSettled(update.settled) }
                let restored = transcriptMessages(from: unique)
                messages = restored.isEmpty
                    ? [.init(role: .status, text: "No earlier transcript was found.")]
                    : restored
            } catch is CancellationError {
                return
            } catch {
                transcriptRestoreFailure = error.localizedDescription
            }
            updatesDuringTranscriptRestore = []
            isRestoringTranscript = false
            transcriptRestoreTask = nil
        }
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
        if let answer = answersByMessage[failedSend.messageID] {
            updatePrompts(answer.proposalID) { $0.answer = .sending(yes: answer.yes) }
        }
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
                if let answer = answersByMessage[messageID] {
                    answersByTurn[acknowledgement.seq] = answer.proposalID
                    updatePrompts(answer.proposalID) { prompt in
                        if case .sending = prompt.answer { prompt.answer = .sent(yes: answer.yes) }
                    }
                }
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
            if isRestoringTranscript {
                updatesDuringTranscriptRestore.append(update)
            }
            handleUpdate(update)
            lastProcessedSeq = update.seq
            persistCursor(update.seq)
            transport.setResumeCursor(update.seq)
        }
    }

    private func handleUpdate(_ update: TrayUpdate) {
        let belongsToActiveTurn = activeTurn?.seq == update.forSeq
        applySettled(update.settled)

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
                        messages.append(.init(role: .omega, text: reply, proposals: prompts(for: update.proposals)))
                    } else {
                        if !update.proposals.isEmpty {
                            proactiveProposals.append((reply, update.proposals))
                        }
                        if let proactivePresentation {
                            proactivePresentation(reply, urgency(for: update.forSeq))
                        } else {
                            proactivePeek = reply
                            hasUnread = true
                        }
                    }
                }
                turnState = .complete("Complete")
            case "silent":
                messages.append(.init(role: .status, text: "omega stayed quiet."))
                turnState = .complete("Complete — no reply needed")
            default:
                turnState = .complete("Complete")
            }
            // A button's answer turn that settled nothing found the proposal
            // already closed (answered elsewhere, expired); the core said so.
            if let proposalID = answersByTurn.removeValue(forKey: update.forSeq) {
                updatePrompts(proposalID) { prompt in
                    if case .sent = prompt.answer { prompt.answer = .closed }
                }
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

    private func transcriptMessages(from updates: [TrayUpdate]) -> [TrayMessage] {
        let foreignTurnSequences = Set(updates.compactMap { update -> Int? in
            guard update.state == "understood",
                  update.kind == "message.inbound",
                  let channel = update.channel,
                  !channel.isEmpty,
                  channel != "tray"
            else { return nil }
            return update.forSeq
        })

        return updates.compactMap { update in
            guard !foreignTurnSequences.contains(update.forSeq) else { return nil }
            switch update.state {
            case "understood" where update.kind == "message.inbound":
                let text = update.text?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
                let visibleText = visibleTranscriptInstruction(text)
                return TrayMessage(
                    role: .user,
                    text: visibleText.isEmpty ? "Context shared with omega" : visibleText,
                    contextDescriptions: update.context.map { "\($0.kind.capitalized) context" },
                    delivery: .sent
                )
            case "complete" where update.outcome == "spoke":
                guard let reply = update.reply, !reply.isEmpty else { return nil }
                return TrayMessage(role: .omega, text: reply, proposals: prompts(for: update.proposals))
            case "failed":
                let detail = update.error?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
                return TrayMessage(role: .status, text: detail.isEmpty ? "A previous turn failed." : detail)
            default:
                return nil
            }
        }
    }

    private func teachingInstruction(for note: String) -> String {
        """
        Teaching note from me. Treat this as something to remember and apply in future conversations, not as a task to execute. Briefly confirm what you learned.

        \(note)
        """
    }

    private func visibleTranscriptInstruction(_ text: String) -> String {
        guard let separator = text.range(of: "\n\n"),
              text[..<separator.lowerBound].contains("remember and apply in future conversations")
        else { return text }

        return String(text[separator.upperBound...])
            .trimmingCharacters(in: .whitespacesAndNewlines)
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
        if let answer = answersByMessage[messageID] {
            answersByTurn = answersByTurn.filter { $0.value != answer.proposalID }
            updatePrompts(answer.proposalID) { prompt in
                switch prompt.answer {
                case .sending, .sent: prompt.answer = .failed
                default: break
                }
            }
        }
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

    func removeContext(id: UUID) {
        sendWhenStored = false
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
        var proposals: [TrayProposal] = []
        if let index = proactiveProposals.firstIndex(where: { $0.text == text }) {
            proposals = proactiveProposals.remove(at: index).proposals
        }
        messages.append(.init(role: .omega, text: text, proposals: prompts(for: proposals)))
        hasUnread = false
        proactivePeek = nil
    }

    /// A drop, item by item. A file always arrives as a file, uploaded with `put`: its
    /// URL is taken before any other representation, because a text file (a `.md`, a
    /// `.txt`) also offers itself as a string, and reading that string made the file a
    /// 48-character snippet with no bytes behind it. A drag with no file URL (an editor's)
    /// is searched for the files it names (`DroppedFiles`) before anything is read as
    /// content. Text that is only text is kept whole, as a stored file (`stageText`).
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
            } else if !provider.registeredTypeIdentifiers.isEmpty {
                accepted = true
                Task { @MainActor [weak self] in
                    let files = await DroppedFiles.references(in: provider)
                    guard let self else { return }
                    if files.isEmpty {
                        self.importContent(provider)
                    } else {
                        files.forEach { self.stageFile($0) }
                    }
                }
            }
        }

        return accepted
    }

    /// A dropped item that names no file: a link, a named document held as bytes, a
    /// picture, or text.
    private func importContent(_ provider: NSItemProvider) {
        if provider.hasItemConformingToTypeIdentifier(UTType.url.identifier) {
            provider.loadItem(forTypeIdentifier: UTType.url.identifier, options: nil) { [weak self] item, _ in
                let url = (item as? URL) ?? (item as? String).flatMap(URL.init(string:))
                guard let url else { return }
                Task { @MainActor in self?.stageURL(url) }
            }
        } else if let type = Self.fileType(of: provider) {
            // A file offered as typed bytes with a name (a document dragged out of an
            // app that holds it) rather than as a URL: copied out before the provider
            // deletes its temporary file, then staged as that file.
            let name = provider.suggestedName ?? "file"
            provider.loadFileRepresentation(forTypeIdentifier: type) { [weak self] url, _ in
                guard let url, let copy = Self.copyOut(url, named: name, type: type) else { return }
                Task { @MainActor in self?.stageFile(copy) }
            }
        } else if provider.canLoadObject(ofClass: NSImage.self) {
            _ = provider.loadObject(ofClass: NSImage.self) { [weak self] object, _ in
                guard let image = object as? NSImage else { return }
                Task { @MainActor in self?.stageImage(image, title: "Dropped image", kind: .image) }
            }
        } else if provider.canLoadObject(ofClass: NSString.self) {
            _ = provider.loadObject(ofClass: NSString.self) { [weak self] object, _ in
                guard let text = object as? String else { return }
                Task { @MainActor in self?.stageDroppedText(text) }
            }
        }
    }

    /// The type of a named file the provider holds as bytes, if it holds one: a
    /// registered type more specific than plain text or a URL, on a provider that names
    /// its item. A text selection registers only the generic string types and no name.
    nonisolated static func fileType(of provider: NSItemProvider) -> String? {
        guard let name = provider.suggestedName, !name.isEmpty else { return nil }
        let generic: Set<String> = [
            UTType.plainText.identifier, UTType.utf8PlainText.identifier,
            UTType.utf16PlainText.identifier, UTType.text.identifier,
            UTType.url.identifier, UTType.fileURL.identifier,
        ]
        return provider.registeredTypeIdentifiers.first { identifier in
            guard !generic.contains(identifier), let type = UTType(identifier) else { return false }
            return type.conforms(to: .data) && !type.conforms(to: .url)
        }
    }

    nonisolated private static func copyOut(_ url: URL, named name: String, type: String) -> URL? {
        var file = name.replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: "\\", with: "_")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if file.isEmpty || file.hasPrefix(".") { file = "file\(file)" }
        if URL(fileURLWithPath: file).pathExtension.isEmpty,
           let ext = UTType(type)?.preferredFilenameExtension {
            file += ".\(ext)"
        }
        let folder = FileManager.default.temporaryDirectory
            .appendingPathComponent("omega-drop-\(UUID().uuidString)", isDirectory: true)
        do {
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            let copy = folder.appendingPathComponent(file)
            try FileManager.default.copyItem(at: url, to: copy)
            return copy
        } catch {
            return nil
        }
    }

    private func stageDroppedText(_ text: String) {
        let files = DroppedFiles.references(in: text)
        if files.isEmpty {
            stageText(text)
        } else {
            files.forEach { stageFile($0) }
        }
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
            let files = DroppedFiles.references(in: text)
            if !files.isEmpty {
                files.forEach { stageFile($0) }
            } else if let url = URL(string: text), url.scheme != nil {
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

    private func prompts(for proposals: [TrayProposal]) -> [TrayProposalPrompt] {
        proposals.map { proposal in
            let answer: TrayProposalPrompt.Answer = switch settledAnswers[proposal.id] {
            case .some(true): .approved
            case .some(false): .declined
            case .none: .open
            }
            return TrayProposalPrompt(proposal: proposal, answer: answer)
        }
    }

    private func recordSettled(_ settled: [TraySettled]) {
        for answer in settled { settledAnswers[answer.id] = answer.yes }
    }

    /// The core recorded these answers, by a button, by typing, or on another
    /// channel: every prompt for them is closed with that outcome.
    private func applySettled(_ settled: [TraySettled]) {
        recordSettled(settled)
        for answer in settled {
            updatePrompts(answer.id) { $0.answer = answer.yes ? .approved : .declined }
        }
    }

    private func updatePrompts(_ proposalID: String, change: (inout TrayProposalPrompt) -> Void) {
        for messageIndex in messages.indices {
            for promptIndex in messages[messageIndex].proposals.indices
            where messages[messageIndex].proposals[promptIndex].id == proposalID {
                change(&messages[messageIndex].proposals[promptIndex])
            }
        }
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

    /// Text that is only text, kept whole: the core's context item has no field for its
    /// words (`episodes.inbound` takes id, kind, title and a blob reference), so the text
    /// is written to a file and uploaded with `put` like any other, and the 48-character
    /// title is only its preview. Before, the title was all that ever left the Mac.
    func stageText(_ text: String) {
        let normalized = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !normalized.isEmpty else { return }
        let title = String(normalized.prefix(48))
        let file = FileManager.default.temporaryDirectory
            .appendingPathComponent("omega-text-\(UUID().uuidString).txt")
        do {
            try Data(normalized.utf8).write(to: file, options: .atomic)
            stage(StagedContext(kind: .text, title: title, detail: "Text · Not sent", fileURL: file, text: normalized))
        } catch {
            stagedContext.append(
                StagedContext(
                    kind: .text,
                    title: title,
                    detail: "Text · Not sent",
                    text: normalized,
                    attachmentState: .failed("The text could not be prepared for storage.")
                )
            )
        }
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

    /// A file whose path they wrote in the message (`/…`, `~/…`), staged as an attachment
    /// so the core gets the file and not only its path; the words are sent as written.
    /// True when something new was staged and the send must wait for it.
    private func stageFilesNamedInDraft() -> Bool {
        let staged = Set(stagedContext.compactMap { $0.fileURL?.standardizedFileURL.resolvingSymlinksInPath().path })
        let named = DroppedFiles.inMessage(draft).filter { !staged.contains($0.path) }
        named.forEach { stageFile($0) }
        return !named.isEmpty
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
                self?.sendWhenStored = false
                self?.updateContext(id: id) {
                    $0.attachmentState = .failed(error.localizedDescription)
                }
            }
            self?.attachmentTasks[id] = nil
            if let self, self.sendWhenStored, self.canSubmit { self.send() }
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
