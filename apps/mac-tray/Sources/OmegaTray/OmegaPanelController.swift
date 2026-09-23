import AppKit
import Combine
import SwiftUI

@MainActor
final class OmegaPanelController: NSWindowController, CapturePresentationControlling {
    enum Presentation {
        case resting
        case peek
        case expanded
    }

    private let viewModel: TrayViewModel
    private var presentation: Presentation = .resting
    private var peekCollapseTask: Task<Void, Never>?
    private var contentObservation: AnyCancellable?
    private var dropObservation: AnyCancellable?
    private weak var previouslyActiveApplication: NSRunningApplication?
    private var isCaptureSuspended = false
    private var captureWasKey = false
    var timeSensitiveFallback: ((String) -> Void)?

    private let restingSize = NSSize(width: 190, height: 38)
    private let peekSize = NSSize(width: 390, height: 108)

    init(viewModel: TrayViewModel) {
        self.viewModel = viewModel

        let panel = OmegaPanel(
            contentRect: NSRect(origin: .zero, size: restingSize),
            styleMask: [.borderless, .nonactivatingPanel],
            backing: .buffered,
            defer: false
        )
        panel.isOpaque = false
        panel.backgroundColor = .clear
        panel.hasShadow = false
        panel.hidesOnDeactivate = false
        panel.level = .floating
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .stationary]
        panel.isReleasedWhenClosed = false
        panel.animationBehavior = .none

        super.init(window: panel)
        viewModel.capturePresentation = self
        updateContent()

        contentObservation = Publishers.CombineLatest(viewModel.$messages, viewModel.$stagedContext)
            .dropFirst()
            .sink { [weak self] _, _ in
                guard let self, self.presentation == .expanded else { return }
                self.resize(to: self.expandedSize(), animate: true)
            }

        dropObservation = viewModel.$isDropTargeted
            .removeDuplicates()
            .dropFirst()
            .sink { [weak self] targeted in
                guard let self, self.presentation == .resting else { return }
                self.resize(to: targeted ? NSSize(width: 320, height: 72) : self.restingSize, animate: true)
            }

        NotificationCenter.default.addObserver(
            self,
            selector: #selector(screenParametersChanged),
            name: NSApplication.didChangeScreenParametersNotification,
            object: nil
        )
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    deinit {
        NotificationCenter.default.removeObserver(self)
        peekCollapseTask?.cancel()
    }

    func showResting() {
        let shouldRestoreFocus = NSWorkspace.shared.frontmostApplication?.processIdentifier
            == ProcessInfo.processInfo.processIdentifier
        presentation = .resting
        viewModel.isExpanded = false
        if !viewModel.hasUnread {
            viewModel.proactivePeek = nil
        }
        resize(to: restingSize, animate: true)
        updateContent()
        window?.orderFrontRegardless()

        if shouldRestoreFocus {
            previouslyActiveApplication?.activate(options: [])
        }
    }

    func showExpanded(focusComposer: Bool = true) {
        peekCollapseTask?.cancel()
        let pendingProactiveMessage = viewModel.hasUnread ? viewModel.proactivePeek : nil
        presentation = .expanded
        viewModel.isExpanded = true
        if let pendingProactiveMessage {
            viewModel.receiveProactiveMessage(pendingProactiveMessage)
        } else {
            viewModel.proactivePeek = nil
        }
        resize(to: expandedSize(), animate: true)
        updateContent()
        window?.orderFrontRegardless()

        if focusComposer {
            let frontmost = NSWorkspace.shared.frontmostApplication
            if frontmost?.processIdentifier != ProcessInfo.processInfo.processIdentifier {
                previouslyActiveApplication = frontmost
            }
            NSApp.activate(ignoringOtherApps: true)
            window?.makeKey()
            viewModel.composerFocusRequest += 1
        }
    }

    func showProactivePeek(_ message: String, urgency: ProactiveUrgency = .normal) {
        if isCaptureSuspended {
            if presentation == .expanded {
                viewModel.receiveProactiveMessage(message)
            } else {
                viewModel.proactivePeek = message
                viewModel.hasUnread = true
            }
            return
        }

        guard presentation != .expanded else {
            viewModel.receiveProactiveMessage(message)
            return
        }

        peekCollapseTask?.cancel()
        presentation = .peek
        viewModel.proactivePeek = message
        viewModel.hasUnread = true
        resize(to: peekSize, animate: true)
        updateContent()
        window?.orderFrontRegardless()

        if urgency == .timeSensitive {
            timeSensitiveFallback?(message)
        }

        peekCollapseTask = Task { [weak self] in
            try? await Task.sleep(for: .seconds(7))
            guard !Task.isCancelled else { return }
            await MainActor.run {
                self?.showResting()
            }
        }
    }

    func toggle() {
        guard !isCaptureSuspended else { return }
        switch presentation {
        case .expanded:
            showResting()
        case .resting, .peek:
            showExpanded()
        }
    }

    func captureArea() {
        guard !isCaptureSuspended else { return }
        showExpanded(focusComposer: false)
        viewModel.captureArea()
    }

    func hideForCapture(then start: @escaping @MainActor () -> Void) {
        guard !isCaptureSuspended else { return }
        isCaptureSuspended = true
        captureWasKey = window?.isKeyWindow == true
        window?.orderOut(nil)

        // Give WindowServer one frame to remove omega before screencapture samples the display.
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.12) {
            start()
        }
    }

    func restoreAfterCapture() {
        guard isCaptureSuspended else { return }
        isCaptureSuspended = false
        resize(to: presentationSize(), animate: false)
        updateContent()
        window?.orderFrontRegardless()

        if captureWasKey {
            NSApp.activate(ignoringOtherApps: true)
            window?.makeKey()
            viewModel.composerFocusRequest += 1
        }
        captureWasKey = false
    }

    @objc private func screenParametersChanged() {
        position(size: window?.frame.size ?? restingSize)
    }

    private func updateContent() {
        guard let panel = window as? OmegaPanel else { return }
        panel.contentView = NSHostingView(
            rootView: TrayRootView(
                viewModel: viewModel,
                presentation: presentation,
                close: { [weak self] in self?.showResting() },
                open: { [weak self] in self?.showExpanded() }
            )
        )
    }

    private func resize(to size: NSSize, animate: Bool) {
        guard let window else { return }
        let targetFrame = frame(for: size)

        if animate, !NSWorkspace.shared.accessibilityDisplayShouldReduceMotion {
            NSAnimationContext.runAnimationGroup { context in
                context.duration = 0.24
                context.timingFunction = CAMediaTimingFunction(controlPoints: 0.22, 0.82, 0.2, 1)
                window.animator().setFrame(targetFrame, display: true)
            }
        } else {
            window.setFrame(targetFrame, display: true)
        }
    }

    private func position(size: NSSize) {
        window?.setFrame(frame(for: size), display: true)
    }

    private func expandedSize() -> NSSize {
        let messageHeight = min(CGFloat(viewModel.messages.count) * 54, 130)
        let contextHeight: CGFloat = viewModel.stagedContext.isEmpty ? 0 : 76
        return NSSize(width: 460, height: min(540, 380 + messageHeight + contextHeight))
    }

    private func presentationSize() -> NSSize {
        switch presentation {
        case .resting: restingSize
        case .peek: peekSize
        case .expanded: expandedSize()
        }
    }

    private func frame(for size: NSSize) -> NSRect {
        let screen = activeScreen()
        let origin = NSPoint(
            x: screen.frame.midX - size.width / 2,
            y: screen.frame.maxY - size.height
        )
        return NSRect(origin: origin, size: size)
    }

    private func activeScreen() -> NSScreen {
        let pointer = NSEvent.mouseLocation
        return NSScreen.screens.first(where: { NSMouseInRect(pointer, $0.frame, false) })
            ?? window?.screen
            ?? NSScreen.main
            ?? NSScreen.screens[0]
    }
}

private final class OmegaPanel: NSPanel {
    override var canBecomeKey: Bool { true }
    override var canBecomeMain: Bool { false }
}
