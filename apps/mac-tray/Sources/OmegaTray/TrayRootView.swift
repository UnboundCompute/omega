import SwiftUI
import UniformTypeIdentifiers

struct TrayRootView: View {
    @ObservedObject var viewModel: TrayViewModel
    let presentation: OmegaPanelController.Presentation
    let close: () -> Void
    let open: () -> Void

    var body: some View {
        Group {
            switch presentation {
            case .resting:
                RestingView(
                    hasUnread: viewModel.hasUnread,
                    isDropTargeted: viewModel.isDropTargeted,
                    open: open
                )
            case .peek:
                ProactivePeekView(message: viewModel.displayedProactivePeek, open: open)
            case .expanded:
                if viewModel.isPrivacyRestricted {
                    PrivacyRestrictedView(close: close)
                } else {
                    ExpandedTrayView(viewModel: viewModel, close: close)
                }
            }
        }
        .onDrop(
            of: [UTType.fileURL, UTType.url, UTType.image, UTType.plainText],
            isTargeted: $viewModel.isDropTargeted,
            perform: viewModel.importProviders
        )
    }
}

private struct PrivacyRestrictedView: View {
    let close: () -> Void

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Spacer()
                Capsule()
                    .fill(Color.white.opacity(0.13))
                    .frame(width: 34, height: 3)
                Spacer()
                Button(action: close) {
                    Image(systemName: "xmark")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundStyle(TrayTheme.secondaryText)
                        .frame(width: 26, height: 26)
                        .background(Color.white.opacity(0.06), in: Circle())
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Close omega")
                .padding(.trailing, 10)
            }
            .frame(height: 38)
            .background(TrayTheme.shell)

            VStack(spacing: 10) {
                Image(systemName: "eye.slash")
                    .font(.system(size: 21, weight: .light))
                    .foregroundStyle(TrayTheme.signal)
                Text("Privacy veil is on")
                    .font(.system(size: 15, weight: .semibold))
                    .foregroundStyle(TrayTheme.primaryText)
                Text("Conversation and context are hidden. Turn off Privacy Veil from the omega menu-bar item when you’re ready.")
                    .font(.system(size: 12))
                    .foregroundStyle(TrayTheme.secondaryText)
                    .multilineTextAlignment(.center)
                    .frame(maxWidth: 300)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(TrayTheme.surface)
        }
        .clipShape(UnevenRoundedRectangle(bottomLeadingRadius: 22, bottomTrailingRadius: 22))
        .overlay {
            UnevenRoundedRectangle(bottomLeadingRadius: 22, bottomTrailingRadius: 22)
                .stroke(Color.white.opacity(0.09), lineWidth: 0.75)
        }
        .shadow(color: .black.opacity(0.34), radius: 28, y: 14)
        .accessibilityElement(children: .contain)
    }
}

private struct RestingView: View {
    let hasUnread: Bool
    let isDropTargeted: Bool
    let open: () -> Void

    var body: some View {
        Button(action: open) {
            ZStack(alignment: .bottom) {
                UnevenRoundedRectangle(
                    bottomLeadingRadius: 14,
                    bottomTrailingRadius: 14
                )
                .fill(TrayTheme.shell)

                if isDropTargeted {
                    HStack(spacing: 8) {
                        Image(systemName: "arrow.down.doc")
                        Text("Drop to stage — nothing sends yet")
                    }
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(TrayTheme.primaryText)
                    .padding(.bottom, 18)
                } else {
                    SignalSeam(isVisible: hasUnread)
                        .padding(.horizontal, 74)
                        .padding(.bottom, 2)
                }
            }
        }
        .buttonStyle(.plain)
        .accessibilityLabel(
            isDropTargeted
                ? "Drop context into omega. It will be staged, not sent."
                : (hasUnread ? "Open omega, unread message" : "Open omega")
        )
    }
}

private struct ProactivePeekView: View {
    let message: String
    let open: () -> Void

    var body: some View {
        Button(action: open) {
            VStack(spacing: 0) {
                TrayTheme.shell.frame(height: 34)
                HStack(alignment: .top, spacing: 10) {
                    OmegaGlyph()
                        .frame(width: 22, height: 22)
                    Text(message)
                        .font(.system(size: 13, weight: .medium))
                        .foregroundStyle(TrayTheme.primaryText)
                        .multilineTextAlignment(.leading)
                        .lineLimit(2)
                    Spacer(minLength: 0)
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 13)
                .background(TrayTheme.surface)
            }
            .clipShape(UnevenRoundedRectangle(bottomLeadingRadius: 18, bottomTrailingRadius: 18))
            .shadow(color: .black.opacity(0.32), radius: 22, y: 10)
        }
        .buttonStyle(.plain)
        .accessibilityLabel("Message from omega: \(message)")
    }
}

private struct ExpandedTrayView: View {
    @ObservedObject var viewModel: TrayViewModel
    let close: () -> Void
    @FocusState private var composerFocused: Bool
    @Environment(\.accessibilityReduceTransparency) private var reduceTransparency
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @Environment(\.controlActiveState) private var controlActiveState

    var body: some View {
        VStack(spacing: 0) {
            cameraHeader
            VStack(spacing: 0) {
                statusRow
                Divider().overlay(Color.white.opacity(0.07))
                conversation
                if !viewModel.stagedContext.isEmpty {
                    contextShelf
                }
                composer
            }
            .background(TrayTheme.surface.opacity(reduceTransparency ? 1 : 0.97))
        }
        .clipShape(UnevenRoundedRectangle(bottomLeadingRadius: 22, bottomTrailingRadius: 22))
        .overlay {
            UnevenRoundedRectangle(bottomLeadingRadius: 22, bottomTrailingRadius: 22)
                .stroke(Color.white.opacity(0.09), lineWidth: 0.75)
        }
        .shadow(color: .black.opacity(0.34), radius: 28, y: 14)
        .opacity(controlActiveState == .inactive ? 0.94 : 1)
        .overlay(alignment: .bottom) {
            WorkingSeam(isActive: viewModel.workState.isBusy)
                .padding(.bottom, 2)
        }
        .overlay {
            if viewModel.isDropTargeted {
                dropOverlay
            }
        }
        .onChange(of: viewModel.composerFocusRequest) {
            composerFocused = true
        }
    }

    private var cameraHeader: some View {
        HStack {
            Spacer()
            Capsule()
                .fill(Color.white.opacity(0.13))
                .frame(width: 34, height: 3)
                .accessibilityHidden(true)
            Spacer()
            Button(action: close) {
                Image(systemName: "xmark")
                    .font(.system(size: 10, weight: .semibold))
                    .foregroundStyle(TrayTheme.secondaryText)
                    .frame(width: 26, height: 26)
                    .background(Color.white.opacity(0.06), in: Circle())
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Close omega")
            .padding(.trailing, 10)
        }
        .frame(height: 38)
        .background(TrayTheme.shell)
    }

    private var statusRow: some View {
        HStack(spacing: 9) {
            OmegaGlyph()
                .frame(width: 22, height: 22)
            VStack(alignment: .leading, spacing: 1) {
                Text("omega")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(TrayTheme.primaryText)
                Text(viewModel.statusLabel)
                    .font(.system(size: 11))
                    .foregroundStyle(statusColor)
            }
            Spacer()
            Menu {
                Button("Capture area", systemImage: "viewfinder") {
                    viewModel.captureScreen(.area)
                }
                Button("Capture window", systemImage: "macwindow") {
                    viewModel.captureScreen(.window)
                }
                Button("Capture active display", systemImage: "display") {
                    viewModel.captureScreen(.display)
                }
            } label: {
                Label("Capture", systemImage: "viewfinder")
                    .labelStyle(.titleAndIcon)
                    .font(.system(size: 11, weight: .medium))
            }
            .buttonStyle(QuietButtonStyle())
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
    }

    private var conversation: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 14) {
                    notices
                    if viewModel.messages.isEmpty {
                        emptyState
                    } else {
                        ForEach(viewModel.messages) { message in
                            MessageView(message: message)
                                .id(message.id)
                        }
                    }
                }
                .padding(16)
            }
            .onChange(of: viewModel.messages.count) {
                guard let id = viewModel.messages.last?.id else { return }
                if reduceMotion {
                    proxy.scrollTo(id, anchor: .bottom)
                } else {
                    withAnimation(.timingCurve(0.22, 0.82, 0.2, 1, duration: 0.2)) {
                        proxy.scrollTo(id, anchor: .bottom)
                    }
                }
            }
        }
        .frame(maxHeight: .infinity)
    }

    @ViewBuilder
    private var notices: some View {
        if viewModel.connectionState == .disconnected {
            RecoveryNotice(
                icon: "bolt.horizontal.circle",
                title: "The agent is offline",
                detail: "Start omega’s local agent. The tray will reconnect automatically.",
                actionTitle: nil,
                action: nil
            )
        }

        if viewModel.hasStagedContextWithoutContentTransport {
            RecoveryNotice(
                icon: "eye.slash",
                title: "Attachment viewing isn’t connected yet",
                detail: "omega will receive item names and types, but cannot read their contents yet. Add the important details in your message.",
                actionTitle: nil,
                action: nil
            )
        }

        if viewModel.capturePermission == .denied {
            RecoveryNotice(
                icon: "rectangle.on.rectangle.slash",
                title: "Screen capture is off",
                detail: "Allow Screen & System Audio Recording in System Settings. If you just enabled it, quit and reopen omega once.",
                actionTitle: "Open Settings",
                action: viewModel.openScreenCaptureSettings
            )
        }

        if case .failed(let detail) = viewModel.localWorkState {
            RecoveryNotice(
                icon: "viewfinder.circle",
                title: "Capture failed",
                detail: detail,
                actionTitle: nil,
                action: nil
            )
        }

        if let failure = viewModel.hotKeyRegistrationFailure {
            RecoveryNotice(
                icon: "keyboard.badge.exclamationmark",
                title: "A shortcut is already in use",
                detail: failure,
                actionTitle: "Settings",
                action: openAppSettings
            )
        }

        if case .failed(let detail) = viewModel.turnState {
            if viewModel.canRetryLastSend {
                RecoveryNotice(
                    icon: "arrow.clockwise",
                    title: "Not delivered",
                    detail: detail,
                    actionTitle: "Retry",
                    action: viewModel.retryLastSend
                )
            } else {
                RecoveryNotice(
                    icon: "exclamationmark.circle",
                    title: "Agent task failed",
                    detail: detail,
                    actionTitle: nil,
                    action: nil
                )
            }
        }
    }

    private var emptyState: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("What are we doing?")
                .font(.system(size: 19, weight: .semibold))
                .foregroundStyle(TrayTheme.primaryText)
            Text("Ask or delegate directly, or stage context to keep with your message. Nothing is sent until you submit it.")
                .font(.system(size: 13))
                .foregroundStyle(TrayTheme.secondaryText)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(.top, 8)
    }

    private var contextShelf: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 9) {
                ForEach(viewModel.stagedContext) { context in
                    ContextCard(
                        context: context,
                        preview: { viewModel.previewContext(context) },
                        remove: { viewModel.removeContext(id: context.id) }
                    )
                }
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 12)
        }
        .background(TrayTheme.raised.opacity(0.72))
        .overlay(alignment: .top) {
            Divider().overlay(Color.white.opacity(0.07))
        }
    }

    private var composer: some View {
        HStack(alignment: .bottom, spacing: 10) {
            Button(action: viewModel.pasteFromClipboard) {
                Image(systemName: "paperclip")
                    .frame(width: 28, height: 28)
            }
            .buttonStyle(.plain)
            .foregroundStyle(TrayTheme.secondaryText)
            .accessibilityLabel("Add from clipboard")

            TextField("Ask or delegate…", text: $viewModel.draft, axis: .vertical)
                .textFieldStyle(.plain)
                .font(.system(size: 13))
                .foregroundStyle(TrayTheme.primaryText)
                .lineLimit(1...5)
                .focused($composerFocused)
                .onSubmit(viewModel.send)

            Button(action: viewModel.send) {
                Group {
                    if viewModel.workState.isBusy {
                        ProgressView()
                            .controlSize(.small)
                    } else {
                        Image(systemName: "arrow.up")
                            .font(.system(size: 12, weight: .bold))
                    }
                }
                .foregroundStyle(viewModel.canSubmit ? TrayTheme.shell : TrayTheme.tertiaryText)
                .frame(width: 29, height: 29)
                .background(viewModel.canSubmit ? TrayTheme.signal : TrayTheme.raised, in: Circle())
            }
            .buttonStyle(.plain)
            .disabled(!viewModel.canSubmit)
            .accessibilityLabel("Send to omega")
        }
        .padding(10)
        .background(TrayTheme.instruction, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .stroke(composerFocused ? TrayTheme.signal.opacity(0.74) : Color.white.opacity(0.06), lineWidth: 1)
        }
        .padding(12)
    }

    private var dropOverlay: some View {
        VStack(spacing: 10) {
            Image(systemName: "arrow.down.doc")
                .font(.system(size: 24, weight: .light))
            Text("Drop into omega")
                .font(.system(size: 15, weight: .semibold))
            Text("It will be staged, not sent")
                .font(.system(size: 12))
                .foregroundStyle(TrayTheme.secondaryText)
        }
        .foregroundStyle(TrayTheme.primaryText)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(TrayTheme.shell.opacity(0.95))
        .overlay {
            RoundedRectangle(cornerRadius: 22, style: .continuous)
                .stroke(TrayTheme.signal.opacity(0.85), lineWidth: 2)
                .padding(8)
        }
        .allowsHitTesting(false)
        .accessibilityLabel("Drop context into omega. Items will not be sent automatically.")
    }

    private var statusColor: Color {
        switch viewModel.workState {
        case .failed: .red
        case .blocked: .orange
        case .complete: .green
        case .working, .sending, .understood: TrayTheme.signal
        case .ready:
            viewModel.connectionState == .disconnected ? .orange : TrayTheme.secondaryText
        }
    }

    private func openAppSettings() {
        NSApp.sendAction(Selector(("showSettingsWindow:")), to: nil, from: nil)
        NSApp.activate(ignoringOtherApps: true)
    }
}

private struct MessageView: View {
    let message: TrayMessage

    var body: some View {
        switch message.role {
        case .user:
            VStack(alignment: .trailing, spacing: 6) {
                Text(message.text)
                    .font(.system(size: 13))
                    .foregroundStyle(TrayTheme.primaryText)

                if !message.contextDescriptions.isEmpty {
                    Text(message.contextDescriptions.joined(separator: " · "))
                        .font(.system(size: 10, weight: .medium))
                        .foregroundStyle(TrayTheme.secondaryText)
                        .lineLimit(2)
                }

                if message.delivery == .failed {
                    Label("Not delivered", systemImage: "exclamationmark.circle")
                        .font(.system(size: 10, weight: .medium))
                        .foregroundStyle(.red)
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 9)
            .background(TrayTheme.instruction, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
            .frame(maxWidth: .infinity, alignment: .trailing)
        case .omega:
            VStack(alignment: .leading, spacing: 7) {
                Text("omega")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(TrayTheme.signal)
                Text(message.text)
                    .font(.system(size: 13))
                    .foregroundStyle(TrayTheme.primaryText)
                    .textSelection(.enabled)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        case .status:
            Text(message.text)
                .font(.system(size: 11))
                .foregroundStyle(TrayTheme.secondaryText)
        }
    }
}

private struct ContextCard: View {
    let context: StagedContext
    let preview: () -> Void
    let remove: () -> Void

    var body: some View {
        HStack(spacing: 9) {
            Button(action: preview) {
                Group {
                    if let preview = context.preview {
                        Image(nsImage: preview)
                            .resizable()
                            .scaledToFill()
                    } else {
                        Image(systemName: context.kind == .link ? "link" : "text.alignleft")
                            .foregroundStyle(TrayTheme.signal)
                    }
                }
                .frame(width: 34, height: 34)
                .background(TrayTheme.surface, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
                .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
            }
            .buttonStyle(.plain)
            .disabled(context.fileURL == nil)
            .accessibilityLabel("Preview \(context.title)")

            VStack(alignment: .leading, spacing: 2) {
                Text(context.title)
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(TrayTheme.primaryText)
                    .lineLimit(1)
                Text(context.detail)
                    .font(.system(size: 10))
                    .foregroundStyle(TrayTheme.secondaryText)
            }

            Button(action: remove) {
                Image(systemName: "xmark.circle.fill")
                    .foregroundStyle(TrayTheme.tertiaryText)
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Remove \(context.title)")
        }
        .padding(8)
        .frame(width: 190)
        .background(TrayTheme.instruction, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
    }
}

private struct RecoveryNotice: View {
    let icon: String
    let title: String
    let detail: String
    let actionTitle: String?
    let action: (() -> Void)?

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: icon)
                .foregroundStyle(TrayTheme.signal)
                .frame(width: 18)
            VStack(alignment: .leading, spacing: 3) {
                Text(title)
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(TrayTheme.primaryText)
                Text(detail)
                    .font(.system(size: 11))
                    .foregroundStyle(TrayTheme.secondaryText)
                    .fixedSize(horizontal: false, vertical: true)
            if let actionTitle, let action {
                Button(actionTitle, action: action)
                    .buttonStyle(.link)
                    .font(.system(size: 11, weight: .medium))
            }
            }
        }
        .padding(12)
        .background(TrayTheme.raised, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
    }
}

private struct OmegaGlyph: View {
    var body: some View {
        Canvas { context, size in
            var path = Path()
            path.move(to: CGPoint(x: size.width * 0.18, y: size.height * 0.38))
            path.addCurve(
                to: CGPoint(x: size.width * 0.5, y: size.height * 0.78),
                control1: CGPoint(x: size.width * 0.16, y: size.height * 0.68),
                control2: CGPoint(x: size.width * 0.32, y: size.height * 0.78)
            )
            path.addCurve(
                to: CGPoint(x: size.width * 0.82, y: size.height * 0.38),
                control1: CGPoint(x: size.width * 0.68, y: size.height * 0.78),
                control2: CGPoint(x: size.width * 0.84, y: size.height * 0.68)
            )
            context.stroke(path, with: .color(TrayTheme.signal), style: .init(lineWidth: 1.8, lineCap: .round))

            var base = Path()
            base.move(to: CGPoint(x: size.width * 0.25, y: size.height * 0.82))
            base.addLine(to: CGPoint(x: size.width * 0.75, y: size.height * 0.82))
            context.stroke(base, with: .color(TrayTheme.signal.opacity(0.72)), style: .init(lineWidth: 1.4, lineCap: .round))
        }
        .accessibilityHidden(true)
    }
}

private struct SignalSeam: View {
    let isVisible: Bool

    var body: some View {
        Capsule()
            .fill(isVisible ? TrayTheme.signal : Color.white.opacity(0.12))
            .frame(height: 2)
            .shadow(color: isVisible ? TrayTheme.signal.opacity(0.35) : .clear, radius: 4)
    }
}

private struct WorkingSeam: View {
    let isActive: Bool
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var bright = false

    var body: some View {
        Capsule()
            .fill(isActive ? TrayTheme.signal : Color.clear)
            .frame(width: bright ? 72 : 42, height: 2)
            .opacity(isActive ? (bright ? 0.95 : 0.42) : 0)
            .animation(
                reduceMotion || !isActive
                    ? nil
                    : .timingCurve(0.22, 0.82, 0.2, 1, duration: 0.8).repeatForever(autoreverses: true),
                value: bright
            )
            .onAppear { bright = isActive }
            .onChange(of: isActive) { bright = isActive }
            .accessibilityHidden(true)
    }
}

private struct QuietButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .foregroundStyle(TrayTheme.secondaryText)
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(
                configuration.isPressed ? Color.white.opacity(0.1) : Color.white.opacity(0.055),
                in: Capsule()
            )
    }
}
