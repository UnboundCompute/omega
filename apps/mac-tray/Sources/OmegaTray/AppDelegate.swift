import AppKit
import Combine

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    private let transport = OmegaChannelClient()
    private lazy var viewModel = TrayViewModel(transport: transport)
    private var panelController: OmegaPanelController?
    private var statusItem: NSStatusItem?
    private var panelHotKey: GlobalHotKey?
    private var captureAreaHotKey: GlobalHotKey?
    private var privacyMenuItem: NSMenuItem?
    private var settingsObservation: AnyCancellable?
    private var notificationController: NotificationController?
    private let settings = AppSettings.shared

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.accessory)

        let panelController = OmegaPanelController(viewModel: viewModel)
        self.panelController = panelController
        let notificationController = NotificationController { [weak panelController] in
            panelController?.showExpanded()
        }
        self.notificationController = notificationController
        panelController.timeSensitiveFallback = { [weak notificationController] message in
            notificationController?.deliverTimeSensitiveFallback(message)
        }
        viewModel.proactivePresentation = { [weak panelController] message, urgency in
            panelController?.showProactivePeek(message, urgency: urgency)
        }

        switch ProcessInfo.processInfo.environment["OMEGA_TRAY_PREVIEW_STATE"] {
        case "expanded":
            panelController.showExpanded(focusComposer: false)
        case "peek":
            panelController.showProactivePeek("I found something worth your attention while you were working.")
        default:
            panelController.showResting()
        }

        registerHotKeys()
        settingsObservation = Publishers.CombineLatest(
            settings.$hotKeyID,
            settings.$captureAreaHotKeyID
        )
            .dropFirst()
            .sink { [weak self] _ in self?.registerHotKeys() }

        DistributedNotificationCenter.default.addObserver(
            self,
            selector: #selector(screenLocked),
            name: .init("com.apple.screenIsLocked"),
            object: nil
        )
        DistributedNotificationCenter.default.addObserver(
            self,
            selector: #selector(screenUnlocked),
            name: .init("com.apple.screenIsUnlocked"),
            object: nil
        )

        installStatusItem()
        viewModel.start()
    }

    func applicationWillTerminate(_ notification: Notification) {
        panelHotKey = nil
        captureAreaHotKey = nil
        viewModel.stop()
        DistributedNotificationCenter.default.removeObserver(self)
    }

    func applicationDidBecomeActive(_ notification: Notification) {
        viewModel.refreshScreenCapturePermission()
    }

    private func installStatusItem() {
        let item = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)
        item.button?.image = OmegaStatusIcon.make()
        item.button?.toolTip = "omega"

        let menu = NSMenu()
        menu.addItem(withTitle: "Open omega", action: #selector(togglePanel), keyEquivalent: "")
        menu.addItem(withTitle: "Capture area…", action: #selector(captureArea), keyEquivalent: "")
        menu.addItem(withTitle: "Show proactive peek", action: #selector(showProactivePeek), keyEquivalent: "")
        let privacyItem = menu.addItem(
            withTitle: "Turn On Privacy Veil",
            action: #selector(togglePrivacyMode),
            keyEquivalent: ""
        )
        privacyMenuItem = privacyItem
        menu.addItem(.separator())
        menu.addItem(withTitle: "Settings…", action: #selector(openSettings), keyEquivalent: ",")
        menu.addItem(withTitle: "Quit omega", action: #selector(quit), keyEquivalent: "q")
        menu.items.forEach { $0.target = self }

        item.menu = menu
        statusItem = item
    }

    @objc private func togglePanel() {
        panelController?.toggle()
    }

    @objc private func captureArea() {
        panelController?.captureArea()
    }

    @objc private func showProactivePeek() {
        panelController?.showProactivePeek(
            "You may want to revisit the screen-capture permission before the first real task."
        )
    }

    @objc private func openSettings() {
        NSApp.sendAction(Selector(("showSettingsWindow:")), to: nil, from: nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    @objc private func quit() {
        NSApp.terminate(nil)
    }

    @objc private func screenLocked() {
        viewModel.isScreenLocked = true
        panelController?.showResting()
    }

    @objc private func screenUnlocked() {
        viewModel.isScreenLocked = false
    }

    @objc private func togglePrivacyMode() {
        viewModel.manualPrivacyMode.toggle()
        privacyMenuItem?.title = viewModel.manualPrivacyMode
            ? "Turn Off Privacy Veil"
            : "Turn On Privacy Veil"
    }

    private func registerHotKeys() {
        panelHotKey = nil
        captureAreaHotKey = nil

        let panelConfiguration = settings.hotKey
        panelHotKey = GlobalHotKey(
            id: 1,
            keyCode: panelConfiguration.keyCode,
            modifiers: panelConfiguration.modifiers
        ) { [weak self] in
            Task { @MainActor in self?.panelController?.toggle() }
        }

        let captureConfiguration = settings.captureAreaHotKey
        captureAreaHotKey = GlobalHotKey(
            id: 2,
            keyCode: captureConfiguration.keyCode,
            modifiers: captureConfiguration.modifiers
        ) { [weak self] in
            Task { @MainActor in self?.captureArea() }
        }

        let failures = [
            panelHotKey == nil ? "Open or close omega (\(panelConfiguration.title))" : nil,
            captureAreaHotKey == nil ? "Capture area (\(captureConfiguration.title))" : nil
        ].compactMap { $0 }
        viewModel.hotKeyRegistrationFailure = failures.isEmpty
            ? nil
            : "\(failures.joined(separator: " and ")) could not be registered. Choose another shortcut in omega Settings."
    }
}
