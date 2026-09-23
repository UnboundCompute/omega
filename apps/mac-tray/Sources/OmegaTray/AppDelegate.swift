import AppKit
import Combine

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    private let viewModel = TrayViewModel(transport: LocalDemoTransport())
    private var panelController: OmegaPanelController?
    private var statusItem: NSStatusItem?
    private var hotKey: GlobalHotKey?
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

        switch ProcessInfo.processInfo.environment["OMEGA_TRAY_PREVIEW_STATE"] {
        case "expanded":
            panelController.showExpanded(focusComposer: false)
        case "peek":
            panelController.showProactivePeek("I found something worth your attention while you were working.")
        default:
            panelController.showResting()
        }

        registerHotKey()
        settingsObservation = settings.$hotKeyID
            .dropFirst()
            .sink { [weak self] _ in self?.registerHotKey() }

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
    }

    func applicationWillTerminate(_ notification: Notification) {
        hotKey = nil
        DistributedNotificationCenter.default.removeObserver(self)
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
        panelController?.showExpanded(focusComposer: false)
        viewModel.captureArea()
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

    private func registerHotKey() {
        hotKey = nil
        let configuration = settings.hotKey
        hotKey = GlobalHotKey(
            keyCode: configuration.keyCode,
            modifiers: configuration.modifiers
        ) { [weak self] in
            Task { @MainActor in self?.panelController?.toggle() }
        }
        viewModel.hotKeyRegistrationFailed = hotKey == nil
    }
}
