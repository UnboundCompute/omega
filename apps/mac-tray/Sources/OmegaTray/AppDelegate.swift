import AppKit
import Carbon.HIToolbox

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    private let viewModel = TrayViewModel(transport: LocalDemoTransport())
    private var panelController: OmegaPanelController?
    private var statusItem: NSStatusItem?
    private var hotKey: GlobalHotKey?

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.accessory)

        let panelController = OmegaPanelController(viewModel: viewModel)
        self.panelController = panelController

        switch ProcessInfo.processInfo.environment["OMEGA_TRAY_PREVIEW_STATE"] {
        case "expanded":
            panelController.showExpanded(focusComposer: false)
        case "peek":
            panelController.showProactivePeek("I found something worth your attention while you were working.")
        default:
            panelController.showResting()
        }

        hotKey = GlobalHotKey(keyCode: UInt32(kVK_Space), modifiers: UInt32(controlKey | optionKey)) { [weak self] in
            Task { @MainActor in
                self?.panelController?.toggle()
            }
        }

        installStatusItem()
    }

    func applicationWillTerminate(_ notification: Notification) {
        hotKey = nil
    }

    private func installStatusItem() {
        let item = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)
        item.button?.image = NSImage(systemSymbolName: "circle.bottomhalf.filled", accessibilityDescription: "omega")
        item.button?.toolTip = "omega"

        let menu = NSMenu()
        menu.addItem(withTitle: "Open omega", action: #selector(togglePanel), keyEquivalent: "")
        menu.addItem(withTitle: "Capture area…", action: #selector(captureArea), keyEquivalent: "")
        menu.addItem(withTitle: "Show proactive peek", action: #selector(showProactivePeek), keyEquivalent: "")
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
}
