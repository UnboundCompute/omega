import Carbon.HIToolbox
import Combine
import Foundation
import ServiceManagement

struct HotKeyChoice: Identifiable, Equatable {
    let id: String
    let title: String
    let keyCode: UInt32
    let modifiers: UInt32

    static let panelChoices = [
        HotKeyChoice(
            id: "control-option-space",
            title: "Control–Option–Space",
            keyCode: UInt32(kVK_Space),
            modifiers: UInt32(controlKey | optionKey)
        ),
        HotKeyChoice(
            id: "control-space",
            title: "Control–Space",
            keyCode: UInt32(kVK_Space),
            modifiers: UInt32(controlKey)
        ),
        HotKeyChoice(
            id: "option-space",
            title: "Option–Space",
            keyCode: UInt32(kVK_Space),
            modifiers: UInt32(optionKey)
        ),
        HotKeyChoice(
            id: "command-shift-space",
            title: "Command–Shift–Space",
            keyCode: UInt32(kVK_Space),
            modifiers: UInt32(cmdKey | shiftKey)
        )
    ]

    static let captureAreaChoices = [
        HotKeyChoice(
            id: "control-option-c",
            title: "Control–Option–C",
            keyCode: UInt32(kVK_ANSI_C),
            modifiers: UInt32(controlKey | optionKey)
        ),
        HotKeyChoice(
            id: "control-shift-c",
            title: "Control–Shift–C",
            keyCode: UInt32(kVK_ANSI_C),
            modifiers: UInt32(controlKey | shiftKey)
        ),
        HotKeyChoice(
            id: "option-shift-c",
            title: "Option–Shift–C",
            keyCode: UInt32(kVK_ANSI_C),
            modifiers: UInt32(optionKey | shiftKey)
        ),
        HotKeyChoice(
            id: "command-option-c",
            title: "Command–Option–C",
            keyCode: UInt32(kVK_ANSI_C),
            modifiers: UInt32(cmdKey | optionKey)
        )
    ]

    static let panelFallback = panelChoices[0]
    static let captureAreaFallback = captureAreaChoices[0]
}

@MainActor
final class AppSettings: ObservableObject {
    static let shared = AppSettings()

    @Published var hotKeyID: String {
        didSet { defaults.set(hotKeyID, forKey: Keys.hotKeyID) }
    }
    @Published var captureAreaHotKeyID: String {
        didSet { defaults.set(captureAreaHotKeyID, forKey: Keys.captureAreaHotKeyID) }
    }
    @Published var hideProactivePreviews: Bool {
        didSet { defaults.set(hideProactivePreviews, forKey: Keys.hideProactivePreviews) }
    }
    @Published private(set) var launchAtLogin = false
    @Published private(set) var launchAtLoginMessage: String?

    private let defaults: UserDefaults

    var hotKey: HotKeyChoice {
        HotKeyChoice.panelChoices.first { $0.id == hotKeyID } ?? .panelFallback
    }

    var captureAreaHotKey: HotKeyChoice {
        HotKeyChoice.captureAreaChoices.first { $0.id == captureAreaHotKeyID } ?? .captureAreaFallback
    }

    private enum Keys {
        static let hotKeyID = "hotKeyID"
        static let captureAreaHotKeyID = "captureAreaHotKeyID"
        static let hideProactivePreviews = "hideProactivePreviews"
    }

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
        hotKeyID = defaults.string(forKey: Keys.hotKeyID) ?? HotKeyChoice.panelFallback.id
        let savedCaptureAreaHotKeyID = defaults.string(forKey: Keys.captureAreaHotKeyID)
        if let savedCaptureAreaHotKeyID,
           HotKeyChoice.captureAreaChoices.contains(where: { $0.id == savedCaptureAreaHotKeyID }) {
            captureAreaHotKeyID = savedCaptureAreaHotKeyID
        } else {
            captureAreaHotKeyID = HotKeyChoice.captureAreaFallback.id
        }
        hideProactivePreviews = defaults.object(forKey: Keys.hideProactivePreviews) as? Bool ?? true
        if savedCaptureAreaHotKeyID != captureAreaHotKeyID {
            defaults.set(captureAreaHotKeyID, forKey: Keys.captureAreaHotKeyID)
        }
        refreshLaunchAtLogin()
    }

    func setLaunchAtLogin(_ enabled: Bool) {
        launchAtLoginMessage = nil

        guard Bundle.main.bundleURL.pathExtension == "app" else {
            launchAtLogin = false
            launchAtLoginMessage = "Launch at login is available in the packaged app."
            return
        }

        do {
            if enabled {
                try SMAppService.mainApp.register()
            } else {
                try SMAppService.mainApp.unregister()
            }
            refreshLaunchAtLogin()
        } catch {
            refreshLaunchAtLogin()
            launchAtLoginMessage = "macOS could not update the login item. Open Login Items in System Settings and try again."
        }
    }

    func refreshLaunchAtLogin() {
        launchAtLogin = SMAppService.mainApp.status == .enabled
    }

    func openLoginItemSettings() {
        SMAppService.openSystemSettingsLoginItems()
    }
}
