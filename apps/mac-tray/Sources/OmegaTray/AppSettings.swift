import Carbon.HIToolbox
import Combine
import Foundation
import ServiceManagement

struct HotKeyChoice: Identifiable, Equatable {
    let id: String
    let title: String
    let keyCode: UInt32
    let modifiers: UInt32

    static let choices = [
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

    static let fallback = choices[0]
}

@MainActor
final class AppSettings: ObservableObject {
    static let shared = AppSettings()

    @Published var hotKeyID: String {
        didSet { defaults.set(hotKeyID, forKey: Keys.hotKeyID) }
    }
    @Published var hideProactivePreviews: Bool {
        didSet { defaults.set(hideProactivePreviews, forKey: Keys.hideProactivePreviews) }
    }
    @Published private(set) var launchAtLogin = false
    @Published private(set) var launchAtLoginMessage: String?

    private let defaults: UserDefaults

    var hotKey: HotKeyChoice {
        HotKeyChoice.choices.first { $0.id == hotKeyID } ?? .fallback
    }

    private enum Keys {
        static let hotKeyID = "hotKeyID"
        static let hideProactivePreviews = "hideProactivePreviews"
    }

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
        hotKeyID = defaults.string(forKey: Keys.hotKeyID) ?? HotKeyChoice.fallback.id
        hideProactivePreviews = defaults.object(forKey: Keys.hideProactivePreviews) as? Bool ?? true
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
