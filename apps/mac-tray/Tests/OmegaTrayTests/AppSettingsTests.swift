import XCTest
@testable import OmegaTray

final class AppSettingsTests: XCTestCase {
    @MainActor
    func testHotKeyChoicePersists() {
        let suiteName = "OmegaTrayTests.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!
        defer { defaults.removePersistentDomain(forName: suiteName) }

        let settings = AppSettings(defaults: defaults)
        settings.hotKeyID = "control-space"

        XCTAssertEqual(defaults.string(forKey: "hotKeyID"), "control-space")
        XCTAssertEqual(settings.hotKey.title, "Control–Space")
    }

    func testEveryShortcutChoiceHasAUniqueIdentifier() {
        let identifiers = (HotKeyChoice.panelChoices + HotKeyChoice.captureAreaChoices).map(\.id)
        XCTAssertEqual(Set(identifiers).count, identifiers.count)
    }

    @MainActor
    func testCaptureAreaHotKeyPersists() {
        let suiteName = "OmegaTrayTests.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!
        defer { defaults.removePersistentDomain(forName: suiteName) }

        let settings = AppSettings(defaults: defaults)
        settings.captureAreaHotKeyID = "option-shift-4"

        XCTAssertEqual(defaults.string(forKey: "captureAreaHotKeyID"), "option-shift-4")
        XCTAssertEqual(settings.captureAreaHotKey.title, "Option–Shift–4")
    }
}
