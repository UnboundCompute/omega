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
        let identifiers = HotKeyChoice.choices.map(\.id)
        XCTAssertEqual(Set(identifiers).count, identifiers.count)
    }
}
