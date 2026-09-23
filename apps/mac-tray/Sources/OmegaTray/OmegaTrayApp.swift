import SwiftUI

@main
struct OmegaTrayApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate

    var body: some Scene {
        Settings {
            SettingsView()
        }
    }
}

private struct SettingsView: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("omega")
                .font(.title2.weight(.semibold))
            Text("The first prototype uses Control–Option–Space to open or close the tray.")
                .foregroundStyle(.secondary)
            Text("Shortcut customization and launch-at-login controls are part of the v1 contract and will land after the core panel behavior is verified.")
                .font(.callout)
                .foregroundStyle(.secondary)
        }
        .frame(width: 420)
        .padding(24)
    }
}
