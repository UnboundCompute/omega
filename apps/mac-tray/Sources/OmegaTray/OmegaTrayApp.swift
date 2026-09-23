import SwiftUI

@main
struct OmegaTrayApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate

    var body: some Scene {
        Settings {
            SettingsView(settings: .shared)
        }
    }
}

private struct SettingsView: View {
    @ObservedObject var settings: AppSettings

    var body: some View {
        Form {
            Section("Presence") {
                Picker("Open or close omega", selection: $settings.hotKeyID) {
                    ForEach(HotKeyChoice.panelChoices) { choice in
                        Text(choice.title).tag(choice.id)
                    }
                }

                Picker("Capture an area", selection: $settings.captureAreaHotKeyID) {
                    ForEach(HotKeyChoice.captureAreaChoices) { choice in
                        Text(choice.title).tag(choice.id)
                    }
                }

                Toggle(
                    "Open omega when I log in",
                    isOn: Binding(
                        get: { settings.launchAtLogin },
                        set: settings.setLaunchAtLogin
                    )
                )

                if let message = settings.launchAtLoginMessage {
                    VStack(alignment: .leading, spacing: 6) {
                        Text(message)
                            .font(.callout)
                            .foregroundStyle(.secondary)
                        Button("Open Login Items") {
                            settings.openLoginItemSettings()
                        }
                    }
                }
            }

            Section("Privacy") {
                Toggle("Hide proactive message previews", isOn: $settings.hideProactivePreviews)
                Text("When enabled, the camera-area peek only says that omega has something for you. Open the tray to read it.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }

            Section {
                Text("The tray collects context and shows omega’s replies. Memory, judgement, and tool execution remain in the single agent core.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
        .frame(width: 480)
        .fixedSize(horizontal: false, vertical: true)
        .onAppear { settings.refreshLaunchAtLogin() }
    }
}
