import UserNotifications

enum ProactiveUrgency {
    case normal
    case timeSensitive
}

@MainActor
final class NotificationController: NSObject, UNUserNotificationCenterDelegate {
    private let openTray: @MainActor () -> Void
    private let center = UNUserNotificationCenter.current()

    init(openTray: @escaping @MainActor () -> Void) {
        self.openTray = openTray
        super.init()
        center.delegate = self
    }

    func deliverTimeSensitiveFallback(_ message: String) {
        Task {
            let settings = await center.notificationSettings()
            var isAllowed = settings.authorizationStatus == .authorized

            if settings.authorizationStatus == .notDetermined {
                isAllowed = (try? await center.requestAuthorization(options: [.alert, .sound])) == true
            }

            guard isAllowed else { return }

            let content = UNMutableNotificationContent()
            content.title = "omega"
            content.body = AppSettings.shared.hideProactivePreviews
                ? "omega has something time-sensitive for you"
                : message
            content.interruptionLevel = .timeSensitive
            content.sound = .default

            let request = UNNotificationRequest(
                identifier: "omega-proactive-\(UUID().uuidString)",
                content: content,
                trigger: nil
            )
            try? await center.add(request)
        }
    }

    nonisolated func userNotificationCenter(
        _ center: UNUserNotificationCenter,
        willPresent notification: UNNotification
    ) async -> UNNotificationPresentationOptions {
        [.banner, .sound]
    }

    nonisolated func userNotificationCenter(
        _ center: UNUserNotificationCenter,
        didReceive response: UNNotificationResponse
    ) async {
        await openTray()
    }
}
