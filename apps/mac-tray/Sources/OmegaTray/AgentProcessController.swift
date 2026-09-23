import Foundation

struct AgentLaunchConfiguration: Codable, Equatable {
    let executable: String
    let workingDirectory: String
    let arguments: [String]

    static func load(from url: URL) throws -> AgentLaunchConfiguration {
        try decode(Data(contentsOf: url))
    }

    static func decode(_ data: Data) throws -> AgentLaunchConfiguration {
        try PropertyListDecoder().decode(AgentLaunchConfiguration.self, from: data)
    }
}

@MainActor
final class AgentProcessController {
    var onFailure: ((String) -> Void)?

    private let configurationURL: URL?
    private let logURL: URL
    private var process: Process?
    private var logHandle: FileHandle?
    private var isStopping = false

    init(
        configurationURL: URL? = Bundle.main.url(
            forResource: "AgentLaunch",
            withExtension: "plist"
        ),
        logURL: URL? = nil
    ) {
        self.configurationURL = configurationURL
        self.logURL = logURL ?? AgentProcessController.defaultLogURL
    }

    func start() {
        guard process == nil else { return }
        guard let configurationURL else {
            onFailure?("The local agent launcher is missing. Reinstall omega from this checkout.")
            return
        }

        do {
            let configuration = try AgentLaunchConfiguration.load(from: configurationURL)
            guard FileManager.default.isExecutableFile(atPath: configuration.executable) else {
                throw AgentProcessError.executableMissing(configuration.executable)
            }

            let launched = Process()
            launched.executableURL = URL(fileURLWithPath: configuration.executable)
            launched.currentDirectoryURL = URL(
                fileURLWithPath: configuration.workingDirectory,
                isDirectory: true
            )
            launched.arguments = configuration.arguments
            var environment = ProcessInfo.processInfo.environment
            environment["PYTHONUNBUFFERED"] = "1"
            launched.environment = environment

            let handle = try openLog()
            launched.standardOutput = handle
            launched.standardError = handle
            launched.terminationHandler = { [weak self] finished in
                Task { @MainActor [weak self] in
                    self?.didTerminate(finished)
                }
            }

            isStopping = false
            process = launched
            logHandle = handle
            try launched.run()
        } catch {
            process = nil
            try? logHandle?.close()
            logHandle = nil
            onFailure?("The local agent could not start: \(error.localizedDescription)")
        }
    }

    func stop() {
        isStopping = true
        guard let process, process.isRunning else {
            finishCleanup()
            return
        }
        process.terminate()
    }

    private func didTerminate(_ finished: Process) {
        let expected = isStopping
        let status = finished.terminationStatus
        finishCleanup()
        if !expected {
            onFailure?(
                "The local agent exited with status \(status). Details are in \(logURL.path)."
            )
        }
    }

    private func finishCleanup() {
        process = nil
        try? logHandle?.close()
        logHandle = nil
    }

    private func openLog() throws -> FileHandle {
        let directory = logURL.deletingLastPathComponent()
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        if !FileManager.default.fileExists(atPath: logURL.path) {
            FileManager.default.createFile(atPath: logURL.path, contents: nil)
        }
        let handle = try FileHandle(forWritingTo: logURL)
        try handle.seekToEnd()
        return handle
    }

    private static var defaultLogURL: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Logs/omega", isDirectory: true)
            .appendingPathComponent("agent.log")
    }
}

private enum AgentProcessError: LocalizedError {
    case executableMissing(String)

    var errorDescription: String? {
        switch self {
        case .executableMissing(let path):
            "The configured agent executable does not exist at \(path)."
        }
    }
}
