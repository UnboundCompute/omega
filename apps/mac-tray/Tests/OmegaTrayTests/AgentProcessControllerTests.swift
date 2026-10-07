import Foundation
import XCTest
@testable import OmegaTray

final class AgentProcessControllerTests: XCTestCase {
    func testLaunchConfigurationRoundTripsThroughAPropertyList() throws {
        let expected = AgentLaunchConfiguration(
            executable: "/repo/.venv/bin/python",
            workingDirectory: "/repo",
            arguments: ["-m", "omega", "--serve"]
        )

        let data = try PropertyListEncoder().encode(expected)

        XCTAssertEqual(try AgentLaunchConfiguration.decode(data), expected)
    }

    func testARemoteHostLaunchesTheRelayInsteadOfTheCore() {
        let packaged = AgentLaunchConfiguration(
            executable: "/repo/.venv/bin/python",
            workingDirectory: "/repo",
            arguments: ["-m", "omega", "--serve"]
        )

        let relay = packaged.relaying(to: "omega.example.ts.net")

        XCTAssertEqual(relay.arguments, ["-m", "omega", "--relay", "omega.example.ts.net"])
        XCTAssertFalse(relay.arguments.contains("--serve"))
        XCTAssertEqual(relay.executable, packaged.executable)
        XCTAssertEqual(relay.workingDirectory, packaged.workingDirectory)
    }
}
