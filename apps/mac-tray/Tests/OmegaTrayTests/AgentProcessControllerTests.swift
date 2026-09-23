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
}
