import XCTest
@testable import OmegaTray

final class MarkdownMessageTests: XCTestCase {
    func testParsesCommonReplyStructureWithoutLeavingBlockMarkers() {
        let blocks = TrayMarkdownParser.parse(
            """
            # Result

            This is **important** and `literal`.

            - First
            2. Second
            > A note

            ```swift
            let answer = 42
            ```
            """
        )

        XCTAssertEqual(blocks, [
            .heading(level: 1, text: "Result"),
            .paragraph("This is **important** and `literal`."),
            .unordered("First"),
            .ordered(marker: "2.", text: "Second"),
            .quote("A note"),
            .code("let answer = 42"),
        ])
    }

    func testPlainLinesBecomeOneReadableParagraph() {
        XCTAssertEqual(
            TrayMarkdownParser.parse("one line\ncontinues here"),
            [.paragraph("one line continues here")]
        )
    }

    func testUnclosedFenceStillRendersAsCode() {
        XCTAssertEqual(
            TrayMarkdownParser.parse("```\nunfinished"),
            [.code("unfinished")]
        )
    }
}
