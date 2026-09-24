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
            .unordered(text: "First"),
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

    func testLearningReceiptKeepsIndentedReplacementWithItsClaim() {
        XCTAssertEqual(
            TrayMarkdownParser.parse(
                """
                I wrote this down:
                - I take my medication at 6:40 on weekdays (when you mention medication)
                  replaces what you told me before: I take my medication in the mornings
                """
            ),
            [
                .paragraph("I wrote this down:"),
                .unordered(
                    text: "I take my medication at 6:40 on weekdays (when you mention medication)",
                    continuation: [
                        "replaces what you told me before: I take my medication in the mornings"
                    ]
                ),
            ]
        )
    }
}
