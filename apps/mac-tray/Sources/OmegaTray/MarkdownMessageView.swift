import SwiftUI

enum TrayMarkdownBlock: Equatable {
    case paragraph(String)
    case heading(level: Int, text: String)
    case unordered(text: String, continuation: [String] = [])
    case ordered(marker: String, text: String, continuation: [String] = [])
    case quote(String)
    case code(String)
    case divider
}

enum TrayMarkdownParser {
    static func parse(_ source: String) -> [TrayMarkdownBlock] {
        let lines = source.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
        var blocks: [TrayMarkdownBlock] = []
        var paragraph: [String] = []
        var code: [String] = []
        var isInCode = false

        func flushParagraph() {
            guard !paragraph.isEmpty else { return }
            blocks.append(.paragraph(paragraph.joined(separator: " ")))
            paragraph.removeAll(keepingCapacity: true)
        }

        for line in lines {
            if line.hasPrefix("```") {
                flushParagraph()
                if isInCode {
                    blocks.append(.code(code.joined(separator: "\n")))
                    code.removeAll(keepingCapacity: true)
                }
                isInCode.toggle()
                continue
            }

            if isInCode {
                code.append(line)
                continue
            }

            let trimmed = line.trimmingCharacters(in: .whitespaces)
            guard !trimmed.isEmpty else {
                flushParagraph()
                continue
            }

            if paragraph.isEmpty,
               isIndented(line),
               appendListContinuation(trimmed, to: &blocks) {
                continue
            }

            if ["---", "***", "___"].contains(trimmed) {
                flushParagraph()
                blocks.append(.divider)
            } else if let heading = heading(from: trimmed) {
                flushParagraph()
                blocks.append(heading)
            } else if let item = unorderedItem(from: trimmed) {
                flushParagraph()
                blocks.append(.unordered(text: item))
            } else if let item = orderedItem(from: trimmed) {
                flushParagraph()
                blocks.append(.ordered(marker: item.marker, text: item.text))
            } else if trimmed.hasPrefix("> ") {
                flushParagraph()
                blocks.append(.quote(String(trimmed.dropFirst(2))))
            } else {
                paragraph.append(trimmed)
            }
        }

        flushParagraph()
        if isInCode || !code.isEmpty {
            blocks.append(.code(code.joined(separator: "\n")))
        }
        return blocks
    }

    private static func heading(from line: String) -> TrayMarkdownBlock? {
        let hashes = line.prefix { $0 == "#" }.count
        guard (1...3).contains(hashes), line.dropFirst(hashes).first == " " else { return nil }
        return .heading(level: hashes, text: String(line.dropFirst(hashes + 1)))
    }

    private static func unorderedItem(from line: String) -> String? {
        for marker in ["- ", "* ", "+ "] where line.hasPrefix(marker) {
            return String(line.dropFirst(marker.count))
        }
        return nil
    }

    private static func orderedItem(from line: String) -> (marker: String, text: String)? {
        guard let dot = line.firstIndex(of: ".") else { return nil }
        let number = line[..<dot]
        let afterDot = line.index(after: dot)
        guard !number.isEmpty,
              number.allSatisfy(\.isNumber),
              afterDot < line.endIndex,
              line[afterDot] == " "
        else { return nil }
        return ("\(number).", String(line[line.index(after: afterDot)...]))
    }

    private static func isIndented(_ line: String) -> Bool {
        line.first == "\t" || line.prefix { $0 == " " }.count >= 2
    }

    private static func appendListContinuation(
        _ text: String,
        to blocks: inout [TrayMarkdownBlock]
    ) -> Bool {
        guard let last = blocks.last else { return false }
        switch last {
        case .unordered(let item, let continuation):
            blocks[blocks.count - 1] = .unordered(
                text: item,
                continuation: continuation + [text]
            )
            return true
        case .ordered(let marker, let item, let continuation):
            blocks[blocks.count - 1] = .ordered(
                marker: marker,
                text: item,
                continuation: continuation + [text]
            )
            return true
        default:
            return false
        }
    }
}

struct MarkdownMessageView: View {
    let source: String

    private var blocks: [TrayMarkdownBlock] {
        let parsed = TrayMarkdownParser.parse(source)
        return parsed.isEmpty ? [.paragraph(source)] : parsed
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 7) {
            ForEach(Array(blocks.enumerated()), id: \.offset) { _, block in
                blockView(block)
            }
        }
        .textSelection(.enabled)
    }

    @ViewBuilder
    private func blockView(_ block: TrayMarkdownBlock) -> some View {
        switch block {
        case .paragraph(let text):
            inlineText(text)
                .font(.system(size: 13))
        case .heading(let level, let text):
            inlineText(text)
                .font(.system(size: headingSize(level), weight: .semibold))
                .padding(.top, level == 1 ? 3 : 1)
        case .unordered(let text, let continuation):
            listRow(marker: "•", text: text, continuation: continuation)
        case .ordered(let marker, let text, let continuation):
            listRow(marker: marker, text: text, continuation: continuation)
        case .quote(let text):
            HStack(alignment: .top, spacing: 9) {
                Rectangle()
                    .fill(TrayTheme.signal.opacity(0.65))
                    .frame(width: 1)
                inlineText(text)
                    .font(.system(size: 13))
                    .foregroundStyle(TrayTheme.secondaryText)
            }
        case .code(let code):
            ScrollView(.horizontal, showsIndicators: false) {
                Text(verbatim: code)
                    .font(.system(size: 12, design: .monospaced))
                    .foregroundStyle(TrayTheme.primaryText)
                    .fixedSize(horizontal: true, vertical: true)
                    .padding(10)
            }
            .background(TrayTheme.raised, in: RoundedRectangle(cornerRadius: 9, style: .continuous))
        case .divider:
            Divider().overlay(Color.white.opacity(0.09))
        }
    }

    private func listRow(marker: String, text: String, continuation: [String]) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 7) {
            Text(marker)
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(TrayTheme.signal)
                .frame(minWidth: 14, alignment: .trailing)
            VStack(alignment: .leading, spacing: 3) {
                inlineText(text)
                    .font(.system(size: 13))
                ForEach(Array(continuation.enumerated()), id: \.offset) { _, line in
                    inlineText(line)
                        .font(.system(size: 12))
                        .foregroundStyle(TrayTheme.secondaryText)
                }
            }
        }
        .accessibilityElement(children: .combine)
    }

    private func inlineText(_ source: String) -> Text {
        let options = AttributedString.MarkdownParsingOptions(
            interpretedSyntax: .inlineOnlyPreservingWhitespace,
            failurePolicy: .returnPartiallyParsedIfPossible
        )
        guard let attributed = try? AttributedString(markdown: source, options: options) else {
            return Text(verbatim: source)
        }
        return Text(attributed)
    }

    private func headingSize(_ level: Int) -> CGFloat {
        switch level {
        case 1: 18
        case 2: 16
        default: 14
        }
    }
}
