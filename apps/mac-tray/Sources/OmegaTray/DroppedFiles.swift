import Foundation
import UniformTypeIdentifiers

/// Finding the files a drop or a message means, so they go to the core as files.
///
/// A drag does not always say "file" the way Finder does. An editor's drag can carry its
/// files as `text/uri-list`, as `file://` URLs, as plain absolute paths, or in its own
/// JSON types (a list of URIs, or objects with an `fsPath`), and no `public.file-url` at
/// all. Every string in such a payload is a candidate, and a candidate counts only when it
/// names an existing, readable regular file on this Mac, so a payload that is only text
/// yields nothing and stays text.
enum DroppedFiles {
    /// `MAX_PUT_IN` in the core, as `OmegaPutUpload.maximumFileBytes`; a path written
    /// in a message is only taken up to it.
    static let maximumMessageFileBytes = 25 << 20
    /// How much of one drag representation is read when looking for file references.
    static let maximumPayloadBytes = 1 << 20

    /// The regular file a URL names, symlinks resolved, when it is one and readable.
    static func regularFile(_ url: URL, maximumBytes: Int? = nil) -> URL? {
        guard url.isFileURL else { return nil }
        let resolved = url.standardizedFileURL.resolvingSymlinksInPath()
        guard let attributes = try? FileManager.default.attributesOfItem(atPath: resolved.path),
              attributes[.type] as? FileAttributeType == .typeRegular,
              FileManager.default.isReadableFile(atPath: resolved.path)
        else { return nil }
        if let maximumBytes,
           let size = (attributes[.size] as? NSNumber)?.intValue,
           size > maximumBytes {
            return nil
        }
        return resolved
    }

    /// One candidate string as a file: a `file://` URL (percent-decoded), an absolute
    /// path (as written, or percent-decoded when only that exists), or `~/`.
    static func file(named candidate: String, home: String = NSHomeDirectory()) -> URL? {
        var text = candidate.trimmingCharacters(in: .whitespacesAndNewlines)
        if text.count >= 2, let first = text.first, first == text.last, "\"'`".contains(first) {
            text = String(text.dropFirst().dropLast())
        }
        if text.lowercased().hasPrefix("file://") {
            guard let url = URL(string: text) ?? text.addingPercentEncoding(withAllowedCharacters: .urlFragmentAllowed).flatMap(URL.init(string:)),
                  url.isFileURL
            else { return nil }
            return regularFile(url)
        }
        if text.hasPrefix("~/") {
            text = (home as NSString).appendingPathComponent(String(text.dropFirst(2)))
        }
        guard text.hasPrefix("/") else { return nil }
        if let file = regularFile(URL(fileURLWithPath: text)) { return file }
        if let decoded = text.removingPercentEncoding, decoded != text {
            return regularFile(URL(fileURLWithPath: decoded))
        }
        return nil
    }

    /// Every file a drag payload names, in order, each once. JSON is walked for every
    /// string in it; anything else is read a line at a time (`#` lines are uri-list
    /// comments).
    static func references(in payload: String) -> [URL] {
        var candidates: [String] = []
        if let data = payload.data(using: .utf8),
           let json = try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed]) {
            collectStrings(json, into: &candidates)
        } else {
            // A list of files, one per line, and nothing else: one line that names no
            // file makes the whole payload text, so a pasted document that happens to
            // hold a path on a line of its own is not cut down to that file.
            let lines = payload.components(separatedBy: .newlines)
                .map { $0.trimmingCharacters(in: .whitespaces) }
                .filter { !$0.isEmpty && !$0.hasPrefix("#") }
            let files = lines.compactMap { file(named: $0) }
            guard files.count == lines.count else { return [] }
            var seen: Set<String> = []
            return files.filter { seen.insert($0.path).inserted }
        }
        var seen: Set<String> = []
        return candidates.compactMap { file(named: $0) }.filter { seen.insert($0.path).inserted }
    }

    private static func collectStrings(_ value: Any, into out: inout [String]) {
        if let string = value as? String {
            out.append(string)
        } else if let array = value as? [Any] {
            array.forEach { collectStrings($0, into: &out) }
        } else if let object = value as? [String: Any] {
            object.keys.sorted().forEach { collectStrings(object[$0] as Any, into: &out) }
        }
    }

    /// The files a provider's representations name, read from every registered type
    /// that is not itself pictures or sound.
    static func references(in provider: NSItemProvider) async -> [URL] {
        var found: [URL] = []
        for identifier in provider.registeredTypeIdentifiers {
            if let type = UTType(identifier),
               type.conforms(to: .image) || type.conforms(to: .audiovisualContent) || type.conforms(to: .pdf) {
                continue
            }
            guard let data = await data(of: provider, type: identifier),
                  data.count <= maximumPayloadBytes,
                  let text = String(data: data, encoding: .utf8)
                    ?? String(data: data, encoding: .utf16)
            else { continue }
            found.append(contentsOf: references(in: text))
        }
        var seen: Set<String> = []
        return found.filter { seen.insert($0.path).inserted }
    }

    private static func data(of provider: NSItemProvider, type: String) async -> Data? {
        await withCheckedContinuation { continuation in
            _ = provider.loadDataRepresentation(forTypeIdentifier: type) { data, _ in
                continuation.resume(returning: data)
            }
        }
    }

    /// Files whose paths are written in a message: each `/…`, `~/…` or `file://…` that
    /// starts a word and, taken as far along its line as names a regular file of at
    /// most `maximumMessageFileBytes`, resolves to one. The longest such reading wins, so
    /// a path with spaces in it is found whole; trailing punctuation is not part of it.
    static func inMessage(_ text: String, home: String = NSHomeDirectory()) -> [URL] {
        var found: [URL] = []
        var seen: Set<String> = []
        for line in text.components(separatedBy: .newlines) where line.count <= 4096 {
            let characters = Array(line)
            var index = 0
            while index < characters.count {
                let startsWord = index == 0 || " \t\"'`(<[".contains(characters[index - 1])
                let rest = String(characters[index...])
                let marked = rest.hasPrefix("/") || rest.hasPrefix("~/") || rest.lowercased().hasPrefix("file://")
                guard startsWord, marked, let (file, length) = longestFile(in: rest, home: home) else {
                    index += 1
                    continue
                }
                if seen.insert(file.path).inserted { found.append(file) }
                index += max(length, 1)
            }
        }
        return found
    }

    private static func longestFile(in rest: String, home: String) -> (URL, Int)? {
        let characters = Array(rest)
        var ends = characters.indices.filter { characters[$0] == " " || characters[$0] == "\t" }
        ends.append(characters.count)
        for end in ends.reversed() {
            let written = String(characters[..<end])
            var trimmed = written
            while let last = trimmed.last, ".,;:!?)]}>\"'`".contains(last) {
                trimmed.removeLast()
            }
            for candidate in [written, trimmed] where !candidate.isEmpty {
                if let file = file(named: candidate, home: home),
                   regularFile(file, maximumBytes: maximumMessageFileBytes) != nil {
                    return (file, end)
                }
            }
        }
        return nil
    }
}
