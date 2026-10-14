import Foundation

/// The tray's half of the core's chunked `put` op (`python/omega/channel.py`).
///
/// The core may run on another machine, so the tray sends a file's bytes rather than its
/// path: base64 chunks of at most `maximumChunkBytes`, each written at its offset into one
/// staging file named by the upload id. Resending a chunk lands it where it landed before,
/// so a retry is harmless; the last chunk stores the file and answers `attached` with the
/// same `{blob, mime, bytes}` that `attach` returns.
@MainActor
enum OmegaPutUpload {
    /// `MAX_PUT_CHUNK` in the core: one chunk's bytes before base64.
    static let maximumChunkBytes = 512 << 10
    /// `MAX_PUT_IN` in the core: the largest file it takes.
    static let maximumFileBytes = 25 << 20

    /// The file's bytes, refused before anything is sent when it is over the core's cap,
    /// so an over-cap file is never half uploaded.
    static func read(fileAt url: URL) throws -> Data {
        guard url.isFileURL else {
            throw OmegaChannelError.invalidMessage("attachment path must be a file URL")
        }
        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        guard attributes[.type] as? FileAttributeType == .typeRegular else {
            throw OmegaChannelError.invalidMessage("attachment must be a regular file")
        }
        if let size = (attributes[.size] as? NSNumber)?.intValue, size > maximumFileBytes {
            throw OmegaChannelError.overCap
        }
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        guard data.count <= maximumFileBytes else { throw OmegaChannelError.overCap }
        return data
    }

    /// The name the core guesses the mime type from, as `attach` did from the path.
    static func fileName(for url: URL) -> String {
        let name = url.lastPathComponent
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: "\\", with: "_")
        return name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? "file" : name
    }

    /// Every `put` line of one upload, in order. An empty file is one empty last chunk.
    static func lines(upload: String, name: String, data: Data) throws -> [Data] {
        guard data.count <= maximumFileBytes else { throw OmegaChannelError.overCap }
        var lines: [Data] = []
        var offset = 0
        repeat {
            let end = min(offset + maximumChunkBytes, data.count)
            let chunk = data.subdata(in: offset..<end)
            lines.append(try OmegaChannelClient.encodeObject([
                "op": "put",
                "upload": upload,
                "name": name,
                "offset": offset,
                "data": chunk.base64EncodedString(),
                "last": end == data.count,
            ]))
            offset = end
        } while offset < data.count
        return lines
    }

    /// Sends the lines in order and returns the stored reference. A dropped connection
    /// starts the upload again from its first chunk under the same id: each chunk is
    /// written at its offset, so the staging file ends up exactly the file's bytes.
    static func send(
        _ lines: [Data],
        isRunning: () -> Bool,
        request: (Data) async throws -> [String: Any]
    ) async throws -> TrayAttachmentReference {
        upload: while isRunning() {
            for (index, line) in lines.enumerated() {
                let response: [String: Any]
                do {
                    response = try await request(line)
                } catch OmegaChannelError.disconnected {
                    continue upload
                }
                guard OmegaChannelClient.integer(response["v"]) == 1 else {
                    throw OmegaChannelClient.responseError(response, expected: "put")
                }
                if index < lines.count - 1 {
                    guard response["op"] as? String == "put" else {
                        throw OmegaChannelClient.responseError(response, expected: "put")
                    }
                    continue
                }
                guard response["op"] as? String == "attached",
                      let blob = response["blob"] as? String,
                      OmegaChannelClient.isDigest(blob),
                      let mime = response["mime"] as? String,
                      !mime.isEmpty,
                      let bytes = OmegaChannelClient.integer(response["bytes"]),
                      bytes >= 0
                else {
                    throw OmegaChannelClient.responseError(response, expected: "attached")
                }
                return TrayAttachmentReference(blob: blob, mime: mime, bytes: bytes)
            }
        }
        throw OmegaChannelError.stopped
    }
}
