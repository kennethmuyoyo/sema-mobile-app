import Foundation

/// Debug-only: writes recognizer inference windows to disk as .npy files so
/// they can be replayed offline with `recognition/v3_infer.py` for parity
/// against the on-device CoreML emission. Files land in the app's Documents
/// directory under `landmark_dumps/`; pull them via
/// Xcode → Devices and Simulators → app → ⚙️ → Download Container.
///
/// Each dump is the raw `(T_in, 135)` buffer the way it's handed to
/// `GlossTagger.predict(features:)` — shoulder-centered, shoulder-width-scaled,
/// xyz only, no mask, no z-score. `v3_infer.py` applies stride-2 + z-score on
/// top, matching what the CoreML graph does internally.
enum LandmarkDump {
    private static let directoryURL: URL = {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let dir = docs.appendingPathComponent("landmark_dumps", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }()

    private static var didLogPath = false

    struct TopKEntry: Sendable {
        let label: String
        let confidence: Float
    }

    static func dump(
        features: [Float],
        inputSeqLen: Int,
        featureDim: Int,
        topK: [TopKEntry],
        timestamp: TimeInterval
    ) {
        precondition(features.count == inputSeqLen * featureDim,
                     "feature count \(features.count) ≠ \(inputSeqLen) × \(featureDim)")
        let ts = String(format: "%.3f", timestamp)
        let rawLabel = topK.first?.label ?? "noemit"
        let conf = topK.first?.confidence ?? 0
        // Filenames must survive HFS + the user's shell + macOS Finder.
        let safeLabel = rawLabel
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: " ", with: "_")
        let base = "\(ts)_\(safeLabel)_\(String(format: "%.2f", conf))"
        let npyURL = directoryURL.appendingPathComponent("\(base).npy")
        let metaURL = directoryURL.appendingPathComponent("\(base).json")
        do {
            try NpyWriter.writeFloat32(features, shape: [inputSeqLen, featureDim], to: npyURL)
            let meta: [String: Any] = [
                "timestamp": timestamp,
                "shape": [inputSeqLen, featureDim],
                "topK": topK.map { ["label": $0.label, "confidence": $0.confidence] },
            ]
            let jsonData = try JSONSerialization.data(withJSONObject: meta, options: [.prettyPrinted, .sortedKeys])
            try jsonData.write(to: metaURL)
            if !didLogPath {
                print("[LandmarkDump] writing to \(directoryURL.path)")
                didLogPath = true
            }
        } catch {
            print("[LandmarkDump] failed to write \(base): \(error)")
        }
    }
}

/// Minimal NumPy v1.0 writer for contiguous Float32 arrays in C order,
/// little-endian. Just enough to produce files `numpy.load` accepts.
enum NpyWriter {
    enum Error: Swift.Error { case headerTooLong }

    static func writeFloat32(_ array: [Float], shape: [Int], to url: URL) throws {
        // npy v1.0 needs (magic + version + len + header) % 64 == 0 and the
        // header itself to end with '\n'. We pad with spaces before the '\n'.
        var header = "{'descr': '<f4', 'fortran_order': False, 'shape': ("
        header += shape.map(String.init).joined(separator: ", ")
        if shape.count == 1 { header += "," }
        header += "), }"
        let prefixLen = 10   // 6 (magic) + 2 (version) + 2 (uint16 length)
        let unpadded = prefixLen + header.utf8.count + 1
        let padding = (64 - (unpadded % 64)) % 64
        header += String(repeating: " ", count: padding)
        header += "\n"
        let headerBytes = Array(header.utf8)
        guard headerBytes.count <= Int(UInt16.max) else { throw Error.headerTooLong }

        var out = Data()
        out.append(contentsOf: [0x93])
        out.append(contentsOf: Array("NUMPY".utf8))
        out.append(contentsOf: [0x01, 0x00])
        let hlen = UInt16(headerBytes.count)
        out.append(UInt8(hlen & 0xff))
        out.append(UInt8((hlen >> 8) & 0xff))
        out.append(contentsOf: headerBytes)
        array.withUnsafeBufferPointer { buf in
            out.append(Data(buffer: buf))
        }
        try out.write(to: url)
    }
}
