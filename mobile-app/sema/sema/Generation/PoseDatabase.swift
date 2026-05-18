import Foundation

/// Bundled pose-clip database. Reads `index.json` at init, lazy-decodes
/// `.npz` clips on first access, holds them in an LRU cache.
///
/// The `.npz` files are produced by `generation/pose_library/build_index.py`
/// and can contain either:
///   - `clip_i8: (T,45,3) int8` + `scale: (45,3) float32` (compact), or
///   - `clip_f32: (T,45,3) float32` (BVH-fidelity, larger).
///
/// We don't pull a full NumPy reader in; we read the `.npz` (a ZIP of two
/// `.npy` blobs) and parse the simple, well-known `.npy` format ourselves.
actor PoseDatabase {

    struct Entry: Sendable, Decodable {
        let path: String
        let n_frames: Int
        let fps: Double
        let source_clip_id: Int
        let source_range: [Int]
    }

    private let entries: [String: Entry]
    /// Subdirectory that owns each token's entry, in case future code wants
    /// to know whether a clip came from the curated demo library or the
    /// alignment-derived full one. Indexed by token.
    private let entrySource: [String: String]
    /// Resource subdirectories this database is loading from, in priority
    /// order. The first one's entry wins on token collisions (so the curated
    /// `PoseLibrary` demo set shadows any duplicate in `PoseLibraryFull`).
    private let bundleSubdirs: [String]
    private let cacheLimit: Int
    private var cache: [String: PoseClip] = [:]
    private var cacheOrder: [String] = []   // simple LRU; small ≤ cacheLimit

    /// Old single-library init — kept for SwiftUI previews and unit tests
    /// that only want one library. Production code should use the
    /// `bundleSubdirs:` initialiser below.
    convenience init(bundleSubdir: String = "PoseLibrary",
                     indexResourceName: String? = nil,
                     cacheLimit: Int = 256) throws {
        try self.init(bundleSubdirs: [bundleSubdir],
                      indexResourceName: indexResourceName,
                      cacheLimit: cacheLimit)
    }

    /// Multi-library init. Loads every subdir's index in order and merges
    /// entries; first-write-wins on token collisions so a curated demo
    /// clip is preferred over an alignment-derived full-library clip with
    /// the same token name.
    ///
    /// Index filename resolution (per subdir):
    ///   - `"PoseLibrary"`     → `index.json`
    ///   - `"PoseLibraryFull"` → `index_full.json`
    ///   - explicit `indexResourceName:` argument overrides for single-subdir
    ///     callers; for multi-subdir callers, leave it nil and the default
    ///     derivation kicks in.
    init(bundleSubdirs: [String],
         indexResourceName: String? = nil,
         cacheLimit: Int = 256) throws {
        precondition(!bundleSubdirs.isEmpty, "bundleSubdirs cannot be empty")
        self.bundleSubdirs = bundleSubdirs
        self.cacheLimit = cacheLimit

        var merged: [String: Entry] = [:]
        var source: [String: String] = [:]
        var loadedAny = false

        for (idx, subdir) in bundleSubdirs.enumerated() {
            // Per-subdir index filename resolution. Honors an explicit
            // override only when it's the single-subdir case (subdir count
            // == 1) — multi-subdir callers can't share one override.
            let indexResource: String
            if let explicit = indexResourceName, bundleSubdirs.count == 1 {
                indexResource = explicit
            } else if subdir == "PoseLibrary" {
                indexResource = "index"
            } else if subdir == "PoseLibraryFull" {
                indexResource = "index_full"
            } else {
                indexResource = "index"
            }

            // Try the natural subdirectory first; fall back to bundle root
            // (Xcode 16 fileSystemSynchronizedGroup flattens everything).
            var indexURL = Bundle.main.url(forResource: indexResource,
                                            withExtension: "json",
                                            subdirectory: subdir)
            if indexURL == nil {
                indexURL = Bundle.main.url(forResource: indexResource,
                                            withExtension: "json")
            }
            guard let url = indexURL else {
                print("[PoseDatabase] \(indexResource).json not found for subdir '\(subdir)'; skipping")
                continue
            }
            let data = try Data(contentsOf: url)
            let part = try JSONDecoder().decode([String: Entry].self, from: data)
            var added = 0
            var shadowed = 0
            for (token, entry) in part {
                if merged[token] != nil {
                    shadowed += 1
                    continue   // first-write-wins; earlier subdir is canonical
                }
                merged[token] = entry
                source[token] = subdir
                added += 1
            }
            loadedAny = true
            print("[PoseDatabase] [\(idx)] '\(subdir)/\(indexResource).json' loaded \(part.count) entries (added \(added), shadowed \(shadowed))")
        }

        guard loadedAny, !merged.isEmpty else {
            print("[PoseDatabase] FAIL: no index found across subdirs=\(bundleSubdirs)")
            throw PoseDatabaseError.indexMissing
        }

        self.entries = merged
        self.entrySource = source
        print("[PoseDatabase] total: \(entries.count) entries across \(bundleSubdirs.count) library(ies)")
    }

    var tokenCount: Int { entries.count }
    func allTokens() -> [String] { Array(entries.keys) }

    func contains(_ token: String) -> Bool { entries[token] != nil }

    /// Look up the clip for a gloss token. Returns nil for unknown tokens.
    func lookup(_ token: String) throws -> PoseClip? {
        if let cached = cache[token] {
            // promote in LRU order
            if let i = cacheOrder.firstIndex(of: token) {
                cacheOrder.remove(at: i)
            }
            cacheOrder.append(token)
            return cached
        }
        guard let entry = entries[token] else { return nil }

        // Resolve the .npz path inside the bundle. We try the token's owning
        // library subdir first (recorded at load time), fall back across all
        // configured subdirs, then to the bundle root since Xcode 16
        // typically flattens.
        let stem = (entry.path as NSString).deletingPathExtension          // "clips/HELLO"
        let stemFile = (stem as NSString).lastPathComponent                 // "HELLO"
        let clipDir = (stem as NSString).deletingLastPathComponent          // "clips"
        let ownerSubdir = entrySource[token] ?? bundleSubdirs.first ?? "PoseLibrary"
        let candidateSubdirs = [ownerSubdir] + bundleSubdirs.filter { $0 != ownerSubdir }
        var url: URL?
        for subdir in candidateSubdirs {
            let full = clipDir.isEmpty ? subdir : "\(subdir)/\(clipDir)"
            if let hit = Bundle.main.url(forResource: stemFile,
                                          withExtension: "npz",
                                          subdirectory: full)
                ?? Bundle.main.url(forResource: stemFile,
                                    withExtension: "npz",
                                    subdirectory: subdir) {
                url = hit
                break
            }
        }
        if url == nil {
            url = Bundle.main.url(forResource: stemFile, withExtension: "npz")
        }
        guard let url else {
            print("[PoseDatabase] clip not in bundle: \(token) (tried subdirs=\(candidateSubdirs), file=\(stemFile).npz)")
            throw PoseDatabaseError.clipFileMissing(token)
        }
        print("[PoseDatabase] lookup '\(token)' → \(url.lastPathComponent)")

        var clip = try decodeNPZ(at: url, token: token, entry: entry)
        do {
            if let rigRotations = try decodeRotationSidecar(
                stemFile: stemFile,
                expectedFrames: clip.frameCount,
                token: token
            ) {
                clip = PoseClip(
                    frames: clip.frames,
                    rigRotations: rigRotations,
                    rigJointCount: BVHRigRotationLayout.jointCount,
                    frameCount: clip.frameCount,
                    fps: clip.fps,
                    sourceClipId: clip.sourceClipId,
                    sourceRange: clip.sourceRange,
                    token: clip.token
                )
            }
        } catch {
            print("[PoseDatabase] rotation sidecar unavailable for '\(token)': \(error)")
        }
        cache[token] = clip
        cacheOrder.append(token)
        if cacheOrder.count > cacheLimit {
            let evict = cacheOrder.removeFirst()
            cache.removeValue(forKey: evict)
        }
        return clip
    }

    // MARK: - NPZ / NPY parsing

    private func decodeNPZ(at url: URL, token: String, entry: Entry) throws -> PoseClip {
        // .npz = ZIP. We support either clip_f32.npy (raw) OR
        // clip_i8.npy+scale.npy (quantized).
        let zipData = try Data(contentsOf: url)
        let blobs = try parseFlatZip(data: zipData)

        let T: Int
        let floats: [Float]
        if let f32Blob = blobs["clip_f32.npy"] {
            let (fShape, fBytes, fDtype) = try parseNPY(blob: f32Blob)
            guard fDtype == "<f4" || fDtype == "|f4" else {
                throw PoseDatabaseError.clipMalformed(token, "clip_f32 dtype \(fDtype) not float32")
            }
            guard fShape.count == 3, fShape[1] == 45, fShape[2] == 3 else {
                throw PoseDatabaseError.clipMalformed(token, "clip_f32 shape \(fShape) not (T, 45, 3)")
            }
            T = fShape[0]
            guard T == entry.n_frames else {
                throw PoseDatabaseError.clipMalformed(token, "frame count mismatch: file=\(T), index=\(entry.n_frames)")
            }
            floats = fBytes.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
            print("[PoseDatabase] decode '\(token)' format=f32 frames=\(T)")
        } else if let i8Blob = blobs["clip_i8.npy"], let scaleBlob = blobs["scale.npy"] {
            let (i8Shape, i8Bytes, i8Dtype) = try parseNPY(blob: i8Blob)
            let (scaleShape, scaleBytes, scaleDtype) = try parseNPY(blob: scaleBlob)
            guard i8Dtype == "<i1" || i8Dtype == "|i1" else {
                throw PoseDatabaseError.clipMalformed(token, "clip_i8 dtype \(i8Dtype) not int8")
            }
            guard scaleDtype == "<f4" || scaleDtype == "|f4" else {
                throw PoseDatabaseError.clipMalformed(token, "scale dtype \(scaleDtype) not float32")
            }
            guard i8Shape.count == 3, i8Shape[1] == 45, i8Shape[2] == 3 else {
                throw PoseDatabaseError.clipMalformed(token, "clip_i8 shape \(i8Shape) not (T, 45, 3)")
            }
            guard scaleShape == [45, 3] else {
                throw PoseDatabaseError.clipMalformed(token, "scale shape \(scaleShape) not (45, 3)")
            }
            T = i8Shape[0]
            guard T == entry.n_frames else {
                throw PoseDatabaseError.clipMalformed(token, "frame count mismatch: file=\(T), index=\(entry.n_frames)")
            }

            // Reconstruct float32 = i8 / 127 * scale
            let total = T * 45 * 3
            var out = [Float](repeating: 0, count: total)
            let scale: [Float] = scaleBytes.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
            let i8: [Int8] = i8Bytes.withUnsafeBytes { Array($0.bindMemory(to: Int8.self)) }
            for f in 0..<T {
                for j in 0..<45 {
                    for k in 0..<3 {
                        let off = f * 45 * 3 + j * 3 + k
                        out[off] = Float(i8[off]) / 127.0 * scale[j * 3 + k]
                    }
                }
            }
            floats = out
            print("[PoseDatabase] decode '\(token)' format=i8 frames=\(T)")
        } else {
            throw PoseDatabaseError.clipMalformed(token, "missing clip_f32.npy or clip_i8.npy+scale.npy")
        }

        return PoseClip(
            frames: floats,
            rigRotations: nil,
            rigJointCount: 0,
            frameCount: T,
            fps: Float(entry.fps),
            sourceClipId: entry.source_clip_id,
            sourceRange: entry.source_range[0]...entry.source_range[1],
            token: token
        )
    }

    private func decodeRotationSidecar(
        stemFile: String,
        expectedFrames: Int,
        token: String
    ) throws -> [Float]? {
        let sidecarResourceName = "\(stemFile).rot"
        let sidecarFileName = "\(sidecarResourceName).npz"
        // Try the token's owning library first, then any other configured
        // library, then bundle root — same fallback ladder as the clip lookup.
        let ownerSubdir = entrySource[token] ?? bundleSubdirs.first ?? "PoseLibrary"
        let candidates = [ownerSubdir] + bundleSubdirs.filter { $0 != ownerSubdir }
        var url: URL?
        for subdir in candidates {
            if let hit = Bundle.main.url(forResource: sidecarResourceName,
                                          withExtension: "npz",
                                          subdirectory: "\(subdir)/rotations")
                ?? Bundle.main.url(forResource: sidecarResourceName,
                                    withExtension: "npz",
                                    subdirectory: subdir) {
                url = hit
                break
            }
        }
        if url == nil {
            url = Bundle.main.url(forResource: sidecarResourceName,
                                   withExtension: "npz",
                                   subdirectory: "rotations")
        }
        if url == nil {
            let target = sidecarFileName
            url = Bundle.main
                .urls(forResourcesWithExtension: "npz", subdirectory: nil)?
                .first(where: { $0.lastPathComponent == target })
        }
        guard let url else { return nil }

        let zipData = try Data(contentsOf: url)
        let blobs = try parseFlatZip(data: zipData)
        guard let quatBlob = blobs["quat_f32.npy"] else {
            throw PoseDatabaseError.clipMalformed(token, "rotation sidecar missing quat_f32.npy")
        }
        let (shape, bytes, dtype) = try parseNPY(blob: quatBlob)
        guard dtype == "<f4" || dtype == "|f4" else {
            throw PoseDatabaseError.clipMalformed(token, "rotation sidecar dtype \(dtype) not float32")
        }
        guard shape.count == 3, shape[0] == expectedFrames, shape[1] == BVHRigRotationLayout.jointCount, shape[2] == 4 else {
            throw PoseDatabaseError.clipMalformed(
                token,
                "rotation sidecar shape \(shape) not (\(expectedFrames), \(BVHRigRotationLayout.jointCount), 4)"
            )
        }
        let floats = bytes.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        print("[PoseDatabase] quaternions '\(token)' joints=\(shape[1])")
        return floats
    }

    /// Minimal flat-ZIP reader: handles stored (method 0) and deflate (8).
    /// Sufficient for `.npz` files (which use deflate for compressed_npz).
    private func parseFlatZip(data: Data) throws -> [String: Data] {
        var out: [String: Data] = [:]
        let bytes = [UInt8](data)

        func le16(_ idx: Int) -> UInt16 {
            UInt16(bytes[idx]) | (UInt16(bytes[idx + 1]) << 8)
        }
        func le32(_ idx: Int) -> UInt32 {
            UInt32(bytes[idx]) |
            (UInt32(bytes[idx + 1]) << 8) |
            (UInt32(bytes[idx + 2]) << 16) |
            (UInt32(bytes[idx + 3]) << 24)
        }
        func le64(_ idx: Int) -> UInt64 {
            UInt64(bytes[idx]) |
            (UInt64(bytes[idx + 1]) << 8) |
            (UInt64(bytes[idx + 2]) << 16) |
            (UInt64(bytes[idx + 3]) << 24) |
            (UInt64(bytes[idx + 4]) << 32) |
            (UInt64(bytes[idx + 5]) << 40) |
            (UInt64(bytes[idx + 6]) << 48) |
            (UInt64(bytes[idx + 7]) << 56)
        }

        var i = 0
        while i + 30 <= bytes.count {
            let sig = le32(i)
            if sig != 0x04034b50 { break }  // hit central directory or EOCD
            let method = le16(i + 8)
            var compSize64 = UInt64(le32(i + 18))
            var uncompSize64 = UInt64(le32(i + 22))
            let nameLen = Int(le16(i + 26))
            let extraLen = Int(le16(i + 28))
            let nameStart = i + 30
            let extraStart = nameStart + nameLen
            let dataStart = extraStart + extraLen
            guard dataStart <= bytes.count else { throw PoseDatabaseError.zipTruncated }

            // ZIP64 local header support: when 32-bit size fields are all-ones,
            // actual 64-bit sizes are stored in extra field id 0x0001.
            if compSize64 == 0xFFFF_FFFF || uncompSize64 == 0xFFFF_FFFF {
                let extraEnd = extraStart + extraLen
                var cursor = extraStart
                while cursor + 4 <= extraEnd {
                    let fieldID = le16(cursor)
                    let fieldLen = Int(le16(cursor + 2))
                    let fieldDataStart = cursor + 4
                    let fieldDataEnd = fieldDataStart + fieldLen
                    guard fieldDataEnd <= extraEnd else { break }
                    if fieldID == 0x0001 {
                        var p = fieldDataStart
                        if uncompSize64 == 0xFFFF_FFFF, p + 8 <= fieldDataEnd {
                            uncompSize64 = le64(p)
                            p += 8
                        }
                        if compSize64 == 0xFFFF_FFFF, p + 8 <= fieldDataEnd {
                            compSize64 = le64(p)
                            p += 8
                        }
                        break
                    }
                    cursor = fieldDataEnd
                }
            }

            guard compSize64 <= UInt64(Int.max) else { throw PoseDatabaseError.zipTruncated }
            guard uncompSize64 <= UInt64(Int.max) else { throw PoseDatabaseError.zipTruncated }
            let compSize = Int(compSize64)
            let uncompSize = Int(uncompSize64)
            guard compSize <= (bytes.count - dataStart) else { throw PoseDatabaseError.zipTruncated }
            let dataEnd = dataStart + compSize

            let name = String(bytes: bytes[nameStart..<nameStart + nameLen], encoding: .utf8) ?? ""
            let payload = Data(bytes[dataStart..<dataEnd])
            let decoded: Data
            switch method {
            case 0:
                decoded = payload
            case 8:
                decoded = try inflate(payload, expectedSize: uncompSize)
            default:
                throw PoseDatabaseError.unsupportedCompression(method)
            }
            out[name] = decoded
            i = dataEnd
        }
        return out
    }

    private func inflate(_ data: Data, expectedSize: Int) throws -> Data {
        // Use libcompression's raw DEFLATE. Imported via the Compression
        // framework which is available on iOS 9+.
        let dst = UnsafeMutablePointer<UInt8>.allocate(capacity: max(expectedSize, data.count * 4 + 16))
        defer { dst.deallocate() }
        let result = data.withUnsafeBytes { (src: UnsafeRawBufferPointer) -> Int in
            guard let base = src.bindMemory(to: UInt8.self).baseAddress else { return 0 }
            return decompress(src: base, srcLen: src.count, dst: dst, dstCap: max(expectedSize, data.count * 4 + 16))
        }
        guard result > 0 else { throw PoseDatabaseError.inflationFailed }
        return Data(bytes: dst, count: result)
    }

    /// Minimal Compression-framework wrapper for raw DEFLATE.
    private nonisolated func decompress(src: UnsafePointer<UInt8>, srcLen: Int, dst: UnsafeMutablePointer<UInt8>, dstCap: Int) -> Int {
        // Forward to Compression API; the Compression import lives in a private extension.
        return _SemaInflate.zlibRawInflate(src: src, srcLen: srcLen, dst: dst, dstCap: dstCap)
    }

    /// Parse a `.npy` blob. Returns shape, raw bytes of array data, and dtype string.
    private func parseNPY(blob: Data) throws -> (shape: [Int], bytes: Data, dtype: String) {
        let prefix: [UInt8] = [0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59]   // \x93NUMPY
        guard blob.count > 10, [UInt8](blob.prefix(6)) == prefix else {
            throw PoseDatabaseError.npyMalformed("bad magic")
        }
        let major = blob[6]
        let _minor = blob[7]
        let headerLen: Int
        let headerStart: Int
        if major == 1 {
            headerLen = Int(blob[8]) | (Int(blob[9]) << 8)
            headerStart = 10
        } else {
            headerLen =
                Int(blob[8]) | (Int(blob[9]) << 8) |
                (Int(blob[10]) << 16) | (Int(blob[11]) << 24)
            headerStart = 12
        }
        let headerData = blob[headerStart..<headerStart + headerLen]
        guard let header = String(data: headerData, encoding: .ascii) else {
            throw PoseDatabaseError.npyMalformed("header not ASCII")
        }
        // header looks like: {'descr': '<i1', 'fortran_order': False, 'shape': (32, 45, 3), }
        let dtype = extractQuoted(in: header, after: "'descr':") ?? "?"
        let shapeText = extractParens(in: header, after: "'shape':") ?? "()"
        let shape = shapeText
            .replacingOccurrences(of: " ", with: "")
            .split(separator: ",", omittingEmptySubsequences: true)
            .compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }

        let dataStart = headerStart + headerLen
        return (shape, Data(blob[dataStart...]), dtype)
    }

    private func extractQuoted(in s: String, after marker: String) -> String? {
        guard let r = s.range(of: marker) else { return nil }
        var i = s.index(after: r.upperBound.samePosition(in: s) ?? r.upperBound)
        // find next '
        while i < s.endIndex && s[i] != "'" { i = s.index(after: i) }
        guard i < s.endIndex else { return nil }
        let start = s.index(after: i)
        var j = start
        while j < s.endIndex && s[j] != "'" { j = s.index(after: j) }
        guard j < s.endIndex else { return nil }
        return String(s[start..<j])
    }

    private func extractParens(in s: String, after marker: String) -> String? {
        guard let r = s.range(of: marker) else { return nil }
        var i = r.upperBound
        while i < s.endIndex && s[i] != "(" { i = s.index(after: i) }
        guard i < s.endIndex else { return nil }
        let start = s.index(after: i)
        var j = start
        while j < s.endIndex && s[j] != ")" { j = s.index(after: j) }
        guard j < s.endIndex else { return nil }
        return String(s[start..<j])
    }
}

enum PoseDatabaseError: Error, CustomStringConvertible {
    case indexMissing
    case clipFileMissing(String)
    case clipMalformed(String, String)
    case zipTruncated
    case unsupportedCompression(UInt16)
    case inflationFailed
    case npyMalformed(String)

    var description: String {
        switch self {
        case .indexMissing:
            return "PoseLibrary/index.json not in bundle. Run generation/pose_library/build_index.py."
        case .clipFileMissing(let t): return "Pose clip for '\(t)' is not in the bundle."
        case .clipMalformed(let t, let m): return "Pose clip '\(t)' malformed: \(m)"
        case .zipTruncated: return ".npz file truncated"
        case .unsupportedCompression(let m): return "ZIP compression method \(m) not supported"
        case .inflationFailed: return "DEFLATE inflation failed"
        case .npyMalformed(let m): return ".npy parse error: \(m)"
        }
    }
}
