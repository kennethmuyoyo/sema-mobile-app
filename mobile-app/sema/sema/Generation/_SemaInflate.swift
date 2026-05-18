import Compression
import Foundation

/// Thin wrapper around the system Compression framework's raw DEFLATE
/// (the algorithm used for stored `.npz` entries that ship from NumPy's
/// `np.savez_compressed`). Pulled out of `PoseDatabase` so the file
/// stays single-purpose.
enum _SemaInflate {
    /// Inflate a raw DEFLATE stream. `dst` must be large enough for the
    /// uncompressed payload; pass `expectedSize` from the ZIP header.
    /// Returns the number of bytes written, or 0 on failure.
    static func zlibRawInflate(src: UnsafePointer<UInt8>, srcLen: Int, dst: UnsafeMutablePointer<UInt8>, dstCap: Int) -> Int {
        compression_decode_buffer(dst, dstCap, src, srcLen, nil, COMPRESSION_ZLIB)
    }
}
