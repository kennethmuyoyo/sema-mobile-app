import Foundation

/// A per-gloss pose clip, decoded from the int8-quantised `.npz` in the
/// app bundle into float32 `(T, 45, 3)` form.
struct PoseClip: Sendable {
    /// Flat row-major `(frameCount, 45, 3)` values. Length = `frameCount * 135`.
    let frames: [Float]
    /// Optional flat row-major `(frameCount, rigJointCount, 4)` rig-local
    /// quaternions in `[x, y, z, w]` order, aligned to
    /// `BVHRigRotationLayout.jointOrder`. Produced offline by
    /// `generation/pose_library/retarget_to_target.py`.
    let rigRotations: [Float]?
    let rigJointCount: Int
    let frameCount: Int
    let fps: Float
    let sourceClipId: Int
    let sourceRange: ClosedRange<Int>
    let token: String

    /// One frame as a `(45, 3)` flat array, length 135.
    func frame(at i: Int) -> [Float] {
        precondition(i >= 0 && i < frameCount, "frame index out of range")
        let stride = 45 * 3
        let start = i * stride
        return Array(frames[start..<start + stride])
    }

    /// One frame of optional rig-local quaternions, flattened `(rigJointCount, 4)`
    /// in `[x, y, z, w]` order.
    func rigRotationFrame(at i: Int) -> [Float]? {
        guard let rigRotations, rigJointCount > 0 else { return nil }
        precondition(i >= 0 && i < frameCount, "frame index out of range")
        let stride = rigJointCount * 4
        let start = i * stride
        return Array(rigRotations[start..<start + stride])
    }
}

/// One frame in the pose stream consumed by the renderer: 45 joints × xyz.
/// Joint ordering matches `Landmark45.jointOrder`.
struct PoseFrame: Sendable {
    /// Length 135 (= 45 × 3) row-major xyz.
    var values: [Float]
    /// Optional flattened `(rigJointCount, 4)` rig-local quaternions in `[x,y,z,w]`.
    var rigRotations: [Float]? = nil
    var timestamp: TimeInterval

    static var jointCount: Int { 45 }
    static var dim: Int { 135 }

    /// (x, y, z) for joint at `index`.
    func point(at index: Int) -> SIMD3<Float> {
        let base = index * 3
        return SIMD3(values[base], values[base + 1], values[base + 2])
    }

    /// Linear interpolation between two frames.
    static func lerp(_ a: PoseFrame, _ b: PoseFrame, t: Float) -> PoseFrame {
        precondition(a.values.count == b.values.count)
        var out = [Float](repeating: 0, count: a.values.count)
        let one = 1.0 - t
        for i in 0..<a.values.count {
            out[i] = a.values[i] * one + b.values[i] * t
        }
        return PoseFrame(
            values: out,
            rigRotations: nil,
            timestamp: a.timestamp + Double(t) * (b.timestamp - a.timestamp)
        )
    }
}
