import Foundation

/// One frame of recognizer-ready features.
///
/// `values` is the **model input**: 45 joints × (x, y, z) interleaved,
/// length 135 — the same `(T, 45, 3)` layout the training data has in
/// `data/mediapipe_landmarks/*.npy`. Missing joints are `(0, 0, 0)`.
/// `mask` is a sibling 45-length Swift-only array — used by the skeleton
/// overlay to fade undetected joints; never goes into the model tensor.
struct NormalizedFrame: Sendable {
    var values: [Float]
    var mask: [Float]
    var timestamp: TimeInterval

    static var dim: Int { Landmark45.featureDim }   // 135

    /// (x, y, z, mask) for the joint at `index`. The mask is read from the
    /// sibling `mask` array; the model input itself carries no mask channel.
    func joint(at index: Int) -> SIMD4<Float> {
        let base = index * 3
        return SIMD4(values[base], values[base + 1], values[base + 2], mask[index])
    }
}
