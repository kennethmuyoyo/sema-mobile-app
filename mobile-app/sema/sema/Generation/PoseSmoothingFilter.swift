import Foundation

/// Savitzky-Golay smoothing filter for pose frames. Applies a 5-tap order-2
/// polynomial filter along the time axis, per-coordinate for `values` and
/// per-quaternion-component for `rigRotations` (with renormalization).
///
/// Output is delayed by `halfWindow` frames (2 @ window=5 → ~83ms at 24fps).
/// Auto-resets when input goes quiet for `quietThreshold` seconds, so a new
/// utterance doesn't blend with stale frames from a previous one.
final class PoseSmoothingFilter {
    private let window: Int
    private let halfWindow: Int
    private let coefficients: [Float]
    private let quietThreshold: TimeInterval = 0.4

    private var buffer: [PoseFrame] = []
    private var lastPushAt: Date?

    init(window: Int = 5) {
        precondition(window % 2 == 1 && window >= 3, "window must be odd and >= 3")
        self.window = window
        self.halfWindow = window / 2
        switch window {
        case 5:
            self.coefficients = [-3, 12, 17, 12, -3].map { $0 / 35.0 }
        case 7:
            self.coefficients = [-2, 3, 6, 7, 6, 3, -2].map { $0 / 21.0 }
        case 9:
            self.coefficients = [-21, 14, 39, 54, 59, 54, 39, 14, -21].map { $0 / 231.0 }
        default:
            // Fallback: uniform moving average. Not Savitzky-Golay, but at
            // least produces a sane result for non-canonical window sizes.
            self.coefficients = Array(repeating: 1.0 / Float(window), count: window)
        }
    }

    /// Push a new frame and return a smoothed frame. On the first call after
    /// a reset or quiescence, the buffer is primed with copies of `frame` so
    /// the very first emission is well-defined and the avatar doesn't pop
    /// backward when the buffer transitions from partial to full.
    func push(_ frame: PoseFrame) -> PoseFrame {
        let now = Date()
        let quiescent = lastPushAt.map { now.timeIntervalSince($0) > quietThreshold } ?? false
        if buffer.isEmpty || quiescent {
            buffer = Array(repeating: frame, count: window - 1)
        }
        lastPushAt = now
        buffer.append(frame)
        let smoothed = smoothCenter()
        buffer.removeFirst()
        return smoothed
    }

    func reset() {
        buffer.removeAll(keepingCapacity: true)
        lastPushAt = nil
    }

    private func smoothCenter() -> PoseFrame {
        let base = buffer[halfWindow]
        let n = base.values.count
        var values = [Float](repeating: 0, count: n)
        for i in 0..<window {
            let w = coefficients[i]
            let src = buffer[i].values
            for k in 0..<n {
                values[k] += w * src[k]
            }
        }

        let rig = smoothRigRotations(base: base)
        return PoseFrame(values: values, rigRotations: rig, timestamp: base.timestamp)
    }

    private func smoothRigRotations(base: PoseFrame) -> [Float]? {
        guard let baseRig = base.rigRotations else { return nil }
        let n = baseRig.count
        var out = [Float](repeating: 0, count: n)
        for i in 0..<window {
            guard let rig = buffer[i].rigRotations, rig.count == n else {
                // A frame in the window is missing rig data — bail and use
                // the center frame's rig as-is so we don't emit garbage.
                return baseRig
            }
            let w = coefficients[i]
            for k in 0..<n {
                out[k] += w * rig[k]
            }
        }
        // Renormalize per-joint quaternion. Component-wise SG on (x,y,z,w)
        // produces near-unit vectors but not exactly unit; renorm restores
        // the manifold so SceneKit treats them as valid rotations.
        let jointCount = n / 4
        for j in 0..<jointCount {
            let b = j * 4
            let x = out[b]
            let y = out[b + 1]
            let z = out[b + 2]
            let w = out[b + 3]
            let norm = sqrt(x * x + y * y + z * z + w * w)
            if norm > 0.0001 {
                out[b] = x / norm
                out[b + 1] = y / norm
                out[b + 2] = z / norm
                out[b + 3] = w / norm
            }
        }
        return out
    }
}
