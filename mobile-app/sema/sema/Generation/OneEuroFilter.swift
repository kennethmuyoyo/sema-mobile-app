import Foundation

/// One-Euro filter — an adaptive low-pass filter for noisy real-time signals.
///
/// Casiez, Roussel & Vogel, *"1€ Filter: A Simple Speed-based Low-pass Filter
/// for Noisy Input in Interactive Systems"* (CHI 2012).
///
/// Idea:
/// - At each frame, compute a smoothed *derivative* of the signal.
/// - Use the magnitude of that derivative to pick a *per-frame cutoff*:
///       cutoff = minCutoff + beta * |edx|
/// - Run a single-pole low-pass on the value with that cutoff.
///
/// Effect: when the signal is steady (small |edx|), the cutoff is low and the
/// filter aggressively kills jitter. When the signal is moving fast (large
/// |edx|), the cutoff opens up and the filter passes the motion through
/// nearly untouched. Causal (no look-ahead), so it adds **zero lag** to the
/// pipeline — the right complement to the centered Savitzky-Golay smoother
/// upstream.
///
/// Channel-wise: each scalar in the input vector keeps its own (prevValue,
/// prevFiltered, prevDx) state. For pose frames we filter the 135 landmark
/// coords and the 196 quaternion components independently, then renormalise
/// each quaternion on the way out.
final class OneEuroFilter {
    /// Hz. Lower → smoother holds, more apparent lag during slow motion.
    /// Paper default 1.0; for 24 fps signing, 1.0–1.5 works well.
    var minCutoff: Float
    /// Responsiveness scaling. Higher → cutoff opens more aggressively on
    /// fast motion (less smoothing during sign transitions). Paper default
    /// 0.007; for fast hand transitions in signing we want a bit more
    /// responsiveness, so 0.05 is a sensible start.
    var beta: Float
    /// Hz. Cutoff for the derivative low-pass. Paper default 1.0.
    var dCutoff: Float

    private let dt: Float
    private var prevValue: [Float] = []
    private var prevFiltered: [Float] = []
    private var prevDx: [Float] = []
    private var primed = false

    init(sampleRate: Float = 24.0, minCutoff: Float = 1.0, beta: Float = 0.05, dCutoff: Float = 1.0) {
        precondition(sampleRate > 0)
        self.dt = 1.0 / sampleRate
        self.minCutoff = minCutoff
        self.beta = beta
        self.dCutoff = dCutoff
    }

    /// Filter a vector of per-channel scalar values. The channel count is
    /// fixed by the first call; subsequent calls must match.
    func filter(_ values: [Float]) -> [Float] {
        if !primed {
            prevValue = values
            prevFiltered = values
            prevDx = Array(repeating: 0, count: values.count)
            primed = true
            return values
        }
        precondition(values.count == prevValue.count,
                     "channel count must stay constant; got \(values.count) vs \(prevValue.count)")

        var out = [Float](repeating: 0, count: values.count)
        for i in 0..<values.count {
            let x = values[i]
            let dx = (x - prevValue[i]) / dt
            let edx = lowPass(dx, prev: prevDx[i], alpha: alpha(cutoff: dCutoff))
            let cutoff = minCutoff + beta * abs(edx)
            let filtered = lowPass(x, prev: prevFiltered[i], alpha: alpha(cutoff: cutoff))

            out[i] = filtered
            prevValue[i] = x
            prevDx[i] = edx
            prevFiltered[i] = filtered
        }
        return out
    }

    /// Re-arm the filter. Call between utterances so the first frame of a
    /// new clip isn't smoothed against stale state from the previous one.
    func reset() {
        primed = false
        prevValue.removeAll(keepingCapacity: true)
        prevFiltered.removeAll(keepingCapacity: true)
        prevDx.removeAll(keepingCapacity: true)
    }

    // MARK: - Math

    @inline(__always)
    private func alpha(cutoff: Float) -> Float {
        let tau = 1.0 / (2.0 * .pi * cutoff)
        return 1.0 / (1.0 + tau / dt)
    }

    @inline(__always)
    private func lowPass(_ value: Float, prev: Float, alpha: Float) -> Float {
        alpha * value + (1.0 - alpha) * prev
    }
}
