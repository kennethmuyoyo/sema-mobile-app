import Foundation
import Observation
import simd

/// Holds the latest `PoseFrame` from a Stitcher and exposes it to SwiftUI so
/// the 3D avatar view can pull `rawFrame` each render tick.
///
/// Also runs an idle animation loop: when no clip frames have arrived
/// recently, the player emits a relaxed pose with a subtle breathing
/// modulation so the avatar doesn't sit in T-pose between utterances.
///
/// Smoothing chain:
///     Stitcher → Savitzky-Golay (centered, kills boundary discontinuities)
///              → One-Euro     (causal, adaptive — kills residual jitter)
///              → apply(rawFrame)
/// SG fixes the velocity discontinuity at gloss boundaries that linear-blend
/// alone leaves behind; One-Euro takes the smoothed but still-jittery output
/// and adapts its cutoff per-frame so slow holds get extra smoothing while
/// fast sign onsets stay crisp. Both filters auto-reset after a quiescence
/// gap so a new utterance doesn't blend against stale state.
@MainActor
@Observable
final class AvatarStreamPlayer {
    /// Most recent frame applied. Read by `SimpleAvatar3DView` each render tick.
    var rawFrame: PoseFrame?

    /// True after the first real (non-idle) frame has been applied.
    private(set) var hasReceivedRealFrame = false

    /// True while clip frames are actively arriving from the Stitcher — i.e.
    /// the avatar is mid-sign rather than holding the idle pose. Read by the
    /// orchestrator's turn-taking loop to know when the avatar has finished
    /// translating a spoken utterance and it's time to switch back to
    /// watching mode for the deaf user's reply.
    var isAvatarSigning: Bool {
        guard let last = lastRealFrameAt else { return false }
        return Date().timeIntervalSince(last) < idleTakeoverThreshold
    }

    @ObservationIgnored private let smoother = PoseSmoothingFilter(window: 5)
    @ObservationIgnored private let oneEuroValues = OneEuroFilter(sampleRate: 24.0, minCutoff: 1.2, beta: 0.05)
    @ObservationIgnored private let oneEuroQuats = OneEuroFilter(sampleRate: 24.0, minCutoff: 1.5, beta: 0.08)
    @ObservationIgnored private var lastIngestAt: Date?
    /// If no Stitcher frame arrives within this window, treat the next frame
    /// as the start of a fresh utterance and reset both filters so we don't
    /// blend the new clip against trailing state from the previous one.
    @ObservationIgnored private let filterQuietThreshold: TimeInterval = 0.4

    @ObservationIgnored private var idleBase: PoseFrame?
    @ObservationIgnored private var idleStart: Date = Date()
    @ObservationIgnored private var lastRealFrameAt: Date?
    @ObservationIgnored private var idleTask: Task<Void, Never>?

    /// Threshold after which idle motion takes over. Slightly larger than the
    /// inter-frame interval at 24fps so idle doesn't fight active playback.
    private let idleTakeoverThreshold: TimeInterval = 0.35

    /// Consume frames from a Stitcher's output stream. Cancellation-safe.
    func attach(_ stream: AsyncStream<PoseFrame>) -> Task<Void, Never> {
        Task { [weak self] in
            for await frame in stream {
                guard let self else { return }
                self.ingest(frame)
            }
        }
    }

    /// Directly push a single frame — useful for manual testing. Bypasses the
    /// smoother since callers using this generally want immediate updates.
    func apply(_ frame: PoseFrame) {
        rawFrame = frame
        hasReceivedRealFrame = true
        lastRealFrameAt = Date()
    }

    /// Register a relaxed pose used between utterances. The first call also
    /// seeds `rawFrame` so the avatar shows the idle pose immediately instead
    /// of the rig's T-pose default.
    func setIdleBase(_ frame: PoseFrame) {
        idleBase = frame
        if rawFrame == nil {
            rawFrame = frame
        }
    }

    /// Start the idle animation loop. No-op if already running.
    func startIdleLoop() {
        guard idleTask == nil else { return }
        idleStart = Date()
        idleTask = Task { [weak self] in
            let dt = UInt64(1_000_000_000 / 24)
            while !Task.isCancelled {
                await MainActor.run { self?.tickIdle() }
                try? await Task.sleep(nanoseconds: dt)
            }
        }
    }

    func stopIdleLoop() {
        idleTask?.cancel()
        idleTask = nil
    }

    // MARK: - Internals

    private func ingest(_ frame: PoseFrame) {
        let now = Date()
        if let last = lastIngestAt, now.timeIntervalSince(last) > filterQuietThreshold {
            // PoseSmoothingFilter has its own auto-reset on quiescence; do
            // the One-Euro side here for symmetry.
            oneEuroValues.reset()
            oneEuroQuats.reset()
        }
        lastIngestAt = now

        let smoothed = smoother.push(frame)
        let dejittered = applyOneEuro(smoothed)
        apply(dejittered)
    }

    /// Pass the SG output through One-Euro per channel.
    /// - Positions: 135 channels (45 joints × 3 coords) filtered straight.
    /// - Quaternions: 196 components (49 joints × 4) filtered component-wise,
    ///   then each quaternion renormalised. Component-wise filtering can
    ///   nudge quats off the unit hypersphere; renorm restores the manifold
    ///   so SceneKit treats them as valid rotations.
    private func applyOneEuro(_ frame: PoseFrame) -> PoseFrame {
        let values = oneEuroValues.filter(frame.values)
        var rig: [Float]? = nil
        if let q = frame.rigRotations {
            var filtered = oneEuroQuats.filter(q)
            let jointCount = filtered.count / 4
            for j in 0..<jointCount {
                let b = j * 4
                let x = filtered[b]
                let y = filtered[b + 1]
                let z = filtered[b + 2]
                let w = filtered[b + 3]
                let n = sqrt(x * x + y * y + z * z + w * w)
                if n > 0.0001 {
                    filtered[b]     = x / n
                    filtered[b + 1] = y / n
                    filtered[b + 2] = z / n
                    filtered[b + 3] = w / n
                }
            }
            rig = filtered
        }
        return PoseFrame(values: values, rigRotations: rig, timestamp: frame.timestamp)
    }

    private func tickIdle() {
        guard let base = idleBase else { return }
        if let last = lastRealFrameAt, Date().timeIntervalSince(last) < idleTakeoverThreshold {
            // A real clip frame arrived recently — don't fight it.
            return
        }
        let elapsed = Float(Date().timeIntervalSince(idleStart))
        rawFrame = breathing(base: base, elapsed: elapsed)
    }

    /// Apply a subtle forward/back chest sway to the idle base pose by
    /// multiplying spine1's local quaternion by a small X-axis rotation.
    /// Amplitude is intentionally tiny (~1.4°) — sign is communicative;
    /// the idle motion just needs to look alive, not distracting.
    private func breathing(base: PoseFrame, elapsed: Float) -> PoseFrame {
        guard var rig = base.rigRotations,
              let spine1Index = BVHRigRotationLayout.jointOrder.firstIndex(of: "spine1")
        else {
            return base
        }
        let period: Float = 4.0
        let amplitude: Float = 0.025  // radians ≈ 1.4°
        let phase = sin(elapsed * 2.0 * .pi / period)
        let extra = simd_quatf(angle: amplitude * phase, axis: SIMD3<Float>(1, 0, 0))

        let b = spine1Index * 4
        let q = simd_quatf(ix: rig[b], iy: rig[b + 1], iz: rig[b + 2], r: rig[b + 3])
        let combined = simd_normalize(q * extra)
        rig[b] = combined.imag.x
        rig[b + 1] = combined.imag.y
        rig[b + 2] = combined.imag.z
        rig[b + 3] = combined.real

        return PoseFrame(values: base.values, rigRotations: rig, timestamp: base.timestamp)
    }
}
