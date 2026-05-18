import Foundation

/// Concatenates per-gloss clips into a continuous PoseFrame stream, with
/// an 8-frame linear-blend handoff between adjacent clips.
///
/// Per `mobile-app/docs/path_b_avatar.md`:
/// - Source fps = 24 (from BVH `Frame Time`).
/// - Unknown tokens (`PoseDatabase.lookup` returns `nil`) are skipped with
///   a logged warning; the renderer holds the last frame.
actor Stitcher {
    private let database: PoseDatabase
    private let handoffFrames: Int
    private let sourceFps: Float
    /// Multiplier on wall-clock playback speed. 1.0 = native 24 fps (how
    /// the BVHs were authored); 1.25 trims the inter-frame sleep so signs
    /// finish a bit quicker without resampling the underlying clips. The
    /// timestamps we attach to each `PoseFrame` still advance at the
    /// native dt so downstream filters (Savitzky-Golay, One-Euro) keep
    /// their time-domain semantics — only the wall-clock pacing changes.
    private let playbackRate: Float
    private var previousFrame: PoseFrame? = nil
    private var nextTimestamp: TimeInterval = 0

    /// Output stream consumed by `AvatarStreamPlayer`.
    let frames: AsyncStream<PoseFrame>
    private let framesContinuation: AsyncStream<PoseFrame>.Continuation

    /// Unknown-gloss events, for telemetry/debug.
    let missing: AsyncStream<String>
    private let missingContinuation: AsyncStream<String>.Continuation

    init(database: PoseDatabase,
         handoffFrames: Int = 8,
         sourceFps: Float = 24,
         playbackRate: Float = 1.25) {
        self.database = database
        self.handoffFrames = handoffFrames
        self.sourceFps = sourceFps
        self.playbackRate = max(0.1, playbackRate)

        var fc: AsyncStream<PoseFrame>.Continuation!
        self.frames = AsyncStream { c in fc = c }
        self.framesContinuation = fc

        var mc: AsyncStream<String>.Continuation!
        self.missing = AsyncStream { c in mc = c }
        self.missingContinuation = mc
    }

    /// Append a gloss token to the stream. Emits the clip's frames (after
    /// an optional blend from the previous frame) onto `frames`.
    func append(_ token: String) async {
        let clip: PoseClip?
        do {
            clip = try await database.lookup(token)
        } catch {
            print("[Stitcher] lookup error token='\(token)' error=\(error)")
            missingContinuation.yield("error:\(token)")
            return
        }
        guard let clip else {
            print("[Stitcher] missing token='\(token)'")
            missingContinuation.yield(token)
            return
        }
        print("[Stitcher] append token='\(token)' frames=\(clip.frameCount) fps=\(clip.fps)")

        let dt = TimeInterval(1.0 / sourceFps)
        // Wall-clock sleep is shorter than `dt` when playbackRate > 1.
        // `nextTimestamp` (advanced by `dt` below) is the *content* time
        // the smoothing filters use, which we keep at native rate.
        let sleepNanos = UInt64((dt / TimeInterval(playbackRate)) * 1_000_000_000)

        if let prev = previousFrame {
            let firstFrame = PoseFrame(
                values: clip.frame(at: 0),
                rigRotations: clip.rigRotationFrame(at: 0),
                timestamp: nextTimestamp
            )
            for i in 1...handoffFrames {
                if Task.isCancelled { return }
                let t = Float(i) / Float(handoffFrames + 1)
                var blended = PoseFrame.lerp(prev, firstFrame, t: t)
                // During cross-gloss handoff, keep the next clip's local rotations
                // so the 3D rig uses real BVH channels whenever available.
                blended.rigRotations = firstFrame.rigRotations
                blended.timestamp = nextTimestamp
                framesContinuation.yield(blended)
                nextTimestamp += dt
                // Rate-limit yields to source fps so the renderer sees motion
                // unfold over wall-clock time instead of a single-frame jump.
                try? await Task.sleep(nanoseconds: sleepNanos)
            }
        }

        for k in 0..<clip.frameCount {
            // Cooperative cancellation: when the caller (PathBCoordinator)
            // cancels the active playback Task — e.g. because the user
            // tapped a new button or the ASR transcript advanced — we
            // bail out of the clip immediately rather than running to
            // completion in the background.
            if Task.isCancelled { return }
            let f = PoseFrame(
                values: clip.frame(at: k),
                rigRotations: clip.rigRotationFrame(at: k),
                timestamp: nextTimestamp
            )
            framesContinuation.yield(f)
            previousFrame = f
            nextTimestamp += dt
            try? await Task.sleep(nanoseconds: sleepNanos)
        }
    }

    /// Reset between distinct user utterances so the next handoff starts fresh.
    func reset(startTime: TimeInterval = 0) {
        previousFrame = nil
        nextTimestamp = startTime
    }

    /// Tear down the stream. Call when Path B is paused/stopped.
    func finish() {
        framesContinuation.finish()
        missingContinuation.finish()
    }
}
