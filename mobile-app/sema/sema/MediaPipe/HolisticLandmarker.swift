import CoreVideo
import Foundation
import MediaPipeTasksVision
import simd

/// Combines MediaPipe `PoseLandmarker` + `HandLandmarker` into a single
/// per-frame producer that emits `NormalizedFrame` directly, in the 45-joint
/// layout the recognizer expects.
///
/// Coordinate convention: MediaPipe returns image-normalized coordinates with
/// `y` pointing **down**, which matches what `bvh_to_landmarks.py` produces
/// for training (it y-flips the BVH y-up frame). So no further y-flip here.
///
/// Normalisation matches `normalize_landmarks` in `bvh_to_landmarks.py`:
///   origin = mid(L_shoulder, R_shoulder)
///   scale  = max(|L_shoulder - R_shoulder|, 1e-3)
///   p'     = (p - origin) / scale
///
/// The per-joint mask channel records whether the source MediaPipe detector
/// produced a confident landmark for that joint (1.0) or dropped it (0.0).
/// Dropped joints have their xyz triple zeroed — matching the training-time
/// dropout-and-mask augmentation in `data/augment.py`.
actor HolisticLandmarker {
    private let pose: PoseLandmarker
    private let hand: HandLandmarker

    init() throws {
        let posePath = try Self.bundleAssetPath(named: "pose_landmarker_full",
                                                 ext: "task",
                                                 fallback: .poseAssetMissing)
        // Match training-time config exactly. `render_bvh_to_mediapipe.py`
        // used 0.5 for detection AND tracking on both pose and hand; the iOS
        // app previously lowered everything to 0.3 / 0.2 to get more frames
        // through, but that let low-confidence (noisy-position) landmarks
        // into the recognizer's input — landmarks the model never saw at
        // training time. Result: extreme logits and random predictions
        // because the input distribution drifted. Bumped back to 0.5 to
        // keep the iOS-side detector output in the same regime training
        // sampled from.
        let poseOptions = PoseLandmarkerOptions()
        poseOptions.baseOptions.modelAssetPath = posePath
        poseOptions.runningMode = .video
        poseOptions.numPoses = 1
        poseOptions.minPoseDetectionConfidence = 0.5
        poseOptions.minPosePresenceConfidence = 0.5
        poseOptions.minTrackingConfidence = 0.5
        self.pose = try PoseLandmarker(options: poseOptions)

        let handPath = try Self.bundleAssetPath(named: "hand_landmarker",
                                                 ext: "task",
                                                 fallback: .handAssetMissing)
        let handOptions = HandLandmarkerOptions()
        handOptions.baseOptions.modelAssetPath = handPath
        handOptions.runningMode = .video
        handOptions.numHands = 2
        handOptions.minHandDetectionConfidence = 0.5
        handOptions.minHandPresenceConfidence = 0.5
        handOptions.minTrackingConfidence = 0.5
        self.hand = try HandLandmarker(options: handOptions)
    }

    /// Resolve a bundled MediaPipe `.task` file to an absolute path and
    /// emit one line of diagnostic output so we can debug "Unable to open
    /// zip archive" errors: path, file size, first 8 bytes (hex). Strict
    /// ZIP parsers fail if anything corrupted the file during bundle copy.
    private static func bundleAssetPath(named name: String,
                                         ext: String,
                                         fallback: HolisticLandmarkerError) throws -> String {
        // Prefer `path(forResource:ofType:)` — it returns a clean filesystem
        // path, whereas `url(forResource:withExtension:).path` has bitten us
        // historically with percent-encoded characters.
        let resolved: String? =
            Bundle.main.path(forResource: name, ofType: ext) ??
            Bundle.main.url(forResource: name, withExtension: ext)?.path
        guard let path = resolved else { throw fallback }

        // Diagnostic: log size and ZIP magic so we can debug bundle-copy issues.
        var summary = "[HolisticLandmarker] \(name).\(ext) → \(path)"
        if let attrs = try? FileManager.default.attributesOfItem(atPath: path),
           let size = attrs[.size] as? NSNumber {
            summary += "  size=\(size.intValue)"
        }
        if let handle = FileHandle(forReadingAtPath: path) {
            defer { try? handle.close() }
            if let head = try? handle.read(upToCount: 8) {
                summary += "  head=" + head.map { String(format: "%02x", $0) }.joined()
            }
        }
        print(summary)
        return path
    }

    /// Run both landmarkers on a video frame. `timestampMs` must be strictly
    /// increasing per session — MediaPipe's `.video` running mode enforces it.
    /// Returns `nil` if either shoulder is missing (we can't normalise).
    func process(pixelBuffer: CVPixelBuffer, timestampMs: Int) throws -> NormalizedFrame? {
        let mpImage = try MPImage(pixelBuffer: pixelBuffer)
        let poseResult = try pose.detect(videoFrame: mpImage, timestampInMilliseconds: timestampMs)
        let handResult = try hand.detect(videoFrame: mpImage, timestampInMilliseconds: timestampMs)
        return assemble(
            poseResult: poseResult,
            handResult: handResult,
            timestampMs: timestampMs
        )
    }

    /// Returns true iff the previous frame had at least one detected hand.
    /// Used by the coordinator to reset the streaming CTC decoder after a
    /// no-hand timeout.
    private(set) var lastFrameHadHand: Bool = false

    /// Wipe the carry-forward buffers so the next detected frame starts
    /// from a clean slate. Called by the orchestrator on mode-switch — the
    /// previous watching session's stale `lastNormalized` would otherwise
    /// fill in joints the new session can't actually see yet.
    func reset() {
        lastNormalized = Array(repeating: nil, count: Landmark45.count)
        framesSinceValid = Array(repeating: .max, count: Landmark45.count)
        lastFrameHadHand = false
    }

    /// Last shoulder-normalised position MediaPipe reported per joint,
    /// used for temporal carry-forward when the detector blinks on a joint
    /// for a frame or two (common during hand occlusion / head turns).
    /// Indexed by `Landmark45.jointOrder` slot.
    private var lastNormalized: [SIMD3<Float>?] = Array(repeating: nil, count: Landmark45.count)
    private var framesSinceValid: [Int] = Array(repeating: .max, count: Landmark45.count)
    /// Hold a stale joint for up to this many consecutive missed frames.
    /// 30 frames ≈ 1.25 s at 24 fps. Bumped up from 6 because PathA feeds
    /// the recognizer's per-frame `LayerNorm(135)`: when a finger joint
    /// drops out and reverts to a literal zero, the frame's variance
    /// collapses and LayerNorm blows up the activations (the training set
    /// only kept frames inside "visibility bursts" where ≥30% of hand
    /// joints were detected, so the model never learned to handle all-zero
    /// hand slots). 1.25 s bridges any realistic MediaPipe blink while
    /// still letting a truly off-screen hand revert to zero eventually.
    private let maxCarryFrames: Int = 30

    // MARK: - Assembly

    private func assemble(
        poseResult: PoseLandmarkerResult,
        handResult: HandLandmarkerResult,
        timestampMs: Int
    ) -> NormalizedFrame? {
        guard let body = poseResult.landmarks.first, body.count >= 33 else {
            lastFrameHadHand = false
            return nil
        }

        // Map MediaPipe handedness labels → our left/right slots.
        var leftHand: [NormalizedLandmark]? = nil
        var rightHand: [NormalizedLandmark]? = nil
        for i in 0..<handResult.landmarks.count where i < handResult.handedness.count {
            guard let category = handResult.handedness[i].first else { continue }
            switch (category.categoryName ?? "").lowercased() {
            case "left":  leftHand = handResult.landmarks[i]
            case "right": rightHand = handResult.landmarks[i]
            default:      break
            }
        }
        lastFrameHadHand = (leftHand != nil) || (rightHand != nil)

        // Collect raw positions and a per-joint detected mask.
        var rawPositions = [SIMD3<Float>](repeating: .zero, count: Landmark45.count)
        var mask = [Float](repeating: 0, count: Landmark45.count)

        for (jointIndex, jointName) in Landmark45.jointOrder.enumerated() {
            if let bodyIdx = Landmark45.mediaPipeBodyIndex[jointName] {
                guard bodyIdx < body.count else { continue }
                let lm = body[bodyIdx]
                // No per-landmark visibility filter — matches training-time
                // `render_bvh_to_mediapipe.py`, which trusts every landmark
                // returned by a pose detection that already cleared the 0.5
                // detection-confidence threshold. The earlier iOS-only
                // `visibility > 0.2` check zeroed out joints the model
                // never expected to see zeroed, producing OOD frame patterns
                // that drove logits to 60000+. Trust the detector here.
                rawPositions[jointIndex] = SIMD3(
                    Float(truncating: lm.x as NSNumber),
                    Float(truncating: lm.y as NSNumber),
                    Float(truncating: lm.z as NSNumber)
                )
                mask[jointIndex] = 1
            } else if let (side, segment) = parseHandJointName(jointName),
                      let segIdx = Landmark45.mediaPipeHandIndex[segment] {
                let arr: [NormalizedLandmark]? = (side == "left") ? leftHand : rightHand
                if let h = arr, segIdx < h.count {
                    let lm = h[segIdx]
                    rawPositions[jointIndex] = SIMD3(
                        Float(truncating: lm.x as NSNumber),
                        Float(truncating: lm.y as NSNumber),
                        Float(truncating: lm.z as NSNumber)
                    )
                    mask[jointIndex] = 1
                }
            }
        }

        // Keep lower-body joints populated: the v3 training data was rendered
        // from a full SMPL-X rig with real hip/knee/ankle positions (mean Y
        // around +1.5 to +3.8 shoulder-widths). Zeroing them here used to
        // happen for overlay cleanliness, but it produces a ~25σ input-stat
        // shift that flattens the recognizer's output distribution.

        // Both shoulders are required for normalisation.
        let lsIdx = Landmark45.index(of: "left_shoulder")
        let rsIdx = Landmark45.index(of: "right_shoulder")
        guard mask[lsIdx] > 0 && mask[rsIdx] > 0 else { return nil }
        let origin = (rawPositions[lsIdx] + rawPositions[rsIdx]) * 0.5
        let scale = max(simd_distance(rawPositions[lsIdx], rawPositions[rsIdx]), 1e-3)

        // 135-dim interleaved layout per joint: [x, y, z]. Missing joints
        // are zero-filled. This matches the `(T, 45, 3)` layout of the
        // training data in `data/mediapipe_landmarks/*.npy`. The detection
        // mask stays on the sibling `mask` array (overlay-only); it does
        // NOT go into the model tensor.
        //
        // Temporal carry-forward: when MediaPipe drops a joint for ≤
        // `maxCarryFrames` consecutive frames, we reuse its last shoulder-
        // normalised position rather than emitting zeros. This bridges
        // hand-occlusion blinks and brief head turns. The sibling mask is
        // set to 1 for carried-forward frames so the overlay still draws
        // the joint.
        var values = [Float](repeating: 0, count: NormalizedFrame.dim)
        for j in 0..<Landmark45.count {
            if mask[j] > 0 {
                let p = (rawPositions[j] - origin) / scale
                values[j * 3 + 0] = p.x
                values[j * 3 + 1] = p.y
                values[j * 3 + 2] = p.z
                lastNormalized[j] = p
                framesSinceValid[j] = 0
            } else if framesSinceValid[j] < maxCarryFrames,
                      let prev = lastNormalized[j] {
                values[j * 3 + 0] = prev.x
                values[j * 3 + 1] = prev.y
                values[j * 3 + 2] = prev.z
                // Half-confidence mask: lets the skeleton overlay keep drawing
                // (it checks `mask > 0`) while motion / signing-gate logic in
                // PathA can require `mask >= 1.0` to exclude these stale,
                // dx=dy=dz=0-by-construction carry-forwards.
                mask[j] = 0.5
                framesSinceValid[j] &+= 1
            } else {
                // Truly absent — emit zeros (mask stays 0 on the sibling).
                if framesSinceValid[j] < .max { framesSinceValid[j] &+= 1 }
            }
        }

        return NormalizedFrame(
            values: values,
            mask: mask,
            timestamp: TimeInterval(timestampMs) / 1000.0
        )
    }

    /// `left_index1` → ("left", "index1"); `right_thumb3` → ("right", "thumb3").
    private nonisolated func parseHandJointName(_ name: String) -> (String, String)? {
        let parts = name.split(separator: "_", maxSplits: 1).map(String.init)
        guard parts.count == 2 else { return nil }
        return (parts[0], parts[1])
    }
}

enum HolisticLandmarkerError: Error, CustomStringConvertible {
    case poseAssetMissing
    case handAssetMissing

    var description: String {
        switch self {
        case .poseAssetMissing:
            return "pose_landmarker_full.task is not in the app bundle. It should live under Sema/Resources/."
        case .handAssetMissing:
            return "hand_landmarker.task is not in the app bundle. It should live under Sema/Resources/."
        }
    }
}
