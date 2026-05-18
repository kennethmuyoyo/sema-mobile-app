import Foundation

/// The 45-joint layout shared between the recognizer's training pipeline
/// (`recognition/data/landmarks_meta.json`) and the iOS HolisticLandmarker
/// output. This is the single source of truth on iOS for joint ordering and
/// the MediaPipe → 45-joint index mapping.
///
/// Body joints come from MediaPipe `PoseLandmarker` (33-landmark layout).
/// Finger joints come from MediaPipe `HandLandmarker` (21 landmarks per hand;
/// we use only MCP/PIP/DIP for each finger and skip the tips).
enum Landmark45 {
    static let count = 45

    /// `count × (x, y, z)` — matches the v11 recognizer's 135-dim input.
    /// MediaPipe-rendered training data is shape `(T, 45, 3)`, so the iOS
    /// app's per-frame feature vector is xyz only, 45 joints, length 135.
    /// A sibling 45-length detection mask is kept on `NormalizedFrame` for
    /// the skeleton overlay (UI-only); the model never sees it.
    static let featureDim = count * 3   // 135

    /// 45 joint names in the exact order of `landmarks_meta.json`.
    /// DO NOT reorder without regenerating the recognizer's input.
    static let jointOrder: [String] = [
        // Body 15
        "head",
        "left_eye_smplhf",
        "right_eye_smplhf",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        // Left hand 15 — index → middle → pinky → ring → thumb
        "left_index1", "left_index2", "left_index3",
        "left_middle1", "left_middle2", "left_middle3",
        "left_pinky1", "left_pinky2", "left_pinky3",
        "left_ring1", "left_ring2", "left_ring3",
        "left_thumb1", "left_thumb2", "left_thumb3",
        // Right hand 15
        "right_index1", "right_index2", "right_index3",
        "right_middle1", "right_middle2", "right_middle3",
        "right_pinky1", "right_pinky2", "right_pinky3",
        "right_ring1", "right_ring2", "right_ring3",
        "right_thumb1", "right_thumb2", "right_thumb3",
    ]

    private static let indexByName: [String: Int] = Dictionary(
        uniqueKeysWithValues: jointOrder.enumerated().map { ($0.element, $0.offset) }
    )

    static func index(of jointName: String) -> Int {
        guard let idx = indexByName[jointName] else {
            fatalError("Unknown joint name '\(jointName)'")
        }
        return idx
    }

    /// Hip / knee / ankle slots. Sign language lives in the upper body and
    /// the hands — the lower body adds noise to the recognizer and clutters
    /// the debug overlay, so we deliberately zero these out (position and
    /// mask) at landmark-assembly time.
    static let lowerBodyJointIndices: [Int] = [
        "left_hip", "right_hip",
        "left_knee", "right_knee",
        "left_ankle", "right_ankle",
    ].map(index(of:))

    /// Per-joint training-time mean (in shoulder-normalised frame), extracted
    /// from `v3_stats.npz` — the same stats the CoreML model's baked-in
    /// z-score layer subtracts internally.
    ///
    /// Why: the model bakes `(x - mean) / std` before the encoder. Some joints
    /// have absurdly tight training stds (e.g. ankle.y std ≈ 0.14 around mean
    /// 3.79, hip.y std ≈ 0.08 around mean 1.50, shoulder.x std ≈ 0.001). When
    /// a joint is missing on iOS and we send 0, the z-score turns into a
    /// ±15…26σ outlier the transformer has never seen → uniform softmax. By
    /// filling missing joints with these means, the z-score maps them to 0,
    /// which is the neutral "no signal" value the model was trained on.
    static let trainingMean: [SIMD3<Float>] = [
        SIMD3(+0.017444, -0.551718, -0.079666),  // head
        SIMD3(+0.060428, -0.706972, +0.094708),  // left_eye_smplhf
        SIMD3(-0.124010, -0.690865, +0.092570),  // right_eye_smplhf
        SIMD3(+0.499428, +0.020980, +0.020343),  // left_shoulder
        SIMD3(-0.499428, -0.020980, -0.020343),  // right_shoulder
        SIMD3(+0.590267, +0.762653, +0.228712),  // left_elbow
        SIMD3(-0.733253, +0.617735, +0.354945),  // right_elbow
        SIMD3(+0.444519, +0.836924, +0.867460),  // left_wrist
        SIMD3(-0.474949, +0.472480, +0.890996),  // right_wrist
        SIMD3(+0.166783, +1.504120, -0.102268),  // left_hip
        SIMD3(-0.216331, +1.537294, -0.084944),  // right_hip
        SIMD3(+0.332577, +2.600080, -0.121494),  // left_knee
        SIMD3(-0.351144, +2.585558, -0.129771),  // right_knee
        SIMD3(+0.200458, +3.783087, -0.219452),  // left_ankle
        SIMD3(-0.306187, +3.789783, -0.192145),  // right_ankle
        SIMD3(+0.366206, +0.837537, +1.115727),  // left_index1
        SIMD3(+0.352961, +0.868890, +1.185706),  // left_index2
        SIMD3(+0.342830, +0.904116, +1.223745),  // left_index3
        SIMD3(+0.417939, +0.862541, +1.141575),  // left_middle1
        SIMD3(+0.403626, +0.902960, +1.201900),  // left_middle2
        SIMD3(+0.395749, +0.942618, +1.239211),  // left_middle3
        SIMD3(+0.497748, +0.929677, +1.083166),  // left_pinky1
        SIMD3(+0.496372, +0.951480, +1.123944),  // left_pinky2
        SIMD3(+0.500458, +0.979802, +1.156106),  // left_pinky3
        SIMD3(+0.466909, +0.898141, +1.115734),  // left_ring1
        SIMD3(+0.453345, +0.926505, +1.176227),  // left_ring2
        SIMD3(+0.448159, +0.962373, +1.215888),  // left_ring3
        SIMD3(+0.353459, +0.846216, +0.954631),  // left_thumb1
        SIMD3(+0.298661, +0.856027, +1.003404),  // left_thumb2
        SIMD3(+0.273937, +0.868913, +1.059030),  // left_thumb3
        SIMD3(-0.369449, +0.385773, +1.060285),  // right_index1
        SIMD3(-0.349830, +0.384643, +1.114915),  // right_index2
        SIMD3(-0.336296, +0.401917, +1.144707),  // right_index3
        SIMD3(-0.408879, +0.398860, +1.097946),  // right_middle1
        SIMD3(-0.389353, +0.413108, +1.143868),  // right_middle2
        SIMD3(-0.378950, +0.441842, +1.167507),  // right_middle3
        SIMD3(-0.481142, +0.476988, +1.087981),  // right_pinky1
        SIMD3(-0.470631, +0.486852, +1.117493),  // right_pinky2
        SIMD3(-0.467508, +0.504421, +1.138705),  // right_pinky3
        SIMD3(-0.452274, +0.438829, +1.098065),  // right_ring1
        SIMD3(-0.429596, +0.445033, +1.141571),  // right_ring2
        SIMD3(-0.419432, +0.467866, +1.167426),  // right_ring3
        SIMD3(-0.380148, +0.450138, +0.936171),  // right_thumb1
        SIMD3(-0.325617, +0.442042, +0.969106),  // right_thumb2
        SIMD3(-0.297114, +0.428849, +1.012003),  // right_thumb3
    ]

    /// Body joint → MediaPipe `PoseLandmarker` landmark index (0..32).
    static let mediaPipeBodyIndex: [String: Int] = [
        "head": 0,                 // MediaPipe "nose" used as the head anchor
        "left_eye_smplhf": 2,
        "right_eye_smplhf": 5,
        "left_shoulder": 11,
        "right_shoulder": 12,
        "left_elbow": 13,
        "right_elbow": 14,
        "left_wrist": 15,
        "right_wrist": 16,
        "left_hip": 23,
        "right_hip": 24,
        "left_knee": 25,
        "right_knee": 26,
        "left_ankle": 27,
        "right_ankle": 28,
    ]

    /// Finger-segment suffix → MediaPipe `HandLandmarker` landmark index (0..20).
    /// MCP = first knuckle, PIP = second, DIP = third. Tips (4, 8, 12, 16, 20) omitted.
    static let mediaPipeHandIndex: [String: Int] = [
        "thumb1": 1, "thumb2": 2, "thumb3": 3,
        "index1": 5, "index2": 6, "index3": 7,
        "middle1": 9, "middle2": 10, "middle3": 11,
        "ring1": 13, "ring2": 14, "ring3": 15,
        "pinky1": 17, "pinky2": 18, "pinky3": 19,
    ]
}
