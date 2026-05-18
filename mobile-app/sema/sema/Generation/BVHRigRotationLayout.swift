import Foundation

/// Joint layout for the quaternion sidecar (`quat_f32: (T, N, 4)`).
/// Must match `generation/pose_library/retarget_to_target.py:TARGET_JOINT_ORDER`
/// and the bone names actually present in `hackathon.usdc`.
///
/// Order is parent-before-child (depth-first), so `parentIndex[i] < i`
/// for every i with a parent in this list, which the iOS retargeting
/// loop relies on when composing world rotations.
enum BVHRigRotationLayout {
    static let jointOrder: [String] = [
        "pelvis",
        "left_hip", "left_knee", "left_ankle", "left_foot",
        "right_hip", "right_knee", "right_ankle", "right_foot",
        "spine1", "spine2", "spine3",
        "neck", "head",
        "jaw",
        "left_eye_smplhf", "right_eye_smplhf",
        "left_collar", "left_shoulder", "left_elbow", "left_wrist",
        "left_index1", "left_index2", "left_index3",
        "left_middle1", "left_middle2", "left_middle3",
        "left_pinky1", "left_pinky2", "left_pinky3",
        "left_ring1", "left_ring2", "left_ring3",
        "left_thumb1", "left_thumb2", "left_thumb3",
        "right_collar", "right_shoulder", "right_elbow", "right_wrist",
        "right_index1", "right_index2", "right_index3",
        "right_middle1", "right_middle2", "right_middle3",
        "right_pinky1", "right_pinky2", "right_pinky3",
        "right_ring1", "right_ring2", "right_ring3",
        "right_thumb1", "right_thumb2", "right_thumb3",
    ]

    static let jointCount = jointOrder.count

    /// Parent map. `pelvis`'s parent in the rig is `root`, which is NOT in
    /// `jointOrder` (we don't retarget the world-placement bone), so pelvis
    /// has no entry here and resolves to parentIndex = -1.
    static let parentByName: [String: String] = [
        "left_hip": "pelvis",
        "left_knee": "left_hip",
        "left_ankle": "left_knee",
        "left_foot": "left_ankle",

        "right_hip": "pelvis",
        "right_knee": "right_hip",
        "right_ankle": "right_knee",
        "right_foot": "right_ankle",

        "spine1": "pelvis",
        "spine2": "spine1",
        "spine3": "spine2",
        "neck": "spine3",
        "head": "neck",
        "jaw": "head",
        "left_eye_smplhf": "head",
        "right_eye_smplhf": "head",

        "left_collar": "spine3",
        "left_shoulder": "left_collar",
        "left_elbow": "left_shoulder",
        "left_wrist": "left_elbow",
        "left_index1": "left_wrist",
        "left_index2": "left_index1",
        "left_index3": "left_index2",
        "left_middle1": "left_wrist",
        "left_middle2": "left_middle1",
        "left_middle3": "left_middle2",
        "left_pinky1": "left_wrist",
        "left_pinky2": "left_pinky1",
        "left_pinky3": "left_pinky2",
        "left_ring1": "left_wrist",
        "left_ring2": "left_ring1",
        "left_ring3": "left_ring2",
        "left_thumb1": "left_wrist",
        "left_thumb2": "left_thumb1",
        "left_thumb3": "left_thumb2",

        "right_collar": "spine3",
        "right_shoulder": "right_collar",
        "right_elbow": "right_shoulder",
        "right_wrist": "right_elbow",
        "right_index1": "right_wrist",
        "right_index2": "right_index1",
        "right_index3": "right_index2",
        "right_middle1": "right_wrist",
        "right_middle2": "right_middle1",
        "right_middle3": "right_middle2",
        "right_pinky1": "right_wrist",
        "right_pinky2": "right_pinky1",
        "right_pinky3": "right_pinky2",
        "right_ring1": "right_wrist",
        "right_ring2": "right_ring1",
        "right_ring3": "right_ring2",
        "right_thumb1": "right_wrist",
        "right_thumb2": "right_thumb1",
        "right_thumb3": "right_thumb2",
    ]

    static let parentIndex: [Int] = {
        var byName: [String: Int] = [:]
        for (i, n) in jointOrder.enumerated() {
            byName[n] = i
        }
        return jointOrder.map { name in
            guard let parentName = parentByName[name],
                  let idx = byName[parentName] else {
                return -1
            }
            return idx
        }
    }()
}
