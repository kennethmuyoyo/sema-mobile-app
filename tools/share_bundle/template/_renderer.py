"""SMPL-X render + MediaPipe Holistic extraction (self-contained bundle copy).

Mirrors the production `recognition/data/render_bvh_to_mediapipe.py` but with
file-path lookups anchored to this bundle's directory (no repo-relative
traversal). All public functions exposed for `render.py`:

  - bvh_to_smplx_pose(bvh_path, zero_translation=True)
  - render_clip(pose, smplx_model_path, video_out)
  - extract_landmarks_from_video(video_path)

Heavy imports (torch / smplx / pyrender / mediapipe) are deferred to the
call sites so just importing this module stays cheap.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent

# Render config — matches iOS phone-camera pin so the SMPL-X render looks
# like what the live MediaPipe pipeline sees.
RENDER_W, RENDER_H = 480, 640
RENDER_FPS = 24
PHONE_FOV_Y_DEG = 65.0
CAMERA_DISTANCE_M = 1.4
SUBJECT_BVH_TO_M = 0.01

# 45-joint TARGET layout — must match Landmark45 in the iOS app.
TARGET_JOINTS: list[str] = [
    "head",
    "left_eye_smplhf", "right_eye_smplhf",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_index1", "left_index2", "left_index3",
    "left_middle1", "left_middle2", "left_middle3",
    "left_pinky1", "left_pinky2", "left_pinky3",
    "left_ring1", "left_ring2", "left_ring3",
    "left_thumb1", "left_thumb2", "left_thumb3",
    "right_index1", "right_index2", "right_index3",
    "right_middle1", "right_middle2", "right_middle3",
    "right_pinky1", "right_pinky2", "right_pinky3",
    "right_ring1", "right_ring2", "right_ring3",
    "right_thumb1", "right_thumb2", "right_thumb3",
]

# MediaPipe joint indices in pose (33-landmark) / hand (21-landmark) outputs.
MEDIAPIPE_BODY_IDX = {
    "head": 0, "left_eye_smplhf": 2, "right_eye_smplhf": 5,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
}
MEDIAPIPE_HAND_IDX = {
    "thumb1": 1, "thumb2": 2, "thumb3": 3,
    "index1": 5, "index2": 6, "index3": 7,
    "middle1": 9, "middle2": 10, "middle3": 11,
    "ring1": 13, "ring2": 14, "ring3": 15,
    "pinky1": 17, "pinky2": 18, "pinky3": 19,
}

# SMPL-X body-pose joint order (21 axis-angle joints, no hands/face).
SMPLX_BODY_JOINTS = [
    "left_hip", "right_hip", "spine1",
    "left_knee", "right_knee", "spine2",
    "left_ankle", "right_ankle", "spine3",
    "left_foot", "right_foot", "neck",
    "left_collar", "right_collar", "head",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
]
SMPLX_LEFT_HAND_JOINTS = [
    f"left_{f}{n}" for f in ("index", "middle", "pinky", "ring", "thumb") for n in (1, 2, 3)
]
SMPLX_RIGHT_HAND_JOINTS = [
    f"right_{f}{n}" for f in ("index", "middle", "pinky", "ring", "thumb") for n in (1, 2, 3)
]


# ---------------------------------------------------------------------------
# Tiny BVH parser — same algorithm as recognition/data/bvh_to_landmarks.py.
# ---------------------------------------------------------------------------

def parse_bvh(path: Path) -> tuple[list[dict], np.ndarray, float]:
    with open(path) as f:
        lines = f.read().split("\n")

    joints: list[dict] = []
    stack: list[int] = []
    pending: int | None = None
    in_motion = False
    motion_lines: list[str] = []
    frame_time = 0.0

    i = 0
    while i < len(lines):
        ln = lines[i].strip()
        if not ln:
            i += 1
            continue
        tok = ln.split()
        head = tok[0]

        if in_motion:
            if head.lower() == "frames:":
                pass
            elif head.lower() == "frame":
                frame_time = float(tok[2])
            elif tok[0].replace("-", "").replace(".", "").replace("e", "").replace("+", "").isdigit() or (
                tok[0].startswith("-") or tok[0][0].isdigit()
            ):
                motion_lines.append(ln)
            i += 1
            continue

        if head == "MOTION":
            in_motion = True
        elif head in ("ROOT", "JOINT"):
            name = tok[1]
            joints.append({
                "name": name,
                "parent": stack[-1] if stack else -1,
                "channels": [],
                "offset": None,
                "ch_offset": 0,
            })
            pending = len(joints) - 1
        elif head == "End":
            pending = None
        elif head == "{":
            if pending is not None:
                stack.append(pending)
            else:
                stack.append(-1)
        elif head == "}":
            stack.pop()
            pending = None
        elif head == "OFFSET" and pending is not None:
            joints[pending]["offset"] = np.array([float(x) for x in tok[1:4]], dtype=np.float64)
        elif head == "CHANNELS" and pending is not None:
            joints[pending]["channels"] = tok[2:]
        i += 1

    # Assign ch_offset
    off = 0
    for j in joints:
        j["ch_offset"] = off
        off += len(j["channels"])

    n_chan = off
    motion = np.array(
        [[float(x) for x in ln.split()] for ln in motion_lines],
        dtype=np.float32,
    ).reshape(-1, n_chan) if motion_lines else np.zeros((0, n_chan), dtype=np.float32)

    return joints, motion, frame_time


# ---------------------------------------------------------------------------
# BVH → SMPL-X pose
# ---------------------------------------------------------------------------

def bvh_to_smplx_pose(bvh_path: Path, zero_translation: bool = True) -> dict[str, Any]:
    """Convert BVH motion to SMPL-X axis-angle pose parameters.

    `zero_translation=True` (default) drops the pelvis world translation, which
    otherwise puts the body outside the render camera's frustum for clips
    authored inside a larger capture scene. Sign content lives in joint
    rotations, not pelvis translation, so this is safe for our use.
    """
    from scipy.spatial.transform import Rotation as R   # noqa: WPS433

    joints, motion, frame_time = parse_bvh(bvh_path)
    if motion.shape[0] == 0:
        raise RuntimeError(f"{bvh_path}: BVH has no motion frames")

    name_to_idx = {j["name"]: i for i, j in enumerate(joints)}
    T = motion.shape[0]

    def euler_for(name: str) -> np.ndarray:
        if name not in name_to_idx:
            return np.zeros((T, 3), dtype=np.float32)
        j = joints[name_to_idx[name]]
        ch_off = j["ch_offset"]
        rot_axes = [c for c in j["channels"] if c.endswith("rotation")]
        if not rot_axes:
            return np.zeros((T, 3), dtype=np.float32)
        rot_idx = np.array([j["channels"].index(c) for c in rot_axes], dtype=np.int64)
        deg = motion[:, ch_off + rot_idx]
        order = "".join(c[0] for c in rot_axes)
        rot = R.from_euler(order, deg, degrees=True)
        return rot.as_rotvec().astype(np.float32)

    pelvis = joints[name_to_idx["pelvis"]]
    pelvis_pos_axes = [c for c in pelvis["channels"] if c.endswith("position")]
    if pelvis_pos_axes:
        pelvis_pos_idx = np.array(
            [pelvis["channels"].index(c) for c in pelvis_pos_axes], dtype=np.int64
        )
        raw_transl = motion[:, pelvis["ch_offset"] + pelvis_pos_idx]
        transl_m = raw_transl.astype(np.float32) * SUBJECT_BVH_TO_M
    else:
        transl_m = np.zeros((T, 3), dtype=np.float32)

    if zero_translation:
        transl_m = np.zeros_like(transl_m)

    global_orient = euler_for("pelvis")
    body_pose = np.stack([euler_for(n) for n in SMPLX_BODY_JOINTS], axis=1)
    lh = np.stack([euler_for(n) for n in SMPLX_LEFT_HAND_JOINTS], axis=1)
    rh = np.stack([euler_for(n) for n in SMPLX_RIGHT_HAND_JOINTS], axis=1)

    return {
        "T": T,
        "frame_time": frame_time,
        "global_orient": global_orient,
        "body_pose": body_pose.reshape(T, -1),
        "left_hand_pose": lh.reshape(T, -1),
        "right_hand_pose": rh.reshape(T, -1),
        "transl": transl_m,
    }


# ---------------------------------------------------------------------------
# SMPL-X forward + pyrender + MP4 encode
# ---------------------------------------------------------------------------

def render_clip(pose: dict, smplx_model_path: Path, video_out: Path) -> None:
    import torch                                              # noqa: WPS433
    import smplx                                              # noqa: WPS433
    import pyrender                                           # noqa: WPS433
    import trimesh                                            # noqa: WPS433
    import cv2                                                # noqa: WPS433

    # smplx.create looks for `<model_folder>/<model_type>/SMPLX_NEUTRAL.npz`.
    # Our bundle layout is `<bundle>/models/smplx/SMPLX_NEUTRAL.npz` — so
    # smplx_model_path.parent.parent is the right model_folder.
    folder = smplx_model_path.parent.parent if smplx_model_path.parent.name == "smplx" \
        else smplx_model_path.parent
    model = smplx.create(
        str(folder),
        model_type="smplx",
        gender="neutral",
        use_pca=False,
        flat_hand_mean=True,
        batch_size=pose["T"],
        ext="npz",
    ).eval()

    with torch.no_grad():
        out = model(
            global_orient=torch.from_numpy(pose["global_orient"]),
            body_pose=torch.from_numpy(pose["body_pose"]),
            left_hand_pose=torch.from_numpy(pose["left_hand_pose"]),
            right_hand_pose=torch.from_numpy(pose["right_hand_pose"]),
            transl=torch.from_numpy(pose["transl"]),
            return_verts=True,
        )
    vertices_all = out.vertices.cpu().numpy()
    faces = model.faces

    yfov = np.radians(PHONE_FOV_Y_DEG)
    aspect = RENDER_W / RENDER_H
    cam = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=aspect)
    cam_pose = np.eye(4)
    cam_pose[:3, 3] = [0.0, 0.55, CAMERA_DISTANCE_M]

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=4.0)
    light_pose = np.eye(4); light_pose[:3, 3] = [1.0, 2.0, 2.0]

    renderer = pyrender.OffscreenRenderer(viewport_width=RENDER_W, viewport_height=RENDER_H)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_out), fourcc, RENDER_FPS, (RENDER_W, RENDER_H))

    try:
        for t in range(pose["T"]):
            mesh = trimesh.Trimesh(vertices=vertices_all[t], faces=faces, process=False)
            scene = pyrender.Scene(bg_color=[0.15, 0.18, 0.22, 1.0], ambient_light=[0.3, 0.3, 0.3])
            scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=True))
            scene.add(cam, pose=cam_pose)
            scene.add(light, pose=light_pose)
            color, _ = renderer.render(scene)
            writer.write(color[:, :, ::-1])   # RGB→BGR
    finally:
        renderer.delete()
        writer.release()


# ---------------------------------------------------------------------------
# MediaPipe extraction (Tasks API — mediapipe ≥ 0.10.20)
# ---------------------------------------------------------------------------

def extract_landmarks_from_video(video_path: Path) -> np.ndarray:
    import cv2                                                # noqa: WPS433
    import mediapipe as mp                                    # noqa: WPS433
    from mediapipe.tasks import python as mp_tasks            # noqa: WPS433
    from mediapipe.tasks.python import vision                 # noqa: WPS433

    pose_task_path = ROOT / "models" / "pose_landmarker_full.task"
    hand_task_path = ROOT / "models" / "hand_landmarker.task"
    if not pose_task_path.exists():
        raise FileNotFoundError(f"pose_landmarker_full.task not found at {pose_task_path}")
    if not hand_task_path.exists():
        raise FileNotFoundError(f"hand_landmarker.task not found at {hand_task_path}")

    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    cap.release()

    pose_options = vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(pose_task_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    hand_options = vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(hand_task_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    out = np.zeros((len(frames), 45, 3), dtype=np.float32)
    name_to_slot = {n: i for i, n in enumerate(TARGET_JOINTS)}
    ms_per_frame = 1000.0 / RENDER_FPS

    with vision.PoseLandmarker.create_from_options(pose_options) as pose_landmarker, \
         vision.HandLandmarker.create_from_options(hand_options) as hand_landmarker:
        for t, frame in enumerate(frames):
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            ts_ms = int(t * ms_per_frame)
            pose_result = pose_landmarker.detect_for_video(mp_image, ts_ms)
            hand_result = hand_landmarker.detect_for_video(mp_image, ts_ms)

            pose_lms = pose_result.pose_landmarks
            if not pose_lms:
                continue
            pose = pose_lms[0]

            body_xyz: dict[str, np.ndarray] = {}
            for jname, mp_idx in MEDIAPIPE_BODY_IDX.items():
                lm = pose[mp_idx]
                body_xyz[jname] = np.array([lm.x, lm.y, lm.z], dtype=np.float32)

            ls = body_xyz["left_shoulder"]
            rs = body_xyz["right_shoulder"]
            origin = (ls + rs) * 0.5
            scale = max(float(np.linalg.norm(ls - rs)), 1e-3)

            def norm(p: np.ndarray) -> np.ndarray:
                return (p - origin) / scale

            for jname, xyz in body_xyz.items():
                out[t, name_to_slot[jname]] = norm(xyz)

            for hand_idx, hand_landmarks in enumerate(hand_result.hand_landmarks):
                if hand_idx >= len(hand_result.handedness):
                    continue
                categories = hand_result.handedness[hand_idx]
                if not categories:
                    continue
                label = categories[0].category_name.lower()
                side_prefix = "left" if "left" in label else "right" if "right" in label else None
                if side_prefix is None:
                    continue
                for seg, mp_idx in MEDIAPIPE_HAND_IDX.items():
                    lm = hand_landmarks[mp_idx]
                    xyz = np.array([lm.x, lm.y, lm.z], dtype=np.float32)
                    out[t, name_to_slot[f"{side_prefix}_{seg}"]] = norm(xyz)
    return out
