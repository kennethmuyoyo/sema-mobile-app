"""Project Motion-S BVH skeletons into MediaPipe-equivalent normalized landmarks.

All BVH files in Motion-S share an identical SMPL-X-style 55-joint rig, so the
hierarchy is parsed per-file but the joint-name layout is fixed. For each
clip we:

  1. Parse the BVH (HIERARCHY + MOTION) into joint definitions and a
     (T, n_channels) array of frame channels.
  2. Forward-kinematics → per-frame world positions of all 55 joints.
  3. Select 45 joints that overlap MediaPipe Holistic (15 body + 15 left-hand
     finger segments + 15 right-hand finger segments). The wrists used by the
     arms also serve as the hand-root in the MediaPipe sense.
  4. Normalize: shoulder-midpoint origin, shoulder-width unit scale, flip y
     so positive-y points downward (matches MediaPipe image-space convention).
  5. Write `(T, 45, 3)` float32 to /Users/ken/sema-mobile-app/data/landmarks/{id}.npy.

Writes a sibling `landmarks_meta.json` documenting the joint ordering and
normalization stats. This file is the single source of truth for the
landmark layout across this folder and `../generation/` and `../mobile-app/`.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# 45 target joints, in fixed order. Must match generation/renderer/ios_contract.md.
TARGET_JOINTS: list[str] = [
    # body (15)
    "head",
    "left_eye_smplhf", "right_eye_smplhf",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    # left hand (15): 5 fingers x 3 segments
    "left_index1", "left_index2", "left_index3",
    "left_middle1", "left_middle2", "left_middle3",
    "left_pinky1", "left_pinky2", "left_pinky3",
    "left_ring1", "left_ring2", "left_ring3",
    "left_thumb1", "left_thumb2", "left_thumb3",
    # right hand (15)
    "right_index1", "right_index2", "right_index3",
    "right_middle1", "right_middle2", "right_middle3",
    "right_pinky1", "right_pinky2", "right_pinky3",
    "right_ring1", "right_ring2", "right_ring3",
    "right_thumb1", "right_thumb2", "right_thumb3",
]
N_JOINTS = len(TARGET_JOINTS)         # 45
FEATURE_DIM = N_JOINTS * 3            # 135


def parse_bvh(path: Path) -> tuple[list[dict], np.ndarray, float]:
    """Return (joints, motion, frame_time).

    joints: list of {name, parent, offset (3,), channels [str], ch_offset int}
            in DFS order. End Sites are skipped (no channels, not a model joint).
    motion: float32 array of shape (T, total_channels).
    """
    with open(path) as f:
        lines = f.read().split("\n")

    joints: list[dict] = []
    stack: list[int] = []   # current open-brace stack: joint indices, -1 for End Site
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
            parent = stack[-1] if stack and stack[-1] >= 0 else -1
            joints.append({"name": name, "parent": parent, "offset": None, "channels": [], "ch_offset": 0})
            pending = len(joints) - 1
        elif head == "End":
            pending = -1
        elif head == "{":
            assert pending is not None, "{ without ROOT/JOINT/End Site"
            stack.append(pending)
            pending = None
        elif head == "}":
            stack.pop()
        elif head == "OFFSET":
            offset = np.array([float(x) for x in tok[1:4]], dtype=np.float64)
            if stack[-1] >= 0:
                joints[stack[-1]]["offset"] = offset
        elif head == "CHANNELS":
            n = int(tok[1])
            joints[stack[-1]]["channels"] = tok[2 : 2 + n]
        i += 1

    # Assign channel offsets in DFS / parse order
    cum = 0
    for j in joints:
        j["ch_offset"] = cum
        cum += len(j["channels"])
    total_channels = cum

    motion = np.array([[float(x) for x in ln.split()] for ln in motion_lines], dtype=np.float64)
    if motion.size and motion.shape[1] != total_channels:
        raise ValueError(
            f"{path}: motion has {motion.shape[1]} channels per frame, hierarchy expects {total_channels}"
        )
    return joints, motion, frame_time


def _rot_matrices(angles_deg: np.ndarray, order: list[str]) -> np.ndarray:
    """Compose rotation matrices from per-axis Euler angles in degrees.

    `angles_deg` has shape (..., 3) aligned to `order` (e.g. ['Xrotation','Yrotation','Zrotation']).
    Returns (..., 3, 3) where R = R_order[0] @ R_order[1] @ R_order[2].
    """
    a = np.deg2rad(angles_deg)
    Rs = []
    for k, axis in enumerate(order):
        ang = a[..., k]
        c, s = np.cos(ang), np.sin(ang)
        zero = np.zeros_like(ang)
        one = np.ones_like(ang)
        if axis.startswith("X"):
            R = np.stack([
                np.stack([one,  zero, zero], -1),
                np.stack([zero, c,    -s ], -1),
                np.stack([zero, s,     c ], -1),
            ], -2)
        elif axis.startswith("Y"):
            R = np.stack([
                np.stack([ c,   zero,  s], -1),
                np.stack([zero, one,  zero], -1),
                np.stack([-s,   zero,  c], -1),
            ], -2)
        else:  # Z
            R = np.stack([
                np.stack([c, -s,  zero], -1),
                np.stack([s,  c,  zero], -1),
                np.stack([zero, zero, one], -1),
            ], -2)
        Rs.append(R)
    out = Rs[0]
    for R in Rs[1:]:
        out = np.einsum("...ij,...jk->...ik", out, R)
    return out


def forward_kinematics(joints: list[dict], motion: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Return (positions (T, J, 3) world coords, joint_names ordered by `joints`)."""
    T = motion.shape[0]
    J = len(joints)
    pos = np.zeros((T, J, 3), dtype=np.float64)
    rot = np.zeros((T, J, 3, 3), dtype=np.float64)

    for j_idx, j in enumerate(joints):
        ch_off = j["ch_offset"]
        ch = j["channels"]
        # rotation channels (assumed last three, may be preceded by 3 position channels for root)
        rot_axes = [c for c in ch if c.endswith("rotation")]
        rot_idx = [ch.index(c) for c in rot_axes]
        rot_vals = motion[:, ch_off + np.array(rot_idx, dtype=np.int64)] if rot_idx else np.zeros((T, 0))
        if len(rot_axes) == 3:
            local_R = _rot_matrices(rot_vals, rot_axes)
        else:
            local_R = np.broadcast_to(np.eye(3), (T, 3, 3)).copy()

        # position channels (only the root pelvis has them)
        pos_axes = [c for c in ch if c.endswith("position")]
        if pos_axes:
            pos_idx = [ch.index(c) for c in pos_axes]
            p_vals = motion[:, ch_off + np.array(pos_idx, dtype=np.int64)]
            # Reorder to xyz
            order_map = {"Xposition": 0, "Yposition": 1, "Zposition": 2}
            xyz = np.zeros_like(p_vals)
            for k, c in enumerate(pos_axes):
                xyz[:, order_map[c]] = p_vals[:, k]
            root_translation = xyz
        else:
            root_translation = None

        parent = j["parent"]
        offset = j["offset"]
        if offset is None:
            offset = np.zeros(3)

        if parent < 0:
            # root: position = translation (override rest offset), rot = local
            pos[:, j_idx] = root_translation if root_translation is not None else offset[None, :]
            rot[:, j_idx] = local_R
        else:
            # child: world_pos = parent_pos + parent_rot @ offset; world_rot = parent_rot @ local_rot
            pos[:, j_idx] = pos[:, parent] + np.einsum("tij,j->ti", rot[:, parent], offset)
            rot[:, j_idx] = np.einsum("tij,tjk->tik", rot[:, parent], local_R)

    names = [j["name"] for j in joints]
    return pos.astype(np.float32), names


def normalize_landmarks(pos_45: np.ndarray, name_to_idx: dict[str, int]) -> np.ndarray:
    """Shoulder-mid origin, shoulder-width unit scale, y-flip for image convention."""
    ls = pos_45[:, name_to_idx["left_shoulder"]]
    rs = pos_45[:, name_to_idx["right_shoulder"]]
    mid = (ls + rs) / 2.0                                   # (T, 3)
    scale = np.linalg.norm(ls - rs, axis=-1, keepdims=True) # (T, 1)
    scale = np.maximum(scale, 1e-3)
    norm = (pos_45 - mid[:, None, :]) / scale[:, None, :]
    norm[..., 1] = -norm[..., 1]                            # y points down like MediaPipe
    return norm.astype(np.float32)


def process_one(bvh_path: Path, out_path: Path) -> dict:
    joints, motion, frame_time = parse_bvh(bvh_path)
    if motion.size == 0:
        raise ValueError(f"{bvh_path}: empty motion")
    pos, names = forward_kinematics(joints, motion)
    name_to_idx = {n: i for i, n in enumerate(names)}
    sel = np.array([name_to_idx[n] for n in TARGET_JOINTS], dtype=np.int64)
    pos_45 = pos[:, sel, :]
    norm = normalize_landmarks(pos_45, {n: i for i, n in enumerate(TARGET_JOINTS)})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, norm)
    return {"frames": int(norm.shape[0]), "frame_time": frame_time}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bvh-root", default="/Users/ken/sema-mobile-app/data/Train")
    ap.add_argument("--out-root", default="/Users/ken/sema-mobile-app/data/landmarks")
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit; otherwise process first N clips")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    bvh_files = sorted(Path(args.bvh_root).glob("*/*.bvh"))
    if args.limit:
        bvh_files = bvh_files[: args.limit]

    n_ok = 0
    n_skip = 0
    n_err = 0
    for bvh in tqdm(bvh_files, desc="bvh→landmarks"):
        clip_id = bvh.stem
        out_path = out_root / f"{clip_id}.npy"
        if out_path.exists() and not args.overwrite:
            n_skip += 1
            continue
        try:
            process_one(bvh, out_path)
            n_ok += 1
        except Exception as e:
            n_err += 1
            print(f"  ERR {bvh}: {e}", file=sys.stderr)

    meta = {
        "joint_order": TARGET_JOINTS,
        "n_joints": N_JOINTS,
        "feature_dim": FEATURE_DIM,
        "coord_frame": "shoulder-mid origin, shoulder-width unit scale, y-down (MediaPipe convention)",
        "source": "Motion-S BVH (SMPL-X rig) projected via forward kinematics",
        "fps_source": "from BVH 'Frame Time' (typically 24 Hz)",
    }
    (repo / "data" / "landmarks_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"ok={n_ok}  skipped={n_skip}  errors={n_err}  total={len(bvh_files)}")
    print(f"out: {out_root}")
    print(f"meta: {repo/'data'/'landmarks_meta.json'}")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
