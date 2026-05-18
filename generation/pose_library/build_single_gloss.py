"""Convert single-gloss BVH clips into the on-device pose library.

This is the **canonical** path for tokens we've curated — `data/recognition_set/`
and `data/generation_set/` each contain one BVH per gloss (e.g.
`hello.bvh` → token `HELLO`). For every such BVH this script:

  1. Runs the existing `recognition.data.bvh_to_landmarks` FK + normalisation
     to produce `(T, 45, 3)` shoulder-normalised landmarks. Stored int8-
     quantised in `clips/{TOKEN}.npz` — exact same on-disk format as
     `generation/pose_library/build_index.py` so the iOS
     `PoseDatabase.decodeNPZ` reads it without code changes.

  2. Shells out to `retarget_to_target.py --bvh ... --out rotations/{TOKEN}.rot.npz`
     to produce parent-local quaternions on the target rig. iOS picks these
     up via `PoseDatabase.decodeRotationSidecar` and prefers them over the
     position-IK fallback in `SimpleAvatar3DView`.

  3. Merges the per-token entry into `index.json` (preserving any equal-
     sliced entries from `build_index.py` for tokens not in the new sets).

  4. Mirrors `clips/`, `rotations/`, and `index.json` into
     `mobile-app/sema/sema/Resources/PoseLibrary/` so Xcode 16's
     synchronised folder picks them up at build time.

Usage:
    python generation/pose_library/build_single_gloss.py
        [--sources data/recognition_set data/generation_set]
        [--lib generation/pose_library]
        [--bundle mobile-app/sema/sema/Resources/PoseLibrary]
        [--overwrite]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Reuse the existing converters — DO NOT reimplement the FK or retargeting math.
from recognition.data.bvh_to_landmarks import (   # noqa: E402
    TARGET_JOINTS,
    forward_kinematics,
    normalize_landmarks,
    parse_bvh,
)
from generation.pose_library.build_index import quantize_int8   # noqa: E402

DEFAULT_SOURCES = [
    REPO_ROOT / "data" / "recognition_set",
    REPO_ROOT / "data" / "generation_set",
]
RETARGET_SCRIPT = REPO_ROOT / "generation" / "pose_library" / "retarget_to_target.py"
TARGET_BIND_POSE = REPO_ROOT / "generation" / "pose_library" / "target_bind_pose.json"
DEFAULT_SMPLX_MODEL = REPO_ROOT / "recognition" / ".cache" / "smplx" / "SMPLX_NEUTRAL.npz"


def token_for(bvh: Path) -> str:
    """Filename stem → uppercase gloss token (matches index.json conventions)."""
    return bvh.stem.upper()


def bvh_to_landmarks_fk(bvh: Path) -> np.ndarray:
    """(T, 45, 3) float32, shoulder-normalised. Reuses existing FK converter.

    Geometry-perfect (it knows the rig's actual joint positions), but the
    distribution doesn't match MediaPipe — MediaPipe estimates joint
    positions from the rendered image surface, which is systematically
    different. Use this when you only care about the avatar playback path.
    """
    joints, motion, _frame_time = parse_bvh(bvh)
    if motion.size == 0:
        raise ValueError(f"{bvh}: empty motion")
    pos, names = forward_kinematics(joints, motion)
    name_to_idx = {n: i for i, n in enumerate(names)}
    sel = np.array([name_to_idx[n] for n in TARGET_JOINTS], dtype=np.int64)
    pos_45 = pos[:, sel, :]
    norm = normalize_landmarks(pos_45, {n: i for i, n in enumerate(TARGET_JOINTS)})
    return norm.astype(np.float32)


def bvh_to_landmarks_mediapipe(bvh: Path, smplx_model: Path) -> np.ndarray:
    """(T, 45, 3) float32 — runs the BVH through the SMPL-X renderer + MediaPipe
    Holistic so the output distribution matches what the iOS HolisticLandmarker
    produces from live camera input. This is the right source for the
    PoseTemplateMatcher templates — same joint estimator on both sides closes
    the synthetic-vs-real domain gap.
    """
    # Imported lazily because pyrender / mediapipe / smplx are heavy and only
    # needed when --landmark-source=mediapipe is selected.
    from recognition.data.render_bvh_to_mediapipe import (   # noqa: PLC0415
        bvh_to_smplx_pose,
        render_clip,
        extract_landmarks_from_video,
    )
    import tempfile

    # `smplx.create(model_folder, model_type="smplx")` resolves the model
    # by appending `<model_type>/SMPLX_NEUTRAL.npz` to the folder we pass.
    # `render_clip` passes `smplx_model.parent` as the folder, so the actual
    # file smplx loads is `smplx_model.parent / "smplx" / "SMPLX_NEUTRAL.npz"`.
    # The CLI argument is treated as a *hint to the parent folder* — the
    # filename itself can be anything; only the parent matters. We probe the
    # real lookup path so the error message points at the right file.
    resolved_npz = smplx_model.parent / "smplx" / "SMPLX_NEUTRAL.npz"
    if not resolved_npz.exists() and not smplx_model.exists():
        raise FileNotFoundError(
            f"SMPL-X model not found at {resolved_npz}. "
            "Download SMPLX_NEUTRAL.npz from https://smpl-x.is.tue.mpg.de/ "
            f"(register, accept license) and place it at that path."
        )

    pose = bvh_to_smplx_pose(bvh)
    # Single-gloss BVHs come from a larger capture scene; their pelvis
    # translation is a world offset from that scene (e.g. x = -2.12 m) that
    # puts the SMPL-X body outside the render camera's frustum and produces
    # an empty frame — MediaPipe then detects nothing. Sign information
    # lives in the joint rotations, not the pelvis translation, so we can
    # safely zero translation and put the body at the origin where the
    # camera is pointed.
    pose["transl"] = np.zeros_like(pose["transl"])
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        video_path = Path(tmp.name)
    try:
        render_clip(pose, smplx_model, video_path)
        landmarks = extract_landmarks_from_video(video_path)
    finally:
        try:
            video_path.unlink()
        except FileNotFoundError:
            pass
    return landmarks.astype(np.float32)


def write_clip_npz(clip: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    q, scale = quantize_int8(clip)
    np.savez(out_path, clip_i8=q, scale=scale)


def write_rotation_sidecar(bvh: Path, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(RETARGET_SCRIPT),
        "--bvh", str(bvh),
        "--out", str(out_path),
        "--bind-pose", str(TARGET_BIND_POSE),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"retarget_to_target.py failed for {bvh.name}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )


def mirror_to_bundle(src_lib: Path, bundle: Path, sub: str, filename: str) -> None:
    """Copy one file from generation/pose_library/{sub}/ to the iOS bundle."""
    src = src_lib / sub / filename
    dst = bundle / sub / filename
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sources",
        nargs="+",
        type=Path,
        default=DEFAULT_SOURCES,
        help="BVH source directories. Each *.bvh stem becomes the gloss token (uppercased).",
    )
    ap.add_argument(
        "--lib",
        type=Path,
        default=REPO_ROOT / "generation" / "pose_library",
        help="Pose-library root. clips/, rotations/, and index.json live here.",
    )
    ap.add_argument(
        "--bundle",
        type=Path,
        default=REPO_ROOT / "mobile-app" / "sema" / "sema" / "Resources" / "PoseLibrary",
        help="iOS bundle mirror. Pass '' to skip mirroring.",
    )
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument(
        "--landmark-source",
        choices=["fk", "mediapipe"],
        default="fk",
        help=(
            "How to build the landmark clip. 'fk' = direct FK on the BVH "
            "(fast, no extra deps); 'mediapipe' = SMPL-X render → MediaPipe "
            "Holistic, so the clip distribution matches live camera input "
            "and the PoseTemplateMatcher can compare apples to apples."
        ),
    )
    ap.add_argument(
        "--smplx-model",
        type=Path,
        default=DEFAULT_SMPLX_MODEL,
        help="SMPLX_NEUTRAL.npz path. Only consulted when --landmark-source=mediapipe.",
    )
    ap.add_argument(
        "--prune",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After merging single-gloss entries, drop every other token from "
            "index.json and delete its clip + rotation files in both the "
            "staging library and the iOS bundle. Default ON — the bundle ends "
            "up with only the curated single-gloss tokens (no equal-sliced "
            "Motion-S leftovers). Pass --no-prune to keep them."
        ),
    )
    args = ap.parse_args()

    index_path = args.lib / "index.json"
    index: dict[str, dict] = {}
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text())
        except json.JSONDecodeError as exc:
            print(f"[single-gloss] WARN: existing index.json malformed ({exc}); starting fresh")

    bvh_files: list[Path] = []
    for src in args.sources:
        if not src.is_dir():
            print(f"[single-gloss] WARN: source not found: {src}", file=sys.stderr)
            continue
        bvh_files.extend(sorted(src.glob("*.bvh")))

    if not bvh_files:
        print("[single-gloss] no BVH files found", file=sys.stderr)
        return 1

    ok = 0
    skipped = 0
    errors = 0

    for bvh in bvh_files:
        token = token_for(bvh)
        rel_clip = f"clips/{token}.npz"
        rel_rot = f"rotations/{token}.rot.npz"
        clip_out = args.lib / rel_clip
        rot_out = args.lib / rel_rot

        already_built = clip_out.exists() and rot_out.exists() and token in index
        if already_built and not args.overwrite:
            skipped += 1
            continue

        try:
            print(f"[single-gloss] {bvh.parent.name}/{bvh.name} → {token}  "
                  f"(landmarks: {args.landmark_source})")
            if args.landmark_source == "mediapipe":
                clip = bvh_to_landmarks_mediapipe(bvh, args.smplx_model)
            else:
                clip = bvh_to_landmarks_fk(bvh)
            write_clip_npz(clip, clip_out)
            write_rotation_sidecar(bvh, rot_out)

            n_frames = int(clip.shape[0])
            index[token] = {
                "path": rel_clip,
                "n_frames": n_frames,
                "fps": float(args.fps),
                # iOS PoseDatabase.Entry requires both — synthesise plausible
                # values for single-clip BVHs (no source slice to reference).
                "source_clip_id": -1,
                "source_range": [0, n_frames],
                "format": "int8",
                "source": f"single_gloss/{bvh.parent.name}",
                "landmark_source": args.landmark_source,
            }

            if str(args.bundle):
                mirror_to_bundle(args.lib, args.bundle, "clips", f"{token}.npz")
                mirror_to_bundle(args.lib, args.bundle, "rotations", f"{token}.rot.npz")

            ok += 1
        except Exception as exc:
            errors += 1
            print(f"[single-gloss] ERROR token={token} bvh={bvh}: {exc}", file=sys.stderr)

    pruned = 0
    orphans_removed = 0
    if args.prune:
        # Step 1: drop non-single_gloss index entries (+ their files).
        keep = {
            tok for tok, entry in index.items()
            if str(entry.get("source", "")).startswith("single_gloss/")
        }
        drop = [tok for tok in index.keys() if tok not in keep]
        for tok in drop:
            entry = index.pop(tok)
            clip_rel = str(entry.get("path", ""))
            for root in (args.lib, args.bundle if str(args.bundle) else None):
                if root is None:
                    continue
                if clip_rel:
                    _safe_delete(root / clip_rel)
                _safe_delete(root / "rotations" / f"{tok}.rot.npz")
            pruned += 1

        # Step 2: sweep orphan files on disk that no longer correspond to any
        # index entry. Earlier batch runs of retarget_to_target.py over the
        # full PoseLibrary index dropped rotations/{TOKEN}.rot.npz for ~3.8k
        # tokens; those tokens are gone but their files linger.
        kept_clip_paths = {entry.get("path", "") for entry in index.values()}
        kept_rotation_names = {f"{tok}.rot.npz" for tok in index.keys()}
        for root in (args.lib, args.bundle if str(args.bundle) else None):
            if root is None:
                continue
            clips_dir = root / "clips"
            if clips_dir.is_dir():
                for p in clips_dir.iterdir():
                    rel = f"clips/{p.name}"
                    if rel not in kept_clip_paths:
                        _safe_delete(p)
                        orphans_removed += 1
            rotations_dir = root / "rotations"
            if rotations_dir.is_dir():
                for p in rotations_dir.iterdir():
                    if p.name not in kept_rotation_names:
                        _safe_delete(p)
                        orphans_removed += 1

    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False))
    if str(args.bundle):
        (args.bundle / "index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False))

    print(
        f"[single-gloss] done ok={ok} skipped={skipped} errors={errors} "
        f"pruned={pruned} orphans_removed={orphans_removed} "
        f"total_entries={len(index)} index={index_path}"
    )
    return 0 if errors == 0 else 1


def _safe_delete(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except Exception as exc:                                      # noqa: BLE001
        print(f"[single-gloss] WARN: couldn't delete {path}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
