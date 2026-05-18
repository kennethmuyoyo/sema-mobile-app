#!/usr/bin/env python3
"""Render BVH motion-capture clips through SMPL-X + MediaPipe to landmark .npy.

Self-contained bundle entry point. Walks `./train/<id>/<id>.bvh`, runs each
through a head-on phone-camera-style SMPL-X render then MediaPipe Pose +
Hand Holistic, and writes (T, 45, 3) float32 to `./output/<id>.npy`.

Resumable — clips whose `<id>.npy` already exists are skipped, so you can
Ctrl-C and re-run anytime.

Usage:
    python render.py                       # process every clip sequentially
    python render.py --shard 0/3           # process every 3rd clip, starting at 0
    python render.py --limit 10            # smoke test on first 10 clips
    python render.py --overwrite           # re-render existing outputs

To run multiple terminals on the same machine in parallel (highly
recommended on M-series Macs):
    Terminal A:  python render.py --shard 0/2
    Terminal B:  python render.py --shard 1/2

Each shard processes a disjoint subset of clips so they never collide.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from _renderer import (
    bvh_to_smplx_pose,
    render_clip,
    extract_landmarks_from_video,
)


def bvh_to_landmarks_mediapipe(bvh: Path, smplx_model: Path) -> np.ndarray:
    """Render → extract. Returns (T, 45, 3) MediaPipe-distribution landmarks."""
    pose = bvh_to_smplx_pose(bvh)  # zeros translation by default
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--train", type=Path, default=ROOT / "train",
                    help="Directory of <id>/<id>.bvh folders. Default: ./train")
    ap.add_argument("--output", type=Path, default=ROOT / "output",
                    help="Where to write <id>.npy. Default: ./output")
    ap.add_argument("--smplx-model", type=Path,
                    default=ROOT / "models" / "smplx" / "SMPLX_NEUTRAL.npz",
                    help="Default: ./models/smplx/SMPLX_NEUTRAL.npz")
    ap.add_argument("--shard", type=str, default=None,
                    help="Shard like 0/3 to process every 3rd clip starting at 0.")
    ap.add_argument("--limit", type=int, default=0,
                    help="0 = all clips; otherwise process the first N.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-render clips whose output already exists.")
    args = ap.parse_args()

    if not args.smplx_model.exists():
        print(f"[render] SMPL-X model not found at {args.smplx_model}", file=sys.stderr)
        print(f"[render] Place SMPLX_NEUTRAL.npz at models/smplx/SMPLX_NEUTRAL.npz", file=sys.stderr)
        print(f"[render] Register at https://smpl-x.is.tue.mpg.de (free) to download.", file=sys.stderr)
        return 1

    args.output.mkdir(parents=True, exist_ok=True)

    bvhs = sorted(args.train.glob("*/*.bvh"))
    if not bvhs:
        print(f"[render] No BVHs found under {args.train}/*/*.bvh", file=sys.stderr)
        return 1

    label = f"{len(bvhs)} total"
    if args.shard:
        rank, world = (int(x) for x in args.shard.split("/"))
        if not (0 <= rank < world):
            print(f"[render] --shard {rank}/{world}: rank must be 0..{world-1}", file=sys.stderr)
            return 1
        bvhs = bvhs[rank::world]
        label = f"shard {rank}/{world} ({len(bvhs)} of total)"
    if args.limit > 0:
        bvhs = bvhs[: args.limit]
        label += f", limited to {len(bvhs)}"

    print(f"[render] {label}")
    print(f"[render] writing to {args.output}")

    ok = 0
    skipped = 0
    errors = 0
    for i, bvh in enumerate(bvhs, start=1):
        out_path = args.output / f"{bvh.stem}.npy"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            print(f"[{i}/{len(bvhs)}] {bvh.parent.name}/{bvh.name} → {out_path.name}")
            landmarks = bvh_to_landmarks_mediapipe(bvh, args.smplx_model)
            np.save(out_path, landmarks)
            ok += 1
        except Exception as exc:                                  # noqa: BLE001
            errors += 1
            print(f"  ERROR {bvh}: {exc}", file=sys.stderr)

    print(f"[render] done ok={ok} skipped={skipped} errors={errors} "
          f"total_attempted={len(bvhs)} out={args.output}")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
