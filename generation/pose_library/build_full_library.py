"""Materialise per-gloss .npz clips from `data/best_takes.json`.

For each gloss in the best-takes manifest:
  1. Slice `data/mediapipe_landmarks/{sentence_id}.npy[start:end]` → (T,45,3).
  2. Symmetric per-joint INT8 quantise (same `quantize_int8` as
     `build_index.py` so iOS `PoseDatabase.decodeNPZ` reads it unchanged).
  3. Write `<lib>/clips/{GLOSS}.npz`  containing `clip_i8` + `scale`.
  4. Add an `index.json` entry stamping how the clip was sourced.

Optionally — `--with-rotations` — shells out to `retarget_to_target.py` with
a frame-range argument so we also emit `<lib>/rotations/{GLOSS}.rot.npz` for
the 3D avatar's quaternion path. Skipped by default because that step needs
the source BVH on disk and the retargeter is slower per-call.

Output goes to a **new** directory (`<lib>` = `generation/pose_library/full`
by default) so we don't clobber the existing demo clips. After building,
optionally mirror into `mobile-app/sema/sema/Resources/PoseLibraryFull/`
with `--mirror`.

That way the iOS app can load **both** sets: the curated demo set
(`PoseLibrary/`) and the full v11-derived set (`PoseLibraryFull/`), and
the user can A/B-test them on the avatar without losing the working demo.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


def quantize_int8(clip: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-joint per-coord symmetric int8 quantisation.

    Inlined from `build_index.py.quantize_int8` so this script doesn't
    inherit the pandas dep build_index brings in.

    Input  : (T, 45, 3) float32
    Output : (q (T, 45, 3) int8, scale (45, 3) float32)
    Reconstruction: clip ≈ q.astype(float32) / 127 * scale  — exactly what
    iOS `PoseDatabase.decodeNPZ` does.
    """
    assert clip.ndim == 3 and clip.shape[1:] == (45, 3), f"unexpected shape {clip.shape}"
    scale = np.maximum(np.abs(clip).max(axis=0), 1e-6).astype(np.float32)
    q = np.round(clip / scale * 127.0).clip(-127, 127).astype(np.int8)
    return q, scale

BEST_TAKES = REPO_ROOT / "data/best_takes.json"
LANDMARKS_DIR = REPO_ROOT / "data/mediapipe_landmarks"
ALIGNMENTS = REPO_ROOT / "data/alignments.json"
DEFAULT_LIB = REPO_ROOT / "generation/pose_library/full"
DEFAULT_BUNDLE_MIRROR = REPO_ROOT / "mobile-app/sema/sema/Resources/PoseLibraryFull"
DEFAULT_FPS = 24.0
# Bundle filename prefix for the full-library clips. Xcode 16's
# fileSystemSynchronizedGroup flattens every .npz into the bundle root, so a
# bare `BANK.npz` here would collide with `PoseLibrary/clips/BANK.npz` from
# the demo set ("Multiple commands produce …" build error). The prefix keeps
# the two libraries co-resident in the bundle without project-file edits.
# `index.json` paths are written with the prefix; PoseDatabase resolves clips
# via the path field, so iOS code needs no awareness of this convention.
DEFAULT_NAME_PREFIX = "_full__"


def slice_clip(landmarks_path: Path, start: int, end: int) -> np.ndarray:
    """Load and slice a sentence's mediapipe landmarks to [start, end).

    Returns float32 (T, 45, 3). Caller is responsible for INT8 quantising.
    """
    arr = np.load(landmarks_path)
    if arr.ndim != 3 or arr.shape[1:] != (45, 3):
        raise ValueError(f"{landmarks_path} has unexpected shape {arr.shape}")
    T = arr.shape[0]
    s = max(0, int(start))
    e = min(T, int(end))
    if e - s < 6:
        raise ValueError(f"slice [{s}:{e}] too short on T={T}")
    return arr[s:e].astype(np.float32, copy=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lib", type=Path, default=DEFAULT_LIB,
                    help="output library dir (default: generation/pose_library/full)")
    ap.add_argument("--mirror", action="store_true",
                    help=f"also copy clips/index.json into {DEFAULT_BUNDLE_MIRROR}")
    ap.add_argument("--mirror-dir", type=Path, default=DEFAULT_BUNDLE_MIRROR,
                    help="override the bundle mirror path")
    ap.add_argument("--max-takes", type=int, default=None,
                    help="cap number of clips built (debug)")
    ap.add_argument("--overwrite", action="store_true",
                    help="rebuild clips even if they already exist")
    ap.add_argument("--name-prefix", default=DEFAULT_NAME_PREFIX,
                    help=("clip filename prefix in the bundle root. Keeps "
                          "the full library from colliding with the demo "
                          "library when Xcode flattens .npz files. Pass "
                          "empty string only if you've added folder-"
                          "reference exceptions to the .xcodeproj."))
    args = ap.parse_args()

    if not BEST_TAKES.exists():
        print(f"ERROR: {BEST_TAKES} not found. Run pick_best_takes.py first.",
              file=sys.stderr)
        return 1
    takes = json.loads(BEST_TAKES.read_text())
    print(f"Loaded {len(takes)} best takes from {BEST_TAKES.name}")

    clips_dir = args.lib / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    items = sorted(takes.items())
    if args.max_takes:
        items = items[:args.max_takes]

    index: dict[str, dict] = {}
    skipped: list[tuple[str, str]] = []
    built = 0
    t0 = time.time()

    for gloss, t in items:
        file_stem = f"{args.name_prefix}{gloss}"
        out_npz = clips_dir / f"{file_stem}.npz"
        if out_npz.exists() and not args.overwrite:
            # Re-read existing clip to populate index entry.
            existing = np.load(out_npz)
            if "clip_i8" in existing.files:
                n_frames = int(existing["clip_i8"].shape[0])
            else:
                n_frames = -1
            index[gloss] = _entry(
                gloss, t, n_frames,
                source=f"alignment/sentence_{t['sentence_id']}",
                file_stem=file_stem,
            )
            continue

        sid = t["sentence_id"]
        start, end = t["start"], t["end"]
        lpath = LANDMARKS_DIR / f"{sid}.npy"
        if not lpath.exists():
            skipped.append((gloss, f"missing landmarks file {lpath.name}"))
            continue
        try:
            clip = slice_clip(lpath, start, end)
        except Exception as exc:
            skipped.append((gloss, f"slice error: {exc}"))
            continue

        q, scale = quantize_int8(clip)
        np.savez(out_npz, clip_i8=q, scale=scale)
        n_frames = int(clip.shape[0])
        index[gloss] = _entry(
            gloss, t, n_frames,
            source=f"alignment/sentence_{sid}",
            file_stem=file_stem,
        )
        built += 1

    print(f"\nBuilt {built} new clips, skipped {len(skipped)}.")
    if skipped:
        for g, reason in skipped[:10]:
            print(f"  skipped {g}: {reason}")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more")

    # Unique index filename so it doesn't collide with the demo set's
    # `index.json` when Xcode flattens the bundle (matches the rename done
    # for the clips). PoseDatabase looks for this name when bundleSubdir is
    # "PoseLibraryFull".
    index_path = args.lib / "index_full.json"
    index_path.write_text(json.dumps(
        {g: index[g] for g in sorted(index.keys())},
        indent=2,
    ))
    print(f"\nWrote {index_path} ({len(index)} entries)")
    print(f"Total wall time: {time.time()-t0:.1f}s")

    if args.mirror:
        mirror_clips = args.mirror_dir / "clips"
        args.mirror_dir.mkdir(parents=True, exist_ok=True)
        mirror_clips.mkdir(parents=True, exist_ok=True)
        # Copy clips
        for src in clips_dir.glob("*.npz"):
            shutil.copy2(src, mirror_clips / src.name)
        shutil.copy2(index_path, args.mirror_dir / index_path.name)
        print(f"\nMirrored to {args.mirror_dir}")
        print(f"  clips/: {sum(1 for _ in mirror_clips.glob('*.npz'))} files")
        print(f"  size:   {sum(p.stat().st_size for p in args.mirror_dir.rglob('*'))/1024/1024:.1f} MB")
    return 0


def _entry(gloss: str, take: dict, n_frames: int, source: str, file_stem: str) -> dict:
    return {
        "path": f"clips/{file_stem}.npz",
        "n_frames": n_frames,
        "fps": DEFAULT_FPS,
        "source_clip_id": int(take["sentence_id"]),
        "source_range": [int(take["start"]), int(take["end"])],
        "format": "int8",
        "source": source,
        "landmark_source": "mediapipe",
        "alignment_score": float(take.get("alignment_score", 0.0)),
        "alignment_quality": float(take.get("quality", 0.0)),
    }


if __name__ == "__main__":
    sys.exit(main())
