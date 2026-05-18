"""Package a self-contained MediaPipe-render bundle for a collaborator.

Each bundle contains:
  - render.py + _renderer.py (portable, no repo-relative imports)
  - setup.sh + requirements.txt + README.md
  - models/{pose,hand}_landmarker.task (reused from the iOS bundle)
  - models/smplx/SMPLX_NEUTRAL.npz       (optional — see --include-smplx)
  - train/<id>/{*.bvh, metadata.txt}     (one shard of data/Train)
  - output/                              (created empty, .gitkeep)

Run it 3 times (once per helper) with different `--shard`:
    python tools/make_share_bundle.py --out ~/Desktop/folder1 --shard 0/3 \\
        --folder-name folder1
    python tools/make_share_bundle.py --out ~/Desktop/folder2 --shard 1/3 \\
        --folder-name folder2
    python tools/make_share_bundle.py --out ~/Desktop/folder3 --shard 2/3 \\
        --folder-name folder3

Each `--out` folder is then ready to zip and ship.

After all helpers finish and return their `output/` folders, you can merge
them all back into `data/mediapipe_landmarks/` by simply unzipping each into
that directory (no name collisions because the shards are disjoint).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TEMPLATE = REPO / "tools" / "share_bundle" / "template"

DEFAULT_TRAIN = REPO / "data" / "Train"
DEFAULT_TASK_DIR = REPO / "mobile-app" / "sema" / "sema" / "Resources"
DEFAULT_SMPLX = REPO / "recognition" / ".cache" / "smplx" / "smplx" / "SMPLX_NEUTRAL.npz"


def parse_shard(spec: str) -> tuple[int, int]:
    try:
        rank_s, world_s = spec.split("/")
        rank = int(rank_s)
        world = int(world_s)
    except Exception as exc:
        raise SystemExit(f"--shard must look like 0/3, got {spec!r}: {exc}")
    if world <= 0 or not (0 <= rank < world):
        raise SystemExit(f"--shard {rank}/{world}: rank must be 0..{world-1}, world > 0")
    return rank, world


def copy_template(out: Path) -> None:
    """Copy the four template files into the bundle root."""
    for name in ("render.py", "_renderer.py", "setup.sh", "requirements.txt", "README.md"):
        src = TEMPLATE / name
        dst = out / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    # setup.sh has to be executable.
    (out / "setup.sh").chmod(0o755)


def copy_models(out: Path, task_dir: Path, smplx_npz: Path | None) -> None:
    models = out / "models"
    models.mkdir(parents=True, exist_ok=True)
    for name in ("pose_landmarker_full.task", "hand_landmarker.task"):
        src = task_dir / name
        if not src.exists():
            raise SystemExit(f"missing {src} — check --task-dir")
        shutil.copy2(src, models / name)
    if smplx_npz is not None:
        if not smplx_npz.exists():
            raise SystemExit(f"--smplx-npz given but not found at {smplx_npz}")
        (models / "smplx").mkdir(parents=True, exist_ok=True)
        shutil.copy2(smplx_npz, models / "smplx" / "SMPLX_NEUTRAL.npz")


def copy_shard(out: Path, train_root: Path, shard: tuple[int, int]) -> int:
    rank, world = shard
    out_train = out / "train"
    out_train.mkdir(parents=True, exist_ok=True)

    subdirs = sorted(d for d in train_root.iterdir() if d.is_dir() and (d / f"{d.name}.bvh").exists())
    shard_dirs = subdirs[rank::world]
    for src_dir in shard_dirs:
        dst_dir = out_train / src_dir.name
        dst_dir.mkdir(parents=True, exist_ok=True)
        for f in src_dir.iterdir():
            if f.is_file():
                shutil.copy2(f, dst_dir / f.name)
    return len(shard_dirs)


def render_readme(out: Path, folder_name: str, clip_count: int) -> None:
    """Substitute {{FOLDER_NAME}} and clip-count placeholders in the README."""
    readme = out / "README.md"
    text = readme.read_text()
    hours_single = max(1, round(clip_count * 8.0 / 3600.0))     # ~8 s/clip avg on M-series
    text = (text
            .replace("{{FOLDER_NAME}}", folder_name)
            .replace("{{CLIP_COUNT}}", f"{clip_count:,}")
            .replace("{{HOURS_SINGLE}}", str(hours_single)))
    readme.write_text(text)


def ensure_output_dir(out: Path) -> None:
    output = out / "output"
    output.mkdir(parents=True, exist_ok=True)
    (output / ".gitkeep").touch()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True,
                    help="Destination folder to populate (will be created).")
    ap.add_argument("--shard", type=str, required=True,
                    help="Shard spec like 0/3 — which subset of data/Train to bundle.")
    ap.add_argument("--folder-name", type=str, default=None,
                    help="Human-readable name embedded in the README. Default: --out's basename.")
    ap.add_argument("--train", type=Path, default=DEFAULT_TRAIN,
                    help=f"Source BVH root. Default: {DEFAULT_TRAIN}")
    ap.add_argument("--task-dir", type=Path, default=DEFAULT_TASK_DIR,
                    help=f"Source dir for MediaPipe .task files. Default: {DEFAULT_TASK_DIR}")
    ap.add_argument("--include-smplx", action="store_true",
                    help="Bundle SMPLX_NEUTRAL.npz with the share folder. "
                         "Skip if you want each helper to register themselves "
                         "(stricter reading of the SMPL-X license).")
    ap.add_argument("--smplx-npz", type=Path, default=DEFAULT_SMPLX,
                    help=f"Source SMPL-X model. Default: {DEFAULT_SMPLX}")
    args = ap.parse_args()

    if not args.train.is_dir():
        raise SystemExit(f"--train dir not found: {args.train}")
    if not TEMPLATE.is_dir():
        raise SystemExit(f"template missing at {TEMPLATE}")

    shard = parse_shard(args.shard)
    folder_name = args.folder_name or args.out.name

    print(f"[bundle] target: {args.out}")
    print(f"[bundle] shard: {shard[0]}/{shard[1]}  source: {args.train}")
    args.out.mkdir(parents=True, exist_ok=True)

    copy_template(args.out)
    copy_models(args.out, args.task_dir, args.smplx_npz if args.include_smplx else None)
    n = copy_shard(args.out, args.train, shard)
    render_readme(args.out, folder_name, n)
    ensure_output_dir(args.out)

    print(f"[bundle] ✓ {n} BVH clips in shard")
    print(f"[bundle] ✓ wrote bundle to {args.out}")
    if not args.include_smplx:
        print(f"[bundle] ⚠ SMPLX_NEUTRAL.npz NOT bundled (license-sensitive).")
        print(f"[bundle]   Helper must register at smpl-x.is.tue.mpg.de and drop")
        print(f"[bundle]   the file at models/smplx/SMPLX_NEUTRAL.npz themselves.")
    print(f"[bundle] zip it:  cd {args.out.parent} && zip -r {args.out.name}.zip {args.out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
