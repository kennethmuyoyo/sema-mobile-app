"""Build the on-device pose-clip database (equal-slicing v0).

For each unique gloss token in `recognition/data/vocab/gloss_vocab.json`,
collect candidate sub-clips by **equally slicing** every full-utterance
landmark clip in `data/landmarks/` whose gloss label contains that token.
Pick the longest candidate as the canonical clip for that token, then write:

  generation/pose_library/clips/{TOKEN}.npz
  generation/pose_library/index.json

The same outputs are mirrored into
`mobile-app/sema/sema/Resources/PoseLibrary/` so the iOS app's bundle picks
them up automatically (Xcode 16 synchronised folder).

Per `mobile-app/docs/path_b_avatar.md`:
- Equal-slicing is the v0 floor; replace with CTC-forced alignment after
  the recognizer is properly trained.
- A token must yield at least `MIN_FRAMES` (6 frames @ 24 fps ≈ 0.25 s)
  to be kept.
- File-name sanitisation: lowercase, replace `/`, `?`, `*`, ` `, `:` with
  `_`. Token-to-path uniqueness is asserted at write time.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

PUNCT_TAIL = re.compile(r"(?://+|[?.,!])+$")
MIN_FRAMES = 6
SANITIZE = re.compile(r"[\\/?*:\s<>|\"]+")


def tokenize_gloss(s: str) -> list[str]:
    """Same tokenisation as `recognition/data/dataset.py`."""
    out = []
    for tok in str(s).split():
        tok = PUNCT_TAIL.sub("", tok)
        if tok:
            out.append(tok)
    return out


def sanitize_filename(token: str) -> str:
    return SANITIZE.sub("_", token)


def quantize_int8(clip: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-joint per-coord symmetric int8 quantisation.

    Input  : (T, 45, 3) float32
    Output : (q (T, 45, 3) int8, scale (45, 3) float32)
    Reconstruction: clip ≈ q.astype(float32) / 127 * scale
    """
    assert clip.ndim == 3 and clip.shape[1:] == (45, 3), f"unexpected shape {clip.shape}"
    scale = np.maximum(np.abs(clip).max(axis=0), 1e-6).astype(np.float32)
    q = np.round(clip / scale * 127.0).clip(-127, 127).astype(np.int8)
    return q, scale


def build_candidates(
    train_csv: Path,
    landmarks_dir: Path,
    vocab: dict[str, int],
) -> dict[str, list[tuple[np.ndarray, int, tuple[int, int]]]]:
    """Walk every row that has both a gloss and an existing landmark file;
    equal-slice and add slices to `candidates[token]`.
    """
    df = pd.read_csv(train_csv)[["id", "gloss"]].dropna()
    have = {int(p.stem) for p in landmarks_dir.glob("*.npy")}
    df = df[df["id"].isin(have)].reset_index(drop=True)

    candidates: dict[str, list[tuple[np.ndarray, int, tuple[int, int]]]] = defaultdict(list)
    for rec in tqdm(df.to_dict("records"), desc="slicing clips"):
        tokens = tokenize_gloss(rec["gloss"])
        if not tokens:
            continue
        clip = np.load(landmarks_dir / f"{rec['id']}.npy").astype(np.float32)
        if clip.ndim != 3 or clip.shape[1:] != (45, 3):
            continue
        T = clip.shape[0]
        n = len(tokens)
        bounds = [round(i * T / n) for i in range(n + 1)]
        for i, tok in enumerate(tokens):
            if tok not in vocab:
                continue
            start, end = bounds[i], bounds[i + 1]
            if end - start < MIN_FRAMES:
                continue
            candidates[tok].append((clip[start:end].copy(), int(rec["id"]), (start, end)))
    return candidates


def write_library(
    candidates: dict[str, list[tuple[np.ndarray, int, tuple[int, int]]]],
    out_root: Path,
    bundle_root: Path | None,
    clip_format: str,
) -> dict[str, dict]:
    clips_dir = out_root / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    if bundle_root is not None:
        (bundle_root / "clips").mkdir(parents=True, exist_ok=True)

    index: dict[str, dict] = {}
    seen_paths: set[str] = set()
    n_skipped = 0

    for tok, lst in tqdm(sorted(candidates.items()), desc="writing clips"):
        if not lst:
            n_skipped += 1
            continue
        clip, source_id, (start, end) = max(lst, key=lambda x: x[0].shape[0])

        fname = sanitize_filename(tok) + ".npz"
        if fname in seen_paths:
            raise ValueError(f"Filename collision for token '{tok}' (path={fname})")
        seen_paths.add(fname)
        rel_path = f"clips/{fname}"

        # Uncompressed .npz — iOS reader uses Compression.framework whose
        # COMPRESSION_ZLIB is RFC 1950 (zlib-wrapped), not raw DEFLATE that
        # `np.savez_compressed` uses inside the ZIP.
        if clip_format == "int8":
            q, scale = quantize_int8(clip)
            np.savez(out_root / rel_path, clip_i8=q, scale=scale)
            if bundle_root is not None:
                np.savez(bundle_root / rel_path, clip_i8=q, scale=scale)
        elif clip_format == "float32":
            clip_f32 = clip.astype(np.float32, copy=False)
            np.savez(out_root / rel_path, clip_f32=clip_f32)
            if bundle_root is not None:
                np.savez(bundle_root / rel_path, clip_f32=clip_f32)
        else:
            raise ValueError(f"unknown clip_format: {clip_format}")

        index[tok] = {
            "path": rel_path,
            "n_frames": int(clip.shape[0]),
            "fps": 24.0,
            "source_clip_id": source_id,
            "source_range": [int(start), int(end)],
            "format": clip_format,
        }

    return index


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", default="/Users/ken/sema-mobile-app/data/train.csv")
    ap.add_argument("--landmarks-dir", default="/Users/ken/sema-mobile-app/data/landmarks")
    ap.add_argument("--vocab", default="/Users/ken/sema-mobile-app/recognition/data/vocab/gloss_vocab.json")
    ap.add_argument("--out", default="/Users/ken/sema-mobile-app/generation/pose_library")
    ap.add_argument(
        "--bundle",
        default="/Users/ken/sema-mobile-app/mobile-app/sema/sema/Resources/PoseLibrary",
        help="Also mirror to the iOS app bundle path; pass empty string to skip.",
    )
    ap.add_argument(
        "--clip-format",
        choices=["int8", "float32"],
        default="int8",
        help="Storage format inside each clip .npz. float32 keeps BVH-fidelity; int8 is smaller.",
    )
    args = ap.parse_args()

    out_root = Path(args.out)
    bundle_root = Path(args.bundle) if args.bundle else None

    vocab = json.loads(Path(args.vocab).read_text())
    landmarks_dir = Path(args.landmarks_dir)
    have_landmarks = sum(1 for _ in landmarks_dir.glob("*.npy"))
    if have_landmarks == 0:
        print(f"no landmark files in {landmarks_dir}", file=sys.stderr)
        return 1

    print(f"vocab tokens     : {len(vocab)}")
    print(f"landmark files   : {have_landmarks}")
    print(f"output           : {out_root}")
    if bundle_root is not None:
        print(f"bundle mirror    : {bundle_root}")
    print(f"clip format      : {args.clip_format}")

    candidates = build_candidates(Path(args.train_csv), landmarks_dir, vocab)
    print(f"tokens with ≥1 candidate clip: {len(candidates)}")

    index = write_library(candidates, out_root, bundle_root, clip_format=args.clip_format)
    (out_root / "index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False))
    if bundle_root is not None:
        (bundle_root / "index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False))

    print(f"wrote {len(index)} pose clips")
    print(f"index            : {out_root/'index.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
