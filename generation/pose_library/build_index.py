"""Build the on-device pose-clip database.

For each gloss in `../../recognition/data/vocab/gloss_vocab.json`:
  - Find clips in Motion-S whose gloss column contains this token.
  - Pick a canonical clip (longest cleanly-segmented exemplar for that token).
  - Slice the per-frame landmark sequence from
    /Users/ken/sema-mobile-app/data/landmarks/{id}.npy to just the frames
    that span this gloss (via base_tokens stride heuristics for now;
    proper per-gloss alignment is a follow-up).
  - int8-quantize per joint with a global symmetric scale; record the scale
    in the index so the iOS client dequantizes correctly.
  - Write `clips/{gloss_token}.npz` with keys {"clip_i8": (T, 45, 3) int8,
    "scale": float32 (45, 3)}.
  - Append to `index.json`: {gloss_token: {"path": ..., "T": ..., "fps": ...}}.

Output paths are designed to drop into mobile-app/Sema/PoseLibrary/.
"""
from __future__ import annotations

import argparse
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--landmarks-root", default="/Users/ken/sema-mobile-app/data/landmarks")
    ap.add_argument("--vocab", default="../../recognition/data/vocab/gloss_vocab.json")
    ap.add_argument("--train-csv", default="/Users/ken/sema-mobile-app/data/train.csv")
    ap.add_argument("--out", default="clips")
    ap.add_argument("--index", default="index.json")
    ap.add_argument("--int8-scale-mode", choices=["global", "per-joint"], default="per-joint")
    args = ap.parse_args()
    print("build_index.py: stub. Implementation tracked under generation/'s plan.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
