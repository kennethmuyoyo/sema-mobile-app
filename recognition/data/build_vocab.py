"""Build the gloss vocabulary and train/val split from train.csv.

Outputs:
  data/vocab/gloss_vocab.json   {token: id, ...} with <blank>=0, <unk>=1
  data/splits/train.txt         one clip id per line
  data/splits/val.txt           one clip id per line

Filters rows whose Motion-Features/{id}.npy is missing.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

PUNCT_TAIL = re.compile(r"(?://+|[?.,!])+$")
BLANK, UNK = "<blank>", "<unk>"


def tokenize_gloss(s: str) -> list[str]:
    out = []
    for tok in str(s).split():
        tok = PUNCT_TAIL.sub("", tok)
        if tok:
            out.append(tok)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/Users/ken/sema-mobile-app/data")
    ap.add_argument("--train-csv", default="train.csv")
    ap.add_argument("--features-dir", default="Motion-Features")
    ap.add_argument("--out-vocab", default="data/vocab/gloss_vocab.json")
    ap.add_argument("--out-splits", default="data/splits")
    ap.add_argument("--min-count", type=int, default=1)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    df = pd.read_csv(Path(args.root) / args.train_csv)
    df = df[["id", "gloss"]].dropna()

    feats = Path(args.root) / args.features_dir
    have = {int(p.stem) for p in feats.glob("*.npy")}
    df = df[df["id"].isin(have)].reset_index(drop=True)

    counts: Counter[str] = Counter()
    for g in df["gloss"]:
        counts.update(tokenize_gloss(g))

    kept = [t for t, c in counts.most_common() if c >= args.min_count]
    vocab = {BLANK: 0, UNK: 1}
    for tok in kept:
        vocab[tok] = len(vocab)

    vocab_path = repo / args.out_vocab
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    vocab_path.write_text(json.dumps(vocab, indent=2, ensure_ascii=False))

    rng = random.Random(args.seed)
    ids = df["id"].tolist()
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * args.val_frac))
    val_ids = sorted(ids[:n_val])
    train_ids = sorted(ids[n_val:])

    splits_dir = repo / args.out_splits
    splits_dir.mkdir(parents=True, exist_ok=True)
    (splits_dir / "train.txt").write_text("\n".join(str(i) for i in train_ids) + "\n")
    (splits_dir / "val.txt").write_text("\n".join(str(i) for i in val_ids) + "\n")

    print(f"clips           : {len(df)}")
    print(f"unique tokens   : {len(counts)}")
    print(f"vocab size      : {len(vocab)} (including <blank>, <unk>)")
    print(f"train / val     : {len(train_ids)} / {len(val_ids)}")
    print(f"vocab           : {vocab_path}")
    print(f"splits          : {splits_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
