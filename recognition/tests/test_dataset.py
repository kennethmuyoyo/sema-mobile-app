"""Tiny shape/dtype sanity checks. Run with: python -m tests.test_dataset"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from data.augment import AugConfig          # noqa: E402
from data.dataset import GlossDataset, collate, load_vocab  # noqa: E402


def main() -> int:
    vocab = load_vocab(REPO / "data/vocab/gloss_vocab.json")
    ds = GlossDataset(
        root="/Users/ken/sema-mobile-app/data",
        features_dir="landmarks",
        train_csv="train.csv",
        split_file=REPO / "data/splits/train.txt",
        vocab=vocab,
        max_frames=512,
        augment=AugConfig(enabled=False, append_mask_channel=True),
    )
    assert len(ds) > 0, "empty dataset"
    s = ds[0]
    assert s["features"].dtype == torch.float32
    assert s["features"].ndim == 2
    # 45 joints * 4 channels (xyz + mask) = 180
    assert s["features"].shape[1] == 180, f"unexpected feature dim: {s['features'].shape[1]}"
    assert s["gloss"].dtype == torch.long and s["gloss"].ndim == 1
    batch = collate([ds[i] for i in range(4)])
    assert batch["features"].ndim == 3 and batch["features"].shape[0] == 4
    assert (batch["feat_lens"] <= batch["features"].shape[1]).all()
    assert (batch["gloss_lens"] <= batch["glosses"].shape[1]).all()
    print(f"ok  N={len(ds)}  feat={tuple(batch['features'].shape)}  gloss={tuple(batch['glosses'].shape)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
