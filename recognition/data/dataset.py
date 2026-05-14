"""Dataset and collate for the gloss tagger.

Reads landmark features written by `data/bvh_to_landmarks.py`:
  /Users/ken/sema-mobile-app/data/landmarks/{id}.npy   shape (T, 45, 3) float32

Each batch item:
  features : FloatTensor (T, J*C)
              C = 3 (raw) or 4 (with augmentation mask channel)
  gloss    : LongTensor  (U,)   gloss vocab ids
  rvq_aux  : LongTensor  (K,) | None   base_tokens, if enabled in config

The collate pads to the longest sequence in the batch and returns per-sample
lengths so CTC can mask correctly.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .augment import AugConfig, augment_clip

PUNCT_TAIL = re.compile(r"(?://+|[?.,!])+$")
BLANK_ID, UNK_ID = 0, 1


def tokenize_gloss(s: str) -> list[str]:
    out = []
    for tok in str(s).split():
        tok = PUNCT_TAIL.sub("", tok)
        if tok:
            out.append(tok)
    return out


def load_vocab(path: str | Path) -> dict[str, int]:
    return json.loads(Path(path).read_text())


class GlossDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        features_dir: str,
        train_csv: str,
        split_file: str | Path,
        vocab: dict[str, int],
        max_frames: int = 512,
        augment: AugConfig | dict | None = None,
        return_rvq_aux: bool = False,
    ) -> None:
        self.root = Path(root)
        self.features_dir = self.root / features_dir
        self.vocab = vocab
        self.max_frames = max_frames
        self.return_rvq_aux = return_rvq_aux

        if isinstance(augment, dict):
            self.aug = AugConfig(**augment)
        elif isinstance(augment, AugConfig):
            self.aug = augment
        else:
            self.aug = AugConfig(enabled=False, append_mask_channel=True)

        cols = ["id", "gloss"] + (["base_tokens"] if return_rvq_aux else [])
        df = pd.read_csv(self.root / train_csv)[cols].dropna()
        ids = [int(x) for x in Path(split_file).read_text().split() if x]
        df = df[df["id"].isin(set(ids))].reset_index(drop=True)
        # Drop rows without a corresponding landmark file
        have = {int(p.stem) for p in self.features_dir.glob("*.npy")}
        df = df[df["id"].isin(have)].reset_index(drop=True)
        self.items = df.to_dict("records")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        rec = self.items[idx]
        feats = np.load(self.features_dir / f"{rec['id']}.npy").astype(np.float32)
        if feats.ndim == 2:
            feats = feats.reshape(feats.shape[0], -1, 3)        # legacy flatten guard
        if feats.shape[0] > self.max_frames:
            start = (feats.shape[0] - self.max_frames) // 2
            feats = feats[start : start + self.max_frames]

        x = torch.from_numpy(feats)
        x = augment_clip(x, self.aug)                             # (T, J, 3) or (T, J, 4)
        T = x.shape[0]
        x = x.reshape(T, -1)                                      # (T, J*C)

        toks = [self.vocab.get(t, UNK_ID) for t in tokenize_gloss(rec["gloss"])]
        if not toks:
            toks = [UNK_ID]

        out = {
            "features": x,
            "gloss": torch.tensor(toks, dtype=torch.long),
            "id": rec["id"],
        }
        if self.return_rvq_aux:
            aux = torch.tensor([int(t) for t in str(rec["base_tokens"]).split()], dtype=torch.long)
            out["rvq_aux"] = aux
        return out


def collate(batch: list[dict]) -> dict:
    B = len(batch)
    feat_lens = torch.tensor([b["features"].shape[0] for b in batch], dtype=torch.long)
    gloss_lens = torch.tensor([b["gloss"].shape[0] for b in batch], dtype=torch.long)
    T = int(feat_lens.max())
    U = int(gloss_lens.max())
    D = batch[0]["features"].shape[1]

    features = torch.zeros(B, T, D, dtype=torch.float32)
    glosses = torch.zeros(B, U, dtype=torch.long)
    for i, b in enumerate(batch):
        t = b["features"].shape[0]
        u = b["gloss"].shape[0]
        features[i, :t] = b["features"]
        glosses[i, :u] = b["gloss"]

    out = {
        "features": features,
        "feat_lens": feat_lens,
        "glosses": glosses,
        "gloss_lens": gloss_lens,
        "ids": [b["id"] for b in batch],
    }
    if "rvq_aux" in batch[0]:
        K = max(b["rvq_aux"].shape[0] for b in batch)
        rvq = torch.full((B, K), -100, dtype=torch.long)          # -100 = CE ignore
        rvq_lens = torch.tensor([b["rvq_aux"].shape[0] for b in batch], dtype=torch.long)
        for i, b in enumerate(batch):
            k = b["rvq_aux"].shape[0]
            rvq[i, :k] = b["rvq_aux"]
        out["rvq_aux"] = rvq
        out["rvq_aux_lens"] = rvq_lens
    return out
