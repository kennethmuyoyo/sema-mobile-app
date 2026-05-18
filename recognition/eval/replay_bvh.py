"""Controlled-recognition harness: replay single-gloss BVH clips through the
trained gloss tagger and report per-token accuracy.

Each `.bvh` in the input directory is treated as ground truth for the gloss
encoded in its filename stem (e.g. `hello.bvh` → `HELLO`). The script runs
the **same** FK + shoulder-normalisation + mask-channel pipeline the trainer
uses (no augmentation), feeds it through the checkpointed Transformer
tagger, greedy-CTC-decodes the logits, and reports top-1 / top-3 hits.

Usage:
    python recognition/eval/replay_bvh.py
        [--bvh-dir data/recognition_set]
        [--ckpt   recognition/checkpoints/transformer_base/best.pt]
        [--device cpu]

Exit code is non-zero if any clip mispredicts top-1 — wire into CI when the
recognizer is stable enough for the bar to make sense.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "recognition"))
sys.path.insert(0, str(REPO))

# Reuse the existing training-side code paths so eval stays in lockstep
# with how the model was trained.
from recognition.data.bvh_to_landmarks import (   # noqa: E402
    TARGET_JOINTS,
    forward_kinematics,
    normalize_landmarks,
    parse_bvh,
)
from recognition.models.transformer_encoder import TransformerTagger   # noqa: E402
from recognition.training.decode import greedy_ctc_decode   # noqa: E402

DEFAULT_BVH_DIR = REPO / "data" / "recognition_set"
DEFAULT_CKPT = REPO / "recognition" / "checkpoints" / "transformer_base" / "best.pt"


def bvh_to_features(bvh: Path) -> torch.Tensor:
    """(T, 180) float32. (T, 45, 3) landmarks + ones mask channel, flattened.
    Mirrors `GlossDataset.__getitem__` with augmentation disabled.
    """
    joints, motion, _ = parse_bvh(bvh)
    if motion.size == 0:
        raise ValueError(f"{bvh}: empty motion")
    pos, names = forward_kinematics(joints, motion)
    name_to_idx = {n: i for i, n in enumerate(names)}
    sel = np.array([name_to_idx[n] for n in TARGET_JOINTS], dtype=np.int64)
    pos_45 = pos[:, sel, :]
    norm = normalize_landmarks(pos_45, {n: i for i, n in enumerate(TARGET_JOINTS)}).astype(np.float32)
    # Append all-ones mask channel (the BVH-derived clip has every joint
    # present, so mask is 1 everywhere). Match `augment_clip(enabled=False)`.
    T, J, _ = norm.shape
    ones = np.ones((T, J, 1), dtype=np.float32)
    norm4 = np.concatenate([norm, ones], axis=-1)            # (T, J, 4)
    feats = norm4.reshape(T, J * 4)                          # (T, 180)
    return torch.from_numpy(feats)


def build_model_from_ckpt(ckpt_path: Path, device: torch.device) -> tuple[TransformerTagger, dict[int, str]]:
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = blob["cfg"]
    vocab: dict[str, int] = blob["vocab"]
    mcfg = cfg["model"]
    dcfg = cfg["data"]
    model = TransformerTagger(
        input_dim=int(dcfg["feature_dim"]),
        vocab_size=len(vocab),
        d_model=int(mcfg["d_model"]),
        n_heads=int(mcfg["n_heads"]),
        n_layers=int(mcfg["n_layers"]),
        ff_dim=int(mcfg["ff_dim"]),
        dropout=float(mcfg.get("dropout", 0.0)),
        input_norm=bool(mcfg.get("input_norm", True)),
        max_len=int(mcfg.get("max_len", 512)),
    )
    model.load_state_dict(blob["model"])
    model.to(device).eval()
    inv_vocab = {idx: label for label, idx in vocab.items()}
    return model, inv_vocab


def top_k_tokens(logits: torch.Tensor, inv_vocab: dict[int, str], k: int = 3) -> list[tuple[str, float]]:
    """Mean-over-time softmax peak ranking (excludes blank=0)."""
    probs = F.softmax(logits, dim=-1).mean(dim=0)            # (V,)
    probs[0] = 0.0                                            # drop CTC blank
    top = torch.topk(probs, k=min(k, probs.shape[0]))
    return [(inv_vocab.get(int(idx), f"?{int(idx)}"), float(p))
            for p, idx in zip(top.values.tolist(), top.indices.tolist())]


def evaluate(bvh_dir: Path, ckpt: Path, device: torch.device, top_k: int) -> int:
    bvh_dir = bvh_dir.resolve()
    ckpt = ckpt.resolve()

    model, inv_vocab = build_model_from_ckpt(ckpt, device)

    bvhs = sorted(bvh_dir.glob("*.bvh"))
    if not bvhs:
        print(f"[replay] no BVH files in {bvh_dir}", file=sys.stderr)
        return 1

    top1_correct = 0
    topk_correct = 0
    rows: list[tuple[str, str, str, float, bool, bool]] = []

    for bvh in bvhs:
        ground_truth = bvh.stem.upper()
        try:
            feats = bvh_to_features(bvh).to(device)
        except Exception as exc:
            print(f"[replay] ERROR {bvh.name}: {exc}", file=sys.stderr)
            continue

        with torch.no_grad():
            x = feats.unsqueeze(0)                            # (1, T, D)
            lens = torch.tensor([x.shape[1]], dtype=torch.long)
            output = model(x, lens)
            logits = output[0] if isinstance(output, (tuple, list)) else output
            logits = logits[0]                                # (T, V)

        # CTC decoded sequence (for diagnosis)
        decoded_ids = greedy_ctc_decode(logits.unsqueeze(0), lens)[0]
        decoded_str = " ".join(inv_vocab.get(i, f"?{i}") for i in decoded_ids) or "(empty)"

        topk = top_k_tokens(logits, inv_vocab, k=top_k)
        top1_label, top1_prob = topk[0]
        topk_labels = {label for label, _ in topk}

        is_top1 = top1_label == ground_truth
        is_topk = ground_truth in topk_labels
        top1_correct += int(is_top1)
        topk_correct += int(is_topk)

        rows.append((ground_truth, top1_label, decoded_str, top1_prob, is_top1, is_topk))

    def short(p: Path) -> str:
        try:
            return str(p.relative_to(REPO))
        except ValueError:
            return str(p)
    print(f"\n[replay] {short(bvh_dir)}  ckpt={short(ckpt)}\n")
    print(f"{'GT':<14} {'TOP-1':<14} {'PROB':<7} {'T1':<3} {'T'+str(top_k):<3} {'CTC DECODE':<40}")
    print("-" * 90)
    for gt, top1, decoded, prob, is_t1, is_tk in rows:
        print(f"{gt:<14} {top1:<14} {prob:<7.3f} "
              f"{'✓' if is_t1 else '✗':<3} {'✓' if is_tk else '✗':<3} {decoded:<40}")
    n = len(rows)
    print("-" * 90)
    print(f"top-1: {top1_correct}/{n} = {top1_correct/n:.1%}")
    print(f"top-{top_k}: {topk_correct}/{n} = {topk_correct/n:.1%}")
    return 0 if top1_correct == n else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bvh-dir", type=Path, default=DEFAULT_BVH_DIR)
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--top-k", type=int, default=3)
    args = ap.parse_args()

    if not args.ckpt.exists():
        print(f"[replay] checkpoint not found: {args.ckpt}", file=sys.stderr)
        return 1
    device = torch.device(args.device)
    return evaluate(args.bvh_dir, args.ckpt, device, args.top_k)


if __name__ == "__main__":
    sys.exit(main())
