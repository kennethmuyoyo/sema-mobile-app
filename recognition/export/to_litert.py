"""Export a trained gloss tagger checkpoint to LiteRT (.tflite) via ai-edge-torch.

Usage:
  python -m export.to_litert \
      --ckpt checkpoints/transformer_base/best.pt \
      --out  ../mobile-app/app/src/main/assets/models/gloss_tagger.tflite \
      --seq-len 256
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from models import build_model  # noqa: E402


class FixedLenWrapper(torch.nn.Module):
    """LiteRT runtimes prefer a static-shaped graph; expose a fixed-len model
    that takes only the features tensor and returns logits."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x, lens=None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--parity-tol", type=float, default=1e-3)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    vocab = ckpt["vocab"]
    model = build_model(cfg["model"], input_dim=cfg["data"]["feature_dim"], vocab_size=len(vocab))
    model.load_state_dict(ckpt["model"])
    model.eval()

    wrapper = FixedLenWrapper(model).eval()
    D = cfg["data"]["feature_dim"]
    sample = torch.randn(1, args.seq_len, D)

    with torch.no_grad():
        ref = wrapper(sample)

    try:
        import ai_edge_torch
    except ImportError:
        print("ai-edge-torch not installed. Run: pip install ai-edge-torch", file=sys.stderr)
        return 2

    edge = ai_edge_torch.convert(wrapper, (sample,))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    edge.export(str(out_path))
    print(f"wrote {out_path}")

    edge_out = torch.from_numpy(edge(sample.numpy()))
    diff = (edge_out - ref).abs().max().item()
    print(f"max |edge - torch|: {diff:.2e}  (tol {args.parity_tol:.0e})")
    if diff > args.parity_tol:
        print("WARNING: parity exceeds tolerance", file=sys.stderr)
        return 1

    sidecar = out_path.with_suffix(".vocab.json")
    sidecar.write_text(__import__("json").dumps(vocab, indent=2, ensure_ascii=False))
    print(f"wrote {sidecar}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
