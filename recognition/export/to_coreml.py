"""Export a trained gloss tagger checkpoint to CoreML (.mlpackage) via coremltools.

Usage:
  python -m export.to_coreml \
      --ckpt checkpoints/transformer_base/best.pt \
      --out  ../mobile-app/Sema/Models/gloss_tagger.mlpackage \
      --seq-len 256
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from models import build_model  # noqa: E402


class FixedLenWrapper(torch.nn.Module):
    """CoreML traces a static graph; expose a fixed-shape forward."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x, lens=None)
        return out[0] if isinstance(out, tuple) else out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--parity-tol", type=float, default=1e-3)
    ap.add_argument("--minimum-deployment-target", default="iOS17")
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
        import coremltools as ct
    except ImportError:
        print("coremltools not installed. Run: pip install -r requirements-export-coreml.txt", file=sys.stderr)
        return 2

    traced = torch.jit.trace(wrapper, sample, strict=False, check_trace=False)

    target_map = {
        "iOS15": ct.target.iOS15,
        "iOS16": ct.target.iOS16,
        "iOS17": ct.target.iOS17,
    }
    target = target_map.get(args.minimum_deployment_target, ct.target.iOS17)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="features", shape=(1, args.seq_len, D), dtype=float)],
        outputs=[ct.TensorType(name="logits")],
        convert_to="mlprogram",
        minimum_deployment_target=target,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(out_path))
    print(f"wrote {out_path}")

    # Parity check
    try:
        pred = mlmodel.predict({"features": sample.numpy()})
        edge_out = torch.from_numpy(pred["logits"])
        diff = (edge_out - ref).abs().max().item()
        print(f"max |coreml - torch|: {diff:.2e}  (tol {args.parity_tol:.0e})")
        if diff > args.parity_tol:
            print("WARNING: parity exceeds tolerance", file=sys.stderr)
    except Exception as e:
        # On non-macOS hosts `predict` is unavailable; skip but warn.
        print(f"parity check skipped: {e}", file=sys.stderr)

    sidecar = out_path.with_suffix("").with_suffix(".vocab.json")
    sidecar.write_text(json.dumps(vocab, indent=2, ensure_ascii=False))
    print(f"wrote {sidecar}")

    landmarks_meta = REPO / "data" / "landmarks_meta.json"
    if landmarks_meta.exists():
        dest_meta = out_path.with_suffix("").with_suffix(".landmarks_meta.json")
        dest_meta.write_text(landmarks_meta.read_text())
        print(f"wrote {dest_meta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
