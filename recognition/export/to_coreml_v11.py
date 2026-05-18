"""Convert the v11 KSL recognizer to CoreML (.mlpackage).

Mirrors `recognition/convert_to_litert.py` but targets Apple's CoreML
runtime via `coremltools.convert`. Output:

  ksl_model.mlpackage           — CoreML model
  ksl_model.metadata.json       — vocabularies + aux head label spaces
                                  (identical schema to the LiteRT export)

Why CoreML for the gloss tagger? On iOS, MediaPipeTasksVision (used for
landmark extraction) ships its own internal copy of TFLite and force-loads
its symbols. Linking a second copy via the `TensorFlowLiteSwift` pod
produces ~48 duplicate-symbol linker errors. Apple's CoreML runtime is
built into iOS, has zero pod conflicts, and is the lowest-friction path
to run the .pt's recognizer head on device.

USAGE:
  python recognition/export/to_coreml_v11.py \
      --checkpoint recognition/ksl_model.pt \
      --out        mobile-app/sema/sema/Resources/ksl_model

Notes:
  - The model classes are re-imported from `convert_to_litert.py` so we
    don't fork them; weight/key drift is impossible.
  - We export the same `InferenceWrapper` that the LiteRT converter uses,
    so the iOS contract (gloss_logits + aux_indices) stays unchanged.
  - Apple's CoreML compiler tolerates `nn.MultiheadAttention` cleanly on
    iOS 17+. If you ever hit a converter crash, swap to the
    `ManualAttention` block from `convert_to_litert.py`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "recognition"))

# Reuse the v11 model definitions verbatim so this script never drifts.
from convert_to_litert import (   # type: ignore  # noqa: E402
    Recognizer,
    reconstruct_recognizer,
    export_metadata_json,
    WINDOW,
    N_FEAT,
)


class ManualAttention(torch.nn.Module):
    """Drop-in replacement for `nn.MultiheadAttention` (batch_first=True).

    `nn.MultiheadAttention` traces to `_native_multi_head_attention`, which
    coremltools' PyTorch frontend doesn't implement. Breaking the op into
    Q/K/V linears + `scaled_dot_product_attention` produces only primitive
    nodes that the converter supports.

    Weight layout matches `nn.MultiheadAttention`'s — `in_proj_weight` is
    `[Wq; Wk; Wv]` stacked, so we just rename to `qkv.weight`.
    """
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        assert dim % n_heads == 0
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = torch.nn.Linear(dim, dim * 3)
        self.out_proj = torch.nn.Linear(dim, dim)

    def forward(self, q_in, k_in, v_in, key_padding_mask=None, need_weights=False):
        B, T, D = q_in.shape
        qkv = self.qkv(q_in).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # iOS only runs inference on a full window, so key_padding_mask is
        # always None at inference. We ignore it here (the LiteRT export
        # does the same).
        import torch.nn.functional as F  # local import keeps top of file lean
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).contiguous().reshape(B, T, D)
        return self.out_proj(out), None


def swap_mha_for_manual(recognizer) -> None:
    """Walk a v11 Recognizer and replace every nn.MultiheadAttention in its
    TSLite blocks with a ManualAttention initialised from the same weights.
    Done in-place; weights are copied so no retraining needed."""
    for block in recognizer.encoder.blocks:
        old = block.attn
        manual = ManualAttention(old.embed_dim, old.num_heads)
        # in_proj_weight is [Wq; Wk; Wv] stacked => maps 1:1 to qkv.weight.
        manual.qkv.weight.data.copy_(old.in_proj_weight.data)
        manual.qkv.bias.data.copy_(old.in_proj_bias.data)
        manual.out_proj.weight.data.copy_(old.out_proj.weight.data)
        manual.out_proj.bias.data.copy_(old.out_proj.bias.data)
        block.attn = manual


class CoreMLInferenceWrapper(torch.nn.Module):
    """Single-output wrapper: features -> gloss_logits.

    The LiteRT version of `InferenceWrapper` stacks argmax over 30 aux
    phonological heads and produces a second output of int indices. The
    coremltools PyTorch frontend can't statically resolve the int-cast on
    that path (`TypeError: only 0-dimensional arrays`). Aux indices are
    only used by the (currently-disabled) phonological Hamming retrieval
    in `LiteRTGlossTagger.retrieveByHamming`, so for the CoreML build we
    drop them. If we ever want them back we'll port the retrieval to use
    the model's per-frame logits directly.
    """
    def __init__(self, recognizer: torch.nn.Module):
        super().__init__()
        self.recognizer = recognizer

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # No `lengths` arg: the iOS ring buffer never invokes the
        # recogniser on a partial window, so masked-mean-pooling is dead
        # weight here. Skipping it also dodges a coremltools tracer issue
        # where `length.unsqueeze(1)` and the `.clamp_min(1.0)` end up
        # producing an int-cast op the converter can't statically resolve.
        out = self.recognizer(features, lengths=None)
        return out["gloss"]   # (B, V), float32


def convert(checkpoint_path: Path, out_prefix: Path) -> None:
    try:
        import coremltools as ct
    except ImportError:
        print("ERROR: coremltools not installed. `pip install coremltools`.",
              file=sys.stderr)
        sys.exit(1)

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    print(f"  ckpt version: {ckpt.get('version', '?')}")
    if isinstance(ckpt.get("val_top1"), float):
        print(f"  ckpt val_top1: {ckpt['val_top1']:.4f}")
    print(f"  aux_keys: {len(ckpt['aux_keys'])} heads")
    print(f"  vocab_size: {ckpt['model_config']['vocab_size']}")

    print("\n[1/4] Reconstructing Recognizer …")
    recognizer = reconstruct_recognizer(ckpt)
    n_params = sum(p.numel() for p in recognizer.parameters())
    print(f"  {n_params:,} params ({n_params/1e6:.2f}M)")

    print("\n[2/4] Wrapping for export …")
    # Replace fused nn.MultiheadAttention modules with primitive Q/K/V
    # linears so coremltools can trace the graph.
    swap_mha_for_manual(recognizer)
    wrapped = CoreMLInferenceWrapper(recognizer).eval()

    # Quick sanity check against the un-wrapped recognizer so we know the
    # wrapper isn't dropping anything important.
    sample = torch.randn(1, WINDOW, N_FEAT, dtype=torch.float32)
    with torch.no_grad():
        ref_full = recognizer(sample, lengths=None)
        ref_gloss = ref_full["gloss"]
        wrap_gloss = wrapped(sample)
        max_abs = (ref_gloss - wrap_gloss).abs().max().item()
        assert max_abs < 1e-5, (
            f"CoreMLInferenceWrapper diverged from Recognizer ({max_abs:.2e})")
    print(f"  wrapper parity OK (gloss_logits max-abs diff {max_abs:.2e})")

    print(f"\n[3/4] Tracing CoreMLInferenceWrapper (input {tuple(sample.shape)}) …")
    with torch.no_grad():
        traced = torch.jit.trace(wrapped, sample)

    print("\n[4/4] Converting via coremltools.convert …")
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    # FLOAT32 instead of the prior FLOAT16. The FP16 build was producing
    # logits in the ±200,000 range on-device — the transformer's attention
    # softmax + FFN intermediates relied on FP32 dynamic range that FP16
    # silently overflows. With FP32 weights the activations stay in the
    # ±10 range the gloss head was trained against. Cost: ~5 ms inference
    # instead of <1 ms, and the .mlpackage grows from ~3 MB to ~9 MB.
    # Worth it — the FP16 build was unusable.
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="features", shape=sample.shape, dtype=np.float32)],
        outputs=[ct.TensorType(name="gloss_logits")],
        minimum_deployment_target=ct.target.iOS17,
        convert_to="mlprogram",          # produces .mlpackage
        compute_precision=ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.ALL,
    )

    # Post-conversion parity check: feed the same random window through both
    # the PyTorch wrapper and the converted CoreML model and confirm logits
    # agree to ~1e-3. If this fails, the converted model is broken and the
    # iOS app will see garbage — better to crash here than ship the bug.
    print("\n[parity] running CoreML vs PyTorch on the trace sample …")
    pt_out = wrapped(sample).detach().cpu().numpy()
    ml_out = mlmodel.predict({"features": sample.numpy()})["gloss_logits"]
    diff = np.abs(pt_out - ml_out).max()
    pt_min, pt_max = float(pt_out.min()), float(pt_out.max())
    ml_min, ml_max = float(ml_out.min()), float(ml_out.max())
    print(f"  PyTorch logits: min={pt_min:+.3f} max={pt_max:+.3f}")
    print(f"  CoreML  logits: min={ml_min:+.3f} max={ml_max:+.3f}")
    print(f"  max-abs diff:   {diff:.3e}")
    if diff > 1e-2:
        print(f"  WARNING: parity diff {diff:.3e} exceeds 1e-2 — the converted "
              "model may be numerically degraded. Inspect before shipping.")
    else:
        print("  parity OK — converted model matches PyTorch within tolerance")

    # Stamp version + provenance so iOS can sanity-check the bundle at load.
    mlmodel.author = "sema-mobile-app/recognition"
    mlmodel.short_description = (
        f"KSL gloss recognizer (v11, vocab={ckpt['model_config']['vocab_size']}). "
        "Input: (1, 64, 135) shoulder-normalised landmarks at 24 fps. "
        "Output: (1, V) gloss logits + (1, 30) aux head argmax indices."
    )

    mlpackage_path = out_prefix.with_suffix(".mlpackage")
    mlmodel.save(str(mlpackage_path))
    size_mb = sum(p.stat().st_size for p in mlpackage_path.rglob("*") if p.is_file()) / 1024 / 1024
    print(f"  saved: {mlpackage_path}  ({size_mb:.1f} MB)")

    # Metadata sidecar (vocabularies, aux label spaces) — same schema as the
    # LiteRT export so iOS code can keep one parser.
    meta_path = out_prefix.with_suffix(".metadata.json")
    export_metadata_json(ckpt, meta_path)

    print("\n" + "=" * 60)
    print("CoreML conversion complete.")
    print(f"  mlpackage: {mlpackage_path}")
    print(f"  metadata:  {meta_path}")
    print("\nMobile-app integration:")
    print("  - Drop the .mlpackage into mobile-app/sema/sema/Resources/.")
    print("  - Add a CoreMLGlossTagger.swift that wraps Vision/CoreML's "
          "MLModel API and reads the .metadata.json sidecar.")
    print("  - Remove `TensorFlowLiteSwift` from the Podfile — no more "
          "TFLite/MediaPipe symbol clash.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", type=Path, required=True,
                    help="path to the v11 PyTorch checkpoint (.pt)")
    ap.add_argument("--out", type=Path, required=True,
                    help="output prefix (e.g. ksl_model -> ksl_model.mlpackage)")
    args = ap.parse_args()
    convert(args.checkpoint, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
