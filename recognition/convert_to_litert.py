"""Convert the v11 KSL recognizer to LiteRT (.tflite) for mobile deployment.

This script loads `ksl_model.pt` (exported by the v11 notebook's section 9),
wraps the Recognizer in a tensor-in/tensor-out module suitable for
torch.export, and converts it to a LiteRT flatbuffer via ai-edge-torch.

Produces three artifacts:
  - ksl_model.float.tflite        ~6 MB, full float32, best accuracy
  - ksl_model.int8.tflite         ~1.5-2 MB, int8 weights, ~similar latency on CPU
  - ksl_model.metadata.json       vocabularies and label spaces for the mobile app

Run this after the v11 notebook has produced `ksl_model.pt`. On Kaggle just
attach the notebook output as a dataset and run this in a fresh notebook with:

    !pip install -q ai-edge-torch tensorflow
    !python convert_to_litert.py --checkpoint ksl_model.pt --out ksl_model

Locally, the install is heavy (~3 GB including TF + NVIDIA deps) so prefer
Kaggle or Colab unless you have the disk space.

USAGE NOTES:
  - The wrapper outputs (gloss_logits, aux_indices) where aux_indices is
    a (B, 30) tensor of argmax IDs over the 30 phonological aux heads.
    The label vocabularies for decoding those IDs are in ksl_model.metadata.json.
  - The contrastive head is dropped at inference time (training-only).
  - No `lengths` input — the wrapper assumes full 64-frame windows since the
    iOS app's ring buffer always feeds a complete window when inference runs.
    If you need variable-length support later, expose `lengths` as a second
    input (see commented-out code in InferenceWrapper).

INPUT CONTRACT (matches data/mediapipe_landmarks/*.npy and the iOS app exactly):
  features: (1, 64, 135) float32  — 45 joints × xyz, shoulder-normalised
"""
from __future__ import annotations

import argparse
import io
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Model class definitions
# ============================================================================
# These must match the v11 notebook's section 6 exactly. We copy them here so
# the converted model can be loaded standalone (no notebook required).

# Constants — keep in sync with the v11 notebook's §6 block.
N_JOINTS = 45
N_FEAT = N_JOINTS * 3   # 135
WINDOW = 64
D_MODEL = 192
N_HEADS = 4
N_LAYERS = 3
FFN_DIM = 384
DROPOUT = 0.15          # not used at inference but kept for the class def
CONTRASTIVE_DIM = 128


class TemporalConvBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.depthwise = nn.Conv1d(dim, dim, kernel_size=5, padding=2, groups=dim)
        self.pointwise = nn.Conv1d(dim, dim, kernel_size=1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y = self.norm(x).transpose(1, 2)
        y = self.depthwise(y)
        y = F.gelu(self.pointwise(y)).transpose(1, 2)
        return x + self.drop(y)


class TSLiteBlock(nn.Module):
    def __init__(self, dim, heads, ffn_dim, dropout):
        super().__init__()
        self.local = TemporalConvBlock(dim, dropout)
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.attn_drop = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim), nn.Dropout(dropout),
        )

    def forward(self, x, key_padding_mask=None):
        x = self.local(x)
        y = self.attn_norm(x)
        attn_out, _ = self.attn(y, y, y, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.attn_drop(attn_out)
        return x + self.ffn(self.ffn_norm(x))


class TSLEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_norm = nn.LayerNorm(N_FEAT)
        self.input_proj = nn.Linear(N_FEAT, D_MODEL)
        self.pos = nn.Parameter(torch.zeros(1, WINDOW, D_MODEL))
        self.blocks = nn.ModuleList([
            TSLiteBlock(D_MODEL, N_HEADS, FFN_DIM, DROPOUT) for _ in range(N_LAYERS)
        ])
        self.final_norm = nn.LayerNorm(D_MODEL)

    def forward(self, x, lengths=None):
        B, T, _ = x.shape
        h = self.input_proj(self.input_norm(x)) + self.pos[:, :T]
        if lengths is not None:
            positions = torch.arange(T, device=x.device).unsqueeze(0)
            kpm = positions >= lengths.unsqueeze(1)
        else:
            kpm = None
        for block in self.blocks:
            h = block(h, key_padding_mask=kpm)
        h = self.final_norm(h)
        return h, kpm


class Recognizer(nn.Module):
    """v11 Recognizer. AUX_KEYS and AUX_SIZES are passed in to support arbitrary
    checkpoints — they're read from the saved ckpt at load time."""
    def __init__(self, V, aux_keys, aux_sizes, use_aux=True):
        super().__init__()
        self.encoder = TSLEncoder()
        self.use_aux = use_aux
        self.aux_keys = aux_keys
        self.gloss = nn.Linear(D_MODEL, V)
        self.proj = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL), nn.GELU(),
            nn.Linear(D_MODEL, CONTRASTIVE_DIM),
        )
        if use_aux:
            for k in aux_keys:
                setattr(self, k, nn.Linear(D_MODEL, aux_sizes[k]))

    def forward(self, x, lengths=None):
        z, kpm = self.encoder(x, lengths=lengths)
        if kpm is not None:
            valid = (~kpm).float().unsqueeze(-1)
            h = (z * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        else:
            h = z.mean(dim=1)
        out = {"gloss": self.gloss(h), "embedding": h,
               "contrastive": F.normalize(self.proj(h), dim=-1)}
        if self.use_aux:
            for k in self.aux_keys:
                out[k] = getattr(self, k)(h)
        return out


# ============================================================================
# Inference wrapper for LiteRT export
# ============================================================================
class InferenceWrapper(nn.Module):
    """Tensor-in/tensor-out wrapper for torch.export and LiteRT conversion.

    The iOS app produces features in exactly this layout already (see
    `HolisticLandmarker.swift` — 45 joints × xyz, shoulder-normalised, flat
    135-dim per frame, 64-frame window). No mask channel, no length plumbing:
    the ring buffer only invokes inference once it's full, so lengths is
    always WINDOW.

    Input:
        x: (B, 64, 135) float32 — pose window matching data/mediapipe_landmarks

    Outputs:
        gloss_logits: (B, V)  float32 — softmax for top-k gloss confidences
        aux_indices:  (B, 30) int64   — argmax per phonological aux head

    The contrastive head (training-only) and embedding tensor (not needed by
    mobile) are dropped here so the converter has a clean, fixed signature.
    """

    def __init__(self, recognizer: Recognizer, aux_keys: list[str]):
        super().__init__()
        self.recognizer = recognizer
        self._aux_keys = tuple(aux_keys)

    def forward(self, x: torch.Tensor):
        B = x.shape[0]
        # Always full window — iOS only invokes once the ring buffer is full.
        lengths = torch.full((B,), WINDOW, dtype=torch.int64, device=x.device)
        out = self.recognizer(x, lengths=lengths)
        gloss_logits = out["gloss"]
        # Argmax each aux head, then stack — turns 30 logit tensors with
        # different vocab sizes into one (B, 30) int64 tensor the converter
        # can trace and the mobile app can index trivially.
        aux_indices = torch.stack(
            [out[k].argmax(dim=-1) for k in self._aux_keys], dim=-1
        )
        return gloss_logits, aux_indices


# ============================================================================
# Conversion pipeline
# ============================================================================
def reconstruct_recognizer(ckpt: dict[str, Any]) -> Recognizer:
    """Build a Recognizer from a v11 checkpoint dict and load its state_dict."""
    cfg = ckpt["model_config"]
    expected_arch = "TSLFormer-lite"
    if cfg.get("architecture") != expected_arch:
        print(f"WARNING: checkpoint architecture is {cfg.get('architecture')!r}, "
              f"expected {expected_arch!r}. Proceeding anyway.")
    aux_keys = ckpt["aux_keys"]
    aux_sizes = ckpt["aux_sizes"]
    for name, val in [("window", WINDOW), ("n_feat", N_FEAT), ("d_model", D_MODEL),
                       ("n_heads", N_HEADS), ("n_layers", N_LAYERS), ("ffn_dim", FFN_DIM)]:
        if cfg.get(name) != val:
            print(f"WARNING: ckpt config {name}={cfg.get(name)} differs from script "
                  f"constant ({val}). Conversion may fail.")
    model = Recognizer(cfg["vocab_size"], aux_keys, aux_sizes,
                        use_aux=cfg.get("use_aux", True))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def parity_check(wrapped: InferenceWrapper, recognizer: Recognizer,
                 x: torch.Tensor, aux_keys: list[str]) -> None:
    """Verify wrapped() agrees with recognizer() for the same inputs."""
    wrapped.eval(); recognizer.eval()
    with torch.no_grad():
        gloss_w, aux_w = wrapped(x)
        # Recognizer expects explicit lengths — use the same default the wrapper does.
        lengths = torch.full((x.shape[0],), WINDOW, dtype=torch.int64, device=x.device)
        out_r = recognizer(x, lengths=lengths)
        gloss_r = out_r["gloss"]
        aux_r = torch.stack([out_r[k].argmax(-1) for k in aux_keys], dim=-1)
    if not torch.allclose(gloss_w, gloss_r):
        raise RuntimeError("parity FAILED: wrapped gloss logits != recognizer gloss logits")
    if not torch.equal(aux_w, aux_r):
        raise RuntimeError("parity FAILED: wrapped aux indices != recognizer aux argmax")
    print("  parity OK: wrapped output matches recognizer output exactly")


def numeric_check(edge_output: tuple, torch_output: tuple, name: str = "") -> bool:
    """Check converted-model output is within tolerance of the original PyTorch output."""
    gloss_pt, aux_pt = torch_output
    gloss_lt, aux_lt = edge_output
    gloss_close = np.allclose(gloss_pt.numpy(), gloss_lt, atol=1e-3, rtol=1e-3)
    aux_close = np.array_equal(aux_pt.numpy(), aux_lt) if aux_lt.dtype == aux_pt.numpy().dtype \
                else np.array_equal(aux_pt.numpy(), aux_lt.astype(aux_pt.numpy().dtype))
    print(f"  {name} parity vs PyTorch: gloss_close={gloss_close}, aux_close={aux_close}")
    return gloss_close and aux_close


def export_metadata_json(ckpt: dict[str, Any], out_path: Path) -> None:
    """Write a lightweight metadata JSON for the mobile app to consume."""
    meta = {
        "version": ckpt.get("version", "v11"),
        "architecture": ckpt["model_config"].get("architecture", "TSLFormer-lite"),
        "window": WINDOW,
        "n_feat": N_FEAT,
        "n_joints": N_JOINTS,
        "vocab_size": ckpt["model_config"]["vocab_size"],
        # Gloss vocab (id <-> name)
        "gloss_id_to_name": {str(k): v for k, v in ckpt["gloss_id_to_name"].items()},
        "gloss_name_to_id": ckpt["gloss_name_to_id"],
        # Aux head vocabularies (the mobile app converts aux_indices to readable
        # phonological descriptions using these)
        "aux_keys": ckpt["aux_keys"],
        "aux_sizes": ckpt["aux_sizes"],
        "location_labels": ckpt["location_labels"],
        "move_types": ckpt["move_types"],
        "move_dirs": ckpt["move_dirs"],
        "orient_labels": ckpt["orient_labels"],
        "contact_hands_labels": ckpt["contact_hands_labels"],
        "contact_body_labels": ckpt["contact_body_labels"],
        # Retrieval centroids (gloss -> { aux_key -> int }). Mobile app uses
        # these for Hamming-distance phonological retrieval.
        "phonological_centroids": ckpt["phonological_centroids"],
        "demo_glosses": ckpt.get("demo_glosses", []),
        # Joint order so the mobile pose extractor maps MediaPipe landmarks
        # to the right feature indices.
        "joint_order": ckpt["joint_order"],
        # K-means handshape centroids — for explainability. Optional.
        "handshape_kmeans_centroids": ckpt["kmeans_centroids"],
    }
    out_path.write_text(json.dumps(meta, indent=2))
    print(f"  wrote metadata: {out_path}  ({out_path.stat().st_size/1024:.1f} KB)")


def convert(checkpoint_path: Path, out_prefix: Path, quantize: bool = True) -> None:
    """Main conversion pipeline."""
    # `ai-edge-torch` was renamed to `litert-torch` in late 2025. Prefer the
    # new name; fall back to the deprecated shim for older Kaggle kernels.
    try:
        import litert_torch as ai_edge_torch
    except ImportError:
        try:
            import ai_edge_torch  # type: ignore[no-redef]
            if not hasattr(ai_edge_torch, "convert"):
                raise ImportError("ai_edge_torch shim is missing convert()")
        except ImportError:
            print("ERROR: litert-torch (or legacy ai-edge-torch) is not installed.")
            print("Install with: pip install litert-torch     # new")
            print("           or: pip install ai-edge-torch   # legacy, Kaggle/Colab")
            sys.exit(1)

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    print(f"  ckpt version: {ckpt.get('version', '?')}")
    if isinstance(ckpt.get("val_top1"), float):
        print(f"  ckpt val_top1: {ckpt['val_top1']:.4f}")
    print(f"  aux_keys: {len(ckpt['aux_keys'])} heads")
    print(f"  vocab_size: {ckpt['model_config']['vocab_size']}")

    print("\n[1/5] Reconstructing Recognizer …")
    recognizer = reconstruct_recognizer(ckpt)
    n_params = sum(p.numel() for p in recognizer.parameters())
    print(f"  {n_params:,} params ({n_params/1e6:.2f}M)")

    print("\n[2/5] Wrapping for export …")
    aux_keys = ckpt["aux_keys"]
    wrapped = InferenceWrapper(recognizer, aux_keys).eval()

    # Sample input — single (1, 64, 135) tensor, matches what iOS sends.
    sample_x = torch.randn(1, WINDOW, N_FEAT, dtype=torch.float32)
    sample_inputs = (sample_x,)

    print("\n[3/5] Parity check (wrapper vs recognizer) …")
    parity_check(wrapped, recognizer, sample_x, aux_keys)
    with torch.no_grad():
        torch_output = wrapped(*sample_inputs)
    print(f"  reference outputs: gloss={tuple(torch_output[0].shape)}, "
          f"aux={tuple(torch_output[1].shape)} dtype={torch_output[1].dtype}")

    print("\n[4/5] Converting to LiteRT (float32) …")
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    float_path = out_prefix.with_suffix(".float.tflite")
    try:
        edge_model = ai_edge_torch.convert(wrapped, sample_inputs)
    except Exception as e:
        print(f"  CONVERSION FAILED: {type(e).__name__}: {e}")
        print("\n  Diagnostics to try next:")
        print("    1. If the error mentions 'bool' tensors or 'key_padding_mask',")
        print("       the MultiheadAttention masking path didn't trace.")
        print("       Workaround: replace nn.MultiheadAttention with manual Q/K/V")
        print("       linears + F.scaled_dot_product_attention (see ManualAttention")
        print("       class at the bottom of this script).")
        print("    2. If the error mentions 'dynamic_shapes' or 'data-dependent',")
        print("       the masked-mean-pool's clamp_min may need to be replaced with")
        print("       a torch.where on a constant lower bound.")
        print("    3. Set USE_TORCH_XLA=1 env var to try the legacy backend.")
        raise

    # Numeric parity
    edge_out = edge_model(*sample_inputs)
    if not numeric_check(edge_out, torch_output, name="float32"):
        print("  WARNING: float32 LiteRT output diverges from PyTorch. Investigate before deploying.")
    edge_model.export(str(float_path))
    print(f"  saved: {float_path}  ({float_path.stat().st_size/1024/1024:.2f} MB)")

    quantized_path = None
    if quantize:
        print("\n[5/5] Converting to LiteRT (int8 dynamic-range) …")
        quantized_path = out_prefix.with_suffix(".int8.tflite")
        try:
            # Newer litert-torch dropped `quant_recipes`; fall back to legacy path.
            try:
                from litert_torch.quantize import quant_config, quant_recipes  # type: ignore
            except ImportError:
                from ai_edge_torch.quantize import quant_config, quant_recipes  # type: ignore
            recipe = quant_recipes.full_int8_dynamic_recipe()
            cfg = quant_config.QuantConfig(pt2e_quantizer=None,
                                            generative_recipe=recipe)
            quant_edge = ai_edge_torch.convert(wrapped, sample_inputs, quant_config=cfg)
            quant_edge.export(str(quantized_path))
            print(f"  saved: {quantized_path}  ({quantized_path.stat().st_size/1024/1024:.2f} MB)")
            quant_out = quant_edge(*sample_inputs)
            numeric_check(quant_out, torch_output, name="int8")
        except Exception as e:
            print(f"  int8 conversion failed: {type(e).__name__}: {e}")
            print("  (float32 .tflite was saved successfully; you can quantize later)")
            quantized_path = None
    else:
        print("\n[5/5] Skipping quantization (--no-quantize)")

    print("\nExporting metadata JSON …")
    meta_path = out_prefix.with_suffix(".metadata.json")
    export_metadata_json(ckpt, meta_path)

    print("\n" + "=" * 60)
    print("Conversion complete. Artifacts:")
    print(f"  float32: {float_path}")
    if quantized_path: print(f"  int8:    {quantized_path}")
    print(f"  meta:    {meta_path}")
    print("\nMobile-app integration:")
    print("  - Copy ksl_model.float.tflite + ksl_model.metadata.json into")
    print("    mobile-app/sema/sema/Resources/.")
    print("  - LiteRTGlossTagger.swift loads both at startup.")
    print("  - Input: x=(1, 64, 135) float32. No lengths input.")
    print("  - Output: gloss_logits=(1, V) float32, aux_indices=(1, 30) int64.")
    print("  - Apply softmax over gloss_logits for top-k confidences.")
    print("  - Look up aux_indices entries via the metadata vocabularies.")


# ============================================================================
# Fallback: manual attention if MultiheadAttention fails to convert
# ============================================================================
# If ai_edge_torch.convert crashes inside MultiheadAttention with the
# key_padding_mask, swap TSLiteBlock's `self.attn` for ManualAttention.
# Manual attention is well-supported by torch.export.

class ManualAttention(nn.Module):
    def __init__(self, dim, n_heads, dropout=0.0):
        super().__init__()
        assert dim % n_heads == 0
        self.dim = dim; self.n_heads = n_heads; self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = dropout

    def forward(self, q_in, k_in, v_in, key_padding_mask=None, need_weights=False):
        B, T, D = q_in.shape
        qkv = self.qkv(q_in).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = key_padding_mask.to(dtype=q.dtype) * -1e4
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                              dropout_p=self.dropout if self.training else 0.0)
        out = out.transpose(1, 2).contiguous().reshape(B, T, D)
        return self.out_proj(out), None


# ============================================================================
# CLI
# ============================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Convert v11 KSL recognizer to LiteRT")
    ap.add_argument("--checkpoint", "-c", required=True, type=Path,
                    help="Path to ksl_model.pt produced by the v11 notebook")
    ap.add_argument("--out", "-o", default=Path("ksl_model"), type=Path,
                    help="Output file prefix (default: ksl_model). "
                         "Produces <prefix>.float.tflite, <prefix>.int8.tflite, "
                         "and <prefix>.metadata.json")
    ap.add_argument("--no-quantize", action="store_true",
                    help="Skip the int8 quantized export")
    args = ap.parse_args()
    convert(args.checkpoint, args.out, quantize=not args.no_quantize)
