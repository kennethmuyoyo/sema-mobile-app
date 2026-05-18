"""Replay the recognition_set BVH-derived clips through the LiteRT model.

For each `.npz` clip under `generation/pose_library/clips/` whose source is
`single_gloss/recognition_set`, this script:

  1. Decodes the int8 clip back to float32 `(T, 45, 3)`.
  2. Slides a `window`-frame view across it with a 16-frame stride.
  3. Runs each window through `ksl_model.float.tflite`.
  4. Averages the per-window softmax probabilities to get a single
     top-K gloss ranking for the clip.
  5. Compares to the filename stem (ground truth).

Reports per-clip top-1 / top-3 + a summary line.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

REPO = Path(__file__).resolve().parents[2]
DEFAULT_TFLITE = REPO / "mobile-app" / "sema" / "sema" / "Resources" / "ksl_model.float.tflite"
DEFAULT_META   = REPO / "mobile-app" / "sema" / "sema" / "Resources" / "ksl_model.metadata.json"
DEFAULT_CLIPS  = REPO / "generation" / "pose_library" / "clips"
DEFAULT_INDEX  = REPO / "generation" / "pose_library" / "index.json"


def load_clip(npz_path: Path) -> np.ndarray:
    """Return (T, 45, 3) float32 — handles both int8-quantised and float32 formats."""
    z = np.load(npz_path)
    if "clip_f32" in z.files:
        return z["clip_f32"].astype(np.float32)
    # int8: clip = q.astype(float32) / 127 * scale
    return (z["clip_i8"].astype(np.float32) / 127.0) * z["scale"]


def sliding_windows(clip: np.ndarray, window: int, stride: int) -> np.ndarray:
    """(T, 45, 3) → (N, window, 135). Pad with zeros if T < window."""
    T = clip.shape[0]
    flat = clip.reshape(T, -1).astype(np.float32)            # (T, 135)
    if T < window:
        pad = np.zeros((window - T, flat.shape[1]), dtype=np.float32)
        flat = np.concatenate([flat, pad], axis=0)
        T = window
    starts = list(range(0, T - window + 1, stride))
    if not starts:
        starts = [0]
    return np.stack([flat[s : s + window] for s in starts], axis=0)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tflite", type=Path, default=DEFAULT_TFLITE)
    ap.add_argument("--metadata", type=Path, default=DEFAULT_META)
    ap.add_argument("--clips", type=Path, default=DEFAULT_CLIPS,
                    help="Folder of <TOKEN>.npz clips. Default: pose_library/clips/")
    ap.add_argument("--index", type=Path, default=DEFAULT_INDEX,
                    help="index.json to filter to recognition_set source.")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--stride", type=int, default=16,
                    help="Sliding-window stride. 16 → ~4× window overlap on 64-frame clips.")
    ap.add_argument("--closed-set", action="store_true",
                    help="Restrict ranking to recognition_set tokens (mimics the "
                         "iOS demo_recognition.json allowlist). Drops ~4133 "
                         "distractor classes before argsort.")
    args = ap.parse_args()

    if not args.tflite.exists():
        print(f"[replay] tflite not found at {args.tflite}", file=sys.stderr)
        return 1

    metadata = json.loads(args.metadata.read_text())
    window: int = int(metadata["window"])
    gloss_id_to_name: dict[str, str] = metadata["gloss_id_to_name"]
    inv_vocab = {int(k): v for k, v in gloss_id_to_name.items()}

    # Filter to recognition_set tokens via the pose library index.
    if args.index.exists():
        idx = json.loads(args.index.read_text())
        recog_tokens = sorted(
            tok for tok, e in idx.items()
            if str(e.get("source", "")) == "single_gloss/recognition_set"
        )
    else:
        # Fallback: process every .npz under --clips.
        recog_tokens = sorted(p.stem for p in args.clips.glob("*.npz"))
    if not recog_tokens:
        print("[replay] no recognition_set tokens found", file=sys.stderr)
        return 1
    print(f"[replay] evaluating {len(recog_tokens)} recognition_set tokens "
          f"against window={window}, stride={args.stride}")

    interp = tf.lite.Interpreter(model_path=str(args.tflite))
    interp.allocate_tensors()
    input_details = interp.get_input_details()
    output_details = interp.get_output_details()
    # The InferenceWrapper exposes (gloss_logits, aux_indices). Identify which
    # output is gloss_logits by its second-dim size matching vocab_size.
    vocab_size = int(metadata["vocab_size"])
    gloss_out_idx = next(
        i for i, od in enumerate(output_details) if od["shape"][-1] == vocab_size
    )

    # Build allowlist mask if --closed-set: only recog_tokens are scoreable.
    allow_mask: np.ndarray | None = None
    if args.closed_set:
        name_to_id = {v: int(k) for k, v in gloss_id_to_name.items()}
        allow_ids = [name_to_id[t] for t in recog_tokens if t in name_to_id]
        allow_mask = np.full(vocab_size, -np.inf, dtype=np.float64)
        allow_mask[allow_ids] = 0.0
        print(f"[replay] closed-set: ranking restricted to {len(allow_ids)} tokens")

    rows: list[tuple[str, str, float, list[str], bool, bool]] = []
    for token in recog_tokens:
        clip_path = args.clips / f"{token}.npz"
        if not clip_path.exists():
            print(f"[replay] WARN: {clip_path} missing, skipping", file=sys.stderr)
            continue
        clip = load_clip(clip_path)
        windows = sliding_windows(clip, window, args.stride)

        # Average softmax over windows.
        probs_acc = np.zeros(vocab_size, dtype=np.float64)
        for w in windows:
            interp.set_tensor(input_details[0]["index"],
                              w.reshape(1, window, -1).astype(np.float32))
            interp.invoke()
            logits = interp.get_tensor(output_details[gloss_out_idx]["index"])[0]
            probs_acc += softmax(logits)
        probs_mean = probs_acc / len(windows)
        if allow_mask is not None:
            probs_mean = probs_mean + allow_mask  # -inf wipes non-allowed classes

        topk_ids = np.argsort(-probs_mean)[: args.top_k]
        topk_labels = [inv_vocab.get(int(i), f"?{int(i)}") for i in topk_ids]
        top1_label = topk_labels[0]
        top1_prob = float(probs_mean[topk_ids[0]])
        is_top1 = top1_label == token
        is_topk = token in topk_labels
        rows.append((token, top1_label, top1_prob, topk_labels, is_top1, is_topk))

    n = len(rows)
    top1_n = sum(r[4] for r in rows)
    topk_n = sum(r[5] for r in rows)

    print(f"\n{'GT':<12} {'TOP-1':<14} {'PROB':<8} {'T1':<3} {'T'+str(args.top_k):<3} TOP-K")
    print("-" * 90)
    for gt, top1, prob, topk, is_t1, is_tk in rows:
        mark1 = "✓" if is_t1 else "✗"
        markk = "✓" if is_tk else "✗"
        print(f"{gt:<12} {top1:<14} {prob:<8.3f} {mark1:<3} {markk:<3} {topk}")
    print("-" * 90)
    print(f"top-1: {top1_n}/{n} = {top1_n/n:.1%}")
    print(f"top-{args.top_k}: {topk_n}/{n} = {topk_n/n:.1%}")
    return 0 if top1_n == n else 1


if __name__ == "__main__":
    sys.exit(main())
