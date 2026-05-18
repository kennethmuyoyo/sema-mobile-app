"""
Merge the gemma_ksl_lora PEFT adapter into the unsloth/gemma-4-E4B-it base
model and save the result as a plain HuggingFace fp16 checkpoint.

This is step 1 of getting the KSL-tuned Gemma onto the iOS device:

    [HF base + LoRA adapter]  ── merge_lora.py ──▶  [merged fp16 HF dir]
                                                        │
                                                        ▼
                                              convert_to_tflite.py
                                                        │
                                                        ▼
                                              [gemma_ksl_int4.tflite]

Usage:
    source ../../.venv_convert/bin/activate
    python merge_lora.py
        [--base unsloth/gemma-4-E4B-it]
        [--adapter ../../mobile-app/sema/sema/Models/gemma_ksl_lora]
        [--out merged_model]

Resource notes:
    - Downloads ~5 GB on first run (cached in ~/.cache/huggingface/hub).
    - Peak RAM: ~16 GB during merge (model loaded in fp16 + LoRA deltas).
    - Output: ~8 GB on disk.
    - The adapter targets `unsloth/gemma-4-E4B-it-unsloth-bnb-4bit` per
      adapter_config.json, but loading that to merge would require dequant.
      The non-bnb base `unsloth/gemma-4-E4B-it` is equivalent in fp16 and
      simpler to merge against, so we use that by default.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    # LoRA lives alongside this script under llm_export/lora/. It was
    # originally dropped into mobile-app/sema/sema/Models/ but Xcode 16's
    # synchronized folder reference auto-bundles every nested file, which
    # caused "Multiple commands produce README.md" collisions and added
    # ~463 MB of training artefacts to the .ipa that no Swift code reads.
    default_adapter = (here / "lora" / "gemma_ksl_lora").resolve()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        default="unsloth/gemma-4-E4B-it",
        help="HF id of the base model to merge into.",
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=default_adapter,
        help="Path to the PEFT LoRA adapter directory.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=here / "merged_model",
        help="Output directory for the merged HF model.",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Precision for the merged weights.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]

    print(f"[merge] base     : {args.base}")
    print(f"[merge] adapter  : {args.adapter}")
    print(f"[merge] out      : {args.out}")
    print(f"[merge] dtype    : {args.dtype}")

    if not args.adapter.is_dir():
        raise SystemExit(f"adapter directory does not exist: {args.adapter}")

    print("[merge] loading base model (this downloads ~5 GB on first run)...")
    base = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )

    print("[merge] loading tokenizer...")
    # Prefer the LoRA's bundled tokenizer (it has the chat template the
    # adapter was trained against). Fall back to the base's tokenizer if
    # the adapter directory doesn't have one.
    tokenizer_source = args.adapter if (args.adapter / "tokenizer.json").exists() else args.base
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)

    print("[merge] attaching LoRA adapter...")
    peft_model = PeftModel.from_pretrained(base, str(args.adapter), torch_dtype=dtype)

    print("[merge] merging LoRA into base weights (peft.merge_and_unload)...")
    merged = peft_model.merge_and_unload()

    print(f"[merge] writing merged model to {args.out}...")
    args.out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.out, safe_serialization=True)
    tokenizer.save_pretrained(args.out)

    print("[merge] done. Next step:")
    print(f"    python convert_to_tflite.py --merged {args.out}")


if __name__ == "__main__":
    main()
