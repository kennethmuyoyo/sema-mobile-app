#!/usr/bin/env python3
"""Merge local LoRA adapter into the Gemma 4 E2B base and save full weights."""

import argparse
import json
import shutil
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Merge Gemma 4 LoRA adapter into base weights.")
    parser.add_argument(
        "--base-model",
        default=None,
        help="HF model id or local path for the base checkpoint. Defaults to google/gemma-4-E2B-it, or the Unsloth 4-bit base when --load-in-4bit is set.",
    )
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=Path("."),
        help="Directory containing adapter_config.json and adapter_model.safetensors.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("merged_model"),
        help="Where to write the merged safetensors checkpoint.",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load the 4-bit Unsloth base (unsloth/gemma-4-E2B-it-unsloth-bnb-4bit).",
    )
    return parser.parse_args()


def copy_tokenizer_files(src_dir: Path, dst_dir: Path) -> None:
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "processor_config.json",
        "preprocessor_config.json",
    ):
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, dst_dir / name)


def load_lora_tensors(adapter_dir: Path):
    from safetensors import safe_open

    adapter_file = adapter_dir / "adapter_model.safetensors"
    tensors = {}
    with safe_open(str(adapter_file), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            tensors[key] = handle.get_tensor(key)
    return tensors


def merge_lora_into_model(model, adapter_dir: Path) -> int:
    import torch

    config_path = adapter_dir / "adapter_config.json"
    with open(config_path, "r", encoding="utf-8") as handle:
        adapter_config = json.load(handle)

    rank = int(adapter_config.get("r", 0))
    alpha = int(adapter_config.get("lora_alpha", rank))
    if rank <= 0:
        raise ValueError(f"Invalid LoRA rank in {config_path}: {rank}")
    scaling = alpha / rank

    tensors = load_lora_tensors(adapter_dir)
    modules = dict(model.named_modules())
    merged_count = 0
    skipped_keys = []

    grouped = {}
    prefix = "base_model.model."
    suffixes = (".lora_A.weight", ".lora_B.weight")
    for key, tensor in tensors.items():
        if not key.startswith(prefix) or not key.endswith(suffixes):
            continue
        base_key = key[len(prefix):]
        for suffix in suffixes:
            if base_key.endswith(suffix):
                base_key = base_key[: -len(suffix)]
                break
        grouped.setdefault(base_key, {})["A" if ".lora_A.weight" in key else "B"] = tensor

    for base_key, pair in grouped.items():
        if "A" not in pair or "B" not in pair:
            continue

        module = modules.get(base_key)
        if module is None:
            module = modules.get(f"{base_key}.linear")
        if module is None:
            skipped_keys.append(base_key)
            continue

        target = getattr(module, "linear", module)
        if not hasattr(target, "weight"):
            skipped_keys.append(base_key)
            continue

        lora_a = pair["A"].to(dtype=torch.float32)
        lora_b = pair["B"].to(dtype=torch.float32)
        delta = (lora_b @ lora_a) * scaling

        weight = target.weight
        if weight.shape != delta.shape:
            skipped_keys.append(base_key)
            continue

        target.weight.data.add_(delta.to(dtype=weight.dtype, device=weight.device))
        merged_count += 1

    if skipped_keys:
        print(f"Warning: skipped {len(skipped_keys)} adapter modules that did not match the loaded base.")
        for key in skipped_keys[:20]:
            print(f"  skipped: {key}")

    return merged_count


def main() -> None:
    args = parse_args()

    try:
        import torch
        from transformers import AutoProcessor, Gemma4ForConditionalGeneration
    except ImportError as exc:
        print("Missing dependencies:", exc)
        print("Install with: pip install torch transformers peft accelerate safetensors")
        sys.exit(1)

    adapter_dir = args.adapter_dir.resolve()
    adapter_file = adapter_dir / "adapter_model.safetensors"
    if not adapter_file.exists():
        print(f"Adapter not found: {adapter_file}")
        sys.exit(1)

    if args.base_model:
        base_name = args.base_model
    elif args.load_in_4bit:
        base_name = "unsloth/gemma-4-E2B-it-unsloth-bnb-4bit"
    else:
        base_name = "google/gemma-4-E2B-it"

    print(f"Loading base model: {base_name}")
    load_kwargs = {"low_cpu_mem_usage": True, "torch_dtype": torch.float16}
    if args.load_in_4bit:
        load_kwargs = {"device_map": "auto", "load_in_4bit": True}

    try:
        model = Gemma4ForConditionalGeneration.from_pretrained(base_name, **load_kwargs)
    except Exception as exc:
        print("Failed to load base model:", exc)
        sys.exit(1)

    print(f"Merging LoRA adapter from {adapter_dir}")
    try:
        merged_count = merge_lora_into_model(model, adapter_dir)
    except Exception as exc:
        print("Failed to merge adapter:", exc)
        sys.exit(1)

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Merged {merged_count} LoRA modules.")
    print(f"Saving merged model to {out_dir}")
    model.save_pretrained(out_dir, safe_serialization=True)

    try:
        processor = AutoProcessor.from_pretrained(base_name)
        processor.save_pretrained(out_dir)
    except Exception as exc:
        print("Warning: could not save processor from base:", exc)
        copy_tokenizer_files(adapter_dir, out_dir)
    else:
        copy_tokenizer_files(adapter_dir, out_dir)

    print("Done. Merged checkpoint:", out_dir)


if __name__ == "__main__":
    main()
