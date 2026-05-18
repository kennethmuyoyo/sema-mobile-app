"""Kaggle: merge gemma_ksl_lora into base Gemma 4 E4B + export ksl_model.litertlm.

Single-cell, idempotent. Paste the whole file into ONE Kaggle code cell.
- First run installs pinned deps and asks you to restart the session.
- After Run -> Restart Session, re-run the same cell. It skips install,
  detects the merged checkpoint (or rebuilds it), and runs the export.

Setup:
  1. Add Input -> Upload Dataset: the contents of
     mobile-app/sema/sema/Models/gemma_ksl_lora/ as a private dataset.
  2. Settings -> Accelerator: GPU T4 x2 or P100. Internet ON.
  3. Paste this entire file as one cell. Run.
  4. Run -> Restart Session. Run the cell again.
  5. Download /kaggle/working/ksl_model.litertlm and drop it into
     mobile-app/sema/sema/Resources/.
"""
import gc
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------- config
LORA_DIR_HINT = Path("/kaggle/input/gemma-ksl-lora")
BASE = "unsloth/gemma-4-E4B-it"
MERGED_DIR = Path("/kaggle/working/gemma_ksl_merged")
EXPORT_DIR = Path("/kaggle/working")
OUTPUT_NAME = "ksl_model"

# Pins that all work together (matches the local Mac venv that successfully
# loaded the model + applied the LoRA + merged it). Kaggle's preinstalled
# torch/torchao/transformers are all skewed for litert-torch 0.9.
PINS_NO_DEPS = [
    "torch==2.11.0",
    "torchao==0.17.0",
    "transformers==5.8.1",
    "tokenizers==0.22.2",
    "accelerate==1.13.0",
    "huggingface_hub==1.15.0",
    "peft==0.19.1",
]
PINS_WITH_DEPS = [
    "litert-torch==0.9.0",
    "ai-edge-litert==2.1.4",
    "ai-edge-quantizer==0.6.0",
    "safetensors",
]

# ---------------------------------------------------------------- deps
def _deps_ready() -> bool:
    """True iff the running interpreter already has the pin set.

    Strips CUDA local-version suffixes (e.g. '2.11.0+cu130' -> '2.11.0')
    before comparing torch, and verifies that the two imports the export
    pipeline actually needs both load cleanly.
    """
    try:
        import torch
        base = torch.__version__.split("+", 1)[0]
        print(f"  [_deps_ready] torch.__version__ = {torch.__version__!r}")
        if not base.startswith("2.11."):
            return False
        from torchao.quantization import Granularity  # noqa: F401
        from litert_torch.generative.export_hf.export import export  # noqa: F401
        return True
    except Exception as exc:
        print(f"  [_deps_ready] import failed: {type(exc).__name__}: {exc}")
        return False


if not _deps_ready():
    print("Installing pinned dependencies (~3 min) ...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet",
        "--upgrade", "--force-reinstall", "--no-deps", *PINS_NO_DEPS,
    ])
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet", *PINS_WITH_DEPS,
    ])
    print()
    print("=" * 64)
    print("DEPENDENCIES INSTALLED.")
    print("Now do:  Kaggle menu -> Run -> Restart Session")
    print("Then re-run this same cell. Merge+export will proceed automatically.")
    print("=" * 64)
    raise SystemExit(0)

# ---------------------------------------------------------------- main
import torch
from transformers import Gemma4ForConditionalGeneration
from peft import PeftModel, LoraConfig

print(f"torch={torch.__version__}  cuda={torch.cuda.is_available()}  ready.\n")

lora_dir = LORA_DIR_HINT
if not (lora_dir / "adapter_model.safetensors").exists():
    for found in Path("/kaggle/input").rglob("adapter_model.safetensors"):
        lora_dir = found.parent
        break
assert (lora_dir / "adapter_config.json").exists(), (
    "LoRA not found under /kaggle/input. Upload gemma_ksl_lora as a dataset.")
print(f"LoRA dir: {lora_dir}")

merged_ok = (MERGED_DIR / "model.safetensors.index.json").exists() and any(
    MERGED_DIR.glob("model-*.safetensors"))
if merged_ok:
    print(f"\n[1-4] Merge already done at {MERGED_DIR}, skipping.")
else:
    print(f"\n[1] Loading base {BASE} ...")
    t0 = time.time()
    model = Gemma4ForConditionalGeneration.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True,
    )
    print(f"    base loaded in {time.time()-t0:.0f}s")

    print(f"\n[2] Attaching LoRA (text decoder only) ...")
    cfg = json.load(open(lora_dir / "adapter_config.json"))
    cfg["exclude_modules"] = "(.*vision_tower.*)|(.*audio_tower.*)"
    for k in ("alora_invocation_tokens", "auto_mapping", "arrow_config",
              "corda_config", "ensure_weight_tying", "eva_config",
              "qalora_group_size", "trainable_token_indices", "use_qalora"):
        cfg.pop(k, None)
    peft_config = LoraConfig(**cfg)
    t1 = time.time()
    model = PeftModel.from_pretrained(model, str(lora_dir), config=peft_config)
    print(f"    attached in {time.time()-t1:.0f}s")

    print(f"\n[3] Merging LoRA into base (merge_and_unload) ...")
    t2 = time.time()
    model = model.merge_and_unload()
    print(f"    merged in {time.time()-t2:.0f}s")

    print(f"\n[4] Saving merged model to {MERGED_DIR} ...")
    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    t3 = time.time()
    model.save_pretrained(MERGED_DIR, safe_serialization=True, max_shard_size="4GB")
    for name in ("tokenizer.json", "tokenizer_config.json",
                 "chat_template.jinja", "processor_config.json"):
        src = lora_dir / name
        if src.exists():
            shutil.copy2(src, MERGED_DIR / name)
    print(f"    saved in {time.time()-t3:.0f}s")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

final = EXPORT_DIR / f"{OUTPUT_NAME}.litertlm"
if final.exists():
    print(f"\n[5-6] {final} already exists "
          f"({final.stat().st_size/1024/1024:.1f} MB). Done.")
else:
    print(f"\n[5] Exporting to .litertlm (INT4 dynamic, text-only) ...")
    from litert_torch.generative.export_hf.export import export as run_export
    t = time.time()
    run_export(
        model=str(MERGED_DIR),
        output_dir=str(EXPORT_DIR),
        quantization_recipe="dynamic_wi4_afp32",
        export_vision_encoder=False,
        externalize_embedder=True,
        bundle_litert_lm=True,
    )
    print(f"    export finished in {time.time()-t:.0f}s")

    cands = sorted(EXPORT_DIR.glob("*.litertlm"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    assert cands, "Export claimed success but no .litertlm landed."
    src = cands[0]
    if src != final:
        src.rename(final)
    size_mb = final.stat().st_size / 1024 / 1024
    print(f"\n[6] artefact: {final}  ({size_mb:.1f} MB)")
    print(f"    Download from the Kaggle Output panel.")
    print(f"    Drop it into mobile-app/sema/sema/Resources/")

    shutil.rmtree(MERGED_DIR, ignore_errors=True)

print("\nDone.")
