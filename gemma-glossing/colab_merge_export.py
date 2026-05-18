"""Google Colab: merge gemma_ksl_lora into Gemma 4 E4B + export ksl_model.litertlm.

Single-cell, idempotent. Paste the whole file into ONE Colab code cell.

Setup (one-time):
  1. Open https://colab.research.google.com -> New notebook.
  2. Runtime -> Change runtime type -> T4 GPU (free) is fine.
  3. In Google Drive, create a folder `MyDrive/gemma_ksl_lora/` and upload
     ALL 7 files from `mobile-app/sema/sema/Models/gemma_ksl_lora/`:
       adapter_config.json, adapter_model.safetensors, chat_template.jinja,
       processor_config.json, README.md, tokenizer_config.json, tokenizer.json
  4. Paste this entire file into one cell and Run.
  5. When prompted, Runtime -> Restart runtime, then Run the cell again.
  6. Final artefact lands at `MyDrive/ksl_model.litertlm` (~3 GB). Download
     from Drive and drop into `mobile-app/sema/sema/Resources/`.

Total runtime ~15 min end-to-end (mostly the INT4 quantization pass).
"""
import gc
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------- config
BASE = "unsloth/gemma-4-E4B-it"
WORK_DIR = Path("/content")
MERGED_DIR = WORK_DIR / "gemma_ksl_merged"
DRIVE_MOUNT = Path("/content/drive")
DRIVE_ROOT = DRIVE_MOUNT / "MyDrive"
OUTPUT_NAME = "ksl_model"

# Same pin set that works locally. Colab's preinstalls are much leaner than
# Kaggle's so we don't need --no-deps gymnastics, but we still pin every
# layer of the stack to known-good versions to avoid surprises.
PINS = [
    "torch==2.11.0",
    "torchao==0.17.0",
    "transformers==5.8.1",
    "tokenizers==0.22.2",
    "accelerate==1.13.0",
    "huggingface_hub==1.15.0",
    "peft==0.19.1",
    "safetensors",
    "litert-torch==0.9.0",
    "ai-edge-litert==2.1.4",
    "ai-edge-quantizer==0.6.0",
]

# ---------------------------------------------------------------- deps
def _deps_ready() -> bool:
    try:
        import torch
        base = torch.__version__.split("+", 1)[0]
        print(f"  [_deps_ready] torch.__version__ = {torch.__version__!r}")
        if not base.startswith("2.11."):
            return False
        from torchao.quantization import Granularity  # noqa: F401
        from transformers.modeling_utils import AttentionInterface  # noqa: F401
        from litert_torch.generative.export_hf.export import export  # noqa: F401
        return True
    except Exception as exc:
        print(f"  [_deps_ready] import failed: {type(exc).__name__}: {exc}")
        return False


if not _deps_ready():
    print("Installing pinned dependencies (~3 min) ...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet", "--upgrade", *PINS,
    ])
    print()
    print("=" * 64)
    print("DEPENDENCIES INSTALLED.")
    print("Now do:  Colab menu -> Runtime -> Restart runtime")
    print("Then re-run this same cell. Merge+export will proceed automatically.")
    print("=" * 64)
    raise SystemExit(0)

# ---------------------------------------------------------------- main
import torch
from transformers import Gemma4ForConditionalGeneration
from peft import PeftModel, LoraConfig

print(f"torch={torch.__version__}  cuda={torch.cuda.is_available()}  ready.\n")

# Mount Drive (Colab will prompt for auth on first run).
if not DRIVE_ROOT.exists():
    print("Mounting Google Drive ...")
    from google.colab import drive
    drive.mount(str(DRIVE_MOUNT))

# Find the LoRA. Look in a few sensible Drive locations.
LORA_CANDIDATES = [
    DRIVE_ROOT / "gemma_ksl_lora",
    DRIVE_ROOT / "sema" / "gemma_ksl_lora",
    DRIVE_ROOT / "models" / "gemma_ksl_lora",
    WORK_DIR / "gemma_ksl_lora",
]
lora_dir = next((p for p in LORA_CANDIDATES
                 if (p / "adapter_model.safetensors").exists()), None)
if lora_dir is None:
    # Last resort: search Drive for it.
    for found in DRIVE_ROOT.rglob("adapter_model.safetensors"):
        if (found.parent / "adapter_config.json").exists():
            lora_dir = found.parent
            break
assert lora_dir is not None, (
    "Could not find gemma_ksl_lora in Drive. Put the 7 LoRA files under "
    "MyDrive/gemma_ksl_lora/ and re-run.")
print(f"LoRA dir: {lora_dir}")

# Skip merge if a complete checkpoint is already on disk.
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
    # Stock PEFT can't graft onto Gemma 4's Gemma4ClippableLinear wrapper
    # (used in the vision/audio towers). Skip them — text-only inference
    # doesn't need those branches.
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

# Export
final_local = WORK_DIR / f"{OUTPUT_NAME}.litertlm"
final_drive = DRIVE_ROOT / f"{OUTPUT_NAME}.litertlm"
if final_local.exists() or final_drive.exists():
    existing = final_drive if final_drive.exists() else final_local
    print(f"\n[5-6] {existing} already exists "
          f"({existing.stat().st_size/1024/1024:.1f} MB). Done.")
else:
    print(f"\n[5] Exporting to .litertlm (INT4 dynamic, text-only) ...")
    from litert_torch.generative.export_hf.export import export as run_export
    t = time.time()
    run_export(
        model=str(MERGED_DIR),
        output_dir=str(WORK_DIR),
        quantization_recipe="dynamic_wi4_afp32",
        export_vision_encoder=False,     # text-only path
        externalize_embedder=True,       # required for Gemma 4 (PLE arch)
        bundle_litert_lm=True,           # single-file .litertlm artefact
    )
    print(f"    export finished in {time.time()-t:.0f}s")

    cands = sorted(WORK_DIR.glob("*.litertlm"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    assert cands, "Export claimed success but no .litertlm landed."
    src = cands[0]
    if src != final_local:
        src.rename(final_local)
    size_mb = final_local.stat().st_size / 1024 / 1024
    print(f"\n[6] artefact: {final_local}  ({size_mb:.1f} MB)")

    # Copy to Drive so the user can download from anywhere.
    print(f"    Copying to {final_drive} ...")
    shutil.copy2(final_local, final_drive)
    print(f"    Done. Download from Google Drive and drop into")
    print(f"    mobile-app/sema/sema/Resources/")

    # Free disk: 15 GB merged checkpoint not needed anymore.
    shutil.rmtree(MERGED_DIR, ignore_errors=True)

print("\nDone.")
