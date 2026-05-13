# recognition/

**Path A, stage 1** — signs to gloss tokens.

This folder owns the **gloss tagger**: a small temporal model that consumes MediaPipe-equivalent landmark sequences and emits a stream of KSL gloss tokens. It is **not** Gemma. Per the architecture in [../README.md](../README.md), gloss recognition is a temporal-classification problem and is handled by a model class suited to that — a Transformer encoder over landmark sequences, with an LSTM fallback.

The MediaPipe Holistic pose extraction itself runs in the iOS client (see `../mobile-app/`); this folder is concerned with **training, evaluation, and export** of the model that consumes those landmarks.

## What this folder produces

- A trained gloss-tagger checkpoint (PyTorch).
- A **CoreML `.mlpackage`** exported via `coremltools`, bundled into the iOS app.
- (Optional) a LiteRT `.tflite` export retained as Android-fallback optionality.
- A label map (gloss vocabulary) shared with `../gemma-glossing/` so the downstream translator sees the same token space.

## Input feature space

The iOS client's MediaPipe Tasks output is normalized and consumed directly by the model — no runtime adapter. Training-time features are synthesized from Motion-S BVH skeletons projected into the **same** joint layout, then normalized identically:

- Joint set: 15 body joints (eyes, shoulders, elbows, wrists, hips, knees, ankles) + 16 per hand (wrist + 15 finger segments) × 2 hands = **47 joints**.
- Per-joint coordinates: `(x, y, z)` after normalization — shoulder-midpoint origin, shoulder-width unit scale; `x,y` are image-plane, `z` is relative depth.
- Per-frame feature vector: `47 × 3 = 141` floats.

This is a *normalized invariant* representation. The recognizer never sees raw camera-space pixels; the iOS client applies the same normalization to live MediaPipe output at inference time.

## Augmentation contract

To narrow the sim2real gap between BVH-projected training features and real MediaPipe iOS output, the training Dataset wraps every clip with `data/augment.py`:

| Augmentation | Magnitude | Rationale |
|---|---|---|
| Per-joint Gaussian jitter | ~σ = 0.005 in normalized units (~ MediaPipe 4 px std at 720p) | matches observed MediaPipe localisation noise |
| Random landmark dropout | 5% body, 10–15% hands (zero-fill + dropout-mask channel) | MediaPipe hand detector often drops frames |
| Random temporal masking | 0–8 frames per clip | regularization; mimics tracking glitches |
| Random global affine | small rotation ±10°, translation ±0.05, scale ±5% | camera framing variance |
| Framerate jitter | drop every n-th frame, n ∈ {0,2,3} | training fps ≠ inference fps |

Augmentations apply during training only; eval runs with the augmentation off.

## Intended layout

```
recognition/
├── README.md
├── requirements.txt                    # core: torch, numpy, pandas, pyyaml, tqdm
├── requirements-export-coreml.txt      # adds coremltools (installs on macOS)
├── requirements-export-litert.txt      # optional: ai-edge-torch + tensorflow (Linux/Py>=3.11)
├── configs/
│   ├── transformer_base.yaml           # primary architecture
│   └── lstm_fallback.yaml              # SLRNet-style baseline
├── data/
│   ├── build_vocab.py                  # parse train.csv → gloss vocab + splits
│   ├── bvh_to_landmarks.py             # BVH skeleton → normalized 47-joint landmark vectors
│   ├── augment.py                      # MediaPipe-noise model (train-only)
│   ├── dataset.py                      # PyTorch Dataset + padded collate (+ optional RVQ-aux targets)
│   ├── vocab/gloss_vocab.json          # generated; shared with ../gemma-glossing/
│   ├── splits/{train,val}.txt          # generated; clip ids per split
│   └── landmarks_meta.json             # generated; joint layout + normalization stats
├── models/
│   ├── transformer_encoder.py          # primary: encoder + CTC head (+ optional RVQ-aux head)
│   └── lstm_tagger.py                  # fallback, SLRNet-style stacked LSTM
├── training/
│   ├── train.py                        # CTC (+ optional aux CE) training loop
│   └── decode.py                       # greedy CTC decode + WER
├── export/
│   ├── to_coreml.py                    # PyTorch → CoreML .mlpackage (primary)
│   └── to_litert.py                    # PyTorch → LiteRT .tflite (retained, optional)
├── eval/
│   └── real_camera_smoke.md            # deployment gate: real MediaPipe iOS clips
├── scripts/
│   └── smoke_train.sh                  # bvh→landmarks → vocab → 200-step smoke run
└── tests/
    └── test_dataset.py                 # shape/dtype sanity
```

## References

The pipeline shape (MediaPipe Holistic → small temporal classifier) follows the Tier-1 references in the root README, with **SLRNet** (Khushi-739) as the primary scaffold. The pose-input modelling follows **Pose-TGCN** from the WLASL repo (Tier 2). See [../README.md](../README.md#reference-implementations) for the full list.

## Inputs and outputs

- **Input:** variable-length sequences of normalized 47-joint landmark vectors, `(T, 141)` float32. The same representation at training (BVH-projected) and at inference (MediaPipe iOS).
- **Output:** a gloss token sequence, e.g. `HOSPITAL TOMORROW I-GO`, produced by CTC over the input sequence.
- **Consumer:** `../gemma-glossing/` (translation) → `../mobile-app/` (TTS).

## Environments

| Environment | Purpose | Notes |
|---|---|---|
| Local macOS, `.venv/` with `requirements.txt` | Editing, smoke tests on CPU, parity checks, **CoreML export** | No CUDA. CTC is not implemented on Apple MPS, so this env is CPU-only. |
| Linux + CUDA, Python ≥ 3.11 | Full training runs | Same `requirements.txt`. Training uses `device = "cuda" if available else "cpu"` — no MPS path is included on purpose. |
| Linux + Python ≥ 3.11 | LiteRT export (optional) | `requirements-export-litert.txt`. `ai-edge-torch` pulls in `torch_xla`, which has no macOS wheels. |

## Quick start (local smoke test)

```bash
cd recognition
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m data.bvh_to_landmarks --limit 50          # subset for smoke
.venv/bin/python -m data.build_vocab                          # writes vocab + splits
.venv/bin/python -m tests.test_dataset                        # shape/dtype check
.venv/bin/python -m training.train --config configs/transformer_base.yaml --smoke
```

The `--smoke` flag runs 200 steps on a small subset and confirms data loading, augmentation, model forward with padding mask, CTC loss, eval/decode, and checkpoint save all work end-to-end. Real runs drop `--smoke` and execute on a CUDA box.

## Export to CoreML (local Mac)

```bash
.venv/bin/pip install -r requirements-export-coreml.txt
.venv/bin/python -m export.to_coreml \
    --ckpt checkpoints/transformer_base/best.pt \
    --out  ../mobile-app/Sema/Models/gloss_tagger.mlpackage \
    --seq-len 256
```

The script wraps the model with a fixed-length input shape, traces with `coremltools.convert`, writes an `.mlpackage`, runs parity vs the PyTorch forward, and writes a `gloss_tagger.vocab.json` sidecar.

## Deployment gate

Before declaring the synthetic-trained recognizer deployable, gate on [`eval/real_camera_smoke.md`](eval/real_camera_smoke.md) — a checklist of real MediaPipe-iOS clips the model must pass.
