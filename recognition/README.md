# recognition/

**Path A, stage 1** — signs to gloss tokens.

This folder owns the **gloss tagger**: a small temporal model that consumes MediaPipe Holistic keypoint sequences and emits a stream of KSL gloss tokens. It is **not** Gemma. Per the architecture in [../README.md](../README.md), gloss recognition is a temporal-classification problem and is handled by a model class suited to that — a Transformer encoder over pose sequences, with an LSTM fallback.

The MediaPipe Holistic pose extraction itself runs in the Android client (see `../mobile-app/`); this folder is concerned with **training, evaluation, and export** of the model that consumes those keypoints.

## What this folder produces

- A trained gloss-tagger checkpoint (PyTorch / TensorFlow).
- A LiteRT (`.tflite`) export bundled with the Android app.
- A label map (gloss vocabulary) shared with `../gemma-glossing/` so the downstream translator sees the same token space.

## Intended layout

```
recognition/
├── README.md
├── requirements.txt
├── configs/
│   ├── transformer_base.yaml      # primary architecture
│   └── lstm_fallback.yaml         # SLRNet-style baseline
├── data/
│   ├── prepare_ksl_pose.py        # ingest KSL Word-Based Pose Dataset
│   ├── mediapipe_extract.py       # offline MediaPipe Holistic on raw video
│   ├── normalize.py               # per-frame keypoint normalisation
│   └── splits/                    # train/val/test gloss ID splits
├── models/
│   ├── transformer_encoder.py     # primary: encoder over keypoint sequences
│   ├── lstm_tagger.py             # fallback, SLRNet-style stacked LSTM
│   └── heads.py                   # CTC / classification heads
├── training/
│   ├── train.py
│   ├── evaluate.py
│   └── callbacks.py
├── transfer/
│   └── wlasl_pose_tgcn_init.py    # optional WLASL pose-pretrained init
│                                  # (license check required, see root README)
├── export/
│   ├── to_litert.py               # PyTorch/TF → LiteRT INT8
│   └── verify_parity.py           # numerical parity vs reference run
├── notebooks/
│   └── exploration.ipynb
└── tests/
    └── test_pipeline.py
```

## References

The pipeline shape (MediaPipe Holistic → small temporal classifier) follows the Tier-1 references in the root README, with **SLRNet** (Khushi-739) as the primary scaffold. The pose-input modelling follows **Pose-TGCN** from the WLASL repo (Tier 2). See [../README.md](../README.md#reference-implementations) for the full list.

## Inputs and outputs

- **Input:** sequences of 543 MediaPipe Holistic landmarks per frame, length ~30 frames.
- **Output:** a gloss token sequence, e.g. `HOSPITAL TOMORROW I-GO`.
- **Consumer:** `../gemma-glossing/` (translation) → `../mobile-app/` (TTS).
