# Sema — Repository Structure

This file is the map of the repo. Each top-level folder owns one piece of the architecture described in [README.md](README.md). The split mirrors the design principle: each sub-task is routed to the model class it is actually suited to, and each folder is the home of exactly one such sub-task plus the glue around it.

```
sema-mobile-app/
├── README.md              # Architecture, pipelines, references, datasets
├── STRUCTURE.md           # (this file) repository map
│
├── recognition/           # Path A, stage 1: signs → gloss tokens
│   │                      # MediaPipe Holistic → small temporal model
│   │                      # Trains and exports the on-device gloss tagger
│   └── README.md
│
├── gemma-glossing/        # Path A stage 2 + Path B stage 2 (shared model)
│   │                      # Gemma 4 E4B fine-tuned bidirectionally on
│   │                      # KSL gloss ↔ English/Swahili sentence pairs
│   └── README.md
│
├── generation/            # Path B, stage 3: gloss → animated signing avatar
│   │                      # ASR glue + stickman renderer over the
│   │                      # KSL Word-Based Pose Dataset
│   └── README.md
│
└── mobile-app/            # Android client. Orchestrates camera, mic,
    │                      # MediaPipe, LiteRT models, TTS, and renderer.
    └── README.md
```

## How the folders compose the two pipelines

### Path A — KSL recognition (Deaf → hearing)

```
[camera frames]
        │
        ▼
mobile-app/ ── runs MediaPipe Holistic on-device
        │
        ▼
recognition/ ── gloss tagger (LiteRT) classifies pose sequence → gloss tokens
        │
        ▼
gemma-glossing/ ── Gemma 4 E4B (LiteRT INT4) translates gloss → fluent EN/SW
        │
        ▼
mobile-app/ ── Android system TTS speaks the sentence
```

### Path B — Speech generation (hearing → Deaf)

```
[microphone audio]
        │
        ▼
mobile-app/ ── Whisper-tiny / Android STT produces text
        │
        ▼
gemma-glossing/ ── Gemma 4 E4B translates EN/SW → KSL gloss sequence
        │
        ▼
generation/ ── stickman renderer looks up each gloss in the KSL Pose
        │      Dataset, stitches the pose sequences, plays the animation
        ▼
mobile-app/ ── displays the animated signing avatar
```

## Ownership at a glance

| Concern | Folder | Notes |
|---|---|---|
| Pose extraction (MediaPipe) | `mobile-app/` | Runs on-device; no training artefact |
| Gloss tagger (training + export) | `recognition/` | Transformer over keypoints; LSTM fallback |
| Gloss ↔ sentence translator | `gemma-glossing/` | Single model, both directions, task tokens |
| ASR (Whisper / system STT) | `mobile-app/` | Integration only; no training here |
| Stickman renderer + pose library | `generation/` | Retrieval + stitching; no motion synthesis |
| TTS | `mobile-app/` | Android system TTS |
| App orchestration / UI | `mobile-app/` | Kotlin / Android |

## A note on what is *not* in this repo

- Raw datasets (Motion-S, KSL Word-Based Pose Dataset) are external. Each folder's README explains how to acquire and stage them.
- Pre-trained third-party weights (WLASL Pose-TGCN, Gemma 4 E4B base) are downloaded by the training scripts; they are not committed.
