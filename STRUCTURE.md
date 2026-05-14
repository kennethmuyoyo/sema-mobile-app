# Sema — Repository Structure

This file is the map of the repo. Each top-level folder owns one piece of the architecture described in [README.md](README.md). The split mirrors the design principle: each sub-task is routed to the model class it is actually suited to, and each folder is the home of exactly one such sub-task plus the glue around it.

```
sema-mobile-app/
├── README.md              # Architecture, pipelines, references, datasets
├── STRUCTURE.md           # (this file) repository map
├── dvc.yaml               # Pipeline DAG (landmarks → vocab → train → export)
├── docs/
│   └── data.md            # DVC + Cloudflare R2 team-onboarding flow
│
├── recognition/           # Path A, stage 1: signs → gloss tokens
│   │                      # MediaPipe-equivalent landmarks → small temporal model
│   │                      # Trains and exports the on-device gloss tagger (CoreML)
│   └── README.md
│
├── gemma-glossing/        # Path A stage 2 + Path B stage 2 (shared model)
│   │                      # Gemma 4 E4B fine-tuned bidirectionally on
│   │                      # KSL gloss ↔ English/Swahili sentence pairs (LiteRT)
│   └── README.md
│
├── generation/            # Path B, stage 3: gloss → animated signing avatar
│   │                      # Pose-clip retrieval, stitching, and the
│   │                      # SwiftUI-renderer contract consumed by mobile-app/
│   └── README.md
│
└── mobile-app/            # iOS client (SwiftUI). Orchestrates camera, mic,
    │                      # MediaPipe iOS, CoreML recognizer, LiteRT Gemma,
    │                      # AVSpeechSynthesizer, and the avatar renderer.
    └── README.md
```

## How the folders compose the two pipelines

### Path A — KSL recognition (Deaf → hearing)

```
[iPhone front camera]
        │
        ▼
mobile-app/ ── runs MediaPipe Tasks for iOS (Holistic)
        │
        ▼
recognition/ ── gloss tagger (CoreML .mlpackage) classifies landmark sequence → gloss tokens
        │
        ▼
gemma-glossing/ ── Gemma 4 E4B (LiteRT INT4) translates gloss → fluent EN/SW
        │
        ▼
mobile-app/ ── AVSpeechSynthesizer speaks the sentence
```

### Path B — Speech generation (hearing → Deaf)

```
[iPhone microphone]
        │
        ▼
mobile-app/ ── SFSpeechRecognizer (continuous) produces text
        │
        ▼
gemma-glossing/ ── Gemma 4 E4B translates EN/SW → KSL gloss sequence
        │
        ▼
generation/ ── on-device pose DB returns per-gloss clips; stitcher
        │      produces a continuous keypoint stream
        ▼
mobile-app/ ── SwiftUI / SpriteKit skeleton renders the animated avatar
```

## Ownership at a glance

| Concern | Folder | Notes |
|---|---|---|
| Pose extraction (MediaPipe Tasks iOS) | `mobile-app/` | Runs on-device; no training artefact |
| Gloss tagger (training + export) | `recognition/` | Transformer over landmark vectors; LSTM fallback. CoreML `.mlpackage` is the primary export; `to_litert.py` retained as Android-fallback optionality |
| Gloss ↔ sentence translator | `gemma-glossing/` | Single model, both directions, task tokens. LiteRT `.tflite` (Google AI Edge) |
| ASR (SFSpeechRecognizer continuous) | `mobile-app/` | Integration only; no training here |
| Pose DB + stitching | `generation/` | Retrieval over per-gloss clips bundled in-app, int8-quantized |
| Avatar renderer | `mobile-app/` (impl) + `generation/` (contract) | SwiftUI / SpriteKit skeleton |
| TTS (AVSpeechSynthesizer) | `mobile-app/` | Apple-native, no extra model |
| App orchestration / UI | `mobile-app/` | Swift / SwiftUI / Xcode |

## Where the large files live

- **Cloudflare R2** via **DVC** (`dvc.yaml` + `.dvc` pointer files). See [`docs/data.md`](docs/data.md) for the onboarding flow. `dvc pull` populates `data/`, `recognition/checkpoints/`, `mobile-app/Sema/Models/`, and `mobile-app/Sema/PoseLibrary/` from R2.
- Pre-trained third-party weights (WLASL Pose-TGCN, Gemma 4 E4B base, Google AI Edge published Gemma `.tflite`) are downloaded by the training/export scripts; they are not tracked in either git or DVC.
- The iOS app's `Sema.xcodeproj`, build settings, and code-signing config live in `mobile-app/`. They are created on first build, not by the scaffolds in this repo.
