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
│   │                      # MediaPipe-equivalent landmarks → small temporal model.
│   │                      # Trains and exports the on-device gloss tagger (CoreML).
│   │                      # NOTE: data/, training/, and checkpoints/ live on the
│   │                      # training workstation only — not in git (see .gitignore).
│   ├── README.md
│   ├── configs/           # transformer_base.yaml, lstm_fallback.yaml
│   ├── export/            # to_coreml_v11.py — ships the .mlpackage to iOS
│   ├── eval/              # replay_litert.py, replay_bvh.py, smoke checklist
│   ├── tests/             # shape/dtype sanity checks
│   ├── convert_to_litert.py
│   └── BENCHMARK.md
│
├── gemma-glossing/        # Path A stage 2 + Path B stage 2 (shared model).
│   │                      # Gemma 4 E2B fine-tuned bidirectionally on
│   │                      # KSL gloss ↔ English/Swahili sentence pairs.
│   │                      # Weights are produced on Kaggle/Colab and copied
│   │                      # into mobile-app/sema/sema/Resources/ — not in git.
│   ├── README.md
│   ├── merge_lora.py
│   ├── colab_merge_export.py
│   └── kaggle_merge_export.{py,ipynb}
│
├── generation/            # Path B, stage 3: gloss → animated signing avatar.
│   │                      # Pose-clip retrieval, stitching, and the
│   │                      # SwiftUI-renderer contract consumed by mobile-app/.
│   ├── README.md
│   ├── asr/contract.md            # SFSpeechRecognizer continuous-mode contract
│   ├── renderer/ios_contract.md   # keypoint-stream format the iOS renderer consumes
│   ├── llm_export/                # LoRA → LiteRT/TFLite staging (artifacts ignored)
│   └── pose_library/
│       ├── build_index.py         # ingest Motion-S → quantised per-gloss clips
│       ├── build_full_library.py  # alignment-derived library (~406 glosses)
│       ├── build_single_gloss.py  # iterate one gloss at a time
│       └── pick_best_takes.py     # curate exemplars per gloss
│                                  # clips/, full/, rotations/, index.json are
│                                  # generated locally and ignored by git.
│
└── mobile-app/            # iOS client (SwiftUI). Orchestrates camera, mic,
    │                      # MediaPipe iOS, CoreML recognizer, llama.cpp Gemma,
    │                      # AVSpeechSynthesizer, and the SceneKit avatar.
    ├── README.md
    ├── ARCHITECTURE.md            # why each runtime was chosen, privacy story
    ├── docs/
    │   ├── bundle_inventory.md    # every file in the IPA + manual setup steps
    │   ├── coreml_path_a.md       # Path A on-device plan
    │   ├── path_a_setup.md        # Path A Xcode checklist
    │   ├── path_b_avatar.md       # Path B on-device plan
    │   └── path_b_setup.md        # Path B Xcode checklist
    └── sema/
        ├── Podfile                # MediaPipeTasksVision, TensorFlowLiteSwift
        ├── Frameworks/            # llama.xcframework (not in git, ~200 MB)
        ├── sema.xcodeproj/        # Xcode project (committed)
        ├── sema.xcworkspace/      # CocoaPods workspace — always open this
        └── sema/                  # app sources, grouped by concern:
            ├── semaApp.swift          # @main entry point
            ├── ContentView.swift      # root SwiftUI view
            ├── CameraSessionController.swift
            ├── VolumeShortcutDetector.swift
            ├── SimpleAvatar3DView.swift  # SceneKit avatar renderer
            ├── MediaPipe/             # HolisticLandmarker.swift (Tasks iOS wrapper)
            ├── ML/                    # gloss tagger, Gemma engine, decoders,
            │                          # normalisation, prompts, metadata
            ├── Generation/            # PoseDatabase, Stitcher, AvatarStreamPlayer,
            │                          # smoothing filters, BVH rotation layout
            ├── Pipelines/             # PathACoordinator, PathBCoordinator,
            │                          # ConversationOrchestrator (half-duplex)
            ├── Speech/                # AudioSession, ContinuousSpeechRecognizer,
            │                          # AVSpeechSynthesizer wrappers
            ├── Views/                 # ConversationScreenView, Debug/, Components/
            ├── Util/                  # camera mirroring, permissions, test env
            ├── Resources/             # bundled models + pose library (not in git)
            ├── Assets.xcassets/       # AppIcon metadata (PNGs not in git)
            ├── PrivacyInfo.xcprivacy
            └── sema.entitlements
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
| Gloss tagger (training + export) | `recognition/` | Transformer over landmark vectors; LSTM fallback. CoreML `.mlpackage` is the primary export; `convert_to_litert.py` retained as Android-fallback optionality |
| Gloss ↔ sentence translator | `gemma-glossing/` | Single model, both directions, task tokens. Ships as a GGUF for llama.cpp on iOS; LiteRT export is the Android-fallback path |
| ASR (SFSpeechRecognizer continuous) | `mobile-app/` | Integration only; no training here |
| Pose DB + stitching | `generation/` | Retrieval over per-gloss clips bundled in-app, int8-quantized |
| Avatar renderer | `mobile-app/` (impl) + `generation/` (contract) | SceneKit (`SimpleAvatar3DView`) — superseded the SwiftUI Canvas in the original spec |
| TTS (AVSpeechSynthesizer) | `mobile-app/` | Apple-native, no extra model |
| App orchestration / UI | `mobile-app/` | Swift / SwiftUI / Xcode |

## What's in git vs what isn't

The repo carries **code, configs, and contracts** — not models, training data, generated bundles, or rendered art. The split is enforced by `.gitignore` at the root and by per-folder ignore rules.

**Not in git** (regenerated, fetched, or DVC-pulled on each workstation):

| What | Where it lives locally | How to get it |
|---|---|---|
| Training data + landmarks | `data/` | `dvc pull` (Cloudflare R2 via `dvc.yaml`) |
| Recognition pipeline (data prep + training loop + checkpoints) | `recognition/data/`, `recognition/training/`, `recognition/checkpoints/` | Sources kept on the training workstation; `dvc pull` for the checkpoints |
| iOS-bundled models | `mobile-app/sema/sema/Resources/*.gguf`, `*.task`, `*.mlpackage/`, `*.metadata.json` | Produced by `recognition/export/to_coreml_v11.py`, `gemma-glossing/*_merge_export.*`, and one-time downloads (MediaPipe `.task` files) |
| iOS pose library | `mobile-app/sema/sema/Resources/PoseLibrary{,Full}/` | `generation/pose_library/build_*.py` |
| iOS avatar rig | `mobile-app/sema/sema/Resources/*.usdc` + textures | Stored locally; bundled at build time |
| App icons | `mobile-app/sema/sema/Assets.xcassets/AppIcon.appiconset/*.png` | Kept locally per design pass |
| llama.cpp framework | `mobile-app/sema/Frameworks/llama.xcframework/` | Downloaded from <https://github.com/ggml-org/llama.cpp/releases> |
| LoRA conversion intermediates | `generation/llm_export/`, `gemma-glossing/merged_model/` | Re-runnable from the export scripts |

**In git**: every Swift source, every Python source under `recognition/` (excluding the data/training pipeline), `generation/`, `gemma-glossing/`, configs, READMEs, contracts, the Xcode project + workspace, Podfile, dvc.yaml, and the per-folder `.dvc` pointer files.

- See [`docs/data.md`](docs/data.md) for the DVC + Cloudflare R2 onboarding flow.
- See [`mobile-app/docs/bundle_inventory.md`](mobile-app/docs/bundle_inventory.md) for the per-file iOS bundle map and the two manual setup steps for a fresh checkout (CocoaPods + the llama.cpp XCFramework).
