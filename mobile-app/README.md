# mobile-app/

**The iOS client.** A SwiftUI app for iOS 17+ that interprets Kenyan Sign Language ↔ English/Swahili in both directions, fully on-device. No training happens here — this folder consumes the artefacts produced by `../recognition/`, `../gemma-glossing/`, and `../generation/`.

## What ships

A single `Sema.ipa` that runs both directions on one screen:

- **Sign → Speech** (camera → MediaPipe pose+hand → CoreML gloss tagger → Gemma KSL→EN → `AVSpeechSynthesizer`)
- **Speech → Sign** (mic → on-device `SFSpeechRecognizer` → Gemma EN→KSL → pose-clip library → SceneKit 3D avatar)

Both directions share one camera, one audio session, one Gemma model. The orchestrator (`ConversationOrchestrator`) runs the session **half-duplex** — one sensor live at a time — and the user taps the camera tile to toggle between *listening* (mic) and *watching* (camera).

Everything runs offline. No network calls in the default configuration; see [`ARCHITECTURE.md`](ARCHITECTURE.md) for the privacy/offline story.

## Where to read next

- **[`ARCHITECTURE.md`](ARCHITECTURE.md)** — what the app is, how the pieces fit, the privacy story, the retargeting math, why the LLM is llama.cpp (not Ollama, not LiteRT), why the recognizer is CoreML (not a second TFLite).
- **[`docs/bundle_inventory.md`](docs/bundle_inventory.md)** — every file in the IPA, where it comes from, the two manual setup steps for a fresh checkout (CocoaPods + the llama.cpp XCFramework).
- **[`docs/path_a_setup.md`](docs/path_a_setup.md)** — Path A (sign → speech) Xcode-side checklist.
- **[`docs/path_b_avatar.md`](docs/path_b_avatar.md)** — the original Path B design doc. Most of it still applies; the only material drift is that the renderer is now SceneKit (`SimpleAvatar3DView`), not the SwiftUI `Canvas` originally specced.
- **[`docs/coreml_path_a.md`](docs/coreml_path_a.md)** — CoreML export notes and why we moved off TFLite for the recognizer.

## First-build setup (in order)

```bash
cd mobile-app/sema
pod install            # MediaPipeTasksVision + TensorFlowLiteSwift
open sema.xcworkspace  # always the workspace, never the .xcodeproj
```

Then in Xcode: drag `mobile-app/sema/Frameworks/llama.xcframework` into the **sema** target, embed-and-sign. The full checklist (and what to do when the model artefacts need re-hydrating from `../recognition/` / `../gemma-glossing/` / `../generation/`) is in [`docs/bundle_inventory.md`](docs/bundle_inventory.md).

> **Path A needs a physical device.** MediaPipe can't see camera frames in the iOS Simulator. Path B (mic → avatar) works in the simulator.

## Models bundled in `sema/sema/Resources/`

None of these live in git — they're produced by the Python pipeline (or fetched once) and copied into `Resources/` on the workstation that's building the IPA. The root `.gitignore` blocks every model extension under `Resources/` plus the two `PoseLibrary*` directories defensively. A fresh checkout will not build until you put these files back; see [`docs/bundle_inventory.md`](docs/bundle_inventory.md) for the per-file rehydration steps.

| File | What | From |
|---|---|---|
| `pose_landmarker_full.task` | MediaPipe pose (33 body landmarks) | Google AI Edge, one-time download |
| `hand_landmarker.task` | MediaPipe hand (21 landmarks/hand) | Google AI Edge, one-time download |
| `ksl_model.mlpackage` | CoreML v11 phonological gloss recognizer | `../recognition/export/to_coreml_v11.py` |
| `ksl_model.metadata.json` | Vocab + phonological label spaces sidecar | same export |
| `gemma-4-e2b-ksl-Q4_K_M.gguf` | Gemma 4 E2B fine-tuned bidirectionally (KSL ↔ EN/SW), Q4_K_M | `../gemma-glossing/kaggle_merge_export.ipynb` |
| `PoseLibrary/` | Curated demo pose clips (~31 glosses) | `../generation/pose_library/build_index.py` |
| `PoseLibraryFull/` | Alignment-derived library (~406 glosses) | `../generation/pose_library/build_full_library.py` |
| `hackathon.usdc` + textures | SceneKit avatar rig | bundled, license-permissible |

The GGUF is ~3 GB; that pushes the IPA past the 4 GB App Store binary limit. Two production options (download-on-first-launch vs. re-quantising to a smaller variant) are discussed in [`docs/bundle_inventory.md`](docs/bundle_inventory.md#production-sizing-app-store-path). For TestFlight and dev builds the full bundle ships as-is.

App icons (`sema/sema/Assets.xcassets/AppIcon.appiconset/SemaIcon*.png`) are also kept locally per design pass and excluded from git; the asset-catalog `Contents.json` is tracked so Xcode still resolves the icon set.

## Current state

- Both pipelines run end-to-end on iPhone 14 / 15-class hardware.
- The shipping default for the LLM is `.stub` (phrasebook covering the Hospital + Bank demo scenarios) because the bundled Q4_K_M GGUF is unstable under simultaneous bring-up with MediaPipe + SceneKit. Switching to fully on-device Gemma is a one-line change in `Pipelines/ConversationOrchestrator.swift` once the model is re-quantised smaller; see [`ARCHITECTURE.md`](ARCHITECTURE.md#5-gemma-on-device-via-llamacpp).
- The recogniser checkpoint is the v11 phonological model. Accuracy on the iOS input distribution is improving as more real-camera data lands; the design includes a deterministic `PoseTemplateMatcher` fallback during demos.

## Out of scope

- Android (iOS-only for the first release).
- A server backend. The `.server` Gemma fallback path exists in code but is off by default and not required for any feature.
- Physical Action Button integration. All controls are on-screen.
- Face landmarks. The recognizer's joint set is body + hands only; non-manual markers are future work.
