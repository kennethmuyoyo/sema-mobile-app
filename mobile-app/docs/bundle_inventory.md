# Bundle inventory & first-build setup

If you just cloned this repo and want the iOS app to build, this is the doc.
It catalogues every file the running app loads at startup, where each one
comes from, and the two manual steps Xcode can't do on its own.

## Manual setup, in order

These are required exactly once per fresh checkout. The Xcode project itself
is committed; only the package dependencies and the heavy model artefacts
need to be hydrated locally.

### 1. CocoaPods — landmarker + recognizer runtimes

```bash
cd mobile-app/sema
pod install
```

This produces `sema.xcworkspace` (open this, **not** the `.xcodeproj`) and
fetches:

- **MediaPipeTasksVision** — `PoseLandmarker` + `HandLandmarker` used by
  `MediaPipe/HolisticLandmarker.swift`.
- **TensorFlowLiteSwift** — the LiteRT runtime used by
  `ML/LiteRTGlossTagger.swift` to run the v11 sign recognizer.


### 2. Swift Package — on-device Gemma

In Xcode (`sema.xcworkspace` open):

1. **File → Add Package Dependencies…**
2. URL: `https://github.com/google-ai-edge/LiteRT-LM.git`
3. **Dependency Rule → Exact Version → `0.11.0`**
4. **Add Package**
5. In the "Choose package products" sheet, tick **LiteRTLM** and add it to
   the **sema** target.

`ML/LiteRTLMEngine.swift` uses `#if canImport(LiteRTLM)` so the project still
compiles before this step — but `GemmaTranslator.onDevice` will throw at
runtime with a self-describing error message until the package is linked.

### 3. (Optional) Real device

Path A (camera) won't run in the iOS Simulator because MediaPipe needs a
camera feed. Path B (mic → avatar) does run in the simulator. Pick "My Mac
(Designed for iPad)" or any physical device to use Path A.

## What ships inside the app bundle

Every file below lives under `mobile-app/sema/sema/Resources/`. Xcode's
file-system-synchronised group picks them up automatically — no `pbxproj`
membership entries to maintain.

| File | Size | What loads it | What it does |
|---|---|---|---|
| `pose_landmarker_full.task` | 9.0 MB | `MediaPipe/HolisticLandmarker.swift` | MediaPipe pose model (33 body landmarks per frame) |
| `hand_landmarker.task` | 7.5 MB | `MediaPipe/HolisticLandmarker.swift` | MediaPipe hand model (21 landmarks per hand) |
| `ksl_model.float.tflite` | 7.4 MB | `ML/LiteRTGlossTagger.swift` | v11 phonological recognizer, exported from `recognition/convert_to_litert.py` |
| `ksl_model.metadata.json` | 3.0 MB | `ML/LiteRTGlossTagger.swift` | Vocabularies + phonological label spaces for the recogniser |
| `gemma-4-e2b-ksl.litertlm` | 4.7 GB | `ML/LiteRTLMEngine.swift` | Gemma 4 E2B fine-tuned bidirectionally on KSL ↔ EN |
| `PoseLibrary/index.json` | ~50 KB | `Generation/PoseDatabase.swift` | Per-gloss clip manifest — built by `generation/pose_library/build_index.py` |
| `PoseLibrary/clips/*.npz` | ~5 MB total | `Generation/PoseDatabase.swift` | One INT8-quantised pose clip per gloss (the avatar plays these) |
| `PoseLibrary/rotations/` | varies | `Generation/PoseDatabase.swift` | Optional per-joint rotation streams used by the 3D rig |
| `PoseLibrary/textures/`, `hackathon.usdc` | varies | `SimpleAvatar3DView.swift` | Avatar mesh + textures for the SceneKit|
| `demo_recognition.json` | 4 KB | `Pipelines/PathACoordinator.swift` | Allowlist of glosses for the sign-to-speech demo scenarios. Setting empty `glosses` disables the allowlist and rerelies on the full vocab. |
| `demo_generation.json` | 4 KB | `Pipelines/PathBCoordinator.swift` | Per-scenario gloss manifest for the speech-to-sign demo |


When the recogniser or the LLM is retrained, these are the canonical commands
to refresh the bundle:

| File | Where it comes from |
|---|---|
| `pose_landmarker_full.task`, `hand_landmarker.task` | Downloaded from Google AI Edge once; not regenerated locally. |
| `ksl_model.float.tflite` + `.metadata.json` | `recognition/convert_to_litert.py --checkpoint ksl_model.pt --out ksl_model` — produces both files. The PyTorch `.pt` comes from `recognition/train_kaggle.ipynb`. |
| `gemma-4-e2b-ksl.litertlm` | `gemma-glossing/kaggle_merge_export.ipynb` — merges the LoRA into base Gemma 4 E2B and exports the INT4 `.litertlm`. Output is downloaded from Kaggle's output panel. |
| `PoseLibrary/` | `generation/pose_library/build_index.py` (manifest) + `generation/pose_library/retarget_to_target.py` (clip data). |
| `demo_recognition.json`, `demo_generation.json` | Hand-edited. Versioned alongside the demo scripts. |

## Production sizing (App Store path)

`gemma-4-e2b-ksl.litertlm` is 4.7 GB — larger than the 4 GB compressed iOS
App Store binary limit. Two production options:

1. **Download on first launch.** Strip the `.litertlm` from `Resources/`,
   host it on a CDN, fetch on first launch into `Documents/`. `LiteRTLMEngine`
   already prefers a path inside the app's caches dir — adding a Documents
   fallback is ~20 lines of Swift. Pros: small App Store binary. Cons: 4.7 GB
   first-launch download, user has to be on Wi-Fi.
2. **Smaller model.** Re-fine-tune against a smaller base (Gemma 4 E2B at
   INT4 weight-only quant comes in around 1.3 GB; Phi-3.5-mini or Llama 3.2
   1B even smaller). Requires retraining; see `gemma-glossing/README.md` for
   the LoRA fine-tune pipeline. Pros: fully bundled, no download UX. Cons:
   quality regression vs. the current E2B fine-tune is unknown until measured.

For now (May 2026), TestFlight + dev builds ship the full bundle. The
download path is a follow-up scoped after the demo.
