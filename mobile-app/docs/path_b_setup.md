# Path B — Xcode setup checklist

Path B's Swift code is in place. This file lists the steps to get a working build with the real microphone path and the avatar driven by the pose library.

## 1. Resources bundled

The synchronized folder picks these up automatically once they exist on disk:

| Path | Source | Notes |
|---|---|---|
| `Sema/Resources/PoseLibrary/index.json` + `clips/*.npz` | `generation/pose_library/build_index.py` | regenerate after every recognition retrain to keep the gloss vocab aligned |
| `Sema/Resources/pose_landmarker_full.task` | already there from Path A | |
| `Sema/Resources/hand_landmarker.task` | already there from Path A | |
| `Sema/Models/gemma4_e4b_int4.tflite` (optional, future) | `gemma-glossing/export/to_litert.py` | not required for v0 — Path B uses the stub translator |

To regenerate the pose library:

```bash
recognition/.venv/bin/python generation/pose_library/build_index.py
```

This writes to both `generation/pose_library/` and `mobile-app/sema/sema/Resources/PoseLibrary/`.

## 2. Info.plist permission strings

Path B adds two keys on top of Path A's camera key:

| Key | Suggested value |
|---|---|
| `NSMicrophoneUsageDescription` | "Sema converts your speech into sign-language animation." |
| `NSSpeechRecognitionUsageDescription` | "Sema recognises what you say so it can translate it to KSL signs." |

## 3. Wire `PathBCoordinator` to the existing UI

`PipelineCoordinator` still runs the original demo loop. To swap Path B in:

```swift
@State private var pathB = PathBCoordinator(translatorMode: .stub)

.onAppear {
    pathB.bootstrap()
}
.onChange(of: pathB.state) { _, new in
    if new == .ready { pathB.start() }
}

// In CallStageView, replace the DemoPoseFrame argument:
CallStageView(frame: pathB.player.currentFrame, isActive: pathB.state == .running)
```

The transcript shows up at `pathB.spokenText`; the active gloss list at `pathB.glossStream`.

## 4. Translator modes

Set the mode at construction time:

```swift
// (a) Demo phrasebook — works offline, ~8 phrases.
PathBCoordinator(translatorMode: .stub)

// (b) Server fallback — points at a hosted Gemma endpoint matching
//     gemma-glossing/README.md's contract.
PathBCoordinator(translatorMode: .server(URL(string: "https://gemma.sema.example/translate")!, token: "..."))

// (c) On-device LiteRT (future) — requires gemma4_e4b_int4.tflite bundled
//     and the LiteRT SPM dep added. Currently throws onDeviceNotWired.
PathBCoordinator(translatorMode: .onDevice)
```

## What works today

- The phrasebook stub maps the README's example phrase ("I am going to the hospital tomorrow.") to `TOMORROW HOSPITAL I-GO`.
- The PoseDatabase reads `.npz` clips from the bundle and dequantises int8 → float32 on demand.
- The Stitcher concatenates with an 8-frame linear-blend handoff between adjacent glosses.
- The avatar bridge feeds the existing `AvatarCanvasView` without changing it.
- SFSpeechRecognizer streams a live transcript; the TTS gate pauses ASR while the synthesizer is speaking.

## Known limitations

- **Pose library coverage** depends on how many landmark files exist on disk. The current build is from the 200 dev-time clips; only ~hundreds of gloss tokens have clips. The script silently skips tokens with no candidate ≥ 6 frames.
- **Equal-slicing** is the v0 alignment. Some clips are visually wrong for mid-utterance gloss tokens. Smart slicing (CTC-forced alignment) is a follow-up after recognizer training completes.
- **Server fallback** has no implementation yet — the contract is documented and the request shape is built, but no Sema-hosted endpoint exists.
- **On-device Gemma** is stubbed out; throws `onDeviceNotWired` when chosen. Wire in once `gemma4_e4b_int4.tflite` is exported.
- **ASR rotation strategy** is v0 (auto-restart on final), not the full 50-second-overlap rotation from `generation/asr/contract.md`.

## File map (Swift source under `mobile-app/sema/sema/`)

```
Generation/
├── PoseClip.swift           Decoded (T, 45, 3) float32 + metadata
├── PoseDatabase.swift       LRU-cached lookup; .npz reader (parses ZIP + .npy)
├── _SemaInflate.swift       Compression-framework wrapper for the zip reader
├── Stitcher.swift           Gloss → continuous PoseFrame with 8-frame blends
└── AvatarStreamPlayer.swift Bridges normalised frames to DemoPoseFrame for the existing renderer

Speech/
├── AudioSession.swift       Shared AVAudioSession configurator
├── TTSGate.swift            Shared "TTS speaking?" flag for ASR pausing
├── Synthesizer.swift        AVSpeechSynthesizer + delegate, flips the gate
└── ContinuousSpeechRecognizer.swift  SFSpeechRecognizer with v0 auto-restart

ML/
└── GemmaTranslator.swift    Stub | server | on-device(TODO) modes

Pipelines/
└── PathBCoordinator.swift   Wires mic → ASR → Gemma → Stitcher → AvatarStreamPlayer
```
