# mobile-app/

**The iOS client.** A SwiftUI app that orchestrates everything that runs on-device: front camera, microphone, MediaPipe Tasks for iOS, the CoreML-exported gloss tagger from `../recognition/`, the LiteRT-exported Gemma 4 E4B from `../gemma-glossing/`, the on-device pose database specified in `../generation/`, `AVSpeechSynthesizer` for TTS, `SFSpeechRecognizer` for ASR, and the SwiftUI/SpriteKit skeleton view that draws the 2D signing avatar.

No training happens here. This folder consumes the artefacts produced by the other three.

## What this folder produces

- An installable iOS app (Sema.ipa) that runs the full Sema demo on-device on iOS 17+.

## UI shape

FaceTime-style, single primary screen:

```
┌──────────────────────────────────────────────┐
│  ●●●               Sema                  ⚙   │  <- nav: settings (history lives here)
│                                              │
│         ┌────────────────────────┐           │
│         │                        │           │
│         │    2D AVATAR CANVAS    │           │  <- SwiftUI/SpriteKit skeleton
│         │   (Path B output)      │           │     drawn from generation/'s stream
│         │                        │           │
│         └────────────────────────┘           │
│                                              │
│  ┌──────┐                                    │
│  │ CAM  │   "I'm going to the hospital..."   │  <- Path A live transcript
│  │ PiP  │                                    │
│  └──────┘                                    │
│                                              │
│          [   ⏺  Action Button   ]            │  <- on-screen: start/pause both paths
└──────────────────────────────────────────────┘
```

Both pipelines run concurrently when the app is foregrounded. The action button is on-screen (no physical Action Button integration — iOS 17+ minimum, all device classes uniformly).

## Intended layout

```
mobile-app/
├── README.md
├── Sema.xcodeproj/                              # created on first build
├── Sema/
│   ├── SemaApp.swift                            # @main App entry
│   ├── ContentView.swift                        # FaceTime-style root view
│   ├── Info.plist                               # camera + mic + speech permission strings
│   ├── Assets.xcassets/
│   ├── Models/
│   │   ├── gloss_tagger.mlpackage               # from ../recognition/export/
│   │   ├── gloss_tagger.vocab.json              # sidecar
│   │   └── gemma4_e4b_int4.tflite               # from ../gemma-glossing/export/
│   ├── PoseLibrary/                             # from ../generation/pose_library/clips/
│   ├── Views/
│   │   ├── AvatarCanvasView.swift               # SwiftUI/SpriteKit skeleton renderer
│   │   ├── CameraPiPView.swift                  # CameraX-equivalent: AVCaptureSession preview
│   │   ├── TranscriptView.swift                 # live Path A transcript
│   │   ├── ActionButton.swift
│   │   └── SettingsView.swift                   # entry to HistoryView
│   ├── Pipelines/
│   │   ├── PathA.swift                          # camera → MediaPipe → CoreML → Gemma → TTS
│   │   ├── PathB.swift                          # mic → ASR → Gemma → PoseDB → avatar
│   │   └── PipelineCoordinator.swift            # concurrent both-paths + thermal observer
│   ├── Capture/
│   │   ├── CameraSession.swift                  # AVCaptureSession (front camera)
│   │   └── AudioSession.swift                   # AVAudioSession config (see below)
│   ├── MediaPipe/
│   │   └── HolisticLandmarker.swift             # MediaPipeTasksVision wrapper
│   ├── ML/
│   │   ├── GlossTagger.swift                    # CoreML runner; normalizes landmarks
│   │   ├── GemmaTranslator.swift                # LiteRT runner (Google AI Edge)
│   │   └── ModelWarmup.swift                    # pre-warm Gemma on background queue
│   ├── Speech/
│   │   ├── ContinuousSpeechRecognizer.swift     # SFSpeechRecognizer with request rotation
│   │   └── Synthesizer.swift                    # AVSpeechSynthesizer wrapper
│   ├── Generation/
│   │   ├── PoseDatabase.swift                   # reads bundled pose clips
│   │   └── Stitcher.swift                       # mirrors generation/stitching/
│   ├── History/
│   │   ├── HistoryStore.swift                   # SwiftData @Model + @Query
│   │   ├── HistoryEntry.swift                   # one interpretation, both directions
│   │   └── HistoryView.swift
│   └── Util/
│       └── Throttle.swift
├── SemaTests/
└── docs/
    ├── permissions.md                           # camera, mic, speech
    └── memory_budget.md                         # see table below
```

## Concurrent paths — audio session contract

The single most common bug class in this UX is microphone-vs-speaker interference between SFSpeechRecognizer (recording) and AVSpeechSynthesizer (playback). The contract:

| Setting | Value |
|---|---|
| `AVAudioSession` category | `.playAndRecord` |
| Mode | `.measurement` (or `.voiceChat` if echo cancellation helps more than the latency hit) |
| Category options | `.allowBluetooth`, `.defaultToSpeaker`, `.allowBluetoothA2DP`, `.mixWithOthers` (deliberate: lets MediaPipe/preview audio coexist) |
| TTS pauses ASR | Yes — `ContinuousSpeechRecognizer` is paused while `AVSpeechSynthesizer.isSpeaking == true`, then resumed after `synthesizer(_:didFinish:)`. |
| Continuous mode rotation | `SFSpeechRecognizer` requests cap at ~1 minute. `ContinuousSpeechRecognizer` rotates requests every 50 s, stitching across the boundary by concatenating final transcripts. |

See [`../generation/asr/contract.md`](../generation/asr/contract.md) for the full ASR contract.

## Memory budget

Target: stable on 6 GB devices (iPhone 13, base iPhone 15). Validate early.

| Component | Resident memory (target) | Notes |
|---|---|---|
| Gemma 4 E4B (LiteRT INT4) | ~3.5 GB | Pre-warmed on background queue at app launch |
| Gloss tagger (CoreML, .mlpackage) | ~50 MB | Runs on Neural Engine where possible |
| MediaPipe Holistic | ~200 MB | One landmarker; reuse across frames |
| Pose database (mmap) | ~100 MB | int8 clips, memory-mapped |
| Camera + preview + buffers | ~200 MB | 720p front camera |
| SwiftUI + app overhead | ~150 MB | |
| Headroom | ~1.8 GB | OS, jetsam buffer |

If `ProcessInfo.thermalState >= .serious`, enter degraded mode: drop MediaPipe to 15 fps, skip Path B avatar repaints when no new gloss is produced, and surface a transient banner.

## Artefacts consumed from sibling folders

| Asset | Comes from | Goes to |
|---|---|---|
| `gloss_tagger.mlpackage` | `../recognition/export/to_coreml.py` | `Sema/Models/` |
| `gloss_tagger.vocab.json` | `../recognition/export/to_coreml.py` | `Sema/Models/` |
| `gemma4_e4b_int4.tflite` | `../gemma-glossing/export/to_litert.py` | `Sema/Models/` |
| Pose clips (`clips/*.npz`) | `../generation/pose_library/build_index.py` | `Sema/PoseLibrary/` |
| Renderer keypoint contract | `../generation/renderer/ios_contract.md` | `Views/AvatarCanvasView.swift` |
| ASR contract | `../generation/asr/contract.md` | `Speech/ContinuousSpeechRecognizer.swift` |
| Landmark joint layout | `../recognition/data/landmarks_meta.json` | `MediaPipe/HolisticLandmarker.swift` + `Generation/PoseDatabase.swift` |

## Out of scope

- Android. The original plan; superseded by iOS-only for the first release.
- A server backend (other than the optional Gemma fallback documented in `../gemma-glossing/README.md`). Sema is on-device by design.
- Physical Action Button integration (iPhone 15 Pro+). All buttons are on-screen; one UI on all iPhones.
- Face landmarks. The recognizer's joint set is body + hands only; non-manual markers (eyebrows, mouth) are a future-work item.
