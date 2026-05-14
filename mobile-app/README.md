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

## Avatar renderer — decision and rationale

The signing avatar in `Views/AvatarCanvasView.swift` uses a **SwiftUI `Canvas` humanized 2D renderer** driven directly from the keypoint stream produced by `Generation/Stitcher.swift`. This decision was made after evaluating three options:

| Option | 3D? | Offline | Build time | Pose-stream driven | Decision |
|---|---|---|---|---|---|
| **SwiftUI Canvas (chosen)** | 2D | ✅ | 1–2 days | ✅ Native | ✅ **Ship this** |
| RealityKit + Mixamo USDZ | ✅ | ✅ | 4–7 days | ⚠️ Requires custom kinematics solver | Future upgrade |
| Rive | 2D + authored states | ✅ | Fast to integrate, slow to author | ❌ Cannot inject raw pose streams | Eliminated |

### Why 2D Canvas wins for v1

- **Your data is already 2D-plus-z keypoints** — the pose clips from Motion-S are MediaPipe-equivalent normalized coordinates. The Canvas renders them directly with zero coordinate-transform overhead.
- **Sign accuracy lives in the joints, not the skin mesh.** The renderer uses the 47-joint body-and-hand contract from `../generation/renderer/ios_contract.md`: 15 body landmarks plus 16 landmarks per hand. This keeps all finger segments explicit while avoiding the unused face landmarks from raw MediaPipe output.
- **RealityKit `SkeletalPosesComponent`** (the correct API for frame-by-frame joint injection) requires **iOS 18.0+ minimum**, a non-trivial Mixamo→USDZ asset pipeline (finger bones frequently break during FBX→USDZ conversion and need Blender fixes), and a custom Swift kinematics solver to convert MediaPipe positions to bone quaternions (no Swift equivalent of Kalidokit exists). Estimated 4–7 days of work.
- **Rive** is a state-machine-driven authoring tool — it is not designed to ingest a continuous external pose stream. Driving it from raw keypoint data requires bypassing its animation loop, which conflicts with the state machine. Eliminated.

### Canvas renderer implementation sketch

```swift
// AvatarCanvasView.swift — driven by Stitcher.currentFrame: [SIMD2<Float>] (47 joints)
Canvas { context, size in
    // 1. Humanized body source geometry
    for (a, b) in PoseSkeleton.bodyEdges {
        var p = Path()
        p.move(to: frame[a].cgPoint(in: size))
        p.addLine(to: frame[b].cgPoint(in: size))
        context.stroke(p, with: .color(.white.opacity(0.7)), lineWidth: 2)
    }
    // 2. Hand geometry (cyan, slightly thicker — sign meaning lives here)
    for (a, b) in PoseSkeleton.handEdges {
        var p = Path()
        p.move(to: frame[a].cgPoint(in: size))
        p.addLine(to: frame[b].cgPoint(in: size))
        context.stroke(p, with: .color(.cyan), lineWidth: 2.5)
    }
    // 3. Joints — hands larger and brighter to anchor viewer's eye
    for (i, pt) in frame.enumerated() {
        let isHand = i >= 15   // body=0..14, left hand=15..30, right hand=31..46
        let r: CGFloat = isHand ? 5 : 3
        let color: Color = isHand ? .cyan : .white
        context.fill(
            Circle().path(in: CGRect(x: pt.cgPoint(in: size).x - r,
                                     y: pt.cgPoint(in: size).y - r,
                                     width: r*2, height: r*2)),
            with: .color(color)
        )
    }
}
.background(.black)
```

### Upgrade path to 3D (post-hackathon)

When time permits, `AvatarCanvasView.swift` can be swapped for a `RealityView`-based renderer that drives a Mixamo USDZ character via `SkeletalPosesComponent`. The `Stitcher` output contract (47 joints × `SIMD3<Float>` per frame) remains identical — only the renderer changes. Minimum iOS 18.0 would be required at that point.

### Offline avatar runtime note

Hosted sign-avatar systems such as Signapse and Hand Talk are useful product references for what a human signer should feel like, but they are not the core Sema runtime path because this app is designed to work fully offline and must render KSL from Sema's own gloss and pose pipeline. CWASA/JASigning and MMS Player are useful research references for sign synthesis, timing, and inflection, but they are web/desktop-generation stacks rather than a native iOS renderer.

The mobile app therefore keeps the renderer on-device and pose-stream driven. The current implementation uses landmarks as an invisible control rig for a filled signer silhouette: head, torso, rounded arms, palms, finger shapes, and hand trails. It avoids visible joint dots and bone lines while preserving the same 47-joint contract that a later Unity or RealityKit renderer would consume.

### Future 3D avatar spike

The next true-3D spike should be isolated from the TestFlight app shell:

1. Bundle one neutral humanoid rig with full finger bones and a permissive license.
2. Build a 47-joint playback fixture from `DemoPoseFrame` and exported pose-library clips.
3. Convert each landmark frame into bone rotations for shoulders, elbows, wrists, palms, and finger segments.
4. Test retargeting in Unity first, because its humanoid avatar and animation tooling are more mature for fingers than a hand-built native solver.
5. Reuse the exact same pose stream as `AvatarCanvasView`; only the renderer changes.

Success criteria: a bundled offline avatar can play the same gloss clip as the SwiftUI renderer, with readable finger shapes, no network calls, stable 30 fps or better on device, and a clear fallback to the SwiftUI renderer if the 3D runtime misses performance or signing-readability targets.


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
│   │   ├── AvatarCanvasView.swift               # SwiftUI Canvas humanized avatar renderer
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

## Build priority order

Everything below needs to be written from scratch. Ordered by dependency — build top to bottom.

### Priority 1 — Core pipeline (must ship for demo)

| File | What it does | Key dependency |
|---|---|---|
| `Capture/CameraSession.swift` | `AVCaptureSession` front camera, 720p, delivers `CMSampleBuffer` | None |
| `Capture/AudioSession.swift` | Configures `AVAudioSession .playAndRecord` contract | None |
| `MediaPipe/HolisticLandmarker.swift` | Wraps `MediaPipeTasksVision` holistic landmarker; projects raw body/hand output into the 47-joint runtime contract | CameraSession |
| `ML/GlossTagger.swift` | CoreML runner for `.mlpackage`; applies torso-anchored normalization from `landmarks_meta.json` | HolisticLandmarker |
| `ML/GemmaTranslator.swift` | LiteRT runner via Google AI Edge SDK; task-token dispatch (`[KSL→EN]` etc.) | GlossTagger |
| `ML/ModelWarmup.swift` | Pre-warms Gemma on a background queue at app launch | GemmaTranslator |
| `Speech/Synthesizer.swift` | `AVSpeechSynthesizer` wrapper; posts `isSpeaking` changes | AudioSession |
| `Speech/ContinuousSpeechRecognizer.swift` | `SFSpeechRecognizer` with 50 s request rotation; pauses while TTS speaks | AudioSession, Synthesizer |
| `Generation/PoseDatabase.swift` | Loads bundled `.npz` clips, memory-maps them, exposes `clip(for gloss:)` | None |
| `Generation/Stitcher.swift` | Concatenates clips with linear interpolation between glosses; emits frame stream | PoseDatabase |
| `Views/AvatarCanvasView.swift` | SwiftUI `Canvas` humanized 2D renderer — **start here, it unblocks Path B end-to-end** | Stitcher |
| `Pipelines/PathA.swift` | Camera → MediaPipe → GlossTagger → GemmaTranslator → TTS | All above |
| `Pipelines/PathB.swift` | Mic → ASR → GemmaTranslator → PoseDatabase → Stitcher → Avatar | All above |
| `Pipelines/PipelineCoordinator.swift` | Runs A+B concurrently; `ProcessInfo.thermalState` observer → degraded mode | PathA, PathB |

### Priority 2 — UI shell

| File | What it does |
|---|---|
| `ContentView.swift` | FaceTime-style root: avatar canvas (large) + camera PiP (small) + transcript + action button |
| `Views/CameraPiPView.swift` | `AVCaptureSession` preview layer in SwiftUI (small PiP corner) |
| `Views/TranscriptView.swift` | Scrolling live text output from Path A |
| `Views/ActionButton.swift` | Start / pause both pipelines |
| `Views/SettingsView.swift` | Entry point to history |
| `History/HistoryEntry.swift` | SwiftData `@Model` — one conversation turn (both directions) |
| `History/HistoryStore.swift` | `@Query` + persistence |
| `History/HistoryView.swift` | List of past interpretations |
| `Util/Throttle.swift` | Frame-rate limiter for Canvas repaints |

### Priority 3 — Hardening

- `Info.plist` permission strings (camera, microphone, speech recognition)
- `docs/permissions.md` — already specced
- `docs/memory_budget.md` — validate against table above on real device
- Degraded-mode banner UI when thermal state ≥ `.serious`
- `SemaTests/` — smoke tests for GlossTagger normalization, PoseDatabase clip lookup, Stitcher continuity

### Do not build

- Any server backend
- ARKit body tracking (MediaPipe is the pose source)
- Rive integration (see avatar decision above)
- RealityKit 3D avatar (post-hackathon upgrade path only)
- Physical Action Button integration
- Face landmarks

## Out of scope

- Android. The original plan; superseded by iOS-only for the first release.
- A server backend (other than the optional Gemma fallback documented in `../gemma-glossing/README.md`). Sema is on-device by design.
- Physical Action Button integration (iPhone 15 Pro+). All buttons are on-screen; one UI on all iPhones.
- Face landmarks. The recognizer's joint set is body + hands only; non-manual markers (eyebrows, mouth) are a future-work item.
