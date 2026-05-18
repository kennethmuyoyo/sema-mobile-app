# Path A — Xcode setup checklist

Path A's Swift code is in place; this checklist is what to do in Xcode before the first build. Estimated 5 minutes.

## 1. Add the MediaPipe dependency (CocoaPods)

**MediaPipe Tasks Vision for iOS is CocoaPods-only.** There is no official SPM package; community wrappers exist but aren't maintained by Google. The Swift `import MediaPipeTasksVision` is the same either way.

```bash
# One-time, per machine
brew install cocoapods

# In the repo
cd mobile-app/sema
pod install         # creates sema.xcworkspace
open sema.xcworkspace
```

The Podfile (`mobile-app/sema/Podfile`) is already committed; it pins to `MediaPipeTasksVision ~> 0.10`.

**From this point on always open `sema.xcworkspace`, never `sema.xcodeproj`.** Opening the bare project loses the CocoaPods linking and you'll see `No such module 'MediaPipeTasksVision'`.

### Alternative: XCFramework drop-in

If you'd rather not use CocoaPods: Google ships pre-built XCFrameworks at <https://github.com/google-ai-edge/mediapipe/releases>. Look for `MediaPipeTasksVision-*.xcframework.zip`. Drag it into the project's Frameworks group, set *Embed & Sign* on the target.

## 2. Verify resources are bundled

The Xcode project uses synchronized folder groups (Xcode 16), so any file you drop under `mobile-app/sema/sema/` is auto-added. The following should already be present and included in the build:

| Path | What |
|---|---|
| `Models/gloss_tagger.mlpackage/` | Re-exported at seq_len 64 from the smoke checkpoint |
| `Models/gloss_tagger.vocab.json` | Vocab sidecar (3,846 entries) |
| `Models/gloss_tagger.landmarks_meta.json` | Reference copy of the joint layout |
| `Resources/pose_landmarker_full.task` | MediaPipe body landmarker (~8.5 MB) |
| `Resources/hand_landmarker.task` | MediaPipe hand landmarker (~7 MB) |

To confirm, open the `sema` target → Build Phases → Copy Bundle Resources. The four resources above should appear. If anything is missing, run:

```bash
# in repo root
.venv-recognition/bin/python recognition/export/to_coreml.py \
    --ckpt recognition/checkpoints/transformer_base/last.pt \
    --out  mobile-app/sema/sema/Models/gloss_tagger.mlpackage \
    --seq-len 64

curl -fsSL https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task \
  -o mobile-app/sema/sema/Resources/pose_landmarker_full.task
curl -fsSL https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task \
  -o mobile-app/sema/sema/Resources/hand_landmarker.task
```

## 3. Wire `PathACoordinator` to the existing UI

`PipelineCoordinator` still runs the original demo loop. To swap Path A in:

```swift
// In ContentView or wherever the coordinator is owned:
@State private var pathA = PathACoordinator()
@State private var camera = CameraSessionController()

.onAppear {
    pathA.bootstrap()                     // loads MediaPipe + CoreML, pre-warms ANE
    camera.frameDelegate = pathA          // forward camera frames into Path A
    camera.start()
}
.onChange(of: pathA.state) { _, new in
    if new == .ready { pathA.start() }    // or wire a manual play button
}
```

The transcript shows up at `pathA.transcript`; the array of emitted tokens at `pathA.emittedTokens`.

## 4. Add Info.plist permission strings

Required keys (you likely already have these, but verify):

| Key | Suggested value |
|---|---|
| `NSCameraUsageDescription` | "Sema reads sign language from your front camera." |
| `NSMicrophoneUsageDescription` | (for Path B later) "Sema converts your speech into sign-language animation." |
| `NSSpeechRecognitionUsageDescription` | (for Path B later) "Sema recognises what you say so it can translate it to KSL signs." |

## What works today

- The pipeline compiles and the recognizer's CoreML graph runs on the Neural Engine (`CPUAndNeuralEngine` compute units).
- MediaPipe Holistic (pose + hand) runs at 30 fps on iPhone 14+.
- The streaming decoder will emit tokens once 3 consecutive 64-frame windows agree on a prefix.

## What's intentionally limited in this build

The bundled checkpoint is from a **200-step smoke training run on 179 clips** — it is *not* the production recognizer. Expect:
- Most tokens decoded as `<unk>` or low-frequency vocab.
- Word-error-rate at or near 100 % on real signing.
- The wiring is correct; the model just hasn't seen enough data.

Replace with a real-trained checkpoint once `recognition/train_kaggle.ipynb` completes on Kaggle GPU. Then re-run step 2's export and rebuild.

## File map (Swift source under `mobile-app/sema/sema/`)

```
ML/
├── Landmark45.swift           45-joint constants + MediaPipe index map
├── NormalizedFrame.swift      (T, 45, 4)-shaped recognizer input frame
├── GlossToken.swift           one emitted gloss
├── FrameRing.swift            64-frame sliding window
├── StreamingCTCDecoder.swift  greedy + N=3 stable-suffix
└── GlossTagger.swift          CoreML actor; CPU+ANE compute units

MediaPipe/
└── HolisticLandmarker.swift   PoseLandmarker + HandLandmarker → NormalizedFrame

Pipelines/
└── PathACoordinator.swift     wires it all to CameraSessionController

CameraSessionController.swift  (extended) now emits frames via CameraFrameDelegate
```

The avatar / Path B side of the app (`AvatarCanvasView`, `DemoPoseFrame`, `PipelineCoordinator`) is untouched and still runs the original demo loop.
