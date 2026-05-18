# Sema — iOS architecture

This document describes the iOS app as it actually ships, not as it was originally planned. Each section starts with a short summary for product / non-engineering readers and then dives into the implementation for engineers working on the codebase.

If you're trying to *build* the project for the first time, start with [`docs/bundle_inventory.md`](docs/bundle_inventory.md) — it covers CocoaPods, the Swift package, and the large model artefacts that have to be hydrated before Xcode can compile.

---

## 1. What the app is

Sema is a Kenyan Sign Language (KSL) interpreter that runs on an iPhone with no internet connection. It does two jobs on the same screen:

- **Sign → Speech.** The front camera watches a Deaf signer; the phone reads the signs and speaks the English (or Swahili) translation aloud.
- **Speech → Sign.** The microphone listens to a hearing speaker; a 3D avatar on screen signs the translation back.

A single SwiftUI screen — `ConversationScreenView`, owned by `ContentView` — hosts both directions. Tapping the camera tile toggles which side is "live": **listening** (mic on, camera off) or **watching** (camera on, mic off). The two paths share one `AVAudioSession`, one `AVCaptureSession`, and one on-device translator, all coordinated by `ConversationOrchestrator` (`sema/Pipelines/ConversationOrchestrator.swift`).

The app is iOS 17+, requires the front camera and the microphone, and ships with every model it needs already inside the IPA.

### Pipeline shape

```
                          ┌───────────────────────────────────┐
                          │       ConversationOrchestrator    │
                          │  half-duplex listening / watching │
                          └────────────┬──────────────────────┘
                                       │
        ┌──────────────── watching ────┴──── listening ───────────────┐
        ▼                                                              ▼
   PathACoordinator                                            PathBCoordinator
   (Sign → Speech)                                             (Speech → Sign)

   front camera                                                microphone
        │                                                              │
        ▼                                                              ▼
   HolisticLandmarker                                  ContinuousSpeechRecognizer
   (MediaPipe pose + hand)                             (SFSpeechRecognizer, on-device)
        │  45 joints × xyz                                             │  Transcript
        ▼                                                              ▼
   CoreMLGlossTagger                                   GemmaTranslator (.stub / .server / .onDevice)
   (ksl_model.mlpackage, ANE)                          (llama.cpp + gemma-4-e2b-ksl-Q4_K_M.gguf)
        │  gloss logits                                                │  gloss tokens
        ▼                                                              ▼
   StreamingCTCDecoder                                          PoseDatabase
        │  GlossToken stream                                           │  PoseClip
        ▼                                                              ▼
   GemmaTranslator (KSL→EN)                                      Stitcher
        │  English sentence                                            │  PoseFrame stream
        ▼                                                              ▼
   AVSpeechSynthesizer                                  AvatarStreamPlayer  ◄── SG + One-Euro smoothing
        │  spoken audio                                                │  rawFrame
        ▼                                                              ▼
   speaker out                                              SimpleAvatar3DView (SceneKit)
```

The two paths share a `TTSGate` so the avatar's spoken output never echoes into the microphone, and they share `GemmaTranslator` so the same model handles both KSL→EN (for Path A's voice) and EN→KSL (for Path B's avatar).

---

## 2. Offline & privacy

**Sema runs fully on-device.** Everything you see on screen — speech recognition, sign recognition, translation, the signing avatar — is computed locally. The app makes no outbound network calls in its default configuration. Put the phone in airplane mode and every feature still works.

This is intentional. KSL interpretation often happens in medical, legal, and personal contexts where sending audio or video off-device is not acceptable.

### What's bundled (no network fetch)

| Model | File | Format | Used by |
|---|---|---|---|
| MediaPipe pose | `pose_landmarker_full.task` | MediaPipe Task | `HolisticLandmarker` |
| MediaPipe hand | `hand_landmarker.task` | MediaPipe Task | `HolisticLandmarker` |
| Gloss recognizer | `ksl_model.mlpackage` | CoreML | `CoreMLGlossTagger` |
| Gemma 4 E2B KSL | `gemma-4-e2b-ksl-Q4_K_M.gguf` | GGUF (llama.cpp) | `LlamaGemmaEngine` |
| Pose clip library | `PoseLibrary/`, `PoseLibraryFull/` | int8 `.npz` + JSON manifest | `PoseDatabase` |
| Avatar rig | `hackathon.usdc` + textures | USDC | `SimpleAvatar3DView` |

Speech recognition uses Apple's `SFSpeechRecognizer` with `requiresOnDeviceRecognition = true` (`Speech/ContinuousSpeechRecognizer.swift:182`). Speech synthesis uses `AVSpeechSynthesizer`, which runs on the OS's own offline voices.

### Privacy declarations

- `sema/PrivacyInfo.xcprivacy` declares `NSPrivacyTracking = false`, no tracking domains, no collected data types, and no accessed API types. Nothing leaves the device.
- `sema/sema.entitlements` requests only two entitlements: `com.apple.developer.kernel.increased-memory-limit` (so the llama.cpp KV cache + MediaPipe + CoreML + SceneKit can coexist) and `com.apple.developer.kernel.extended-virtual-addressing` (so the GGUF mmap can span >4 GB of virtual address space). Neither is a privacy-sensitive entitlement; both exist purely to keep the on-device model alive without being jetsam-killed.

### The one optional network exit

`GemmaTranslator` has three modes (`sema/ML/GemmaTranslator.swift:25`):

- `.stub` — phrasebook lookup, zero compute, zero network. **This is the current default** because the bundled Q4_K_M GGUF (~3.2 GB resident) is unstable on the device under the simultaneous-bring-up of MediaPipe and SceneKit (`ConversationOrchestrator.swift:82-92`). The phrasebook covers the Hospital and Bank demo scenarios; anything else falls back to per-word uppercase glosses + PoseLibrary lookup.
- `.onDevice` — llama.cpp loads the GGUF, runs Gemma fully on-device. Still no network. Re-enable by editing the mode in `ConversationOrchestrator`'s init.
- `.server` — HTTPS POST to a configured endpoint. **Off by default.** Only engaged if a caller explicitly constructs `GemmaTranslator(mode: .server(url, token: ...))`. The contract is described in `gemma-glossing/README.md`.

If you want a hard "no network, no exceptions" build, leave the mode at `.stub` or `.onDevice` and the URLSession code in `callServer` is never reached.

### Engineering note: increased memory limit

The `com.apple.developer.kernel.increased-memory-limit` entitlement is what lets supported devices (A14+ iPhones, M-series iPads) raise the per-app resident-memory cap from ~3 GB toward ~5–6 GB. Without it, `llama.cpp`'s GGUF mmap + MediaPipe's TFLite contexts + the pose-library LRU cache + SceneKit's rig all-resident triggers jetsam code 9. The entitlement is a public Apple API; it does not give Sema any data-access privileges.

---

## 3. Speech-to-text (Path B input)

The hearing speaker's voice is transcribed live by `ContinuousSpeechRecognizer` (`sema/Speech/ContinuousSpeechRecognizer.swift`). The class is a SwiftUI-`@Observable` wrapper around Apple's `SFSpeechRecognizer`, with three pieces of glue that matter:

1. **On-device by default.** Each recognition request is created with `requiresOnDeviceRecognition = true`. If the OS reports the on-device model is unavailable (a fresh install on a slow network sometimes returns error code 216), the recognizer transparently downgrades to the server path *for that one request only* — but this only happens if Apple's own ASR is not yet ready, never as a Sema choice.

2. **Echo cancellation via `TTSGate`.** Path A's avatar-translated reply is spoken by `AVSpeechSynthesizer`. While TTS is playing, the system's own speaker output bleeds back into the microphone — without a gate, the ASR would hear "I have a sore throat" come out of the speaker and re-transcribe it. `Synthesizer` flips `TTSGate.shared.setSpeaking(true/false)` on its delegate callbacks (`Speech/Synthesizer.swift:39-58`); `ContinuousSpeechRecognizer` observes the gate and pauses while it's true, with a 200 ms tail before re-opening the mic (matches the `generation/asr/contract.md` contract).

3. **Audio session contract.** Both paths share one `AVAudioSession` configured for `.playAndRecord` with mode `.videoChat` (`Speech/AudioSession.swift:54-58`). `.videoChat` keeps the mic stable alongside the front camera; `.measurement` (the lower-latency choice) fights `AVCaptureSession` for the input route. The session is configured once on session start and torn down on pause.

The implementation today is v0: one long-lived `SFSpeechAudioBufferRecognitionRequest` per session, restarted on the system's 1-minute cap. The full 50-second rotation-with-LCS-stitching strategy from the design doc is a follow-up (`ContinuousSpeechRecognizer.swift:10-13`); for current demo lengths the simpler path is enough.

---

## 4. Sign recognition — CoreML tagger (Path A)

The front camera feeds `HolisticLandmarker` (`sema/MediaPipe/HolisticLandmarker.swift`), which runs MediaPipe's pose + hand landmarkers and emits a `NormalizedFrame`: 45 joints × (x, y, z, mask). Joints are normalized to the signer's shoulder width with origin at mid-shoulder — the exact same transform used in training (`recognition/data/bvh_to_landmarks.py`'s `normalize_landmarks`). A 64-frame ring buffer (`FrameRing`) provides the sliding input window for the tagger.

The tagger itself is **CoreML**, not TFLite. The model is `Resources/ksl_model.mlpackage`, run by `CoreMLGlossTagger` (`sema/ML/CoreMLGlossTagger.swift`).

### Why CoreML and not TFLite

The natural choice would have been to keep the recognizer in TFLite to match the rest of the on-device ML stack. We tried that. `MediaPipeTasksVision` bundles its own TFLite runtime and force-loads its symbols at link time; adding a second TFLite framework via `TensorFlowLiteSwift` produced **48 duplicate-symbol linker errors** that modern `ld64` won't suppress. CoreML uses Apple's Espresso runtime, which shares zero libraries with MediaPipe's TFLite, so the two coexist cleanly — and we get Neural Engine acceleration for free.

This is documented in the file header (`CoreMLGlossTagger.swift:12-19`) and called out in `PathACoordinator.swift:99-102` so future contributors don't relitigate the decision.

### Inference contract

- **Input.** `(1, 64, 135)` `MLMultiArray` of `Float32` — 64 frames × (45 joints × xyz). Shape is verified against `KSLModelMetadata` at load time, so a mismatched export trips on first launch rather than silently producing garbage (`CoreMLGlossTagger.swift:70-97`).
- **Output.** A flat `[Float]` of length `vocabSize` — gloss logits. The auxiliary phonological heads from the v11 training architecture were dropped from the iOS export (coremltools couldn't trace the int-cast in their argmax+stack path); the comment at `CoreMLGlossTagger.swift:21-26` records the path back if Hamming retrieval gets re-enabled.
- **Compute units.** `MLModelConfiguration.computeUnits = .all` — Neural Engine first, GPU fallback, CPU last (`CoreMLGlossTagger.swift:59`).
- **Warmup.** `prewarm()` runs one zero-input prediction so the first user-visible inference doesn't pay the JIT-compile cost (`CoreMLGlossTagger.swift:140-144`). Called from `PathACoordinator.bootstrap()`.

`PathACoordinator` runs one inference every 4 frames (~167 ms at 24 fps capture, `PathACoordinator.swift:47`) and feeds the logits into a `StreamingCTCDecoder`. Gloss tokens are emitted when the decoder's prefix stabilizes; a no-hand timeout (`signingWristYMax`, `signingMotionMin`, `noHandTimeout` — same file, lines 53-72) resets the decoder so a held-down hand at the signer's side doesn't churn out spurious tokens.

The whole gloss stream is then handed to `GemmaTranslator` with task `.kslToEnglish` for synthesis into a fluent sentence, which `Synthesizer` speaks.

---

## 5. Gemma on-device via llama.cpp

The translator behind both directions is the same Gemma 4 E2B model, fine-tuned on KSL ↔ EN/SW pairs (`gemma-glossing/` in the repo root). The fine-tune is exported as a `Q4_K_M` GGUF and run on iOS through **llama.cpp**, not Ollama.

> Ollama is a desktop CLI/server that itself wraps llama.cpp; it does not run on iOS. The earlier plan in the original README mentions LiteRT/Google AI Edge; that path was tried and shelved in favor of llama.cpp because llama.cpp gave us a stable Metal backend for Q4_K_M without the LiteRT-LM SDK's runtime tokenizer constraints.

### LlamaGemmaEngine

`sema/ML/LlamaGemmaEngine.swift` is a thin Swift `actor` over llama.cpp's C API (linked as an XCFramework — see `sema/Frameworks/`). The contract:

- **Bundled model.** `gemma-4-e2b-ksl-Q4_K_M.gguf` in `Resources/`. Located via `Bundle.main.path(forResource:ofType:)` (no on-disk download).
- **One method that matters.** `generate(prompt:maxTokens:) async throws -> String` wraps the prompt in Gemma's `<start_of_turn>user/model<end_of_turn>` chat template, runs prefill, then greedy-decodes until EOG, `<end_of_turn>`, or `maxTokens` — whichever fires first. The KV cache is cleared between calls so prior generations don't leak into a new prompt (`LlamaGemmaEngine.swift:50-52`).
- **Sampler.** Greedy only. Deterministic, matches the temperature=0 / top-k=1 config the earlier LiteRT-LM path used. Top-p / temperature are one `llama_sampler_chain_add` call away if needed.
- **Threads.** `n_threads = min(8, cores - 2)` — leaves headroom for the camera + Metal + main thread.
- **GPU layers.** Default Metal offload on device; forced to 0 on the simulator (`LlamaGemmaEngine.swift:128-131`).

### Memory and lifecycle

The Q4_K_M GGUF loads as a ~1.5–2 GB mmap region; the KV cache (n_ctx = 2048) adds another ~700 MB resident. Combined with MediaPipe (~200 MB), CoreML (~50 MB), the pose library + SceneKit rig (~250 MB), and the rest of the app, the **on-device path is right at the edge of the 5.7 GB Metal budget on iPhone 15-class devices**. Two consequences:

1. **`.stub` is the shipping default.** `ConversationOrchestrator.swift:82-92` calls out why: simultaneous bring-up with the front camera and MediaPipe contexts was crashing the llama context at `sched_reserve` (the camera's CoreMediaIO worker and llama's Metal init were racing for the same Metal command queue). Switch to `.onDevice` by editing the `translatorMode:` parameter in that file's no-arg init.
2. **Memory-warning autoclose.** On a `UIApplication.didReceiveMemoryWarningNotification`, the orchestrator calls `GemmaTranslator.closeSharedOnDeviceEngine()`, which frees the model, context, sampler, and KV cache. The next `translate` call lazily reloads (paying the ~2–4 s warmup again). Better a slow first reply than a jetsam kill mid-conversation.

### Two directions, same engine

`GemmaTranslator` (`sema/ML/GemmaTranslator.swift:30-35`) defines four tasks: `[EN→KSL]`, `[SW→KSL]`, `[KSL→EN]`, `[KSL→SW]`. Each is a task-token prefix in the prompt rendered by `KSLPrompts.render(...)`. The few-shot prompts live in `sema/ML/KSLPrompts.swift`. Path B's coordinator calls `translate(text, task: .englishToKSL)`; Path A's `SignToSpeechBridge` calls `translate(glossList, task: .kslToEnglish)`. Same engine instance, different prompts.

Replies are parsed by `parseReply` (`GemmaTranslator.swift:144-165`):
- For `→KSL` directions: split on whitespace, strip punctuation, uppercase. Anything lowercase is treated as the model wandering off-script and dropped.
- For `→EN/SW` directions: return the cleaned single-string sentence.

---

## 6. Motion retargeting to the 3D avatar

The avatar is **SceneKit**, not RealityKit. `SimpleAvatar3DView` (`sema/SimpleAvatar3DView.swift`) is a `UIViewRepresentable` wrapping an `SCNView` that loads `hackathon.usdc` — a humanoid rig in SMPL-X-style topology with body, hands, fingers, eyes, and clothing meshes. The view renders at 30 fps (capped via `preferredFramesPerSecond`; source clips are 24 fps so 30 is more than enough).

### The retargeting story, end to end

The pose clips in `Resources/PoseLibrary/` and `PoseLibraryFull/` are produced offline from motion-capture BVH data. Each `.npz` clip carries two arrays:

| Array | Shape | What it is |
|---|---|---|
| `clip_i8` (or `clip_f32`) | `(T, 45, 3)` | 45-joint positional landmarks (the same layout MediaPipe produces). Used for the 2D-style positional fallback path and for any view that wants raw landmarks. |
| `quat_f32` | `(T, 52, 4)` | Parent-local quaternions for the 52-joint rig (`BVHRigRotationLayout.jointOrder`), in `[x, y, z, w]` order. **This is the retargeted sidecar** — Python has already converted from BVH source-rig rotations into target-rig parent-local quaternions, so the iOS side is a straight assignment. |

The Python that builds the sidecar lives at `generation/pose_library/retarget_to_target.py`; the joint order and parent map on the iOS side **must exactly match** the Python's `TARGET_JOINT_ORDER` and parent table. That contract is enforced by name lookup at runtime (see below), not by a tagged file format — drift will silently zero unmatched joints.

### Iterative joint order and the parent map

`sema/Generation/BVHRigRotationLayout.swift` defines the 52-joint layout the sidecar uses:

```
pelvis
  → spine1 → spine2 → spine3
                       → neck → head → jaw / eyes
                       → left_collar → left_shoulder → left_elbow → left_wrist
                                                          → finger chains (5 × 3 joints)
                       → right_collar → ...  (mirrored)
  → left_hip → left_knee → left_ankle → left_foot
  → right_hip → right_knee → right_ankle → right_foot
```

Order is parent-before-child (depth-first), so for every joint `i` with a parent in the list, `parentIndex[i] < i`. The iOS retargeting loop relies on this so it can compose world rotations in a single forward pass.

### How a frame becomes a pose on the rig

`SimpleAvatar3DView.Coordinator.apply(frame:)` (`SimpleAvatar3DView.swift:441-513`) tries two paths in order:

1. **Quaternion sidecar path** — preferred. `applyRigRotationsIfAvailable(frame:)` reads `frame.rigRotations` (the 52 × 4 flat array sliced out of the `.npz` for this timestep). For each joint `i`, it constructs `simd_quatf(ix:iy:iz:r:)` and assigns to `binding.node.simdOrientation`. SceneKit handles forward kinematics from there. This is the path used for any clip that came from BVH-to-rig retargeting, which is currently all of them (`SimpleAvatar3DView.swift:527-550`).

2. **Positional fallback** — used only if the sidecar is missing. Reads `left_shoulder/elbow/wrist` (and mirrors) from `frame.values` (the 45-joint positional data), maps from MediaPipe's coordinate system to the rig's, scales by `modelShoulderDist / mpShoulderDist`, then calls `orientBone(...)` per limb. `orientBone` computes the quaternion that rotates the bone's rest direction onto the target direction via `simd_quatf(from:to:)`. The hand chain (`thumb / index / middle / ring / pinky × 3 joints per side`) uses the same per-finger logic in `applyFingerChain` (`SimpleAvatar3DView.swift:692-725`).

Both paths preserve the rest orientations captured at `setup(in:)` so the rig settles back to its bind pose cleanly when no frame is provided.

### Smoothing chain

Linear-blended pose handoffs between adjacent gloss clips (the Stitcher's 8-frame lerp at `Stitcher.swift:60-79`) leave a faint velocity discontinuity at clip boundaries, and MediaPipe-derived positions carry frame-to-frame jitter. Two filters in series clean both:

```
Stitcher
   → Savitzky-Golay (window=5, order=2, centered)
       → kills velocity discontinuities at clip boundaries
   → One-Euro (causal, adaptive)
       → kills residual jitter; adapts cutoff per-frame so slow holds get
         extra smoothing while fast sign onsets stay crisp
   → apply(rawFrame)
```

Both live inside `AvatarStreamPlayer` (`sema/Generation/AvatarStreamPlayer.swift:39-44, 106-150`). One-Euro filters quaternion components independently and then renormalizes per joint to keep them on the unit hypersphere — component-wise filtering nudges quats off the manifold, and SceneKit treats off-manifold rotations as garbage. Both filters auto-reset after a 0.4 s quiet gap so a fresh utterance never blends against trailing state from the previous one.

### Idle animation

When no clip frame has arrived in `idleTakeoverThreshold` (0.35 s), the player switches to an idle pose with a subtle ±1.4° X-axis sway on `spine1` at a 4-second period (`AvatarStreamPlayer.swift:166-186`). Sign is communicative; the idle motion exists only to keep the avatar from looking like a frozen T-pose between turns.

---

## 7. Orchestration & lifecycle

`ContentView` instantiates a single `ConversationOrchestrator`. `bootstrap()` runs at `.onAppear` and warms up both pipelines in parallel; `start()` brings the live session up; `pause()` tears it back down on `.onDisappear`.

The orchestrator is **half-duplex**. Originally both paths ran simultaneously, but a CoreMediaIO race between AVCaptureSession's camera worker and AVAudioEngine's mic tap made the gloss tagger and ASR fight each other and produce intermittent dropouts. The current design forces only one sensor active at a time:

- `enterListeningMode` — `.playAndRecord` audio session, ASR running, camera stopped. PathA paused. Default startup mode.
- `enterWatchingMode` — `.playback` audio session (preserves TTS), camera running, ASR paused. PathA active.

Switching is manual (`toggleMode()` on a tap). The `SessionMode` enum is exposed for any caller that needs to know which sensor is live; downstream views just read `isListening` / `isWatching` flags.

### Threading

| Component | Isolation | Why |
|---|---|---|
| `ContentView`, `ConversationOrchestrator`, `PathACoordinator`, `PathBCoordinator`, `AvatarStreamPlayer` | `@MainActor` | UI-touching state. SwiftUI observation via `@Observable`. |
| `HolisticLandmarker`, `CoreMLGlossTagger`, `PoseDatabase`, `Stitcher`, `GemmaTranslator`, `LlamaGemmaEngine` | `actor` | Off-main background work with serial access to a heavy resource (CoreML, llama.cpp, mmap'd clips). |
| Camera frames | `DispatchQueue` (background) | `AVCaptureVideoDataOutput` delegate callbacks. Hand off to Path A's actor and return immediately. |

`Sendable` is enforced repo-wide under Swift 6 strict concurrency.

---

## 8. File map (Swift source under `sema/sema/`)

```
semaApp.swift                      @main entry; mounts ContentView (or a black
                                    screen in TestEnvironment)
ContentView.swift                   thin wrapper; owns the orchestrator
SimpleAvatar3DView.swift            SceneKit UIViewRepresentable; rig binding +
                                    retargeting (sidecar + positional fallback)
CameraSessionController.swift       AVCaptureSession front camera, BGRA 720p
VolumeShortcutDetector.swift        hardware-volume gesture to start a session
sema.entitlements                   increased memory + extended VA only
PrivacyInfo.xcprivacy               no tracking, no collected data, no APIs

Speech/
  AudioSession.swift                shared AVAudioSession (.playAndRecord/.playback)
  ContinuousSpeechRecognizer.swift  on-device SFSpeechRecognizer + TTSGate
  Synthesizer.swift                 AVSpeechSynthesizer + TTSGate flips
  TTSGate.swift                     single shared flag; ASR pauses while true
  SpeechOutputting.swift            tiny abstraction so tests can fake TTS

MediaPipe/
  HolisticLandmarker.swift          PoseLandmarker + HandLandmarker → 45-joint
                                     NormalizedFrame, with carry-forward across
                                     short detector dropouts

ML/
  CoreMLGlossTagger.swift           ksl_model.mlpackage runner (ANE+GPU+CPU)
  LlamaGemmaEngine.swift            llama.cpp actor; loads gemma-4-e2b-ksl GGUF
  GemmaTranslator.swift             .stub / .server / .onDevice modes + parser
  KSLPrompts.swift                  few-shot prompt templates per task
  KSLModelMetadata.swift            vocab + label spaces sidecar loader
  FrameRing.swift                   64-frame sliding window for the tagger
  StreamingCTCDecoder.swift         CTC + N=3 stable-suffix gate
  Landmark45.swift                  45-joint constants + MediaPipe index map
  NormalizedFrame.swift             tagger input frame (xyz + mask)
  LandmarkDump.swift                debug: persist input/output for offline replay
  PoseTemplateMatcher.swift         deterministic fallback recognizer (template
                                     match against PoseLibrary clips)
  GlossToken.swift                  emitted gloss with confidence + timestamp

Generation/
  BVHRigRotationLayout.swift        52-joint rig layout + parent map (must match
                                     retarget_to_target.py)
  PoseClip.swift                    PoseClip + PoseFrame value types
  PoseDatabase.swift                .npz loader + LRU cache, multi-library merge
  Stitcher.swift                    gloss stream → PoseFrame stream, 8-frame
                                     linear handoff
  AvatarStreamPlayer.swift          ingest + SG + One-Euro + idle animation
  PoseSmoothingFilter.swift         Savitzky-Golay (5/7/9-tap, order-2)
  OneEuroFilter.swift               One-Euro adaptive low-pass

Pipelines/
  ConversationOrchestrator.swift    half-duplex listening/watching, permissions,
                                     TTS observation, memory-warning autoclose
  PathACoordinator.swift            camera → MediaPipe → CoreML → CTC → Gemma → TTS
  PathBCoordinator.swift            mic → ASR → Gemma → PoseDB → Stitcher → player

Views/
  ConversationScreenView.swift      single FaceTime-style screen
  SignToSpeechBridge.swift          Path A's gloss → Gemma → Synthesizer wiring
  Design.swift                      shared design tokens
  Components/                       caption cards, control dock, top bar, etc.
  Debug/                            landmark/skeleton overlays, gloss player

Resources/
  ksl_model.mlpackage               recognizer (with .metadata.json sidecar)
  pose_landmarker_full.task         MediaPipe pose (~9 MB)
  hand_landmarker.task              MediaPipe hand (~7.5 MB)
  gemma-4-e2b-ksl-Q4_K_M.gguf       llama.cpp model (~3 GB)
  PoseLibrary/                      curated demo clips + index.json
  PoseLibraryFull/                  alignment-derived clips + index_full.json
  demo_recognition.json             Path A scenario allowlist
  demo_generation.json              Path B scenario manifest

Util/
  MediaPermissions.swift            camera/mic/speech permission snapshot
  TestEnvironment.swift             flags for previews + test runs
  FrontCameraMirroring.swift        SwiftUI helper for mirrored preview
```

---

## 9. See also

- [`README.md`](README.md) — quick-start, what's shipped, build steps
- [`docs/bundle_inventory.md`](docs/bundle_inventory.md) — every file in the IPA, where it comes from, how to refresh it
- [`docs/path_a_setup.md`](docs/path_a_setup.md) — Path A Xcode-side checklist (CocoaPods, model export)
- [`docs/path_b_avatar.md`](docs/path_b_avatar.md) — original Path B design (most still applies; the renderer is now SceneKit, not SwiftUI Canvas)
- [`docs/coreml_path_a.md`](docs/coreml_path_a.md) — CoreML export & integration notes
- `../generation/renderer/ios_contract.md` — the renderer contract the Python retargeter writes against
- `../generation/asr/contract.md` — the ASR contract `ContinuousSpeechRecognizer` implements
- `../gemma-glossing/README.md` — how the Gemma fine-tune is trained and exported to GGUF
- `../recognition/README.md` — how `ksl_model.mlpackage` is trained and exported
