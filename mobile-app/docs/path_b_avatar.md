# Path B — On-device plan (mic → ASR → Gemma → pose DB → SwiftUI avatar)

This is the implementation plan for Path B's on-device half of Sema: the hearing user speaks, the Deaf user sees an animated KSL signing avatar. Plan only — no Swift is written until this is approved.

## Why this is the easier path right now

The data we need is already on disk (Motion-S BVH → 45-joint landmarks via `recognition/data/bvh_to_landmarks.py`). Every piece of Path B is either deterministic Swift drawing code, a built-in Apple API, or a model artefact we can substitute behind a server fallback. Path A by contrast still has a sim2real gate (`recognition/eval/real_camera_smoke.md`). So Path B becomes the first demo-able pipeline end-to-end.

## Context

- Generation folder (`generation/`) is currently README-only; `pose_library/build_index.py` exists as a stub. This plan fills it.
- Renderer + ASR contracts: [`../../generation/renderer/ios_contract.md`](../../generation/renderer/ios_contract.md), [`../../generation/asr/contract.md`](../../generation/asr/contract.md). The 45-joint layout is authoritative in `recognition/data/landmarks_meta.json`.
- Gemma 4 E4B fine-tune is tracked in `gemma-glossing/`; not yet trained. The plan integrates the LiteRT artefact when it lands and falls back to an HTTPS server endpoint until then (see `gemma-glossing/README.md`).
- The data the user is referring to: 12,463 full-utterance landmark clips of shape `(T, 45, 3)` produced by `bvh_to_landmarks.py`, paired with English sentences and KSL gloss labels in `train.csv`. ~3,844 unique gloss tokens after punctuation stripping.

## Decisions locked

| Area | Choice |
|---|---|
| Per-gloss slicing (v0) | **Equal slicing** — for a clip with N gloss tokens, divide its frames into N equal ranges and assign the i-th range to the i-th token. Crude; replace with CTC-forced alignment after the recognizer is trained on real data. |
| Vocabulary scope | **Full ~3.8k vocab** bundled, int8-quantised. ~80 MB in the IPA. |
| Input source | **Full Path B from day one** — SFSpeechRecognizer (continuous) → Gemma (LiteRT) → PoseDB → Renderer. Server fallback for Gemma until the local artefact is ready. |
| Renderer | **SwiftUI `Canvas` + `TimelineView`**. 45 joints + ~40 edges drawn at the display refresh. Pure Swift, no SpriteKit. |
| Stitching | Linear interpolation over an **8-frame handoff window** between adjacent gloss clips. |
| Quantisation | **Per-joint per-coord int8** symmetric scale (135 scales per clip). Stored as `int8` values + `(45, 3)` float32 scale matrix in an `.npz`. |
| Source / playback fps | Source 24 fps (from BVH `Frame Time`). Renderer drives off `TimelineView` at display refresh (60–120 Hz) and **resamples**. |

## Top-level data flow

```
AVAudioEngine input tap (16 kHz mono)
        │  AVAudioPCMBuffer
        ▼
ContinuousSpeechRecognizer (actor)         SFSpeechRecognizer with request rotation
        │  Transcript { text, isFinal }    (see generation/asr/contract.md)
        ▼
GemmaTranslator (actor, wraps LiteRT)      task token [EN→KSL] | [SW→KSL]
        │  AsyncStream<GlossToken>         (streamed token-by-token)
        ▼
PoseDatabase (actor)                       gloss → cached .npz lookup
        │  PoseClip { frames: (T, 45, 3) float32 }
        ▼
Stitcher (actor)                           concatenate + 8-frame linear blend
        │  AsyncStream<PoseFrame>          continuous (45, 3) float32 stream at 24 fps
        ▼
AvatarStreamPlayer (@MainActor)            resamples 24 → display refresh, exposes current frame
        │  @Published var currentFrame
        ▼
AvatarCanvasView (SwiftUI Canvas)          draws skeleton; updates via TimelineView
```

## Module layout (interfaces, no code)

All under `mobile-app/Sema/`. As with Path A, only **new** code is listed.

```
Sema/
├── Speech/
│   ├── AudioSession.swift                AVAudioSession config (.playAndRecord, options),
│   │                                     install/remove engine taps, handle interruptions
│   ├── ContinuousSpeechRecognizer.swift  actor; SFSpeechRecognizer + request rotation;
│   │                                     pauses/resumes via TTSGate (see Path A doc)
│   └── Synthesizer.swift                 AVSpeechSynthesizer wrapper (used by Path A;
│                                         exposes TTSGate that ContinuousSpeechRecognizer reads)
├── ML/
│   └── GemmaTranslator.swift             actor; loads gemma4_e4b_int4.tflite via LiteRT;
│                                         applies task token prefix; greedy decode loop;
│                                         streams gloss tokens as soon as available
├── Generation/
│   ├── PoseDatabase.swift                actor; loads PoseLibrary/index.json at launch;
│   │                                     lazy-decodes int8 clip on first access; LRU cache
│   ├── PoseClip.swift                    value type; (T, 45, 3) float32 + source metadata
│   ├── Stitcher.swift                    actor; receives gloss tokens, looks up clips,
│   │                                     concatenates with 8-frame linear blend
│   ├── AvatarStreamPlayer.swift          @MainActor; resamples 24 fps → display refresh;
│   │                                     @Published currentFrame
│   └── AvatarCanvasView.swift            SwiftUI Canvas with TimelineView; draws 45 joints
│                                         + edges from currentFrame
├── Pipelines/
│   └── PathBCoordinator.swift            wires mic → ASR → Gemma → stitcher → player → view
└── Resources/
    ├── gemma4_e4b_int4.tflite            (bundled, DVC-tracked)
    ├── gemma_tokenizer.model             SentencePiece, bundled
    └── PoseLibrary/
        ├── index.json
        └── clips/{TOKEN}.npz             one per gloss token, int8 quantised
```

### Key types (signatures only)

```swift
struct GlossToken {                       // shared with Path A
    var id: Int
    var label: String
    var timestamp: TimeInterval
    var confidence: Float
}

struct PoseClip {
    var frames: [Float]                   // length T * 45 * 3, row-major
    var frameCount: Int
    var fps: Float                        // 24
    var sourceClipId: Int
    var sourceRange: ClosedRange<Int>     // for debugging / future re-alignment
}

struct PoseFrame {
    var values: [Float]                   // length 135 (45 joints × 3 coords)
    var timestamp: TimeInterval
}

struct Transcript {                       // matches generation/asr/contract.md
    var text: String
    var isFinal: Bool
    var confidence: Float?
    var language: Locale
}
```

## Pose library build (Python, offline)

Owner: `generation/pose_library/build_index.py`. Runs as a DVC stage so teammates pull the result from R2 instead of re-running.

### Algorithm (v0, equal-slicing)

```
load train.csv → rows (id, gloss, ...)
candidates : dict[token, list[(slice, source_id, source_range)]] = {}

for row in rows:
    tokens = tokenize_gloss(row.gloss)          # same regex as data/dataset.py
    if not tokens: continue
    landmarks = np.load(data/landmarks/{row.id}.npy)   # (T, 45, 3) float32
    T = landmarks.shape[0]
    n = len(tokens)
    bounds = [round(i * T / n) for i in range(n + 1)]
    for i, tok in enumerate(tokens):
        clip = landmarks[bounds[i] : bounds[i + 1]]
        if clip.shape[0] >= MIN_FRAMES:         # MIN_FRAMES = 6 (~0.25 s @ 24 fps)
            candidates[tok].append((clip, row.id, (bounds[i], bounds[i+1])))

# Canonical clip per token: longest of the equal-sliced candidates
for tok, lst in candidates.items():
    best = max(lst, key=lambda x: x[0].shape[0])
    int8, scale = quantize_int8_per_joint(best[0])
    np.savez_compressed(f"clips/{sanitize(tok)}.npz",
                        clip_i8=int8, scale=scale)
    index[tok] = {
        "path": f"clips/{sanitize(tok)}.npz",
        "n_frames": best[0].shape[0],
        "source_clip_id": best[1],
        "source_range": list(best[2]),
        "fps": 24
    }
write_index_json(index)
```

Filename sanitization: replace `/`, `?`, `*` etc. with `_`; preserve token in `index.json` so lookup is by raw token. The vocab is alphanumeric + `-` after the punctuation strip, so collisions are unlikely; assert uniqueness at write time.

### Quantisation

```python
def quantize_int8_per_joint(clip: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # clip: (T, 45, 3) float32
    scale = np.maximum(np.abs(clip).max(axis=0), 1e-6)        # (45, 3)
    q = np.round(clip / scale * 127).clip(-127, 127).astype(np.int8)
    return q, scale.astype(np.float32)

def dequantize(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (q.astype(np.float32) / 127.0) * scale
```

Reconstruction error on a typical signing clip is well under 1% of the shoulder-width unit — visually imperceptible.

### Bundle size estimate (full vocab)

Average per token: ~30 frames × 135 int8 values + 540 float32 scale bytes ≈ 4.6 KB compressed.
With 3,844 tokens → ~18 MB compressed, ~80 MB uncompressed in memory if all loaded simultaneously.
The PoseDatabase loads lazily with an LRU cache (size: 256 clips ≈ 35 MB resident).

### DVC stage

```yaml
pose_library:
  desc: Build per-gloss int8 pose clips (equal-slicing v0).
  cmd: cd generation && ../recognition/.venv/bin/python -m pose_library.build_index
  deps:
    - generation/pose_library/build_index.py
    - data/landmarks
    - data/train.csv
    - recognition/data/vocab/gloss_vocab.json
  outs:
    - generation/pose_library/index.json
    - generation/pose_library/clips
    - mobile-app/Sema/Resources/PoseLibrary
```

The same script writes to **both** `generation/pose_library/` (canonical, for inspection) and `mobile-app/Sema/Resources/PoseLibrary/` (consumed by the app). The DVC stage caches both.

## PoseDatabase + Stitcher (Swift)

### PoseDatabase

- Init: load `index.json` from `Bundle.main.url(forResource: "PoseLibrary/index", withExtension: "json")`.
- `lookup(_ token: String) async -> PoseClip?`:
  - LRU cache check first.
  - Else: read `Resources/PoseLibrary/clips/{token}.npz`, dequantise to `(T, 45, 3) Float32`, store in cache.
- Unknown token policy: return `nil`; the Stitcher emits a `glossUnknown` event upward (logged + counted; renderer holds the last frame).

### Stitcher

- Input: `AsyncStream<GlossToken>` (from Gemma).
- For each incoming token: `clip = await poseDB.lookup(token.label)`. If `nil`, skip.
- For the **first** clip in a session, the stream starts at clip frame 0.
- For each **subsequent** clip:
  - Emit an 8-frame linear interpolation between `out.last` (the last frame already streamed) and `clip[0]`.
  - Emit the clip's own frames sequentially.
- Output: `AsyncStream<PoseFrame>` at the source fps (24).

Pseudocode:

```
prev: PoseFrame? = nil
for await tok in glossStream:
    guard let clip = await db.lookup(tok.label) else { continue }
    if let prev = prev {
        for i in 1...H {                  // H = 8
            let alpha = Float(i) / Float(H + 1)
            yield lerp(prev.values, clip.frame(0), alpha) at t
            t += 1.0 / 24.0
        }
    }
    for k in 0..<clip.frameCount {
        yield clip.frame(k) at t
        t += 1.0 / 24.0
        prev = current
    }
```

For long pauses between gloss tokens (Gemma emits sparsely), the stream stalls; the renderer holds the last frame. Resume on the next token.

### Why not crossfade with velocity matching for v0

Linear lerp over 8 frames produces a faintly visible "snap" between dissimilar end/start poses but is **far** simpler than velocity-matched easing. Defer until users report it as a problem.

## Renderer (SwiftUI Canvas + TimelineView)

Pure SwiftUI, no SpriteKit, no Metal. 45 joints + ~40 edges per frame at display refresh is well within `Canvas` budget.

```swift
struct AvatarCanvasView: View {
    @ObservedObject var player: AvatarStreamPlayer

    var body: some View {
        TimelineView(.animation) { context in
            Canvas { ctx, size in
                let frame = player.currentFrame(for: context.date)
                drawSkeleton(in: ctx, size: size, frame: frame)
            }
        }
    }
}
```

### Edges (drawn as 3-point-thick lines)

Defined in `Generation/Skeleton.swift` (small constant), mirroring [`generation/renderer/ios_contract.md`](../../generation/renderer/ios_contract.md):

| Section | Joint pairs |
|---|---|
| Spine | `head → mid(L_shoulder, R_shoulder) → mid(L_hip, R_hip)` |
| Eyes | `L_eye ↔ R_eye` |
| Arms | `shoulder → elbow → wrist` (both sides) |
| Legs | `hip → knee → ankle` (both sides) |
| Hands | `wrist → finger{1,2,3}` for each of {thumb, index, middle, ring, pinky}, both sides |

### Frame resampling (24 → display refresh)

`AvatarStreamPlayer` keeps a ring buffer of the last ~120 frames (5 s at 24 fps) and the wall-clock timestamp of each. On `currentFrame(for: Date)`:
- Find the two source frames bracketing the requested timestamp.
- Linearly interpolate the 135 floats between them.
- Return the interpolated frame.

This decouples source rate from display refresh; no glitching when ProMotion switches between 60 and 120 Hz.

## ASR (SFSpeechRecognizer)

See [`generation/asr/contract.md`](../../generation/asr/contract.md) — the implementation matches it without divergence:

- `requiresOnDeviceRecognition = true`.
- Rotate `SFSpeechAudioBufferRecognitionRequest` every 50 s with a 1 s overlap; stitch transcripts by longest-common-suffix.
- Pause while `AVSpeechSynthesizer.isSpeaking == true` (Path A's TTS); resume on `didFinish` + 200 ms tail.
- Emits a `Transcript` stream; the translator subscribes.

## Gemma translator on iOS (LiteRT)

### What we bundle

- `gemma4_e4b_int4.tflite` — produced by `gemma-glossing/export/to_litert.py` (Google AI Edge starting point + LoRA adapter merged).
- `gemma_tokenizer.model` — the SentencePiece model from the same export.

Both are DVC-tracked and copied into `mobile-app/Sema/Resources/` at build time.

### Generation loop

```
on transcript.isFinal == true OR transcript.text.count - lastTranslatedCount > 80:
    prompt = "[\(taskToken)] \(transcript.text)"          // e.g. [EN→KSL] I'm going home
    tokens = tokenizer.encode(prompt)
    output: [Int] = []
    while output.count < maxNewTokens && tokens.count < ctxLimit:
        logits = await gemma.runStep(tokens)              // LiteRT single-step
        nextId = argmax(logits)                           // greedy v0; sampler later
        if nextId == eosId: break
        output.append(nextId)
        tokens.append(nextId)
        // Stream the new token immediately if it ends a gloss boundary
        if isGlossBoundary(nextId):                       // space or [EOS]
            yield GlossToken(label: tokenizer.decode(output[lastBoundary..]),
                             confidence: softmax(logits)[nextId])
            lastBoundary = output.count
```

Streaming gloss tokens during generation (rather than after the full sequence completes) gets the avatar moving within ~300 ms of the first Gemma token. This matters for perceived latency.

### Compute path on iOS

- LiteRT iOS delegate: **CPU-only** until verified otherwise (per `gemma-glossing/README.md`'s assumption).
- Pre-warm at app launch on a background queue: load model, run one step on a dummy token, hold the model alive for the session.
- Expected resident memory: ~3.5 GB. Validate on a 6 GB device; if it OOMs, the server-fallback engages automatically (see below).

### Server-fallback contract

When **any** of these happens, route translation through the HTTPS endpoint in [`../../gemma-glossing/README.md`](../../gemma-glossing/README.md):
- `MLLoadError` on the LiteRT artefact at launch.
- `ProcessInfo.thermalState == .critical` for >5 s.
- 95th-percentile per-token latency >300 ms over a 30-token rolling window.
- Memory-pressure notification at level 2 or higher.

Fallback is silent to the user (a small "cloud" badge in the corner is the only signal). Identical I/O contract — task-token-prefixed prompt, gloss-sequence response.

## Threading & lifecycle

| Component | Isolation | Notes |
|---|---|---|
| `AudioSession` | `@MainActor` for config; nonisolated for buffer plumbing | Activated on session start, deactivated on app background |
| `ContinuousSpeechRecognizer` | `actor` | Serial request rotation; reads `TTSGate` |
| `GemmaTranslator` | `actor` | One concurrent generation at a time; new transcripts cancel in-flight if `isFinal` arrives mid-generation |
| `PoseDatabase` | `actor` | LRU cache, mmap-backed reads |
| `Stitcher` | `actor` | Sequential gloss stream → frame stream |
| `AvatarStreamPlayer` | `@MainActor` | `@Published var currentFrame` drives the view |
| `AvatarCanvasView` | View | Pure rendering |
| `PathBCoordinator` | `@MainActor` | Owns lifecycles, observes thermal state |

**Pre-warm at app launch (background queue):**
- Load Gemma `.tflite`, run one dummy step.
- Load `index.json`, decode the top-N most-frequent gloss clips into the LRU cache.
- Install audio engine taps (but don't start ASR — wait for foreground).

Estimated cold-start cost: 1.5–4 s total. The UI shows a "preparing…" indicator until both Path A and Path B finish warming.

**Thermal/memory adjustments:**
- `.serious`: drop renderer resampling to source rate (no interpolation); 25 % fewer redraws.
- `.critical`: pause Gemma's on-device path; route to server fallback; keep ASR + renderer running.

## Tests (before any real-device run)

1. **`BuildIndexTests`** (Python, in `generation/tests/`) — feed a synthetic 60-frame clip with 3 gloss tokens, assert equal-slicing produces three 20-frame slices and that the longest survives per token.
2. **`QuantisationRoundtripTests`** (Python) — random clip → quantise → dequantise → max-abs-error well below 1% of shoulder-width unit (target: 0.5%).
3. **`PoseDatabaseLoadTests`** (Swift) — load a known small fixture index + clips from `SemaTests/Fixtures/PoseLibrary/`; assert `lookup("HOSPITAL")` returns the right frame count and that the int8 dequantisation matches the Python reference within 1 LSB.
4. **`StitcherTests`** (Swift) — given a hand-built sequence `[A, B, A]` and synthetic clips of length 10, assert output length = 10 + 8 + 10 + 8 + 10 = 46 and that the 8-frame bridges are linear interpolation of the boundary frames.
5. **`AvatarStreamPlayerTests`** (Swift) — push 24 fps frames; query at irregular wall-clock timestamps; assert returned frames are correct linear interpolations.
6. **`GemmaTranslatorParityTest`** (skipped until Gemma is available) — Swift loads `gemma4_e4b_int4.tflite`, runs on a fixed prompt, asserts top-1 next-token matches a Python reference. Marked as skipped in CI; runs locally once the artefact lands.

No real-device tests in this plan — those start after a first build exists. Battery and thermals get measured then and recorded in `mobile-app/docs/memory_budget.md`.

## Risks and where they bite

| Risk | Likely impact | Mitigation in this plan |
|---|---|---|
| **Equal-slicing produces wrong sub-clips** | Avatar shows the wrong sub-motion for some glosses (especially mid-sentence) | Accept for v0; document in `mobile-app/README.md` that the avatar is illustrative until smart slicing lands. Track "wrong-looking" gloss tokens via in-app debug toggle. |
| **Gemma not yet trained / shipped** | The translator returns nothing | Server-fallback wired from day one. v0 demos can run against a temporary server endpoint that wraps either a hosted Gemma or a hand-coded English-to-gloss phrasebook (~50 phrases) for early demos. |
| **Gemma OOMs on 6 GB devices** | App jetsams | Server-fallback triggers on memory-pressure level 2. Memory budget table in `mobile-app/README.md` already reserves ~3.5 GB for Gemma; degraded mode is the safety net. |
| **Stitching looks jerky** between dissimilar end/start poses | Visible "snap" | 8-frame linear blend is the v0 floor; if user-visible, upgrade to velocity-matched easing in a follow-up. |
| **SFSpeechRecognizer cuts at 1 minute** | Transcripts truncate mid-sentence | Rotation + LCS stitching per `generation/asr/contract.md`. |
| **TTS (from Path A) collides with mic** | Echo loop | `TTSGate` (lives in `Speech/Synthesizer.swift`) is read by `ContinuousSpeechRecognizer`; ASR pauses while TTS speaks. |
| **PoseLibrary build is slow** | First teammate run blocks | DVC stage caches the result; teammates `dvc pull` in seconds instead of running the build. |
| **PoseLibrary bundle bloats the IPA** | App Store size warning | int8 quantisation keeps it ~18 MB compressed. If it grows past 100 MB later, switch to on-demand resources keyed by gloss frequency. |

## Open questions

1. **Unknown-gloss policy** — when Gemma emits a gloss not in the PoseLibrary (e.g. a typo or out-of-vocab token), do we hold the last frame, fingerspell letter-by-letter, or show a small text bubble? Pick before integration. Current default: hold last frame + log.
2. **Punctuation in the avatar stream** — Gemma's gloss output may include trailing markers (`//`, `?`). These already get stripped by `tokenize_gloss`. Confirm the same regex is used in the Swift tokenizer-decode path so we don't look up `LOOK?` when the library has `LOOK`.
3. **Resampling artefacts** at extreme fps mismatches (e.g. 120 Hz display vs 24 fps source = 5× upsample) — usually fine with linear blend, but worth a real-device check.

## What this plan does NOT include

- The Xcode project itself.
- Path A pieces other than the `TTSGate` that ASR reads. Path A is in [`coreml_path_a.md`](coreml_path_a.md).
- History storage, settings UI — covered separately in `mobile-app/README.md`.
- Smart per-gloss alignment (CTC-forced or otherwise). Equal-slicing is the v0 floor; smart slicing is its own follow-up project.

## Build order

When this plan and Path A are both approved, suggested sequence — pick what's blocking and parallelise the rest:

1. `generation/pose_library/build_index.py` (Python, equal-slicing) + DVC stage → ships the data.
2. `PoseDatabase`, `PoseClip`, `Stitcher`, `AvatarStreamPlayer`, `AvatarCanvasView` (Swift, deterministic) — testable in isolation against a `[GlossToken]` stub.
3. `ContinuousSpeechRecognizer` + `AudioSession` + `Synthesizer` (Swift, SFSpeechRecognizer + AVSpeechSynthesizer) — testable with the user's voice.
4. `GemmaTranslator` (server-fallback first) — substitute the local LiteRT path when the artefact lands.
5. `PathBCoordinator` — last wire.

Steps 1+2 alone produce a usable internal demo: hard-code a gloss sequence in the coordinator and watch the avatar sign. Steps 3–5 turn it into the real Path B.
