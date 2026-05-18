# Path A — On-device plan (camera → MediaPipe → CoreML recognizer → gloss tokens)

This is the implementation plan for Path A's on-device half of Sema. It stops at the gloss token stream — the downstream Gemma + TTS hop is covered separately. No Swift is written until this plan is approved.

## Context

- The recognizer artefact is `mobile-app/Sema/Models/gloss_tagger.mlpackage`, produced by `recognition/export/to_coreml.py`. Input `features: (1, T, 180)` float32, output `logits: (1, T, V=3846)` float32. `T` is fixed at conversion time; the current export uses 256, which is too long for live UX.
- The 45-joint layout, normalisation, and mask channel are authoritative in `recognition/data/landmarks_meta.json` (45 joints, body 15 + 30 finger joints; shoulder-mid origin, shoulder-width unit scale, y-down image-plane).
- MediaPipe Tasks for iOS provides `PoseLandmarker` (33-landmark body) and `HandLandmarker` (21 landmarks × 2 hands). There is no longer a single Holistic task on iOS; the adapter stitches them.
- The recognizer was trained on synthetic-MediaPipe features projected from Motion-S BVH, with `data/augment.py` modelling MediaPipe jitter/dropout. The real-camera deployment gate is `recognition/eval/real_camera_smoke.md`.

## Decisions locked

| Area | Choice |
|---|---|
| Window length | **64 frames** (~2.7 s at 24 fps). Re-export the recognizer at `--seq-len 64` before integration. |
| Run cadence | **Sliding** with stride 8. New inference every ~0.27 s at 30 fps. |
| Compute units | `CPUAndNeuralEngine` (skip GPU; the GPU is reserved for camera preview and avatar render). |
| Decoder | Greedy CTC + stable-suffix emission across N=3 consecutive windows. |
| MediaPipe assets | `pose_landmarker_full.task` + `hand_landmarker.task` bundled in-app, fetched via a DVC stage from Google's CDN into `mobile-app/Sema/Resources/`. |
| Asset and gloss-tagger inputs | Both Swift and the renderer use the same joint ordering as `recognition/data/landmarks_meta.json` — single source of truth. |

## Top-level data flow

```
AVCaptureSession (front camera, 1080p @ 30 fps)
        │  CMSampleBuffer
        ▼
HolisticLandmarker (actor)               ── PoseLandmarker + HandLandmarker run in parallel;
        │                                   internally remaps to 45-joint order, normalizes
        │                                   (shoulder-mid origin, shoulder-width scale),
        │                                   appends per-joint mask channel.
        │  NormalizedFrame (Float32 × 180)
        ▼
FrameRing (actor)                        ── ring buffer of last 64 frames
        │  every 8 frames pushed: run inference
        ▼
GlossTagger (actor, wraps MLModel)       ── CoreML inference (~10–20 ms ANE)
        │  Float32 logits (1, 64, 3846)
        ▼
StreamingCTCDecoder                      ── greedy argmax → collapse → stable-suffix
        │  newly-stable gloss token(s)
        ▼
PathACoordinator → Publisher<GlossToken> ── handed to Gemma in the next stage
```

## Module layout (interfaces, no code)

All under `mobile-app/Sema/`. Files listed are the **only** new code in this plan; existing files in `mobile-app/README.md` that this plan does **not** touch are deliberately omitted. There is **no separate `LandmarkAdapter` file** — the MediaPipe→model-input preprocessing (index remap + normalize + mask) lives inside `HolisticLandmarker` because the index mapping is intrinsic to MediaPipe's own layout. The downstream `GlossTagger` never sees raw MediaPipe types.

```
Sema/
├── MediaPipe/
│   ├── PoseLandmarker.swift          actor; loads pose_landmarker_full.task; emits raw 33-landmark frames
│   ├── HandLandmarker.swift          actor; loads hand_landmarker.task; emits raw 21×2 hand frames
│   └── HolisticLandmarker.swift      actor; runs both in parallel, joins on timestamp,
│                                     remaps to 45-joint order, normalizes, builds mask,
│                                     emits NormalizedFrame
├── ML/
│   ├── GlossTagger.swift             actor; owns MLModel; fixed (1, 64, 180) shape; pre-warmed at launch
│   ├── StreamingCTCDecoder.swift     stateful decoder; stable-suffix emission
│   └── FrameRing.swift               actor; bounded ring buffer of 64 NormalizedFrames
├── Pipelines/
│   └── PathACoordinator.swift        wires everything; owns the publisher feeding the next stage
└── Resources/
    ├── pose_landmarker_full.task     (bundled, DVC-tracked)
    └── hand_landmarker.task          (bundled, DVC-tracked)
```

### Key types (signatures only)

```swift
struct NormalizedFrame {
    var values: [Float]                  // length 180 (45 joints × 4: x, y, z, mask)
    var timestamp: TimeInterval
}

struct GlossToken {
    var id: Int
    var label: String
    var timestamp: TimeInterval
    var confidence: Float                // softmax peak at emission point
}
```

The raw MediaPipe landmark types (`Landmark33`, `Landmark21`) are internal to `MediaPipe/` and never cross the module boundary.

## MediaPipe → NormalizedFrame spec (inside HolisticLandmarker)

Authoritative source: `recognition/data/landmarks_meta.json` + `recognition/data/bvh_to_landmarks.py` (the `normalize_landmarks` function). The Swift code inside `HolisticLandmarker` must produce **bit-equivalent** output to that function for a given MediaPipe input (modulo single-precision rounding).

### MediaPipe → 45-joint mapping

Body 15 (MediaPipe `PoseLandmarker` indices):

| Target | MP index | Notes |
|---|---|---|
| `head` | 0 | nose (`recognition/` calls this `head` for trained-layout parity) |
| `left_eye_smplhf` | 2 | `left_eye` |
| `right_eye_smplhf` | 5 | `right_eye` |
| `left_shoulder` | 11 | |
| `right_shoulder` | 12 | |
| `left_elbow` | 13 | |
| `right_elbow` | 14 | |
| `left_wrist` | 15 | |
| `right_wrist` | 16 | |
| `left_hip` | 23 | |
| `right_hip` | 24 | |
| `left_knee` | 25 | |
| `right_knee` | 26 | |
| `left_ankle` | 27 | |
| `right_ankle` | 28 | |

Hand 30 (MediaPipe `HandLandmarker` indices; same for left/right, mirror the source). The trained order is **index → middle → pinky → ring → thumb**, and each finger uses **MCP, PIP, DIP** (we drop the tip):

| Target | MP hand index |
|---|---|
| `{l/r}_index1` | 5 (MCP) |
| `{l/r}_index2` | 6 (PIP) |
| `{l/r}_index3` | 7 (DIP) |
| `{l/r}_middle1` | 9 |
| `{l/r}_middle2` | 10 |
| `{l/r}_middle3` | 11 |
| `{l/r}_pinky1` | 17 |
| `{l/r}_pinky2` | 18 |
| `{l/r}_pinky3` | 19 |
| `{l/r}_ring1` | 13 |
| `{l/r}_ring2` | 14 |
| `{l/r}_ring3` | 15 |
| `{l/r}_thumb1` | 1 |
| `{l/r}_thumb2` | 2 |
| `{l/r}_thumb3` | 3 |

### Normalization steps (per frame)

1. Build a `(45, 3)` matrix `p` from the mapping above, in the order listed in `landmarks_meta.json`.
2. Compute `origin = 0.5 * (p[L_shoulder] + p[R_shoulder])`.
3. Compute `scale = max(||p[L_shoulder] - p[R_shoulder]||, 1e-3)`.
4. Output `p' = (p - origin) / scale`. **Do NOT negate y** — MediaPipe is already y-down (matches the `-y` step in `bvh_to_landmarks.py`, which was applied to BVH's y-up).
5. Build the per-joint mask: `mask[j] = (presence[j] > 0.5) ? 1.0 : 0.0`. For dropped joints (mask=0), zero the xyz triple. Source for presence: MediaPipe's per-landmark `presence` field on body; `score` from `HandLandmarker` for hands (whole-hand confidence, broadcast to all 21).
6. Flatten to length 180 by interleaving `(x, y, z, mask)` per joint, joint order from `landmarks_meta.json`. **Order matters** — the trained model assumes this exact layout.

### Parity unit test (planned)

`SemaTests/HolisticLandmarkerNormalizationTests.swift`:
- Hard-coded synthetic MediaPipe input (5 frames).
- Expected normalised output captured by running `bvh_to_landmarks.py`'s `normalize_landmarks` on the same input in Python and dumping the result into a `.json` fixture in `SemaTests/Fixtures/`.
- Assert max-abs-diff ≤ 1e-6 between Swift and Python over `NormalizedFrame.values`.

## Streaming window

State per active session:

```
ring  : FrameRing(capacity = 64)
stride: Int = 8                  // frames between inferences
since : Int = 0                  // frames since last inference
```

Per incoming `NormalizedFrame`:
1. `ring.push(frame)`
2. `since += 1`
3. If `ring.isFull && since >= stride`:
   - Snapshot the 64 frames in order → `(64, 180)` Float32 buffer
   - `since = 0`
   - Dispatch async to `GlossTagger.run(snapshot)`

Until the ring is full, no inference runs and no tokens emit — first ~2.1 s of camera input is "warm-up". An on-screen state ("listening…") covers it.

If frames drop (thermal, CPU saturation), `since` keeps incrementing; the next available frame triggers inference. Better to skip a window than to back up.

## Streaming CTC decoder

State:

```
history : Deque<[Int]>           // last N=3 decoded sequences
emitted : [Int]                  // tokens already published downstream
N       : Int = 3
```

Greedy decode:
```
decode(logits):                          // logits shape (64, V)
    pred = argmax(logits, axis=-1)       // length 64
    out  = []
    prev = -1
    for p in pred:
        if p != prev and p != BLANK:
            out.append(p)
        prev = p
    return out
```

Emission:
```
on new inference output (logits):
    seq = decode(logits)
    history.append(seq)
    if history.count < N: return
    stable = longest_common_prefix(history)
    new   = stable[emitted.count ..< stable.count]
    if !new.isEmpty:
        emit(new)                        // publish GlossTokens with confidence = mean softmax peak
        emitted = stable
```

Edge cases:
- **History reset.** If the user stops signing (no hands detected for >2 s), reset `history` and `emitted` so the next utterance starts fresh.
- **Drift bound.** If `emitted.count` grows unbounded (long monologue), trim the prefix older than 10 s — emitted tokens stay in the publisher's downstream buffer.
- **Confidence.** At emission, look up the softmax probability for each new token at its position in the most recent window; attach as `GlossToken.confidence`.

Unit test fixture: hand-crafted CTC logits sequences with known expected emissions.

## Threading & lifecycle

| Component | Isolation | Queue |
|---|---|---|
| `AVCaptureSession` callback | `nonisolated` | Dedicated `DispatchQueue("camera")` |
| `PoseLandmarker`, `HandLandmarker` | `actor` | Each on its own background queue |
| `HolisticLandmarker` | `actor` | Awaits both; joins on timestamp |
| `LandmarkAdapter` | pure functions | Runs on caller's actor |
| `FrameRing` | `actor` | Serializes ring writes |
| `GlossTagger` | `actor` | ANE-bound; serialized inference |
| `StreamingCTCDecoder` | `actor` (owned by tagger) | |
| `PathACoordinator` | `@MainActor` for state, background for plumbing | |

**Pre-warm:** `PathACoordinator.start()` schedules a single `GlossTagger.run(zerosTensor)` on a background queue and a single dummy frame through each MediaPipe landmarker. Expected cost: 300–800 ms total. The user-facing "ready" indicator waits on this.

**Thermal degradation:** `ProcessInfo.thermalStateDidChangeNotification` observer in the coordinator. On `.serious`, stride increases to 16 and the camera frame rate halves to 15 fps. On `.critical`, Path A pauses and a banner appears.

**Memory:**
- One `MLModel` instance, reused. Sleep on `applicationDidEnterBackground`, reload on resume.
- Ring buffer: 64 × 180 × 4 bytes = ~46 KB. Trivial.
- MediaPipe internal allocations: ~150 MB total (validated empirically once we have a build).

## Re-export the recognizer

`recognition/export/to_coreml.py` currently defaults `--seq-len 256`. Two options:

**Option A — flip the default to 64.** Cleanest if we never want a 256-frame artefact again. Edit one line.

**Option B — leave the script flexible, document the deployment command.** Better if we want to keep producing both windows for experiments.

Plan: **Option B**. The deployment artefact comes from this exact invocation, runnable on the Mac with the existing venv:

```bash
cd recognition
.venv/bin/python -m export.to_coreml \
    --ckpt checkpoints/transformer_base/best.pt \
    --out  ../mobile-app/Sema/Models/gloss_tagger.mlpackage \
    --seq-len 64
```

This becomes the canonical command in `docs/data.md` and the `export_coreml` stage in `dvc.yaml` is updated to use `--seq-len 64`. The DVC stage definition needs no other changes — the output path is identical.

Parity vs PyTorch at FP16 is the existing tolerance (5e-2 default). Verified post-export.

## MediaPipe asset pipeline (DVC stage)

New stage in `dvc.yaml`:

```yaml
mediapipe_assets:
  desc: Fetch MediaPipe iOS landmarker .task files (pinned versions).
  cmd: bash mobile-app/scripts/fetch_mediapipe_tasks.sh
  outs:
    - mobile-app/Sema/Resources/pose_landmarker_full.task
    - mobile-app/Sema/Resources/hand_landmarker.task
```

`mobile-app/scripts/fetch_mediapipe_tasks.sh` (new, ~15 lines):

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p Sema/Resources
POSE_URL="https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
HAND_URL="https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
curl -fsSL "$POSE_URL" -o Sema/Resources/pose_landmarker_full.task
curl -fsSL "$HAND_URL" -o Sema/Resources/hand_landmarker.task
```

URL versions are explicit (`/1/`) so we don't silently roll over. When upstream cuts a new version, we bump the URL in the script and `dvc repro mediapipe_assets`.

Total size: ~30 MB (pose ~10 MB + hand ~6 MB). Acceptable in the IPA and in R2.

## Tests (before any real-device run)

1. **`HolisticLandmarkerNormalizationTests`** — exact-parity test against `bvh_to_landmarks.py::normalize_landmarks`. Fixture: 5 synthetic frames, both as MediaPipe-shaped input and as the expected normalised output.
2. **`GlossTaggerLoadTest`** — load the `.mlpackage`, run on a Float32 `(1, 64, 180)` zeros tensor, assert output shape and that no logit is NaN.
3. **`GlossTaggerParityTest`** — load the same checkpoint in Python, run on a fixed random `(1, 64, 180)` seed-derived tensor, save logits to a `.json` fixture in `SemaTests/Fixtures/`. Swift asserts max-abs-diff ≤ 5e-2 (FP16 envelope).
4. **`StreamingCTCDecoderTests`** — three hand-crafted scenarios:
   - Stable token emerges at frame 30, stays in window — emits exactly once.
   - Confusable token oscillates A→B→A — never emits.
   - Long monologue — emitted prefix trims correctly, no duplicate emissions.
5. **`FrameRingTests`** — push past capacity, snapshot order is FIFO.

No real-device tests in this plan — those gate on `recognition/eval/real_camera_smoke.md` once a real iOS build exists.

## Risks and where they bite

| Risk | Likely impact | Mitigation in this plan |
|---|---|---|
| **Sim2real gap** (training projected from BVH; inference from real camera) | Dominant. Tokens never emit; or emit constantly. | Mask channel; `recognition/data/augment.py` MediaPipe-noise model already in place. Real-camera gate at `recognition/eval/real_camera_smoke.md`. |
| **MediaPipe per-frame latency** at 30 fps | If pose+hand together exceed ~33 ms, frames drop, the ring underfills, no inference. | Run pose + hand in parallel actors; on `thermalState >= .serious`, drop camera to 15 fps. |
| **Joint mapping drift** between Python and Swift | Recognizer sees garbage; silently bad accuracy. | The adapter unit test compares against the Python reference. Failing CI gates the merge. |
| **Streaming decoder over-emits** during transitions | Spurious tokens between signs. | N=3 stable-suffix; reset on >2 s of no-hand. Increase N if needed. |
| **ANE warmup latency** on cold start | First inference is 300–800 ms — visible "blank" window. | Pre-warm in `PathACoordinator.start()`; UI shows "ready" only after warmup completes. |
| **History gloss vocab drift** | If `gloss_tagger.vocab.json` and the model disagree, every token is wrong. | The sidecar is co-located with the `.mlpackage`; Swift loads them together and asserts vocab size equals model output dim at load time. |

## Open questions (call out before coding)

1. **Confidence handling downstream.** The plan emits a `confidence` per token. Should the Gemma translator filter low-confidence tokens, or always pass them through? — Decide before wiring Gemma.
2. **No-hand timeout.** 2 s feels right for "stop signing"; calibrate against real users.
3. **Camera framing assumption.** Normalization is shoulder-width-anchored. If the user steps to half-frame or leaves the frame, both shoulders may not be detected — adapter must define behaviour (skip frame? carry last good frame?). Proposal: skip the frame, log a counter; if skips exceed 5 in a row, emit a "lost tracking" event to the coordinator. Confirm.

## What this plan does NOT include

- The Xcode project itself (target, signing, capabilities). Created out-of-band; this plan only specifies the source-file tree.
- Gemma translator integration. Path A here stops at the gloss token stream.
- Avatar rendering and Path B (mic → STT → Gemma → renderer). Separate plans.
- Real-device profiling. Done after a first build lands; results recorded in `mobile-app/docs/memory_budget.md`.
