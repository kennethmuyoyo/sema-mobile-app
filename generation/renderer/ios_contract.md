# Renderer contract — iOS SwiftUI skeleton

This is the contract `mobile-app/Sema/Views/AvatarCanvasView.swift` implements against. The contract is dimensionally identical to the **recognizer's** training-time joint layout, so there is exactly one joint ordering in this project — see `../../recognition/data/landmarks_meta.json` for the authoritative list.

## Input stream

A `StitchedPoseStream` from `../stitching/stitch.py` (or the iOS port in `Sema/Generation/Stitcher.swift`):

| Field | Type | Notes |
|---|---|---|
| `frames` | `[Float]` flattened, shape `(T, 45, 3)` | 45-joint normalized landmark vectors, same ordering as `landmarks_meta.json` |
| `frame_rate` | `Float` | Hz; typically 24 from BVH source. Renderer may upsample for smoothness. |
| `glosses` | `[GlossSpan]` | per-gloss frame ranges, for caption rendering |

Coordinates are normalized: shoulder-midpoint origin, shoulder-width unit scale. `x, y` are image-plane (right, down). `z` is relative depth (positive = away from camera) used for parallax / line-thickness cues; the renderer is otherwise 2D.

## Skeleton edges (drawn as line segments)

| Section | Joint pairs |
|---|---|
| Spine | `head` → midpoint(`left_shoulder`, `right_shoulder`) → midpoint(`left_hip`, `right_hip`) |
| Eyes | `left_eye_smplhf` ↔ `right_eye_smplhf` (visual only) |
| Left arm | `left_shoulder` → `left_elbow` → `left_wrist` |
| Right arm | `right_shoulder` → `right_elbow` → `right_wrist` |
| Left leg | `left_hip` → `left_knee` → `left_ankle` |
| Right leg | `right_hip` → `right_knee` → `right_ankle` |
| Left hand | `left_wrist` → `left_{finger}1` → `left_{finger}2` → `left_{finger}3` for each of {thumb, index, middle, ring, pinky} |
| Right hand | mirror of left hand |

Joints are drawn as filled circles; edges as lines. Line thickness ∝ `1 - clamp(z, 0, 1)` to give a soft depth cue.

## Frame pacing

The renderer drives off a `CADisplayLink` and resamples the input stream to display refresh (60–120 Hz). When `ProcessInfo.thermalState >= .serious`, drop the resampling to source rate (no interpolation).

## What the renderer does NOT do

- Inverse kinematics, retargeting, or motion synthesis. It draws what it's given.
- Face landmarks. The recognizer's joint set is body + hands only.
- Audio. Audio is handled in `Sema/Speech/Synthesizer.swift`.
