# Renderer contract — iOS SwiftUI skeleton

This is the contract `mobile-app/sema/sema/AvatarCanvasView.swift` implements against. The contract is dimensionally identical to the **recognizer's** training-time joint layout in [`../../recognition/data/landmarks_meta.json`](../../recognition/data/landmarks_meta.json), so there is one canonical joint set in this project: **45 normalized body-and-hand landmarks** (15 body + 15 left hand + 15 right hand).

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

The 45 landmarks are 15 body + 15 finger joints per hand:

- **Body (15):** `head`, `left_eye_smplhf`, `right_eye_smplhf`, left/right shoulders, elbows, wrists, hips, knees, ankles.
- **Each hand (15):** three joints (MCP, PIP, DIP) per finger × five fingers (thumb, index, middle, ring, pinky). The wrist used by the hand chain is the same joint as the arm chain's `*_wrist`.

Joints are drawn as filled circles or humanized hand/body shapes; edges are used as the geometric source of truth. Line thickness ∝ `1 - clamp(z, 0, 1)` to give a soft depth cue.

## Synthesized renderer-only joints

The renderer may compute additional cosmetic anchor points that **do not** live in the recognizer's 45-joint set:

- `*_body_wrist` — interpolated 85 % of the way from elbow to wrist on each side. Used to round the forearm/hand joint visually; never sent to the recognizer or the pose database.

If you add more renderer-only joints, document them here and keep them strictly downstream of the recognizer — they must never appear in `landmarks_meta.json` or the recognizer's input shape.

## Frame pacing

The renderer drives off `TimelineView(.animation)` / `CADisplayLink` and resamples the input stream to display refresh (60–120 Hz). When `ProcessInfo.thermalState >= .serious`, drop the resampling to source rate (no interpolation).

## What the renderer does NOT do

- Inverse kinematics, retargeting, or motion synthesis. It draws what it's given.
- Face landmarks. The recognizer's joint set is body + hands only.
