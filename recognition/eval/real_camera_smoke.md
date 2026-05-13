# Real-camera deployment gate

Before declaring a synthetic-trained gloss-tagger checkpoint deployable to the iOS app, it must pass this checklist on **real MediaPipe iOS output**, not on synthetic BVH-projected features. This gate exists because the dominant risk for this model is the sim2real gap between BVH-projected training landmarks and live MediaPipe Holistic landmarks.

## Pre-requisites

- A built iOS dev build of the app with a debug toggle that **logs MediaPipe landmark sequences to JSON** alongside the inferred gloss output, for each recorded clip.
- A small calibration set: 5–10 short clips you record yourself in roughly app-realistic conditions (front camera, indoor/outdoor lighting, neutral background). Sign one well-known KSL gloss per clip (drawn from the top-100 frequency tier so the model has plenty of training exposure).

## Pass criteria

| # | Check | Pass condition |
|---|---|---|
| 1 | Top-1 gloss matches the intended sign | ≥ 6 / 10 clips |
| 2 | Top-3 gloss contains the intended sign | ≥ 9 / 10 clips |
| 3 | The greedy CTC output is non-degenerate (not all blanks, not all `<unk>`) | All clips |
| 4 | Per-clip end-to-end latency on iPhone 13 or 15 | < 300 ms after the last MediaPipe frame |
| 5 | No `NaN` / `Inf` in logits across the calibration set | All clips |
| 6 | Activations on the **dropout-mask channel** are present when MediaPipe drops hand landmarks | Logged for ≥ 1 clip |

## Failure modes and what they mean

| Symptom | Likely cause | Fix path |
|---|---|---|
| All clips decode to `<blank>` | sim2real gap too large; model didn't generalize | Strengthen `data/augment.py` (more dropout, more affine jitter); collect a small real-MediaPipe set and train a domain-adapter head |
| Top-1 wrong but top-3 right | recognizer is "close" but motion-detail signals are wrong | Increase per-joint jitter magnitude in augmentation; inspect whether finger joints are being dropped at training time too aggressively |
| Latency exceeds budget | Model too big for ANE / falls back to CPU | Re-export with explicit `coremltools.precision.FLOAT16`; shrink `d_model` or `n_layers` in `configs/transformer_base.yaml` |
| `NaN`s in logits | Normalization stats mismatch between training and runtime | Verify `landmarks_meta.json` is bundled and `Sema/ML/GlossTagger.swift` applies identical normalization |

## What we do NOT gate on here

- Translation quality. That's `gemma-glossing/`'s eval gate.
- Avatar visual quality. That's a UI gate, not a recognition gate.
- Battery / thermals. Tracked separately in `mobile-app/docs/memory_budget.md`.
