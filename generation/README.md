# generation/

**Path B, stage 3** — gloss tokens to an animated signing avatar.

This folder owns the **on-device pose database** and the **stitching/renderer contract** the iOS app implements against. Given a gloss sequence from `../gemma-glossing/`, it provides each gloss's pre-recorded pose clip, blends adjacent clips, and hands a continuous keypoint stream to the SwiftUI renderer in `../mobile-app/`.

This is **retrieval + playback**, not motion synthesis. Per [../README.md](../README.md), an RVQ-VAE-based motion generator is **explicitly out of scope**: retrieval over a curated, quantised pose database is honest, demoable, and avoids fabricating signs that do not exist in KSL. The Motion-S `train.csv` already contains 6×512 RVQ tokens per clip (`base_tokens, residual_1..5`), but we use those tokens only as an auxiliary supervision signal for the recognizer (see `../recognition/README.md`) — we do not build a generative decoder from them.

ASR (SFSpeechRecognizer continuous) is upstream of this folder, but its integration lives in `../mobile-app/`. This folder owns the contract it expects from ASR; it does not bundle an ASR model.

## What this folder produces

- An indexed pose library keyed by gloss token, **int8-quantised**, bundled in the iOS app.
- A stitcher/blender that produces a continuous keypoint stream from a gloss sequence.
- A renderer contract spec consumed by `../mobile-app/`'s SwiftUI skeleton view.
- An ASR contract spec consumed by `../mobile-app/`'s SFSpeechRecognizer wrapper.

## Bundle-size estimate

Full vocabulary (≈ 3,800 unique gloss tokens), ~3 s average per clip at 24 fps, 47 joints × 3 coords × int8 ≈ 27 KB/clip → **~100 MB** total. Acceptable inside an App Store binary; if it grows, fall back to iOS on-demand resources keyed by gloss token. Document the actual bundle size in `../mobile-app/README.md` once `pose_library/build_index.py` has run.

## Repo layout

```
generation/
├── README.md
├── asr/
│   └── contract.md                  # SFSpeechRecognizer continuous-mode contract
├── renderer/
│   └── ios_contract.md              # keypoint-stream format the iOS renderer consumes
├── llm_export/                      # LoRA → LiteRT/TFLite staging (artefacts git-ignored)
└── pose_library/
    ├── build_index.py               # ingest Motion-S → quantised per-gloss clips (~31 demo glosses)
    ├── build_full_library.py        # alignment-derived library (~406 glosses)
    ├── build_single_gloss.py        # iterate one gloss at a time
    └── pick_best_takes.py           # curate exemplars per gloss
```

Stitching / blending is implemented in Swift in `../mobile-app/sema/sema/Generation/Stitcher.swift` rather than Python — the avatar renderer composes clips at runtime, so the Python side stops at the indexed clip library.

**Not in git** (regenerated locally per workstation; see `../.gitignore`):

```
generation/pose_library/
├── clips/                           # int8 per-gloss keypoint clips (.npz)
├── full/                            # full ~406-gloss library output
├── rotations/                       # BVH-derived rotation streams
├── index.json                       # gloss_token → {clip_path, frame_count, …}
├── retarget_to_target.py            # source-skeleton → target-skeleton retargeter
└── target_bind_pose.json            # cached bind pose
```

## Inputs and outputs

- **Input:** a gloss token sequence from `../gemma-glossing/` (Path B direction).
- **Output:** a time-stamped keypoint stream for the SwiftUI/SpriteKit skeleton renderer in `../mobile-app/`. Joint ordering matches `../recognition/data/landmarks_meta.json`, so the recognizer's training-time layout and the avatar's runtime layout are identical — one source of truth.

## Dataset

The pose-clip database is derived from **Motion-S** BVH (forward-kinematics → MediaPipe-equivalent 47-joint layout, the same projection used by `../recognition/data/bvh_to_landmarks.py`). One clip per gloss; if a gloss appears multiple times in Motion-S, we select the longest cleanly-segmented exemplar.

