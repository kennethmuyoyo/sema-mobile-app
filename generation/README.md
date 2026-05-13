# generation/

**Path B, stage 3** — gloss tokens to an animated signing avatar.

This folder owns the **stickman renderer**. Given a gloss sequence from `../gemma-glossing/`, it looks each gloss up in the KSL Word-Based Pose Dataset, stitches the per-gloss pose clips into a continuous sequence, and plays it back as an animated avatar in the Android client.

This is deliberately **retrieval + playback**, not motion synthesis. Per [../README.md](../README.md), a generative motion model is out of scope for the 7-day demo target and would require resources we do not have; retrieval over a curated pose dataset is honest, demoable, and avoids fabricating signs that do not exist in KSL.

ASR (Whisper-tiny / Android STT) is upstream of this folder, but its integration lives in `../mobile-app/`. This folder only describes the contract it expects from ASR; it does not bundle an ASR model.

## What this folder produces

- An indexed pose library keyed by gloss token.
- A stitcher/blender that produces a continuous keypoint stream from a gloss sequence.
- A renderer spec consumed by `../mobile-app/` to draw the stickman on-screen.

## Intended layout

```
generation/
├── README.md
├── requirements.txt
├── pose_library/
│   ├── build_index.py             # ingest KSL Word-Based Pose Dataset
│   ├── index.json                 # gloss_token -> clip_id, frame_range
│   └── clips/                     # per-gloss keypoint clips (.npz)
├── stitching/
│   ├── stitch.py                  # concatenate clips with handoff frames
│   ├── interpolate.py             # linear / spline blending between glosses
│   └── timing.py                  # per-gloss duration, pauses, emphasis
├── renderer/
│   ├── stickman_spec.md           # joint set, edges, coordinate frame
│   ├── reference_renderer.py      # Python preview renderer (matplotlib)
│   └── android_contract.md        # what mobile-app/ expects to receive
├── asr/
│   └── contract.md                # expected text format from mobile-app/
├── tests/
│   ├── test_index.py
│   └── test_stitch.py
└── notebooks/
    └── render_preview.ipynb
```

## Inputs and outputs

- **Input:** a gloss token sequence from `../gemma-glossing/` (Path B direction).
- **Output:** a time-stamped keypoint stream for the Android stickman renderer in `../mobile-app/`.

## Dataset

**KSL Word-Based Pose Dataset.** One pose clip per gloss is the unit of retrieval. The same dataset is reused by `../recognition/` for training the gloss tagger, so the gloss vocabulary in `pose_library/index.json` must stay aligned with `../gemma-glossing/data/vocab/gloss_vocab.json`.
