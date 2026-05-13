# Sema — Architecture & References

This document describes the technical architecture of Sema, the two pipelines that compose the system, the reference implementations we draw from, and the licensing constraints on the datasets we use.

## Design principle

Sema is **not** a single end-to-end model. Gemma 4 — even multimodal Gemma 4 — is a still-image-plus-text model. It does not consume video, it does not consume time-series pose data, and it cannot, by fine-tuning alone, learn to recognise or generate motion.

Sema therefore decomposes the problem into the two tasks each model class is actually suited to:

1. **Motion recognition (signs → gloss tokens).** Handled by a small specialised temporal model over MediaPipe Holistic keypoints. Not Gemma.
2. **Grammar-aware translation (gloss ↔ spoken-language sentences).** Handled by Gemma 4 E4B, fine-tuned via Unsloth on KSL gloss / English-Swahili sentence pairs.

This decomposition is the project's Cactus-track thesis: route each sub-task to the right kind of model, on-device, with intelligent activation.

---

## The two pipelines

### Path A — KSL recognition (Deaf user signs → hearing user hears speech)

```
Camera frames
  ↓
MediaPipe Holistic
  → per-frame keypoints (33 body + 21 + 21 hands, optionally face)
  ↓
Gloss Tagger (small temporal model, ~5 MB, NOT Gemma)
  → gloss token stream, e.g. "HOSPITAL TOMORROW I-GO"
  ↓
Gemma 4 E4B (LiteRT, INT4, fine-tuned via Unsloth)
  → fluent Swahili / English: "I am going to the hospital tomorrow"
  ↓
Android TTS
  → audio out
```

**Pieces:**

| Piece | Role | Model | Why |
|---|---|---|---|
| Pose extraction | Per-frame skeletal keypoints | MediaPipe Holistic | Fast, on-device, well-supported on Android |
| Gloss tagger | Pose sequence → gloss token | Small Transformer / LSTM over keypoints | Right tool for temporal classification; tiny and quick |
| Translator | Gloss sequence → fluent sentence | Gemma 4 E4B, fine-tuned | True low-resource MT task; KSL grammar ≠ English grammar |
| TTS | Text → audio | Android system TTS (Swahili / English) | Built-in, no extra model footprint |

### Path B — Speech generation (hearing user speaks → Deaf user sees signs)

```
Microphone audio
  ↓
Whisper-tiny (or Android on-device STT)
  → text (Swahili / English)
  ↓
Gemma 4 E4B (LiteRT, INT4, fine-tuned via Unsloth)
  → KSL gloss sequence, respecting KSL grammar
    (topicalization, time-before-event, classifier predicates)
  ↓
Stickman renderer
  → look up each gloss in the KSL Pose Dataset
  → stitch pose sequences and play back as animated avatar
```

**Pieces:**

| Piece | Role | Model | Why |
|---|---|---|---|
| ASR | Audio → text | Whisper-tiny or Android STT | Swahili support, small, on-device |
| Translator | Sentence → gloss sequence | Gemma 4 E4B, fine-tuned (same model as Path A) | Bidirectional fine-tune; KSL grammar generation |
| Renderer | Gloss sequence → animated stickman | Retrieval + playback from KSL Pose Dataset | Honest, demoable in 7 days, no motion-synthesis model required |

The same fine-tuned Gemma 4 instance serves both directions, with task tokens (`[KSL→EN]`, `[EN→KSL]`, `[KSL→SW]`, `[SW→KSL]`).

---

## Reference implementations

We draw from two distinct tiers of prior work. Tier 1 informs the on-device pipeline style; Tier 2 informs the scale and the temporal-model architecture.

### Tier 1 — MediaPipe + light temporal model (on-device, small vocabulary)

These are the closest references for the **pipeline shape** of our gloss tagger: MediaPipe Holistic for keypoint extraction, followed by a small temporal classifier (LSTM, GRU, or light Transformer), running locally on a CPU or mobile device. The vocabularies in these projects are small (3–36 signs), but the code organisation and on-device deployment patterns are directly applicable.

| Project | Link | Notes |
|---|---|---|
| **SLRNet** (Khushi-739) | https://github.com/Khushi-739/SLRNet | Most recent and best-documented. MediaPipe Holistic → stacked LSTM, 543 landmarks/frame, 30-frame sequences, 86.7% validation accuracy on 26 ASL letters + 10 functional words. Accompanying paper: arXiv:2506.11154. **Our primary code-scaffold reference.** |
| **Sign-Language-Detection-Using-LSTM** (evarghese563) | https://github.com/evarghese563/Sign-Language-Detection-Using-LSTM | Classic 3-sign demo. Good pedagogical reference for the MediaPipe-Holistic-to-LSTM data flow. |
| **Sign-Language-Recognition** (RSBhoomika) | https://github.com/RSBhoomika/Sign-Language-Recognition | Similar pattern, MediaPipe + LSTM. |
| **AI-Based-Sign-Language-Recognition** (gianlucapargaetzi) | https://github.com/gianlucapargaetzi/AI-Based-Sign-Language-Recognition | Includes an iOS demo app — closest existing example of MediaPipe-based classifier deployed to a real phone. |
| **Sign-Language-Recognition–MediaPipe-DTW** (gabguerin) | https://github.com/gabguerin/Sign-Language-Recognition--MediaPipe-DTW | Uses Dynamic Time Warping rather than a learned classifier. Useful as a no-training-required baseline / fallback. |

### Tier 2 — WLASL ecosystem (large vocabulary, research-grade)

These references inform our **modelling choices** at vocabulary scale and provide an option for transfer-learning initialisation (subject to the licensing notes below). The WLASL dataset contains 21,083 videos across ~2,000 ASL signs from 119 signers.

| Project | Link | Notes |
|---|---|---|
| **WLASL (canonical)** (dxli94) | https://github.com/dxli94/WLASL | The WACV 2020 reference repo. Provides pre-trained **I3D** weights for WLASL100/300/1000/2000 and pre-trained **Pose-TGCN** weights using body keypoints. The Pose-TGCN branch is architecturally the closest precedent to what Sema needs: temporal classification over pose graphs. |
| **Sign-Language-Recognition (I3D + Transformer)** (sumedhsp) | https://github.com/sumedhsp/Sign-Language-Recognition | More recent. I3D feature extraction + Transformer temporal modelling on WLASL. Pre-trained weights available for 100, 300, 1000 subsets. Reference for the modern Transformer-over-pose pattern. |
| **WLASL-CLASSIFICATION** (oussamaouardini) | https://github.com/oussamaouardini/WLASL-CLASSIFICATION | Comparative study (ConvLSTM / LRCN / CNN) on WLASL. Useful for understanding architecture tradeoffs. |

### Temporal-model choice for Sema's gloss tagger

We adopt a **Transformer encoder over MediaPipe Holistic keypoint sequences**, architecturally derived from SLRNet (Khushi-739, 2025) for the pipeline shape and from the Pose-TGCN baseline (Li et al., WACV 2020) for the pose-input modelling. LSTM remains a fallback if Transformer training is unstable on our smaller KSL Pose Dataset.

Optional initialisation from WLASL pose-pretrained weights, followed by fine-tuning on the KSL Pose Dataset, is under consideration as a transfer-learning experiment — pose-level features (hand shapes, trajectories) generalise across sign languages at the low level even though the linguistic gloss space differs. **License compatibility for this step must be confirmed before use** (see below).

### Translation model

Gemma 4 E4B, fine-tuned via Unsloth on AI4KSL English–KSL gloss pairs. Bidirectional with task tokens. Deployed via Google AI Edge LiteRT, INT4 quantised, on Android.

---

## Datasets

### Primary datasets

| Dataset | Used for | Source | Citation |
|---|---|---|---|
| **Motion-S** | Fine-tuning Gemma 4 on KSL gloss ↔ English/Swahili pairs | Signvrse Kaggle Dataset |
| **KSL Word-Based Pose Dataset** | Training the gloss tagger; rendering the stickman avatar | 

### Optional supplementary dataset


### Licensing

- **Motion-S** — released by Signvrsey; we treat access as research-purpose under the authors' terms. Citation and credit required. Verify any redistribution constraints with the authors before publishing fine-tuned weights derived from this dataset.

This documentation, and the Sema submission, are explicitly non-commercial research artefacts and consistent with all three licences.
