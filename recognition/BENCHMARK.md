# KSL phonological-intermediate sign recognition — benchmark suite

A protocol and baseline for evaluating sign-language recognition methods on
the KSL MediaPipe-rendered corpus, with a compositional phonological
intermediate. **Multi-seed (≥3), layered metrics, stratified by data
density** — the v4-vs-v3 single-seed drift (+1.65 → +0.14 pp on the same
ablation) is the failure mode this protocol is designed to prevent.

The corresponding notebook (`recognition/phonological_pipeline.ipynb`)
implements the full suite and produces every number in this document.

---

## 1. Dataset specification

| | Value |
|---|---|
| Source corpus | 12,467 KSL BVH sentences (`data/Train/<id>/<id>.bvh`) |
| Currently rendered through MediaPipe | 756 (`data/mediapipe_landmarks/`) |
| Aligned isolated-sign segments | 3,629 |
| Segmentation | Visibility-burst + finger-velocity-peak; targeted at GT gloss count |
| Unique glosses (in 756-clip subset) | 1,133 |
| Median examples / gloss | 1 |
| Demo subset | 42 glosses with ≥10 training examples |
| Train / val split | 80 / 20 random by segment, seed = 17 (deterministic) |
| Window | 64 frames @ 24 fps (≈ 2.7 s) |
| Input dim | 45 joints × xyz = 135 / frame, shoulder-normalised |
| Phonological label dim | 10 per segment (5 features × 2 hands) |

Per-gloss density today vs at full 12K corpus scale (80/20 projection):

| threshold | currently rendered | full 12K render |
|---|---|---|
| glosses with ≥3 train ex. | 251 | 1,534 |
| glosses with ≥5 train ex. | ~170 | 1,044 |
| glosses with ≥10 train ex. | 42 | 660 |
| glosses with ≥20 train ex. | 14 | 395 |

Until the BVH→MediaPipe render queue completes, the metrics below should be
read as a **lower bound** on what's achievable.

---

## 2. Required metric suite

A method evaluating on this corpus should report **all** of the following.
Single-number top-1 accuracy is insufficient — the data distribution
(median 1 example per gloss) makes any aggregate over the full vocabulary
misleading.

### 2.1 Gloss top-k (primary recognition)

| subset | report |
|---|---|
| **All vocab** (1,133 classes) | top-1, top-3, top-5, mean rank, **mean ± std over ≥3 seeds** |
| **Demo vocab** (42 classes, ≥10 train ex.) | top-1, top-3, top-5, mean rank, **mean ± std over ≥3 seeds** |

### 2.2 Stratified-by-density top-1 (data-scale dependence)

| per-gloss train count | n_glosses | top-1 |
|---|---|---|
| 1–2 | — | — |
| 3–9 | — | — |
| 10–19 | — | — |
| 20+ | — | — |

This is more informative than the average — it exposes whether a method's
gain comes from the dense head of the distribution or from the long tail.

### 2.3 Phonological description quality

If the method has a compositional intermediate, report:

- **Per-feature aux accuracy** for each of the 10 phonological features
- **Mean features correct / instance** (out of 10)
- **Mean features correct on top-1 MISSES** (near-miss quality)

### 2.4 Retrieval-style eval (dense subset, ≥3 train ex.)

- **Top-1 / top-5** retrieval rank (Hamming distance over predicted phonological feature IDs vs per-gloss training centroids)
- **Hamming spread on errors**: mean Hamming(pred → TRUE) vs mean Hamming(pred → RANDOM) over ≥30 random centroids per error
- **Random baseline**: 1 / |dense gloss set|

### 2.5 Compositional generalisation (the §17 protocol)

For each of (at minimum) 3 feature pairs × 3 combinations per pair:

1. Pick a (feature_A_value, feature_B_value) combination with 12–40 instances.
2. Hold out all instances with that combination from training.
3. Hold out a random subset of equal size (control).
4. Train.
5. Report:
   - Per-head accuracy on **combo held-out** for both feat A and feat B
   - Per-head accuracy on **random held-out** (control)
   - **Δ (compositional cost)** = combo-acc − random-acc, per head

| feature pair | Δ feat A | Δ feat B |
|---|---|---|
| loc × hs | — | — |
| loc × palm | — | — |
| hs × palm | — | — |

Negative Δ = the head depends on the joint distribution. Positive Δ = the
head generalises compositionally. A competing method's contribution to
"compositional sign recognition" should push these closer to zero.

### 2.6 Diagnostics (mandatory)

- **Train / val gap** on gloss accuracy — overfitting check
- **Number of trainable parameters**
- **Training wall time per seed**

---

## 3. v5 baseline numbers (the reference to beat)

3 seeds {17, 23, 42}; mean ± std unless stated.

### 3.1 Gloss top-k

| | top-1 | top-3 | top-5 | mean rank |
|---|---|---|---|---|
| All vocab (1,133) | 23.26 ± 0.65 % | 29.01 ± 0.35 % | 31.54 ± 0.56 % | 306.7 |
| **Demo vocab (42)** | **53.00 ± 2.13 %** | **62.01 ± 1.82 %** | **66.52 ± 1.71 %** | **5.95** |

### 3.2 Stratified-by-density top-1

| train count | n_glosses | top-1 |
|---|---|---|
| 1–2 | 734 | 13.42 ± 0.75 % |
| 3–9 | 209 | 20.92 ± 0.81 % |
| 10–19 | 28 | 29.96 ± 2.64 % |
| **20+** | **14** | **59.44 ± 1.85 %** |

The slope of this curve is the dominant predictor of method quality at
this corpus scale. A method that lifts the 1–2 bucket without lifting the
20+ bucket would be unusual; the reverse (lifting 20+ without 1–2) is the
data-saturation regime.

### 3.3 Phonological description quality

Per-feature aux accuracy:

| feature | accuracy |
|---|---|
| right_loc | 82.16 ± 1.34 % |
| left_palm | 81.20 ± 0.35 % |
| right_palm | 79.82 ± 0.80 % |
| left_loc | 72.97 ± 0.72 % |
| left_hs | 70.53 ± 1.12 % |
| left_mt | 63.36 ± 1.24 % |
| right_mt | 59.17 ± 3.37 % |
| right_md | 58.02 ± 1.66 % |
| right_hs | 51.49 ± 1.87 % |
| left_md | 36.60 ± 0.48 % |

Aggregate:
- Mean features correct (all instances): **6.55 ± 0.09 / 10**
- Mean features correct on top-1 MISSES (near-miss): **6.19 ± 0.10 / 10**

The near-miss number is the key claim: **when the gloss head is wrong, the
phonological description is still ~62 % correct.**

### 3.4 Retrieval-style eval (dense subset, 251 glosses)

| | mean ± std (3 seeds) |
|---|---|
| Top-1 | 23.98 ± 1.83 % |
| Top-5 | 36.18 ± 2.07 % |
| Hamming spread on errors | +1.30 ± 0.04 |
| Random baseline | 0.40 % |

### 3.5 Compositional generalisation matrix

3 combinations × 3 seeds per pair; reported as mean Δ across combinations:

| feature pair | Δ feat A | Δ feat B |
|---|---|---|
| loc × hs | −16.0 pp | −51.4 pp |
| loc × palm | −25.9 pp | −63.6 pp |
| hs × palm | −67.0 pp | −32.9 pp |

**All feature pairs fail compositionally** at this data scale. A clean
asymmetry observed on a single combination in earlier v4 runs (location +8.3
pp, handshape −86.7 pp) **did not replicate** when averaged across
combinations — a single-(combo, seed) artifact. The honest baseline:
compositional generalisation across held-out combinations is weak across
the board.

Two-stream architecture (separate body and hand encoders) tested on the
loc × hs pair: **mean Δ_hs improvement = +2.3 pp** (within seed-variance
noise band). Architectural priors alone don't fix this at current scale.

### 3.6 Diagnostics

- Recognizer trainable params: **0.45 M**
- VQ-VAE pretrain: 20 epochs, ~70 s CPU
- Recognizer training: 20 epochs, ~120 s CPU per seed
- **Train / val gloss-accuracy gap: +55 pp** (77.0 / 22.3) — heavy overfit
- Aux signal vs single-permutation shuffled control: **+28.9 pp** (real phonology, not rule-correlation)

---

## 4. Reporting protocol for new methods

A paper claiming improvement over this baseline should:

1. Use the same train/val split (random seed = 17, ratio 80/20 by segment).
2. Run **≥ 3 seeds** for every reported number; report mean ± std.
3. Fill **all six sections** of §2 (top-k, stratified, phonological if applicable, retrieval, compositional Δ, diagnostics).
4. State which subset of the data was used (756 clips vs. full 12K render).
5. Disclose the limitations of §5 explicitly.

If a paper reports only top-1 on the full vocab without stratification or
seed variance, the comparison is incomplete. The v4-vs-v3 drift on the
same ablation (+1.65 vs +0.14 pp) demonstrates this concretely: one of
those numbers misrepresented the architecture.

---

## 5. Limitations of this baseline (mandatory to acknowledge)

1. **Single-signer rendering** (SMPL-X neutral mesh). Cross-signer
   generalisation is not measurable at this data scale.
2. **BVH-synthetic input** — real-MediaPipe-from-video has occlusion,
   lighting, and tracking-jitter distributions that the rendered corpus
   does not capture. Models trained here may not transfer without
   domain adaptation.
3. **Forced 1 : 1 sentence segmentation** to GT gloss count.
   Boundary positions are heuristic (visibility bursts + finger-velocity
   peaks); segmentation error is not separated from recognition error.
4. **+55 pp train / val gap** indicates the model is at its data-limit
   ceiling, not its model-limit ceiling. Most architectural variants at
   this scale will be within noise.
5. **Held-out-*gloss* zero-shot is not tested.** §2.4 is held-out-*instance*
   retrieval (the gloss appears in training; only the specific instance
   is held out). True open-vocabulary zero-shot needs the dense-gloss
   pool that only the full 12K render unlocks.
6. **No real-camera baseline.** Inference performance on actual
   MediaPipe-from-video input is unmeasured.

---

## 6. Comparison context (sanity-check anchors only — not direct comparisons)

Different datasets, different vocab sizes, different input modalities
(video vs landmarks). These are orientation references, not benchmarks
against which to claim parity:

| benchmark | vocab | top-1 (typical) |
|---|---|---|
| WLASL-100 (ASL, video) | 100 | 65–85 % |
| WLASL-2000 (ASL, video) | 2000 | 30–40 % |
| MS-ASL-100 | 100 | 50–65 % |
| **This work — demo subset** | **42** | **53 %** |
| **This work — full vocab** | **1,133** | **23 %** |

Our 42-class 53 % top-1 sits in the typical range for moderate-vocab
isolated SLR. The all-vocab 23 % reflects the unfair distribution (median
1 example per class) rather than a fundamental model limitation; the
stratified 20+-bucket 59 % top-1 is the more honest indicator of what
the architecture can do when given enough data per class.

---

## 7. Roadmap (open work)

1. **Complete BVH → MediaPipe render** for the remaining 11.7K BVH files.
   Unlocks the held-out-gloss zero-shot test (§2.4's caveat).
2. **Decouple handshape from location.** Two-stream encoder (§3.5) gave
   only +2.3 pp — architectural priors alone don't fix the entanglement.
   Worth trying: gradient-reversal between feature heads; conditional
   training of handshape only on segments where finger visibility > 0.5.
3. **Real-MediaPipe-from-video baseline.** Record demo signers using the
   iOS app, run the trained model on real input, report the
   synthetic→real generalisation gap.
4. **Multi-signer rendering.** Render the same BVH animations through
   multiple SMPL-X body parameters to introduce signer variance, then
   measure cross-signer generalisation.

---

*Numbers in §3 are taken from the v5 run logged in
`recognition/phonological_pipeline.ipynb`. Regenerating that notebook
end-to-end reproduces every number here (deterministic up to floating-
point determinism on CPU; std bands derived from seeds 17/23/42).*
