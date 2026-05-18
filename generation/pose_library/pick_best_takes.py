"""Pick the best take per gloss from `data/alignments.json`.

Reads the per-sentence alignment table produced by
`recognition/data/align_sentences.py`, scores every candidate
`(sentence_id, gloss, start, end, score)` tuple, and writes a flat manifest
of the single best take per gloss to `data/best_takes.json`:

  {
    "<GLOSS>": {
      "sentence_id": 1234567,
      "start": 18,
      "end":   47,
      "peak_frame": 32,
      "alignment_score": 0.83,
      "quality": 0.42      # combined score used for ranking (see below)
    },
    ...
  }

The quality score is:

  quality = alignment_score
            * duration_factor(end - start)
            * boundary_factor(start, end, n_frames)
            * (0.7 if peak_frame == -1 else 1.0)   # in-vocab penalty

`duration_factor` is 1.0 if 0.5s ≤ duration ≤ 3.0s @ 24fps (12–72 frames),
falling off linearly outside that band. `boundary_factor` discounts spans
that hit the first or last 15% of the sentence (coarticulation is
usually nastiest at the edges).

By default the picker only considers glosses listed in
`ksl_model.metadata.demo_glosses` — the 705 high-support ones we'd actually
want to render on the avatar.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ALIGNMENTS_IN = REPO_ROOT / "data/alignments.json"
METADATA = REPO_ROOT / "mobile-app/sema/sema/Resources/ksl_model.metadata.json"
BEST_TAKES_OUT = REPO_ROOT / "data/best_takes.json"

DEFAULT_FPS = 24.0
DUR_MIN_F = 12     # 0.5 s at 24 fps
DUR_MAX_F = 72     # 3.0 s at 24 fps
EDGE_FRAC = 0.15   # peaks in the first/last 15% are slightly penalised


def duration_factor(duration: int) -> float:
    if duration < 6:
        return 0.0
    if duration < DUR_MIN_F:
        return duration / DUR_MIN_F
    if duration <= DUR_MAX_F:
        return 1.0
    # Allow up to 2× max with linear falloff to 0.2 at the cap.
    cap = DUR_MAX_F * 2
    if duration >= cap:
        return 0.2
    return 1.0 - 0.8 * (duration - DUR_MAX_F) / (cap - DUR_MAX_F)


def boundary_factor(peak_frame: int, n_frames: int) -> float:
    if peak_frame < 0 or n_frames <= 1:
        return 0.0
    rel = peak_frame / max(1, n_frames - 1)
    if EDGE_FRAC <= rel <= 1.0 - EDGE_FRAC:
        return 1.0
    # Linear falloff to 0.5 at the very edge.
    if rel < EDGE_FRAC:
        return 0.5 + 0.5 * (rel / EDGE_FRAC)
    return 0.5 + 0.5 * ((1.0 - rel) / EDGE_FRAC)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--target-set", default="demo",
                    choices=["demo", "all"],
                    help="`demo` = 705 demo_glosses; `all` = every gloss seen in alignments")
    ap.add_argument("--min-score", type=float, default=0.05,
                    help="drop candidates whose raw alignment score is below this")
    ap.add_argument("--out", type=Path, default=BEST_TAKES_OUT)
    args = ap.parse_args()

    print(f"Loading {ALIGNMENTS_IN} ...")
    alignments = json.loads(ALIGNMENTS_IN.read_text())
    print(f"  {len(alignments):,} sentences")

    meta = json.loads(METADATA.read_text())
    demo_glosses = set(meta["demo_glosses"])
    print(f"  demo_glosses (target): {len(demo_glosses)}")

    if args.target_set == "demo":
        target = demo_glosses
    else:
        target = None   # accept everything

    # Group candidates by gloss.
    by_gloss: dict[str, list[dict]] = defaultdict(list)
    for sid, payload in alignments.items():
        n = payload["n_frames"]
        for s in payload["spans"]:
            g = s["gloss"]
            if target is not None and g not in target:
                continue
            if s["score"] < args.min_score:
                continue
            duration = s["end"] - s["start"]
            q = (s["score"]
                 * duration_factor(duration)
                 * boundary_factor(s["peak_frame"], n))
            if q <= 0:
                continue
            by_gloss[g].append({
                "sentence_id": int(sid),
                "start": s["start"],
                "end":   s["end"],
                "peak_frame": s["peak_frame"],
                "alignment_score": s["score"],
                "n_frames_sentence": n,
                "quality": q,
            })

    # Pick the top-1 per gloss.
    best: dict[str, dict] = {}
    for g, cands in by_gloss.items():
        cands.sort(key=lambda c: c["quality"], reverse=True)
        best[g] = cands[0]

    # Summary stats.
    if target is not None:
        covered = sum(1 for g in target if g in best)
        print(f"\nCovered {covered:,} / {len(target):,} demo glosses")
        missing = sorted(target - best.keys())
        if missing:
            print(f"  missing {len(missing)} (first 20): {missing[:20]}")

    # Histogram of quality scores.
    if best:
        qs = sorted(b["quality"] for b in best.values())
        n = len(qs)
        print(f"\nQuality distribution over {n} picks:")
        print(f"  min={qs[0]:.3f}  p25={qs[n//4]:.3f}  med={qs[n//2]:.3f}"
              f"  p75={qs[3*n//4]:.3f}  max={qs[-1]:.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Sort by gloss for deterministic, diff-able output.
    out = {g: best[g] for g in sorted(best.keys())}
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.out} ({len(out)} glosses)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
