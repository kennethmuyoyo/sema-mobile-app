#!/usr/bin/env bash
# End-to-end sanity check:
#   BVH → landmarks (50 clips) → vocab/splits → dataset shape → 200-step CTC smoke.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=".venv/bin/python"
"$PY" -m data.bvh_to_landmarks --limit 50
"$PY" -m data.build_vocab
"$PY" -m tests.test_dataset
"$PY" -m training.train --config configs/transformer_base.yaml --smoke
