#!/usr/bin/env bash
# Sets up the Python venv and installs all dependencies for the bundle.
# Run once after unzipping. Then `python render.py` to start rendering.

set -euo pipefail
cd "$(dirname "$0")"

# Prefer Python 3.12 — it has the broadest ML wheel coverage (torch, smplx,
# pyrender, mediapipe). 3.11 and 3.13 work too but 3.14 is risky.
PY=""
for cand in python3.12 python3.11 python3.13 python3; do
    if command -v "$cand" >/dev/null 2>&1; then
        PY="$cand"
        break
    fi
done

if [[ -z "$PY" ]]; then
    echo "[setup] ERROR: Python 3.11+ not found on PATH."
    echo "  macOS:  brew install python@3.12"
    echo "  Ubuntu: sudo apt install python3.12 python3.12-venv"
    exit 1
fi

echo "[setup] using $PY ($($PY --version))"

if [[ ! -d .venv ]]; then
    echo "[setup] creating venv at .venv ..."
    "$PY" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "[setup] upgrading pip / wheel ..."
python -m pip install --upgrade pip wheel >/dev/null

echo "[setup] installing dependencies (this can take a few minutes) ..."
python -m pip install -r requirements.txt

# Sanity check — fail loudly if any heavy dep is missing.
python -c "import smplx, pyrender, cv2, mediapipe, trimesh, scipy, torch, numpy" \
    && echo "[setup] ✓ all imports loaded successfully"

if [[ ! -f models/smplx/SMPLX_NEUTRAL.npz ]]; then
    echo
    echo "[setup] ⚠️  models/smplx/SMPLX_NEUTRAL.npz is missing."
    echo "[setup]    Register at https://smpl-x.is.tue.mpg.de (free, ~5 min),"
    echo "[setup]    download SMPLX_NEUTRAL.npz, then:"
    echo "[setup]       mkdir -p models/smplx"
    echo "[setup]       mv ~/Downloads/SMPLX_NEUTRAL.npz models/smplx/"
fi

echo
echo "[setup] done. To render:"
echo "    source .venv/bin/activate"
echo "    python render.py                    # process every clip sequentially"
echo "    python render.py --shard 0/2        # multi-terminal parallelism"
