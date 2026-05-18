#!/usr/bin/env bash
# Install Python deps for the gemma_ksl_lora → TFLite conversion pipeline.
#
# Why this is a shell script and not a `pip install -r` against the venv I
# created: the agent running these scripts can't reach pypi.org from inside
# its sandbox. You run this once in a normal terminal, then the rest of the
# pipeline can be driven by the agent.
#
# Usage:
#   ./install.sh
#
# Expects:
#   - The venv at repo-root `.venv_convert/` already exists (Python 3.13).
#   - ~5 GB free disk for wheels.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="$REPO_ROOT/.venv_convert"

if [[ ! -d "$VENV" ]]; then
    echo "Creating venv at $VENV (Python 3.13)..."
    /opt/homebrew/bin/python3.13 -m venv "$VENV"
fi

PIP="$VENV/bin/python -m pip"

echo "Upgrading pip / wheel / setuptools..."
$PIP install --upgrade pip wheel setuptools

# torch first — pin to CPU build for macOS (we run conversion on CPU; the
# MPS backend isn't needed for a one-shot merge + export).
echo "Installing torch (CPU)..."
$PIP install --index-url https://download.pytorch.org/whl/cpu \
    'torch>=2.5,<2.7'

echo "Installing HuggingFace stack..."
$PIP install \
    'transformers>=4.50' \
    'peft>=0.13' \
    'accelerate>=1.0' \
    'safetensors>=0.4' \
    'sentencepiece>=0.2' \
    'tokenizers>=0.20'

# ai_edge_torch pulls TensorFlow + ai_edge_litert. This is the big one
# (~2 GB of wheels). If this fails with a "no matching distribution" error
# we'll know Gemma 4 / Python 3.13 isn't yet on ai_edge_torch's wheel matrix
# and need to fall back to Python 3.12.
echo "Installing ai_edge_torch (this pulls TensorFlow, may take a while)..."
$PIP install 'ai_edge_torch>=0.4'

echo
echo "✓ Done. Activate with:"
echo "    source $VENV/bin/activate"
echo
echo "Then verify with:"
echo "    python -c 'import torch, transformers, peft, ai_edge_torch; print(\"ok\")'"
