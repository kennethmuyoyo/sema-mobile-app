#!/usr/bin/env bash
# Fetch the MediaPipe iOS landmarker .task assets pinned to known-good versions.
# Driven by the `mediapipe_assets` DVC stage; teammates `dvc pull` to skip.
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p Resources

POSE_URL="https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
HAND_URL="https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

echo "Fetching pose_landmarker_full.task…"
curl -fsSL --retry 3 --retry-delay 2 --max-time 120 "$POSE_URL" -o Resources/pose_landmarker_full.task

echo "Fetching hand_landmarker.task…"
curl -fsSL --retry 3 --retry-delay 2 --max-time 120 "$HAND_URL" -o Resources/hand_landmarker.task

echo "Done. Sizes:"
ls -lh Resources/*.task
