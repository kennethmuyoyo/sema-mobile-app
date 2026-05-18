# Sema MediaPipe Landmark Renderer — {{FOLDER_NAME}}

You've been given **{{CLIP_COUNT}} BVH motion-capture clips** to render through
SMPL-X + MediaPipe Holistic. Each clip becomes a `(T, 45, 3)` `.npy` file in
`output/`. Send that `output/` folder back when you're done — Ken stitches
all helpers' outputs together for training.

## Requirements

- **Python 3.11 or 3.12** (3.13 mostly OK, 3.14 risky). On Mac: `brew install python@3.12`. On Ubuntu: `sudo apt install python3.12 python3.12-venv`.
- **macOS or Linux.** Windows works only inside WSL (pyrender needs a GL backend).
- **~5 GB free disk** for the venv + pip wheels.
- **Apple Silicon strongly preferred** (or any decent dGPU on Linux). The pipeline renders an SMPL-X mesh and runs MediaPipe per frame, both of which lean heavily on the GPU.

## One-time setup

```bash
bash setup.sh
```

This creates `.venv/` and installs torch, pyrender, mediapipe, smplx, etc.
First run takes ~10 min.

### If the SMPL-X model is not bundled

After setup.sh, you may see:

> ⚠️ models/smplx/SMPLX_NEUTRAL.npz is missing.

Apple/SMPL-X's license requires individual registration, so the model file
may not be shipped in this bundle. Get it yourself:

1. Register at https://smpl-x.is.tue.mpg.de (free, 5 min, accept the
   research license).
2. Download `SMPLX_NEUTRAL.npz`.
3. `mkdir -p models/smplx && mv ~/Downloads/SMPLX_NEUTRAL.npz models/smplx/`

## Running

### One terminal (simplest, slowest)

```bash
source .venv/bin/activate
python render.py
```

Expected throughput on Apple M-series: ~5–15 s per clip. **For {{CLIP_COUNT}}
clips this is roughly {{HOURS_SINGLE}} hours.** Resumable — Ctrl-C anytime,
re-running picks up where it stopped.

### Multiple terminals on the same machine (faster)

Open 2 or 3 terminals. In each, activate the venv and run with a different
shard spec:

```bash
# Terminal A
source .venv/bin/activate
python render.py --shard 0/2

# Terminal B
source .venv/bin/activate
python render.py --shard 1/2
```

(Or `0/3, 1/3, 2/3` for three terminals.) Each shard processes a disjoint
subset of clips — they won't collide. **Sweet spot is 2–3 terminals on a
modern Mac.** Beyond that the GPU is the bottleneck and the laptop is
basically a heater.

Plug in the laptop. Don't set it on a soft surface. Expect fans at max.

## Quick smoke test before the big run

```bash
source .venv/bin/activate
python render.py --limit 5
```

Check that 5 `.npy` files appeared under `output/` and each is non-empty.
If it works, kill it and start the real run.

## Sending output back

When `render.py` reports `done` with `errors=0`, just zip the `output/`
folder and send it back:

```bash
zip -r {{FOLDER_NAME}}_output.zip output/
```

Each `.npy` is ~135 KB; the full zip is ~500 MB.

## Troubleshooting

- **`Unable to create OpenGL context`** — pyrender's GL backend can't init on macOS sometimes. Try: `PYOPENGL_PLATFORM=osmesa python render.py` (slower software backend).
- **`No module named cv2`** — `pip install opencv-python` (not `cv2`).
- **`module 'mediapipe' has no attribute 'solutions'`** — your mediapipe is too new; we already use the new Tasks API, so this means a *different* `mediapipe` is installed. Re-run `bash setup.sh`.
- **All outputs are zero-filled** — the SMPL-X model file might be missing or the bundle's `models/` paths are wrong. Check the smoke-test output first.

Any other weird error: send Ken the last 20 lines of stderr.
