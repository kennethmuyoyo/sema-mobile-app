# Team data + model storage (DVC + Cloudflare R2)

Sema's large files — raw BVH motion capture, derived landmarks, model checkpoints, exported CoreML/LiteRT bundles, and the iOS pose library — are versioned with **DVC** and stored in a private **Cloudflare R2** bucket. Git only carries the small `.dvc` pointer files and `dvc.lock`.

R2 has **zero egress fees**, which matters because every teammate pulls data on first clone and after every retraining round.

## TL;DR — fresh clone

```bash
git clone <repo>
cd sema-mobile-app

# (1) Python deps incl. DVC
python3 -m venv recognition/.venv
recognition/.venv/bin/pip install -r recognition/requirements.txt 'dvc[s3]'

# (2) R2 credentials (one time per machine) — see below
recognition/.venv/bin/dvc remote modify --local r2 access_key_id     "$R2_ACCESS_KEY_ID"
recognition/.venv/bin/dvc remote modify --local r2 secret_access_key "$R2_SECRET_ACCESS_KEY"

# (3) Pull all tracked data and model artefacts
recognition/.venv/bin/dvc pull
```

After `dvc pull`, the directories below are populated:

| Path | What lives there |
|---|---|
| `data/Train/` | Motion-S BVH source files (one folder per clip) |
| `data/train.csv`, `test.csv`, `sample_submission.csv` | Motion-S labels and tokens |
| `data/landmarks/` | Derived 45-joint landmark `.npy` files (output of the `landmarks` DVC stage) |
| `recognition/data/vocab/gloss_vocab.json`, `recognition/data/splits/` | `vocab` stage output |
| `recognition/checkpoints/transformer_base/` | Trained checkpoint(s) |
| `mobile-app/Sema/Models/gloss_tagger.mlpackage` etc. | Exported CoreML bundle |

## What is and isn't in DVC

| Tracked by DVC | Tracked by git |
|---|---|
| Raw Motion-S BVH (`data/Train/`) and CSVs (`data/*.csv`) | All code (.py, .swift), configs (.yaml), READMEs, contracts |
| Derived landmark `.npy` files (`data/landmarks/`) | The gloss vocab JSON and splits txt are small and **committed to git** for now — DVC treats them as pipeline outputs and will cache them, but git is the source of truth |
| Trained checkpoints (`recognition/checkpoints/`) | `dvc.yaml`, `dvc.lock`, `.dvc` pointer files |
| Exported `.mlpackage` and pose-library bundle | |

`data/Motion-Features/` (the legacy 668-dim BVH-derived features from before the iOS pivot) is **not** tracked. The current pipeline goes BVH → landmarks directly. Add it back if a teammate needs it: `dvc add data/Motion-Features && dvc push`.

## Cloudflare R2 setup (one-time, done by the data owner)

1. Sign in to Cloudflare → R2.
2. Create a bucket, e.g. `sema-data`.
3. Create an **R2 API token** with **Object Read & Write** scoped to that bucket.
4. Note: account ID, bucket name, access key id, secret access key.
5. Fill in `.dvc/config`:
   ```ini
   [core]
       remote = r2
   ['remote "r2"']
       url = s3://sema-data/sema
       endpointurl = https://<ACCOUNT_ID>.r2.cloudflarestorage.com
   ```
6. Commit `.dvc/config` (no secrets in it).

Each teammate then runs the two `dvc remote modify --local …` lines above to plant their personal credentials into `.dvc/config.local` (which is gitignored).

### Alternative: AWS named profile

If your team already manages cloud credentials via `~/.aws/credentials`, add:

```ini
[cloudflare-r2]
aws_access_key_id = ...
aws_secret_access_key = ...
```

and configure DVC once to use it:

```bash
recognition/.venv/bin/dvc remote modify --local r2 profile cloudflare-r2
```

## Day-to-day workflow

### Producing new artefacts

```bash
# Edit code that affects a DVC stage's deps (e.g. recognition/data/bvh_to_landmarks.py)
recognition/.venv/bin/dvc repro           # reruns affected stages, in dependency order
git add dvc.lock recognition/data/landmarks_meta.json   # commit the new pipeline state
git commit -m "landmarks: …"
recognition/.venv/bin/dvc push            # uploads new outputs to R2
git push
```

### Consuming someone else's artefacts

```bash
git pull
recognition/.venv/bin/dvc pull            # downloads anything new from R2
```

DVC compares hashes; if the cached file already matches what's referenced in `dvc.lock`, nothing is transferred.

### Adding a new tracked-but-not-pipeline file

For external inputs that are not produced by a pipeline stage (e.g. the raw BVH source):

```bash
recognition/.venv/bin/dvc add data/Train
git add data/Train.dvc data/.gitignore
git commit -m "track data/Train via DVC"
recognition/.venv/bin/dvc push
```

The `.dvc` pointer file is small and goes into git; the actual data goes to R2.

## DVC pipeline

The full DAG is in [`../dvc.yaml`](../dvc.yaml). Summary:

```
data/Train  ──►  landmarks  ──►  data/landmarks
                                       │
data/train.csv  ──►  vocab  ──►  recognition/data/{vocab,splits}
                                       │
                                       ▼
                            train_transformer  ──►  recognition/checkpoints/transformer_base
                                       │
                                       ▼
                              export_coreml  ──►  mobile-app/Sema/Models/gloss_tagger.mlpackage
```

`dvc repro <stage>` runs the named stage plus any stale upstream deps.

## Cost expectations

R2 pricing (Mar 2026): $0.015/GB/month storage, **zero egress**, $4.50/million Class A ops (writes), $0.36/million Class B ops (reads). For Sema's scale (~3 GB raw + ~400 MB derived + small checkpoints) this is on the order of **$0.05/month** in storage with hundreds of free pull operations.

## Troubleshooting

- **`dvc pull` says "no remote storage"** — fill in `.dvc/config` per the R2 setup above and re-run.
- **`dvc pull` says "Unable to find credentials"** — run the `dvc remote modify --local` lines or set up the AWS profile.
- **`dvc repro` reruns a stage you didn't change** — a dep file's hash changed. `dvc status` shows what's stale.
- **Want to skip pulling huge files** — `dvc pull --targets data/Train` pulls only the named path.
