"""Train the gloss tagger with CTC loss (+ optional RVQ-aux cross-entropy).

Usage:
  python -m training.train --config configs/transformer_base.yaml
  python -m training.train --config configs/transformer_base.yaml --smoke
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Subset

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from data.augment import AugConfig                       # noqa: E402
from data.dataset import GlossDataset, collate, load_vocab  # noqa: E402
from models import build_model                            # noqa: E402
from training.decode import greedy_ctc_decode, wer        # noqa: E402


def cosine_with_warmup(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def _split_logits(out):
    """Model may return logits or (logits, aux_logits). Normalize."""
    if isinstance(out, tuple):
        return out[0], out[1]
    return out, None


def evaluate(model, loader, device, aux_enabled: bool, aux_weight: float, blank=0) -> dict:
    model.eval()
    losses = []
    all_hyps: list[list[int]] = []
    all_refs: list[list[int]] = []
    ctc = nn.CTCLoss(blank=blank, zero_infinity=True)
    ce = nn.CrossEntropyLoss(ignore_index=-100)
    with torch.no_grad():
        for batch in loader:
            x = batch["features"].to(device)
            xl = batch["feat_lens"].to(device)
            y = batch["glosses"].to(device)
            yl = batch["gloss_lens"].to(device)
            out = model(x, xl)
            logits, aux_logits = _split_logits(out)
            logp = logits.log_softmax(dim=-1).transpose(0, 1)
            loss = ctc(logp, y, xl, yl)
            if aux_enabled and aux_logits is not None and "rvq_aux" in batch:
                # average-pool over time to match (B, K) aux target; here we
                # use the simpler per-frame CE against base_tokens replicated
                # to per-frame; for the smoke path the aux head is OFF.
                pass
            losses.append(float(loss.item()))
            hyps = greedy_ctc_decode(logits, xl, blank=blank)
            refs = [y[i, : yl[i]].tolist() for i in range(y.size(0))]
            all_hyps.extend(hyps)
            all_refs.extend(refs)
    model.train()
    return {"loss": sum(losses) / max(1, len(losses)), "wer": wer(all_hyps, all_refs)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true", help="quick run on a subset for sanity")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    dcfg, mcfg, tcfg = cfg["data"], cfg["model"], cfg["train"]
    acfg = cfg.get("augment", {}) or {}

    torch.manual_seed(tcfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    vocab = load_vocab(REPO / dcfg["vocab_path"])
    V = len(vocab)
    print(f"vocab size: {V}")

    aux = (mcfg.get("aux_rvq") or {}) if isinstance(mcfg.get("aux_rvq"), dict) else {}
    aux_enabled = bool(aux.get("enabled"))
    aux_weight = float(aux.get("loss_weight", 0.1))

    aug_train = AugConfig(**acfg)
    aug_eval = AugConfig(**{**acfg, "enabled": False})

    train_ds = GlossDataset(
        root=dcfg["root"],
        features_dir=dcfg["features_dir"],
        train_csv=dcfg["train_csv"],
        split_file=REPO / dcfg["splits_dir"] / "train.txt",
        vocab=vocab,
        max_frames=dcfg["max_frames"],
        augment=aug_train,
        return_rvq_aux=dcfg.get("return_rvq_aux", False) and aux_enabled,
    )
    val_ds = GlossDataset(
        root=dcfg["root"],
        features_dir=dcfg["features_dir"],
        train_csv=dcfg["train_csv"],
        split_file=REPO / dcfg["splits_dir"] / "val.txt",
        vocab=vocab,
        max_frames=dcfg["max_frames"],
        augment=aug_eval,
        return_rvq_aux=False,
    )

    if args.smoke:
        train_ds = Subset(train_ds, list(range(min(512, len(train_ds)))))
        val_ds = Subset(val_ds, list(range(min(64, len(val_ds)))))
        tcfg = {**tcfg, "total_steps": 200, "warmup_steps": 20, "eval_every": 50, "log_every": 10}

    print(f"train: {len(train_ds)}  val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=tcfg["batch_size"],
        shuffle=True,
        num_workers=tcfg["num_workers"],
        collate_fn=collate,
        drop_last=True,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=tcfg["batch_size"],
        shuffle=False,
        num_workers=tcfg["num_workers"],
        collate_fn=collate,
        pin_memory=device.type == "cuda",
    )

    model = build_model(mcfg, input_dim=dcfg["feature_dim"], vocab_size=V).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {mcfg['name']}  params: {n_params/1e6:.2f}M  aux={aux_enabled}")

    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    ce = nn.CrossEntropyLoss(ignore_index=-100)

    ckpt_dir = REPO / tcfg["ckpt_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    best_wer = float("inf")
    t0 = time.time()
    model.train()

    while step < tcfg["total_steps"]:
        for batch in train_loader:
            if step >= tcfg["total_steps"]:
                break
            lr_scale = cosine_with_warmup(step, tcfg["warmup_steps"], tcfg["total_steps"])
            for g in opt.param_groups:
                g["lr"] = tcfg["lr"] * lr_scale

            x = batch["features"].to(device)
            xl = batch["feat_lens"].to(device)
            y = batch["glosses"].to(device)
            yl = batch["gloss_lens"].to(device)

            out = model(x, xl)
            logits, aux_logits = _split_logits(out)
            logp = logits.log_softmax(dim=-1).transpose(0, 1)
            loss = ctc(logp, y, xl, yl)

            if aux_enabled and aux_logits is not None and "rvq_aux" in batch:
                # Stretch the (B, K) aux target uniformly to T per-frame
                # via nearest-neighbor index lookup. Cheap and matches the
                # observation that base_tokens stride non-uniformly across
                # source frames.
                aux_tgt = batch["rvq_aux"].to(device)
                aux_lens = batch["rvq_aux_lens"].to(device)
                B, T, _ = aux_logits.shape
                t_idx = torch.arange(T, device=device).float().unsqueeze(0)
                stretch = (t_idx * (aux_lens.float().unsqueeze(1) / xl.float().unsqueeze(1))).clamp_max(
                    aux_lens.float().unsqueeze(1) - 1
                ).long()
                tgt = aux_tgt.gather(1, stretch)
                pad_mask = (t_idx >= xl.float().unsqueeze(1))
                tgt[pad_mask] = -100
                aux_loss = ce(aux_logits.reshape(-1, aux_logits.size(-1)), tgt.reshape(-1))
                loss = loss + aux_weight * aux_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            opt.step()

            if step % tcfg["log_every"] == 0:
                elapsed = time.time() - t0
                print(f"step {step:>6}  loss {loss.item():.4f}  lr {tcfg['lr']*lr_scale:.2e}  {elapsed:.1f}s")

            if step > 0 and step % tcfg["eval_every"] == 0:
                metrics = evaluate(model, val_loader, device, aux_enabled, aux_weight)
                print(f"  eval @ {step}: loss {metrics['loss']:.4f}  wer {metrics['wer']:.3f}")
                if metrics["wer"] < best_wer:
                    best_wer = metrics["wer"]
                    torch.save(
                        {"step": step, "model": model.state_dict(), "cfg": cfg, "vocab": vocab},
                        ckpt_dir / "best.pt",
                    )
                    print(f"  saved best -> {ckpt_dir/'best.pt'} (wer {best_wer:.3f})")

            step += 1

    metrics = evaluate(model, val_loader, device, aux_enabled, aux_weight)
    print(f"final eval: loss {metrics['loss']:.4f}  wer {metrics['wer']:.3f}")
    torch.save(
        {"step": step, "model": model.state_dict(), "cfg": cfg, "vocab": vocab},
        ckpt_dir / "last.pt",
    )
    print(f"saved last -> {ckpt_dir/'last.pt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
