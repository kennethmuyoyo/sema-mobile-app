"""MediaPipe-noise augmentation model applied to training landmarks.

Goal: narrow the sim2real gap between BVH-projected synthetic landmarks
(clean, no occlusion, no detector failure) and real MediaPipe iOS output
(jitter, hand-detector dropouts, framerate variance, framing variance).

Augmentations are sampled per-clip at train time. Values are tuned against
the joint-set defined in `landmarks_meta.json`.

Add to the model input an extra "dropout-mask" channel per joint so the
recognizer can condition on whether a landmark was dropped or jittered.
This doubles feature_dim from J*3 to J*3 + J*1 = J*4. The dataset/config
control whether this channel is appended.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

# Joint indices into the 45-joint layout for hand subsets — used for biased dropout.
HAND_JOINT_INDICES = list(range(15, 45))   # everything after body 15


@dataclass
class AugConfig:
    enabled: bool = True
    sigma_jitter: float = 0.005           # gaussian std in normalized units
    body_dropout_p: float = 0.05
    hand_dropout_p: float = 0.12
    temporal_mask_max: int = 8            # frames
    affine_rot_deg: float = 10.0
    affine_trans: float = 0.05
    affine_scale: float = 0.05
    frame_drop_p: float = 0.15            # probability of dropping any given frame
    append_mask_channel: bool = True


def _make_affine_2d(rot_deg: float, tx: float, ty: float, scale: float) -> np.ndarray:
    r = math.radians(rot_deg)
    s = scale
    c, sn = math.cos(r) * s, math.sin(r) * s
    return np.array([[c, -sn, tx], [sn, c, ty], [0.0, 0.0, 1.0]], dtype=np.float32)


def augment_clip(features: torch.Tensor, cfg: AugConfig, rng: np.random.Generator | None = None) -> torch.Tensor:
    """features: (T, J, 3) float32. Returns (T, J, 3) or (T, J, 4) if append_mask_channel.

    The augmentation is intentionally non-differentiable; runs on CPU
    (cheap relative to the model forward).
    """
    if not cfg.enabled:
        if cfg.append_mask_channel:
            ones = torch.ones(*features.shape[:-1], 1, dtype=features.dtype)
            return torch.cat([features, ones], dim=-1)
        return features

    rng = rng or np.random.default_rng()
    x = features.numpy().copy()
    T, J, _ = x.shape

    # --- per-joint Gaussian jitter ---
    x += rng.normal(0.0, cfg.sigma_jitter, size=x.shape).astype(np.float32)

    # --- random global 2D affine (rotation + translation + scale) ---
    A = _make_affine_2d(
        rot_deg=float(rng.uniform(-cfg.affine_rot_deg, cfg.affine_rot_deg)),
        tx=float(rng.uniform(-cfg.affine_trans, cfg.affine_trans)),
        ty=float(rng.uniform(-cfg.affine_trans, cfg.affine_trans)),
        scale=1.0 + float(rng.uniform(-cfg.affine_scale, cfg.affine_scale)),
    )
    xy = x[..., :2]                                       # (T, J, 2)
    h = np.concatenate([xy, np.ones((*xy.shape[:-1], 1), dtype=np.float32)], axis=-1)
    xy2 = np.einsum("ij,tkj->tki", A, h)[..., :2]
    x[..., :2] = xy2

    # --- per-joint dropout (zero-fill) + mask channel ---
    mask = np.ones((T, J), dtype=np.float32)
    body_drop = rng.random((T, 15)) < cfg.body_dropout_p
    hand_drop = rng.random((T, J - 15)) < cfg.hand_dropout_p
    drop = np.concatenate([body_drop, hand_drop], axis=1)
    x[drop] = 0.0
    mask[drop] = 0.0

    # --- temporal masking: zero out a contiguous window ---
    if cfg.temporal_mask_max > 0:
        w = int(rng.integers(0, cfg.temporal_mask_max + 1))
        if w > 0 and T > w:
            start = int(rng.integers(0, T - w))
            x[start : start + w] = 0.0
            mask[start : start + w] = 0.0

    # --- framerate jitter: drop every n-th frame ---
    if cfg.frame_drop_p > 0 and rng.random() < cfg.frame_drop_p:
        n = int(rng.integers(2, 4))                       # drop every 2nd or 3rd
        keep = np.arange(T) % n != 0
        if keep.sum() >= 4:
            x = x[keep]
            mask = mask[keep]

    out = torch.from_numpy(x)
    if cfg.append_mask_channel:
        out = torch.cat([out, torch.from_numpy(mask)[..., None]], dim=-1)
    return out
