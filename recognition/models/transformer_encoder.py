"""Transformer encoder over per-frame landmark features → CTC logits.

Shape conventions:
  features : (B, T, D_in)
  mask     : (B, T) bool, True = padding
  logits   : (B, T, V) before log_softmax

Optionally produces a per-frame auxiliary head over the RVQ codebook
(`base_tokens` from Motion-S). The aux head is a free regularization
signal — Plain CTC stays the primary objective.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerTagger(nn.Module):
    def __init__(
        self,
        input_dim: int,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        input_norm: bool = True,
        max_len: int = 1024,
        aux_codebook_size: int = 0,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim) if input_norm else nn.Identity()
        self.proj = nn.Linear(input_dim, d_model)
        self.pe = SinusoidalPE(d_model, max_len=max_len)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, vocab_size)
        self.aux_head = nn.Linear(d_model, aux_codebook_size) if aux_codebook_size > 0 else None

    def forward(self, x: torch.Tensor, lens: torch.Tensor | None = None):
        x = self.input_norm(x)
        x = self.proj(x)
        x = self.pe(x)
        x = self.drop(x)
        key_padding_mask = None
        if lens is not None:
            B, T = x.shape[:2]
            idx = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
            key_padding_mask = idx >= lens.to(x.device).unsqueeze(1)
        h = self.encoder(x, src_key_padding_mask=key_padding_mask)
        logits = self.head(h)
        if self.aux_head is None:
            return logits
        aux_logits = self.aux_head(h)
        return logits, aux_logits
