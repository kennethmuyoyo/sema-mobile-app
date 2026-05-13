"""SLRNet-style stacked LSTM fallback.

Kept as a fallback per the architecture note in ../README.md: if the
Transformer is unstable on the available data scale, swap to this with
no other code changes (the same CTC head + training loop apply).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LSTMTagger(nn.Module):
    def __init__(
        self,
        input_dim: int,
        vocab_size: int,
        hidden: int = 256,
        n_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.1,
        input_norm: bool = True,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim) if input_norm else nn.Identity()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        out_dim = hidden * (2 if bidirectional else 1)
        self.head = nn.Linear(out_dim, vocab_size)

    def forward(self, x: torch.Tensor, lens: torch.Tensor | None = None) -> torch.Tensor:
        x = self.input_norm(x)
        if lens is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lens.cpu(), batch_first=True, enforce_sorted=False
            )
            out, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.shape[1])
        else:
            out, _ = self.lstm(x)
        return self.head(out)
