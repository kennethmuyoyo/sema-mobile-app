"""Greedy CTC decode + word error rate over gloss tokens."""
from __future__ import annotations

import torch


def greedy_ctc_decode(logits: torch.Tensor, lens: torch.Tensor, blank: int = 0) -> list[list[int]]:
    pred = logits.argmax(dim=-1).cpu()
    out: list[list[int]] = []
    for row, L in zip(pred, lens.cpu().tolist()):
        prev = -1
        seq: list[int] = []
        for t in range(int(L)):
            p = int(row[t])
            if p != prev and p != blank:
                seq.append(p)
            prev = p
        out.append(seq)
    return out


def edit_distance(a: list[int], b: list[int]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ai in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, bj in enumerate(b, 1):
            cost = 0 if ai == bj else 1
            cur[j] = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]


def wer(hyps: list[list[int]], refs: list[list[int]]) -> float:
    edits = sum(edit_distance(h, r) for h, r in zip(hyps, refs))
    n = sum(len(r) for r in refs) or 1
    return edits / n
