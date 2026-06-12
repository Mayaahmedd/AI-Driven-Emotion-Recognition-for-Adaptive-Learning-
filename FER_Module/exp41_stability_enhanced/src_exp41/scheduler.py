"""
Cosine annealing with linear warm-up, stepped per iteration.

The schedule is composed of two phases:

    Phase 1 (linear warm-up, first ``warmup_iters`` iterations):
        lr(it) = lr_max * (it + 1) / warmup_iters

    Phase 2 (cosine annealing, until ``total_iters``):
        progress = (it - warmup_iters) / (total_iters - warmup_iters)
        lr(it) = eta_min + 0.5 * (lr_max - eta_min) * (1 + cos(pi * progress))

``lr_max`` is taken from each parameter group's existing ``lr`` field at
scheduler construction time, so any per-group LR multipliers in the
optimiser are preserved.

This wrapper is implemented with the standard ``LambdaLR`` so it composes
cleanly with ``torch.cuda.amp`` and with ``optimizer.state_dict`` /
checkpoint resume logic.
"""

from __future__ import annotations

import math
from typing import List

import torch


def build_cosine_with_warmup(
    optimizer: torch.optim.Optimizer,
    total_iters: int,
    warmup_iters: int,
    eta_min_ratio: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Return a per-iteration LambdaLR scheduler.

    Parameters
    ----------
    optimizer
        The optimiser whose learning rate(s) will be scheduled.
    total_iters
        Total number of iterations across the entire training run
        (i.e. ``epochs * len(train_loader)``).
    warmup_iters
        Number of iterations in the linear warm-up phase. With a
        one-epoch warm-up this equals ``len(train_loader)``.
    eta_min_ratio
        Minimum LR expressed as a fraction of the initial LR. With
        ``eta_min_ratio = 0.0`` the LR cosines down to zero; common
        choices are ``0.0`` to ``0.05``.
    """
    if total_iters <= 0:
        raise ValueError("total_iters must be positive.")
    if warmup_iters < 0 or warmup_iters >= total_iters:
        raise ValueError("warmup_iters must satisfy 0 <= warmup_iters < total_iters.")

    def lr_factor(it: int) -> float:
        if warmup_iters > 0 and it < warmup_iters:
            return float(it + 1) / float(warmup_iters)
        progress = (it - warmup_iters) / max(1, total_iters - warmup_iters)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(eta_min_ratio + (1.0 - eta_min_ratio) * cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)
