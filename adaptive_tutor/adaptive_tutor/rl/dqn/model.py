"""Factorised Q-network: shared encoder + three discrete heads (Phase 8).

Each head outputs logits (Q-values) for one axis of :class:`MesoAction`:
``content_variant`` (strategy), ``pace`` (difficulty), ``ui_variant``
(progression).  Total Q for a joint choice is the **sum** of the three
head values - a standard factorisation that keeps capacity modest for a
bachelor thesis.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from adaptive_tutor.rl.dqn.decode import DIFFICULTY_DIM, PROGRESSION_DIM, STRATEGY_DIM


class FactorisedDQN(nn.Module):
    """MLP encoder with three linear heads (no transformers)."""

    def __init__(
        self,
        state_dim: int = 13,
        hidden: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__()
        if not hidden:
            raise ValueError("hidden must be non-empty")
        layers: list[nn.Module] = []
        d = state_dim
        for h in hidden:
            layers.append(nn.Linear(d, h))
            layers.append(nn.ReLU(inplace=True))
            d = h
        self.encoder = nn.Sequential(*layers)
        self.head_strategy = nn.Linear(d, STRATEGY_DIM)
        self.head_difficulty = nn.Linear(d, DIFFICULTY_DIM)
        self.head_progression = nn.Linear(d, PROGRESSION_DIM)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(Q_strategy, Q_difficulty, Q_progression)`` each shape ``[..., n_i]``."""
        h = self.encoder(x)
        return self.head_strategy(h), self.head_difficulty(h), self.head_progression(h)
