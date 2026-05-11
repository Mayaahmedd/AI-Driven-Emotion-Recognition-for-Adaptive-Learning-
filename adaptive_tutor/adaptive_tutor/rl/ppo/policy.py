"""Categorical curriculum policy + pacing / difficulty scalars (Phase 9).

Shared MLP trunk; one discrete head (concept index) and two deterministic scalars
used only inside ``TutoringEnvironment.start_concept_segment`` (no extra PPO ratio).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from adaptive_tutor.rl.ppo.types import CurriculumAction


class CurriculumPolicy(nn.Module):
    """Maps normalised state vector -> concept logits + segment knobs."""

    def __init__(
        self,
        *,
        num_concepts: int,
        state_dim: int = 13,
        hidden: tuple[int, ...] = (64, 64),
    ) -> None:
        super().__init__()
        if num_concepts < 1:
            raise ValueError("num_concepts must be >= 1")
        layers: list[nn.Module] = []
        d = state_dim
        for h in hidden:
            layers.extend([nn.Linear(d, h), nn.ReLU(inplace=True)])
            d = h
        self.trunk = nn.Sequential(*layers) if layers else nn.Identity()
        self.head_concept = nn.Linear(d, num_concepts)
        self.head_pacing = nn.Linear(d, 1)
        self.head_difficulty = nn.Linear(d, 1)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``logits [B,C]``, ``pacing [B]`` in ~[0.5, 2], ``difficulty [B]`` in [-1,1]."""
        h = self.trunk(x)
        logits = self.head_concept(h)
        pace = 0.5 + F.softplus(self.head_pacing(h)).squeeze(-1).clamp(max=1.5)
        diff = torch.tanh(self.head_difficulty(h)).squeeze(-1)
        return logits, pace, diff

    def sample_action(self, obs_1d: torch.Tensor) -> tuple[CurriculumAction, torch.Tensor]:
        """Sample concept from Categorical; return action + log_prob (scalartensor)."""
        dev = next(self.parameters()).device
        logits, pace, diff = self.forward(obs_1d.to(dev).unsqueeze(0))
        dist = Categorical(logits=logits)
        idx = dist.sample()
        lp = dist.log_prob(idx)
        return (
            CurriculumAction(
                concept_index=int(idx.item()),
                pacing_factor=float(pace[0].item()),
                difficulty_bias=float(diff[0].item()),
            ),
            lp,
        )

    def log_prob_of(
        self, obs_1d: torch.Tensor, concept_index: int
    ) -> torch.Tensor:
        """Log-prob of a stored discrete decision (for PPO ratio)."""
        dev = next(self.parameters()).device
        logits, _, _ = self.forward(obs_1d.to(dev).unsqueeze(0))
        dist = Categorical(logits=logits)
        return dist.log_prob(
            torch.tensor([concept_index], device=logits.device, dtype=torch.long)
        )
