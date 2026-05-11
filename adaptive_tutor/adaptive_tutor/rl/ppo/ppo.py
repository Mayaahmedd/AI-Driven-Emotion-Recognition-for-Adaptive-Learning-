"""Clipped PPO policy loss on the discrete curriculum head (Phase 9)."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.distributions import Categorical

from adaptive_tutor.rl.ppo.buffer import PPOStepRecord
from adaptive_tutor.rl.ppo.policy import CurriculumPolicy


def ppo_policy_update(
    policy: CurriculumPolicy,
    optimizer: torch.optim.Optimizer,
    batch: Sequence[PPOStepRecord],
    *,
    clip_epsilon: float = 0.2,
    normalize_advantage: bool = True,
) -> float:
    """One clipped surrogate step on stored concept decisions.

    Advantages are normalised returns across the minibatch (simple baseline,
    no learned critic) - sufficient for low-variance curriculum credit.
    """
    if not batch:
        raise ValueError("batch must be non-empty")
    device = next(policy.parameters()).device
    obs = torch.stack([r.obs_1d for r in batch]).to(device)
    actions = torch.tensor(
        [r.concept_index for r in batch], device=device, dtype=torch.long
    )
    old_logp = torch.tensor(
        [r.log_prob for r in batch], device=device, dtype=torch.float32
    )
    returns = torch.tensor(
        [r.reward for r in batch], device=device, dtype=torch.float32
    )
    adv = returns
    if normalize_advantage:
        adv = adv - adv.mean()
        std = adv.std(unbiased=False)
        if torch.isfinite(std) and std > 1e-8:
            adv = adv / std

    logits, _, _ = policy(obs)
    dist = Categorical(logits=logits)
    new_logp = dist.log_prob(actions)
    ratio = torch.exp(new_logp - old_logp)
    unclipped = ratio * adv
    clip = float(clip_epsilon)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip, 1.0 + clip)
    clipped = clipped_ratio * adv
    loss = -torch.min(unclipped, clipped).mean()

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(loss.item())
