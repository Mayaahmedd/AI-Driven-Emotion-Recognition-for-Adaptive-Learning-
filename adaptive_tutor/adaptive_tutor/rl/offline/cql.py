"""Offline conservative Q-learning (CQL-style regularisation).

All optimisation steps consume **only** minibatches from a fixed offline
dataset (:class:`~adaptive_tutor.memory.providers.dataset_provider.RawTransition`
or tensorised equivalents). There is no online explore step tied to a
simulator during ``train_step``.

Reference: Kumar et al., *Conservative Q-Learning* (CQL). This module
implements a minimal discrete-action penalty suitable for bachelor-thesis
scope: discouraging over-estimation on actions that are absent from the logged
support.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def cql_discrete_penalty(q_logits: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Mean CQL penalty per batch row.

    Parameters
    ----------
    q_logits:
        ``(batch, n_actions)`` predicted Q-values (or logits treated as Q).
    actions:
        ``(batch,)`` long indices of the behaviour action in the offline data.
    """
    if q_logits.ndim != 2:
        raise ValueError("q_logits must be (batch, n_actions)")
    logsum = torch.logsumexp(q_logits, dim=1)
    on_data = q_logits.gather(1, actions.unsqueeze(1).long()).squeeze(1)
    return (logsum - on_data).mean()


class CQLDiscreteTrainer:
    """One-step TD + CQL penalty for a discrete Q-network (batch RL only)."""

    def __init__(
        self,
        q_online: nn.Module,
        q_target: nn.Module,
        *,
        optimizer: torch.optim.Optimizer,
        gamma: float = 0.99,
        alpha_cql: float = 1.0,
    ) -> None:
        self.q_online = q_online
        self.q_target = q_target
        self._optim = optimizer
        self._gamma = float(gamma)
        self._alpha_cql = float(alpha_cql)

    def train_step(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ) -> float:
        """Single gradient step from offline tensors (no env sampling)."""
        self.q_online.train()
        actions = actions.long().view(-1)
        rewards = rewards.float().view(-1)
        done = done.float().view(-1)

        q_all = self.q_online(obs)
        qa = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            q_next = self.q_target(next_obs).max(dim=1).values
            target = rewards + (1.0 - done) * self._gamma * q_next

        td = nn.functional.mse_loss(qa, target)
        cql = cql_discrete_penalty(q_all, actions) if self._alpha_cql > 0 else torch.zeros((), device=td.device)
        loss = td + self._alpha_cql * cql

        self._optim.zero_grad(set_to_none=True)
        loss.backward()
        self._optim.step()
        return float(loss.detach().item())


__all__ = ["CQLDiscreteTrainer", "cql_discrete_penalty"]
