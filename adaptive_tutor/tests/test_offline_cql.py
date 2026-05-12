"""Offline CQL penalty smoke test."""

from __future__ import annotations

import torch
import torch.nn as nn

from adaptive_tutor.rl.offline.cql import CQLDiscreteTrainer, cql_discrete_penalty


def test_cql_penalty_positive_when_on_policy_high() -> None:
    q = torch.tensor([[0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
    a = torch.tensor([1, 2])
    pen = cql_discrete_penalty(q, a)
    assert pen.ndim == 0
    assert float(pen.detach()) > 0


def test_cql_trainer_step_runs() -> None:
    net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
    target = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
    target.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    trainer = CQLDiscreteTrainer(net, target, optimizer=opt, alpha_cql=0.5)
    obs = torch.randn(16, 4)
    next_obs = torch.randn(16, 4)
    actions = torch.randint(0, 3, (16,))
    rews = torch.randn(16)
    done = torch.zeros(16)
    loss = trainer.train_step(obs, actions, rews, next_obs, done)
    assert loss == loss and loss >= 0.0
