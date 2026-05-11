"""Training utilities for factorised Double DQN (Phase 8).

The canonical interaction loop (**DQN** -> decode string -> **Phase 7 env**
``step`` -> **reward** -> **replay**) lives here so
:class:`~adaptive_tutor.simulator.environment.TutoringEnvironment` stays
policy-agnostic.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F

from adaptive_tutor.core.types import CompositeAction, MesoAction
from adaptive_tutor.rl.dqn.dqn import DoubleDQNAgent, q_sum_gather
from adaptive_tutor.rl.dqn.model import FactorisedDQN
from adaptive_tutor.replay import UniformReplayBuffer
from adaptive_tutor.simulator.environment import TutoringEnvironment
from adaptive_tutor.state.state import LearnerState, Transition


def soft_update_target(online: FactorisedDQN, target: FactorisedDQN, tau: float) -> None:
    """Polyak averaging: ``target = (1-tau)*target + tau*online``."""
    tau = float(tau)
    with torch.no_grad():
        for tp, op in zip(target.parameters(), online.parameters(), strict=True):
            tp.data.mul_(1.0 - tau).add_(op.data, alpha=tau)


def train_double_dqn_batch(
    agent: DoubleDQNAgent,
    batch: list[Transition],
    *,
    num_concepts: int,
    gamma: float,
    optimizer: torch.optim.Optimizer,
) -> float:
    """One gradient step; returns scalar loss."""
    if not batch:
        raise ValueError("batch must be non-empty")
    dev = agent.device
    s = torch.stack([tr.s.to_tensor(num_concepts=num_concepts) for tr in batch]).to(dev)
    s_n = torch.stack([tr.s_next.to_tensor(num_concepts=num_concepts) for tr in batch]).to(dev)
    r = torch.tensor([tr.r for tr in batch], device=dev, dtype=torch.float32)
    done = torch.tensor([1.0 if tr.done else 0.0 for tr in batch], device=dev, dtype=torch.float32)

    i_s = torch.tensor(
        [tr.a.meso.content_variant for tr in batch], device=dev, dtype=torch.long
    )
    i_d = torch.tensor([tr.a.meso.pace for tr in batch], device=dev, dtype=torch.long)
    i_p = torch.tensor([tr.a.meso.ui_variant for tr in batch], device=dev, dtype=torch.long)

    qs, qd, qp = agent.online(s)
    q_sa = q_sum_gather(qs, qd, qp, i_s, i_d, i_p)

    with torch.no_grad():
        v_next = agent.next_state_value_double(s_n)
        y = r + float(gamma) * (1.0 - done) * v_next

    loss = F.mse_loss(q_sa, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(loss.item())


def _info_to_transition_tuple(info: Mapping[str, Any]) -> tuple[tuple[str, float], ...]:
    pairs: list[tuple[str, float]] = []
    for key in ("correct", "p_correct"):
        if key in info and isinstance(info[key], int | float):
            pairs.append((key, float(info[key])))
    return tuple(pairs)


def collect_transition(
    env: TutoringEnvironment,
    agent: DoubleDQNAgent,
    buffer: UniformReplayBuffer,
    state: LearnerState,
    *,
    num_concepts: int,
    epsilon: float,
    rng: random.Random,
) -> tuple[LearnerState, bool, dict[str, Any], float]:
    """Select meso heads -> decode -> ``env.step`` (Phase 7 inside) -> store transition.

    Returns
    -------
    next_state, done, info, reward
    """
    s_idx, d_idx, p_idx, _decoded_preview = agent.act(
        state, num_concepts=num_concepts, epsilon=epsilon, rng=rng
    )
    action_str = _decoded_preview
    s_next, reward, done, info = env.step(action_str)
    action = CompositeAction(
        macro=None, meso=MesoAction(s_idx, d_idx, p_idx, 0)
    )
    tr = Transition(
        s=state,
        a=action,
        r=reward,
        r_components=tuple(info["r_components"]),
        s_next=s_next,
        done=done,
        info=_info_to_transition_tuple(info),
    )
    buffer.add(tr)
    return s_next, done, info, reward
