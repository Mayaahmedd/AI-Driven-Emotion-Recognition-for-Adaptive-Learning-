"""Tests for factorised Double DQN (Phase 8)."""

from __future__ import annotations

import random

import pytest
import torch

from adaptive_tutor.memory.providers.dataset_provider import ASSISTMENTS_ACTIONS
from adaptive_tutor.replay import UniformReplayBuffer
from adaptive_tutor.rl.dqn import (
    DoubleDQNAgent,
    FactorisedDQN,
    assistments_to_meso,
    collect_transition,
    decode_meso_to_assistments,
    q_sum_gather,
    soft_update_target,
    train_double_dqn_batch,
)
from adaptive_tutor.simulator import TutoringEnvironment
from adaptive_tutor.state.state import LearnerState, PerformanceFeatures, Transition
from adaptive_tutor.core.types import CompositeAction, EmotionVector, MesoAction


def test_decode_hits_every_assistments_action() -> None:
    seen: set[str] = set()
    for s in range(4):
        for d in range(3):
            for p in range(2):
                seen.add(decode_meso_to_assistments(s, d, p))
    assert seen == set(ASSISTMENTS_ACTIONS)


def test_assistments_round_trip_table() -> None:
    for a in ASSISTMENTS_ACTIONS:
        t = assistments_to_meso(a)
        assert decode_meso_to_assistments(*t) == a


def test_factorised_forward_shapes() -> None:
    m = FactorisedDQN(state_dim=13, hidden=(32,))
    x = torch.zeros(5, 13)
    qs, qd, qp = m(x)
    assert qs.shape == (5, 4)
    assert qd.shape == (5, 3)
    assert qp.shape == (5, 2)


def test_q_sum_gather() -> None:
    qs = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]])
    qd = torch.tensor([[0.0, 3.0, 0.0], [0.0, 0.0, 4.0]])
    qp = torch.tensor([[5.0, 0.0], [0.0, 6.0]])
    out = q_sum_gather(
        qs,
        qd,
        qp,
        torch.tensor([0, 1], dtype=torch.long),
        torch.tensor([1, 2], dtype=torch.long),
        torch.tensor([0, 1], dtype=torch.long),
    )
    assert out[0].item() == pytest.approx(9.0)
    assert out[1].item() == pytest.approx(12.0)


def test_double_dqn_target_finite() -> None:
    agent = DoubleDQNAgent(hidden=(16, 16), device=torch.device("cpu"))
    x = torch.randn(3, 13)
    v = agent.next_state_value_double(x)
    assert torch.isfinite(v).all()


def test_soft_update_moves_target() -> None:
    agent = DoubleDQNAgent(hidden=(8,), device=torch.device("cpu"))
    with torch.no_grad():
        agent.online.head_strategy.bias.fill_(1.0)
    soft_update_target(agent.online, agent.target, tau=1.0)
    assert torch.allclose(agent.target.head_strategy.bias, agent.online.head_strategy.bias)


def test_train_batch_non_negative_loss() -> None:
    torch.manual_seed(0)
    agent = DoubleDQNAgent(hidden=(32,), device=torch.device("cpu"))
    opt = torch.optim.Adam(agent.online.parameters(), lr=1e-2)

    s0 = LearnerState(
        current_concept_id="c",
        current_concept_index=0,
        mastery=0.3,
        perf=PerformanceFeatures(0.4, 0.1, 1),
        rolling_emotions=EmotionVector.neutral(),
        engagement_trend=0.0,
        confusion_trend=0.0,
        frustration_trend=0.0,
        boredom_trend=0.0,
        timestep=0,
    )
    s1 = LearnerState(
        current_concept_id="c",
        current_concept_index=0,
        mastery=0.4,
        perf=PerformanceFeatures(0.5, 0.0, 1),
        rolling_emotions=EmotionVector.neutral(),
        engagement_trend=0.0,
        confusion_trend=0.0,
        frustration_trend=0.0,
        boredom_trend=0.0,
        timestep=1,
    )
    a = CompositeAction(None, MesoAction(1, 1, 0, 0))
    tr = Transition(
        s=s0,
        a=a,
        r=0.5,
        r_components=(("c", 0.5),),
        s_next=s1,
        done=False,
        info=(),
    )
    loss = train_double_dqn_batch(
        agent, [tr] * 16, num_concepts=1, gamma=0.99, optimizer=opt
    )
    assert loss >= 0.0
    assert loss == loss


def test_collect_transition_pushes_replay() -> None:
    env = TutoringEnvironment(
        concept_slug="z",
        concept_index=0,
        num_concepts=1,
        seed=0,
        max_episode_steps=50,
        mastery_threshold=0.999,
        frustration_terminal_threshold=0.999,
        enable_action_filter=False,
    )
    agent = DoubleDQNAgent(hidden=(16,), device=torch.device("cpu"))
    buf = UniformReplayBuffer(100, seed=1)
    s, _ = env.reset()
    _, _done, _info, _r = collect_transition(
        env,
        agent,
        buf,
        s,
        num_concepts=1,
        epsilon=1.0,
        rng=random.Random(0),
    )
    assert len(buf) == 1
