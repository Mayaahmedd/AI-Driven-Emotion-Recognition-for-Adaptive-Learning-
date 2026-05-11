"""Phase 9: PPO curriculum manager over concept segments (above DQN)."""

from __future__ import annotations

import random
from pathlib import Path

import torch

from adaptive_tutor.memory.providers.dataset_provider import ASSISTMENTS_ACTIONS
from adaptive_tutor.memory.providers.teacher_provider import TeacherCurriculumProvider
from adaptive_tutor.replay import UniformReplayBuffer
from adaptive_tutor.rl.dqn import DoubleDQNAgent
from adaptive_tutor.rl.ppo import (
    CurriculumPolicy,
    PPOStepRecord,
    collect_concept_rollout,
    concept_rollout_reward,
    dataset_skill_slug_for_concept,
    ppo_policy_update,
)
from adaptive_tutor.simulator import TutoringEnvironment

_REPO = Path(__file__).resolve().parent.parent
_MATH_YAML = _REPO / "configs" / "curriculum" / "examples" / "math_basic.yaml"


def test_concept_rollout_reward_increases_with_mastery() -> None:
    base = concept_rollout_reward(
        mastery_start=0.2,
        mastery_end=0.2,
        engaged_start=0.5,
        engaged_end=0.5,
        frustration_end=0.1,
        n_steps=5,
    )
    better = concept_rollout_reward(
        mastery_start=0.2,
        mastery_end=0.8,
        engaged_start=0.5,
        engaged_end=0.5,
        frustration_end=0.1,
        n_steps=5,
    )
    assert better > base


def test_dataset_skill_slug_for_concept() -> None:
    teacher = TeacherCurriculumProvider(_MATH_YAML)
    assert (
        dataset_skill_slug_for_concept(teacher, "addition_whole_numbers")
        == "addition_whole_numbers"
    )


def test_collect_concept_rollout_runs_with_dqn_under_phase7() -> None:
    teacher = TeacherCurriculumProvider(_MATH_YAML)
    n = teacher.num_concepts()
    idx0 = teacher.concept_index("addition_whole_numbers")
    env = TutoringEnvironment(
        concept_slug="addition_whole_numbers",
        concept_index=idx0,
        num_concepts=n,
        teacher=teacher,
        seed=2,
        max_episode_steps=8,
        mastery_threshold=0.995,
        frustration_terminal_threshold=0.999,
        enable_action_filter=True,
    )
    curriculum_state, _ = env.reset()
    torch.manual_seed(0)
    policy = CurriculumPolicy(num_concepts=n, hidden=(32,))
    agent = DoubleDQNAgent(hidden=(32, 32), device=torch.device("cpu"))
    buf = UniformReplayBuffer(500, seed=3)

    out = collect_concept_rollout(
        env,
        teacher,
        agent,
        buf,
        policy,
        curriculum_state=curriculum_state,
        num_concepts=n,
        epsilon=1.0,
        rng=random.Random(4),
    )

    assert 0 <= out.record.concept_index < n
    assert len(out.micro_trace) >= 1
    m0, action0, _r0, _c0, _h0, _f0 = out.micro_trace[0]
    assert 0.0 <= m0 <= 1.0
    assert action0 in ASSISTMENTS_ACTIONS
    assert len(buf) >= 1


def test_prereq_masking_still_applies_under_ppo_segment() -> None:
    """Phase 7 must still rewrite illegal ops after PPO switches concept."""
    teacher = TeacherCurriculumProvider(_MATH_YAML)
    n = teacher.num_concepts()
    idx = teacher.concept_index("subtraction_whole_numbers")
    env = TutoringEnvironment(
        concept_slug="subtraction_whole_numbers",
        concept_index=idx,
        num_concepts=n,
        teacher=teacher,
        initial_mastery_by_slug={"addition_whole_numbers": 0.1},
        prereq_mastery_threshold=0.6,
        seed=0,
        max_episode_steps=6,
        mastery_threshold=0.99,
        frustration_terminal_threshold=0.99,
        enable_action_filter=True,
    )
    env.reset()
    env.start_concept_segment(
        concept_slug="subtraction_whole_numbers",
        concept_index=idx,
        dataset_skill_slug=dataset_skill_slug_for_concept(
            teacher, "subtraction_whole_numbers"
        ),
        pacing_factor=1.0,
        difficulty_bias=0.0,
    )
    _s, _r, _d, info = env.step("harder_problem")
    assert info["action_requested"] == "harder_problem"
    assert info["action"] != "harder_problem"


def test_ppo_policy_update_finite() -> None:
    torch.manual_seed(1)
    n_concepts = 4
    policy = CurriculumPolicy(num_concepts=n_concepts, hidden=(16,))
    opt = torch.optim.Adam(policy.parameters(), lr=1e-2)
    obs = torch.zeros(13, dtype=torch.float32)
    batch = [
        PPOStepRecord(
            obs.clone(),
            concept_index=0,
            log_prob=-1.2,
            reward=0.5,
        ),
        PPOStepRecord(
            obs.clone(),
            concept_index=1,
            log_prob=-1.1,
            reward=-0.1,
        ),
    ]
    loss = ppo_policy_update(policy, opt, batch)
    assert loss == loss
    assert abs(loss) < 1e6
