"""Tests for :class:`TutoringEnvironment` and calibration helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from adaptive_tutor.memory.providers.dataset_provider import (
    ASSISTMENTS_ACTIONS,
    DatasetCurriculumProvider,
    SkillStats,
    skill_stats_as_tuple,
)
from adaptive_tutor.rewards import assistments_reward
from adaptive_tutor.simulator import (
    TutoringEnvironment,
    calibrate_from_stats,
    default_learner_params,
    random_tutor_policy,
)
from adaptive_tutor.simulator.learner import SyntheticLearner

_REPO = Path(__file__).resolve().parent.parent
_CSV = _REPO / "configs" / "curriculum" / "examples" / "assistments_synthetic.csv"


def test_reset_zeros_timestep_and_clears_windows() -> None:
    env = TutoringEnvironment(
        concept_slug="addition_whole_numbers",
        concept_index=0,
        num_concepts=6,
        seed=42,
        max_episode_steps=1000,
        mastery_threshold=0.99,
        enable_action_filter=False,
    )
    s0, _ = env.reset()
    assert s0.timestep == 0
    s1, _, _, _ = env.step("give_hint")
    assert s1.timestep >= 1
    s2, _ = env.reset()
    assert s2.timestep == 0


def test_deterministic_rollout_under_fixed_seed() -> None:
    def rollout(seed: int) -> list[tuple[float, float]]:
        env = TutoringEnvironment(
            concept_slug="x",
            concept_index=0,
            num_concepts=1,
            seed=seed,
            max_episode_steps=5,
            mastery_threshold=0.99,
            frustration_terminal_threshold=0.99,
            enable_action_filter=False,
        )
        env.reset()
        out: list[tuple[float, float]] = []
        for a in (
            "give_hint",
            "easier_problem",
            "encouragement",
            "retry_current_skill",
            "give_hint",
        ):
            _s, r, _d, _i = env.step(a)
            out.append((round(r, 5), env._snapshot().mastery))  # noqa: SLF001
        return out

    assert rollout(7) == rollout(7)
    assert rollout(7) != rollout(8)


def test_step_reward_consistent_with_assistments_reward() -> None:
    env = TutoringEnvironment(
        concept_slug="z",
        concept_index=0,
        num_concepts=1,
        seed=0,
        max_episode_steps=3,
        mastery_threshold=0.99,
        frustration_terminal_threshold=0.99,
        enable_action_filter=False,
    )
    env.reset()
    _s, r, _, info = env.step("advance_to_next_skill")
    correct = int(info["correct"])
    hints = 1 if info["action"] == "give_hint" else 0
    em = env._learner.to_emotion_vector()  # noqa: SLF001
    expected = assistments_reward(
        correct,
        hints,
        {
            "engaged": em.engaged,
            "confused": em.confused,
            "bored": em.bored,
            "frustrated": em.frustrated,
        },
    )
    assert r == pytest.approx(expected)


def test_reward_stays_in_reasonable_band() -> None:
    env = TutoringEnvironment(
        concept_slug="z",
        concept_index=0,
        num_concepts=1,
        seed=123,
        max_episode_steps=50,
        mastery_threshold=0.99,
        frustration_terminal_threshold=0.99,
        enable_action_filter=False,
    )
    env.reset()
    policy = random_tutor_policy(seed=5)
    for _ in range(50):
        s = env._snapshot()  # noqa: SLF001
        _, r, done, _ = env.step(policy(s))
        assert -2.0 < r < 2.0
        if done:
            break


def test_episode_terminates_at_max_steps() -> None:
    env = TutoringEnvironment(
        concept_slug="z",
        concept_index=0,
        num_concepts=1,
        seed=1,
        max_episode_steps=3,
        mastery_threshold=0.999,
        frustration_terminal_threshold=0.999,
        enable_action_filter=False,
    )
    env.reset()
    done = False
    steps = 0
    while not done and steps < 10:
        _, _, done, _ = env.step("retry_current_skill")
        steps += 1
    assert done and steps == 3


def test_invalid_action_raises() -> None:
    env = TutoringEnvironment(
        concept_slug="z",
        concept_index=0,
        num_concepts=1,
        seed=0,
        max_episode_steps=5,
        enable_action_filter=False,
    )
    env.reset()
    with pytest.raises(ValueError, match="unknown action"):
        env.step("not_an_action")


def test_emotions_stay_clamped_inside_learner() -> None:
    learner = SyntheticLearner.from_params(default_learner_params())
    for _ in range(30):
        learner.apply_action_effect("encouragement")
        learner.feedback_after_outcome(True)
    assert 0.0 <= learner.engaged <= 1.0
    learner2 = SyntheticLearner.from_params(default_learner_params())
    for _ in range(30):
        learner2.feedback_after_outcome(False)
    assert 0.0 <= learner2.frustrated <= 1.0


def test_calibration_from_real_stats_shape() -> None:
    ds = DatasetCurriculumProvider(_CSV)
    st = ds.stats("addition_whole_numbers")
    p = calibrate_from_stats(st)
    assert 0.0 < p.learning_rate < 1.0
    assert 0.0 <= p.engaged_bias <= 1.0


def test_skill_stats_round_trip_tuple() -> None:
    st = SkillStats(
        skill="S",
        n_records=10,
        mean_correctness=0.7,
        mean_hint_count=0.5,
        mean_attempts=1.2,
        mean_response_time_ms=5000.0,
        mean_frustrated=0.1,
        mean_engaged=0.8,
    )
    t = skill_stats_as_tuple(st)
    d = dict(t)
    assert d["historical_success_rate"] == pytest.approx(0.7)
    assert d["n_records"] == pytest.approx(10.0)


def test_scripted_policies_emit_valid_actions_only() -> None:
    from adaptive_tutor.simulator import (
        heuristic_tutor_policy,
        scripted_advanced_learner_policy,
        scripted_struggling_learner_policy,
    )

    env = TutoringEnvironment(
        concept_slug="addition_whole_numbers",
        concept_index=0,
        num_concepts=6,
        seed=0,
        max_episode_steps=200,
        mastery_threshold=0.999,
        frustration_terminal_threshold=0.999,
        enable_action_filter=False,
    )
    s, _ = env.reset()
    for factory in (
        heuristic_tutor_policy,
        scripted_struggling_learner_policy,
        scripted_advanced_learner_policy,
    ):
        pol = factory()
        for _ in range(20):
            a = pol(s)
            assert a in ASSISTMENTS_ACTIONS
            s, _, done, _ = env.step(a)
            if done:
                s, _ = env.reset()
