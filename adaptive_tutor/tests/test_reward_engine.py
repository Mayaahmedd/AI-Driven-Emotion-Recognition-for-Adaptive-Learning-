"""Tests for :class:`~adaptive_tutor.rewards.RewardEngine`."""

from __future__ import annotations

from adaptive_tutor.rewards import RewardEngine, assistments_reward, default_reward_engine


def test_components_sum_to_scalar() -> None:
    eng = RewardEngine()
    emotion = {"engaged": 0.5, "frustrated": 0.2, "confused": 0.0, "bored": 0.0}
    comp = eng.components(correct=1, hint_count=2, emotion=emotion)
    s = eng.compute_scalar(correct=1, hint_count=2, emotion=emotion)
    assert s == sum(v for _, v in comp)


def test_assistments_reward_matches_default_engine() -> None:
    e = {"engaged": 0.4, "frustrated": 0.3}
    assert assistments_reward(1, 0, e) == default_reward_engine().compute_scalar(
        correct=1, hint_count=0, emotion=e
    )


def test_deterministic_default_singleton() -> None:
    assert default_reward_engine() is default_reward_engine()
