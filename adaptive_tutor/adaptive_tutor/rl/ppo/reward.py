"""Scalar reward for one concept-level rollout (Phase 9).

Aggregates learner state movement over **DQN micro-steps** while PPO trains on
a low-frequency curriculum signal (reduced variance vs per-step reward).
"""

from __future__ import annotations


def concept_rollout_reward(
    *,
    mastery_start: float,
    mastery_end: float,
    engaged_start: float,
    engaged_end: float,
    frustration_end: float,
    n_steps: int,
    step_cost: float = 0.02,
) -> float:
    """Hand-written thesis reward (interpretable, no learning inside).

    ``R = mastery_gain + engagement_gain - frustration_end - step_cost * n_steps``
    """
    mg = float(mastery_end) - float(mastery_start)
    eg = float(engaged_end) - float(engaged_start)
    fr = float(frustration_end)
    return mg + eg - fr - step_cost * max(0, int(n_steps))
