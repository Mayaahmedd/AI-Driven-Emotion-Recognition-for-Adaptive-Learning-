"""Evaluation and reporting (Phase 11)."""

from adaptive_tutor.evaluation.dataset_metrics import compute_dataset_metrics
from adaptive_tutor.evaluation.policy_eval import (
    evaluate_dqn_policy,
    evaluate_heuristic_policy,
    evaluate_ppo_policy,
    evaluate_random_policy,
    learning_curve_slope_from_rewards,
    run_callable_policy_episode,
    run_dqn_episode,
    run_ppo_curriculum_session,
)
from adaptive_tutor.evaluation.run_evaluation import run_full_evaluation
from adaptive_tutor.evaluation.simulator_metrics import (
    compute_episode_metrics,
    compute_frustration_rate,
    compute_hint_efficiency,
    compute_learning_rate,
    compute_mastery_gain,
)

__all__ = [
    "compute_dataset_metrics",
    "compute_episode_metrics",
    "compute_frustration_rate",
    "compute_hint_efficiency",
    "compute_learning_rate",
    "compute_mastery_gain",
    "evaluate_dqn_policy",
    "evaluate_heuristic_policy",
    "evaluate_ppo_policy",
    "evaluate_random_policy",
    "learning_curve_slope_from_rewards",
    "run_callable_policy_episode",
    "run_dqn_episode",
    "run_full_evaluation",
    "run_ppo_curriculum_session",
]
