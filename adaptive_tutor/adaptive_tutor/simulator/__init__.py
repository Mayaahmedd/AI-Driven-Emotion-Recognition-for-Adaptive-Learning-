"""Synthetic student environment for RL development without live learners.

This package sits **between** curriculum memory (teacher YAML) and future
DQN/PPO trainers. It consumes ASSISTments-shaped statistics for
calibration but never mutates the curriculum graph.

Components
----------
* :mod:`adaptive_tutor.simulator.calibration` - ``LearnerParams`` from ``SkillStats``
* :mod:`adaptive_tutor.simulator.learner` - latent emotion dynamics
* :mod:`adaptive_tutor.simulator.environment` - ``TutoringEnvironment``
* :mod:`adaptive_tutor.simulator.policies` - baseline action functions
"""

from adaptive_tutor.simulator.calibration import (
    LearnerParams,
    calibrate_from_stats,
    default_learner_params,
)
from adaptive_tutor.simulator.environment import TutoringEnvironment
from adaptive_tutor.simulator.learner import SyntheticLearner
from adaptive_tutor.simulator.policies import (
    TutorPolicy,
    heuristic_tutor_policy,
    random_tutor_policy,
    scripted_advanced_learner_policy,
    scripted_struggling_learner_policy,
)

__all__ = [
    "LearnerParams",
    "SyntheticLearner",
    "TutoringEnvironment",
    "TutorPolicy",
    "calibrate_from_stats",
    "default_learner_params",
    "heuristic_tutor_policy",
    "random_tutor_policy",
    "scripted_advanced_learner_policy",
    "scripted_struggling_learner_policy",
]
