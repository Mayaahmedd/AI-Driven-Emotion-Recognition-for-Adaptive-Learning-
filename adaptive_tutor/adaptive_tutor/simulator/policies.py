"""Lightweight tutor policies for baselines and regression tests.

Each policy is a **callable** ``(LearnerState) -> str`` returning an
action name from :data:`~adaptive_tutor.memory.providers.dataset_provider.ASSISTMENTS_ACTIONS`.
"""

from __future__ import annotations

import random
from collections.abc import Callable

from adaptive_tutor.memory.providers.dataset_provider import ASSISTMENTS_ACTIONS
from adaptive_tutor.state.state import LearnerState

TutorPolicy = Callable[[LearnerState], str]


def random_tutor_policy(*, seed: int = 0) -> TutorPolicy:
    """Uniform random over the ASSISTments action vocabulary."""
    rng = random.Random(seed)

    def choose(_state: LearnerState) -> str:
        return rng.choice(ASSISTMENTS_ACTIONS)

    return choose


def heuristic_tutor_policy() -> TutorPolicy:
    """Simple if-then rules using only actions in ``ASSISTMENTS_ACTIONS``."""

    def choose(state: LearnerState) -> str:
        if state.rolling_emotions.frustrated > 0.55 or state.frustration_trend > 0.35:
            return "encouragement"
        if state.mastery < 0.35 and state.perf.recent_accuracy < 0.4:
            return "give_hint"
        if state.rolling_emotions.bored > 0.5:
            return "harder_problem"
        if state.mastery > 0.75:
            return "advance_to_next_skill"
        return "retry_current_skill"

    return choose


def scripted_struggling_learner_policy(*, seed: int = 1) -> TutorPolicy:
    """Bias toward ``easier_problem`` and ``encouragement``."""
    rng = random.Random(seed)
    weights = (
        ("easier_problem", 0.35),
        ("encouragement", 0.30),
        ("give_hint", 0.20),
        ("retry_current_skill", 0.15),
    )

    def choose(_state: LearnerState) -> str:
        r = rng.random()
        acc = 0.0
        for name, w in weights:
            acc += w
            if r <= acc:
                return name
        return "easier_problem"

    return choose


def scripted_advanced_learner_policy(*, seed: int = 2) -> TutorPolicy:
    """Bias toward ``harder_problem`` and ``advance_to_next_skill``."""
    rng = random.Random(seed)
    weights = (
        ("harder_problem", 0.35),
        ("advance_to_next_skill", 0.35),
        ("retry_current_skill", 0.20),
        ("give_hint", 0.10),
    )

    def choose(_state: LearnerState) -> str:
        r = rng.random()
        acc = 0.0
        for name, w in weights:
            acc += w
            if r <= acc:
                return name
        return "harder_problem"

    return choose
