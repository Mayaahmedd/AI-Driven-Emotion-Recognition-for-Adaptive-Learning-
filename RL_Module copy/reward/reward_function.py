"""
Multi-objective educational reward (action-agnostic).

r_t = wk * delta_k_norm + we * engagement - wf * frustration - wb * boredom - wc * confusion

Weights come from config.REWARD_PRESETS / REWARD_WEIGHTS or sensitivity overrides.
Optional legacy terms (zone bonus, wrong-answer penalty) via config flags.

Numerical weights are literature-inspired simulator hyperparameters calibrated
through sensitivity analysis — not universal psychological constants.
"""

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np

from RL_Module import config
from RL_Module.mdp_definition import ACTION_TO_ID, StudentState


def _to_state(state: Union[np.ndarray, StudentState]) -> StudentState:
    if isinstance(state, np.ndarray):
        return StudentState.from_vec(state)
    return state


def compute_reward(
    prev_state: Union[np.ndarray, StudentState],
    action: int,
    new_state: Union[np.ndarray, StudentState],
    persistent_flag: bool = False,
    last_answer_correct: bool = True,
    last_answer_wrong: Optional[bool] = None,
    strict_no_emotion: bool = False,
    **kwargs: Any,
) -> float:
    """
    Multi-objective reward. Optional no_action struggle penalty when student needs intervention.

    strict_no_emotion: knowledge-only reward r = wk * delta_k_norm (+ optional wrong-answer term).
    Same weight preset as full mode; affect terms are excluded, not re-weighted.
    """
    weights = config.get_reward_weights()
    wk = float(weights.get("wk", 0.5))

    prev = _to_state(prev_state)
    new = _to_state(new_state)

    if last_answer_wrong is None:
        last_answer_wrong = not last_answer_correct
    elif "last_answer_wrong" in kwargs and kwargs["last_answer_wrong"] is not None:
        last_answer_wrong = bool(kwargs["last_answer_wrong"])

    delta_k_norm = np.clip(
        (new.knowledge - prev.knowledge) / (1.0 - prev.knowledge + 1e-8),
        -1.0,
        1.0,
    )

    if strict_no_emotion:
        total = wk * delta_k_norm
        if config.USE_WRONG_ANSWER_PENALTY and last_answer_wrong:
            total -= wk * float(config.WRONG_ANSWER_PENALTY)
        return float(np.clip(total, -1.0, 1.0))

    we = float(weights.get("we", 0.2))
    wf = float(weights.get("wf", 0.15))
    wb = float(weights.get("wb", 0.10))
    wc = float(weights.get("wc", 0.05))

    total = (
        wk * delta_k_norm
        + we * new.engagement
        - wf * new.frustration
        - wb * new.boredom
        - wc * new.confusion
    )

    if persistent_flag:
        total -= wf * float(config.PERSISTENT_PENALTY)

    if config.USE_ZONE_BONUS:
        in_zone = (
            new.knowledge > config.ZONE_MIN_KNOWLEDGE
            and new.engagement > config.ZONE_MIN_ENGAGEMENT
            and new.frustration < config.ZONE_MAX_FRUSTRATION
            and new.confusion < config.ZONE_MAX_CONFUSION
            and new.boredom < config.ZONE_MAX_BOREDOM
        )
        if in_zone:
            total += wk * float(config.ZONE_BONUS)

    if config.USE_WRONG_ANSWER_PENALTY and last_answer_wrong:
        total -= wk * float(config.WRONG_ANSWER_PENALTY)

    if action == ACTION_TO_ID["no_action"]:
        struggling = (
            new.confusion > config.NO_ACTION_CONFUSION_THRESH
            or new.frustration > config.NO_ACTION_FRUSTRATION_THRESH
        )
        if struggling:
            total -= float(config.NO_ACTION_STRUGGLE_PENALTY)

    return float(np.clip(total, -1.0, 1.0))
