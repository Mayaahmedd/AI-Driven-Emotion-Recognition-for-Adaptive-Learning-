"""
Reward function R - equal weighting across outcome dimensions.

Equal weights (W = 1/7) are used deliberately to avoid introducing researcher
bias into the learning signal. This design choice follows the principle of
minimal assumptions: in the absence of empirical data quantifying the relative
importance of each affective dimension, equal weighting is the most
epistemically honest approach (Dawes, 1979).

The priority ordering of outcomes is instead encoded in:
  - The transition function (which dimensions change most per action)
  - The action masking rules (which prevent harmful actions)
  - The optimal zone bonus (which rewards holistic good states)

Reference: Dawes, R.M. (1979). The robust beauty of improper linear models
in decision making. American Psychologist, 34(7), 571-582.

This module contains ZERO references to action IDs in the reward signal.
"""

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np

from RL_Module import config
from RL_Module.mdp_definition import StudentState


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
    **kwargs: Any,
) -> float:
    """
    Equal-weight reward (Dawes, 1979).

    Seven terms with per-term weights (default W = 1/7 each):
      + delta_k_norm, + delta_e, - confusion, - boredom,
      + frustration_term, + zone_bonus, - wrong_answer_term

    The action parameter is retained for API compatibility but is never used.
    Sensitivity overrides use W_knowledge, W_engagement, W_affective, etc.
    """
    sw = config.SENSITIVITY_WEIGHTS if config.USE_SENSITIVITY_WEIGHTS else {}

    w_default = sw.get("W", 1.0 / 7.0)
    w_k = sw.get("W_knowledge", w_default)
    w_e = sw.get("W_engagement", w_default)
    w_conf = sw.get("W_confusion", sw.get("W_affective", w_default))
    w_bored = sw.get("W_boredom", sw.get("W_affective", w_default))
    w_frust = sw.get("W_frustration", sw.get("W_affective", w_default))
    w_zone = sw.get("W_zone", w_default)
    w_wrong = sw.get("W_wrong", w_default)

    frust_high = sw.get("frustration_high_threshold", config.FRUSTRATION_HIGH_THRESHOLD)
    frust_pen = sw.get("frustration_penalty_high", config.FRUSTRATION_PENALTY_HIGH)
    persist_pen = sw.get("persistent_penalty", config.PERSISTENT_PENALTY)
    zone_bonus_val = sw.get("zone_bonus", config.ZONE_BONUS)
    wrong_pen = sw.get("wrong_answer_penalty", config.WRONG_ANSWER_PENALTY)

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
    delta_e = new.engagement - prev.engagement

    if persistent_flag:
        frustration_term = -persist_pen
    elif new.frustration > frust_high:
        frustration_term = -frust_pen
    else:
        frustration_term = 0.0  # productive frustration (0.3 < f <= high): no penalty

    in_zone = (
        new.knowledge > config.ZONE_MIN_KNOWLEDGE
        and new.engagement > config.ZONE_MIN_ENGAGEMENT
        and new.frustration < config.ZONE_MAX_FRUSTRATION
        and new.confusion < config.ZONE_MAX_CONFUSION
        and new.boredom < config.ZONE_MAX_BOREDOM
    )
    zone_bonus = zone_bonus_val if in_zone else 0.0
    wrong_term = wrong_pen if last_answer_wrong else 0.0

    total = (
        w_k * delta_k_norm
        + w_e * delta_e
        - w_conf * new.confusion
        - w_bored * new.boredom
        + w_frust * frustration_term
        + w_zone * zone_bonus
        - w_wrong * wrong_term
    )
    return float(np.clip(total, -1.0, 1.0))
