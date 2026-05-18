"""
Reward function R - weighted sum with thesis citations in comments.
"""

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np

from RL_Module.mdp_definition import ID_TO_EMOTION, StudentState, normalized_knowledge_gain

# Weights (cite in thesis)
W_KNOWLEDGE = 0.40      # Hake (1998)
W_ENGAGEMENT = 0.20     # Fredricks et al. (2004)
W_CONFUSION = 0.15      # Sweller (1988)
W_BOREDOM = 0.10        # D'Mello et al. (2012)
W_FRUSTRATION = 0.15    # Kort et al. (2001) - high frustration only
W_PERSISTENT = 0.30     # systemic failure
W_OPTIMAL = 0.20        # Csikszentmihalyi (1990)

BONUS_MAP = {
    ("confused", 2): 0.20,
    ("confused", 9): 0.25,
    ("confused", 7): 0.20,
    ("frustrated", 5): 0.25,
    ("frustrated", 8): 0.15,
    ("bored", 3): 0.20,
    ("bored", 8): 0.20,
    ("engaged", 6): 0.25,
    ("engaged", 1): 0.20,
}

PENALTY_MAP = {
    ("frustrated", 1): -0.25,
    ("frustrated", 6): -0.20,
    ("confused", 1): -0.20,
    ("bored", 9): -0.10,
}


def _to_state(state: Union[np.ndarray, StudentState]) -> StudentState:
    if isinstance(state, np.ndarray):
        return StudentState.from_vec(state)
    return state


def _in_optimal_zone(s: StudentState) -> bool:
    return (
        s.knowledge > 0.5
        and s.engagement > 0.6
        and s.frustration < 0.3
        and s.confusion < 0.4
        and s.boredom < 0.3
    )


def _frustration_penalty(s: StudentState, persistent_flag: bool) -> float:
    """Productive frustration - MaTHiSiS Paper 9."""
    if persistent_flag:
        return W_PERSISTENT
    if s.frustration > 0.6:
        return W_FRUSTRATION
    if 0.3 < s.frustration <= 0.6:
        return 0.0
    return 0.0


def compute_reward(
    prev_state: Union[np.ndarray, StudentState],
    action: int,
    new_state: Union[np.ndarray, StudentState],
    persistent_flag: bool = False,
    last_answer_correct: bool = True,
    last_answer_wrong: Optional[bool] = None,
    use_emotion_bonuses: bool = True,
    **kwargs: Any,
) -> float:
    """
  R = weighted sum, clipped to [-1, +1].
  Accepts obs vectors (6,) or StudentState. Supports last_answer_correct or last_answer_wrong.
    """
    prev = _to_state(prev_state)
    new = _to_state(new_state)

    if last_answer_wrong is None:
        last_answer_wrong = not last_answer_correct
    elif "last_answer_wrong" in kwargs and kwargs["last_answer_wrong"] is not None:
        last_answer_wrong = bool(kwargs["last_answer_wrong"])

    emotion = ID_TO_EMOTION[new.emotion_id]

    dk_norm = normalized_knowledge_gain(prev.knowledge, new.knowledge)
    r = W_KNOWLEDGE * dk_norm
    r += W_ENGAGEMENT * (new.engagement - prev.engagement)
    r -= W_CONFUSION * new.confusion
    r -= W_BOREDOM * new.boredom
    r -= _frustration_penalty(new, persistent_flag)

    if _in_optimal_zone(new):
        r += W_OPTIMAL

    if use_emotion_bonuses:
        key = (emotion, action)
        r += BONUS_MAP.get(key, 0.0)
        r += PENALTY_MAP.get(key, 0.0)
        if last_answer_wrong:
            if action == 9:
                r += 0.20
            elif action == 2:
                r += 0.15

    return float(np.clip(r, -1.0, 1.0))
