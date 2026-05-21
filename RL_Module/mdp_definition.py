"""MDP definition: S, A, P, R, gamma - imported by every other module.

CLIPPING EXPLAINED
==================
Clipping constrains a value to stay within [low, high].
Formula: clip(x, low, high) = max(low, min(high, x))

Used in three places:

1. State variables (student_model.py):
   knowledge = clip(knowledge + delta, 0, 1)
   Prevents knowledge from going below 0 (impossible) or above 1 (fully mastered).

2. Reward function (reward_function.py):
   reward = clip(total, -1.0, 1.0)
   Prevents extreme reward values from destabilizing neural network weight updates.
   SOURCE: Mnih et al. (2015) DQN paper clips rewards for the same reason.

3. Normalized knowledge gain (reward_function.py):
   delta_k_norm = clip((k_new - k_old) / (1 - k_old + 1e-8), -1, 1)
   The 1e-8 epsilon prevents division by zero when k_old = 1.0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, Set, Tuple

import numpy as np

# ?? Emotions (FER output) ????????????????????????????????????????????????????
EMOTIONS: Tuple[str, ...] = ("confused", "bored", "frustrated", "engaged")
# emotion_id: 0=confused, 1=bored, 2=frustrated, 3=engaged

EMOTION_TO_ID: Dict[str, int] = {e: i for i, e in enumerate(EMOTIONS)}
ID_TO_EMOTION: Dict[int, str] = {i: e for i, e in enumerate(EMOTIONS)}

# ?? Actions ???????????????????????????????????????????????????????????????????
ACTIONS: Tuple[str, ...] = (
    "easier_question",       # 0
    "harder_question",       # 1
    "give_hint",             # 2
    "motivational_message",  # 3
    "scaffold",              # 4
    "change_pacing",         # 5
    "reflection_prompt",     # 6
    "strategy_guidance",     # 7
    "autonomy_support",      # 8
    "explanation",           # 9
)

ACTION_TO_ID: Dict[str, int] = {a: i for i, a in enumerate(ACTIONS)}
ID_TO_ACTION: Dict[int, str] = {i: a for i, a in enumerate(ACTIONS)}

# ?? MDP components ????????????????????????????????????????????????????????????
GAMMA: float = 0.99
STATE_DIM: int = 6
ACTION_DIM: int = 10
TRANSITION_NOISE_STD: float = 0.02

# Persistent frustration thresholds
FRUSTRATION_PERSISTENT_ON: float = 0.7
FRUSTRATION_PERSISTENT_OFF: float = 0.5
FRUSTRATION_PERSISTENT_STEPS: int = 3

# Action masking thresholds
FRUSTRATION_BLOCK_HARDER: float = 0.6
FRUSTRATION_BLOCK_STRATEGY: float = 0.8
ENGAGEMENT_MIN_REFLECTION: float = 0.5
KNOWLEDGE_MIN_AUTONOMY: float = 0.4
KNOWLEDGE_MAX_SCAFFOLD: float = 0.5

# Cooldown steps after using an action (action_id -> steps)
COOLDOWNS: Dict[int, int] = {
    2: 3,   # give_hint
    3: 4,   # motivational_message
    5: 5,   # change_pacing
    6: 5,   # reflection_prompt
    7: 4,   # strategy_guidance
    8: 6,   # autonomy_support
    9: 3,   # explanation
}

# Emergency mode: only these actions allowed when persistent_frustration_flag
EMERGENCY_ALLOWED: FrozenSet[int] = frozenset({5, 8, 9})

# Pedagogically optimal actions per emotion_id (for adaptation_accuracy metric)
BEST_ACTION_MAP: Dict[int, Set[int]] = {
    0: {2, 9, 7},   # confused -> hint, explanation, strategy_guidance
    3: {1, 4, 6},   # engaged -> harder_question, scaffold, reflection_prompt
    2: {5, 0, 8},   # frustrated -> change_pacing, easier_question, autonomy_support
    1: {3, 8, 4},   # bored -> motivational_message, autonomy_support, scaffold
}

# String-key view for human-readable lookups (backward compatibility)
BEST_ACTION_MAP_BY_NAME: Dict[str, Set[int]] = {
    ID_TO_EMOTION[eid]: actions for eid, actions in BEST_ACTION_MAP.items()
}

# Student personality types (for result tables)
STUDENT_TYPES: Tuple[str, ...] = (
    "fast_learner",
    "slow_learner",
    "easily_frustrated",
    "patient",
)


@dataclass
class StudentState:
    """S = [knowledge, engagement, frustration, confusion, boredom, emotion_id]"""

    knowledge: float
    engagement: float
    frustration: float
    confusion: float
    boredom: float
    emotion_id: int

    def as_vec(self) -> np.ndarray:
        return np.array(
            [
                self.knowledge,
                self.engagement,
                self.frustration,
                self.confusion,
                self.boredom,
                float(self.emotion_id),
            ],
            dtype=np.float32,
        )

    @classmethod
    def from_vec(cls, vec: np.ndarray) -> "StudentState":
        eid = int(round(float(vec[5])))
        eid = max(0, min(3, eid))
        return cls(
            knowledge=float(vec[0]),
            engagement=float(vec[1]),
            frustration=float(vec[2]),
            confusion=float(vec[3]),
            boredom=float(vec[4]),
            emotion_id=eid,
        )

    @property
    def emotion(self) -> str:
        return ID_TO_EMOTION[self.emotion_id]

    def copy(self) -> "StudentState":
        return StudentState(
            knowledge=self.knowledge,
            engagement=self.engagement,
            frustration=self.frustration,
            confusion=self.confusion,
            boredom=self.boredom,
            emotion_id=self.emotion_id,
        )


def clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def normalized_knowledge_gain(k_old: float, k_new: float, eps: float = 1e-8) -> float:
    """Hake (1998) normalized gain: (k_new - k_old) / (1 - k_old)."""
    denom = max(1.0 - k_old, eps)
    return (k_new - k_old) / denom
