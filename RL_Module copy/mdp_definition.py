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

# Actions (8) — simplified space; removed redundant pedagogical actions
ACTIONS: Tuple[str, ...] = (
    "hint",              # 0
    "scaffold",          # 1
    "encouragement",     # 2
    "simplify_problem",  # 3
    "harder_problem",    # 4
    "break",             # 5
    "explanation",       # 6
    "no_action",         # 7
)

ACTION_TO_ID: Dict[str, int] = {a: i for i, a in enumerate(ACTIONS)}
ID_TO_ACTION: Dict[int, str] = {i: a for i, a in enumerate(ACTIONS)}

# ?? MDP components ????????????????????????????????????????????????????????????
GAMMA: float = 0.99
STATE_DIM: int = 6
ACTION_DIM: int = 8
TRANSITION_NOISE_STD: float = 0.02

# IRT correctness model (van der Linden & Hambleton, 1997) — tunable defaults
IRT_BETA: float = 3.0
IRT_ALPHA: float = 1.0

# Emotion persistence AR(1) — literature-inspired simulator hyperparameters (Pekrun CVT)
LAMBDA_FRUSTRATION: float = 0.69
LAMBDA_ENGAGEMENT: float = 0.60
LAMBDA_CONFUSION: float = 0.47
LAMBDA_BOREDOM: float = 0.36
EMOTION_NOISE_STD: float = 0.03

# Challenge-skill mismatch (Flow Theory) — tunable band on [0,1] scale
MISMATCH_HIGH: float = 0.3
MISMATCH_LOW: float = -0.3

# Asymmetric learning (productive failure; Guzmán & Cruz-Mercado, 2025)
GAIN_INCORRECT_FACTOR: float = 1.0
GAIN_CORRECT_FACTOR: float = 0.1

# Episode difficulty sampling (internal latent; not in observation)
DIFFICULTY_INIT_LOW: float = 0.3
DIFFICULTY_INIT_HIGH: float = 0.8
DIFFICULTY_STEP: float = 0.1

# Persistent frustration thresholds
FRUSTRATION_PERSISTENT_ON: float = 0.7
FRUSTRATION_PERSISTENT_OFF: float = 0.5
FRUSTRATION_PERSISTENT_STEPS: int = 3

# Action masking thresholds
FRUSTRATION_BLOCK_HARDER: float = 0.6

# Cooldown steps after using an action (action_id -> steps)
COOLDOWNS: Dict[int, int] = {
    0: 1,   # hint — lightweight, rapidly reusable
    1: 1,   # scaffold
    2: 3,   # encouragement
    3: 2,   # simplify_problem — overload tool, not universal filler
    5: 3,   # break
    6: 2,   # explanation
}
# harder_problem(4), no_action(7): no cooldown

# Emergency: hint, scaffold, encouragement, break (cooldowns still apply)
EMERGENCY_ALLOWED: FrozenSet[int] = frozenset({0, 1, 2, 5})

# Pedagogically optimal actions per emotion_id (adaptation_accuracy metric only)
BEST_ACTION_MAP: Dict[int, Set[int]] = {
    0: {0, 6, 1},   # confused -> hint, explanation, scaffold
    3: {4, 1},      # engaged -> harder_problem, scaffold
    2: {5, 3, 2},   # frustrated -> break, simplify_problem, encouragement
    1: {2, 4},      # bored -> encouragement, harder_problem
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
