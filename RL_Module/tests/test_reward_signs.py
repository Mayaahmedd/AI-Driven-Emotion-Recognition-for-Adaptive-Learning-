"""Smoke test: reward signs for pedagogically correct actions."""

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from RL_Module.mdp_definition import EMOTION_TO_ID, StudentState
from RL_Module.reward.reward_function import compute_reward


def test_confused_explanation_bonus():
    prev = StudentState(0.3, 0.6, 0.2, 0.7, 0.2, EMOTION_TO_ID["confused"])
    new = StudentState(0.35, 0.62, 0.18, 0.65, 0.18, EMOTION_TO_ID["confused"])
    r_good = compute_reward(prev, 9, new)  # explanation
    r_bad = compute_reward(prev, 1, new)   # harder_question
    assert r_good > r_bad


def test_reward_clipped():
    prev = StudentState(0.1, 0.9, 0.0, 0.0, 0.0, 3)
    new = StudentState(0.99, 1.0, 0.0, 0.0, 0.0, 3)
    r = compute_reward(prev, 0, new)
    assert -1.0 <= r <= 1.0
