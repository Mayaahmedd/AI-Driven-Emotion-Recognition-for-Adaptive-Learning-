"""Smoke test: reward signs for state-based outcomes (action-agnostic)."""

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from RL_Module.mdp_definition import EMOTION_TO_ID, StudentState
from RL_Module.reward.reward_function import compute_reward


def test_knowledge_gain_positive():
    """Higher knowledge after transition should yield higher reward (same action)."""
    prev = StudentState(0.3, 0.6, 0.2, 0.4, 0.2, EMOTION_TO_ID["engaged"])
    new_good = StudentState(0.5, 0.62, 0.18, 0.35, 0.18, EMOTION_TO_ID["engaged"])
    new_bad = StudentState(0.25, 0.58, 0.22, 0.45, 0.22, EMOTION_TO_ID["engaged"])
    r_good = compute_reward(prev, 0, new_good)
    r_bad = compute_reward(prev, 0, new_bad)
    assert r_good > r_bad


def test_reward_clipped():
    prev = StudentState(0.1, 0.9, 0.0, 0.0, 0.0, 3)
    new = StudentState(0.99, 1.0, 0.0, 0.0, 0.0, 3)
    r = compute_reward(prev, 0, new)
    assert -1.0 <= r <= 1.0


def test_reward_action_invariant():
    """Same state transition must give identical reward for any action."""
    prev = StudentState(0.4, 0.6, 0.3, 0.4, 0.2, 2)
    new = StudentState(0.5, 0.7, 0.2, 0.3, 0.1, 3)
    rewards = [compute_reward(prev, a, new) for a in range(8)]
    assert len(set(round(r, 6) for r in rewards)) == 1
