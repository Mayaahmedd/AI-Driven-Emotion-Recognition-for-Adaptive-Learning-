"""Smoke test: action masking blocks harder_problem when frustrated."""

import numpy as np

from RL_Module.environment.student_env import StudentEnv
from RL_Module.mdp_definition import ACTION_TO_ID, EMOTION_TO_ID


def test_harder_blocked_when_frustrated():
    env = StudentEnv(max_episode_steps=5, population_seed=7)
    env.reset(seed=7)
    env._state.frustration = 0.8
    env._state.emotion_id = EMOTION_TO_ID["frustrated"]
    mask = env.get_action_mask()
    assert mask[ACTION_TO_ID["harder_problem"]] == 0, (
        "harder_problem should be blocked when frustrated"
    )


def test_emergency_mode_emergency_actions():
    env = StudentEnv(max_episode_steps=5, population_seed=7)
    env.reset(seed=7)
    env._persistent_frustration_flag = True
    mask = env.get_action_mask()
    allowed = set(np.where(mask == 1)[0])
    assert allowed == {0, 1, 2, 5}  # hint, scaffold, encouragement, break
    env.close()


def test_emergency_always_has_scaffold():
    """Scaffold has no cooldown and is in EMERGENCY_ALLOWED — mask never all-zero."""
    env = StudentEnv(max_episode_steps=5, population_seed=7)
    env.reset(seed=7)
    env._persistent_frustration_flag = True
    for a in (0, 2, 5, 6):
        env._cooldowns[a] = 10
    mask = env.get_action_mask()
    assert mask[ACTION_TO_ID["scaffold"]] == 1
    env.close()
