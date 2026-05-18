"""Smoke test: action masking blocks harder_question when frustrated."""

import numpy as np

from RL_Module.environment.student_env import StudentEnv
from RL_Module.mdp_definition import EMOTION_TO_ID


def test_harder_blocked_when_frustrated():
    env = StudentEnv(max_episode_steps=5, population_seed=7)
    env.reset(seed=7)
    env._state.frustration = 0.8
    env._state.emotion_id = EMOTION_TO_ID["frustrated"]
    mask = env.get_action_mask()
    assert mask[1] == 0, "harder_question should be blocked when frustrated"


def test_emergency_mode_only_three_actions():
    env = StudentEnv(max_episode_steps=5, population_seed=7)
    env.reset(seed=7)
    env._persistent_frustration_flag = True
    mask = env.get_action_mask()
    allowed = set(np.where(mask == 1)[0])
    assert allowed == {5, 8, 9}
    env.close()
