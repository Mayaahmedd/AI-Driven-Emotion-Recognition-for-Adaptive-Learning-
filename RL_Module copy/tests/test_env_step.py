"""Smoke test: env reset and step."""

import numpy as np

from RL_Module.environment.student_env import StudentEnv


def test_reset_step():
    env = StudentEnv(max_episode_steps=10, population_seed=42)
    obs, info = env.reset(seed=42)
    assert obs.shape == (6,)
    assert obs.dtype == np.float32
    assert obs.min() >= 0
    assert obs.max() <= 3.0  # emotion_id raw 0-3
    assert "action_masks" in info
    mask = env.get_action_mask()
    action = int(np.where(mask == 1)[0][0])
    obs2, reward, term, trunc, info2 = env.step(action)
    assert obs2.shape == (6,)
    assert -1 <= reward <= 1
    assert "terminated" in info2
    assert "truncated" in info2
    env.close()


def test_masked_action_raises():
    env = StudentEnv(max_episode_steps=5, population_seed=0)
    env.reset(seed=0)
    mask = env.get_action_mask()
    blocked = [i for i in range(8) if mask[i] == 0]
    if blocked:
        try:
            env.step(blocked[0])
            assert False, "Should have raised"
        except ValueError:
            pass
    env.close()
