"""Phase 9 integration smoke test."""

import numpy as np

from RL_Module.agents.rule_based import RuleBasedAgent
from RL_Module.environment.student_env import StudentEnv
from RL_Module.logging_utils.csv_logger import CSVLogger
from RL_Module.reward.reward_function import compute_reward


def test_phase9_smoke():
    env = StudentEnv(render_mode=None)
    obs, info = env.reset(seed=42)
    agent = RuleBasedAgent()
    logger = CSVLogger(algorithm="Rule", seed=42, run_name="test_run")

    total_reward = 0.0
    step = 0
    for step in range(50):
        mask = env.get_action_mask()
        action = agent.predict(obs, mask)
        obs, reward, terminated, truncated, info = env.step(action)
        logger.log_step(step + 1, 1, action, reward, total_reward, obs, info)
        total_reward += reward
        if terminated or truncated:
            break

    env.reset(seed=0)
    assert obs.shape == (6,)
    assert obs.dtype == np.float32
    assert isinstance(total_reward, float)
    logger.close()
    env.close()
