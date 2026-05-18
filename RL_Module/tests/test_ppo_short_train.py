"""Smoke test: short MaskablePPO training run."""

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

import pytest

pytest.importorskip("sb3_contrib")
pytest.importorskip("stable_baselines3")

from RL_Module.agents.ppo_agent import PPOAgent, make_masked_env


def test_ppo_short_train():
    env = make_masked_env(seed=42)
    agent = PPOAgent()
    agent.train(env, total_timesteps=512, seed=42)
    obs, info = env.reset(seed=42)
    action = agent.predict(obs, info["action_masks"])
    assert 0 <= action < 10
    env.close()
