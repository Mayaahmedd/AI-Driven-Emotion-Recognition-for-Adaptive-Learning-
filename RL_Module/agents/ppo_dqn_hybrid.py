"""ALG 3 - PPO + DQN hybrid: 0.6*PPO_probs + 0.4*norm(DQN_Q). Eval-only combination."""

from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np
import torch

from RL_Module import config
from RL_Module.agents.base_agent import BaseAgent
from RL_Module.agents.dqn_agent import DQNAgent, make_env
from RL_Module.agents.ppo_agent import PPOAgent, make_masked_env


class PPODQNHybridAgent(BaseAgent):
    name = "PPO_DQN"

    def __init__(self):
        self.ppo = PPOAgent()
        self.dqn = DQNAgent()

    def train(self, env: gym.Env, total_timesteps: int, seed: int) -> None:
        config.set_all_seeds(seed)
        masked_env = make_masked_env(seed)
        plain_env = make_env(seed)
        self.ppo.train(masked_env, total_timesteps, seed)
        self.dqn.train(plain_env, total_timesteps, seed)
        masked_env.close()
        plain_env.close()

    def predict(self, obs: np.ndarray, action_mask: Optional[np.ndarray] = None) -> int:
        if action_mask is None:
            action_mask = np.ones(10, dtype=np.int8)
        mask_bool = action_mask.astype(bool) if action_mask.dtype != bool else action_mask

        ppo_probs = self.ppo.get_action_probs(obs, mask_bool)
        q_vals = self.dqn.get_q_values(obs)
        q_min, q_max = q_vals.min(), q_vals.max()
        if q_max - q_min < 1e-8:
            q_norm = np.zeros_like(q_vals)
        else:
            q_norm = (q_vals - q_min) / (q_max - q_min + 1e-8)
        q_norm[action_mask == 0] = 0.0

        combined = config.HYBRID_PPO_WEIGHT * ppo_probs + config.HYBRID_DQN_WEIGHT * q_norm
        combined[action_mask == 0] = 0.0
        return int(np.argmax(combined))

    def save(self, path: str) -> None:
        self.ppo.save(path + "_ppo")
        self.dqn.save(path + "_dqn")

    def load(self, path: str) -> None:
        self.ppo.load(path + "_ppo")
        self.dqn.load(path + "_dqn")
