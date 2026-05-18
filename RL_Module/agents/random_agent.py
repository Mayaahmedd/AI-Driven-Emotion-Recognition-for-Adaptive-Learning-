"""ALG 6 - Random baseline over non-masked actions only."""

from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np

from RL_Module.agents.base_agent import BaseAgent


class RandomAgent(BaseAgent):
    name = "Random"

    def __init__(self, seed: int = 42):
        self._rng = np.random.default_rng(seed)

    def train(self, env: gym.Env, total_timesteps: int, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def needs_training(self) -> bool:
        return False

    def predict(self, obs: np.ndarray, action_mask: Optional[np.ndarray] = None) -> int:
        if action_mask is None:
            return int(self._rng.integers(0, 10))
        available = np.where(action_mask == 1)[0]
        if len(available) == 0:
            return 0
        return int(self._rng.choice(available))

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass
