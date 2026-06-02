"""
Abstract base agent - Stage-2 extension point.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import gymnasium as gym
import numpy as np


class BaseAgent(ABC):
    name: str = "BaseAgent"

    @abstractmethod
    def train(self, env: gym.Env, total_timesteps: int, seed: int) -> None:
        ...

    @abstractmethod
    def predict(self, obs: np.ndarray, action_mask: Optional[np.ndarray] = None) -> int:
        ...

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass

    def needs_training(self) -> bool:
        return True
