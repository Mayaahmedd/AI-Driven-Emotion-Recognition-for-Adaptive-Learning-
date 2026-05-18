"""ALG 4 - LinUCB bandit (fast emotion changes) + DQN (long-term)."""

from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np

from RL_Module import config
from RL_Module.agents.base_agent import BaseAgent
from RL_Module.agents.dqn_agent import DQNAgent, make_env
from RL_Module.mdp_definition import ACTION_DIM


class LinUCB:
    """Contextual LinUCB - sklearn-free implementation."""

    def __init__(self, n_arms: int, context_dim: int, alpha: float = 1.0):
        self.n_arms = n_arms
        self.d = context_dim
        self.alpha = alpha
        self.A = [np.eye(context_dim) for _ in range(n_arms)]
        self.b = [np.zeros(context_dim) for _ in range(n_arms)]

    def select(self, context: np.ndarray, mask: np.ndarray) -> int:
        ucb = np.full(self.n_arms, -np.inf)
        for a in range(self.n_arms):
            if mask[a] == 0:
                continue
            try:
                A_inv = np.linalg.inv(self.A[a])
            except np.linalg.LinAlgError:
                A_inv = np.eye(self.d)
            theta = A_inv @ self.b[a]
            p = float(theta @ context + self.alpha * np.sqrt(context @ A_inv @ context))
            ucb[a] = p
        return int(np.argmax(ucb))

    def update(self, arm: int, context: np.ndarray, reward: float) -> None:
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context


class BanditDQNAgent(BaseAgent):
    name = "Bandit_DQN"

    def __init__(self):
        self.dqn = DQNAgent()
        self.bandit = LinUCB(ACTION_DIM, context_dim=3, alpha=config.BANDIT_ALPHA)
        self._prev_emotion_id: int = 3
        self._last_reward: float = 0.0
        self._persistent_flag: bool = False

    def _context(self, obs: np.ndarray) -> np.ndarray:
        """Context = [emotion_id/3.0, frustration, boredom]."""
        eid = int(round(float(obs[5])))
        eid = max(0, min(3, eid))
        return np.array([eid / 3.0, float(obs[2]), float(obs[4])], dtype=np.float64)

    def _bandit_trigger(self, obs: np.ndarray) -> bool:
        eid = int(round(float(obs[5])))
        eid = max(0, min(3, eid))
        trigger = self._persistent_flag or abs(eid - self._prev_emotion_id) >= 1
        self._prev_emotion_id = eid
        return trigger

    def train(self, env: gym.Env, total_timesteps: int, seed: int) -> None:
        plain_env = make_env(seed)
        self.dqn.train(plain_env, total_timesteps, seed)
        plain_env.close()

    def predict(
        self,
        obs: np.ndarray,
        action_mask: Optional[np.ndarray] = None,
        persistent_flag: bool = False,
    ) -> int:
        if action_mask is None:
            action_mask = np.ones(10, dtype=np.int8)
        self._persistent_flag = persistent_flag

        if self._bandit_trigger(obs):
            ctx = self._context(obs)
            action = self.bandit.select(ctx, action_mask)
            self.bandit.update(action, ctx, self._last_reward)
            return action

        return self.dqn.predict(obs, action_mask)

    def set_last_reward(self, reward: float) -> None:
        self._last_reward = reward

    def save(self, path: str) -> None:
        self.dqn.save(path)

    def load(self, path: str) -> None:
        self.dqn.load(path)
