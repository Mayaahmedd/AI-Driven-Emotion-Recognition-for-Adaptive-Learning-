"""ALG 5 - Rule-based baseline with priority waterfall and mask fallthrough."""

from __future__ import annotations

from typing import List, Optional, Tuple

import gymnasium as gym
import numpy as np

from RL_Module.agents.base_agent import BaseAgent
from RL_Module.mdp_definition import EMOTION_TO_ID, StudentState


class RuleBasedAgent(BaseAgent):
    name = "Rule"

    def train(self, env: gym.Env, total_timesteps: int, seed: int) -> None:
        pass

    def needs_training(self) -> bool:
        return False

    def _rules(self, s: StudentState, persistent: bool) -> List[int]:
        """Priority-ordered candidate actions (spec order)."""
        candidates = []
        if persistent:
            candidates.append(5)
        if s.emotion_id == EMOTION_TO_ID["frustrated"]:
            candidates.append(5)
        if s.knowledge < 0.3:
            candidates.append(4)
        if s.confusion > 0.6:
            candidates.append(9)
        if s.boredom > 0.6:
            candidates.append(3)
        if s.frustration > 0.4:
            candidates.append(2)
        if s.knowledge > 0.7 and s.engagement > 0.6:
            candidates.append(1)
        candidates.append(0)
        return candidates

    def predict(self, obs: np.ndarray, action_mask: Optional[np.ndarray] = None) -> int:
        s = StudentState.from_vec(obs)
        persistent = bool(obs[2] > 0.7) if action_mask is None else False

        if action_mask is None:
            action_mask = np.ones(10, dtype=np.int8)

        for action in self._rules(s, persistent):
            if action_mask[action] == 1:
                return action

        allowed = np.where(action_mask == 1)[0]
        if len(allowed) == 0:
            return 0
        return int(allowed[0])

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass
