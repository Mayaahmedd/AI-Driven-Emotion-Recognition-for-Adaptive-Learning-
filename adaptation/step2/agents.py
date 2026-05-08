from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List

from adaptation.step2.config import EXPLANATION_ACTIONS, PRACTICE_ACTIONS


@dataclass
class TabularQAgent:
    actions: List[str]
    alpha: float = 0.2
    gamma: float = 0.95
    epsilon: float = 0.2
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.05
    q: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def choose_action(self, state_key: str, explore: bool = True) -> str:
        if state_key not in self.q:
            self.q[state_key] = {a: 0.0 for a in self.actions}
        if explore and random.random() < self.epsilon:
            return random.choice(self.actions)
        return max(self.q[state_key], key=self.q[state_key].get)

    def update(self, state_key: str, action: str, reward: float, next_key: str, done: bool) -> None:
        if state_key not in self.q:
            self.q[state_key] = {a: 0.0 for a in self.actions}
        if next_key not in self.q:
            self.q[next_key] = {a: 0.0 for a in self.actions}
        current = self.q[state_key][action]
        best_next = 0.0 if done else max(self.q[next_key].values())
        target = reward + self.gamma * best_next
        self.q[state_key][action] = current + self.alpha * (target - current)

    def decay(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class ExplanationAgent(TabularQAgent):
    def __init__(self) -> None:
        super().__init__(actions=list(EXPLANATION_ACTIONS))


class PracticeAgent(TabularQAgent):
    def __init__(self) -> None:
        super().__init__(actions=list(PRACTICE_ACTIONS))
