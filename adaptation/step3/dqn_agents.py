from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from adaptation.step3.config import DQNConfig, EXPLANATION_ACTIONS, PRACTICE_ACTIONS


Transition = Tuple[List[float], int, float, List[float], bool]


class MLPQ(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class DQNAgent:
    actions: List[str]
    config: DQNConfig

    def __post_init__(self) -> None:
        self.state_dim = 10
        self.action_dim = len(self.actions)
        self.online = MLPQ(self.state_dim, self.action_dim)
        self.target = MLPQ(self.state_dim, self.action_dim)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = optim.Adam(self.online.parameters(), lr=self.config.lr)
        self.replay: Deque[Transition] = deque(maxlen=self.config.replay_size)
        self.epsilon = self.config.epsilon_start
        self.learn_steps = 0

    def choose_action(self, state_vec: List[float], explore: bool = True) -> str:
        if explore and random.random() < self.epsilon:
            return random.choice(self.actions)
        with torch.no_grad():
            q_vals = self.online(torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0))
            idx = int(torch.argmax(q_vals, dim=1).item())
        return self.actions[idx]

    def action_index(self, action: str) -> int:
        return self.actions.index(action)

    def remember(self, transition: Transition) -> None:
        self.replay.append(transition)

    def train_step(self) -> float | None:
        if len(self.replay) < self.config.batch_size:
            return None

        batch = random.sample(self.replay, self.config.batch_size)
        states, acts, rewards, next_states, dones = zip(*batch)
        states_t = torch.tensor(states, dtype=torch.float32)
        acts_t = torch.tensor(acts, dtype=torch.int64).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states_t = torch.tensor(next_states, dtype=torch.float32)
        dones_t = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        q = self.online(states_t).gather(1, acts_t)
        with torch.no_grad():
            max_next_q = self.target(next_states_t).max(dim=1, keepdim=True).values
            target_q = rewards_t + (1.0 - dones_t) * self.config.gamma * max_next_q

        loss = nn.MSELoss()(q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % self.config.target_update_steps == 0:
            self.target.load_state_dict(self.online.state_dict())

        self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)
        return float(loss.item())


class ExplanationDQNAgent(DQNAgent):
    def __init__(self, config: DQNConfig):
        super().__init__(actions=list(EXPLANATION_ACTIONS), config=config)


class PracticeDQNAgent(DQNAgent):
    def __init__(self, config: DQNConfig):
        super().__init__(actions=list(PRACTICE_ACTIONS), config=config)
