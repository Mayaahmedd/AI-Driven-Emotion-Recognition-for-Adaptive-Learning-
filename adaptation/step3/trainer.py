from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List

from adaptation.step3.config import ARCHETYPES
from adaptation.step3.dqn_agents import ExplanationDQNAgent, PracticeDQNAgent
from adaptation.step3.simulator import StudentSimulator


@dataclass
class TrainStats:
    episode_rewards: List[float]
    mean_reward: float


class Trainer:
    def __init__(self, simulator: StudentSimulator, explanation_agent: ExplanationDQNAgent, practice_agent: PracticeDQNAgent):
        self.simulator = simulator
        self.explanation_agent = explanation_agent
        self.practice_agent = practice_agent

    def train(self, episodes: int = 350) -> TrainStats:
        rewards: List[float] = []
        for _ in range(episodes):
            archetype = random.choice(self.simulator.config.train_archetypes or ARCHETYPES)
            self.simulator.set_archetype(archetype)
            state = self.simulator.reset()
            done = False
            total_reward = 0.0
            while not done:
                phase = self.simulator.phase()
                state_vec = self.simulator.to_vector(state)
                agent = self.explanation_agent if phase == "explanation" else self.practice_agent
                action = agent.choose_action(state_vec, explore=True)
                result = self.simulator.step(action)
                next_vec = self.simulator.to_vector(result.state)
                agent.remember((state_vec, agent.action_index(action), result.reward, next_vec, result.done))
                agent.train_step()
                total_reward += result.reward
                state = result.state
                done = result.done
            rewards.append(total_reward)
        return TrainStats(episode_rewards=rewards, mean_reward=sum(rewards) / max(1, len(rewards)))
