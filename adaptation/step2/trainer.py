from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List

from adaptation.step2.agents import ExplanationAgent, PracticeAgent
from adaptation.step2.config import ARCHETYPES
from adaptation.step2.simulator import StudentSimulator


@dataclass
class TrainStats:
    episode_rewards: List[float]
    mean_reward: float


class Trainer:
    def __init__(self, simulator: StudentSimulator, explanation_agent: ExplanationAgent, practice_agent: PracticeAgent):
        self.simulator = simulator
        self.explanation_agent = explanation_agent
        self.practice_agent = practice_agent

    def train(self, episodes: int = 300) -> TrainStats:
        rewards: List[float] = []
        for _ in range(episodes):
            archetype = random.choice(self.simulator.config.train_archetypes or ARCHETYPES)
            self.simulator.set_archetype(archetype)
            state = self.simulator.reset()
            done = False
            total_reward = 0.0
            while not done:
                phase = self.simulator.phase()
                key = self.simulator.discretize_emotion(state)
                agent = self.explanation_agent if phase == "explanation" else self.practice_agent
                action = agent.choose_action(key, explore=True)
                result = self.simulator.step(action)
                next_key = self.simulator.discretize_emotion(result.state)
                agent.update(key, action, result.reward, next_key, result.done)
                state = result.state
                total_reward += result.reward
                done = result.done
            self.explanation_agent.decay()
            self.practice_agent.decay()
            rewards.append(total_reward)
        return TrainStats(episode_rewards=rewards, mean_reward=sum(rewards) / max(1, len(rewards)))
