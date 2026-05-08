from __future__ import annotations

from dataclasses import dataclass
from typing import List

from adaptation.step2.agents import ExplanationAgent, PracticeAgent
from adaptation.step2.baselines import BaselinePolicy
from adaptation.step2.config import ARCHETYPES
from adaptation.step2.simulator import StudentSimulator


@dataclass
class EvalStats:
    avg_reward: float
    avg_correct_rate: float
    avg_engagement: float
    avg_frustration: float
    avg_boredom: float


@dataclass
class ComparisonRow:
    policy: str
    archetype: str
    avg_reward: float
    avg_correct_rate: float
    avg_engagement: float
    avg_frustration: float
    avg_boredom: float


class Evaluator:
    def __init__(self, simulator: StudentSimulator, explanation_agent: ExplanationAgent, practice_agent: PracticeAgent):
        self.simulator = simulator
        self.explanation_agent = explanation_agent
        self.practice_agent = practice_agent

    def evaluate(self, episodes: int = 100) -> EvalStats:
        rewards: List[float] = []
        correct_rates: List[float] = []
        avg_engagements: List[float] = []
        avg_frustrations: List[float] = []
        avg_boredoms: List[float] = []
        for _ in range(episodes):
            state = self.simulator.reset()
            done = False
            total_reward = 0.0
            correct_sum = 0
            correct_count = 0
            engagement_sum = 0.0
            frustration_sum = 0.0
            boredom_sum = 0.0
            steps = 0
            while not done:
                phase = self.simulator.phase()
                key = self.simulator.discretize_emotion(state)
                agent = self.explanation_agent if phase == "explanation" else self.practice_agent
                action = agent.choose_action(key, explore=False)
                result = self.simulator.step(action)
                total_reward += result.reward
                correct_sum += result.correct
                correct_count += 1 if phase == "practice" else 0
                engagement_sum += result.state["engaged"]
                frustration_sum += result.state["frustrated"]
                boredom_sum += result.state["bored"]
                steps += 1
                state = result.state
                done = result.done
            rewards.append(total_reward)
            correct_rates.append(correct_sum / max(1, correct_count))
            avg_engagements.append(engagement_sum / max(1, steps))
            avg_frustrations.append(frustration_sum / max(1, steps))
            avg_boredoms.append(boredom_sum / max(1, steps))
        return EvalStats(
            avg_reward=sum(rewards) / len(rewards),
            avg_correct_rate=sum(correct_rates) / len(correct_rates),
            avg_engagement=sum(avg_engagements) / len(avg_engagements),
            avg_frustration=sum(avg_frustrations) / len(avg_frustrations),
            avg_boredom=sum(avg_boredoms) / len(avg_boredoms),
        )

    def compare_policies(self, baselines: List[BaselinePolicy], episodes_per_archetype: int = 80) -> List[ComparisonRow]:
        rows: List[ComparisonRow] = []
        for archetype in (self.simulator.config.eval_archetypes or ARCHETYPES):
            self.simulator.set_archetype(archetype)
            rl_stats = self.evaluate(episodes=episodes_per_archetype)
            rows.append(ComparisonRow("rl_q_learning", archetype, rl_stats.avg_reward, rl_stats.avg_correct_rate, rl_stats.avg_engagement, rl_stats.avg_frustration, rl_stats.avg_boredom))
            for baseline in baselines:
                stats = self._evaluate_baseline_policy(baseline, episodes_per_archetype, archetype)
                rows.append(ComparisonRow(baseline.name, archetype, stats.avg_reward, stats.avg_correct_rate, stats.avg_engagement, stats.avg_frustration, stats.avg_boredom))
        return rows

    def _evaluate_baseline_policy(self, baseline: BaselinePolicy, episodes: int, archetype: str) -> EvalStats:
        rewards: List[float] = []
        correct_rates: List[float] = []
        avg_engagements: List[float] = []
        avg_frustrations: List[float] = []
        avg_boredoms: List[float] = []
        self.simulator.set_archetype(archetype)
        for _ in range(episodes):
            state = self.simulator.reset()
            done = False
            total_reward = 0.0
            correct_sum = 0
            correct_count = 0
            engagement_sum = 0.0
            frustration_sum = 0.0
            boredom_sum = 0.0
            steps = 0
            while not done:
                phase = self.simulator.phase()
                action = baseline.choose_action(phase, state)
                result = self.simulator.step(action)
                total_reward += result.reward
                correct_sum += result.correct
                correct_count += 1 if phase == "practice" else 0
                engagement_sum += result.state["engaged"]
                frustration_sum += result.state["frustrated"]
                boredom_sum += result.state["bored"]
                steps += 1
                state = result.state
                done = result.done
            rewards.append(total_reward)
            correct_rates.append(correct_sum / max(1, correct_count))
            avg_engagements.append(engagement_sum / max(1, steps))
            avg_frustrations.append(frustration_sum / max(1, steps))
            avg_boredoms.append(boredom_sum / max(1, steps))
        return EvalStats(
            avg_reward=sum(rewards) / len(rewards),
            avg_correct_rate=sum(correct_rates) / len(correct_rates),
            avg_engagement=sum(avg_engagements) / len(avg_engagements),
            avg_frustration=sum(avg_frustrations) / len(avg_frustrations),
            avg_boredom=sum(avg_boredoms) / len(avg_boredoms),
        )
