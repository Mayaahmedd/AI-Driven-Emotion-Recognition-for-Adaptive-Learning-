from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

from adaptation.step2.config import ARCHETYPES, EMOTIONS, EXPLANATION_ACTIONS, PRACTICE_ACTIONS, SimConfig

State = Dict[str, float]


@dataclass
class StepResult:
    state: State
    reward: float
    done: bool
    correct: int


class StudentSimulator:
    def __init__(self, config: SimConfig, seed: int | None = None):
        self.config = config
        self.rng = random.Random(seed)
        self.step_idx = 0
        self.state: State = {}
        self.archetype = "balanced"

    def set_archetype(self, archetype: str) -> None:
        if archetype not in ARCHETYPES:
            raise ValueError(f"Unknown archetype: {archetype}")
        self.archetype = archetype

    def reset(self) -> State:
        self.step_idx = 0
        engaged, confused, frustrated, bored = self._initial_emotions_for_archetype(self.archetype)
        self.state = self._normalize(
            {"engaged": engaged, "confused": confused, "frustrated": frustrated, "bored": bored, "delta_engaged": 0.0,
             "delta_confused": 0.0, "delta_frustrated": 0.0, "delta_bored": 0.0, "last_correct": 0.0, "progress": 0.0}
        )
        return dict(self.state)

    def phase(self) -> str:
        return "explanation" if self.step_idx < self.config.explanation_steps else "practice"

    def step(self, action: str) -> StepResult:
        if self.phase() == "explanation":
            if action not in EXPLANATION_ACTIONS:
                raise ValueError(f"Invalid explanation action: {action}")
            next_state, correct = self._transition_explanation(action)
            reward = self._reward_explanation(self.state, next_state)
        else:
            if action not in PRACTICE_ACTIONS:
                raise ValueError(f"Invalid practice action: {action}")
            next_state, correct = self._transition_practice(action)
            reward = self._reward_practice(self.state, next_state, correct)
        self.step_idx += 1
        done = self.step_idx >= self.config.steps_per_episode
        self.state = next_state
        return StepResult(state=dict(next_state), reward=reward, done=done, correct=correct)

    def _transition_explanation(self, action: str) -> Tuple[State, int]:
        prev = self.state
        engaged, confused, frustrated, bored = prev["engaged"], prev["confused"], prev["frustrated"], prev["bored"]
        if action == "slow_down":
            engaged += 0.05; confused -= 0.04; frustrated -= 0.02
        elif action == "give_micro_example":
            engaged += 0.04; confused -= 0.03
        elif action == "motivate":
            engaged += 0.03; frustrated -= 0.04; bored -= 0.02
        elif action == "quick_check":
            engaged += 0.02; bored -= 0.03; confused += 0.01
        else:
            bored += 0.02
        engaged, confused, frustrated, bored = self._apply_archetype_shift(engaged, confused, frustrated, bored)
        engaged += self.rng.uniform(-0.015, 0.015); confused += self.rng.uniform(-0.012, 0.012)
        frustrated += self.rng.uniform(-0.012, 0.012); bored += self.rng.uniform(-0.012, 0.012)
        return self._build_next_state(engaged, confused, frustrated, bored, prev["last_correct"]), int(prev["last_correct"])

    def _transition_practice(self, action: str) -> Tuple[State, int]:
        prev = self.state
        diff = {"question_easy": 0.2, "question_medium": 0.5, "question_hard": 0.8, "question_review": 0.35}[action]
        stress = prev["frustrated"] + prev["confused"] - prev["engaged"]
        correct_prob = self.config.base_correct_prob + 0.10 * prev["engaged"] - 0.22 * max(0.0, stress) - 0.25 * (diff - 0.5)
        correct_prob = self._adjust_correct_prob_for_archetype(correct_prob, action, prev)
        correct = 1 if self.rng.random() < max(0.1, min(0.95, correct_prob)) else 0
        engaged, confused, frustrated, bored = prev["engaged"], prev["confused"], prev["frustrated"], prev["bored"]
        if correct:
            engaged += 0.05; confused -= 0.03; frustrated -= 0.04; bored -= 0.01
        else:
            engaged -= 0.05; confused += 0.04; frustrated += 0.05; bored += 0.01
        if action == "question_hard":
            confused += 0.03; frustrated += 0.02
        elif action == "question_easy":
            bored += 0.02
        elif action == "question_review":
            confused -= 0.02
        engaged, confused, frustrated, bored = self._apply_archetype_shift(engaged, confused, frustrated, bored)
        engaged += self.rng.uniform(-0.015, 0.015); confused += self.rng.uniform(-0.015, 0.015)
        frustrated += self.rng.uniform(-0.015, 0.015); bored += self.rng.uniform(-0.015, 0.015)
        return self._build_next_state(engaged, confused, frustrated, bored, float(correct)), correct

    def _build_next_state(self, engaged: float, confused: float, frustrated: float, bored: float, last_correct: float) -> State:
        prev = self.state
        n = self._normalize_probs(engaged, confused, frustrated, bored)
        return self._normalize({"engaged": n[0], "confused": n[1], "frustrated": n[2], "bored": n[3],
                                "delta_engaged": n[0]-prev["engaged"], "delta_confused": n[1]-prev["confused"],
                                "delta_frustrated": n[2]-prev["frustrated"], "delta_bored": n[3]-prev["bored"],
                                "last_correct": last_correct, "progress": (self.step_idx + 1) / self.config.steps_per_episode})

    def _reward_explanation(self, prev: State, curr: State) -> float:
        return 0.55 * (curr["engaged"] - prev["engaged"]) - 0.25 * max(0.0, curr["confused"] - prev["confused"]) - 0.20 * max(0.0, curr["frustrated"] - prev["frustrated"])

    def _reward_practice(self, prev: State, curr: State, correct: int) -> float:
        return 0.35 * correct + 0.35 * (curr["engaged"] - prev["engaged"]) - 0.20 * curr["frustrated"] - 0.10 * curr["bored"]

    def _normalize(self, state: State) -> State:
        p = self._normalize_probs(state["engaged"], state["confused"], state["frustrated"], state["bored"])
        state["engaged"], state["confused"], state["frustrated"], state["bored"] = p
        return state

    def _normalize_probs(self, engaged: float, confused: float, frustrated: float, bored: float) -> List[float]:
        vals = [max(0.001, x) for x in (engaged, confused, frustrated, bored)]
        s = sum(vals)
        return [x / s for x in vals]

    def _initial_emotions_for_archetype(self, archetype: str) -> Tuple[float, float, float, float]:
        if archetype == "anxious":
            return 0.33 + self.rng.uniform(-0.05, 0.05), 0.26 + self.rng.uniform(-0.04, 0.04), 0.30 + self.rng.uniform(-0.04, 0.04), 0.11 + self.rng.uniform(-0.03, 0.03)
        if archetype == "boredom_prone":
            return 0.30 + self.rng.uniform(-0.05, 0.05), 0.14 + self.rng.uniform(-0.03, 0.03), 0.12 + self.rng.uniform(-0.03, 0.03), 0.44 + self.rng.uniform(-0.05, 0.05)
        if archetype == "struggling":
            return 0.28 + self.rng.uniform(-0.05, 0.05), 0.34 + self.rng.uniform(-0.05, 0.05), 0.26 + self.rng.uniform(-0.04, 0.04), 0.12 + self.rng.uniform(-0.03, 0.03)
        return 0.45 + self.rng.uniform(-0.08, 0.08), 0.20 + self.rng.uniform(-0.05, 0.05), 0.20 + self.rng.uniform(-0.05, 0.05), 0.15 + self.rng.uniform(-0.05, 0.05)

    def _apply_archetype_shift(self, engaged: float, confused: float, frustrated: float, bored: float) -> Tuple[float, float, float, float]:
        if self.archetype == "anxious":
            frustrated += 0.015; confused += 0.010; engaged -= 0.010
        elif self.archetype == "boredom_prone":
            bored += 0.020; engaged -= 0.015
        elif self.archetype == "struggling":
            confused += 0.018; frustrated += 0.012; engaged -= 0.015
        return engaged, confused, frustrated, bored

    def _adjust_correct_prob_for_archetype(self, correct_prob: float, action: str, prev: State) -> float:
        if self.archetype == "anxious":
            if action == "question_hard": correct_prob -= 0.10
            if action == "question_review": correct_prob += 0.06
        elif self.archetype == "boredom_prone":
            if action == "question_easy": correct_prob -= 0.06
            if action == "question_medium": correct_prob += 0.03
        elif self.archetype == "struggling":
            correct_prob -= 0.08
            if action == "question_review": correct_prob += 0.05
            if prev["confused"] > 0.30: correct_prob -= 0.05
        return correct_prob

    @staticmethod
    def discretize_emotion(state: State) -> str:
        top = max(EMOTIONS, key=lambda k: state[k])
        trend = "up" if state[f"delta_{top}"] >= 0 else "down"
        return f"{top}_{trend}"
