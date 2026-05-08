from __future__ import annotations

import random

from adaptation.step3.config import EXPLANATION_ACTIONS, PRACTICE_ACTIONS
from adaptation.step3.simulator import State


class BaselinePolicy:
    name = "baseline"

    def choose_action(self, phase: str, state: State) -> str:
        raise NotImplementedError


class RandomPolicy(BaselinePolicy):
    name = "random"

    def choose_action(self, phase: str, state: State) -> str:
        if phase == "explanation":
            return random.choice(EXPLANATION_ACTIONS)
        return random.choice(PRACTICE_ACTIONS)


class FixedPolicy(BaselinePolicy):
    name = "fixed"

    def choose_action(self, phase: str, state: State) -> str:
        if phase == "explanation":
            return "keep_pace"
        return "question_medium"


class HeuristicPolicy(BaselinePolicy):
    name = "heuristic"

    def choose_action(self, phase: str, state: State) -> str:
        if phase == "explanation":
            if state["frustrated"] > 0.32 or state["confused"] > 0.32:
                return "slow_down"
            if state["bored"] > 0.35:
                return "quick_check"
            if state["engaged"] < 0.30:
                return "motivate"
            return "give_micro_example"

        if state["frustrated"] > 0.30 or state["confused"] > 0.35:
            return "question_review"
        if state["bored"] > 0.35:
            return "question_hard"
        if state["engaged"] > 0.60:
            return "question_medium"
        return "question_easy"
