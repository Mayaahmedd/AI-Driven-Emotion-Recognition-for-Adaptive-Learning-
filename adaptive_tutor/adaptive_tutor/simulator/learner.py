"""Synthetic latent learner: emotions plus simple tutor-action effects.

This is **not** a cognitive model; it is a deterministic finite-state
emotion prior that makes the environment respond differently to tutor
actions. The observed ``LearnerState`` that RL trains on still comes
from :class:`~adaptive_tutor.state.builder.StateBuilder`; this class
only feeds believable :class:`~adaptive_tutor.core.types.EmotionVector`
readings into ``observe`` and shapes the Bernoulli success rate.

Keeping the latent vector 4-dimensional matches the real FER
interface, so swapping the simulator for a live learner later does not
change tensor shapes.
"""

from __future__ import annotations

from dataclasses import dataclass

from adaptive_tutor.core.types import EmotionVector
from adaptive_tutor.simulator.calibration import LearnerParams


@dataclass(slots=True)
class SyntheticLearner:
    """Mutable emotional + persistence state for one synthetic student."""

    engaged: float
    confused: float
    bored: float
    frustrated: float
    learning_rate: float
    persistence: float

    @classmethod
    def from_params(cls, p: LearnerParams) -> SyntheticLearner:
        return cls(
            engaged=p.engaged_bias,
            confused=p.confused_bias,
            bored=p.bored_bias,
            frustrated=p.frustrated_bias,
            learning_rate=p.learning_rate,
            persistence=p.persistence,
        )

    def apply_action_effect(self, action: str) -> None:
        """Deterministic tutor-effect on latent emotions *before* outcome."""
        if action == "encouragement":
            self.frustrated = max(0.0, self.frustrated - 0.10)
            self.engaged = min(1.0, self.engaged + 0.06)
        elif action == "give_hint":
            self.confused = max(0.0, self.confused - 0.08)
            self.bored = min(1.0, self.bored + 0.03)
        elif action == "easier_problem":
            self.frustrated = max(0.0, self.frustrated - 0.06)
            self.bored = min(1.0, self.bored + 0.04)
        elif action == "harder_problem":
            self.confused = min(1.0, self.confused + 0.07)
            self.engaged = min(1.0, self.engaged + 0.02)
        elif action == "slow_pacing":
            self.bored = max(0.0, self.bored - 0.06)
        elif action == "retry_current_skill":
            self.frustrated = min(1.0, self.frustrated + 0.02)
        elif action == "advance_to_next_skill":
            self.engaged = min(1.0, self.engaged + 0.03)
        self._clamp()

    def feedback_after_outcome(self, correct: bool) -> None:
        """Nudge emotions after correctness is revealed (post-step)."""
        if correct:
            self.engaged = min(1.0, self.engaged + 0.04 * self.persistence)
            self.confused = max(0.0, self.confused - 0.06 * self.persistence)
            self.frustrated = max(0.0, self.frustrated - 0.03)
        else:
            self.frustrated = min(1.0, self.frustrated + 0.06 * (2.0 - self.persistence))
            self.confused = min(1.0, self.confused + 0.05)
            self.bored = min(1.0, self.bored + 0.03)
        self._clamp()

    def success_probability(self, mastery: float, action: str) -> float:
        """Bernoulli success probability before sampling ``correct``."""
        m = max(0.0, min(1.0, float(mastery)))
        boost = 0.0
        if action == "easier_problem":
            boost += 0.16
        elif action == "harder_problem":
            boost -= 0.12
        elif action == "give_hint":
            boost += 0.22
        elif action == "encouragement":
            boost += 0.04

        # Baseline rises with mastery; frustration suppresses, engagement helps.
        base = 0.12 + 0.78 * m
        base *= (1.0 - 0.45 * self.frustrated) * (0.45 + 0.55 * self.engaged)
        p = base + boost
        return max(0.05, min(0.95, p))

    def to_emotion_vector(self) -> EmotionVector:
        return EmotionVector(
            engaged=self.engaged,
            confused=self.confused,
            bored=self.bored,
            frustrated=self.frustrated,
        )

    def _clamp(self) -> None:
        self.engaged = max(0.0, min(1.0, self.engaged))
        self.confused = max(0.0, min(1.0, self.confused))
        self.bored = max(0.0, min(1.0, self.bored))
        self.frustrated = max(0.0, min(1.0, self.frustrated))
