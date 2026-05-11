"""Centralised scalar reward for tutoring steps.

Why a separate module
---------------------
The bachelor-thesis formula must match everywhere: ASSISTments offline
``iter_transitions``, synthetic :class:`~adaptive_tutor.simulator.environment.TutoringEnvironment`
rollouts, and future DQN/PPO training. A single :class:`RewardEngine`
avoid scattering coefficients across adapters.

Design
------
* Coefficients default to the documented ASSISTments-style spec.
* :meth:`RewardEngine.compute` returns scalar + named components for
  :class:`~adaptive_tutor.state.state.Transition.r_components`.
* Optional ``state`` / ``action`` parameters are accepted for future
  shaping; they are ignored in v1 to keep behaviour identical to the
  legacy formula.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from adaptive_tutor.state.state import LearnerState


@dataclass(frozen=True, slots=True)
class RewardEngine:
    """Interpretable reward: correctness minus frustration and hints plus engagement."""

    coef_correct: float = 1.0
    coef_frustrated: float = -0.5
    coef_hint_scale: float = -0.2
    coef_engaged: float = 0.3
    max_hint_for_ratio: int = 3

    def hint_ratio(self, hint_count: int) -> float:
        m = max(1, self.max_hint_for_ratio)
        return min(int(hint_count), m) / float(m)

    def components(
        self,
        *,
        correct: int,
        hint_count: int,
        emotion: Mapping[str, float],
    ) -> tuple[tuple[str, float], ...]:
        hint_r = self.hint_ratio(hint_count)
        perf = self.coef_correct * float(correct)
        fr = self.coef_frustrated * float(emotion.get("frustrated", 0.0))
        hi = self.coef_hint_scale * hint_r
        eng = self.coef_engaged * float(emotion.get("engaged", 0.0))
        return (
            ("correctness", perf),
            ("frustration", fr),
            ("hint_penalty", hi),
            ("engagement", eng),
        )

    def compute_scalar(
        self,
        *,
        correct: int,
        hint_count: int,
        emotion: Mapping[str, float],
    ) -> float:
        return sum(v for _, v in self.components(correct=correct, hint_count=hint_count, emotion=emotion))

    def compute(
        self,
        *,
        correct: int,
        hint_count: int,
        emotion: Mapping[str, float],
        state: LearnerState | None = None,
        action: str | None = None,
    ) -> tuple[float, tuple[tuple[str, float], ...]]:
        """Return ``(scalar_reward, r_components)``.

        ``state`` and ``action`` are reserved for future shaping hooks;
        they do not affect the returned values in v1.
        """
        _ = (state, action)
        comp = self.components(correct=correct, hint_count=hint_count, emotion=emotion)
        return sum(v for _, v in comp), comp


_DEFAULT = RewardEngine()


def default_reward_engine() -> RewardEngine:
    """Process-wide default (bachelor-thesis coefficients)."""
    return _DEFAULT


def assistments_reward(
    correct: int,
    hint_count: int,
    emotion: Mapping[str, float],
) -> float:
    """Backward-compatible name: same scalar as :meth:`RewardEngine.compute_scalar`."""
    return _DEFAULT.compute_scalar(correct=correct, hint_count=hint_count, emotion=emotion)
