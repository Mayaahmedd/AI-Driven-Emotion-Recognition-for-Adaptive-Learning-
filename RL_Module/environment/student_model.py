"""
Synthetic student: personality parameters and state transition rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from RL_Module.mdp_definition import StudentState, TRANSITION_NOISE_STD, clip01


@dataclass
class SyntheticStudent:
    """One synthetic student with personality gamma_s, beta_s, lambda_s, rho_s."""

    gamma_s: float  # learning rate
    beta_s: float   # sensitivity to hard tasks
    lambda_s: float  # forgetting rate
    rho_s: float    # patience (patience_s in spec)
    student_id: int = 0
    student_type: str = "general"
    state: StudentState = field(
        default_factory=lambda: StudentState(0.3, 0.6, 0.1, 0.1, 0.1, 3)
    )

    def classify_type(self) -> str:
        if self.gamma_s > 0.7 and self.lambda_s < 0.1:
            return "fast_learner"
        if self.gamma_s < 0.3:
            return "slow_learner"
        if self.rho_s < 0.3 and self.beta_s > 0.7:
            return "easily_frustrated"
        if self.rho_s > 0.7:
            return "patient"
        return "general"

    def apply_action(self, action: int, prev: StudentState) -> StudentState:
        """Apply transition rules with Gaussian noise per dimension."""
        s = prev.copy()
        patience_s = self.rho_s

        def noise() -> float:
            return float(np.random.normal(0, TRANSITION_NOISE_STD))

        # Knowledge update
        delta_k_map = {
            0: 0.04 * self.gamma_s,
            1: 0.07 * self.gamma_s,
            2: 0.02 * self.gamma_s,
            3: 0.00,
            4: 0.06 * self.gamma_s,
            5: 0.00,
            6: 0.00,
            7: 0.01 * self.gamma_s,
            8: 0.00,
            9: 0.03 * self.gamma_s,
        }
        delta_k = delta_k_map.get(action, 0.0)
        decay = self.lambda_s * 0.01 if s.engagement < 0.3 else 0.0
        s.knowledge = clip01(s.knowledge + delta_k - decay + noise())

        # Frustration update
        reducing_actions = {2, 4, 5, 7, 8}
        if action in reducing_actions:
            delta_f = -0.05 * patience_s
        elif action == 1 and prev.emotion_id in (0, 2):
            delta_f = 0.10 * self.beta_s
        else:
            delta_f = 0.01 * (1.0 - patience_s)
        s.frustration = clip01(s.frustration + delta_f + noise())

        # Engagement update
        positive_actions = {3, 6, 8}
        if action in positive_actions:
            delta_e = 0.06
        elif action == 1 and s.frustration > 0.5:
            delta_e = -0.06
        else:
            delta_e = 0.01
        s.engagement = clip01(s.engagement + delta_e + noise())

        # Confusion update
        clearing_actions = {2, 7, 9}
        if action in clearing_actions:
            delta_c = -0.06
        else:
            delta_c = 0.01
        s.confusion = clip01(s.confusion + delta_c + noise())

        # Boredom update
        if s.engagement < 0.3:
            delta_b = 0.03
        elif action in (3, 8):
            delta_b = -0.06
        elif action == 0 and s.knowledge > 0.6:
            delta_b = 0.04
        else:
            delta_b = -0.01
        s.boredom = clip01(s.boredom + delta_b + noise())

        self.state = s
        return s

    def sample_initial_state(self, rng: np.random.Generator) -> StudentState:
        self.state = StudentState(
            knowledge=float(rng.uniform(0.1, 0.5)),
            engagement=float(rng.uniform(0.4, 0.8)),
            frustration=float(rng.uniform(0.0, 0.3)),
            confusion=float(rng.uniform(0.0, 0.3)),
            boredom=float(rng.uniform(0.0, 0.2)),
            emotion_id=3,
        )
        return self.state
