"""
Synthetic student: theory-grounded stochastic transitions P(s'|s,a).

- IRT correctness: P(correct) = sigmoid(beta * (k - alpha * d))
- Asymmetric mastery learning with saturation (1 - k)
- AR(1) emotion persistence with challenge-skill mismatch targets
- Pedagogical action effects as stochastic blends toward interaction targets

Coefficients are literature-inspired simulator hyperparameters (see config.SIMULATOR_PARAMS).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import numpy as np

from RL_Module import config
from RL_Module.mdp_definition import (
    ACTION_TO_ID,
    TRANSITION_NOISE_STD,
    clip01,
)
from RL_Module.mdp_definition import StudentState

_A = ACTION_TO_ID

MismatchDict = Dict[str, float]
TargetFn = Callable[[MismatchDict], float]


def _sim(name: str, default: float) -> float:
    """Runtime-tunable coefficient (sensitivity patches via config.SIMULATOR_PARAMS)."""
    return config.get_simulator_param(name, default)


def _const_target(value: float) -> TargetFn:
    def _fn(_mm: MismatchDict) -> float:
        return value

    return _fn


def _mismatch_key(key: str) -> TargetFn:
    def _fn(mm: MismatchDict) -> float:
        return mm[key]

    return _fn


# 8-action space; None / "mismatch" -> challenge-skill default for that dimension
ACTION_EFFECT_SPECS: Dict[int, Dict[str, Any]] = {
    _A["hint"]: {
        "base_gain": 0.05,
        "frustration": 0.25,
        "confusion": 0.15,  # lightweight cognitive support — moderate relief
        "boredom": None,
        "engagement": 0.65,
    },
    _A["scaffold"]: {
        "base_gain": 0.06,
        "frustration": 0.2,
        "confusion": 0.08,  # stronger guided support — deeper confusion reduction than hint
        "boredom": None,
        "engagement": 0.7,
    },
    _A["encouragement"]: {
        "base_gain": 0.0,
        "frustration": 0.15,
        "confusion": 0.25,  # slight secondary relief — weaker than hint; distinct from simplify_problem
        "boredom": 0.15,
        "engagement": 0.75,
    },
    _A["simplify_problem"]: {
        "base_gain": 0.05,
        "frustration": 0.2,
        "confusion": 0.2,
        "boredom": "mismatch",
        "engagement": None,
    },
    _A["harder_problem"]: {
        "base_gain": 0.06,
        "frustration": None,
        "confusion": "mismatch",
        "boredom": 0.1,
        "engagement": 0.7,
    },
    _A["break"]: {
        "base_gain": 0.0,
        "frustration": 0.10,  # strongest frustration reducer — pacing recovery niche
        "confusion": None,
        "boredom": None,
        "engagement": 0.65,
    },
    # Over-selection risk: strongest confusion reduction, engagement boost, and knowledge gain.
    # Cooldown-2 is the primary guard; monitor post-training via action_frequency_report().
    _A["explanation"]: {
        "base_gain": 0.06,  # thesis final: nearly equal gains
        "frustration": None,
        "confusion": 0.05,  # strongest confusion reducer — conceptual clarification
        "boredom": None,
        "engagement": 0.65,  # conceptual breakthrough restores interest
    },
    # Deliberate non-intervention: all None -> mismatch dynamics only.
    # Not reward-exploitable: bad states worsen affect (negative reward terms);
    # base_gain=0 limits knowledge growth under repeated use; flow-zone reward is intentional.
    _A["no_action"]: {
        "base_gain": 0.0,
        "frustration": None,
        "confusion": None,
        "boredom": None,
        "engagement": None,
    },
}


def _build_target_fn(spec_val: Any, mm_key: str) -> TargetFn:
    if spec_val == "mismatch":
        return _mismatch_key(mm_key)
    if spec_val is None:
        return _mismatch_key(mm_key)
    return _const_target(float(spec_val))


def build_action_effect_map() -> Dict[int, Dict[str, Any]]:
    """Compile ACTION_EFFECT_SPECS into callable target functions per action."""
    out: Dict[int, Dict[str, Any]] = {}
    for action_id, spec in ACTION_EFFECT_SPECS.items():
        out[action_id] = {
            "base_gain": float(spec["base_gain"]),
            "frustration": _build_target_fn(spec.get("frustration"), "frustration"),
            "confusion": _build_target_fn(spec.get("confusion"), "confusion"),
            "boredom": _build_target_fn(spec.get("boredom"), "boredom"),
            "engagement": _build_target_fn(spec.get("engagement"), "engagement"),
        }
    return out


ACTION_EFFECT_MAP: Dict[int, Dict[str, Any]] = build_action_effect_map()

# Pinned affect in strict_no_emotion mode (must not affect reward or transitions).
NEUTRAL_AFFECT: Dict[str, float] = {
    "engagement": 0.5,
    "frustration": 0.0,
    "confusion": 0.0,
    "boredom": 0.0,
}
NEUTRAL_EMOTION_ID: int = 3  # engaged (inert under strict dynamics)


@dataclass
class SyntheticStudent:
    """One synthetic student with cognitive-affective personality parameters."""

    gamma_s: float  # learning rate
    beta_s: float  # sensitivity to hard tasks (legacy; scales mismatch frustration)
    lambda_s: float  # forgetting rate
    rho_s: float  # patience
    frustration_tolerance: float = 0.5
    boredom_sensitivity: float = 0.5
    engagement_recovery: float = 0.5
    confidence: float = 0.0
    persistence: float = 0.5
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

    def _noise(self, std: Optional[float] = None) -> float:
        s = std if std is not None else TRANSITION_NOISE_STD
        return float(np.random.normal(0, s))

    def _correctness_prob(self, k: float, d: float) -> float:
        """P(correct) = sigma(beta * (k_eff - alpha * d)); IRT 1PL on [0,1] scale."""
        beta = _sim("IRT_BETA", 3.0)
        alpha = _sim("IRT_ALPHA", 1.0)
        k_eff = k + self.confidence * 0.1
        logit = beta * (k_eff - alpha * d)
        return float(1.0 / (1.0 + np.exp(-logit)))

    def _knowledge_update(
        self, k: float, correct: bool, base_gain: float, engagement: float
    ) -> float:
        """k' = k + gain * (1-k) with asymmetric correct/incorrect factors."""
        gain_correct = _sim("GAIN_CORRECT_FACTOR", 0.1)
        gain_incorrect = _sim("GAIN_INCORRECT_FACTOR", 1.0)
        factor = gain_correct if correct else gain_incorrect
        gain = base_gain * self.gamma_s * factor
        decay = (
            self.lambda_s * 0.01 * (1.0 - self.persistence) if not correct else 0.0
        )
        if engagement < 0.3:
            decay += self.lambda_s * 0.01
        return clip01(k + gain * (1.0 - k) - decay + self._noise())

    def _mismatch_effects(self, k: float, d: float) -> MismatchDict:
        """
        Challenge-skill mismatch m = d - k drives interaction targets.
        Flow zone: moderate affect; too hard -> frustration/confusion; too easy -> boredom.
        """
        mismatch = d - k
        high = _sim("MISMATCH_HIGH", 0.3)
        low = _sim("MISMATCH_LOW", -0.3)
        frust_scale = (1.0 - self.frustration_tolerance) * self.beta_s + 0.3
        bored_scale = self.boredom_sensitivity

        if mismatch > high:
            frust_target = clip01(0.7 * frust_scale)
            conf_target = 0.6
            bored_target = 0.1
            eng_target = clip01(0.55 * self.engagement_recovery)
        elif mismatch < low:
            frust_target = 0.1
            conf_target = 0.1
            bored_target = clip01(0.7 * bored_scale)
            eng_target = clip01(0.5 + 0.2 * self.engagement_recovery)
        else:
            frust_target = 0.2
            conf_target = 0.2
            bored_target = 0.15
            eng_target = clip01(0.65 + 0.15 * self.engagement_recovery)

        return {
            "frustration": frust_target,
            "confusion": conf_target,
            "boredom": bored_target,
            "engagement": eng_target,
        }

    def _emotion_update(
        self, current: float, target: float, lambda_coef: float
    ) -> float:
        """e_{t+1} = lambda * e_t + (1-lambda) * target + noise."""
        noise_std = _sim("EMOTION_NOISE_STD", 0.03)
        return clip01(
            lambda_coef * current
            + (1.0 - lambda_coef) * target
            + np.random.normal(0, noise_std)
        )

    def _scale_engagement_target(self, target: float) -> float:
        """Personality modulates how strongly interventions restore engagement."""
        return clip01(target * (0.6 + 0.4 * self.engagement_recovery))

    def apply_action(
        self,
        action: int,
        prev: StudentState,
        difficulty: float,
        strict_no_emotion: bool = False,
    ) -> tuple[StudentState, bool]:
        """
        Stochastic transition: sample correctness, update mastery and emotions.
        Returns (new_state, correct).

        strict_no_emotion: knowledge-only MDP — affect pinned to NEUTRAL_AFFECT and
        does not evolve; engagement does not modulate knowledge decay.
        """
        if strict_no_emotion:
            return self._apply_action_strict(action, prev, difficulty)

        s = prev.copy()
        correct = bool(np.random.random() < self._correctness_prob(prev.knowledge, difficulty))

        effects = ACTION_EFFECT_MAP.get(action, ACTION_EFFECT_MAP[_A["scaffold"]])
        base_gain = float(effects["base_gain"])

        s.knowledge = self._knowledge_update(
            prev.knowledge, correct, base_gain, prev.engagement
        )

        mm = self._mismatch_effects(s.knowledge, difficulty)

        lam_f = _sim("LAMBDA_FRUSTRATION", 0.69)
        lam_e = _sim("LAMBDA_ENGAGEMENT", 0.60)
        lam_c = _sim("LAMBDA_CONFUSION", 0.47)
        lam_b = _sim("LAMBDA_BOREDOM", 0.36)

        s.frustration = self._emotion_update(
            prev.frustration, effects["frustration"](mm), lam_f
        )
        s.confusion = self._emotion_update(
            prev.confusion, effects["confusion"](mm), lam_c
        )
        s.boredom = self._emotion_update(
            prev.boredom, effects["boredom"](mm), lam_b
        )
        eng_target = self._scale_engagement_target(effects["engagement"](mm))
        s.engagement = self._emotion_update(prev.engagement, eng_target, lam_e)

        self.state = s
        return s, correct

    def _apply_action_strict(
        self, action: int, prev: StudentState, difficulty: float
    ) -> tuple[StudentState, bool]:
        """Knowledge-only transition: IRT correctness + base_gain; affect inert."""
        s = prev.copy()
        correct = bool(
            np.random.random() < self._correctness_prob(prev.knowledge, difficulty)
        )
        effects = ACTION_EFFECT_MAP.get(action, ACTION_EFFECT_MAP[_A["scaffold"]])
        base_gain = float(effects["base_gain"])

        gain_correct = _sim("GAIN_CORRECT_FACTOR", 0.1)
        gain_incorrect = _sim("GAIN_INCORRECT_FACTOR", 1.0)
        factor = gain_correct if correct else gain_incorrect
        gain = base_gain * self.gamma_s * factor
        decay = (
            self.lambda_s * 0.01 * (1.0 - self.persistence) if not correct else 0.0
        )
        s.knowledge = clip01(
            prev.knowledge + gain * (1.0 - prev.knowledge) - decay + self._noise()
        )
        s.engagement = NEUTRAL_AFFECT["engagement"]
        s.frustration = NEUTRAL_AFFECT["frustration"]
        s.confusion = NEUTRAL_AFFECT["confusion"]
        s.boredom = NEUTRAL_AFFECT["boredom"]
        s.emotion_id = NEUTRAL_EMOTION_ID
        self.state = s
        return s, correct

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
