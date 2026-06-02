"""
Expert Rule Tutor (ERT) — theory-grounded handcrafted baseline.

Priority hierarchy (T0 → T6 → fallback). Confusion-led rules precede
frustration-led rules. Thresholds calibrated to synthetic student dynamics.

Action IDs (must match StudentEnv):
  0 hint, 1 scaffold, 2 encouragement, 3 simplify_problem,
  4 harder_problem, 5 break, 6 explanation, 7 no_action
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

from RL_Module.agents.base_agent import BaseAgent
from RL_Module.mdp_definition import (
    ACTION_DIM,
    ACTION_TO_ID,
    EMOTION_TO_ID,
    StudentState,
)

# ── Thresholds (fixed — not tuned vs DQN) ───────────────────────────────────
FRUSTRATED_BREAK = 0.27
HIGH_FRUSTRATION = 0.265
MODERATE_FRUSTRATION = 0.22

HIGH_CONFUSION = 0.32
MODERATE_CONFUSION = 0.22

HIGH_BOREDOM = 0.17
BOREDOM_DOMINANCE_MARGIN = 0.03

LOW_KNOWLEDGE = 0.35
HIGH_KNOWLEDGE = 0.75

LOW_ENGAGEMENT = 0.45
HIGH_ENGAGEMENT = 0.58
FLOW_AFFECT_CEILING = 0.22


@dataclass(frozen=True)
class RuleSpec:
    """Single rule metadata for tracing and tests."""

    rule_id: str
    tier: str
    description: str
    educational_rationale: str
    emotional_rationale: str
    expected_learning: str
    expected_affective: str
    theory: str


RULE_REGISTRY: Tuple[RuleSpec, ...] = (
    RuleSpec(
        "T0_break",
        "T0",
        "Acute frustration → break",
        "Stabilize session before instruction; pacing recovery prevents dropout.",
        "Sustained or acute frustration blocks working memory (Pekrun CVT).",
        "Preserves remaining episode time for learning after recovery.",
        "Sharp reduction in frustration; restores engagement baseline.",
        "D'Mello & Graesser (2012); Baker et al. (2008)",
    ),
    RuleSpec(
        "T1_high_confusion_explain",
        "T1",
        "High confusion → explanation",
        "Conceptual impasse requires direct instruction, not hints alone.",
        "Deep confusion creates extraneous cognitive load.",
        "Strong confusion reduction; moderate Δk.",
        "↓ confusion; ↑ engagement on breakthrough.",
        "VanLehn (2011); Chi et al. (1994)",
    ),
    RuleSpec(
        "T1_mod_confusion_scaffold",
        "T1",
        "Moderate confusion → scaffold",
        "Structured step-by-step support within ZPD.",
        "Moderate confusion predicts impending frustration.",
        "Highest support-action Δk among confusion interventions.",
        "↓ confusion, ↓ frustration.",
        "Wood et al. (1976) Scaffolding",
    ),
    RuleSpec(
        "T2_high_frustration_simplify",
        "T2",
        "High frustration → simplify_problem",
        "Reduce task demand to re-enter Zone of Proximal Development.",
        "High frustration signals loss of control and competence.",
        "Increases p(correct); steady Δk via simplify base_gain.",
        "↓ frustration, ↓ confusion.",
        "Vygotsky (1978); Sweller (1988) Cognitive Load Theory",
    ),
    RuleSpec(
        "T2_mod_frustration_encourage",
        "T2",
        "Moderate frustration → encouragement",
        "Attribution support before escalating cognitive intervention.",
        "Mild frustration is recoverable without full demand reduction.",
        "Maintains attempt cycle; small or zero direct Δk.",
        "↓ frustration trajectory; ↑ engagement.",
        "Pekrun (2006); Graesser et al. (2008) AutoTutor",
    ),
    RuleSpec(
        "T3_boredom_harder",
        "T3",
        "Boredom-led challenge → harder_problem",
        "Increase challenge when skill exceeds task demand (flow maintenance).",
        "Boredom = under-stimulation and low task value.",
        "↑ difficulty; high base_gain action.",
        "↓ boredom, ↑ engagement.",
        "Csikszentmihalyi (1990); Bloom (1968)",
    ),
    RuleSpec(
        "T4_low_knowledge_scaffold",
        "T4",
        "Low knowledge → scaffold",
        "Entry-level learners need maximal structured support.",
        "Low k correlates with confusion and frustration risk.",
        "Rapid early Δk progression.",
        "↓ confusion, ↑ engagement.",
        "Bloom (1968) Mastery Learning",
    ),
    RuleSpec(
        "T4_high_knowledge_harder",
        "T4",
        "High knowledge → harder_problem",
        "Mastery achieved; extend to transfer-level problems.",
        "Positive affect supports challenge uptake.",
        "Push toward episode mastery termination (k ≥ 0.95).",
        "Sustained engagement via optimal challenge.",
        "Bloom (1968); Kalyuga (2007) expertise reversal",
    ),
    RuleSpec(
        "T5_flow_no_action",
        "T5",
        "Narrow flow zone → no_action",
        "Do not interrupt successful learning (minimal guidance principle).",
        "All affects in optimal band — intervention is unnecessary.",
        "Learner-driven Δk via attempts and productive failure.",
        "Sustained positive affect and autonomy.",
        "Csikszentmihalyi (1990); Clark (1989)",
    ),
    RuleSpec(
        "T6_low_engagement_encourage",
        "T6",
        "Low engagement → encouragement",
        "Re-engage before content push when attention is fading.",
        "Disengagement predicts dropout and shallow processing.",
        "Prerequisite for germane load processing.",
        "↑ engagement.",
        "Fredrickson (2001); ITS engagement literature",
    ),
    RuleSpec(
        "fallback_hint",
        "Fallback",
        "Default → hint",
        "Standard ZPD support for unmatched mid-mastery states.",
        "Preventive support before affective escalation.",
        "Steady Δk progression.",
        "Stable engagement.",
        "Vygotsky (1978)",
    ),
)


def dominant_affect(s: StudentState) -> str:
    """Return the highest among frustration, confusion, boredom."""
    scores = {
        "frustration": s.frustration,
        "confusion": s.confusion,
        "boredom": s.boredom,
    }
    return max(scores, key=scores.get)


def _affect_margin(s: StudentState, key: str) -> float:
    """Gap between the named affect and the next-highest."""
    scores = {
        "frustration": s.frustration,
        "confusion": s.confusion,
        "boredom": s.boredom,
    }
    ranked = sorted(scores.values(), reverse=True)
    if scores[key] < ranked[0]:
        return 0.0
    return ranked[0] - ranked[1]


def _boredom_triggers_challenge(s: StudentState) -> bool:
    return s.boredom >= HIGH_BOREDOM or (
        dominant_affect(s) == "boredom"
        and _affect_margin(s, "boredom") >= BOREDOM_DOMINANCE_MARGIN
    )


def flow_zone(s: StudentState) -> bool:
    """Narrow flow: high engagement with all negative affects below ceiling."""
    return (
        s.engagement >= HIGH_ENGAGEMENT
        and s.frustration < FLOW_AFFECT_CEILING
        and s.confusion < FLOW_AFFECT_CEILING
        and s.boredom < FLOW_AFFECT_CEILING
    )


def _first_valid(candidates: List[int], action_mask: np.ndarray) -> Optional[int]:
    for action in candidates:
        if action_mask[action] == 1:
            return action
    return None


# ERT rule tiers disabled under each observation ablation (knowledge_only = affect-blind ERT)
ERT_ABLATION_DISABLED_RULES: Dict[str, frozenset] = {
    "full_emotion": frozenset(),
    "knowledge_only": frozenset({
        "T0_break",
        "T1_high_confusion_explain",
        "T1_mod_confusion_scaffold",
        "T2_high_frustration_simplify",
        "T2_mod_frustration_encourage",
        "T3_boredom_harder",
        "T5_flow_no_action",
        "T6_low_engagement_encourage",
    }),
    "no_engagement": frozenset({"T5_flow_no_action", "T6_low_engagement_encourage"}),
    "no_frustration": frozenset({
        "T0_break",
        "T2_high_frustration_simplify",
        "T2_mod_frustration_encourage",
    }),
    "no_confusion": frozenset({
        "T1_high_confusion_explain",
        "T1_mod_confusion_scaffold",
    }),
    "no_boredom": frozenset({"T3_boredom_harder"}),
    "no_emotion_id": frozenset(),
    "emotion_id_only": frozenset({
        "T1_high_confusion_explain",
        "T1_mod_confusion_scaffold",
        "T2_high_frustration_simplify",
        "T2_mod_frustration_encourage",
        "T3_boredom_harder",
        "T5_flow_no_action",
        "T6_low_engagement_encourage",
    }),
}


def _rule_candidates(
    s: StudentState, persistent_flag: bool, obs_ablation: str = "full_emotion"
) -> List[Tuple[str, List[int]]]:
    """
    Return ordered (rule_id, candidate_actions) pairs.
    T1 (confusion) precedes T2 (frustration) so explanation is not
    pre-empted by simplify when confusion is elevated.
    """
    disabled = ERT_ABLATION_DISABLED_RULES.get(obs_ablation, frozenset())
    rules: List[Tuple[str, List[int]]] = []

    # T0 — Acute frustration
    if "T0_break" not in disabled and (
        persistent_flag
        or s.frustration >= FRUSTRATED_BREAK
        or (
            obs_ablation != "no_emotion_id"
            and s.emotion_id == EMOTION_TO_ID["frustrated"]
        )
    ):
        rules.append(("T0_break", [ACTION_TO_ID["break"]]))

    # T1 — Confusion-led support (priority over frustration)
    if "T1_high_confusion_explain" not in disabled and s.confusion >= HIGH_CONFUSION:
        rules.append(
            (
                "T1_high_confusion_explain",
                [
                    ACTION_TO_ID["explanation"],
                    ACTION_TO_ID["scaffold"],
                    ACTION_TO_ID["hint"],
                ],
            )
        )
    elif "T1_mod_confusion_scaffold" not in disabled and s.confusion >= MODERATE_CONFUSION:
        rules.append(
            (
                "T1_mod_confusion_scaffold",
                [
                    ACTION_TO_ID["scaffold"],
                    ACTION_TO_ID["explanation"],
                    ACTION_TO_ID["hint"],
                ],
            )
        )

    # T2 — Frustration-led support
    if "T2_high_frustration_simplify" not in disabled and s.frustration >= HIGH_FRUSTRATION:
        rules.append(
            (
                "T2_high_frustration_simplify",
                [
                    ACTION_TO_ID["simplify_problem"],
                    ACTION_TO_ID["encouragement"],
                ],
            )
        )
    elif "T2_mod_frustration_encourage" not in disabled and s.frustration >= MODERATE_FRUSTRATION:
        rules.append(
            (
                "T2_mod_frustration_encourage",
                [ACTION_TO_ID["encouragement"], ACTION_TO_ID["hint"]],
            )
        )

    # T3 — Boredom-led challenge
    if "T3_boredom_harder" not in disabled and _boredom_triggers_challenge(s):
        rules.append(
            (
                "T3_boredom_harder",
                [
                    ACTION_TO_ID["harder_problem"],
                    ACTION_TO_ID["scaffold"],
                    ACTION_TO_ID["encouragement"],
                ],
            )
        )

    # T4 — Knowledge progression
    if s.knowledge < LOW_KNOWLEDGE:
        rules.append(
            (
                "T4_low_knowledge_scaffold",
                [
                    ACTION_TO_ID["scaffold"],
                    ACTION_TO_ID["hint"],
                    ACTION_TO_ID["explanation"],
                ],
            )
        )
    elif s.knowledge > HIGH_KNOWLEDGE:
        rules.append(
            (
                "T4_high_knowledge_harder",
                [
                    ACTION_TO_ID["harder_problem"],
                    ACTION_TO_ID["scaffold"],
                    ACTION_TO_ID["hint"],
                ],
            )
        )

    # T5 — Narrow flow zone
    if "T5_flow_no_action" not in disabled and flow_zone(s):
        rules.append(("T5_flow_no_action", [ACTION_TO_ID["no_action"]]))

    # T6 — Engagement recovery
    if "T6_low_engagement_encourage" not in disabled and s.engagement < LOW_ENGAGEMENT:
        rules.append(
            (
                "T6_low_engagement_encourage",
                [ACTION_TO_ID["encouragement"], ACTION_TO_ID["hint"]],
            )
        )

    rules.append(
        ("fallback_hint", [ACTION_TO_ID["hint"], ACTION_TO_ID["scaffold"]])
    )
    return rules


class ExpertRuleBasedAgent(BaseAgent):
    """Expert Rule Tutor (ERT) — mask-respecting tiered policy."""

    name = "ERT"
    uses_persistent_flag = True

    def __init__(self, obs_ablation: str = "full_emotion") -> None:
        self.obs_ablation = obs_ablation
        self.last_rule_id: Optional[str] = None

    def train(self, env: gym.Env, total_timesteps: int, seed: int) -> None:
        pass

    def needs_training(self) -> bool:
        return False

    def resolve_rule(
        self, s: StudentState, persistent_flag: bool, action_mask: np.ndarray
    ) -> Tuple[str, List[int], Optional[int]]:
        """Expose decision logic for tests: (rule_id, candidates, chosen_action)."""
        for rule_id, candidates in _rule_candidates(
            s, persistent_flag, self.obs_ablation
        ):
            action = _first_valid(candidates, action_mask)
            if action is not None:
                return rule_id, candidates, action
        allowed = np.where(action_mask == 1)[0]
        fallback = int(allowed[0]) if len(allowed) else ACTION_TO_ID["no_action"]
        return "fallback_hint", [ACTION_TO_ID["hint"]], fallback

    def predict(
        self,
        obs: np.ndarray,
        action_mask: Optional[np.ndarray] = None,
        persistent_flag: bool = False,
    ) -> int:
        s = StudentState.from_vec(obs)

        if action_mask is None:
            action_mask = np.ones(ACTION_DIM, dtype=np.int8)

        self.last_rule_id = None
        for rule_id, candidates in _rule_candidates(
            s, persistent_flag, self.obs_ablation
        ):
            action = _first_valid(candidates, action_mask)
            if action is not None:
                self.last_rule_id = rule_id
                return action

        allowed = np.where(action_mask == 1)[0]
        if len(allowed) == 0:
            self.last_rule_id = "fallback_hint"
            return ACTION_TO_ID["no_action"]
        self.last_rule_id = "fallback_hint"
        return int(allowed[0])

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass


# Backward-compatible alias used across the codebase
RuleBasedAgent = ExpertRuleBasedAgent
