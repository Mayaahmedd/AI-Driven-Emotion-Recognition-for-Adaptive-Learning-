"""Unit tests and validation for Expert Rule Tutor (ERT)."""

from __future__ import annotations

import numpy as np
import pytest

from RL_Module.agents.rule_based import (
    ExpertRuleBasedAgent,
    RULE_REGISTRY,
    dominant_affect,
    flow_zone,
)
from RL_Module.environment.student_env import StudentEnv
from RL_Module.mdp_definition import ACTION_TO_ID, EMOTION_TO_ID, ID_TO_ACTION, StudentState


def _state(**kwargs) -> StudentState:
    defaults = dict(
        knowledge=0.5,
        engagement=0.6,
        frustration=0.1,
        confusion=0.1,
        boredom=0.1,
        emotion_id=EMOTION_TO_ID["engaged"],
    )
    defaults.update(kwargs)
    return StudentState(**defaults)


def _full_mask() -> np.ndarray:
    return np.ones(8, dtype=np.int8)


class TestFlowZone:
    def test_flow_true(self):
        s = _state(engagement=0.7, frustration=0.15, confusion=0.15, boredom=0.15)
        assert flow_zone(s)

    def test_flow_false_high_frustration(self):
        s = _state(engagement=0.7, frustration=0.30, confusion=0.15, boredom=0.15)
        assert not flow_zone(s)

    def test_flow_requires_high_engagement(self):
        s = _state(engagement=0.50, frustration=0.15, confusion=0.15, boredom=0.15)
        assert not flow_zone(s)


class TestDominantAffect:
    def test_confusion_dominant(self):
        s = _state(frustration=0.20, confusion=0.35, boredom=0.10)
        assert dominant_affect(s) == "confusion"


class TestTierPriority:
    def setup_method(self):
        self.agent = ExpertRuleBasedAgent()

    def test_t0_persistent_break(self):
        s = _state(frustration=0.9)
        rule_id, _, action = self.agent.resolve_rule(
            s, persistent_flag=True, action_mask=_full_mask()
        )
        assert rule_id == "T0_break"
        assert action == ACTION_TO_ID["break"]

    def test_t0_frustrated_emotion_break(self):
        s = _state(frustration=0.34, emotion_id=EMOTION_TO_ID["frustrated"])
        rule_id, _, action = self.agent.resolve_rule(s, False, _full_mask())
        assert rule_id == "T0_break"
        assert action == ACTION_TO_ID["break"]

    def test_t0_frustration_threshold_break(self):
        s = _state(frustration=0.28, confusion=0.10)
        rule_id, _, action = self.agent.resolve_rule(s, False, _full_mask())
        assert rule_id == "T0_break"
        assert action == ACTION_TO_ID["break"]

    def test_t2_high_frustration_simplify(self):
        s = _state(frustration=0.266, confusion=0.10)
        rule_id, _, action = self.agent.resolve_rule(s, False, _full_mask())
        assert rule_id == "T2_high_frustration_simplify"
        assert action == ACTION_TO_ID["simplify_problem"]

    def test_t2_moderate_frustration_encouragement(self):
        s = _state(frustration=0.25, confusion=0.10, boredom=0.24)
        rule_id, _, action = self.agent.resolve_rule(s, False, _full_mask())
        assert rule_id == "T2_mod_frustration_encourage"
        assert action == ACTION_TO_ID["encouragement"]

    def test_t1_high_confusion_explanation(self):
        s = _state(confusion=0.35, frustration=0.10)
        rule_id, _, action = self.agent.resolve_rule(s, False, _full_mask())
        assert rule_id == "T1_high_confusion_explain"
        assert action == ACTION_TO_ID["explanation"]

    def test_t1_moderate_confusion_scaffold(self):
        s = _state(confusion=0.25, knowledge=0.60)
        rule_id, _, action = self.agent.resolve_rule(s, False, _full_mask())
        assert rule_id == "T1_mod_confusion_scaffold"
        assert action == ACTION_TO_ID["scaffold"]

    def test_t3_boredom_harder(self):
        s = _state(boredom=0.18, knowledge=0.55, frustration=0.10, confusion=0.10)
        rule_id, _, action = self.agent.resolve_rule(s, False, _full_mask())
        assert rule_id == "T3_boredom_harder"
        assert action == ACTION_TO_ID["harder_problem"]

    def test_t3_boredom_dominant_harder(self):
        s = _state(boredom=0.16, frustration=0.12, confusion=0.12)
        rule_id, _, action = self.agent.resolve_rule(s, False, _full_mask())
        assert rule_id == "T3_boredom_harder"
        assert action == ACTION_TO_ID["harder_problem"]

    def test_t4_low_knowledge_scaffold(self):
        s = _state(knowledge=0.30, confusion=0.1, boredom=0.1, frustration=0.1)
        rule_id, _, action = self.agent.resolve_rule(s, False, _full_mask())
        assert rule_id == "T4_low_knowledge_scaffold"
        assert action == ACTION_TO_ID["scaffold"]

    def test_t4_high_knowledge_harder(self):
        s = _state(knowledge=0.80, frustration=0.15, confusion=0.15, boredom=0.15)
        rule_id, _, action = self.agent.resolve_rule(s, False, _full_mask())
        assert rule_id == "T4_high_knowledge_harder"
        assert action == ACTION_TO_ID["harder_problem"]

    def test_t5_flow_no_action(self):
        s = _state(
            knowledge=0.60,
            engagement=0.70,
            frustration=0.15,
            confusion=0.15,
            boredom=0.15,
        )
        rule_id, _, action = self.agent.resolve_rule(s, False, _full_mask())
        assert rule_id == "T5_flow_no_action"
        assert action == ACTION_TO_ID["no_action"]

    def test_t6_low_engagement(self):
        s = _state(engagement=0.40, frustration=0.1, confusion=0.1, boredom=0.1)
        rule_id, _, action = self.agent.resolve_rule(s, False, _full_mask())
        assert rule_id == "T6_low_engagement_encourage"
        assert action == ACTION_TO_ID["encouragement"]

    def test_fallback_hint(self):
        s = _state(
            knowledge=0.50,
            engagement=0.50,
            frustration=0.1,
            confusion=0.1,
            boredom=0.1,
        )
        rule_id, _, action = self.agent.resolve_rule(s, False, _full_mask())
        assert rule_id == "fallback_hint"
        assert action == ACTION_TO_ID["hint"]


class TestMaskFallthrough:
    def test_persistent_falls_through_to_simplify_when_break_on_cooldown(self):
        agent = ExpertRuleBasedAgent()
        s = _state(frustration=0.9)
        mask = np.zeros(8, dtype=np.int8)
        mask[ACTION_TO_ID["simplify_problem"]] = 1
        mask[ACTION_TO_ID["encouragement"]] = 1
        rule_id, _, action = agent.resolve_rule(
            s, persistent_flag=True, action_mask=mask
        )
        assert rule_id == "T2_high_frustration_simplify"
        assert action == ACTION_TO_ID["simplify_problem"]

    def test_harder_blocked_falls_to_scaffold_on_boredom(self):
        agent = ExpertRuleBasedAgent()
        s = _state(boredom=0.18, knowledge=0.60, frustration=0.1, confusion=0.1)
        mask = _full_mask()
        mask[ACTION_TO_ID["harder_problem"]] = 0
        rule_id, _, action = agent.resolve_rule(s, False, mask)
        assert rule_id == "T3_boredom_harder"
        assert action == ACTION_TO_ID["scaffold"]


class TestActionReachability:
    """Every action must be selectable under some state + mask."""

    def test_all_actions_reachable(self):
        agent = ExpertRuleBasedAgent()
        scenarios = [
            _state(frustration=0.9),
            _state(frustration=0.266, confusion=0.10),
            _state(frustration=0.25, confusion=0.10, boredom=0.24),
            _state(confusion=0.35, frustration=0.10),
            _state(confusion=0.25, knowledge=0.40, frustration=0.10, boredom=0.10),
            _state(
                knowledge=0.50,
                engagement=0.50,
                frustration=0.10,
                confusion=0.10,
                boredom=0.10,
            ),
            _state(boredom=0.18, knowledge=0.55, frustration=0.10, confusion=0.10),
            _state(
                knowledge=0.60,
                engagement=0.70,
                frustration=0.15,
                confusion=0.15,
                boredom=0.15,
            ),
        ]
        persistent_flags = [True, False, False, False, False, False, False, False]
        reached = set()
        for s, pf in zip(scenarios, persistent_flags):
            obs = s.as_vec()
            action = agent.predict(obs, _full_mask(), persistent_flag=pf)
            reached.add(action)
        assert reached == set(range(8)), f"Missing actions: {set(range(8)) - reached}"


class TestActionDistributionTargets:
    """Pedagogical actions used across seeds; no_action capped."""

    def test_aggregate_action_frequencies(self):
        seeds = [13, 21, 42, 99, 314]
        agent = ExpertRuleBasedAgent()
        totals = {a: 0 for a in ID_TO_ACTION.values()}
        total_steps = 0
        for seed in seeds:
            env = StudentEnv(max_episode_steps=50, population_seed=seed)
            for ep in range(100):
                obs, info = env.reset(seed=seed * 1000 + ep)
                done = False
                while not done:
                    action = agent.predict(
                        obs,
                        info["action_masks"],
                        persistent_flag=info.get(
                            "persistent_frustration_flag", False
                        ),
                    )
                    totals[ID_TO_ACTION[action]] += 1
                    total_steps += 1
                    obs, _, term, trunc, info = env.step(action)
                    done = term or trunc
            env.close()

        for action in ("explanation", "break", "harder_problem"):
            freq = totals[action] / total_steps
            assert freq >= 0.04, f"{action} aggregate freq {freq:.3f} below 4%"
            assert freq <= 0.25, f"{action} aggregate freq {freq:.3f} above 25%"

        assert totals["simplify_problem"] / total_steps >= 0.02
        assert totals["no_action"] / total_steps <= 0.20


class TestStudentEnvIntegration:
    def test_ert_never_selects_masked_action(self):
        env = StudentEnv(max_episode_steps=20, population_seed=42)
        agent = ExpertRuleBasedAgent()
        obs, info = env.reset(seed=42)
        for _ in range(20):
            mask = info["action_masks"]
            action = agent.predict(
                obs,
                mask,
                persistent_flag=info.get("persistent_frustration_flag", False),
            )
            assert mask[action] == 1, f"Selected masked action {action}"
            obs, _, term, trunc, info = env.step(action)
            if term or trunc:
                break
        env.close()

    def test_ert_runs_full_episode(self):
        env = StudentEnv(max_episode_steps=50, population_seed=7)
        agent = ExpertRuleBasedAgent()
        obs, info = env.reset(seed=7)
        done = False
        steps = 0
        while not done:
            action = agent.predict(
                obs,
                info["action_masks"],
                persistent_flag=info.get("persistent_frustration_flag", False),
            )
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
            steps += 1
        assert steps > 0
        env.close()


class TestRuleRegistry:
    def test_registry_covers_all_rule_ids(self):
        ids = {r.rule_id for r in RULE_REGISTRY}
        assert "T0_break" in ids
        assert "fallback_hint" in ids
        assert len(ids) >= 10


class TestPriorityConflicts:
    def test_confusion_beats_frustration_when_confusion_higher(self):
        agent = ExpertRuleBasedAgent()
        s = _state(frustration=0.26, confusion=0.35)
        rule_id, _, action = agent.resolve_rule(s, False, _full_mask())
        assert rule_id == "T1_high_confusion_explain"
        assert action == ACTION_TO_ID["explanation"]

    def test_low_knowledge_beats_flow(self):
        agent = ExpertRuleBasedAgent()
        s = _state(
            knowledge=0.30,
            engagement=0.70,
            frustration=0.15,
            confusion=0.15,
            boredom=0.15,
        )
        rule_id, _, _ = agent.resolve_rule(s, False, _full_mask())
        assert rule_id == "T4_low_knowledge_scaffold"
