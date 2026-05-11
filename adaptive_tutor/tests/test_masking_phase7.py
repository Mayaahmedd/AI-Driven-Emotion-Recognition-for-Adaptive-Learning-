"""Phase 7 rule-based action control."""

from __future__ import annotations

from pathlib import Path

import pytest

from adaptive_tutor.core.types import EmotionVector
from adaptive_tutor.masking import (
    ActionFilterContext,
    CooldownTracker,
    WhipsawTracker,
    mask_by_curriculum_and_dataset,
    prerequisites_satisfied,
    run_filter_pipeline,
)
from adaptive_tutor.memory.providers.dataset_provider import ASSISTMENTS_ACTIONS
from adaptive_tutor.memory.providers.teacher_provider import TeacherCurriculumProvider
from adaptive_tutor.simulator import TutoringEnvironment
from adaptive_tutor.state.state import LearnerState, PerformanceFeatures

_REPO = Path(__file__).resolve().parent.parent
_MATH_YAML = _REPO / "configs" / "curriculum/examples/math_basic.yaml"


def _minimal_state(timestep: int = 0) -> LearnerState:
    return LearnerState(
        current_concept_id="x",
        current_concept_index=0,
        mastery=0.5,
        perf=PerformanceFeatures(0.5, 0.0, 1),
        rolling_emotions=EmotionVector.neutral(),
        engagement_trend=0.0,
        confusion_trend=0.0,
        frustration_trend=0.0,
        boredom_trend=0.0,
        timestep=timestep,
    )


def test_mask_strict_dataset_drops_escalation() -> None:
    allowed, reasons = mask_by_curriculum_and_dataset(
        ASSISTMENTS_ACTIONS,
        concept_slug="addition_whole_numbers",
        teacher=None,
        dataset_slugs=frozenset({"other_skill"}),
        dataset_skill_slug="addition_whole_numbers",
        strict_dataset=True,
    )
    assert "harder_problem" not in allowed
    assert "advance_to_next_skill" not in allowed
    assert reasons


def test_mask_unknown_concept_with_teacher_is_restrictive() -> None:
    teacher = TeacherCurriculumProvider(_MATH_YAML)
    allowed, reasons = mask_by_curriculum_and_dataset(
        ASSISTMENTS_ACTIONS,
        concept_slug="not_in_graph",
        teacher=teacher,
        dataset_slugs=None,
        dataset_skill_slug=None,
        strict_dataset=False,
    )
    assert allowed <= frozenset({"retry_current_skill", "encouragement"})
    assert reasons


def test_prerequisites_satisfied_respects_threshold() -> None:
    teacher = TeacherCurriculumProvider(_MATH_YAML)
    assert prerequisites_satisfied(
        teacher,
        "subtraction_whole_numbers",
        {"addition_whole_numbers": 0.59},
        threshold=0.6,
    ) is False
    assert prerequisites_satisfied(
        teacher,
        "subtraction_whole_numbers",
        {"addition_whole_numbers": 0.6},
        threshold=0.6,
    )


def test_cooldown_blocks_immediate_repeat() -> None:
    cd = CooldownTracker({"give_hint": 2})
    s0 = _minimal_state(0)
    a0 = frozenset(ASSISTMENTS_ACTIONS)
    allowed, _ = cd.filter(a0, s0.timestep)
    assert "give_hint" in allowed
    cd.record("give_hint", post_step_timestep=1)
    s1 = _minimal_state(1)
    allowed2, rs = cd.filter(a0, s1.timestep)
    assert "give_hint" not in allowed2
    assert rs


def test_whipsaw_blocks_immediate_pace_reversal() -> None:
    w = WhipsawTracker(window=4, max_pace_switches=3)
    w.record("easier_problem")
    allowed, rs = w.opposite_pair_filter(frozenset(ASSISTMENTS_ACTIONS))
    assert "harder_problem" not in allowed
    assert rs


def test_run_filter_pipeline_deterministic_fallback() -> None:
    teacher = TeacherCurriculumProvider(_MATH_YAML)
    cd = CooldownTracker()
    w = WhipsawTracker()
    ctx = ActionFilterContext(
        teacher=teacher,
        concept_slug="addition_whole_numbers",
        dataset_skill_slug=None,
        dataset_slugs=None,
        strict_dataset=False,
        mastery_by_slug={},
    )
    chosen, allowed, rs, trace = run_filter_pipeline(
        "give_hint",
        _minimal_state(0),
        context=ctx,
        cooldown=cd,
        whipsaw=w,
    )
    assert chosen in allowed
    assert chosen in ASSISTMENTS_ACTIONS
    assert trace.mask_passed


def test_environment_prereq_clamps_escalation() -> None:
    teacher = TeacherCurriculumProvider(_MATH_YAML)
    idx = teacher.concept_index("subtraction_whole_numbers")
    env = TutoringEnvironment(
        concept_slug="subtraction_whole_numbers",
        concept_index=idx,
        num_concepts=teacher.num_concepts(),
        teacher=teacher,
        initial_mastery_by_slug={"addition_whole_numbers": 0.1},
        prereq_mastery_threshold=0.6,
        seed=0,
        max_episode_steps=20,
        mastery_threshold=0.99,
        frustration_terminal_threshold=0.99,
        enable_action_filter=True,
    )
    env.reset()
    *_, info = env.step("harder_problem")
    assert info["action_requested"] == "harder_problem"
    assert info["action"] != "harder_problem"


def test_environment_cooldown_clamps_hint_spam() -> None:
    env = TutoringEnvironment(
        concept_slug="x",
        concept_index=0,
        num_concepts=1,
        seed=0,
        max_episode_steps=10,
        mastery_threshold=0.99,
        frustration_terminal_threshold=0.99,
        teacher=None,
        enable_action_filter=True,
    )
    env.reset()
    info1 = env.step("give_hint")[3]
    assert info1["action"] == "give_hint"
    info2 = env.step("give_hint")[3]
    assert info2["action_requested"] == "give_hint"
    assert info2["action"] != "give_hint"
