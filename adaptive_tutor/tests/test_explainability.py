"""Phase 10: rule-based explainer."""

from __future__ import annotations

from pathlib import Path

from adaptive_tutor.explainability import (
    explain_action,
    explain_action_from_env_info,
    validate_explanation_dict,
)
from adaptive_tutor.memory.providers.teacher_provider import TeacherCurriculumProvider
from adaptive_tutor.simulator import TutoringEnvironment
from adaptive_tutor.state.state import LearnerState, PerformanceFeatures
from adaptive_tutor.core.types import EmotionVector

_REPO = Path(__file__).resolve().parent.parent
_MATH = _REPO / "configs" / "curriculum" / "examples" / "math_basic.yaml"


def test_explain_action_validates() -> None:
    st = LearnerState(
        current_concept_id="c",
        current_concept_index=0,
        mastery=0.25,
        perf=PerformanceFeatures(0.3, 0.2, 2),
        rolling_emotions=EmotionVector(
            engaged=0.4, confused=0.62, bored=0.1, frustrated=0.15
        ),
        engagement_trend=0.0,
        confusion_trend=0.35,
        frustration_trend=0.0,
        boredom_trend=0.0,
        timestep=2,
    )
    raw = explain_action(
        st,
        "give_hint",
        action_requested="give_hint",
        phase7={
            "mask_passed": True,
            "cooldown_blocked": False,
            "prerequisites_met": True,
            "whipsaw_blocked": False,
        },
        r_components=(
            ("correctness", 0.0),
            ("hint_penalty", -0.2),
            ("engagement", 0.1),
        ),
        correct=0,
    )
    validate_explanation_dict(raw)
    assert raw["action"] == "give_hint"
    assert raw["phase7_checks"]["mask_passed"] is True
    assert "explanation_text" in raw


def test_explain_action_from_env_step() -> None:
    teacher = TeacherCurriculumProvider(_MATH)
    env = TutoringEnvironment(
        concept_slug="addition_whole_numbers",
        concept_index=0,
        num_concepts=teacher.num_concepts(),
        teacher=teacher,
        seed=1,
        max_episode_steps=5,
        mastery_threshold=0.99,
        frustration_terminal_threshold=0.99,
        enable_action_filter=True,
    )
    st0, _ = env.reset()
    _, _r, _d, info = env.step("give_hint")
    exp = explain_action_from_env_info(st0, info)
    validate_explanation_dict(exp)
    assert exp["action"] == info["action"]


def test_explain_action_dataset_grounded() -> None:
    from adaptive_tutor.memory.providers.dataset_provider import DatasetCurriculumProvider

    csv = _REPO / "configs" / "curriculum" / "examples" / "assistments_synthetic.csv"
    ds = DatasetCurriculumProvider(csv)
    skill_stats = ds.stats("addition_whole_numbers")
    st = LearnerState(
        current_concept_id="addition_whole_numbers",
        current_concept_index=0,
        mastery=0.35,
        perf=PerformanceFeatures(0.35, 0.2, 2),
        rolling_emotions=EmotionVector(
            engaged=0.5, confused=0.4, bored=0.1, frustrated=0.2
        ),
        engagement_trend=0.0,
        confusion_trend=0.0,
        frustration_trend=0.0,
        boredom_trend=0.0,
        timestep=1,
    )
    raw = explain_action(
        st,
        "give_hint",
        skill_stats=skill_stats,
        phase7={
            "mask_passed": True,
            "cooldown_blocked": False,
            "prerequisites_met": True,
            "whipsaw_blocked": False,
        },
    )
    validate_explanation_dict(raw)
    assert raw["dataset_evidence"]["skill_correctness"] == skill_stats.mean_correctness
    joined = " ".join(str(d["feature"]) for d in raw["top_drivers"])
    assert "ASSISTments" in joined
    assert raw["phase7_flags"]["mask_passed"] is True
    assert "correctness" in raw["reward_components"]
