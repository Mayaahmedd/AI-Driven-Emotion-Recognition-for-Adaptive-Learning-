"""Trajectory list + ASSISTments replay environment."""

from __future__ import annotations

from pathlib import Path

from adaptive_tutor.datasets.trajectory_builder import build_episode_list
from adaptive_tutor.memory.providers.dataset_provider import DatasetCurriculumProvider
from adaptive_tutor.memory.providers.teacher_provider import TeacherCurriculumProvider
from adaptive_tutor.simulator.environment import TutoringEnvironment

_REPO = Path(__file__).resolve().parent.parent
_CSV = _REPO / "configs" / "curriculum" / "examples" / "assistments_synthetic.csv"
_MATH = _REPO / "configs" / "curriculum" / "examples" / "math_basic.yaml"


def test_build_episode_list_groups_by_student_skill() -> None:
    ds = DatasetCurriculumProvider(_CSV)
    eps = build_episode_list(ds)
    assert len(eps) >= 3
    assert all(len(ep) >= 1 for ep in eps)
    assert eps[0][0].skill == "addition_whole_numbers"


def test_replay_transition_source_when_dataset_wired() -> None:
    teacher = TeacherCurriculumProvider(_MATH)
    ds = DatasetCurriculumProvider(_CSV)
    env = TutoringEnvironment(
        concept_slug="addition_whole_numbers",
        concept_index=0,
        num_concepts=teacher.num_concepts(),
        teacher=teacher,
        dataset=ds,
        dataset_slugs_for_mask=frozenset(ds.all_stats().keys()),
        dataset_skill_slug="addition_whole_numbers",
        seed=42,
        max_episode_steps=20,
        mastery_threshold=0.99,
        frustration_terminal_threshold=0.99,
        enable_action_filter=True,
    )
    env.reset()
    _s, _r, _d, info = env.step("give_hint")
    assert info.get("transition_source") == "assistments_replay"
