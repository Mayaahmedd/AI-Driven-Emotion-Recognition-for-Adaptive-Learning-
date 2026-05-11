"""Cross-layer integration (teacher + ASSISTments + state + simulator)."""

from __future__ import annotations

from pathlib import Path

import pytest

from adaptive_tutor.demos.run_episode import run_demo_episode

_REPO = Path(__file__).resolve().parent.parent
_SCIENCEQA_EXCERPT = _REPO / "configs/curriculum/examples/scienceqa_excerpt.yaml"
_SYNTHETIC_CSV = _REPO / "configs/curriculum/examples/assistments_synthetic.csv"


def test_teacher_dataset_state_simulator_integration(tmp_path: Path) -> None:
    """ScienceQA-shaped teacher YAML + ASSISTments rows on a shared skill slug."""
    mini = tmp_path / "overlap.csv"
    mini.write_text(
        "user_id,skill,correct,hint_count,attempt_count,ms_first_response,"
        "Average_confidence(FRUSTRATED),Average_confidence(CONFUSED),"
        "Average_confidence(CONCENTRATING),Average_confidence(BORED)\n"
        "u1,event_probability,1,0,1,2000,0.1,0.1,0.8,0.05\n",
        encoding="utf-8",
    )

    traj, buf = run_demo_episode(
        teacher_path=_SCIENCEQA_EXCERPT,
        dataset_path=mini,
        max_steps=5,
        seed=0,
        policy="heuristic",
        print_fn=None,
    )

    assert len(traj) >= 1
    assert len(buf) == len(traj)
    assert all(t.s.timestep <= t.s_next.timestep for t in traj)
    assert all(isinstance(t.r, float) for t in traj)


def test_math_basic_plus_synthetic_csv_integration() -> None:
    """Smaller fixture path used in most unit tests."""
    math_yaml = _REPO / "configs/curriculum/examples/math_basic.yaml"
    traj, buf = run_demo_episode(
        teacher_path=math_yaml,
        dataset_path=_SYNTHETIC_CSV,
        max_steps=8,
        seed=1,
        policy="random",
        print_fn=None,
    )
    assert len(traj) >= 1
    assert len(buf) == len(traj)
