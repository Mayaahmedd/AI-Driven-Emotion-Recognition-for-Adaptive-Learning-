"""Phase 11 evaluation metrics and runner."""

from __future__ import annotations

from pathlib import Path

import pytest

from adaptive_tutor.evaluation import (
    compute_dataset_metrics,
    compute_episode_metrics,
    compute_learning_rate,
    run_full_evaluation,
)
from adaptive_tutor.memory.providers.dataset_provider import DatasetCurriculumProvider

_REPO = Path(__file__).resolve().parent.parent
_CSV = _REPO / "configs" / "curriculum" / "examples" / "assistments_synthetic.csv"


def test_compute_learning_rate_stable() -> None:
    curve = [0.1, 0.2, 0.5]
    assert compute_learning_rate(curve) == pytest.approx((0.5 - 0.1) / 2.0)
    assert compute_learning_rate([0.4]) == 0.0


def test_compute_episode_metrics_shapes() -> None:
    tr = [
        {"mastery": 0.2, "reward": 0.1, "correct": 1, "hint": False, "frustrated": 0.1},
        {"mastery": 0.4, "reward": 0.2, "correct": 0, "hint": True, "frustrated": 0.6},
    ]
    m = compute_episode_metrics(tr, mastery_threshold=0.35)
    assert m["mastery_gain"] == pytest.approx(0.2)
    assert m["learning_rate"] == pytest.approx(0.2)
    assert m["episode_length"] == 2.0


def test_dataset_metrics_snapshot() -> None:
    ds = DatasetCurriculumProvider(_CSV)
    rep = compute_dataset_metrics(ds)
    assert "baseline_correctness" in rep
    assert "per_skill" in rep
    assert rep["n_rows"] > 0


def test_run_full_evaluation_smoke() -> None:
    rep = run_full_evaluation(
        episodes_per_policy=2,
        max_episode_steps=24,
        ppo_segments_per_run=2,
        seed=42,
    )
    assert "simulator_metrics" in rep
    assert "dataset_metrics" in rep
    pc = rep["policy_comparison"]
    for k in ("random", "heuristic", "dqn", "ppo"):
        assert k in pc
        assert "reward_mean" in pc[k]
        assert "learning_curve_slope" in pc[k]
