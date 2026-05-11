"""Phase 12 experiment runner."""

from __future__ import annotations

import pytest

from adaptive_tutor.experiments import clear_last_experiment, get_last_experiment, run_experiment


@pytest.fixture(autouse=True)
def _clear_store() -> None:
    clear_last_experiment()
    yield
    clear_last_experiment()


def test_run_experiment_heuristic_smoke() -> None:
    out = run_experiment(
        {
            "seed": 7,
            "episodes": 2,
            "policy": "heuristic",
            "max_episode_steps": 12,
        }
    )
    assert out["policy"] == "heuristic"
    assert "experiment_id" in out
    assert len(out["episode_summaries"]) == 2
    assert "dataset_metrics" in out
    assert get_last_experiment() is not None
    assert get_last_experiment()["avg_reward"] == out["avg_reward"]


def test_run_experiment_random_and_ppo_smoke() -> None:
    r = run_experiment(
        {
            "seed": 1,
            "episodes": 1,
            "policy": "random",
            "max_episode_steps": 8,
        }
    )
    assert r["policy"] == "random"
    p = run_experiment(
        {
            "seed": 2,
            "episodes": 1,
            "policy": "ppo",
            "max_episode_steps": 10,
            "ppo_segments_per_episode": 2,
        }
    )
    assert p["policy"] == "ppo"
