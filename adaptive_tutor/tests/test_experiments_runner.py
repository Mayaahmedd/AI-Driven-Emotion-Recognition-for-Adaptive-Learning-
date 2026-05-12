"""Phase 12 experiment runner."""

from __future__ import annotations

import json

import pytest

from adaptive_tutor.experiments import clear_last_experiment, get_last_experiment, run_experiment
import adaptive_tutor.experiments.store as store_module


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


def test_store_reload_from_disk_after_memory_clear(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    cache = tmp_path / "r.json"
    monkeypatch.setenv("ADAPTIVE_TUTOR_EXPERIMENT_CACHE", str(cache))
    clear_last_experiment()
    run_experiment(
        {"seed": 0, "episodes": 1, "policy": "heuristic", "max_episode_steps": 3}
    )
    assert cache.is_file()
    store_module._last = None  # type: ignore[attr-defined]  # simulate other process
    again = get_last_experiment()
    assert again is not None
    assert again["policy"] == "heuristic"
    data = json.loads(cache.read_text(encoding="utf-8"))
    assert data["policy"] == "heuristic"
