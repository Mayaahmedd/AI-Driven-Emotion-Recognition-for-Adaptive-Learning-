"""Phase 13 read-only FastAPI dashboard."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from adaptive_tutor.dashboard.api import create_app
from adaptive_tutor.experiments import clear_last_experiment, run_experiment


@pytest.fixture
def client() -> TestClient:
    clear_last_experiment()
    return TestClient(create_app())


def test_dashboard_html_home(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers.get("content-type", "")
    assert "Adaptive Tutor" in r.text
    assert "DQN and PPO" in r.text

    alias = client.get("/dashboard")
    assert alias.status_code == 200
    assert alias.text == r.text


def test_dashboard_explain_view_html(client: TestClient) -> None:
    r = client.get("/explain/view", params={"action": "give_hint", "mastery": 0.2})
    assert r.status_code == 200
    assert "Explain" in r.text

    short = client.get("/explain", params={"action": "give_hint", "mastery": 0.2})
    assert short.status_code == 200
    assert short.text == r.text


def test_dashboard_endpoints_after_run(client: TestClient) -> None:
    run_experiment(
        {"seed": 0, "episodes": 1, "policy": "heuristic", "max_episode_steps": 5}
    )
    e = client.get("/experiment/latest")
    assert e.status_code == 200
    body = e.json()
    assert body["ok"] is True
    assert body["result"]["policy"] == "heuristic"

    s = client.get("/metrics/simulator")
    assert s.status_code == 200
    assert "learning_rate" in s.json()

    d = client.get("/metrics/dataset")
    assert d.status_code == 200
    assert "baseline_correctness" in d.json()

    x = client.get("/explain/action", params={"action": "give_hint", "mastery": 0.2})
    assert x.status_code == 200
    assert x.json()["action"] == "give_hint"
    clear_last_experiment()
