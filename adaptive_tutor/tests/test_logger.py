"""Lifecycle + sink contracts for ``adaptive_tutor.logging``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from adaptive_tutor.logging import ExperimentLogger, LoggerConfig


def _cfg(tmp_path: Path, *, jsonl: bool = True, tb: bool = False) -> LoggerConfig:
    return LoggerConfig(
        experiment_id="exp_test",
        root_dir=str(tmp_path),
        enable_tensorboard=tb,
        enable_wandb=False,
        trace_jsonl=jsonl,
        flush_every_n=1,  # flush on every write so reads see it immediately
    )


def test_logger_creates_run_dir(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    with ExperimentLogger(cfg):
        pass
    assert (tmp_path / "exp_test").exists()


def test_logger_jsonl_records_have_required_schema(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    with ExperimentLogger(cfg) as log:
        log.log_trace("startup", {"version": "0.1.0"}, step=0)
        log.log_trace("episode_end", {"return": 1.2, "len": 17}, step=42)

    lines = (tmp_path / "exp_test" / "traces" / "events.jsonl").read_text().strip().split("\n")
    assert len(lines) == 2
    for line in lines:
        rec = json.loads(line)
        assert rec["schema"] == "adaptive_tutor.trace.v1"
        assert "ts" in rec
        assert "kind" in rec
        assert "payload" in rec


def test_logger_drops_non_finite_scalars(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    cfg = _cfg(tmp_path, tb=False)
    with (
        caplog.at_level("WARNING", logger="adaptive_tutor.logging.logger"),
        ExperimentLogger(cfg) as log,
    ):
        log.log_scalar("metric/nan", float("nan"), step=0)
        log.log_scalar("metric/inf", float("inf"), step=0)
    assert any("non-finite" in r.message for r in caplog.records)


def test_logger_log_dict_expands_prefix(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    # Exercises just the routing logic — there are no TB/W&B sinks, so
    # the call should not raise.
    with ExperimentLogger(cfg) as log:
        log.log_dict("eval", {"gain": 0.5, "retention": 0.7}, step=1)


def test_logger_idempotent_close(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    log = ExperimentLogger(cfg)
    log.close()
    log.close()  # must not raise
