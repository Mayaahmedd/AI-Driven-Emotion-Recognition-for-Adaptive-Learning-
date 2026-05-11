"""Unified experiment logger.

Why one class
-------------
We want every metric to land in three places without the call site having
to know any of them exist:

1. **TensorBoard**  always on; the local truth.
2. **Weights & Biases**  opt-in via ``WANDB_ENABLED=1`` (decision 20.11).
3. **JSONL trace files**  append-only structured records, parsed later by
   the offline-eval harness and the dashboard back-end.

A single ``ExperimentLogger`` owns the artifact directory and routes
``log_scalar / log_dict / log_trace`` calls to all enabled sinks. Failures
in one sink never propagate: a flaky W&B network must not crash a long
training run.

Concurrency
-----------
Phase 0 assumes single-process logging. The internal lock guards file
writes only so it's safe to share one logger across threads inside a
process; multi-process logging will be tackled when (and if) we move
trainers to ``torch.multiprocessing`` workers.

Schema discipline
-----------------
Every JSONL record carries:

    {
      "schema": "adaptive_tutor.trace.v1",
      "ts":      "<ISO-8601 UTC>",
      "step":    <int>,
      "kind":    "<event-name>",
      "payload": {...}
    }

This pre-empts the same versioning headache we already discussed for the
explainer schema (ADR-005).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Heavy deps are imported lazily so that ``from adaptive_tutor.logging import
# ExperimentLogger`` works in a minimal environment (Phase 0 / unit tests).

_TRACE_SCHEMA = "adaptive_tutor.trace.v1"
_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoggerConfig:
    """Configuration for :class:`ExperimentLogger`.

    Attributes:
        experiment_id: Unique identifier for the run. Used as artifact
            directory name. Conventionally ``<exp_name>__<sha>__seed<k>__<ts>``.
        root_dir: Base directory under which ``<root_dir>/<experiment_id>/``
            will be created.
        enable_tensorboard: Defaults to True. Falls back gracefully if the
            ``tensorboard`` package is not importable.
        enable_wandb: Defaults to ``WANDB_ENABLED=1`` env var. When True the
            ``wandb`` package must be importable.
        wandb_project: Used only when W&B is enabled.
        wandb_entity:  Used only when W&B is enabled.
        wandb_tags:    Free-form tags for W&B filtering.
        trace_jsonl:   When True, structured records are appended to
                       ``<root>/<exp_id>/traces/events.jsonl``.
        flush_every_n: Flush JSONL to disk every N records. Smaller values
                       give better crash-resistance, larger values are faster.
    """

    experiment_id: str
    root_dir: str = "runs"
    enable_tensorboard: bool = True
    enable_wandb: bool = field(
        default_factory=lambda: os.environ.get("WANDB_ENABLED", "").lower()
        in {"1", "true", "yes"}
    )
    wandb_project: str = "adaptive-tutor"
    wandb_entity: str | None = None
    wandb_tags: tuple[str, ...] = ()
    trace_jsonl: bool = True
    flush_every_n: int = 32

    @property
    def run_dir(self) -> Path:
        return Path(self.root_dir) / self.experiment_id

    @property
    def trace_path(self) -> Path:
        return self.run_dir / "traces" / "events.jsonl"

    @property
    def tb_dir(self) -> Path:
        return self.run_dir / "tb"


# ---------------------------------------------------------------------------
# Sinks
# ---------------------------------------------------------------------------


class _TBSink:
    """Lazy wrapper around ``torch.utils.tensorboard.SummaryWriter``.

    Why ``torch.utils.tensorboard`` and not the standalone ``tensorboard``
    package: PyTorch's wrapper is the de facto interface and ships with our
    main dep.
    """

    def __init__(self, log_dir: Path) -> None:
        self._writer: Any | None = None
        self._log_dir = log_dir

    def _ensure(self) -> Any | None:
        if self._writer is not None:
            return self._writer
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            _LOG.warning("TensorBoard requested but not importable; sink disabled.")
            return None
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(log_dir=str(self._log_dir))
        return self._writer

    def log_scalar(self, name: str, value: float, step: int) -> None:
        w = self._ensure()
        if w is None:
            return
        try:
            w.add_scalar(name, float(value), global_step=step)
        except Exception as e:  # pragma: no cover - non-fatal sink failure
            _LOG.warning("TB scalar %s failed: %s", name, e)

    def flush(self) -> None:
        if self._writer is not None:
            try:
                self._writer.flush()
            except Exception as e:  # pragma: no cover - non-fatal sink failure
                _LOG.warning("TB flush failed: %s", e)
                return

    def close(self) -> None:
        if self._writer is not None:
            with contextlib.suppress(Exception):  # pragma: no cover
                self._writer.close()
            self._writer = None


class _WandbSink:
    """Lazy wrapper around the optional ``wandb`` client."""

    def __init__(self, cfg: LoggerConfig) -> None:
        self._cfg = cfg
        self._run: Any | None = None
        self._init_attempted = False

    def _ensure(self) -> Any | None:
        if self._init_attempted:
            return self._run
        self._init_attempted = True
        try:
            import wandb
        except ImportError:
            _LOG.warning("W&B enabled but ``wandb`` not importable; sink disabled.")
            return None
        try:
            self._run = wandb.init(
                project=self._cfg.wandb_project,
                entity=self._cfg.wandb_entity,
                name=self._cfg.experiment_id,
                dir=str(self._cfg.run_dir),
                tags=list(self._cfg.wandb_tags),
                reinit=True,
            )
        except Exception as e:  # pragma: no cover - network/auth flake
            _LOG.warning("W&B init failed: %s", e)
            self._run = None
        return self._run

    def log_scalar(self, name: str, value: float, step: int) -> None:
        run = self._ensure()
        if run is None:
            return
        try:
            run.log({name: float(value)}, step=int(step))
        except Exception as e:  # pragma: no cover
            _LOG.warning("W&B scalar %s failed: %s", name, e)

    def close(self) -> None:
        if self._run is not None:
            with contextlib.suppress(Exception):  # pragma: no cover
                self._run.finish()
            self._run = None


class _JsonlSink:
    """Append-only JSONL trace sink with a write-buffer.

    Crash-resistance vs throughput is tuned by ``flush_every_n``. We hold
    a lock around the buffer + file so writes from multiple threads (e.g.,
    a trainer thread + a metric thread) interleave atomically.
    """

    def __init__(self, path: Path, flush_every_n: int = 32) -> None:
        self._path = path
        self._flush_every_n = max(1, int(flush_every_n))
        self._buf: list[str] = []
        self._lock = threading.Lock()
        self._fh: Any | None = None

    def _ensure(self) -> Any:
        if self._fh is not None:
            return self._fh
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Append-mode handle kept open for the lifetime of the sink.
        # The matching ``close`` method is responsible for releasing it; the
        # ``SIM115`` rule is suppressed because a context-manager would close
        # the file after every write and defeat the buffer.
        self._fh = open(self._path, mode="a", encoding="utf-8")  # noqa: SIM115
        return self._fh

    def write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, separators=(",", ":"), ensure_ascii=False)
        with self._lock:
            self._buf.append(line)
            if len(self._buf) >= self._flush_every_n:
                self._flush_locked()

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        if not self._buf:
            return
        fh = self._ensure()
        fh.write("\n".join(self._buf) + "\n")
        fh.flush()
        # Not all filesystems support fsync; suppress when unavailable.
        with contextlib.suppress(OSError):  # pragma: no cover
            os.fsync(fh.fileno())
        self._buf.clear()

    def close(self) -> None:
        self.flush()
        if self._fh is not None:
            with contextlib.suppress(Exception):  # pragma: no cover
                self._fh.close()
            self._fh = None


# ---------------------------------------------------------------------------
# The unified logger
# ---------------------------------------------------------------------------


class ExperimentLogger:
    """Unified per-experiment metric + trace sink.

    Typical usage::

        cfg = LoggerConfig(experiment_id="exp_dev__seed0__20260511T1430Z")
        with ExperimentLogger(cfg) as logger:
            logger.log_scalar("env/episode_return", 12.3, step=42)
            logger.log_trace("episode_end", {"return": 12.3, "len": 137})

    Failures inside any sink are logged at WARNING level but do not raise 
    a flaky W&B network must never crash a 10-hour training run.
    """

    def __init__(self, cfg: LoggerConfig) -> None:
        self.cfg = cfg
        self._tb = _TBSink(cfg.tb_dir) if cfg.enable_tensorboard else None
        self._wb = _WandbSink(cfg) if cfg.enable_wandb else None
        self._jsonl = (
            _JsonlSink(cfg.trace_path, flush_every_n=cfg.flush_every_n)
            if cfg.trace_jsonl
            else None
        )
        cfg.run_dir.mkdir(parents=True, exist_ok=True)
        self._t0 = time.time()
        _LOG.info(
            "ExperimentLogger started: id=%s, dir=%s, tb=%s, wandb=%s, jsonl=%s",
            cfg.experiment_id,
            cfg.run_dir,
            cfg.enable_tensorboard,
            cfg.enable_wandb,
            cfg.trace_jsonl,
        )

    # ---- context manager ------------------------------------------------
    def __enter__(self) -> ExperimentLogger:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    # ---- public API -----------------------------------------------------
    def log_scalar(self, name: str, value: float, step: int) -> None:
        """Log a scalar metric ``value`` under ``name`` at ``step``.

        Args:
            name:  hierarchical name, e.g. ``"dqn/loss/td_perf"``.
            value: any finite float (NaN/Inf are dropped with a warning).
            step:  global env step or trainer step.
        """
        v = float(value)
        if v != v or v in (float("inf"), float("-inf")):  # NaN / +-inf
            _LOG.warning("Dropped non-finite scalar %s=%s at step %d", name, v, step)
            return
        if self._tb is not None:
            self._tb.log_scalar(name, v, step)
        if self._wb is not None:
            self._wb.log_scalar(name, v, step)

    def log_dict(self, prefix: str, values: dict[str, float], step: int) -> None:
        """Convenience: log every ``v`` under ``f"{prefix}/{k}"``."""
        for k, v in values.items():
            self.log_scalar(f"{prefix}/{k}", v, step)

    def log_trace(self, kind: str, payload: dict[str, Any], step: int | None = None) -> None:
        """Append a structured record to the JSONL trace.

        Args:
            kind:    short event identifier ("episode_end", "explanation", ...).
            payload: JSON-serializable dict.
            step:    optional global step.
        """
        if self._jsonl is None:
            return
        record = {
            "schema": _TRACE_SCHEMA,
            "ts": datetime.now(UTC).isoformat(),
            "step": int(step) if step is not None else None,
            "kind": kind,
            "payload": payload,
        }
        self._jsonl.write(record)

    def flush(self) -> None:
        if self._tb is not None:
            self._tb.flush()
        if self._jsonl is not None:
            self._jsonl.flush()

    def close(self) -> None:
        if self._tb is not None:
            self._tb.close()
        if self._wb is not None:
            self._wb.close()
        if self._jsonl is not None:
            self._jsonl.close()
        _LOG.info(
            "ExperimentLogger closed: id=%s, elapsed=%.1fs",
            self.cfg.experiment_id,
            time.time() - self._t0,
        )
