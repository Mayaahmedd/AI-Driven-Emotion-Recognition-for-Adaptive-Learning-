"""In-memory store for the latest experiment (Phase 12/13, thesis demos)."""

from __future__ import annotations

from typing import Any

_last: dict[str, Any] | None = None


def set_last_experiment(result: dict[str, Any]) -> None:
    global _last
    _last = dict(result)


def get_last_experiment() -> dict[str, Any] | None:
    return None if _last is None else dict(_last)


def clear_last_experiment() -> None:
    global _last
    _last = None
