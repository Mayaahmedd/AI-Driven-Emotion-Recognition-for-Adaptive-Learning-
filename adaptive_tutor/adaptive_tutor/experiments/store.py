"""Latest experiment: in-process cache + JSON file (Phase 12/13).

The file makes the dashboard work when ``run_experiment`` runs in another
terminal than ``uvicorn``: both read the same path.

Override with env ``ADAPTIVE_TUTOR_EXPERIMENT_CACHE`` (absolute path to a
``.json`` file).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_last: dict[str, Any] | None = None


def _cache_path() -> Path:
    env = os.environ.get("ADAPTIVE_TUTOR_EXPERIMENT_CACHE")
    if env:
        return Path(env).expanduser().resolve()
    base = os.environ.get("XDG_CACHE_HOME")
    if base:
        root = Path(base)
    else:
        root = Path.home() / ".cache"
    d = root / "adaptive_tutor"
    d.mkdir(parents=True, exist_ok=True)
    return d / "last_experiment.json"


def _write_disk(data: dict[str, Any]) -> None:
    path = _cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def _read_disk() -> dict[str, Any] | None:
    path = _cache_path()
    if not path.is_file():
        return None
    try:
        raw = path.read_text(encoding="utf-8")
        out = json.loads(raw)
        return out if isinstance(out, dict) else None
    except (json.JSONDecodeError, OSError):
        return None


def set_last_experiment(result: dict[str, Any]) -> None:
    global _last
    _last = dict(result)
    try:
        _write_disk(_last)
    except OSError:
        pass


def get_last_experiment() -> dict[str, Any] | None:
    global _last
    disk = _read_disk()
    if disk is not None:
        _last = disk
        return dict(_last)
    if _last is not None:
        return dict(_last)
    return None


def clear_last_experiment() -> None:
    global _last
    _last = None
    path = _cache_path()
    try:
        if path.is_file():
            path.unlink()
    except OSError:
        pass
