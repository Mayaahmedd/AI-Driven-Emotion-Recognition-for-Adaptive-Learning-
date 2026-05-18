"""RL-based adaptive tutoring system (MDP + 6 algorithms)."""

from __future__ import annotations

import os
from pathlib import Path

__version__ = "0.1.0"

_HOME_CACHE = Path.home() / ".cache" / "rl_module"
(_HOME_CACHE / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_HOME_CACHE / "matplotlib"))
