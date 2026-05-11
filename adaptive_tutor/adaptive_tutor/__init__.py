"""adaptive_tutor — Hybrid Adaptive Tutoring System.

Top-level package. Re-exports the most commonly used public symbols from
``adaptive_tutor.core`` so consumers can write::

    from adaptive_tutor import LearnerState, EmotionVector, CompositeAction

The package is intentionally split into small sub-packages (see ``docs/adrs/``
and the planning document) so that the RL stack, bandit stack, curriculum
providers, simulator, and dashboard back-end can evolve independently.
"""

from __future__ import annotations

__version__ = "0.1.0"

# Re-exports populated by Phase 1 once core/types.py and core/protocols.py
# exist. We import lazily-safe symbols here; failure to import these is a
# packaging bug and should surface at import time.
from adaptive_tutor.core.types import (
    CompositeAction,
    EmotionVector,
    LearnerState,
    MacroAction,
    MesoAction,
    MicroAction,
    PerformanceFeatures,
    Transition,
)

__all__ = [
    "CompositeAction",
    "EmotionVector",
    "LearnerState",
    "MacroAction",
    "MesoAction",
    "MicroAction",
    "PerformanceFeatures",
    "Transition",
    "__version__",
]
