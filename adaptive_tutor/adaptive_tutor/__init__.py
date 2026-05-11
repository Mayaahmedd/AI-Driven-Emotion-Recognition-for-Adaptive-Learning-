"""adaptive_tutor — Adaptive tutoring with FER, curriculum memory, and RL.

Top-level re-exports::

    from adaptive_tutor import (
        EmotionVector, MacroAction, MesoAction, CompositeAction,
        LearnerState, PerformanceFeatures, Transition,
    )

Modules are split so curriculum structure (teacher/ScienceQA YAML) stays
separate from behavioural data (ASSISTments) and from RL state, simulator,
and future trainers (see ``docs/adrs/006-scope-reset-bachelor-thesis.md``).
"""

from __future__ import annotations

__version__ = "0.1.0"

from adaptive_tutor.core.types import (
    CompositeAction,
    EmotionVector,
    MacroAction,
    MesoAction,
)
from adaptive_tutor.state.state import (
    LearnerState,
    PerformanceFeatures,
    Transition,
)

__all__ = [
    "CompositeAction",
    "EmotionVector",
    "LearnerState",
    "MacroAction",
    "MesoAction",
    "PerformanceFeatures",
    "Transition",
    "__version__",
]
