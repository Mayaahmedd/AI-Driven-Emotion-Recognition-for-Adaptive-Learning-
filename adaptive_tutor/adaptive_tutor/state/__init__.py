"""adaptive_tutor.state - RL state representation.

Three modules:

* :mod:`adaptive_tutor.state.state` - frozen ``LearnerState`` +
  ``PerformanceFeatures`` + ``Transition``. The interchange format
  every controller, replay buffer, and explainer reads.
* :mod:`adaptive_tutor.state.features` - pure-function rolling
  statistics (mean, OLS slope, accuracy, hint rate).
* :mod:`adaptive_tutor.state.builder` - mutable ``StateBuilder`` that
  owns the rolling-window deques and emits ``LearnerState`` snapshots.

Why three files and not one
---------------------------
* ``state.py`` is dependency-light (no torch import at module top),
  so it can be imported in offline-eval scripts and minimal test envs.
* ``features.py`` is pure and trivial to unit-test in isolation.
* ``builder.py`` is the only file with mutable state; isolating it
  makes the data flow obvious in a thesis defence.
"""

from adaptive_tutor.state.builder import StateBuilder
from adaptive_tutor.state.features import (
    WINDOW_LENGTH,
    rolling_accuracy,
    rolling_hint_rate,
    rolling_mean,
    rolling_slope,
)
from adaptive_tutor.state.state import (
    LearnerState,
    PerformanceFeatures,
    Transition,
)

__all__ = [
    "LearnerState",
    "PerformanceFeatures",
    "StateBuilder",
    "Transition",
    "WINDOW_LENGTH",
    "rolling_accuracy",
    "rolling_hint_rate",
    "rolling_mean",
    "rolling_slope",
]
