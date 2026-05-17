"""Map ASSISTments logged dict states into :class:`LearnerState` (offline RL / replay)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from adaptive_tutor.core.types import EmotionVector
from adaptive_tutor.state.state import LearnerState, PerformanceFeatures


def learner_state_from_assistments_dict(
    d: Mapping[str, Any],
    *,
    concept_index: int,
    dataset_stats: tuple[tuple[str, float], ...] = (),
    timestep: int = 0,
) -> LearnerState:
    """Build :class:`LearnerState` from a ``state`` / ``next_state`` dict on :class:`RawTransition`.

    Rolling trends are zeroed: logged snapshots already encode mastery and emotion;
    offline training uses the tensor features the DQN head was built for.
    """
    slug = str(d.get("current_skill", ""))
    em = d.get("emotion") or {}
    return LearnerState(
        current_concept_id=slug,
        current_concept_index=int(concept_index),
        mastery=float(d.get("mastery", 0.0)),
        perf=PerformanceFeatures(
            recent_accuracy=float(d.get("recent_accuracy", 0.0)),
            hint_usage=float(d.get("hint_usage", 0.0)),
            attempts=int(d.get("attempts", 0)),
        ),
        rolling_emotions=EmotionVector(
            engaged=float(em.get("engaged", 0.0)),
            confused=float(em.get("confused", 0.0)),
            bored=float(em.get("bored", 0.0)),
            frustrated=float(em.get("frustrated", 0.0)),
        ),
        engagement_trend=0.0,
        confusion_trend=0.0,
        frustration_trend=0.0,
        boredom_trend=0.0,
        dataset_stats=tuple(dataset_stats),
        timestep=int(timestep),
    )


__all__ = ["learner_state_from_assistments_dict"]
