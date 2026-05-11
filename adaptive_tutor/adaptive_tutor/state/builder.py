"""Mutable :class:`StateBuilder` that emits immutable ``LearnerState`` snapshots.

Role in the architecture
------------------------
The StateBuilder is the single point of contact between observations
and the RL state. It owns the rolling windows; the rest of the system
just calls ``builder.observe(...)`` and ``builder.snapshot(...)`` and
treats the resulting :class:`LearnerState` as immutable.

Lifecycle
---------

::

    builder = StateBuilder(window_length=8)
    builder.reset(concept_id="basic_probability", concept_index=0)

    for step in range(T):
        # 1. Observe the env outcome of the previous tutor action.
        builder.observe(
            emotion=fer_reading,
            correct=outcome.correct,
            hints=outcome.hints_used,
            attempts=outcome.attempts,
        )
        # 2. Snapshot the current state for the controller.
        s = builder.snapshot(
            num_concepts=N,
            dataset_stats=(("mean_correctness", 0.6),),
        )
        # 3. Controller picks an action, env applies it, repeat.

Why mutable
-----------
The rolling windows are append-only across the episode. Making the
builder immutable would require rebuilding it on every step, which is
both expensive and hard to read. We keep the *output* immutable - that
is what matters for replay, hashing and explainability.

Determinism
-----------
The builder uses pure Python lists/deques and the helpers from
:mod:`adaptive_tutor.state.features`. Given the same observation
sequence it produces byte-identical snapshots across runs and
machines.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence

from adaptive_tutor.core.types import EmotionVector
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
)


class StateBuilder:
    """Owns the rolling windows and the per-concept mastery counters.

    All public methods are documented; private helpers stay underscored.
    """

    def __init__(self, window_length: int = WINDOW_LENGTH) -> None:
        if window_length < 2:
            raise ValueError(
                f"window_length must be >= 2 (need >= 2 points for a slope), "
                f"got {window_length}"
            )
        self._W = int(window_length)
        # Six parallel deques. They share the same maxlen so that
        # snapshot computations are aligned step-for-step.
        self._engaged: deque[float] = deque(maxlen=self._W)
        self._confused: deque[float] = deque(maxlen=self._W)
        self._bored: deque[float] = deque(maxlen=self._W)
        self._frustrated: deque[float] = deque(maxlen=self._W)
        self._correct: deque[float] = deque(maxlen=self._W)
        self._hints: deque[float] = deque(maxlen=self._W)

        # Per-concept mastery: running mean correctness.
        # Stored as (sum_correct, count) so we can compute the mean
        # without storing every observation.
        self._mastery_sum: dict[str, float] = {}
        self._mastery_n: dict[str, int] = {}

        # Identity of the current concept.
        self._current_id: str = ""
        self._current_index: int = 0

        # Step counter (within the current episode).
        self._timestep: int = 0
        self._last_attempts: int = 0

    # ---- Public state-machine API --------------------------------------

    @property
    def window_length(self) -> int:
        return self._W

    @property
    def timestep(self) -> int:
        return self._timestep

    def reset(self, *, concept_id: str, concept_index: int) -> None:
        """Clear all rolling windows and start a new episode.

        Per-concept mastery is **not** cleared: a learner who already
        passed ``basic_probability`` retains that mastery across
        episodes. Pass a fresh builder if you want a true cold-start.
        """
        self._engaged.clear()
        self._confused.clear()
        self._bored.clear()
        self._frustrated.clear()
        self._correct.clear()
        self._hints.clear()
        self._current_id = concept_id
        self._current_index = int(concept_index)
        self._timestep = 0
        self._last_attempts = 0

    def switch_concept(self, *, concept_id: str, concept_index: int) -> None:
        """Change which concept ``mastery`` tracks, without resetting windows.

        Use this when the PPO curriculum manager moves the learner to
        the next concept. Emotion / performance windows are kept (they
        characterise the learner, not the concept).
        """
        self._current_id = concept_id
        self._current_index = int(concept_index)

    def observe(
        self,
        *,
        emotion: EmotionVector,
        correct: bool,
        hints: int = 0,
        attempts: int = 0,
    ) -> None:
        """Push one env outcome into the rolling windows.

        Args
        ----
        emotion:
            Latest FER reading (post-sigmoid, in ``[0, 1]^4``).
        correct:
            Whether the learner answered the latest item correctly.
        hints:
            Number of hints used on the latest item (>= 0).
        attempts:
            Attempts used on the latest item (>= 1 in real episodes,
            0 only on the synthetic cold-start observation).
        """
        if hints < 0:
            raise ValueError(f"hints must be >= 0, got {hints}")
        if attempts < 0:
            raise ValueError(f"attempts must be >= 0, got {attempts}")
        self._engaged.append(emotion.engaged)
        self._confused.append(emotion.confused)
        self._bored.append(emotion.bored)
        self._frustrated.append(emotion.frustrated)
        c = 1.0 if correct else 0.0
        self._correct.append(c)
        self._hints.append(float(hints))
        # Update per-concept mastery (running mean correctness).
        if self._current_id:
            self._mastery_sum[self._current_id] = (
                self._mastery_sum.get(self._current_id, 0.0) + c
            )
            self._mastery_n[self._current_id] = (
                self._mastery_n.get(self._current_id, 0) + 1
            )
        self._last_attempts = int(attempts)
        self._timestep += 1

    # ---- Read-only accessors -------------------------------------------

    def mastery_of(self, concept_id: str) -> float:
        """Running mean correctness for ``concept_id`` in ``[0, 1]``.

        Returns ``0.0`` for concepts never observed; that is the
        cold-start convention used everywhere else in the codebase.
        """
        n = self._mastery_n.get(concept_id, 0)
        if n == 0:
            return 0.0
        return float(self._mastery_sum[concept_id] / n)

    def emotion_window(self) -> dict[str, Sequence[float]]:
        """Read-only snapshot of the current emotion windows.

        Useful for explainer/dashboard inspection. Returns *tuples*,
        not the live deques, so callers cannot mutate the builder.
        """
        return {
            "engaged": tuple(self._engaged),
            "confused": tuple(self._confused),
            "bored": tuple(self._bored),
            "frustrated": tuple(self._frustrated),
            "correct": tuple(self._correct),
            "hints": tuple(self._hints),
        }

    # ---- Snapshot the state -------------------------------------------

    def snapshot(
        self,
        *,
        num_concepts: int = 1,
        dataset_stats: tuple[tuple[str, float], ...] = (),
    ) -> LearnerState:
        """Build an immutable :class:`LearnerState` from the current windows.

        Args
        ----
        num_concepts:
            Curriculum size. Used by downstream ``to_tensor`` to
            normalise ``current_concept_index``; not used here.
            (Kept in the signature so consumers see it together.)
        dataset_stats:
            Optional behavioural lookup for the current concept (e.g.
            from :meth:`DatasetCurriculumProvider.all_stats`). Pass an
            empty tuple when no dataset is wired in.
        """
        del num_concepts  # consumed by LearnerState.to_tensor, not here
        perf = PerformanceFeatures(
            recent_accuracy=rolling_accuracy(tuple(self._correct)),
            hint_usage=rolling_hint_rate(tuple(self._hints)),
            attempts=self._last_attempts,
        )
        rolling_em = EmotionVector(
            engaged=_clip01(rolling_mean(tuple(self._engaged))),
            confused=_clip01(rolling_mean(tuple(self._confused))),
            bored=_clip01(rolling_mean(tuple(self._bored))),
            frustrated=_clip01(rolling_mean(tuple(self._frustrated))),
        )
        return LearnerState(
            current_concept_id=self._current_id,
            current_concept_index=self._current_index,
            mastery=self.mastery_of(self._current_id),
            perf=perf,
            rolling_emotions=rolling_em,
            engagement_trend=rolling_slope(tuple(self._engaged)),
            confusion_trend=rolling_slope(tuple(self._confused)),
            frustration_trend=rolling_slope(tuple(self._frustrated)),
            boredom_trend=rolling_slope(tuple(self._bored)),
            dataset_stats=tuple(dataset_stats),
            timestep=self._timestep,
        )


def _clip01(v: float) -> float:
    """Clamp into ``[0, 1]``. Cheap defence against floating-point drift."""
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return float(v)
