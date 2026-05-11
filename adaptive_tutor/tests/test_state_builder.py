"""Tests for the mutable :class:`StateBuilder`.

Coverage:

* rolling-window correctness (window length, overflow, alignment),
* trend computation,
* deterministic snapshots,
* mastery accumulation per concept,
* cold-start behaviour (no observations).
"""

from __future__ import annotations

import pytest

from adaptive_tutor.core.types import EmotionVector
from adaptive_tutor.state.builder import StateBuilder


def _e(*, eng=0.5, con=0.0, bor=0.0, fru=0.0) -> EmotionVector:
    return EmotionVector(eng, con, bor, fru)


# ---- Constructor + invariants -------------------------------------------


def test_window_length_default_is_eight() -> None:
    b = StateBuilder()
    assert b.window_length == 8


def test_window_length_must_be_at_least_two() -> None:
    with pytest.raises(ValueError, match=">= 2"):
        StateBuilder(window_length=1)


# ---- Cold-start snapshot ------------------------------------------------


def test_cold_start_snapshot_is_zeroed() -> None:
    """No observations yet: every rolling stat is 0.0, every trend 0.0,
    mastery 0.0. This is the cold-start contract every consumer relies
    on."""
    b = StateBuilder()
    b.reset(concept_id="probability", concept_index=0)
    s = b.snapshot()
    assert s.mastery == 0.0
    assert s.perf.recent_accuracy == 0.0
    assert s.perf.hint_usage == 0.0
    assert s.perf.attempts == 0
    assert s.rolling_emotions.engaged == 0.0
    assert s.engagement_trend == 0.0
    assert s.timestep == 0


# ---- Rolling windows ----------------------------------------------------


def test_window_overflow_drops_oldest() -> None:
    """Once the window is full, the oldest observation is dropped."""
    b = StateBuilder(window_length=3)
    b.reset(concept_id="c", concept_index=0)
    for v in (0.0, 0.0, 0.0, 1.0):
        b.observe(emotion=_e(eng=v), correct=False)
    s = b.snapshot()
    # Window of length 3: [0, 0, 1] -> mean 1/3.
    assert abs(s.rolling_emotions.engaged - 1 / 3) < 1e-9


def test_observation_aligned_across_channels() -> None:
    """One ``observe`` call must touch every window once; emotion mean
    and correctness mean must be computed over the same N steps."""
    b = StateBuilder(window_length=4)
    b.reset(concept_id="c", concept_index=0)
    for i, c in enumerate((True, False, True, False)):
        b.observe(emotion=_e(eng=0.5), correct=c, hints=i % 2, attempts=1)
    s = b.snapshot()
    # 2 corrects of 4 -> 0.5; 2 of 4 used >=1 hint -> 0.5.
    assert s.perf.recent_accuracy == 0.5
    assert s.perf.hint_usage == 0.5
    assert s.rolling_emotions.engaged == 0.5


# ---- Trends -------------------------------------------------------------


def test_engagement_trend_rising() -> None:
    """Monotone-rising engagement -> trend close to +1 in our rescaling."""
    b = StateBuilder()
    b.reset(concept_id="c", concept_index=0)
    for v in (0.0, 0.143, 0.286, 0.429, 0.571, 0.714, 0.857, 1.0):
        b.observe(emotion=_e(eng=v), correct=True)
    s = b.snapshot()
    assert s.engagement_trend > 0.9


def test_frustration_trend_falling() -> None:
    b = StateBuilder()
    b.reset(concept_id="c", concept_index=0)
    for v in reversed([0.0, 0.143, 0.286, 0.429, 0.571, 0.714, 0.857, 1.0]):
        b.observe(emotion=_e(fru=v), correct=True)
    s = b.snapshot()
    assert s.frustration_trend < -0.9


# ---- Mastery accumulation -----------------------------------------------


def test_mastery_running_mean_for_current_concept() -> None:
    b = StateBuilder()
    b.reset(concept_id="addition", concept_index=0)
    for c in (True, True, False, True):  # 3 of 4 correct
        b.observe(emotion=_e(), correct=c)
    s = b.snapshot()
    assert abs(s.mastery - 0.75) < 1e-9


def test_mastery_is_per_concept() -> None:
    """Switching concept does not wipe the prior concept's mastery."""
    b = StateBuilder()
    b.reset(concept_id="addition", concept_index=0)
    for c in (True, True, True, True):
        b.observe(emotion=_e(), correct=c)
    assert b.mastery_of("addition") == 1.0

    b.switch_concept(concept_id="subtraction", concept_index=1)
    for c in (False, False):
        b.observe(emotion=_e(), correct=c)

    assert b.mastery_of("addition") == 1.0  # preserved
    assert b.mastery_of("subtraction") == 0.0


def test_mastery_unknown_concept_is_zero() -> None:
    b = StateBuilder()
    assert b.mastery_of("never_observed") == 0.0


# ---- Determinism --------------------------------------------------------


def test_snapshots_are_deterministic_for_same_inputs() -> None:
    def replay() -> object:
        b = StateBuilder()
        b.reset(concept_id="c", concept_index=0)
        for c in (True, False, True, False, True, True):
            b.observe(emotion=_e(eng=0.6, con=0.2), correct=c, hints=1, attempts=1)
        return b.snapshot()

    assert replay() == replay()


# ---- Dataset-stats lookup (runtime composition, replaces Hybrid) --------


def test_dataset_stats_pass_through() -> None:
    b = StateBuilder()
    b.reset(concept_id="c", concept_index=0)
    b.observe(emotion=_e(), correct=True)
    s = b.snapshot(dataset_stats=(("mean_correctness", 0.7), ("mean_hint_count", 0.3)))
    assert s.dataset_stats_dict() == {
        "mean_correctness": 0.7,
        "mean_hint_count": 0.3,
    }
