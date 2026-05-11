"""Unit tests for the pure rolling-window helpers."""

from __future__ import annotations

import pytest

from adaptive_tutor.state.features import (
    WINDOW_LENGTH,
    rolling_accuracy,
    rolling_hint_rate,
    rolling_mean,
    rolling_slope,
)


def test_window_length_default() -> None:
    """The shared window length is 8 per the bachelor-thesis spec."""
    assert WINDOW_LENGTH == 8


def test_rolling_mean_empty_is_zero() -> None:
    # Cold-start convention: no signal -> 0.0, not NaN, not raise.
    assert rolling_mean([]) == 0.0


def test_rolling_mean_basic() -> None:
    assert rolling_mean([0.2, 0.4, 0.6]) == pytest.approx(0.4)


# ---- rolling_slope -------------------------------------------------------


def test_rolling_slope_short_window_is_zero() -> None:
    # Need >= 2 points; otherwise slope is undefined - return 0.0.
    assert rolling_slope([]) == 0.0
    assert rolling_slope([0.5]) == 0.0


def test_rolling_slope_constant_series() -> None:
    # No movement => slope == 0.
    assert rolling_slope([0.5, 0.5, 0.5, 0.5]) == 0.0


def test_rolling_slope_perfect_rise_is_one() -> None:
    # A monotone ramp from 0 to 1 over n steps maps to +1 in our rescaling.
    values = [i / 7 for i in range(8)]  # 0, 1/7, ..., 1
    assert abs(rolling_slope(values) - 1.0) < 1e-9


def test_rolling_slope_perfect_fall_is_minus_one() -> None:
    values = [1 - i / 7 for i in range(8)]
    assert abs(rolling_slope(values) + 1.0) < 1e-9


def test_rolling_slope_clamped_to_unit_interval() -> None:
    # Even pathological inputs stay in [-1, 1] (defensive clamp).
    weird = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    s = rolling_slope(weird)
    assert -1.0 <= s <= 1.0


# ---- rolling_accuracy ----------------------------------------------------


def test_rolling_accuracy_corrects() -> None:
    # 3 correct out of 4 -> 0.75.
    assert rolling_accuracy([1.0, 0.0, 1.0, 1.0]) == 0.75


def test_rolling_accuracy_empty() -> None:
    assert rolling_accuracy([]) == 0.0


# ---- rolling_hint_rate ---------------------------------------------------


def test_rolling_hint_rate_is_fraction_not_average() -> None:
    # 2 steps used hints (regardless of count) out of 4 -> 0.5.
    assert rolling_hint_rate([0, 3, 0, 2]) == 0.5


def test_rolling_hint_rate_empty() -> None:
    assert rolling_hint_rate([]) == 0.0


def test_rolling_hint_rate_bounded() -> None:
    # All steps with hints -> 1.0 regardless of how many hints each.
    assert rolling_hint_rate([1, 9, 2, 4, 7]) == 1.0
