"""FER rolling-window cache: math + edge cases.

These tests are the closest thing we have to a unit test of the
state-builder's pre-RL feature pipeline; everything downstream relies on
the slope/mean/drift semantics defined here.
"""

from __future__ import annotations

import numpy as np
import pytest

from adaptive_tutor.core.types import EmotionVector
from adaptive_tutor.fer.cache import FERRollingCache
from adaptive_tutor.fer.client import MockFERClient


def test_window_validation() -> None:
    with pytest.raises(ValueError):
        FERRollingCache(window=0)


def test_empty_cache_returns_neutral() -> None:
    c = FERRollingCache(window=4)
    assert c.mean().as_tuple() == (0.0, 0.0, 0.0, 0.0)
    # slope undefined with <2 samples -> neutral
    assert c.slope().as_tuple() == (0.0, 0.0, 0.0, 0.0)
    assert c.latest().as_tuple() == (0.0, 0.0, 0.0, 0.0)


def test_mean_is_arithmetic_mean() -> None:
    c = FERRollingCache(window=3)
    c.push(EmotionVector(0.2, 0.0, 0.0, 0.0))
    c.push(EmotionVector(0.4, 0.0, 0.0, 0.0))
    c.push(EmotionVector(0.6, 0.0, 0.0, 0.0))
    m = c.mean()
    assert m.engaged == pytest.approx(0.4)


def test_slope_rising_engagement_is_above_half() -> None:
    # Monotonic increase -> slope rescaled to > 0.5 on the engaged channel.
    c = FERRollingCache(window=4)
    for v in (0.1, 0.3, 0.5, 0.7):
        c.push(EmotionVector(v, 0.0, 0.0, 0.0))
    s = c.slope()
    assert s.engaged > 0.5
    assert s.confused == pytest.approx(0.5, abs=1e-6)


def test_slope_falling_frustration_is_below_half() -> None:
    c = FERRollingCache(window=4)
    for v in (0.8, 0.6, 0.4, 0.2):
        c.push(EmotionVector(0.0, 0.0, 0.0, v))
    s = c.slope()
    assert s.frustrated < 0.5


def test_window_overflow_drops_oldest() -> None:
    c = FERRollingCache(window=2)
    c.push(EmotionVector(1.0, 0.0, 0.0, 0.0))
    c.push(EmotionVector(0.5, 0.0, 0.0, 0.0))
    c.push(EmotionVector(0.0, 0.0, 0.0, 0.0))  # evicts the first
    assert len(c) == 2
    assert c.mean().engaged == pytest.approx(0.25)


def test_drift_metric_zero_when_unchanged() -> None:
    c = FERRollingCache(window=4)
    for _ in range(3):
        c.push(EmotionVector(0.5, 0.5, 0.5, 0.5))
    assert c.drift_metric(lag=1) == pytest.approx(0.0)


def test_drift_metric_max_channel_delta() -> None:
    c = FERRollingCache(window=4)
    c.push(EmotionVector(0.1, 0.2, 0.3, 0.4))
    c.push(EmotionVector(0.1, 0.2, 0.3, 0.9))  # frustrated jumped 0.5
    assert c.drift_metric(lag=1) == pytest.approx(0.5, abs=1e-6)


def test_summary_returns_pair_of_emotion_vectors() -> None:
    c = FERRollingCache(window=3)
    c.push(EmotionVector(0.1, 0.0, 0.0, 0.0))
    c.push(EmotionVector(0.3, 0.0, 0.0, 0.0))
    c.push(EmotionVector(0.5, 0.0, 0.0, 0.0))
    mean, slope = c.summary()
    assert isinstance(mean, EmotionVector) and isinstance(slope, EmotionVector)


# ---------------------- MockFERClient quick sanity ------------------------


def test_mock_fer_client_cycles_sequence_then_holds_last() -> None:
    seq = [
        EmotionVector(0.8, 0.0, 0.0, 0.0),
        EmotionVector(0.2, 0.0, 0.0, 0.7),
    ]
    client = MockFERClient(seq)
    dummy = np.zeros((1, 3, 4, 4), dtype=np.float32)
    assert client.predict_frames(dummy).engaged == 0.8
    assert client.predict_frames(dummy).frustrated == 0.7
    # Sequence exhausted -> repeats the last entry.
    assert client.predict_frames(dummy).frustrated == 0.7


def test_mock_fer_client_neutral_when_empty_sequence() -> None:
    client = MockFERClient()
    dummy = np.zeros((1, 3, 4, 4), dtype=np.float32)
    assert client.predict_frames(dummy).as_tuple() == (0.0, 0.0, 0.0, 0.0)
