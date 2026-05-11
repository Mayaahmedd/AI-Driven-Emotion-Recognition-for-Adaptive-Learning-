"""Rolling-window cache over recent FER readings.

Why it exists
-------------
The state representation (decision 20.10) carries both the *instantaneous*
:class:`EmotionVector` and a *summary* over the most-recent W readings.
The rolling summary feeds:

* the contextual bandit's drift trigger
  (``|E_t - E_{t-?}|_? > ?_drift``),
* the DQN's input features (engagement slope, sustained-frustration
  indicator),
* the explainer's provenance section (it quotes the recent window mean
  and slope in the rationale).

We therefore want a *single* O(1) ring buffer that all three consumers
read from rather than each maintaining their own.

Math
----
For a window of length W with most-recent reading at index ``t`` and the
oldest still-in-window reading at index ``t - W + 1``:

* **mean** per channel is the simple arithmetic mean of the window.
* **slope** per channel is the OLS slope of the values against an
  evenly-spaced time axis ``[0, 1, ..., W-1]``. For a fixed-length window
  the denominator ``sum((x - x_mean)^2)`` is constant, so we can
  precompute ``X = [0..W-1] - mean(X)`` and reuse it.

Concretely:

.. math::

    \\text{slope}(y) = \\frac{\\sum_i (x_i - \\bar x)(y_i - \\bar y)}
                            {\\sum_i (x_i - \\bar x)^2}.

We return mean and slope **as** :class:`EmotionVector` instances so the
state builder can treat them uniformly with the raw reading.

Edge cases
----------
* ``len(buffer) < 2`` -> slope is undefined; we return 0.0 per channel.
* ``len(buffer) < W`` -> mean uses the *available* samples; we do not
  zero-pad. This matches what the bandit and explainer expect at warmup.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from adaptive_tutor.core.types import EmotionVector


class FERRollingCache:
    """Fixed-length deque of :class:`EmotionVector` readings.

    Parameters
    ----------
    window:
        Maximum number of recent readings to keep. Older readings fall
        off the back. Default 8 matches ``configs/base.yaml`` (decision
        20.10).

    Complexity
    ----------
    * ``push``:    O(1)
    * ``mean``:    O(W)
    * ``slope``:   O(W)
    * ``summary``: O(W) total (computed in one pass via NumPy)
    """

    _CHANNELS: tuple[str, str, str, str] = ("engaged", "confused", "bored", "frustrated")

    def __init__(self, window: int = 8) -> None:
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        self._window = int(window)
        self._buf: deque[EmotionVector] = deque(maxlen=self._window)
        # Pre-computed regressor for slope OLS. Updated when buffer length
        # changes; cached for the common steady-state where len == window.
        self._x_cached_len: int = -1
        self._x_centered: np.ndarray | None = None
        self._x_var: float = 1.0

    # ---- basic ops ------------------------------------------------------
    def push(self, e: EmotionVector) -> None:
        self._buf.append(e)

    def __len__(self) -> int:
        return len(self._buf)

    def clear(self) -> None:
        self._buf.clear()
        self._x_cached_len = -1

    @property
    def window(self) -> int:
        return self._window

    # ---- vectorized view ------------------------------------------------
    def _matrix(self) -> np.ndarray:
        """Stack the buffer to a ``[n, 4]`` float32 matrix."""
        if not self._buf:
            return np.zeros((0, 4), dtype=np.float32)
        return np.asarray([e.as_tuple() for e in self._buf], dtype=np.float32)

    def _ensure_regressor(self, n: int) -> None:
        """Cache the centered time axis for OLS slope."""
        if self._x_cached_len == n and self._x_centered is not None:
            return
        x = np.arange(n, dtype=np.float32)
        x_centered = x - x.mean()
        var = float((x_centered ** 2).sum())
        self._x_cached_len = n
        self._x_centered = x_centered
        self._x_var = var if var > 0.0 else 1.0  # n==1 ? 0; fall back to 1.0

    # ---- summary ops ----------------------------------------------------
    def mean(self) -> EmotionVector:
        """Mean :class:`EmotionVector` over the current buffer.

        Returns :py:meth:`EmotionVector.neutral` when the buffer is empty
        (warmup before the first FER reading).
        """
        m = self._matrix()
        if m.shape[0] == 0:
            return EmotionVector.neutral()
        v = m.mean(axis=0)
        return EmotionVector(
            engaged=float(v[0]),
            confused=float(v[1]),
            bored=float(v[2]),
            frustrated=float(v[3]),
        )

    def slope(self) -> EmotionVector:
        """Per-channel OLS slope over the current buffer.

        Slope is in units of ``probability / step``. Channels increasing
        over the window get a positive slope; decreasing get negative.
        Clamped into ``[-1, 1]`` because legal values are in ``[0, 1]``
        and across W steps the per-step slope can't exceed 1 in absolute
        value.

        Returned as an :class:`EmotionVector` BUT note: the channels of
        this vector are *slopes* (in ``[0, 1]`` after the clamp +
        rescaling step below) — see :meth:`summary` for the rescaling
        convention that puts the slope back into ``[0, 1]`` for
        :class:`EmotionVector` validation.
        """
        m = self._matrix()
        n = m.shape[0]
        if n < 2:
            return EmotionVector.neutral()
        self._ensure_regressor(n)
        x = self._x_centered  # type: ignore[assignment]
        assert x is not None  # for mypy
        # slope_j = sum((x - x_bar)(y_j - y_bar)) / sum((x - x_bar)^2)
        # = (x_centered @ (y_j - y_bar)) / var
        # The y_bar term cancels because x_centered.sum() == 0.
        slopes = (x @ m) / self._x_var  # [4]
        # Rescale slope into [-1, 1] (it already is given y in [0,1]),
        # then map to [0, 1] so it fits EmotionVector's validation.
        # Convention: 0.5 == flat, >0.5 == rising, <0.5 == falling.
        rescaled = np.clip(0.5 + 0.5 * np.clip(slopes, -1.0, 1.0), 0.0, 1.0)
        return EmotionVector(
            engaged=float(rescaled[0]),
            confused=float(rescaled[1]),
            bored=float(rescaled[2]),
            frustrated=float(rescaled[3]),
        )

    def summary(self) -> tuple[EmotionVector, EmotionVector]:
        """Return ``(mean, slope)`` — the value embedded in :class:`LearnerState`.

        See ``LearnerState.emotion_summary``.
        """
        return self.mean(), self.slope()

    def latest(self) -> EmotionVector:
        """Most recent reading, or neutral if empty."""
        return self._buf[-1] if self._buf else EmotionVector.neutral()

    def drift_metric(self, lag: int = 1) -> float:
        """L? drift between the most recent reading and the one ``lag`` steps ago.

        Used by the contextual bandit's micro-intervention trigger
        (master plan §10). Returns ``0.0`` if the buffer is too short.
        """
        n = len(self._buf)
        if n <= lag:
            return 0.0
        latest = np.asarray(self._buf[-1].as_tuple(), dtype=np.float32)
        previous = np.asarray(self._buf[-1 - lag].as_tuple(), dtype=np.float32)
        return float(np.max(np.abs(latest - previous)))
