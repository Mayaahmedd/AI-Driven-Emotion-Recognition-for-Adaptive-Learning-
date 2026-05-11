"""Pure-function rolling-window helpers used by the state builder.

This module is deliberately tiny:

* No state - every function takes a ``Sequence[float]`` and returns a
  ``float``.
* No torch - we want this importable from the dataset adapter, the
  simulator, and from offline evaluation scripts without paying the
  torch import cost.
* No numpy - the windows are at most 8 elements; pure Python is
  faster *and* deterministic across machines.

The single shared constant is :data:`WINDOW_LENGTH`. Every rolling
statistic the state builder tracks uses the same window length so the
state shape stays predictable.

Determinism
-----------
Floating-point sums are computed by left-to-right Kahan-free
accumulation (the default ``sum`` builtin). For an 8-element window
the error is bounded and well below any signal we care about.
"""

from __future__ import annotations

from collections.abc import Sequence

# ---------------------------------------------------------------------------
# Window length - one constant, used everywhere.
# ---------------------------------------------------------------------------

WINDOW_LENGTH: int = 8
"""Length of every rolling-window deque in the state builder.

Rationale: 8 timesteps covers roughly one tutoring micro-episode at our
default pacing and is short enough that emotion trends still respond
to recent events. Configurable through the builder constructor.
"""


# ---------------------------------------------------------------------------
# Rolling statistics - all return values in interpretable ranges.
# ---------------------------------------------------------------------------


def rolling_mean(values: Sequence[float]) -> float:
    """Arithmetic mean of ``values``; ``0.0`` for an empty window.

    Cold-start convention: an empty window returns ``0.0``. This is
    safe because every consumer downstream interprets ``0.0`` as
    "no signal yet" rather than "perfectly low".
    """
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def rolling_slope(values: Sequence[float]) -> float:
    """OLS slope of ``values`` against integer x = 0, 1, ..., n-1.

    Returns a value in ``[-1, 1]`` if ``values`` themselves are in
    ``[0, 1]`` (which is the case for emotion / correctness signals).
    Mathematical reasoning:

    The OLS slope for ``y vs x = 0..n-1`` is

        b = sum((x_i - mean_x)(y_i - mean_y)) / sum((x_i - mean_x)^2)

    With ``x_i = 0..n-1``, ``sum((x_i - mean_x)^2) = n*(n^2 - 1) / 12``.
    For an emotion or correctness signal in ``[0, 1]``, the maximum
    achievable absolute value of the numerator is
    ``sum_i (x_i - mean_x) * 1`` summed over the half of indices with
    the same sign as ``(x_i - mean_x)`` - which works out to
    ``n*(n^2 - 1) / (4 * (n - 1))`` for moderate ``n``.

    For convenience and matching the FER cache convention
    (``[-1, 1]``), the raw slope is multiplied by ``(n - 1)`` so a
    perfectly monotone ramp from 0 to 1 over an ``n``-element window
    produces a value of ``+1``. This is the same rescaling used by
    :class:`adaptive_tutor.fer.cache.FERRollingCache`.

    Edge cases
    ----------
    * ``len(values) < 2``: no slope is computable; we return ``0.0``.
    * Constant series: slope ``0.0``.
    """
    n = len(values)
    if n < 2:
        return 0.0
    mean_x = (n - 1) / 2.0
    mean_y = sum(values) / n
    num = 0.0
    den = 0.0
    for i, y in enumerate(values):
        dx = i - mean_x
        num += dx * (y - mean_y)
        den += dx * dx
    if den == 0.0:
        return 0.0
    raw_slope = num / den
    # Rescale: full ramp 0->1 across n steps has raw_slope = 1 / (n-1),
    # so multiplying by (n-1) maps it to +1. This puts the field in
    # the same units as the FER cache (signed [-1, 1]).
    rescaled = raw_slope * (n - 1)
    # Clamp; OLS over a bounded signal cannot exceed [-1, 1] but
    # numerical noise can sneak in.
    return float(max(-1.0, min(rescaled, 1.0)))


def rolling_accuracy(corrects: Sequence[float]) -> float:
    """Rolling mean of correctness labels in ``{0, 1}``.

    Same semantics as :func:`rolling_mean`, but kept as a separate name
    for readability at the call site. Cold-start returns ``0.0``.
    """
    return rolling_mean(corrects)


def rolling_hint_rate(hints: Sequence[float]) -> float:
    """Rolling fraction of steps that used at least one hint, in ``[0, 1]``.

    Accepts raw hint counts (``int >= 0``) - we clamp each entry to
    ``{0, 1}`` so the result is interpretable as a *frequency*, not an
    average count. The state builder cares about "how often does this
    learner reach for hints", not "how many hints exactly".
    """
    if not hints:
        return 0.0
    used = sum(1 for h in hints if h > 0)
    return float(used / len(hints))
