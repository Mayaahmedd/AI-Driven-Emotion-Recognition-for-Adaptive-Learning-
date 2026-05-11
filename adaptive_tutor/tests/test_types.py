"""Validation contracts for ``adaptive_tutor.core.types``.

``LearnerState``, ``PerformanceFeatures`` and ``Transition`` moved to
``adaptive_tutor.state.state`` in the bachelor-thesis scope reset; see
``tests/test_state_state.py`` for their tests. This file now only
covers what remains in ``core/types.py``: :class:`EmotionVector` and
the PPO/DQN action dataclasses (:class:`MacroAction`,
:class:`MesoAction`, :class:`CompositeAction`).
"""

from __future__ import annotations

import pytest

from adaptive_tutor.core.types import (
    CompositeAction,
    EmotionVector,
    MacroAction,
    MesoAction,
)

# ----------------------------- EmotionVector -------------------------------


def test_emotion_vector_valid() -> None:
    e = EmotionVector(0.7, 0.1, 0.0, 0.2)
    assert e.as_tuple() == (0.7, 0.1, 0.0, 0.2)


def test_emotion_vector_neutral_is_zero() -> None:
    e = EmotionVector.neutral()
    assert e.as_tuple() == (0.0, 0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"engaged": -0.01, "confused": 0.0, "bored": 0.0, "frustrated": 0.0},
        {"engaged": 0.0, "confused": 1.5, "bored": 0.0, "frustrated": 0.0},
        {"engaged": 0.0, "confused": 0.0, "bored": float("nan"), "frustrated": 0.0},
    ],
)
def test_emotion_vector_rejects_out_of_range(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        EmotionVector(**kwargs)


def test_emotion_vector_is_immutable() -> None:
    e = EmotionVector(0.5, 0.5, 0.5, 0.5)
    # ``dataclasses.FrozenInstanceError`` subclasses ``AttributeError`` when
    # ``slots=True`` is also set, so we catch the broader base class.
    with pytest.raises(AttributeError):
        e.engaged = 0.9  # type: ignore[misc]


# ------------------------------- Actions ----------------------------------


def test_composite_action_requires_meso() -> None:
    meso = MesoAction(0, 0, 0, 0)
    a = CompositeAction(macro=None, meso=meso)
    assert a.meso is meso


def test_macro_and_meso_construct() -> None:
    a = CompositeAction(
        macro=MacroAction(instruction_type=1, concept_id=5),
        meso=MesoAction(0, 1, 2, 3),
    )
    assert a.macro is not None and a.macro.concept_id == 5
    assert a.meso.content_variant == 0
