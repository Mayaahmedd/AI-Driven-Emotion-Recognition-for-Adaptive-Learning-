"""Validation contracts for ``adaptive_tutor.core.types``."""

from __future__ import annotations

import pytest

from adaptive_tutor.core.types import (
    CompositeAction,
    EmotionVector,
    LearnerState,
    MacroAction,
    MesoAction,
    MicroAction,
    PerformanceFeatures,
    Transition,
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


# --------------------------- PerformanceFeatures ---------------------------


def test_performance_features_valid() -> None:
    pf = PerformanceFeatures(0.5, 1, 2, 0.0, 3)
    assert pf.streak_correct == 3


def test_performance_features_rejects_bad_correctness() -> None:
    with pytest.raises(ValueError):
        PerformanceFeatures(1.2, 0, 0, 0.0, 0)


def test_performance_features_rejects_bad_hint_level() -> None:
    with pytest.raises(ValueError):
        PerformanceFeatures(0.5, 0, 4, 0.0, 0)


# ------------------------------- LearnerState ------------------------------


def _state(mastery: tuple[float, ...] = (0.1, 0.2, 0.3)) -> LearnerState:
    return LearnerState(
        emotion=EmotionVector(0.5, 0.1, 0.0, 0.0),
        emotion_summary=(EmotionVector.neutral(), EmotionVector.neutral()),
        perf=PerformanceFeatures(0.5, 0, 0, 0.0, 0),
        mastery=mastery,
        current_concept_id=0,
        session_step=0,
    )


def test_learner_state_valid() -> None:
    s = _state()
    assert s.num_concepts == 3
    assert s.cooldowns_dict() == {}


def test_learner_state_rejects_negative_step() -> None:
    with pytest.raises(ValueError):
        LearnerState(
            emotion=EmotionVector.neutral(),
            emotion_summary=(EmotionVector.neutral(), EmotionVector.neutral()),
            perf=PerformanceFeatures(0.5, 0, 0, 0.0, 0),
            mastery=(0.5,),
            current_concept_id=0,
            session_step=-1,
        )


def test_learner_state_rejects_bad_mastery() -> None:
    with pytest.raises(ValueError):
        _state(mastery=(0.5, 1.7))


def test_learner_state_is_hashable() -> None:
    # Hashability matters for explainer caches; this verifies the
    # tuple-of-pairs design for ``cooldowns`` and ``extras``.
    s = _state()
    d: dict[LearnerState, bool] = {s: True}
    assert d[s] is True


# ------------------------------- Actions ----------------------------------


def test_composite_action_requires_meso() -> None:
    meso = MesoAction(0, 0, 0, 0)
    a = CompositeAction(macro=None, meso=meso, micro=None)
    assert a.meso is meso


def test_macro_meso_micro_construct() -> None:
    a = CompositeAction(
        macro=MacroAction(instruction_type=1, concept_id=5),
        meso=MesoAction(0, 1, 2, 3),
        micro=MicroAction(intervention_id=4),
    )
    assert a.macro is not None and a.macro.concept_id == 5
    assert a.micro is not None and a.micro.intervention_id == 4


# ------------------------------- Transition --------------------------------


def test_transition_components_dict_roundtrip() -> None:
    s = _state()
    t = Transition(
        s=s,
        a=CompositeAction(None, MesoAction(0, 0, 0, 0), None),
        r=0.5,
        r_components=(("performance", 0.4), ("emotion", 0.1)),
        s_next=s,
        done=False,
        info=(("segment_id", 7),),
    )
    assert t.r_components_dict() == {"performance": 0.4, "emotion": 0.1}
    assert t.info_dict() == {"segment_id": 7}
