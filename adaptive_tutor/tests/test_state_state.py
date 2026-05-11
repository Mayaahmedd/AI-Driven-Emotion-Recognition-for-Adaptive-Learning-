"""Validation contracts and tensor schema for ``adaptive_tutor.state.state``."""

from __future__ import annotations

import pytest

from adaptive_tutor.core.types import (
    CompositeAction,
    EmotionVector,
    MesoAction,
)
from adaptive_tutor.state.state import (
    LearnerState,
    PerformanceFeatures,
    Transition,
)

torch = pytest.importorskip("torch")


def _state(
    *,
    mastery: float = 0.4,
    timestep: int = 3,
    concept_index: int = 2,
    dataset_stats: tuple[tuple[str, float], ...] = (),
) -> LearnerState:
    return LearnerState(
        current_concept_id="basic_probability",
        current_concept_index=concept_index,
        mastery=mastery,
        perf=PerformanceFeatures(
            recent_accuracy=0.5,
            hint_usage=0.25,
            attempts=2,
        ),
        rolling_emotions=EmotionVector(0.6, 0.2, 0.1, 0.1),
        engagement_trend=0.3,
        confusion_trend=-0.1,
        frustration_trend=0.0,
        boredom_trend=-0.2,
        dataset_stats=dataset_stats,
        timestep=timestep,
    )


# ---- PerformanceFeatures -------------------------------------------------


def test_performance_features_valid() -> None:
    pf = PerformanceFeatures(0.7, 0.1, 2)
    assert pf.recent_accuracy == 0.7
    assert pf.attempts == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"recent_accuracy": -0.01, "hint_usage": 0.0, "attempts": 0},
        {"recent_accuracy": 0.0, "hint_usage": 1.2, "attempts": 0},
        {"recent_accuracy": 0.0, "hint_usage": 0.0, "attempts": -1},
    ],
)
def test_performance_features_rejects_bad_values(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        PerformanceFeatures(**kwargs)


# ---- LearnerState validation --------------------------------------------


def test_learner_state_valid() -> None:
    s = _state()
    assert s.current_concept_id == "basic_probability"
    assert s.timestep == 3


def test_learner_state_rejects_bad_mastery() -> None:
    with pytest.raises(ValueError):
        _state(mastery=1.5)


def test_learner_state_rejects_negative_timestep() -> None:
    with pytest.raises(ValueError):
        _state(timestep=-1)


def test_learner_state_rejects_negative_concept_index() -> None:
    with pytest.raises(ValueError):
        _state(concept_index=-1)


def test_learner_state_rejects_out_of_range_trend() -> None:
    with pytest.raises(ValueError):
        LearnerState(
            current_concept_id="x",
            current_concept_index=0,
            mastery=0.5,
            perf=PerformanceFeatures(0.5, 0.0, 0),
            rolling_emotions=EmotionVector.neutral(),
            engagement_trend=1.5,
            confusion_trend=0.0,
            frustration_trend=0.0,
            boredom_trend=0.0,
        )


def test_learner_state_is_hashable() -> None:
    # Hashability matters for explainer caches that key on the state.
    s = _state()
    d: dict[LearnerState, bool] = {s: True}
    assert d[s] is True


# ---- Human-readable view ------------------------------------------------


def test_to_dict_is_complete_and_human_readable() -> None:
    """The explainer/dashboard must be able to print the state without
    touching tensors. This pins the public dict shape."""
    s = _state(dataset_stats=(("mean_correctness", 0.6),))
    d = s.to_dict()
    assert d["current_concept_id"] == "basic_probability"
    assert d["mastery"] == 0.4
    assert d["recent_accuracy"] == 0.5
    assert d["rolling_emotions"]["engaged"] == 0.6
    assert d["trends"]["engagement"] == 0.3
    assert d["dataset_stats"] == {"mean_correctness": 0.6}


# ---- to_tensor schema ---------------------------------------------------


def test_to_tensor_shape_and_range() -> None:
    s = _state()
    t = s.to_tensor(num_concepts=5)
    assert t.shape == (len(LearnerState.TENSOR_FEATURE_NAMES),)
    # Every feature is normalised to [0, 1].
    assert bool(((t >= 0.0) & (t <= 1.0)).all())


def test_to_tensor_is_deterministic_across_calls() -> None:
    s = _state()
    t1 = s.to_tensor(num_concepts=5)
    t2 = s.to_tensor(num_concepts=5)
    assert torch.equal(t1, t2)


def test_to_tensor_attempts_saturation() -> None:
    """attempts >= MAX_ATTEMPTS_FOR_NORM saturates to 1.0."""
    s = LearnerState(
        current_concept_id="x",
        current_concept_index=0,
        mastery=0.0,
        perf=PerformanceFeatures(0.0, 0.0, 99),  # huge
        rolling_emotions=EmotionVector.neutral(),
        engagement_trend=0.0,
        confusion_trend=0.0,
        frustration_trend=0.0,
        boredom_trend=0.0,
    )
    t = s.to_tensor(num_concepts=1)
    idx = LearnerState.TENSOR_FEATURE_NAMES.index("attempts_norm")
    assert float(t[idx]) == 1.0


def test_to_tensor_trend_shift_to_unit_interval() -> None:
    """Signed trends in [-1, 1] are mapped to [0, 1] via 0.5*(t+1)."""
    s = LearnerState(
        current_concept_id="x",
        current_concept_index=0,
        mastery=0.5,
        perf=PerformanceFeatures(0.5, 0.0, 0),
        rolling_emotions=EmotionVector.neutral(),
        engagement_trend=1.0,
        confusion_trend=-1.0,
        frustration_trend=0.0,
        boredom_trend=0.0,
    )
    t = s.to_tensor(num_concepts=1)
    eng = float(t[LearnerState.TENSOR_FEATURE_NAMES.index("engagement_trend_pos")])
    conf = float(t[LearnerState.TENSOR_FEATURE_NAMES.index("confusion_trend_pos")])
    assert eng == 1.0
    assert conf == 0.0


# ---- Transition ---------------------------------------------------------


def test_transition_roundtrip() -> None:
    s = _state()
    t = Transition(
        s=s,
        a=CompositeAction(None, MesoAction(0, 0, 0, 0)),
        r=0.5,
        r_components=(("performance", 0.4), ("emotion", 0.1)),
        s_next=s,
        done=False,
        info=(("segment_id", 7),),
    )
    assert t.r_components_dict() == {"performance": 0.4, "emotion": 0.1}
    assert t.info_dict() == {"segment_id": 7}
