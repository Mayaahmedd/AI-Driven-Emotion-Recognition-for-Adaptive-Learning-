"""Uniform replay buffer tests."""

from __future__ import annotations

import pytest

from adaptive_tutor.core.types import CompositeAction, EmotionVector, MesoAction
from adaptive_tutor.integration.actions import assistments_action_to_composite
from adaptive_tutor.replay import UniformReplayBuffer
from adaptive_tutor.state.state import LearnerState, PerformanceFeatures, Transition


def _state(slug: str = "c", idx: int = 0, mastery: float = 0.5) -> LearnerState:
    return LearnerState(
        current_concept_id=slug,
        current_concept_index=idx,
        mastery=mastery,
        perf=PerformanceFeatures(0.5, 0.0, 1),
        rolling_emotions=EmotionVector.neutral(),
        engagement_trend=0.0,
        confusion_trend=0.0,
        frustration_trend=0.0,
        boredom_trend=0.0,
    )


def test_add_len_and_sample() -> None:
    buf = UniformReplayBuffer(100, seed=3)
    s0, s1 = _state(mastery=0.1), _state(mastery=0.2)
    a = assistments_action_to_composite("give_hint")
    t = Transition(
        s=s0,
        a=a,
        r=0.5,
        r_components=(("x", 0.5),),
        s_next=s1,
        done=False,
    )
    buf.add(t)
    assert len(buf) == 1
    batch = buf.sample(1)
    assert len(batch) == 1
    assert batch[0].r == pytest.approx(0.5)


def test_sample_empty_returns_empty() -> None:
    buf = UniformReplayBuffer(10, seed=0)
    assert buf.sample(5) == []


def test_capacity_drops_oldest() -> None:
    buf = UniformReplayBuffer(2, seed=0)
    for i in range(3):
        s = _state(mastery=0.1 * i)
        buf.add(
            Transition(
                s=s,
                a=CompositeAction(None, MesoAction(0, 0, 0, 0)),
                r=float(i),
                r_components=(),
                s_next=s,
                done=False,
            )
        )
    assert len(buf) == 2
