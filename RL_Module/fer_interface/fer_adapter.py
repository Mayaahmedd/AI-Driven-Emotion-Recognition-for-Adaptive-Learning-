"""
FER adapter: synthetic emotion (training) or live FER output (deployment).
"""

from __future__ import annotations

from collections import deque
from typing import Optional, Tuple, Union

import numpy as np

from RL_Module import config
from RL_Module.environment.student_model import SyntheticStudent
from RL_Module.mdp_definition import EMOTION_TO_ID, ID_TO_EMOTION, StudentState


def emotion_from_state(state: StudentState) -> Tuple[int, float]:
    """Simulation mode: derive emotion from student state thresholds."""
    if state.frustration > 0.6:
        return 2, 0.95
    if state.confusion > 0.5:
        return 0, 0.90
    if state.boredom > 0.5:
        return 1, 0.85
    return 3, 0.80


class FERAdapter:
    """Connects FER output to emotion_id in the student state vector."""

    def __init__(self, use_real_fer: Optional[bool] = None):
        self.use_real_fer = use_real_fer if use_real_fer is not None else config.USE_REAL_FER
        self._current_emotion_id: int = 3
        self._confidence: float = 1.0
        self._persist_flag: bool = False
        self._emotion_history: deque = deque(maxlen=3)
        self._prev_emotion_id: int = 3

    @property
    def confidence(self) -> float:
        return self._confidence

    @property
    def persist_flag(self) -> bool:
        return self._persist_flag

    @property
    def prev_emotion_id(self) -> int:
        return self._prev_emotion_id

    def reset(self) -> None:
        self._current_emotion_id = 3
        self._confidence = 1.0
        self._persist_flag = False
        self._emotion_history.clear()
        self._prev_emotion_id = 3

    def synthetic_emotion(self, student: SyntheticStudent) -> Tuple[int, float]:
        return emotion_from_state(student.state)

    def ingest_live(self, label: str, confidence: float, frustration: float = 0.0) -> Tuple[int, float]:
        if confidence < config.FER_CONFIDENCE_THRESHOLD:
            return self._current_emotion_id, self._confidence

        eid = EMOTION_TO_ID.get(label.lower(), self._current_emotion_id)
        self._prev_emotion_id = self._current_emotion_id
        self._current_emotion_id = eid
        self._confidence = confidence
        self._emotion_history.append(eid)

        if (
            len(self._emotion_history) == 3
            and len(set(self._emotion_history)) == 1
            and frustration > 0.7
        ):
            self._persist_flag = True

        return eid, confidence

    def get_emotion(
        self,
        state: Union[np.ndarray, StudentState],
        fer_raw_output: Optional[Tuple[str, float]] = None,
    ) -> Tuple[int, float]:
        """
        Public interface (spec Phase 8).
        Accepts obs vector (6,) or StudentState.
        fer_raw_output = ("frustrated", 0.82) in deployment mode.
        """
        if isinstance(state, np.ndarray):
            state = StudentState.from_vec(state)

        if self.use_real_fer and fer_raw_output is not None:
            label, conf = fer_raw_output
            return self.ingest_live(label, conf, state.frustration)

        eid, conf = emotion_from_state(state)
        self._prev_emotion_id = self._current_emotion_id
        self._current_emotion_id = eid
        self._confidence = conf
        return eid, conf

    def update(
        self,
        student: SyntheticStudent,
        live_label: Optional[str] = None,
        live_confidence: Optional[float] = None,
    ) -> Tuple[int, float]:
        return self.get_emotion(
            student.state,
            fer_raw_output=(live_label, live_confidence) if live_label else None,
        )

    def current_emotion_id(self) -> int:
        return self._current_emotion_id

    def current_emotion(self) -> str:
        return ID_TO_EMOTION[self._current_emotion_id]
