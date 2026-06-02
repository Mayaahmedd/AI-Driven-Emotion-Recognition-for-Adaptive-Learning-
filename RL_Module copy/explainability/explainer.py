"""
JSONL explainer for each tutoring action (spec Phase 4).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from RL_Module import config
from RL_Module.mdp_definition import ACTION_TO_ID, BEST_ACTION_MAP, ID_TO_ACTION, ID_TO_EMOTION, StudentState


def _to_state(obs: Union[np.ndarray, StudentState]) -> StudentState:
    if isinstance(obs, np.ndarray):
        return StudentState.from_vec(obs)
    return obs


def reason_for(
    obs: Union[np.ndarray, StudentState],
    action: int,
    persistent_flag: bool = False,
    confidence: float = 0.85,
) -> str:
    """Module-level reason lookup (no file IO) for use inside StudentEnv.step()."""
    state = _to_state(obs)
    emotion = ID_TO_EMOTION[state.emotion_id]
    action_name = ID_TO_ACTION[action]

    if persistent_flag:
        return "Emergency - reducing cognitive overload"
    if state.knowledge < 0.3 and action == ACTION_TO_ID["scaffold"]:
        return "Low knowledge - scaffolding basics"
    if state.knowledge > 0.7 and action == ACTION_TO_ID["harder_problem"]:
        return "Mastery approaching - advancing level"
    if action == ACTION_TO_ID["no_action"]:
        return "Agent observing - no intervention needed"

    return Explainer.REASONS.get(
        (emotion, action_name), "Agent selected optimal policy action"
    )


class Explainer:
    REASONS = {
        ("confused", "hint"): "Confusion detected - providing hint",
        ("confused", "explanation"): "Confusion detected - clarifying concept",
        ("confused", "scaffold"): "Confusion detected - structured scaffolding",
        ("frustrated", "break"): "Frustration detected - slowing pace",
        ("frustrated", "encouragement"): "Frustration detected - motivating student",
        ("frustrated", "simplify_problem"): "Frustration detected - reducing difficulty",
        ("bored", "encouragement"): "Boredom detected - motivating student",
        ("bored", "harder_problem"): "Boredom detected - increasing challenge",
        ("engaged", "harder_problem"): "High engagement - increasing challenge",
        ("engaged", "scaffold"): "High engagement - structured scaffolding",
        ("engaged", "explanation"): "High engagement - deepening understanding",
    }

    def __init__(self, algorithm: str = "default", log_dir: Optional[Path] = None):
        self.algorithm = algorithm
        self.path = (log_dir or config.LOGS_DIR) / f"explanations_{algorithm}.jsonl"
        self._file = open(self.path, "a")
        self._write_count = 0

    def explain(
        self,
        step: int,
        episode: int,
        obs: Union[np.ndarray, StudentState],
        action: int,
        persistent_flag: bool = False,
        confidence: float = 0.85,
    ) -> Dict[str, Any]:
        state = _to_state(obs)
        emotion = ID_TO_EMOTION[state.emotion_id]
        action_name = ID_TO_ACTION[action]

        reason = reason_for(state.as_vec(), action, persistent_flag, confidence)

        if action in BEST_ACTION_MAP.get(state.emotion_id, set()):
            confidence = min(1.0, confidence + 0.1)

        record = {
            "step": step,
            "episode": episode,
            "emotion": emotion,
            "knowledge": round(state.knowledge, 2),
            "engagement": round(state.engagement, 2),
            "frustration": round(state.frustration, 2),
            "action": action_name,
            "reason": reason,
            "confidence": round(confidence, 2),
        }
        self._write_count += 1
        if self._write_count % 10 == 0:
            self._file.write(json.dumps(record) + "\n")
            self._file.flush()
        return record

    def close(self) -> None:
        self._file.close()
