"""CSV replay engine and playback controller."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from RL_Module.evaluation.viewer.theme import (
    EMOTION_NAMES,
    HIGH_REWARD_THRESHOLD,
    KNOWLEDGE_THRESHOLDS,
    PLAYBACK_SPEEDS,
)


class PlaybackStatus(Enum):
    """Playback state machine status."""

    STOPPED = auto()
    PLAYING = auto()
    PAUSED = auto()
    FINISHED = auto()


@dataclass
class ReplayRow:
    """Single step from a training log."""

    index: int
    step: int
    episode: int
    algorithm: str
    seed: int
    action_id: int
    action_name: str
    reward: float
    cumulative_reward: float
    knowledge: float
    engagement: float
    frustration: float
    confusion: float
    boredom: float
    emotion_id: int
    emotion_name: str
    persistent_flag: bool
    terminated: bool
    truncated: bool
    explainer_reason: str

    @classmethod
    def from_series(cls, index: int, row: pd.Series) -> "ReplayRow":
        """Build a replay row from a CSV record."""
        emotion_id = int(float(row.get("emotion_id", 3)))
        emotion_name = str(row.get("emotion_name", EMOTION_NAMES[min(emotion_id, 3)]))
        action_id = int(row.get("action_id", 0))
        return cls(
            index=index,
            step=int(row.get("step", 0)),
            episode=int(row.get("episode", 0)),
            algorithm=str(row.get("algorithm", "")),
            seed=int(row.get("seed", 0)),
            action_id=action_id,
            action_name=str(row.get("action_name", f"action_{action_id}")),
            reward=float(row.get("reward", 0.0)),
            cumulative_reward=float(row.get("cumulative_reward", 0.0)),
            knowledge=float(row.get("knowledge", 0.0)),
            engagement=float(row.get("engagement", 0.0)),
            frustration=float(row.get("frustration", 0.0)),
            confusion=float(row.get("confusion", 0.0)),
            boredom=float(row.get("boredom", 0.0)),
            emotion_id=emotion_id,
            emotion_name=emotion_name,
            persistent_flag=bool(row.get("persistent_flag", False)),
            terminated=bool(row.get("terminated", False)),
            truncated=bool(row.get("truncated", False)),
            explainer_reason=str(row.get("explainer_reason", "")),
        )


@dataclass
class ReplayAnnotation:
    """Timeline annotation for thesis demonstration."""

    index: int
    kind: str
    caption: str


@dataclass
class SessionSummary:
    """Aggregate statistics for the final analytics screen."""

    total_steps: int
    total_episodes: int
    final_knowledge: float
    initial_knowledge: float
    learning_gain: float
    average_reward: float
    total_interventions: int
    emotion_counts: Dict[str, int]
    action_counts: Dict[str, int]
    most_used_action: str
    peak_engagement: float
    peak_knowledge: float
    persistent_frustration_steps: int


@dataclass
class ReplayState:
    """Mutable display state updated each frame."""

    index: int = 0
    display_knowledge: float = 0.0
    display_engagement: float = 0.0
    display_frustration: float = 0.0
    display_confusion: float = 0.0
    display_boredom: float = 0.0
    active_annotations: List[ReplayAnnotation] = field(default_factory=list)
    caption: str = ""
    show_analytics: bool = False


class CSVReplayEngine:
    """Load a step CSV once and provide fast indexed access."""

    NUMERIC_COLUMNS = (
        "step",
        "episode",
        "action_id",
        "reward",
        "cumulative_reward",
        "knowledge",
        "engagement",
        "frustration",
        "confusion",
        "boredom",
        "emotion_id",
    )

    def __init__(self, csv_path: Path) -> None:
        self.csv_path = Path(csv_path)
        self._df = pd.read_csv(self.csv_path)
        if len(self._df) == 0:
            raise ValueError(f"CSV is empty: {self.csv_path}")
        self._rows = [ReplayRow.from_series(i, self._df.iloc[i]) for i in range(len(self._df))]
        self._arrays: Dict[str, np.ndarray] = {}
        for col in self.NUMERIC_COLUMNS:
            if col in self._df.columns:
                self._arrays[col] = self._df[col].to_numpy(dtype=np.float64)
        self._emotion_names = self._df.get("emotion_name", pd.Series(["engaged"] * len(self._df)))
        self._action_names = self._df.get("action_name", pd.Series(["no_action"] * len(self._df)))
        self._unique_actions = sorted(self._action_names.unique())
        self._action_to_idx = {name: idx for idx, name in enumerate(self._unique_actions)}
        self._emotion_to_idx = {name: idx for idx, name in enumerate(EMOTION_NAMES)}
        self._cumulative_emotion_counts, self._cumulative_action_counts = (
            self._build_cumulative_counts()
        )
        self._annotations = self._build_annotations()
        self._episode_starts = self._build_episode_index()
        self._summary = self._build_summary()

    @property
    def length(self) -> int:
        """Number of rows in the replay."""
        return len(self._rows)

    @property
    def summary(self) -> SessionSummary:
        """Precomputed session summary."""
        return self._summary

    @property
    def annotations(self) -> Sequence[ReplayAnnotation]:
        """All timeline annotations."""
        return self._annotations

    def row(self, index: int) -> ReplayRow:
        """Return row at index, clamped to valid range."""
        idx = max(0, min(index, self.length - 1))
        return self._rows[idx]

    def slice_array(self, column: str, start: int, end: int) -> np.ndarray:
        """Return numeric column slice [start, end]."""
        arr = self._arrays.get(column)
        if arr is None:
            return np.array([], dtype=np.float64)
        start = max(0, start)
        end = min(end, self.length - 1)
        if end < start:
            return np.array([], dtype=np.float64)
        return arr[start : end + 1]

    def jump_to_episode(self, episode: int) -> int:
        """Return first index for episode, or closest match."""
        if episode in self._episode_starts:
            return self._episode_starts[episode]
        episodes = sorted(self._episode_starts.keys())
        if not episodes:
            return 0
        closest = min(episodes, key=lambda ep: abs(ep - episode))
        return self._episode_starts[closest]

    def jump_to_step(self, step: int) -> int:
        """Return first index whose step column equals step."""
        steps = self._arrays.get("step")
        if steps is None:
            return 0
        matches = np.where(steps == step)[0]
        if len(matches) == 0:
            return int(np.argmin(np.abs(steps - step)))
        return int(matches[0])

    def index_for_episode_step(self, episode: int, step: int) -> int:
        """Return index matching both episode and step when possible."""
        for i, row in enumerate(self._rows):
            if row.episode == episode and row.step == step:
                return i
        return self.jump_to_episode(episode)

    def emotion_counts_up_to(self, index: int) -> Dict[str, int]:
        """Count emotions from start through index."""
        idx = max(0, min(index, self.length - 1))
        counts_arr = self._cumulative_emotion_counts[idx]
        return {
            name: int(counts_arr[self._emotion_to_idx[name]])
            for name in EMOTION_NAMES
        }

    def action_counts_up_to(self, index: int) -> Dict[str, int]:
        """Count actions from start through index."""
        idx = max(0, min(index, self.length - 1))
        counts_arr = self._cumulative_action_counts[idx]
        return {name: int(counts_arr[i]) for i, name in enumerate(self._unique_actions)}

    def _build_cumulative_counts(
        self,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Precompute cumulative emotion and action counts for O(1) lookup."""
        n = self.length
        emo_counts = np.zeros((n, len(EMOTION_NAMES)), dtype=np.int32)
        act_counts = np.zeros((n, len(self._unique_actions)), dtype=np.int32)
        for i, row in enumerate(self._rows):
            if i > 0:
                emo_counts[i] = emo_counts[i - 1]
                act_counts[i] = act_counts[i - 1]
            emo_idx = self._emotion_to_idx.get(row.emotion_name, 3)
            act_idx = self._action_to_idx.get(row.action_name, 0)
            emo_counts[i, emo_idx] += 1
            act_counts[i, act_idx] += 1
        return emo_counts, act_counts

    def annotations_near(self, index: int, window: int = 30) -> List[ReplayAnnotation]:
        """Return annotations visible around the current index."""
        return [a for a in self._annotations if abs(a.index - index) <= window]

    def _build_episode_index(self) -> Dict[int, int]:
        """Map episode number to first row index."""
        mapping: Dict[int, int] = {}
        for i, row in enumerate(self._rows):
            if row.episode not in mapping:
                mapping[row.episode] = i
        return mapping

    def _build_annotations(self) -> List[ReplayAnnotation]:
        """Detect thesis-relevant events in the trajectory."""
        annotations: List[ReplayAnnotation] = []
        prev_emotion: Optional[str] = None
        prev_persistent = False
        crossed_thresholds: set[tuple[float, int]] = set()
        last_high_reward_index = -100

        for i, row in enumerate(self._rows):
            if prev_emotion is not None and row.emotion_name != prev_emotion:
                annotations.append(
                    ReplayAnnotation(
                        index=i,
                        kind="emotion_transition",
                        caption=f"Emotion shift: {prev_emotion} -> {row.emotion_name}",
                    )
                )
            prev_emotion = row.emotion_name

            for threshold in KNOWLEDGE_THRESHOLDS:
                key = (threshold, row.episode)
                if row.knowledge >= threshold and key not in crossed_thresholds:
                    crossed_thresholds.add(key)
                    annotations.append(
                        ReplayAnnotation(
                            index=i,
                            kind="learning_breakthrough",
                            caption=f"Knowledge threshold reached ({threshold:.0%})",
                        )
                    )

            if row.persistent_flag and not prev_persistent:
                annotations.append(
                    ReplayAnnotation(
                        index=i,
                        kind="persistent_frustration",
                        caption="Persistent frustration detected",
                    )
                )
            prev_persistent = row.persistent_flag

            if row.reward >= HIGH_REWARD_THRESHOLD and i - last_high_reward_index > 10:
                last_high_reward_index = i
                annotations.append(
                    ReplayAnnotation(
                        index=i,
                        kind="high_reward",
                        caption=f"High-reward intervention ({row.action_name})",
                    )
                )

            if i > 0:
                prev = self._rows[i - 1]
                if prev.confusion > 0.5 and row.confusion < prev.confusion - 0.1:
                    annotations.append(
                        ReplayAnnotation(
                            index=i,
                            kind="confusion_resolved",
                            caption="Confusion resolved",
                        )
                    )
                if row.engagement - prev.engagement > 0.08:
                    annotations.append(
                        ReplayAnnotation(
                            index=i,
                            kind="engagement_gain",
                            caption="Student became more engaged",
                        )
                    )
                if row.knowledge - prev.knowledge > 0.05:
                    annotations.append(
                        ReplayAnnotation(
                            index=i,
                            kind="learning_gain",
                            caption="Learning gain increased",
                        )
                    )

        return annotations

    def _build_summary(self) -> SessionSummary:
        """Compute session-level analytics."""
        first = self._rows[0]
        last = self._rows[-1]
        emotion_counts: Dict[str, int] = {name: 0 for name in EMOTION_NAMES}
        action_counts: Dict[str, int] = {}
        persistent_steps = 0
        peak_engagement = 0.0
        peak_knowledge = 0.0

        for row in self._rows:
            emotion_counts[row.emotion_name] = emotion_counts.get(row.emotion_name, 0) + 1
            action_counts[row.action_name] = action_counts.get(row.action_name, 0) + 1
            if row.persistent_flag:
                persistent_steps += 1
            peak_engagement = max(peak_engagement, row.engagement)
            peak_knowledge = max(peak_knowledge, row.knowledge)

        most_used = max(action_counts, key=action_counts.get) if action_counts else "none"
        rewards = self._arrays.get("reward", np.zeros(self.length))
        episodes = int(self._arrays["episode"].max()) if "episode" in self._arrays else last.episode

        return SessionSummary(
            total_steps=self.length,
            total_episodes=episodes,
            final_knowledge=last.knowledge,
            initial_knowledge=first.knowledge,
            learning_gain=last.knowledge - first.knowledge,
            average_reward=float(np.mean(rewards)),
            total_interventions=self.length,
            emotion_counts=emotion_counts,
            action_counts=action_counts,
            most_used_action=most_used,
            peak_engagement=peak_engagement,
            peak_knowledge=peak_knowledge,
            persistent_frustration_steps=persistent_steps,
        )


class PlaybackController:
    """Video-player style transport controls for replay."""

    BASE_STEPS_PER_SECOND = 12.0

    def __init__(
        self,
        engine: CSVReplayEngine,
        start_index: int = 0,
        speed: float = 1.0,
    ) -> None:
        self.engine = engine
        self.index = max(0, min(start_index, engine.length - 1))
        self.speed = speed
        self.status = PlaybackStatus.PAUSED
        self._accumulator = 0.0

    @property
    def at_end(self) -> bool:
        """True when replay index is on the final row."""
        return self.index >= self.engine.length - 1

    @property
    def progress(self) -> float:
        """Normalized playback progress in [0, 1]."""
        if self.engine.length <= 1:
            return 1.0
        return self.index / (self.engine.length - 1)

    def play(self) -> None:
        """Start or resume playback."""
        if self.at_end:
            self.restart()
        self.status = PlaybackStatus.PLAYING

    def pause(self) -> None:
        """Pause playback."""
        if self.status == PlaybackStatus.PLAYING:
            self.status = PlaybackStatus.PAUSED

    def toggle_play_pause(self) -> None:
        """Toggle between playing and paused."""
        if self.status == PlaybackStatus.PLAYING:
            self.pause()
        else:
            self.play()

    def stop(self) -> None:
        """Stop and reset to the beginning."""
        self.status = PlaybackStatus.STOPPED
        self.index = 0
        self._accumulator = 0.0

    def restart(self) -> None:
        """Restart from the first frame."""
        self.index = 0
        self._accumulator = 0.0
        self.status = PlaybackStatus.PLAYING

    def next_frame(self) -> None:
        """Advance one row."""
        if self.index < self.engine.length - 1:
            self.index += 1
        else:
            self.status = PlaybackStatus.FINISHED

    def previous_frame(self) -> None:
        """Go back one row."""
        self.index = max(0, self.index - 1)
        if self.status == PlaybackStatus.FINISHED:
            self.status = PlaybackStatus.PAUSED

    def set_speed(self, speed: float) -> None:
        """Set playback speed multiplier."""
        if speed in PLAYBACK_SPEEDS:
            self.speed = speed

    def cycle_speed(self, direction: int = 1) -> None:
        """Cycle through predefined speed presets."""
        speeds = list(PLAYBACK_SPEEDS)
        try:
            idx = speeds.index(self.speed)
        except ValueError:
            idx = 2
        idx = (idx + direction) % len(speeds)
        self.speed = speeds[idx]

    def jump_to_index(self, index: int) -> None:
        """Jump to an absolute row index."""
        self.index = max(0, min(index, self.engine.length - 1))
        if self.status == PlaybackStatus.FINISHED and not self.at_end:
            self.status = PlaybackStatus.PAUSED

    def jump_to_episode(self, episode: int) -> None:
        """Jump to the start of an episode."""
        self.jump_to_index(self.engine.jump_to_episode(episode))

    def jump_to_step(self, step: int) -> None:
        """Jump to a global step number."""
        self.jump_to_index(self.engine.jump_to_step(step))

    def update(self, dt_seconds: float) -> None:
        """Advance playback according to elapsed time."""
        if self.status != PlaybackStatus.PLAYING:
            return
        self._accumulator += dt_seconds * self.speed * self.BASE_STEPS_PER_SECOND
        while self._accumulator >= 1.0:
            self._accumulator -= 1.0
            if self.index < self.engine.length - 1:
                self.index += 1
            else:
                self.status = PlaybackStatus.FINISHED
                break

    def current_row(self) -> ReplayRow:
        """Return the row at the current playback index."""
        return self.engine.row(self.index)
