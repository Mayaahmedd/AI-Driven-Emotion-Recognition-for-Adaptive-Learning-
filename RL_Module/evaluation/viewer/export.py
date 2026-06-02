"""Video export and snapshot capture for replay demonstrations."""

from __future__ import annotations

from pathlib import Path
from typing import Deque, List, Optional, Set

import numpy as np
import pygame

from RL_Module.evaluation.viewer.engine import (
    CSVReplayEngine,
    PlaybackController,
    PlaybackStatus,
    ReplayState,
)
from RL_Module.evaluation.viewer.renderers import DashboardRenderer, ensure_pygame_initialized


def _import_cv2():
    """Import OpenCV lazily to avoid SDL conflicts with pygame on macOS."""
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "opencv-python is required for video export. "
            "Install with: pip install opencv-python"
        ) from exc
    return cv2


class SnapshotManager:
    """Capture milestone screenshots during replay."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._saved: Set[str] = set()
        self._peak_engagement = -1.0
        self._peak_knowledge = -1.0

    def maybe_capture(
        self,
        surf: pygame.Surface,
        engine: CSVReplayEngine,
        controller: PlaybackController,
    ) -> None:
        """Save screenshots for thesis milestones."""
        row = controller.current_row()
        index = controller.index

        if row.persistent_flag:
            self._save(surf, f"persistent_frustration_ep{row.episode}_step{row.step}")

        if row.engagement > self._peak_engagement:
            self._peak_engagement = row.engagement
            self._save(surf, f"peak_engagement_{row.engagement:.3f}")

        if row.knowledge > self._peak_knowledge:
            self._peak_knowledge = row.knowledge
            self._save(surf, f"peak_knowledge_{row.knowledge:.3f}")

        for annotation in engine.annotations_near(index, window=0):
            key = f"{annotation.kind}_{annotation.index}"
            if key not in self._saved:
                self._saved.add(key)
                self._save(surf, f"milestone_{annotation.kind}_{annotation.index}")

        if controller.at_end and controller.status == PlaybackStatus.FINISHED:
            self._save(surf, "final_state")

    def _save(self, surf: pygame.Surface, name: str) -> None:
        """Save a unique PNG snapshot."""
        if name in self._saved and not name.startswith("peak_"):
            return
        self._saved.add(name)
        path = self.output_dir / f"{name}.png"
        pygame.image.save(surf, str(path))


class VideoExporter:
    """Render replay frames directly to MP4."""

    def __init__(
        self,
        engine: CSVReplayEngine,
        output_path: Path,
        fps: int = 30,
        presentation: bool = True,
        speed: float = 4.0,
    ) -> None:
        self.engine = engine
        self.output_path = Path(output_path)
        self.fps = fps
        self.presentation = presentation
        self.speed = speed

    def export(self) -> Path:
        """Export the full replay to MP4 without user interaction."""
        cv2 = _import_cv2()
        ensure_pygame_initialized()
        renderer = DashboardRenderer(presentation=self.presentation)
        size = (1920, 1080)
        surf = pygame.Surface(size)
        controller = PlaybackController(self.engine, speed=self.speed)
        controller.play()

        state = ReplayState()
        row = controller.current_row()
        state.display_knowledge = row.knowledge
        state.display_engagement = row.engagement
        state.display_frustration = row.frustration
        state.display_confusion = row.confusion
        state.display_boredom = row.boredom

        emotion_history: List[str] = []
        action_history: List[str] = []
        prev_emotion: Optional[str] = None
        prev_action: Optional[str] = None

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            self.fps,
            size,
        )

        dt = 1.0 / self.fps
        steps_per_frame = max(
            1.0 / self.fps,
            1.0 / (controller.BASE_STEPS_PER_SECOND * self.speed),
        )
        step_accumulator = 0.0

        def write_frame() -> None:
            frame = pygame.surfarray.array3d(surf)
            frame = np.transpose(frame, (1, 0, 2))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame)

        try:
            while True:
                row = controller.current_row()
                if prev_emotion != row.emotion_name:
                    if prev_emotion is not None:
                        emotion_history.append(row.emotion_name)
                    prev_emotion = row.emotion_name
                action_history.append(row.action_name)
                prev_action = row.action_name

                near = self.engine.annotations_near(controller.index, window=0)
                state.caption = near[0].caption if near else row.explainer_reason[:100]
                renderer.update_display_state(state, row, dt)
                renderer.draw(
                    surf,
                    self.engine,
                    controller,
                    state,
                    prev_action=prev_action,
                    emotion_history=emotion_history,
                    action_history=action_history,
                    show_debug=False,
                )
                write_frame()

                if controller.at_end:
                    state.show_analytics = True
                    for _ in range(self.fps * 2):
                        renderer.draw(
                            surf,
                            self.engine,
                            controller,
                            state,
                            prev_action=prev_action,
                            emotion_history=emotion_history,
                            action_history=action_history,
                            show_debug=False,
                        )
                        write_frame()
                    break

                step_accumulator += dt
                while step_accumulator >= steps_per_frame:
                    step_accumulator -= steps_per_frame
                    controller.next_frame()
                    if controller.at_end:
                        break
        finally:
            writer.release()
            pygame.quit()

        return self.output_path
