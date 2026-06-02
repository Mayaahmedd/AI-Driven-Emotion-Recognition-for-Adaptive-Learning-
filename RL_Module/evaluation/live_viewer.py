"""
Replay and live viewer for RL student-learning trajectories.

Replay mode (thesis demonstration):
  python -m RL_Module.evaluation.live_viewer --replay --algo DQN --seed 42

Live monitor (during training):
  python -m RL_Module.evaluation.live_viewer --live --algo PPO --seed 42

Presentation mode:
  python -m RL_Module.evaluation.live_viewer --replay --presentation --algo DQN --seed 42

Video export:
  python -m RL_Module.evaluation.live_viewer --replay --export-video thesis_demo.mp4 \\
      --algo DQN --seed 42 --export-fps 30

Snapshots:
  python -m RL_Module.evaluation.live_viewer --replay --snapshot-dir screenshots/ \\
      --algo DQN --seed 42
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional

import pandas as pd
import pygame

from RL_Module import config
from RL_Module.evaluation.viewer.engine import (
    CSVReplayEngine,
    PlaybackController,
    PlaybackStatus,
    ReplayRow,
    ReplayState,
)
from RL_Module.evaluation.viewer.export import SnapshotManager
from RL_Module.evaluation.viewer.renderers import (
    DashboardRenderer,
    LiveDashboardRenderer,
    ensure_pygame_initialized,
)
from RL_Module.evaluation.viewer.theme import PLAYBACK_SPEEDS


def resolve_csv_path(algo: str, seed: int, csv_path: Optional[str]) -> Path:
    """Resolve CSV path from explicit path or default logs location."""
    if csv_path:
        return Path(csv_path)
    return config.LOGS_DIR / f"steps_{algo}_seed{seed}.csv"


def get_latest_row(csv_path: Path) -> Optional[ReplayRow]:
    """Read the latest row from a growing CSV (live mode)."""
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
        if len(df) == 0:
            return None
        return ReplayRow.from_series(len(df) - 1, df.iloc[-1])
    except Exception:
        return None


class ReplayApplication:
    """Interactive replay player with video-player controls."""

    def __init__(
        self,
        engine: CSVReplayEngine,
        presentation: bool = False,
        start_episode: Optional[int] = None,
        start_step: Optional[int] = None,
        snapshot_dir: Optional[Path] = None,
        initial_speed: float = 1.0,
    ) -> None:
        self.engine = engine
        start_index = 0
        if start_episode is not None and start_step is not None:
            start_index = engine.index_for_episode_step(start_episode, start_step)
        elif start_episode is not None:
            start_index = engine.jump_to_episode(start_episode)
        elif start_step is not None:
            start_index = engine.jump_to_step(start_step)

        self.controller = PlaybackController(engine, start_index=start_index, speed=initial_speed)
        self.presentation = presentation
        self.renderer: Optional[DashboardRenderer] = None
        self.state = ReplayState()
        self.snapshot_manager = SnapshotManager(snapshot_dir) if snapshot_dir else None

        row = self.controller.current_row()
        self.state.display_knowledge = row.knowledge
        self.state.display_engagement = row.engagement
        self.state.display_frustration = row.frustration
        self.state.display_confusion = row.confusion
        self.state.display_boredom = row.boredom

        self.emotion_history: Deque[str] = deque(maxlen=20)
        self.action_history: Deque[str] = deque(maxlen=40)
        self.prev_emotion: Optional[str] = None
        self.prev_action: Optional[str] = None
        self.input_mode: Optional[str] = None
        self.input_text: str = ""

    def _update_histories(self) -> None:
        """Track emotion and action histories."""
        row = self.controller.current_row()
        if self.prev_emotion is not None and row.emotion_name != self.prev_emotion:
            self.emotion_history.append(row.emotion_name)
        self.prev_emotion = row.emotion_name
        self.action_history.append(row.action_name)
        self.prev_action = row.action_name

    def _update_caption(self) -> None:
        """Set caption from nearest annotation."""
        near = self.engine.annotations_near(self.controller.index, window=0)
        if near:
            self.state.caption = near[0].caption
        else:
            self.state.caption = ""

    def _handle_keydown(self, event: pygame.event.Event) -> bool:
        """Handle keyboard shortcuts. Returns False to quit."""
        if self.input_mode:
            if event.key == pygame.K_RETURN:
                try:
                    value = int(self.input_text)
                    if self.input_mode == "episode":
                        self.controller.jump_to_episode(value)
                    else:
                        self.controller.jump_to_step(value)
                except ValueError:
                    pass
                self.input_mode = None
                self.input_text = ""
            elif event.key == pygame.K_ESCAPE:
                self.input_mode = None
                self.input_text = ""
            elif event.key == pygame.K_BACKSPACE:
                self.input_text = self.input_text[:-1]
            elif event.unicode.isdigit():
                self.input_text += event.unicode
            return True

        if event.key == pygame.K_ESCAPE:
            return False
        if event.key == pygame.K_SPACE:
            self.controller.toggle_play_pause()
            self.state.show_analytics = False
        elif event.key == pygame.K_r:
            self.controller.restart()
            self.state.show_analytics = False
        elif event.key == pygame.K_s:
            self.controller.stop()
            self.state.show_analytics = False
        elif event.key == pygame.K_LEFT:
            self.controller.pause()
            self.controller.previous_frame()
        elif event.key == pygame.K_RIGHT:
            self.controller.pause()
            self.controller.next_frame()
        elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
            self.controller.cycle_speed(1)
        elif event.key == pygame.K_MINUS:
            self.controller.cycle_speed(-1)
        elif event.key == pygame.K_e:
            self.input_mode = "episode"
            self.input_text = ""
        elif event.key == pygame.K_t:
            self.input_mode = "step"
            self.input_text = ""
        elif event.key == pygame.K_a:
            self.state.show_analytics = not self.state.show_analytics
        elif event.key == pygame.K_HOME:
            self.controller.stop()
        elif event.key == pygame.K_PAGEUP:
            row = self.controller.current_row()
            self.controller.jump_to_episode(max(1, row.episode - 1))
        elif event.key == pygame.K_PAGEDOWN:
            row = self.controller.current_row()
            self.controller.jump_to_episode(row.episode + 1)
        elif pygame.K_1 <= event.key <= pygame.K_6:
            speed_idx = event.key - pygame.K_1
            if speed_idx < len(PLAYBACK_SPEEDS):
                self.controller.set_speed(PLAYBACK_SPEEDS[speed_idx])
        return True

    def run(self) -> None:
        """Main replay loop."""
        ensure_pygame_initialized()
        self.renderer = DashboardRenderer(presentation=self.presentation)
        size = self.renderer.window_size
        flags = pygame.FULLSCREEN if self.presentation else 0
        screen = pygame.display.set_mode(size, flags)
        pygame.display.set_caption(
            f"Replay Viewer - {self.engine.csv_path.name}"
        )
        clock = pygame.time.Clock()
        target_fps = 60

        running = True
        while running:
            dt_ms = clock.tick(target_fps)
            dt = dt_ms / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_f and self.presentation:
                        pygame.display.toggle_fullscreen()
                    elif not self._handle_keydown(event):
                        running = False

            self.controller.update(dt)
            self._update_histories()
            self._update_caption()

            row = self.controller.current_row()
            self.renderer.update_display_state(self.state, row, dt)
            self.renderer.draw(
                screen,
                self.engine,
                self.controller,
                self.state,
                prev_action=self.prev_action,
                emotion_history=self.emotion_history,
                action_history=self.action_history,
                input_text=self.input_text,
                input_mode=self.input_mode,
                show_debug=not self.presentation,
            )

            if self.snapshot_manager is not None:
                self.snapshot_manager.maybe_capture(screen, self.engine, self.controller)

            pygame.display.flip()

        pygame.quit()


def run_live_monitor(algo: str, seed: int, csv_path: Optional[str] = None) -> None:
    """Original live monitor: poll CSV and show latest row."""
    path = resolve_csv_path(algo, seed, csv_path)
    ensure_pygame_initialized()
    screen = pygame.display.set_mode((660, 520))
    pygame.display.set_caption(f"Live Viewer - {algo} seed={seed}")
    clock = pygame.time.Clock()
    renderer = LiveDashboardRenderer()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()

        row = get_latest_row(path)
        if row is None:
            renderer.draw_waiting(
                screen,
                f"Waiting for {path} ...",
            )
        else:
            renderer.draw_row(screen, row)

        pygame.display.flip()
        clock.tick(2)


def build_parser() -> argparse.ArgumentParser:
    """Build command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Live monitor and replay viewer for RL step logs.",
    )
    mode = parser.add_mutually_exclusive_group(required=False)
    mode.add_argument(
        "--live",
        action="store_true",
        help="Monitor a growing CSV in real time (original behavior).",
    )
    mode.add_argument(
        "--replay",
        action="store_true",
        help="Replay an existing CSV like a video player.",
    )

    parser.add_argument("--algo", default="DQN", help="Algorithm name for default CSV path.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for default CSV path.")
    parser.add_argument("--csv", dest="csv_path", default=None, help="Explicit CSV file path.")
    parser.add_argument(
        "--presentation",
        action="store_true",
        help="Full-screen friendly thesis presentation layout.",
    )
    parser.add_argument(
        "--export-video",
        dest="export_video",
        default=None,
        metavar="OUTPUT.mp4",
        help="Export replay to MP4 (implies --replay, no window).",
    )
    parser.add_argument(
        "--export-fps",
        type=int,
        default=30,
        choices=(30, 60),
        help="FPS for video export (30 or 60).",
    )
    parser.add_argument(
        "--export-speed",
        type=float,
        default=4.0,
        help="Playback speed multiplier used during export.",
    )
    parser.add_argument(
        "--snapshot-dir",
        default=None,
        help="Directory to auto-save milestone screenshots during replay.",
    )
    parser.add_argument("--start-episode", type=int, default=None, help="Start replay at episode.")
    parser.add_argument("--start-step", type=int, default=None, help="Start replay at step.")
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        choices=PLAYBACK_SPEEDS,
        help="Initial replay speed.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """Entry point."""
    args = build_parser().parse_args(argv)
    csv_file = resolve_csv_path(args.algo, args.seed, args.csv_path)

    if args.export_video:
        if not csv_file.exists():
            print(f"CSV not found: {csv_file}", file=sys.stderr)
            sys.exit(1)
        engine = CSVReplayEngine(csv_file)
        from RL_Module.evaluation.viewer.export import VideoExporter

        exporter = VideoExporter(
            engine,
            Path(args.export_video),
            fps=args.export_fps,
            presentation=True,
            speed=args.export_speed,
        )
        output = exporter.export()
        print(f"Exported video: {output}")
        return

    if args.live:
        run_live_monitor(args.algo, args.seed, args.csv_path)
        return

    if args.replay:
        if not csv_file.exists():
            print(f"CSV not found: {csv_file}", file=sys.stderr)
            sys.exit(1)
        print(f"Loading {csv_file} ({csv_file.stat().st_size // 1024} KB) ...")
        t0 = time.time()
        engine = CSVReplayEngine(csv_file)
        print(f"Loaded {engine.length} rows in {time.time() - t0:.2f}s")
        app = ReplayApplication(
            engine,
            presentation=args.presentation,
            start_episode=args.start_episode,
            start_step=args.start_step,
            snapshot_dir=Path(args.snapshot_dir) if args.snapshot_dir else None,
            initial_speed=args.speed,
        )
        app.run()
        return

    parser.error("One of --live or --replay is required (unless using --export-video).")


if __name__ == "__main__":
    main()
