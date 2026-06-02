"""Pygame renderers for replay and live monitoring."""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pygame

from RL_Module import config
from RL_Module.evaluation.viewer.engine import (
    CSVReplayEngine,
    PlaybackController,
    PlaybackStatus,
    ReplayRow,
    ReplayState,
    SessionSummary,
)
from RL_Module.evaluation.viewer.theme import EMOTION_NAMES, Theme
from RL_Module.mdp_definition import ACTIONS

Color = Tuple[int, int, int]


def ensure_pygame_initialized() -> None:
    """Initialize pygame before creating fonts or surfaces."""
    if not pygame.get_init():
        pygame.init()


class FontBundle:
    """Font set scaled for normal or presentation mode."""

    def __init__(self, presentation: bool = False) -> None:
        ensure_pygame_initialized()
        scale = 1.5 if presentation else 1.18
        font_name = "Avenir Next,Helvetica Neue,Arial,sans-serif"
        self.sm = pygame.font.SysFont(font_name, max(13, int(14 * scale)))
        self.md = pygame.font.SysFont(font_name, max(16, int(18 * scale)), bold=True)
        self.lg = pygame.font.SysFont(font_name, max(23, int(26 * scale)), bold=True)
        self.xl = pygame.font.SysFont(font_name, max(30, int(42 * scale)), bold=True)
        self.title = pygame.font.SysFont(font_name, max(20, int(24 * scale)), bold=True)


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation."""
    return a + (b - a) * t


def draw_panel(
    surf: pygame.Surface,
    rect: pygame.Rect,
    theme: Theme,
    radius: int = 8,
    fill: Optional[Color] = None,
) -> None:
    """Draw a rounded panel background."""
    pygame.draw.rect(surf, fill or theme.panel, rect, border_radius=radius)


def draw_progress_bar(
    surf: pygame.Surface,
    font: pygame.font.Font,
    label: str,
    value: float,
    color: Color,
    rect: pygame.Rect,
    theme: Theme,
) -> None:
    """Draw an animated progress bar with label."""
    pygame.draw.rect(surf, theme.progress_bg, rect, border_radius=4)
    fill_w = max(0, min(rect.width, int(rect.width * float(np.clip(value, 0.0, 1.0)))))
    if fill_w > 0:
        pygame.draw.rect(
            surf,
            color,
            pygame.Rect(rect.x, rect.y, fill_w, rect.height),
            border_radius=4,
        )
    text = font.render(label, True, theme.text)
    value_text = font.render(f"{value:.2f}", True, theme.highlight)
    text_x = rect.right + 8
    surf.blit(text, (text_x, rect.y - 1))
    surf.blit(value_text, (text_x + 120, rect.y - 1))


class ChartRenderer:
    """Scrolling time-series charts."""

    CHART_SPECS = (
        ("knowledge", "Knowledge", "bar_k"),
        ("engagement", "Engagement", "bar_e"),
        ("frustration", "Frustration", "bar_f"),
        ("confusion", "Confusion", "bar_c"),
        ("boredom", "Boredom", "bar_b"),
        ("reward", "Reward", "accent"),
        ("cumulative_reward", "Cumulative Reward", "highlight"),
    )

    def __init__(self, window_size: int = 120) -> None:
        self.window_size = window_size

    def draw_grid(
        self,
        surf: pygame.Surface,
        rect: pygame.Rect,
        engine: CSVReplayEngine,
        index: int,
        theme: Theme,
        fonts: FontBundle,
        presentation: bool,
    ) -> None:
        """Draw all charts in a grid layout."""
        cols = 4
        rows = 2
        pad = 8 if presentation else 6
        chart_w = (rect.width - pad * (cols + 1)) // cols
        chart_h = (rect.height - pad * (rows + 1)) // rows

        for i, (column, title, color_key) in enumerate(self.CHART_SPECS):
            row_i, col_i = divmod(i, cols)
            x = rect.x + pad + col_i * (chart_w + pad)
            y = rect.y + pad + row_i * (chart_h + pad)
            chart_rect = pygame.Rect(x, y, chart_w, chart_h)
            color = getattr(theme, color_key)
            self._draw_series(
                surf,
                chart_rect,
                engine,
                index,
                column,
                title,
                color,
                theme,
                fonts.sm,
            )

        # Emotion timeline intentionally removed for cleaner layout.

    def _draw_series(
        self,
        surf: pygame.Surface,
        rect: pygame.Rect,
        engine: CSVReplayEngine,
        index: int,
        column: str,
        title: str,
        color: Color,
        theme: Theme,
        font: pygame.font.Font,
    ) -> None:
        """Draw one scrolling line chart."""
        draw_panel(surf, rect, theme, radius=6)
        surf.blit(font.render(title, True, theme.dim), (rect.x + 8, rect.y + 4))

        start = max(0, index - self.window_size + 1)
        values = engine.slice_array(column, start, index)
        if len(values) < 2:
            return

        plot = pygame.Rect(rect.x + 8, rect.y + 22, rect.width - 16, rect.height - 30)
        pygame.draw.rect(surf, theme.chart_grid, plot, 1, border_radius=3)

        if column == "reward":
            mn, mx = float(np.min(values)), float(np.max(values))
            span = max(mx - mn, 0.05)
            zero_y = plot.bottom - int((0 - mn) / span * (plot.height - 2)) - 1
            pygame.draw.line(
                surf,
                theme.dim,
                (plot.x, zero_y),
                (plot.right, zero_y),
                1,
            )
        elif column == "cumulative_reward":
            mn, mx = float(np.min(values)), float(np.max(values))
            span = max(mx - mn, 0.1)
        else:
            mn, mx = 0.0, 1.0
            span = max(mx - mn, 1e-6)

        points: List[Tuple[int, int]] = []
        for i, val in enumerate(values):
            px = plot.x + int(i / max(len(values) - 1, 1) * (plot.width - 1))
            py = plot.bottom - int((float(val) - mn) / span * (plot.height - 2)) - 1
            points.append((px, py))

        if len(points) >= 2:
            pygame.draw.lines(surf, color, False, points, 2)
        pygame.draw.circle(surf, color, points[-1], 3)

    def draw_emotion_timeline(
        self,
        surf: pygame.Surface,
        rect: pygame.Rect,
        engine: CSVReplayEngine,
        index: int,
        theme: Theme,
        font: pygame.font.Font,
    ) -> None:
        """Draw color-coded emotion timeline."""
        draw_panel(surf, rect, theme, radius=6)
        surf.blit(font.render("Emotion Timeline", True, theme.dim), (rect.x + 8, rect.y + 4))
        start = max(0, index - self.window_size + 1)
        plot = pygame.Rect(rect.x + 8, rect.y + 22, rect.width - 16, rect.height - 30)
        count = index - start + 1
        segment_w = max(1, plot.width // max(count, 1))

        for i in range(start, index + 1):
            row = engine.row(i)
            color = theme.emotion_colors.get(row.emotion_name, theme.text)
            offset = i - start
            x = plot.x + offset * segment_w
            seg = pygame.Rect(x, plot.y, max(1, segment_w), plot.height)
            pygame.draw.rect(surf, color, seg)


class EmotionRenderer:
    """Emotion state visualization."""

    def draw(
        self,
        surf: pygame.Surface,
        rect: pygame.Rect,
        row: ReplayRow,
        engine: CSVReplayEngine,
        index: int,
        theme: Theme,
        fonts: FontBundle,
        history: Sequence[str],
    ) -> None:
        """Draw emotion panel with history and statistics."""
        draw_panel(surf, rect, theme)
        surf.blit(fonts.sm.render("EMOTION STATE", True, theme.dim), (rect.x + 12, rect.y + 10))

        prev_row = engine.row(index - 1) if index > 0 else None
        face_emotion = self._rising_emotion(row, prev_row)
        avatar_rect = pygame.Rect(rect.x + rect.width // 2 - 48, rect.y + 30, 96, 96)
        self._draw_emotion_avatar(surf, avatar_rect, face_emotion, theme)

        if row.persistent_flag:
            badge = fonts.sm.render(
                "PERSISTENT FRUSTRATION",
                True,
                theme.danger,
            )
            surf.blit(badge, (rect.x + 12, rect.y + rect.height - 28))

    def _draw_pie(
        self,
        surf: pygame.Surface,
        rect: pygame.Rect,
        counts: Dict[str, int],
        theme: Theme,
    ) -> None:
        """Draw emotion distribution pie chart."""
        total = sum(counts.values())
        center = rect.center
        radius = rect.width // 2
        if total <= 0:
            pygame.draw.circle(surf, theme.progress_bg, center, radius)
            return

        start_angle = 0.0
        for name in EMOTION_NAMES:
            count = counts.get(name, 0)
            if count <= 0:
                continue
            sweep = 360.0 * count / total
            color = theme.emotion_colors[name]
            arc_rect = pygame.Rect(center[0] - radius, center[1] - radius, radius * 2, radius * 2)
            end_angle = start_angle + sweep
            steps = max(3, int(sweep / 8))
            points = [center]
            for step in range(steps + 1):
                angle = math.radians(start_angle + sweep * step / steps)
                points.append(
                    (
                        center[0] + int(math.cos(angle) * radius),
                        center[1] + int(math.sin(angle) * radius),
                    )
                )
            if len(points) > 2:
                pygame.draw.polygon(surf, color, points)
            start_angle = end_angle

        inner = max(4, radius // 2)
        pygame.draw.circle(surf, theme.panel, center, inner)

    def _rising_emotion(self, row: ReplayRow, prev_row: Optional[ReplayRow]) -> str:
        """Pick face emotion from the strongest positive change since last step."""
        if prev_row is None:
            return row.emotion_name

        deltas = {
            "engaged": row.engagement - prev_row.engagement,
            "frustrated": row.frustration - prev_row.frustration,
            "confused": row.confusion - prev_row.confusion,
            "bored": row.boredom - prev_row.boredom,
        }
        rising = {k: v for k, v in deltas.items() if v > 1e-6}
        if rising:
            return max(rising, key=rising.get)
        return row.emotion_name

    def _draw_emotion_avatar(
        self,
        surf: pygame.Surface,
        rect: pygame.Rect,
        emotion_name: str,
        theme: Theme,
    ) -> None:
        """Draw a student-like avatar with emotion-specific face."""
        card = pygame.Rect(rect.x - 4, rect.y - 4, rect.width + 8, rect.height + 8)
        pygame.draw.rect(surf, theme.panel_light, card, border_radius=10)

        center_x = rect.centerx
        center_y = rect.centery
        face_radius = min(rect.width, rect.height) // 2 - 10
        skin = (250, 225, 199)
        hair = (115, 86, 68)
        eye = (56, 63, 90)
        mouth = theme.emotion_colors.get(emotion_name, theme.text)

        pygame.draw.circle(surf, hair, (center_x, center_y - 20), face_radius - 2)
        pygame.draw.circle(surf, skin, (center_x, center_y), face_radius)
        left_eye = (center_x - 11, center_y - 5)
        right_eye = (center_x + 11, center_y - 5)

        if emotion_name == "engaged":
            # Happy face
            pygame.draw.circle(surf, eye, left_eye, 3)
            pygame.draw.circle(surf, eye, right_eye, 3)
            pygame.draw.arc(surf, mouth, pygame.Rect(center_x - 16, center_y - 2, 32, 18), 0.2, 2.95, 3)
            pygame.draw.line(surf, eye, (center_x - 16, center_y - 14), (center_x - 6, center_y - 12), 2)
            pygame.draw.line(surf, eye, (center_x + 6, center_y - 12), (center_x + 16, center_y - 14), 2)
        elif emotion_name == "confused":
            # Confused face with tilted brows and zigzag mouth
            pygame.draw.circle(surf, eye, left_eye, 3)
            pygame.draw.circle(surf, eye, right_eye, 3)
            pygame.draw.line(surf, eye, (center_x - 17, center_y - 14), (center_x - 6, center_y - 11), 2)
            pygame.draw.line(surf, eye, (center_x + 6, center_y - 11), (center_x + 17, center_y - 14), 2)
            points = [
                (center_x - 14, center_y + 10),
                (center_x - 6, center_y + 14),
                (center_x + 2, center_y + 9),
                (center_x + 12, center_y + 13),
            ]
            pygame.draw.lines(surf, mouth, False, points, 3)
        elif emotion_name == "bored":
            # Bored face with half-closed eyes and flat mouth
            pygame.draw.line(surf, eye, (center_x - 15, center_y - 5), (center_x - 7, center_y - 5), 3)
            pygame.draw.line(surf, eye, (center_x + 7, center_y - 5), (center_x + 15, center_y - 5), 3)
            pygame.draw.line(surf, mouth, (center_x - 12, center_y + 10), (center_x + 12, center_y + 10), 3)
        else:  # frustrated
            # Frustrated face with angry brows and frown
            pygame.draw.circle(surf, eye, left_eye, 3)
            pygame.draw.circle(surf, eye, right_eye, 3)
            pygame.draw.line(surf, eye, (center_x - 17, center_y - 10), (center_x - 6, center_y - 14), 2)
            pygame.draw.line(surf, eye, (center_x + 6, center_y - 14), (center_x + 17, center_y - 10), 2)
            pygame.draw.arc(
                surf,
                mouth,
                pygame.Rect(center_x - 15, center_y + 6, 30, 14),
                3.35,
                6.05,
                3,
            )

        pygame.draw.circle(surf, theme.chip_border, (center_x, center_y), face_radius, 2)


class ActionRenderer:
    """Action visualization panel."""

    def draw(
        self,
        surf: pygame.Surface,
        rect: pygame.Rect,
        row: ReplayRow,
        prev_action: Optional[str],
        engine: CSVReplayEngine,
        index: int,
        theme: Theme,
        fonts: FontBundle,
        history: Sequence[str],
    ) -> None:
        """Draw current action, history, and frequency."""
        draw_panel(surf, rect, theme)
        surf.blit(fonts.sm.render("INTERVENTION", True, theme.dim), (rect.x + 12, rect.y + 10))

        current = fonts.lg.render(row.action_name.replace("_", " "), True, theme.highlight)
        surf.blit(current, (rect.x + 12, rect.y + 32))

        counts = engine.action_counts_up_to(index)
        total = max(sum(counts.values()), 1)

        bar_y = rect.y + 78
        max_count = max(counts.values()) if counts else 1
        label_col_w = 124
        bar_w = rect.width - 24 - label_col_w
        for i, action in enumerate(sorted(counts, key=counts.get, reverse=True)[:6]):
            count = counts[action]
            width = int(bar_w * count / max_count)
            y = bar_y + i * 18
            pygame.draw.rect(
                surf,
                theme.progress_bg,
                pygame.Rect(rect.x + 12, y, bar_w, 12),
                border_radius=3,
            )
            if width > 0:
                pygame.draw.rect(
                    surf,
                    theme.accent if action == row.action_name else theme.dim,
                    pygame.Rect(rect.x + 12, y, width, 12),
                    border_radius=3,
                )
            action_text = action.replace("_", " ")
            label = fonts.sm.render(f"{action_text[:14]}", True, theme.text)
            label_x = rect.x + 12 + bar_w + 8
            surf.blit(label, (label_x, y - 1))

        # Action timeline intentionally removed for cleaner layout.


class StatisticsPanel:
    """Final analytics screen."""

    def draw(
        self,
        surf: pygame.Surface,
        rect: pygame.Rect,
        summary: SessionSummary,
        theme: Theme,
        fonts: FontBundle,
    ) -> None:
        """Render end-of-session analytics."""
        overlay = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        overlay.fill((8, 8, 18, 230))
        surf.blit(overlay, rect.topleft)

        draw_panel(surf, rect, theme, radius=12, fill=theme.panel_light)
        title = fonts.title.render("Session Summary", True, theme.highlight)
        surf.blit(title, (rect.x + 24, rect.y + 20))

        metrics = [
            ("Final knowledge", f"{summary.final_knowledge:.3f}"),
            ("Learning gain", f"{summary.learning_gain:+.3f}"),
            ("Average reward", f"{summary.average_reward:.4f}"),
            ("Total steps", str(summary.total_steps)),
            ("Episodes", str(summary.total_episodes)),
            ("Interventions", str(summary.total_interventions)),
            ("Peak engagement", f"{summary.peak_engagement:.3f}"),
            ("Peak knowledge", f"{summary.peak_knowledge:.3f}"),
            ("Persistent frustration steps", str(summary.persistent_frustration_steps)),
            ("Most-used intervention", summary.most_used_action.replace("_", " ")),
        ]

        col_w = rect.width // 2 - 32
        row_h = 48
        metrics_y = rect.y + 64
        for i, (label, value) in enumerate(metrics):
            col = i // 5
            row = i % 5
            x = rect.x + 24 + col * col_w
            y = metrics_y + row * row_h
            surf.blit(fonts.sm.render(label, True, theme.dim), (x, y))
            surf.blit(fonts.md.render(value, True, theme.text), (x, y + 22))

        act_x = rect.x + 24
        act_y = metrics_y + 5 * row_h + 20
        surf.blit(fonts.sm.render("Action distribution", True, theme.dim), (act_x, act_y))
        for i, (action, count) in enumerate(
            sorted(summary.action_counts.items(), key=lambda kv: kv[1], reverse=True)[:8]
        ):
            txt = fonts.sm.render(
                f"{action.replace('_', ' ')}: {count}",
                True,
                theme.text,
            )
            surf.blit(txt, (act_x, act_y + 22 + i * 20))


class DashboardRenderer:
    """Main dashboard layout and metrics."""

    def __init__(self, presentation: bool = False) -> None:
        self.presentation = presentation
        self.theme = Theme()
        self.fonts = FontBundle(presentation)
        self.chart_renderer = ChartRenderer(window_size=150 if presentation else 120)
        self.emotion_renderer = EmotionRenderer()
        self.action_renderer = ActionRenderer()
        self.stats_panel = StatisticsPanel()

    @property
    def window_size(self) -> Tuple[int, int]:
        """Return window dimensions."""
        return (1920, 1080) if self.presentation else (1360, 780)

    def update_display_state(
        self,
        state: ReplayState,
        row: ReplayRow,
        dt: float,
    ) -> None:
        """Smoothly interpolate bar values toward the current row."""
        t = min(1.0, dt * 12.0)
        state.display_knowledge = lerp(state.display_knowledge, row.knowledge, t)
        state.display_engagement = lerp(state.display_engagement, row.engagement, t)
        state.display_frustration = lerp(state.display_frustration, row.frustration, t)
        state.display_confusion = lerp(state.display_confusion, row.confusion, t)
        state.display_boredom = lerp(state.display_boredom, row.boredom, t)

    def draw(
        self,
        surf: pygame.Surface,
        engine: CSVReplayEngine,
        controller: PlaybackController,
        state: ReplayState,
        prev_action: Optional[str],
        emotion_history: Sequence[str],
        action_history: Sequence[str],
        input_text: str = "",
        input_mode: Optional[str] = None,
        show_debug: bool = True,
    ) -> None:
        """Render the full dashboard."""
        row = controller.current_row()
        w, h = surf.get_size()
        theme = self.theme
        surf.fill(theme.bg)

        header_h = 44 if self.presentation else 36
        controls_h = 22
        metrics_h = 50
        caption_h = 26
        top_gap = 10
        bottom_safe = 160
        pygame.draw.rect(surf, theme.panel, pygame.Rect(0, 0, w, header_h))
        header = self.fonts.md.render(
            f"{row.algorithm}  |  Episode {row.episode}  |  Step {row.step}  |  "
            f"Cum. reward {row.cumulative_reward:.2f}  |  "
            f"{'PERSISTENT' if row.persistent_flag else 'Stable'}",
            True,
            theme.dim if not row.persistent_flag else theme.danger,
        )
        surf.blit(header, (12, 8 if self.presentation else 10))

        status_text = controller.status.name
        speed_text = f"{controller.speed}x"
        transport = self.fonts.sm.render(
            f"{status_text}  {speed_text}  [{controller.index + 1}/{engine.length}]",
            True,
            theme.text,
        )
        surf.blit(transport, (w - transport.get_width() - 12, 12))

        controls_y = header_h + 4
        metrics_rect = pygame.Rect(8, controls_y + controls_h + 4, w - 16, metrics_h)
        self._draw_metric_cards(surf, metrics_rect, row, theme)

        caption_rect = pygame.Rect(8, metrics_rect.bottom + 4, w - 16, caption_h)
        caption = state.caption or row.explainer_reason[:100]
        if caption:
            pygame.draw.rect(surf, theme.annotation_bg, caption_rect, border_radius=6)
            surf.blit(self.fonts.sm.render(caption, True, theme.highlight), (16, caption_rect.y + 4))

        top_y = caption_rect.bottom + top_gap
        left_w = 330 if self.presentation else 320
        right_w = 340 if self.presentation else 300
        mid_x = left_w + 16
        mid_w = w - left_w - right_w - 32

        emotion_rect = pygame.Rect(8, top_y, left_w, 220 if self.presentation else 190)
        self.emotion_renderer.draw(
            surf, emotion_rect, row, engine, controller.index, theme, self.fonts, emotion_history
        )

        state_rect = pygame.Rect(8, emotion_rect.bottom + 8, left_w, 220 if self.presentation else 190)
        self._draw_student_state(surf, state_rect, state, theme)

        chart_rect = pygame.Rect(mid_x, top_y, mid_w, h - top_y - bottom_safe)
        self.chart_renderer.draw_grid(
            surf, chart_rect, engine, controller.index, theme, self.fonts, self.presentation
        )

        action_rect = pygame.Rect(w - right_w - 8, top_y, right_w, h - top_y - bottom_safe)
        self.action_renderer.draw(
            surf,
            action_rect,
            row,
            prev_action,
            engine,
            controller.index,
            theme,
            self.fonts,
            action_history,
        )

        reward_rect = pygame.Rect(mid_x, h - bottom_safe + 8, w - mid_x - 8, 68)
        prev_row = engine.row(controller.index - 1) if controller.index > 0 else row
        self._draw_reward_substitutions(surf, reward_rect, row, prev_row, theme)

        if not self.presentation or show_debug:
            controls = (
                "Space: Play/Pause  |  Left/Right: Frame  |  R: Restart  |  S: Stop  |  "
                "+/-: Speed  |  E: Episode  |  T: Step  |  A: Analytics  |  Esc: Quit"
            )
            dock_safe_y = h - 72
            surf.blit(self.fonts.sm.render(controls, True, theme.dim), (8, dock_safe_y))

        if input_mode:
            prompt = f"Jump to {input_mode}: {input_text}_"
            box = pygame.Rect(w // 2 - 180, h // 2 - 20, 360, 40)
            draw_panel(surf, box, theme, fill=theme.panel_light)
            surf.blit(self.fonts.md.render(prompt, True, theme.highlight), (box.x + 12, box.y + 10))

        if state.show_analytics or controller.status == PlaybackStatus.FINISHED:
            self.stats_panel.draw(
                surf,
                pygame.Rect(80, 80, w - 160, h - 160),
                engine.summary,
                theme,
                self.fonts,
            )

    def _draw_student_state(
        self,
        surf: pygame.Surface,
        rect: pygame.Rect,
        state: ReplayState,
        theme: Theme,
    ) -> None:
        """Draw student state progress bars."""
        draw_panel(surf, rect, theme)
        surf.blit(self.fonts.sm.render("STUDENT STATE", True, theme.dim), (rect.x + 12, rect.y + 10))
        specs = (
            ("Knowledge", state.display_knowledge, theme.bar_k),
            ("Engagement", state.display_engagement, theme.bar_e),
            ("Frustration", state.display_frustration, theme.bar_f),
            ("Confusion", state.display_confusion, theme.bar_c),
            ("Boredom", state.display_boredom, theme.bar_b),
        )
        bar_w = rect.width - 232
        for i, (label, value, color) in enumerate(specs):
            bar_rect = pygame.Rect(rect.x + 12, rect.y + 34 + i * 34, bar_w, 18)
            draw_progress_bar(surf, self.fonts.sm, label, value, color, bar_rect, theme)

    def _draw_metric_cards(
        self,
        surf: pygame.Surface,
        rect: pygame.Rect,
        row: ReplayRow,
        theme: Theme,
    ) -> None:
        """Draw reward and step metric cards."""
        card_w = (rect.width - 8) // 2
        cards = (
            ("Step reward", f"{row.reward:.4f}"),
            ("Cumulative", f"{row.cumulative_reward:.2f}"),
        )
        for i, (label, value) in enumerate(cards):
            x = rect.x + i * (card_w + 8)
            card = pygame.Rect(x, rect.y, card_w, rect.height)
            draw_panel(surf, card, theme)
            value_font = self.fonts.md
            label_font = self.fonts.sm
            value_surface = value_font.render(value, True, theme.highlight)
            label_surface = label_font.render(label, True, theme.dim)

            inner_left = x + 10
            inner_top = rect.y + 6
            inner_bottom = rect.bottom - 6

            # Keep both lines fully visible inside short cards.
            surf.blit(value_surface, (inner_left, inner_top))
            label_y = min(inner_bottom - label_surface.get_height(), inner_top + value_surface.get_height() + 2)
            surf.blit(label_surface, (inner_left, label_y))

    def _draw_reward_substitutions(
        self,
        surf: pygame.Surface,
        rect: pygame.Rect,
        row: ReplayRow,
        prev_row: ReplayRow,
        theme: Theme,
    ) -> None:
        """Show reward equation with current-step substitutions."""
        draw_panel(surf, rect, theme)
        weights = config.get_reward_weights()
        wk = float(weights.get("wk", 0.5))
        we = float(weights.get("we", 0.2))
        wf = float(weights.get("wf", 0.15))
        wb = float(weights.get("wb", 0.10))
        wc = float(weights.get("wc", 0.05))

        delta_k_norm = float(
            np.clip(
                (row.knowledge - prev_row.knowledge) / (1.0 - prev_row.knowledge + 1e-8),
                -1.0,
                1.0,
            )
        )
        k_term = wk * delta_k_norm
        e_term = we * row.engagement
        f_term = -wf * row.frustration
        b_term = -wb * row.boredom
        c_term = -wc * row.confusion
        base_total = k_term + e_term + f_term + b_term + c_term

        x = rect.x + 10
        y = rect.y + 8
        title = self.fonts.sm.render("Reward equation (current step substitution)", True, theme.dim)
        surf.blit(title, (x, y))
        form = self.fonts.sm.render("r = wk*dK + we*E - wf*F - wb*B - wc*C", True, theme.text)
        surf.blit(form, (x, y + 18))
        subs_text = (
            f"= {wk:.2f}*({delta_k_norm:+.3f}) + {we:.2f}*({row.engagement:.3f}) - "
            f"{wf:.2f}*({row.frustration:.3f}) - {wb:.2f}*({row.boredom:.3f}) - "
            f"{wc:.2f}*({row.confusion:.3f}) = {base_total:+.3f}"
        )
        subs = self.fonts.sm.render(subs_text, True, theme.highlight)
        surf.blit(subs, (x, y + 36))


class LiveDashboardRenderer:
    """Compact live monitor preserving original functionality."""

    def __init__(self) -> None:
        self.theme = Theme()
        self.fonts = FontBundle(False)
        self.reward_history: Deque[float] = deque(maxlen=60)

    def draw_waiting(self, surf: pygame.Surface, message: str) -> None:
        """Draw waiting screen when CSV is not yet available."""
        surf.fill(self.theme.bg)
        msg = self.fonts.md.render(message, True, self.theme.dim)
        surf.blit(msg, (20, surf.get_height() // 2))

    def draw_row(self, surf: pygame.Surface, row: ReplayRow) -> None:
        """Draw the latest live row."""
        theme = self.theme
        surf.fill(theme.bg)
        self.reward_history.append(row.cumulative_reward)

        pygame.draw.rect(surf, theme.panel, (0, 0, 660, 32))
        header = self.fonts.md.render(
            f"{row.algorithm}  |  Episode {row.episode}  |  Step {row.step}  "
            f"|  {'PERSISTENT' if row.persistent_flag else ''}",
            True,
            theme.dim,
        )
        surf.blit(header, (10, 8))

        emo_color = theme.emotion_colors.get(row.emotion_name, theme.text)
        pygame.draw.rect(surf, theme.panel, (10, 40, 140, 90), border_radius=8)
        surf.blit(self.fonts.lg.render(row.emotion_name.upper(), True, emo_color), (18, 58))
        surf.blit(self.fonts.sm.render("FER OUTPUT", True, theme.dim), (18, 44))

        pygame.draw.rect(surf, theme.panel, (160, 40, 490, 90), border_radius=8)
        surf.blit(self.fonts.sm.render("STUDENT STATE", True, theme.dim), (168, 44))
        bars = [
            ("knowledge", row.knowledge, theme.bar_k),
            ("engagement", row.engagement, theme.bar_e),
            ("frustration", row.frustration, theme.bar_f),
            ("confusion", row.confusion, theme.bar_c),
            ("boredom", row.boredom, theme.bar_b),
        ]
        for i, (name, val, col) in enumerate(bars):
            bar_rect = pygame.Rect(168, 56 + i * 17, 240, 16)
            draw_progress_bar(surf, self.fonts.sm, name, val, col, bar_rect, theme)

        pygame.draw.rect(surf, theme.panel, (10, 138, 640, 40), border_radius=8)
        surf.blit(self.fonts.sm.render("ACTIONS", True, theme.dim), (18, 142))
        for i, name in enumerate(ACTIONS):
            x = 10 + i * 63
            active = i == row.action_id
            bg = theme.active_chip if active else theme.bg
            bd = theme.accent if active else (50, 50, 90)
            pygame.draw.rect(surf, bg, (x, 154, 60, 18), border_radius=4)
            pygame.draw.rect(surf, bd, (x, 154, 60, 18), 1, border_radius=4)
            c = theme.highlight if active else theme.dim
            short = name.replace("_", " ")[:8]
            surf.blit(self.fonts.sm.render(short, True, c), (x + 2, 157))

        for i, (label, val) in enumerate([
            ("Step reward", f"{row.reward:.3f}"),
            ("Cumulative", f"{row.cumulative_reward:.2f}"),
            ("Episode", str(row.episode)),
        ]):
            x = 10 + i * 216
            pygame.draw.rect(surf, theme.panel, (x, 182, 210, 48), border_radius=8)
            surf.blit(self.fonts.lg.render(val, True, theme.highlight), (x + 8, 192))
            surf.blit(self.fonts.sm.render(label, True, theme.dim), (x + 8, 218))

        pygame.draw.rect(surf, theme.panel, (10, 238, 640, 32), border_radius=8)
        reason = row.explainer_reason[:80]
        surf.blit(self.fonts.sm.render("WHY: " + reason, True, theme.text), (18, 248))

        pygame.draw.rect(surf, theme.panel, (10, 278, 640, 90), border_radius=8)
        surf.blit(
            self.fonts.sm.render("CUMULATIVE REWARD (last 60 steps)", True, theme.dim),
            (18, 284),
        )
        if len(self.reward_history) > 1:
            mn = min(self.reward_history)
            mx = max(self.reward_history)
            rng = max(mx - mn, 0.1)
            pts = [
                (
                    18 + int(i / (len(self.reward_history) - 1) * 624),
                    358 - int((v - mn) / rng * 60),
                )
                for i, v in enumerate(self.reward_history)
            ]
            pygame.draw.lines(surf, theme.accent, False, pts, 2)
