"""Visual theme constants for the replay viewer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from RL_Module.mdp_definition import EMOTIONS

Color = Tuple[int, int, int]


@dataclass(frozen=True)
class Theme:
    """Color palette and layout constants."""

    bg: Color = (246, 244, 252)
    panel: Color = (236, 241, 251)
    panel_light: Color = (228, 236, 248)
    text: Color = (53, 60, 90)
    dim: Color = (103, 112, 148)
    accent: Color = (138, 132, 219)
    highlight: Color = (93, 126, 194)
    success: Color = (125, 208, 167)
    warning: Color = (243, 199, 116)
    danger: Color = (229, 140, 139)
    bar_k: Color = (133, 167, 227)
    bar_e: Color = (123, 208, 189)
    bar_f: Color = (232, 147, 140)
    bar_c: Color = (244, 202, 123)
    bar_b: Color = (176, 170, 220)
    chart_grid: Color = (198, 203, 224)
    annotation_bg: Color = (223, 217, 244)
    progress_bg: Color = (213, 220, 239)
    active_chip: Color = (225, 220, 246)
    chip_border: Color = (167, 160, 218)

    emotion_colors: Dict[str, Color] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "emotion_colors",
            {
                "confused": self.warning,
                "bored": (136, 136, 187),
                "frustrated": self.danger,
                "engaged": self.success,
            },
        )


PRESENTATION_THEME = Theme(
    bg=(244, 242, 250),
    panel=(232, 238, 249),
    text=(53, 60, 90),
    dim=(103, 112, 148),
)

DEFAULT_THEME = Theme()

EMOTION_NAMES = list(EMOTIONS)

KNOWLEDGE_THRESHOLDS = (0.3, 0.5, 0.7, 0.9)
HIGH_REWARD_THRESHOLD = 0.04
PLAYBACK_SPEEDS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
