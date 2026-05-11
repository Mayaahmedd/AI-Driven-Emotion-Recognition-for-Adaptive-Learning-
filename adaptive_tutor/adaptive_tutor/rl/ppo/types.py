"""Curriculum-level types for PPO (Phase 9)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CurriculumAction:
    """One PPO decision at concept boundaries."""

    concept_index: int
    pacing_factor: float
    """Scales ``TutoringEnvironment.max_episode_steps`` for the segment cap."""
    difficulty_bias: float
    """In ``[-1, 1]``; positive slightly lowers the mastery bar for this segment."""
