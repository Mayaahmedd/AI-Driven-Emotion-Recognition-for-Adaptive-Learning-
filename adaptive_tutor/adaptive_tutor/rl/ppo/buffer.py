"""Uniform storage for concept-level PPO transitions (Phase 9)."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class PPOStepRecord:
    """One curriculum decision and its Monte Carlo return."""

    obs_1d: torch.Tensor
    concept_index: int
    log_prob: float
    reward: float


class PPOConceptBuffer:
    """Append-only buffer; deterministic reads for reproducible PPO updates."""

    def __init__(self) -> None:
        self._rows: list[PPOStepRecord] = []

    def __len__(self) -> int:
        return len(self._rows)

    def append(self, row: PPOStepRecord) -> None:
        self._rows.append(row)

    def clear(self) -> None:
        self._rows.clear()

    def all_records(self) -> list[PPOStepRecord]:
        return list(self._rows)

    def first_n(self, n: int) -> list[PPOStepRecord]:
        """Stable slice (no shuffling) - useful for debugging / CI."""
        if n <= 0:
            return []
        return self._rows[:n]
