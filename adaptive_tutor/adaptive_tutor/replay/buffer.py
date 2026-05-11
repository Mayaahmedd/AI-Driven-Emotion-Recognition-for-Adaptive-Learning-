"""Uniform replay buffer (Phase 5).

Stores frozen :class:`~adaptive_tutor.state.state.Transition` records
with uniform random sampling. No priorities, no PER - suitable for
small-scale bachelor experiments and simulator data collection.
"""

from __future__ import annotations

import random
from collections import deque

from adaptive_tutor.state.state import Transition


class UniformReplayBuffer:
    """Fixed-capacity FIFO with :meth:`sample` without replacement semantics via ``random.sample``."""

    def __init__(self, capacity: int, *, seed: int | None = None) -> None:
        self._capacity = max(1, int(capacity))
        self._buf: deque[Transition] = deque(maxlen=self._capacity)
        self._rng = random.Random(seed)

    def add(self, transition: Transition) -> None:
        self._buf.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        n = len(self._buf)
        if n == 0:
            return []
        k = min(int(batch_size), n)
        return self._rng.sample(list(self._buf), k)

    def __len__(self) -> int:
        return len(self._buf)
