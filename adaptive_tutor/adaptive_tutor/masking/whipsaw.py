"""Anti-whipsaw stabiliser for pacing actions (Phase 7).

Detects rapid ``easier_problem`` / ``harder_problem`` oscillation in a
fixed window and temporarily removes **both** pace actions when churn
exceeds a threshold.
"""

from __future__ import annotations

from collections import deque

PACE_ACTIONS = frozenset({"easier_problem", "harder_problem"})


class WhipsawTracker:
    """Track recent executed actions; filter pacing subset."""

    __slots__ = ("_max_pace_switches", "recent", "window")

    def __init__(self, *, window: int = 4, max_pace_switches: int = 2) -> None:
        self.window = max(2, int(window))
        self._max_pace_switches = max(1, int(max_pace_switches))
        self.recent: deque[str] = deque(maxlen=self.window)

    def reset(self) -> None:
        self.recent.clear()

    def _pace_switches(self) -> int:
        """Count adjacent changes between easier/harder in recent window."""
        seq = [a for a in self.recent if a in PACE_ACTIONS]
        if len(seq) < 2:
            return 0
        switches = 0
        for i in range(1, len(seq)):
            if seq[i] != seq[i - 1]:
                switches += 1
        return switches

    def filter(self, actions: frozenset[str]) -> tuple[frozenset[str], tuple[str, ...]]:
        """If pace churn is high, strip both pace actions from ``actions``."""
        if self._pace_switches() < self._max_pace_switches:
            return actions, ()
        dropped = PACE_ACTIONS & actions
        if not dropped:
            return actions, ()
        return actions - PACE_ACTIONS, ("anti_whipsaw:high_pace_churn",)

    def opposite_pair_filter(self, actions: frozenset[str]) -> tuple[frozenset[str], tuple[str, ...]]:
        """Remove immediate reversal of the last pace action (one-step deadband)."""
        if not self.recent:
            return actions, ()
        last = self.recent[-1]
        out = set(actions)
        reasons: list[str] = []
        if last == "easier_problem" and "harder_problem" in out:
            out.remove("harder_problem")
            reasons.append("anti_whipsaw:no_immediate_harder_after_easier")
        elif last == "harder_problem" and "easier_problem" in out:
            out.remove("easier_problem")
            reasons.append("anti_whipsaw:no_immediate_easier_after_harder")
        return frozenset(out), tuple(reasons)

    def record(self, executed_action: str) -> None:
        self.recent.append(executed_action)
