"""Per-action cooldowns (Phase 7).

Blocks spam (e.g. repeated ``give_hint``) using the **learner state's**
``timestep`` as a discrete clock so behaviour matches :class:`StateBuilder`.
"""

from __future__ import annotations

from collections.abc import Mapping

from adaptive_tutor.memory.providers.dataset_provider import ASSISTMENTS_ACTIONS


# Default cooldown lengths in **state timesteps** between consecutive uses.
DEFAULT_COOLDOWN_STEPS: dict[str, int] = {
    "give_hint": 1,
    "retry_current_skill": 2,
    "advance_to_next_skill": 3,
    "encouragement": 5,
    "easier_problem": 1,
    "harder_problem": 1,
}


class CooldownTracker:
    """Tracks last timestep each action was executed (post-step timestep)."""

    __slots__ = ("_cooldowns", "_last_used")

    def __init__(self, cooldowns: Mapping[str, int] | None = None) -> None:
        self._cooldowns: dict[str, int] = dict(DEFAULT_COOLDOWN_STEPS)
        if cooldowns is not None:
            self._cooldowns.update({k: int(v) for k, v in cooldowns.items()})
        for a in ASSISTMENTS_ACTIONS:
            self._cooldowns.setdefault(a, 0)
        self._last_used: dict[str, int] = {}

    def reset(self) -> None:
        self._last_used.clear()

    def can_use(self, action: str, current_timestep: int) -> bool:
        """True if ``current_timestep - last_use >= cooldown`` (or never used)."""
        cd = int(self._cooldowns.get(action, 0))
        if cd <= 0:
            return True
        last = self._last_used.get(action)
        if last is None:
            return True
        return (current_timestep - last) >= cd

    def filter(self, actions: frozenset[str], current_timestep: int) -> tuple[frozenset[str], tuple[str, ...]]:
        allowed = frozenset(a for a in actions if self.can_use(a, current_timestep))
        dropped = sorted(actions - allowed)
        reasons = tuple(f"cooldown:{a}" for a in dropped) if dropped else ()
        return allowed, reasons

    def record(self, action: str, post_step_timestep: int) -> None:
        """Call after the environment applies the action and observed outcomes."""
        self._last_used[action] = int(post_step_timestep)
