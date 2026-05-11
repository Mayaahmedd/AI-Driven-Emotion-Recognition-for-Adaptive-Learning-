"""Map factorised meso head indices to ASSISTments string actions.

Conventions (Phase 8)
---------------------
* **strategy** (``content_variant``): 0=explain, 1=hint, 2=encourage, 3=skip
* **difficulty** (``pace``): 0=easier, 1=same, 2=harder
* **progression** (``ui_variant``): 0=stay, 1=advance

``emotional_support`` stays 0 until a fourth head is wired.

The decoder is a **fixed priority list** (no learning) so thesis readers can
trace ``(strategy, difficulty, progression)`` to a single tutoring opcode.
"""

from __future__ import annotations

from adaptive_tutor.memory.providers.dataset_provider import ASSISTMENTS_ACTIONS

STRATEGY_DIM = 4
DIFFICULTY_DIM = 3
PROGRESSION_DIM = 2


def decode_meso_to_assistments(strategy: int, difficulty: int, progression: int) -> str:
    """Deterministic map to one of :data:`~adaptive_tutor.memory.providers.dataset_provider.ASSISTMENTS_ACTIONS`."""
    if not (0 <= strategy < STRATEGY_DIM):
        raise ValueError(f"strategy must be in [0, {STRATEGY_DIM}), got {strategy}")
    if not (0 <= difficulty < DIFFICULTY_DIM):
        raise ValueError(f"difficulty must be in [0, {DIFFICULTY_DIM}), got {difficulty}")
    if not (0 <= progression < PROGRESSION_DIM):
        raise ValueError(f"progression must be in [0, {PROGRESSION_DIM}), got {progression}")

    if progression == 1:
        return "advance_to_next_skill"
    if strategy == 0:
        return "retry_current_skill"
    if strategy == 1:
        return "give_hint"
    if strategy == 2:
        return "encouragement"
    if difficulty == 0:
        return "easier_problem"
    if difficulty == 2:
        return "harder_problem"
    return "retry_current_skill"


def assistments_to_meso(action: str) -> tuple[int, int, int]:
    """Inverse mapping for logging / behavioural cloning (fixed table)."""
    if action not in ASSISTMENTS_ACTIONS:
        raise ValueError(f"unknown action {action!r}")
    table: dict[str, tuple[int, int, int]] = {
        "advance_to_next_skill": (3, 1, 1),
        "retry_current_skill": (0, 1, 0),
        "give_hint": (1, 1, 0),
        "encouragement": (2, 1, 0),
        "easier_problem": (3, 0, 0),
        "harder_problem": (3, 2, 0),
    }
    return table[action]
