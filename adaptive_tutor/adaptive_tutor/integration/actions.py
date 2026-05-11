"""Map ASSISTments string actions to :class:`CompositeAction` for replay / RL."""

from __future__ import annotations

from adaptive_tutor.core.types import CompositeAction, MesoAction
from adaptive_tutor.memory.providers.dataset_provider import ASSISTMENTS_ACTIONS


def assistments_action_to_composite(action: str) -> CompositeAction:
    """Encode the discrete ASSISTments label in ``MesoAction.content_variant``.

    Other meso heads stay at 0 until Phase 8 factorised DQN is wired.
    This keeps :class:`~adaptive_tutor.state.state.Transition` compatible
    with the existing dataclass without a parallel action type.
    """
    if action not in ASSISTMENTS_ACTIONS:
        raise ValueError(f"unknown ASSISTments action {action!r}")
    idx = ASSISTMENTS_ACTIONS.index(action)
    return CompositeAction(macro=None, meso=MesoAction(idx, 0, 0, 0))
