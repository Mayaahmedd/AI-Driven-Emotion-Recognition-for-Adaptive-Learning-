"""Factorised Double DQN (Phase 8)."""

from adaptive_tutor.rl.dqn.decode import (
    STRATEGY_DIM,
    DIFFICULTY_DIM,
    PROGRESSION_DIM,
    assistments_to_meso,
    decode_meso_to_assistments,
)
from adaptive_tutor.rl.dqn.dqn import DoubleDQNAgent, q_sum_gather
from adaptive_tutor.rl.dqn.model import FactorisedDQN
from adaptive_tutor.rl.dqn.trainer import (
    collect_transition,
    soft_update_target,
    train_double_dqn_batch,
)

__all__ = [
    "STRATEGY_DIM",
    "DIFFICULTY_DIM",
    "PROGRESSION_DIM",
    "DoubleDQNAgent",
    "FactorisedDQN",
    "assistments_to_meso",
    "collect_transition",
    "decode_meso_to_assistments",
    "q_sum_gather",
    "soft_update_target",
    "train_double_dqn_batch",
]
