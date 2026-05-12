"""Offline / batch RL helpers (no online exploration in training loops here)."""

from adaptive_tutor.rl.offline.cql import CQLDiscreteTrainer, cql_discrete_penalty

__all__ = ["CQLDiscreteTrainer", "cql_discrete_penalty"]
