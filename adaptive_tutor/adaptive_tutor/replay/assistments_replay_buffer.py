"""ASSISTments offline replay: trajectories from CSV, rewards from :class:`~adaptive_tutor.rewards.RewardEngine`."""

from __future__ import annotations

import torch

from adaptive_tutor.integration.actions import assistments_action_to_composite
from adaptive_tutor.memory.providers.base import BaseCurriculumProvider
from adaptive_tutor.memory.providers.dataset_provider import (
    DatasetCurriculumProvider,
    skill_stats_as_tuple,
)
from adaptive_tutor.replay.buffer import UniformReplayBuffer
from adaptive_tutor.rewards import RewardEngine, default_reward_engine
from adaptive_tutor.state.assistments_state import learner_state_from_assistments_dict
from adaptive_tutor.state.state import Transition


def _concept_index_safe(teacher: BaseCurriculumProvider, skill_slug: str) -> int:
    try:
        return int(teacher.concept_index(skill_slug))
    except KeyError:
        return 0


class ASSISTMENTSReplayBuffer:
    """Stores :class:`Transition` rows from ``DatasetCurriculumProvider.iter_transitions``.

    Rewards and ``r_components`` are materialised here via :class:`RewardEngine` so the
    dataset layer stays read-only. PPO offline rows reuse the same per-step returns with
    the *pre-action* state tensor and the teacher concept index for the logged skill.
    """

    def __init__(self, capacity: int = 200_000, *, seed: int | None = None) -> None:
        self._inner = UniformReplayBuffer(capacity, seed=seed)
        self._ppo_rows: list[tuple[torch.Tensor, int, float]] = []

    def load_from_provider(
        self,
        provider: DatasetCurriculumProvider,
        teacher: BaseCurriculumProvider,
        reward_engine: RewardEngine | None = None,
        *,
        num_concepts: int,
    ) -> int:
        """Append all CSV transitions; return count loaded."""
        engine = reward_engine or default_reward_engine()
        n = 0
        self._ppo_rows.clear()
        for raw in provider.iter_transitions():
            cidx = _concept_index_safe(teacher, raw.skill)
            st_stats = skill_stats_as_tuple(provider.stats(raw.skill))
            s = learner_state_from_assistments_dict(
                raw.state, concept_index=cidx, dataset_stats=st_stats
            )
            s_next = learner_state_from_assistments_dict(
                raw.next_state, concept_index=cidx, dataset_stats=st_stats
            )
            em = raw.next_state.get("emotion") or {}
            r, comps = engine.compute(
                correct=int(raw.correct),
                hint_count=int(raw.hint_count),
                emotion=em,
                action=raw.action,
            )
            a = assistments_action_to_composite(raw.action)
            tr = Transition(
                s=s,
                a=a,
                r=float(r),
                r_components=tuple(comps),
                s_next=s_next,
                done=bool(raw.done),
            )
            self._inner.add(tr)
            obs = s.to_tensor(num_concepts=num_concepts).detach().cpu()
            self._ppo_rows.append((obs, cidx, float(r)))
            n += 1
        return n

    def sample(self, batch_size: int) -> list[Transition]:
        return self._inner.sample(batch_size)

    def __len__(self) -> int:
        return len(self._inner)

    @property
    def ppo_rows(self) -> list[tuple[torch.Tensor, int, float]]:
        """``(obs_1d, concept_index, per_step_reward)`` for offline PPO microbatches."""
        return self._ppo_rows


__all__ = ["ASSISTMENTSReplayBuffer"]
