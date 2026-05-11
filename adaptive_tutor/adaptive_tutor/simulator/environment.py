"""Gym-style tutoring environment over a :class:`StateBuilder`.

API
---
``reset()`` -> ``(state, info)``  
``step(action: str)`` -> ``(state, reward, done, info)``

Actions are filtered **before** execution when ``enable_action_filter`` is
``True`` (default): curriculum membership, optional strict ASSISTments stats,
prerequisite mastery, cooldowns, and anti-whipsaw pacing. See
:mod:`adaptive_tutor.masking`.

Episode termination
-------------------
* mastery >= ``mastery_threshold`` (running mean correctness on the active concept),
* ``env_steps`` >= ``max_episode_steps``,
* frustration >= ``frustration_terminal_threshold`` (simulated burnout).
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any

from adaptive_tutor.masking import (
    ActionFilterContext,
    CooldownTracker,
    WhipsawTracker,
    run_filter_pipeline,
)
from adaptive_tutor.memory.providers.base import BaseCurriculumProvider
from adaptive_tutor.memory.providers.dataset_provider import (
    ASSISTMENTS_ACTIONS,
    SkillStats,
    skill_stats_as_tuple,
)
from adaptive_tutor.rewards import RewardEngine, default_reward_engine
from adaptive_tutor.simulator.calibration import LearnerParams, default_learner_params
from adaptive_tutor.simulator.learner import SyntheticLearner
from adaptive_tutor.state.builder import StateBuilder
from adaptive_tutor.state.state import LearnerState


class TutoringEnvironment:
    """Single-concept tutoring loop with synthetic latent learner."""

    def __init__(
        self,
        *,
        concept_slug: str,
        concept_index: int,
        num_concepts: int,
        params: LearnerParams | None = None,
        skill_stats: SkillStats | None = None,
        max_episode_steps: int = 200,
        mastery_threshold: float = 0.85,
        frustration_terminal_threshold: float = 0.92,
        seed: int = 0,
        window_length: int = 8,
        reward_engine: RewardEngine | None = None,
        teacher: BaseCurriculumProvider | None = None,
        dataset_slugs_for_mask: frozenset[str] | None = None,
        dataset_skill_slug: str | None = None,
        strict_dataset_stats: bool = False,
        initial_mastery_by_slug: Mapping[str, float] | None = None,
        prereq_mastery_threshold: float = 0.6,
        enable_action_filter: bool = True,
        cooldown_steps: Mapping[str, int] | None = None,
        whipsaw_window: int = 4,
        whipsaw_max_pace_switches: int = 2,
    ) -> None:
        self._slug = concept_slug
        self._index = int(concept_index)
        self._num_concepts = max(1, int(num_concepts))
        self._reward_engine = reward_engine or default_reward_engine()
        self._params = params or default_learner_params()
        self._stats_tuple = (
            skill_stats_as_tuple(skill_stats) if skill_stats is not None else ()
        )
        self._max_episode_steps = int(max_episode_steps)
        self._mastery_threshold = float(mastery_threshold)
        self._frustration_terminal_threshold = float(frustration_terminal_threshold)
        self._seed = int(seed)
        self._builder = StateBuilder(window_length=window_length)
        self._learner = SyntheticLearner.from_params(self._params)
        self._rng = random.Random(self._seed)
        self._env_step = 0
        self._teacher = teacher
        self._dataset_slugs = dataset_slugs_for_mask
        self._dataset_skill_slug = dataset_skill_slug or concept_slug
        self._strict_dataset = bool(strict_dataset_stats)
        self._initial_mastery_snapshot: dict[str, float] = dict(
            initial_mastery_by_slug or {}
        )
        self._mastery_by_slug: dict[str, float] = dict(self._initial_mastery_snapshot)
        self._prereq_threshold = float(prereq_mastery_threshold)
        self._enable_action_filter = bool(enable_action_filter)
        self._cooldown = CooldownTracker(cooldown_steps)
        self._whipsaw = WhipsawTracker(
            window=whipsaw_window, max_pace_switches=whipsaw_max_pace_switches
        )
        self._in_concept_segment: bool = False
        self._segment_step_cap: int = int(max_episode_steps)
        self._segment_mastery_threshold: float = float(mastery_threshold)

    def get_state(self) -> LearnerState:
        """Current learner-state snapshot (read-only builder view)."""
        return self._snapshot()

    def start_concept_segment(
        self,
        *,
        concept_slug: str,
        concept_index: int,
        dataset_skill_slug: str | None = None,
        pacing_factor: float = 1.0,
        difficulty_bias: float = 0.0,
    ) -> LearnerState:
        """Switch the active concept for a PPO macro-step without a full cold reset.

        Resets Phase 7 trackers and the env step counter for this segment only.
        Per-concept mastery accumulated in :class:`StateBuilder` persists.
        Rolling emotion windows persist (learner-centric signal).

        Parameters
        ----------
        pacing_factor:
            Multiplies ``max_episode_steps`` to cap micro-steps on this concept
            (clamped to ``[0.25, 3.0]`` ground room for thesis demos).
        difficulty_bias:
            In ``[-1, 1]`` shifts the mastery termination threshold for this
            segment only (easier teaching when positive).
        """
        self._slug = str(concept_slug)
        self._index = int(concept_index)
        self._dataset_skill_slug = (dataset_skill_slug or self._slug)
        self._builder.switch_concept(concept_id=self._slug, concept_index=self._index)
        self._env_step = 0
        self._cooldown.reset()
        self._whipsaw.reset()
        pf = max(0.25, min(3.0, float(pacing_factor)))
        self._segment_step_cap = max(1, int(self._max_episode_steps * pf))
        db = max(-1.0, min(1.0, float(difficulty_bias)))
        self._segment_mastery_threshold = max(
            0.4, min(0.98, self._mastery_threshold + 0.12 * db)
        )
        self._in_concept_segment = True
        return self._snapshot()

    @property
    def current_concept_slug(self) -> str:
        return self._slug

    @property
    def segment_micro_step_cap(self) -> int:
        """Max env steps for the active segment, or full episode when not in a segment."""
        return self._segment_step_cap if self._in_concept_segment else self._max_episode_steps

    def reset(self) -> tuple[LearnerState, dict[str, Any]]:
        """Start a new episode; windows cleared, latent learner re-seeded from params."""
        self._rng = random.Random(self._seed)
        self._env_step = 0
        self._cooldown.reset()
        self._whipsaw.reset()
        self._mastery_by_slug = dict(self._initial_mastery_snapshot)
        self._builder.reset(concept_id=self._slug, concept_index=self._index)
        self._learner = SyntheticLearner.from_params(self._params)
        self._in_concept_segment = False
        self._segment_step_cap = int(self._max_episode_steps)
        self._segment_mastery_threshold = float(self._mastery_threshold)
        return self._snapshot(), {}

    def step(self, action: str) -> tuple[LearnerState, float, bool, dict[str, Any]]:
        """Apply one tutor action after Phase 7 filtering.

        For RL training, use :func:`~adaptive_tutor.rl.dqn.trainer.collect_transition`
        so Double DQN selects meso heads, decodes to an ASSISTments opcode, and
        then this method applies masks/cooldowns/prerequisites before dynamics.
        """
        if action not in ASSISTMENTS_ACTIONS:
            raise ValueError(
                f"unknown action {action!r}; expected one of {ASSISTMENTS_ACTIONS}"
            )

        state_pre_action = self._snapshot()
        action_requested = action
        filter_reasons: tuple[str, ...] = ()
        if self._enable_action_filter:
            ctx = ActionFilterContext(
                teacher=self._teacher,
                concept_slug=self._slug,
                dataset_skill_slug=self._dataset_skill_slug,
                dataset_slugs=self._dataset_slugs,
                strict_dataset=self._strict_dataset,
                mastery_by_slug=self._mastery_by_slug,
                prereq_mastery_threshold=self._prereq_threshold,
            )
            action, _allowed, filter_reasons = run_filter_pipeline(
                action,
                state_pre_action,
                context=ctx,
                cooldown=self._cooldown,
                whipsaw=self._whipsaw,
            )

        hints_used = 1 if action == "give_hint" else 0
        self._learner.apply_action_effect(action)

        mastery = self._builder.mastery_of(self._slug)
        p_correct = self._learner.success_probability(mastery, action)
        correct = int(self._rng.random() < p_correct)

        self._learner.feedback_after_outcome(bool(correct))
        emotion = self._learner.to_emotion_vector()
        self._builder.observe(
            emotion=emotion,
            correct=bool(correct),
            hints=hints_used,
            attempts=1,
        )

        emotion_map: Mapping[str, float] = {
            "engaged": emotion.engaged,
            "confused": emotion.confused,
            "bored": emotion.bored,
            "frustrated": emotion.frustrated,
        }
        reward, r_components = self._reward_engine.compute(
            correct=correct,
            hint_count=hints_used,
            emotion=emotion_map,
            action=action,
        )
        self._env_step += 1

        state = self._snapshot()
        step_cap = self._segment_step_cap if self._in_concept_segment else self._max_episode_steps
        mastery_target = (
            self._segment_mastery_threshold
            if self._in_concept_segment
            else self._mastery_threshold
        )
        done = (
            state.mastery >= mastery_target
            or self._env_step >= step_cap
            or emotion.frustrated >= self._frustration_terminal_threshold
        )
        info: dict[str, Any] = {
            "correct": correct,
            "p_correct": p_correct,
            "action": action,
            "action_requested": action_requested,
            "action_filter": filter_reasons,
            "r_components": r_components,
        }
        self._mastery_by_slug[self._slug] = state.mastery
        if self._enable_action_filter:
            self._cooldown.record(action, state.timestep)
            self._whipsaw.record(action)
        return state, reward, done, info

    def _snapshot(self) -> LearnerState:
        return self._builder.snapshot(
            num_concepts=self._num_concepts,
            dataset_stats=self._stats_tuple,
        )
