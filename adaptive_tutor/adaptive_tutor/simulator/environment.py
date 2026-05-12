"""Gym-style tutoring environment over a :class:`StateBuilder`.

API
---
``reset()`` -> ``(state, info)``  
``step(action: str)`` -> ``(state, reward, done, info)``

When a :class:`~adaptive_tutor.memory.providers.dataset_provider.DatasetCurriculumProvider`
is attached with ``prefer_dataset_replay=True``, each step **prefers** a logged
ASSISTments transition (real correctness, hints, attempts, emotion channels) for
the active skill. :class:`~adaptive_tutor.simulator.learner.SyntheticLearner` is
used only for explicit fallback (no logged episode, trace exhausted, or cold-start
policies that never wire a dataset).

Actions are still filtered through Phase 7 when ``enable_action_filter`` is
``True``.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any

from adaptive_tutor.core.types import EmotionVector
from adaptive_tutor.datasets.replay_learner import ASSISTMENTSReplayLearner
from adaptive_tutor.datasets.trajectory_builder import build_episode_list
from adaptive_tutor.masking import (
    ActionFilterContext,
    ActionFilterTrace,
    CooldownTracker,
    WhipsawTracker,
    run_filter_pipeline,
)
from adaptive_tutor.memory.providers.base import BaseCurriculumProvider
from adaptive_tutor.memory.providers.dataset_provider import (
    ASSISTMENTS_ACTIONS,
    DatasetCurriculumProvider,
    RawTransition,
    SkillStats,
    skill_stats_as_tuple,
)
from adaptive_tutor.rewards import RewardEngine, default_reward_engine
from adaptive_tutor.simulator.calibration import (
    LearnerParams,
    calibrate_from_stats,
    default_learner_params,
)
from adaptive_tutor.simulator.learner import SyntheticLearner
from adaptive_tutor.state.builder import StateBuilder
from adaptive_tutor.state.state import LearnerState

_REPLAY_MISMATCH_PENALTY = 0.35


class TutoringEnvironment:
    """Tutoring loop: ASSISTments replay when available, else synthetic learner."""

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
        dataset: DatasetCurriculumProvider | None = None,
        prefer_dataset_replay: bool = True,
    ) -> None:
        self._slug = concept_slug
        self._index = int(concept_index)
        self._num_concepts = max(1, int(num_concepts))
        self._reward_engine = reward_engine or default_reward_engine()
        self._dataset = dataset
        self._prefer_dataset_replay = bool(prefer_dataset_replay)
        self._max_episode_steps = int(max_episode_steps)
        self._mastery_threshold = float(mastery_threshold)
        self._frustration_terminal_threshold = float(frustration_terminal_threshold)
        self._seed = int(seed)
        self._builder = StateBuilder(window_length=window_length)
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

        self._replay_learner: ASSISTMENTSReplayLearner | None = None
        self._replay_episode: list[RawTransition] | None = None
        self._replay_idx: int = 0

        if self._dataset is not None and self._prefer_dataset_replay:
            episodes = build_episode_list(self._dataset)
            if episodes:
                self._replay_learner = ASSISTMENTSReplayLearner(episodes)

        if self._dataset is not None:
            self._apply_dataset_calibration()
        else:
            self._params = params or default_learner_params()
            self._stats_tuple = (
                skill_stats_as_tuple(skill_stats) if skill_stats is not None else ()
            )
        self._learner = SyntheticLearner.from_params(self._params)

    def _apply_dataset_calibration(self) -> None:
        if self._dataset is None:
            return
        slug = self._dataset_skill_slug or self._slug
        try:
            st = self._dataset.stats(slug)
        except KeyError:
            self._params = default_learner_params()
            self._stats_tuple = ()
            return
        self._params = calibrate_from_stats(st)
        self._stats_tuple = skill_stats_as_tuple(st)

    def _activate_replay_episode(self) -> None:
        self._replay_episode = None
        self._replay_idx = 0
        if self._replay_learner is None:
            return
        slug = self._dataset_skill_slug or self._slug
        ep = self._replay_learner.pick_episode(slug, self._rng)
        if ep:
            self._replay_episode = ep

    def get_state(self) -> LearnerState:
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
        self._slug = str(concept_slug)
        self._index = int(concept_index)
        self._dataset_skill_slug = dataset_skill_slug or self._slug
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
        if self._dataset is not None:
            self._apply_dataset_calibration()
            self._learner = SyntheticLearner.from_params(self._params)
        self._activate_replay_episode()
        return self._snapshot()

    @property
    def current_concept_slug(self) -> str:
        return self._slug

    @property
    def segment_micro_step_cap(self) -> int:
        return self._segment_step_cap if self._in_concept_segment else self._max_episode_steps

    def reset(self) -> tuple[LearnerState, dict[str, Any]]:
        self._rng = random.Random(self._seed)
        self._env_step = 0
        self._cooldown.reset()
        self._whipsaw.reset()
        self._mastery_by_slug = dict(self._initial_mastery_snapshot)
        self._builder.reset(concept_id=self._slug, concept_index=self._index)
        if self._dataset is not None:
            self._apply_dataset_calibration()
        self._learner = SyntheticLearner.from_params(self._params)
        return self._snapshot(), {}

    def step(self, action: str) -> tuple[LearnerState, float, bool, dict[str, Any]]:
        if action not in ASSISTMENTS_ACTIONS:
            raise ValueError(
                f"unknown action {action!r}; expected one of {ASSISTMENTS_ACTIONS}"
            )

        state_pre_action = self._snapshot()
        action_requested = action
        filter_reasons: tuple[str, ...] = ()
        filt_trace: ActionFilterTrace | None = None
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
            action, _allowed, filter_reasons, filt_trace = run_filter_pipeline(
                action,
                state_pre_action,
                context=ctx,
                cooldown=self._cooldown,
                whipsaw=self._whipsaw,
            )

        use_replay = (
            self._replay_episode is not None
            and self._replay_idx < len(self._replay_episode)
        )
        if use_replay:
            return self._step_dataset_replay(
                action,
                action_requested,
                filter_reasons,
                filt_trace,
                state_pre_action,
            )
        return self._step_synthetic(
            action,
            action_requested,
            filter_reasons,
            filt_trace,
            state_pre_action,
        )

    def _emotion_from_logged(self, emotion: Mapping[str, Any]) -> EmotionVector:
        return EmotionVector(
            engaged=float(emotion.get("engaged", 0.0)),
            confused=float(emotion.get("confused", 0.0)),
            bored=float(emotion.get("bored", 0.0)),
            frustrated=float(emotion.get("frustrated", 0.0)),
        )

    def _step_dataset_replay(
        self,
        action: str,
        action_requested: str,
        filter_reasons: tuple[str, ...],
        filt_trace: ActionFilterTrace | None,
        state_pre_action: LearnerState,
    ) -> tuple[LearnerState, float, bool, dict[str, Any]]:
        assert self._replay_episode is not None
        logged = self._replay_episode[self._replay_idx]
        action_match = action == logged.action
        em_d = logged.next_state.get("emotion") or {}
        emotion = self._emotion_from_logged(em_d)
        correct = int(logged.correct)
        hint_obs = min(3, max(0, int(logged.hint_count)))
        attempts = max(1, int(logged.attempt_count))

        emotion_map: Mapping[str, float] = {
            "engaged": emotion.engaged,
            "confused": emotion.confused,
            "bored": emotion.bored,
            "frustrated": emotion.frustrated,
        }
        reward, r_components = self._reward_engine.compute(
            correct=correct,
            hint_count=int(logged.hint_count),
            emotion=emotion_map,
            action=action,
        )
        p_correct = 1.0 if correct else 0.0
        if not action_match:
            reward -= _REPLAY_MISMATCH_PENALTY
            r_components = tuple(r_components) + (
                ("replay_action_mismatch", -float(_REPLAY_MISMATCH_PENALTY)),
            )

        self._builder.observe(
            emotion=emotion,
            correct=bool(correct),
            hints=1 if logged.hint_count > 0 else 0,
            attempts=attempts,
        )
        self._env_step += 1
        self._replay_idx += 1
        if self._replay_idx >= len(self._replay_episode):
            self._replay_episode = None

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
            or bool(logged.done)
        )
        info: dict[str, Any] = {
            "correct": correct,
            "p_correct": p_correct,
            "action": action,
            "action_requested": action_requested,
            "action_filter": filter_reasons,
            "action_filter_trace": (
                filt_trace.to_json_dict()
                if filt_trace is not None
                else {
                    "mask_passed": True,
                    "cooldown_blocked": False,
                    "prerequisites_met": True,
                    "whipsaw_blocked": False,
                }
            ),
            "r_components": r_components,
            "transition_source": "assistments_replay",
            "logged_action": logged.action,
            "replay_action_match": action_match,
        }
        self._mastery_by_slug[self._slug] = state.mastery
        if self._enable_action_filter:
            self._cooldown.record(action, state.timestep)
            self._whipsaw.record(action)
        return state, float(reward), done, info

    def _step_synthetic(
        self,
        action: str,
        action_requested: str,
        filter_reasons: tuple[str, ...],
        filt_trace: ActionFilterTrace | None,
        state_pre_action: LearnerState,
    ) -> tuple[LearnerState, float, bool, dict[str, Any]]:
        del state_pre_action
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
            "action_filter_trace": (
                filt_trace.to_json_dict()
                if filt_trace is not None
                else {
                    "mask_passed": True,
                    "cooldown_blocked": False,
                    "prerequisites_met": True,
                    "whipsaw_blocked": False,
                }
            ),
            "r_components": r_components,
            "transition_source": "synthetic_learner_fallback",
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
