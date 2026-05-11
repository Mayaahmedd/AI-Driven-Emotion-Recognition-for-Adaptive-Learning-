"""Concept-level rollouts: PPO chooses a segment, DQN fills micro-steps (Phase 9)."""

from __future__ import annotations

import random
from dataclasses import dataclass

from adaptive_tutor.memory.providers.base import BaseCurriculumProvider
from adaptive_tutor.replay import UniformReplayBuffer
from adaptive_tutor.rl.dqn.dqn import DoubleDQNAgent
from adaptive_tutor.rl.dqn.trainer import collect_transition
from adaptive_tutor.rl.ppo.buffer import PPOStepRecord
from adaptive_tutor.rl.ppo.policy import CurriculumPolicy
from adaptive_tutor.rl.ppo.reward import concept_rollout_reward
from adaptive_tutor.rl.ppo.teacher_utils import dataset_skill_slug_for_concept
from adaptive_tutor.simulator.environment import TutoringEnvironment
from adaptive_tutor.state.state import LearnerState


@dataclass(frozen=True, slots=True)
class ConceptRolloutOutcome:
    """PPO training tuple plus an interpretable micro-step trace."""

    record: PPOStepRecord
    concept_slug: str
    micro_trace: tuple[tuple[float, str, float, int, int, float], ...]
    """Per DQN step: ``(mastery, action, reward, correct, hint_flag, frustrated)``."""
    final_state: LearnerState


def collect_concept_rollout(
    env: TutoringEnvironment,
    teacher: BaseCurriculumProvider,
    agent: DoubleDQNAgent,
    dqn_buffer: UniformReplayBuffer,
    policy: CurriculumPolicy,
    *,
    curriculum_state: LearnerState,
    num_concepts: int,
    epsilon: float,
    rng: random.Random,
) -> ConceptRolloutOutcome:
    """PPO picks ``(concept, pacing, difficulty)``; DQN runs until segment ends.

    ``curriculum_state`` must be a snapshot *before* switching concepts -
    typically :meth:`TutoringEnvironment.get_state` after the previous segment
    (or the state returned from :meth:`TutoringEnvironment.reset`).
    """
    obs_cpu = curriculum_state.to_tensor(num_concepts=num_concepts).detach().cpu()
    act, logp = policy.sample_action(obs_cpu)
    slug = teacher.concept_slug(act.concept_index)
    skill = dataset_skill_slug_for_concept(teacher, slug)
    seg_state = env.start_concept_segment(
        concept_slug=slug,
        concept_index=act.concept_index,
        dataset_skill_slug=skill,
        pacing_factor=act.pacing_factor,
        difficulty_bias=act.difficulty_bias,
    )
    mastery0 = seg_state.mastery
    eng0 = seg_state.rolling_emotions.engaged

    state = seg_state
    done = False
    steps = 0
    trace: list[tuple[float, str, float, int, int, float]] = []
    guard = env.segment_micro_step_cap + 32

    while (not done) and steps < guard:
        state, done, info, step_r = collect_transition(
            env,
            agent,
            dqn_buffer,
            state,
            num_concepts=num_concepts,
            epsilon=epsilon,
            rng=rng,
        )
        steps += 1
        trace.append(
            (
                state.mastery,
                str(info["action"]),
                float(step_r),
                int(info["correct"]),
                1 if str(info["action"]) == "give_hint" else 0,
                float(state.rolling_emotions.frustrated),
            )
        )

    r_concept = concept_rollout_reward(
        mastery_start=mastery0,
        mastery_end=state.mastery,
        engaged_start=eng0,
        engaged_end=state.rolling_emotions.engaged,
        frustration_end=state.rolling_emotions.frustrated,
        n_steps=steps,
    )
    record = PPOStepRecord(
        obs_1d=obs_cpu.clone(),
        concept_index=act.concept_index,
        log_prob=float(logp.detach().item()),
        reward=r_concept,
    )
    return ConceptRolloutOutcome(
        record=record,
        concept_slug=slug,
        micro_trace=tuple(trace),
        final_state=state,
    )
