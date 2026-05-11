"""Policy rollouts and aggregated metrics (Phase 11)."""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from adaptive_tutor.evaluation.simulator_metrics import compute_episode_metrics
from adaptive_tutor.memory.providers.base import BaseCurriculumProvider
from adaptive_tutor.memory.providers.dataset_provider import ASSISTMENTS_ACTIONS
from adaptive_tutor.replay import UniformReplayBuffer
from adaptive_tutor.rl.dqn import DoubleDQNAgent
from adaptive_tutor.rl.ppo import CurriculumPolicy, collect_concept_rollout
from adaptive_tutor.simulator.environment import TutoringEnvironment
from adaptive_tutor.simulator.policies import TutorPolicy, heuristic_tutor_policy, random_tutor_policy

Trajectory = list[dict[str, Any]]


def _std(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return float(var**0.5)


def run_callable_policy_episode(
    env: TutoringEnvironment,
    policy: TutorPolicy,
    *,
    max_steps: int,
) -> tuple[Trajectory, float]:
    traj: Trajectory = []
    state, _ = env.reset()
    total_r = 0.0
    for _ in range(max_steps):
        a = policy(state)
        if a not in ASSISTMENTS_ACTIONS:
            a = ASSISTMENTS_ACTIONS[0]
        state, r, done, info = env.step(a)
        total_r += r
        traj.append(
            {
                "mastery": float(state.mastery),
                "reward": float(r),
                "correct": int(info["correct"]),
                "hint": str(info["action"]) == "give_hint",
                "frustrated": float(state.rolling_emotions.frustrated),
                "action": str(info["action"]),
            }
        )
        if done:
            break
    return traj, total_r


def run_dqn_episode(
    env: TutoringEnvironment,
    agent: DoubleDQNAgent,
    *,
    num_concepts: int,
    max_steps: int,
    epsilon: float,
    rng: random.Random,
) -> tuple[Trajectory, float]:
    traj = []
    state, _ = env.reset()
    total_r = 0.0
    for _ in range(max_steps):
        s_idx, d_idx, p_idx, action_str = agent.act(
            state, num_concepts=num_concepts, epsilon=epsilon, rng=rng
        )
        state, r, done, info = env.step(action_str)
        total_r += r
        traj.append(
            {
                "mastery": float(state.mastery),
                "reward": float(r),
                "correct": int(info["correct"]),
                "hint": str(info["action"]) == "give_hint",
                "frustrated": float(state.rolling_emotions.frustrated),
                "action": str(info["action"]),
            }
        )
        if done:
            break
    return traj, total_r


def run_dqn_greedy_episode(
    env: TutoringEnvironment,
    agent: DoubleDQNAgent,
    *,
    num_concepts: int,
    max_steps: int,
) -> tuple[Trajectory, float]:
    return run_dqn_episode(
        env,
        agent,
        num_concepts=num_concepts,
        max_steps=max_steps,
        epsilon=0.0,
        rng=random.Random(0),
    )


def _rows_from_ppo_micro(
    rows: Sequence[tuple[float, str, float, int, int, float]],
) -> Trajectory:
    out: Trajectory = []
    for m, a, r, c, h, f in rows:
        out.append(
            {
                "mastery": float(m),
                "reward": float(r),
                "correct": int(c),
                "hint": bool(h),
                "frustrated": float(f),
                "action": str(a),
            }
        )
    return out


def run_ppo_curriculum_session(
    env: TutoringEnvironment,
    teacher: BaseCurriculumProvider,
    curriculum_policy: CurriculumPolicy,
    dqn_agent: DoubleDQNAgent,
    *,
    num_concepts: int,
    n_segments: int,
    epsilon: float,
    rng: random.Random,
    replay: UniformReplayBuffer | None = None,
) -> tuple[Trajectory, list[float], float]:
    """Several concept segments; returns flat micro trajectory + concept rewards."""
    buf = replay if replay is not None else UniformReplayBuffer(10_000, seed=0)
    curriculum_state, _ = env.reset()
    flat: Trajectory = []
    concept_returns: list[float] = []
    concepts_seen: set[str] = set()
    for _ in range(n_segments):
        out = collect_concept_rollout(
            env,
            teacher,
            dqn_agent,
            buf,
            curriculum_policy,
            curriculum_state=curriculum_state,
            num_concepts=num_concepts,
            epsilon=epsilon,
            rng=rng,
        )
        concepts_seen.add(out.concept_slug)
        flat.extend(_rows_from_ppo_micro(out.micro_trace))
        concept_returns.append(float(out.record.reward))
        curriculum_state = out.final_state
    total_micro_r = sum(float(x["reward"]) for x in flat)
    cov = len(concepts_seen) / float(max(1, num_concepts))
    for row in flat:
        row["concept_coverage_scalar"] = cov
    return flat, concept_returns, total_micro_r


@dataclass(frozen=True, slots=True)
class PolicyEvalSummary:
    mean_episode_metrics: dict[str, float]
    reward_mean: float
    reward_std: float
    concept_coverage: float


def summarize_rollouts(
    episode_trajs: Sequence[Trajectory],
    episode_rewards: Sequence[float],
) -> PolicyEvalSummary:
    ep_m = [compute_episode_metrics(t) for t in episode_trajs]
    keys = (
        "mastery_gain",
        "learning_rate",
        "frustration_rate",
        "hint_efficiency",
        "episode_length",
        "success_rate",
        "convergence_steps",
    )
    mean_ep = {k: float(sum(d[k] for d in ep_m) / len(ep_m)) for k in keys}
    cov_vals = [
        float(t[-1].get("concept_coverage_scalar", 1.0)) if t else 1.0 for t in episode_trajs
    ]
    cov = float(sum(cov_vals) / len(cov_vals))
    rw = [float(x) for x in episode_rewards]
    return PolicyEvalSummary(
        mean_episode_metrics=mean_ep,
        reward_mean=float(sum(rw) / len(rw)),
        reward_std=_std(rw),
        concept_coverage=cov,
    )


def _summary_to_json(s: PolicyEvalSummary) -> dict[str, float]:
    out = dict(s.mean_episode_metrics)
    out.update(
        {
            "reward_mean": s.reward_mean,
            "reward_std": s.reward_std,
            "concept_coverage": s.concept_coverage,
        }
    )
    return out


def learning_curve_slope_from_rewards(rewards: Sequence[float]) -> float:
    """Simple OLS slope across evaluation episodes (higher = improving between runs)."""
    y = [float(r) for r in rewards]
    n = len(y)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = sum(y) / n
    num = sum((i - x_mean) * (y[i] - y_mean) for i in range(n))
    den = sum((i - x_mean) ** 2 for i in range(n))
    return float(num / den) if den > 1e-9 else 0.0


def _pack_policy_result(rews: list[float], s: PolicyEvalSummary) -> dict[str, float]:
    out = _summary_to_json(s)
    out["learning_curve_slope"] = learning_curve_slope_from_rewards(rews)
    return out


def evaluate_random_policy(
    env_factory: Callable[[], TutoringEnvironment],
    *,
    n_episodes: int,
    max_steps: int,
    seed: int,
) -> dict[str, float]:
    pol = random_tutor_policy(seed=seed)
    trajs: list[Trajectory] = []
    rews: list[float] = []
    for _ in range(n_episodes):
        env = env_factory()
        tr, R = run_callable_policy_episode(env, pol, max_steps=max_steps)
        trajs.append(tr)
        rews.append(R)
    return _pack_policy_result(rews, summarize_rollouts(trajs, rews))


def evaluate_heuristic_policy(
    env_factory: Callable[[], TutoringEnvironment],
    *,
    n_episodes: int,
    max_steps: int,
) -> dict[str, float]:
    pol = heuristic_tutor_policy()
    trajs: list[Trajectory] = []
    rews: list[float] = []
    for _ in range(n_episodes):
        env = env_factory()
        tr, R = run_callable_policy_episode(env, pol, max_steps=max_steps)
        trajs.append(tr)
        rews.append(R)
    return _pack_policy_result(rews, summarize_rollouts(trajs, rews))


def evaluate_dqn_policy(
    env_factory: Callable[[], TutoringEnvironment],
    agent: DoubleDQNAgent,
    *,
    num_concepts: int,
    n_episodes: int,
    max_steps: int,
    epsilon: float,
    seed: int,
) -> dict[str, float]:
    trajs: list[Trajectory] = []
    rews: list[float] = []
    for i in range(n_episodes):
        rng = random.Random(seed + i)
        env = env_factory()
        tr, R = run_dqn_episode(
            env,
            agent,
            num_concepts=num_concepts,
            max_steps=max_steps,
            epsilon=epsilon,
            rng=rng,
        )
        trajs.append(tr)
        rews.append(R)
    return _pack_policy_result(rews, summarize_rollouts(trajs, rews))


def evaluate_ppo_policy(
    env_factory: Callable[[], TutoringEnvironment],
    teacher: BaseCurriculumProvider,
    curriculum_policy: CurriculumPolicy,
    dqn_agent: DoubleDQNAgent,
    *,
    num_concepts: int,
    n_runs: int,
    segments_per_run: int,
    max_steps_guard: int,
    epsilon: float,
    seed: int,
) -> dict[str, float]:
    trajs: list[Trajectory] = []
    rews: list[float] = []
    for i in range(n_runs):
        rng = random.Random(seed + i)
        env = env_factory()
        flat, concept_rets, micro_r = run_ppo_curriculum_session(
            env,
            teacher,
            curriculum_policy,
            dqn_agent,
            num_concepts=num_concepts,
            n_segments=segments_per_run,
            epsilon=epsilon,
            rng=rng,
        )
        trajs.append(flat[:max_steps_guard] if len(flat) > max_steps_guard else flat)
        combined_r = micro_r + sum(concept_rets)
        rews.append(combined_r)
    return _pack_policy_result(rews, summarize_rollouts(trajs, rews))
