"""Experiment orchestration (Phase 12): single entrypoint for reproducible runs."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import torch

from adaptive_tutor.evaluation.dataset_metrics import compute_dataset_metrics
from adaptive_tutor.evaluation.policy_eval import (
    run_callable_policy_episode,
    run_dqn_episode,
    run_ppo_curriculum_session,
)
from adaptive_tutor.evaluation.simulator_metrics import compute_episode_metrics
from adaptive_tutor.experiments.store import set_last_experiment
from adaptive_tutor.memory.providers.dataset_provider import DatasetCurriculumProvider
from adaptive_tutor.memory.providers.teacher_provider import TeacherCurriculumProvider
from adaptive_tutor.replay import ASSISTMENTSReplayBuffer, UniformReplayBuffer
from adaptive_tutor.rewards import default_reward_engine
from adaptive_tutor.rl.dqn import DoubleDQNAgent
from adaptive_tutor.rl.dqn.trainer import train_double_dqn_batch
from adaptive_tutor.rl.ppo import CurriculumPolicy, PPOStepRecord, ppo_policy_update
from adaptive_tutor.simulator.environment import TutoringEnvironment
from adaptive_tutor.simulator.policies import heuristic_tutor_policy, random_tutor_policy

Trajectory = list[dict[str, Any]]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_teacher_path() -> Path:
    env = os.environ.get("ADAPTIVE_TUTOR_TEACHER_PATH")
    if env:
        return Path(env).expanduser().resolve()
    return _project_root() / "configs" / "curriculum" / "examples" / "math_basic.yaml"


def _default_dataset_path() -> Path:
    env = os.environ.get("ADAPTIVE_TUTOR_DATASET_PATH")
    if env:
        return Path(env).expanduser().resolve()
    return _project_root() / "configs" / "curriculum" / "examples" / "assistments_synthetic.csv"


def seed_all(seed: int) -> None:
    """Best-effort deterministic process state (thesis reproducibility)."""
    s = int(seed)
    random.seed(s)
    torch.manual_seed(s)
    try:
        import numpy as np

        np.random.seed(s)
    except ImportError:
        pass


def _experiment_id(cfg: dict[str, Any]) -> str:
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def _mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def run_experiment(config: dict[str, Any]) -> dict[str, Any]:
    """Run ``N`` episodes with a named policy; aggregate Phase 11-style metrics.

    Config keys (all optional except policy intent):
        seed (int): default 0
        episodes (int): default 10
        policy (str): ``random`` | ``heuristic`` | ``dqn`` | ``ppo``
        teacher_path (str): default bundled math_basic.yaml, or env
            ``ADAPTIVE_TUTOR_TEACHER_PATH``
        dataset_path (str): default bundled CSV, or env
            ``ADAPTIVE_TUTOR_DATASET_PATH`` (your ASSISTments export)
        strict_dataset_stats (bool): if True, Phase-7 mask blocks escalation
            when the active skill is missing from the cohort CSV
        max_episode_steps (int): default 80
        dqn_epsilon (float): default 0.1
        ppo_segments_per_episode (int): default 3
        log_steps (bool): default False - if True, include per-step rows (large)
        mastery_threshold (float): forwarded to :func:`compute_episode_metrics`
        concept_index (int): starting teacher concept index, default 0
    """
    defaults: dict[str, Any] = {
        "seed": 0,
        "episodes": 10,
        "policy": "heuristic",
        "max_episode_steps": 80,
        "dqn_epsilon": 0.1,
        "ppo_segments_per_episode": 3,
        "log_steps": False,
        "mastery_threshold": 0.85,
        "concept_index": 0,
        "strict_dataset_stats": False,
        "offline_replay_capacity": 200_000,
        "dqn_batch_size": 32,
        "dqn_gamma": 0.99,
        "dqn_lr": 1e-3,
        "dqn_offline_updates_per_ep": 50,
        "ppo_lr": 3e-4,
        "ppo_offline_updates_per_ep": 40,
        "ppo_offline_batch_size": 32,
    }
    cfg: dict[str, Any] = {**defaults, **config}
    cfg["teacher_path"] = cfg.get("teacher_path") or str(_default_teacher_path())
    cfg["dataset_path"] = cfg.get("dataset_path") or str(_default_dataset_path())

    seed_all(int(cfg["seed"]))
    pol_name = str(cfg["policy"]).lower().strip()
    if pol_name not in {"random", "heuristic", "dqn", "ppo"}:
        raise ValueError(f"unknown policy {pol_name!r}")

    teacher = TeacherCurriculumProvider(Path(cfg["teacher_path"]))
    dataset = DatasetCurriculumProvider(Path(cfg["dataset_path"]))
    n_concepts = teacher.num_concepts()
    idx0 = max(0, min(n_concepts - 1, int(cfg["concept_index"])))
    slug0 = teacher.concept_slug(idx0)
    max_steps = max(1, int(cfg["max_episode_steps"]))
    n_ep = max(1, int(cfg["episodes"]))
    log_steps = bool(cfg["log_steps"])
    mt = float(cfg["mastery_threshold"])
    strict_ds = bool(cfg["strict_dataset_stats"])
    ds_slugs = frozenset(dataset.all_stats().keys())

    dqn: DoubleDQNAgent | None = None
    ppo: CurriculumPolicy | None = None
    if pol_name in {"dqn", "ppo"}:
        dqn = DoubleDQNAgent(hidden=(64, 64), device=torch.device("cpu"))
    if pol_name == "ppo":
        ppo = CurriculumPolicy(num_concepts=n_concepts, hidden=(48, 48))

    episode_trajs: list[Trajectory] = []
    episode_rewards: list[float] = []
    episode_summaries: list[dict[str, Any]] = []
    step_logs: list[dict[str, Any]] = []

    base_seed = int(cfg["seed"])

    offline_buffer: ASSISTMENTSReplayBuffer | None = None
    dqn_opt: torch.optim.Optimizer | None = None
    ppo_opt: torch.optim.Optimizer | None = None
    reward_engine = default_reward_engine()
    if pol_name in {"dqn", "ppo"}:
        offline_buffer = ASSISTMENTSReplayBuffer(
            capacity=max(1000, int(cfg["offline_replay_capacity"])),
            seed=base_seed,
        )
        loaded = offline_buffer.load_from_provider(
            dataset,
            teacher,
            reward_engine,
            num_concepts=n_concepts,
        )
        if loaded == 0:
            raise RuntimeError(
                "ASSISTMENTS replay buffer is empty; check dataset_path and CSV contents"
            )
    if pol_name == "dqn" and dqn is not None:
        dqn_opt = torch.optim.Adam(dqn.online.parameters(), lr=float(cfg["dqn_lr"]))
    if pol_name == "ppo" and ppo is not None:
        ppo_opt = torch.optim.Adam(ppo.parameters(), lr=float(cfg["ppo_lr"]))

    batch_size = max(4, int(cfg["dqn_batch_size"]))
    gamma = float(cfg["dqn_gamma"])
    dqn_updates = max(1, int(cfg["dqn_offline_updates_per_ep"]))
    ppo_updates = max(1, int(cfg["ppo_offline_updates_per_ep"]))
    ppo_batch = max(4, int(cfg["ppo_offline_batch_size"]))

    for ep in range(n_ep):
        dqn_train_loss = 0.0
        ppo_train_loss = 0.0

        if pol_name == "dqn" and offline_buffer is not None and dqn_opt is not None:
            assert dqn is not None
            losses: list[float] = []
            for _ in range(dqn_updates):
                batch = offline_buffer.sample(batch_size)
                if len(batch) < batch_size:
                    break
                losses.append(
                    train_double_dqn_batch(
                        dqn,
                        batch,
                        num_concepts=n_concepts,
                        gamma=gamma,
                        optimizer=dqn_opt,
                    )
                )
            dqn_train_loss = _mean(losses)

        if pol_name == "ppo" and offline_buffer is not None and ppo_opt is not None:
            assert ppo is not None
            rows = offline_buffer.ppo_rows
            rng_off = random.Random(base_seed + ep * 71)
            ppo_losses: list[float] = []
            for _ in range(ppo_updates):
                k = min(ppo_batch, len(rows))
                if k < 4:
                    break
                sample = rng_off.sample(rows, k)
                batch_recs: list[PPOStepRecord] = []
                for obs, cidx, r in sample:
                    with torch.no_grad():
                        lp = float(ppo.log_prob_of(obs, cidx).item())
                    batch_recs.append(PPOStepRecord(obs, cidx, lp, r))
                ppo_losses.append(ppo_policy_update(ppo, ppo_opt, batch_recs))
            ppo_train_loss = _mean(ppo_losses)

        env = TutoringEnvironment(
            concept_slug=slug0,
            concept_index=idx0,
            num_concepts=n_concepts,
            teacher=teacher,
            dataset=dataset,
            dataset_slugs_for_mask=ds_slugs,
            dataset_skill_slug=slug0,
            strict_dataset_stats=strict_ds,
            seed=base_seed + ep,
            max_episode_steps=max_steps,
            mastery_threshold=0.9,
            frustration_terminal_threshold=0.95,
            enable_action_filter=True,
            reward_engine=reward_engine,
        )
        traj: Trajectory = []
        R = 0.0

        if pol_name == "random":
            pol = random_tutor_policy(seed=base_seed + ep * 17)
            traj, R = run_callable_policy_episode(env, pol, max_steps=max_steps)
        elif pol_name == "heuristic":
            pol = heuristic_tutor_policy()
            traj, R = run_callable_policy_episode(env, pol, max_steps=max_steps)
        elif pol_name == "dqn":
            assert dqn is not None
            rng = random.Random(base_seed + ep * 31)
            traj, R = run_dqn_episode(
                env,
                dqn,
                num_concepts=n_concepts,
                max_steps=max_steps,
                epsilon=float(cfg["dqn_epsilon"]),
                rng=rng,
            )
        else:  # ppo
            assert dqn is not None and ppo is not None
            rng = random.Random(base_seed + ep * 97)
            buf = UniformReplayBuffer(10_000, seed=base_seed + ep)
            segs = max(1, int(cfg["ppo_segments_per_episode"]))
            traj, concept_rets, micro_r = run_ppo_curriculum_session(
                env,
                teacher,
                ppo,
                dqn,
                num_concepts=n_concepts,
                n_segments=segs,
                epsilon=float(cfg["dqn_epsilon"]),
                rng=rng,
                replay=buf,
            )
            R = micro_r + sum(concept_rets)

        ep_m = compute_episode_metrics(traj, mastery_threshold=mt)
        episode_trajs.append(traj)
        episode_rewards.append(R)
        if pol_name == "dqn" and log_steps:
            print(
                f"[experiment] episode {ep + 1}/{n_ep} "
                f"dqn_offline_loss_mean={dqn_train_loss:.6f} "
                f"rollout_return={R:.4f} replay_size={len(offline_buffer or [])}",
                flush=True,
            )
        if pol_name == "ppo" and log_steps:
            print(
                f"[experiment] episode {ep + 1}/{n_ep} "
                f"ppo_offline_loss_mean={ppo_train_loss:.6f} "
                f"session_return={R:.4f} replay_size={len(offline_buffer or [])}",
                flush=True,
            )
        summary_ep: dict[str, Any] = {
            "episode": int(ep),
            "total_reward": float(R),
            "length": int(ep_m["episode_length"]),
            "mastery_gain": float(ep_m["mastery_gain"]),
            "learning_rate": float(ep_m["learning_rate"]),
            "frustration_rate": float(ep_m["frustration_rate"]),
        }
        if pol_name == "dqn":
            summary_ep["dqn_offline_loss_mean"] = float(dqn_train_loss)
        if pol_name == "ppo":
            summary_ep["ppo_offline_loss_mean"] = float(ppo_train_loss)
        episode_summaries.append(summary_ep)
        if log_steps:
            for si, row in enumerate(traj):
                step_logs.append({"episode": int(ep), "step": si, **dict(row)})

    metrics_list = [compute_episode_metrics(t, mastery_threshold=mt) for t in episode_trajs]
    metric_keys = (
        "mastery_gain",
        "learning_rate",
        "frustration_rate",
        "hint_efficiency",
        "episode_length",
        "success_rate",
        "convergence_steps",
    )
    comparison_ready_metrics = {
        k: _mean([float(m[k]) for m in metrics_list]) for k in metric_keys
    }

    result: dict[str, Any] = {
        "experiment_id": _experiment_id(
            {
                "policy": pol_name,
                "seed": cfg["seed"],
                "episodes": n_ep,
                "max_episode_steps": max_steps,
                "teacher_path": cfg["teacher_path"],
                "dataset_path": cfg["dataset_path"],
            }
        ),
        "policy": pol_name,
        "offline_replay_transitions": (
            int(len(offline_buffer)) if offline_buffer is not None else 0
        ),
        "avg_reward": _mean(episode_rewards),
        "avg_learning_rate": comparison_ready_metrics["learning_rate"],
        "frustration_rate": comparison_ready_metrics["frustration_rate"],
        "comparison_ready_metrics": comparison_ready_metrics,
        "dataset_metrics": compute_dataset_metrics(dataset),
        "dataset_source_label": "cohort_csv",
        "episode_summaries": episode_summaries,
        "config_resolved": cfg,
    }
    if log_steps:
        result["step_logs"] = step_logs

    set_last_experiment(result)
    return result


__all__ = ["run_experiment", "seed_all"]
