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
from adaptive_tutor.replay import UniformReplayBuffer
from adaptive_tutor.rl.dqn import DoubleDQNAgent
from adaptive_tutor.rl.ppo import CurriculumPolicy
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

    for ep in range(n_ep):
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
        episode_summaries.append(
            {
                "episode": int(ep),
                "total_reward": float(R),
                "length": int(ep_m["episode_length"]),
                "mastery_gain": float(ep_m["mastery_gain"]),
                "learning_rate": float(ep_m["learning_rate"]),
                "frustration_rate": float(ep_m["frustration_rate"]),
            }
        )
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
