"""Master evaluation runner (Phase 11): simulator + dataset + policy comparison."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from adaptive_tutor.evaluation.dataset_metrics import compute_dataset_metrics
from adaptive_tutor.evaluation.policy_eval import (
    evaluate_dqn_policy,
    evaluate_heuristic_policy,
    evaluate_ppo_policy,
    evaluate_random_policy,
)
from adaptive_tutor.memory.providers.dataset_provider import DatasetCurriculumProvider
from adaptive_tutor.memory.providers.teacher_provider import TeacherCurriculumProvider
from adaptive_tutor.rl.dqn import DoubleDQNAgent
from adaptive_tutor.rl.ppo import CurriculumPolicy
from adaptive_tutor.simulator.environment import TutoringEnvironment


def _project_root() -> Path:
    # .../adaptive_tutor/adaptive_tutor/evaluation/run_evaluation.py -> parents[2] = install root
    return Path(__file__).resolve().parents[2]


def _default_teacher_yaml() -> Path:
    env = os.environ.get("ADAPTIVE_TUTOR_TEACHER_PATH")
    if env:
        return Path(env).expanduser().resolve()
    return _project_root() / "configs" / "curriculum" / "examples" / "math_basic.yaml"


def _default_dataset_csv() -> Path:
    env = os.environ.get("ADAPTIVE_TUTOR_DATASET_PATH")
    if env:
        return Path(env).expanduser().resolve()
    return _project_root() / "configs" / "curriculum" / "examples" / "assistments_synthetic.csv"


def run_full_evaluation(
    *,
    teacher_yaml: Path | None = None,
    dataset_csv: Path | None = None,
    episodes_per_policy: int = 4,
    max_episode_steps: int = 64,
    ppo_segments_per_run: int = 3,
    seed: int = 0,
) -> dict[str, Any]:
    """Return one nested JSON-ready report (deterministic for fixed ``seed``)."""
    ty = Path(teacher_yaml) if teacher_yaml is not None else _default_teacher_yaml()
    dcsv = Path(dataset_csv) if dataset_csv is not None else _default_dataset_csv()

    torch.manual_seed(seed)
    teacher = TeacherCurriculumProvider(ty)
    dataset = DatasetCurriculumProvider(dcsv)
    n = teacher.num_concepts()
    slug0 = teacher.concept_slug(0)

    def env_factory() -> TutoringEnvironment:
        return TutoringEnvironment(
            concept_slug=slug0,
            concept_index=0,
            num_concepts=n,
            teacher=teacher,
            dataset=dataset,
            dataset_slugs_for_mask=frozenset(dataset.all_stats().keys()),
            dataset_skill_slug=slug0,
            strict_dataset_stats=False,
            seed=seed,
            max_episode_steps=max_episode_steps,
            mastery_threshold=0.9,
            frustration_terminal_threshold=0.95,
            enable_action_filter=True,
        )

    dqn = DoubleDQNAgent(hidden=(64, 64), device=torch.device("cpu"))
    ppo_pol = CurriculumPolicy(num_concepts=n, hidden=(48, 48))

    rnd = evaluate_random_policy(
        env_factory,
        n_episodes=episodes_per_policy,
        max_steps=max_episode_steps,
        seed=seed,
    )
    heu = evaluate_heuristic_policy(
        env_factory,
        n_episodes=episodes_per_policy,
        max_steps=max_episode_steps,
    )
    dq = evaluate_dqn_policy(
        env_factory,
        dqn,
        num_concepts=n,
        n_episodes=episodes_per_policy,
        max_steps=max_episode_steps,
        epsilon=0.15,
        seed=seed + 7,
    )
    pp = evaluate_ppo_policy(
        env_factory,
        teacher,
        ppo_pol,
        dqn,
        num_concepts=n,
        n_runs=episodes_per_policy,
        segments_per_run=ppo_segments_per_run,
        max_steps_guard=max_episode_steps * ppo_segments_per_run,
        epsilon=0.15,
        seed=seed + 19,
    )

    sim_ref = {
        "mastery_gain": heu["mastery_gain"],
        "learning_rate": heu["learning_rate"],
        "frustration_rate": heu["frustration_rate"],
        "hint_efficiency": heu["hint_efficiency"],
    }

    return {
        "simulator_metrics": sim_ref,
        "dataset_metrics": compute_dataset_metrics(dataset),
        "policy_comparison": {
            "random": rnd,
            "heuristic": heu,
            "dqn": dq,
            "ppo": pp,
        },
    }


__all__ = ["run_full_evaluation"]
