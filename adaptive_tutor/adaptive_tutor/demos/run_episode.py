"""End-to-end demo: teacher YAML + ASSISTments CSV + state + env + replay.

Paths may be absolute or relative to the ``adaptive_tutor`` project directory
(the folder that contains ``pyproject.toml`` and ``configs/``).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from adaptive_tutor.integration import (
    assistments_action_to_composite,
    select_first_joint_concept,
)
from adaptive_tutor.memory.providers.base import BaseCurriculumProvider
from adaptive_tutor.memory.providers.dataset_provider import DatasetCurriculumProvider
from adaptive_tutor.memory.providers.teacher_provider import TeacherCurriculumProvider
from adaptive_tutor.replay import UniformReplayBuffer
from adaptive_tutor.rewards import RewardEngine, default_reward_engine
from adaptive_tutor.simulator import (
    TutoringEnvironment,
    calibrate_from_stats,
    heuristic_tutor_policy,
    random_tutor_policy,
)
from adaptive_tutor.simulator.policies import TutorPolicy
from adaptive_tutor.state.state import LearnerState, Transition


def _project_root() -> Path:
    # adaptive_tutor/adaptive_tutor/demos/run_episode.py -> parents[2] == project root
    return Path(__file__).resolve().parents[2]


def _resolve_path(p: str | Path, *, project_root: Path) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def _seed_mastery_for_concept(
    teacher: BaseCurriculumProvider, concept_id: str
) -> dict[str, float]:
    """Mark all immediate prerequisites as mastered so Phase-7 gates do not stall demos.

    The active concept starts at ``0.0`` in this dict; the builder still learns
    mastery from observations. Prerequisite slugs default to ``1.0`` so
    ``harder_problem`` / ``advance_to_next_skill`` stay available when the DAG
    requires prior nodes.
    """
    out: dict[str, float] = {}
    for c in teacher.get_concepts():
        if str(c["concept_id"]) == concept_id:
            for p in c.get("prerequisites") or []:
                out[str(p)] = 1.0
            break
    out[str(concept_id)] = float(out.get(concept_id, 0.0))
    return out


def run_demo_episode(
    *,
    teacher_path: str | Path,
    dataset_path: str | Path,
    max_steps: int = 20,
    seed: int = 0,
    policy: str = "random",
    buffer_capacity: int = 512,
    reward_engine: RewardEngine | None = None,
    print_fn: Callable[..., None] | None = print,
) -> tuple[list[Transition], UniformReplayBuffer]:
    """Run one synthetic episode on the first teacher/dataset slug intersection.

    Returns
    -------
    transitions
        Ordered list of :class:`Transition` records for inspection.
    buffer
        :class:`UniformReplayBuffer` populated with the same transitions.
    """
    project_root = _project_root()
    tp = _resolve_path(teacher_path, project_root=project_root)
    dp = _resolve_path(dataset_path, project_root=project_root)

    engine = reward_engine or default_reward_engine()
    teacher = TeacherCurriculumProvider(tp)
    dataset = DatasetCurriculumProvider(dp, reward_engine=engine)
    joint = select_first_joint_concept(teacher, dataset)

    n_concepts = len(teacher.get_concepts())
    params = calibrate_from_stats(joint.skill_stats)
    mastery_seed = _seed_mastery_for_concept(teacher, joint.concept_id)

    env = TutoringEnvironment(
        concept_slug=joint.concept_id,
        concept_index=joint.concept_index,
        num_concepts=n_concepts,
        params=params,
        skill_stats=joint.skill_stats,
        max_episode_steps=max_steps,
        seed=seed,
        reward_engine=engine,
        teacher=teacher,
        dataset_slugs_for_mask=frozenset(dataset.all_stats().keys()),
        dataset_skill_slug=joint.dataset_skill_slug,
        strict_dataset_stats=False,
        initial_mastery_by_slug=mastery_seed,
    )

    if policy == "heuristic":
        pol: TutorPolicy = heuristic_tutor_policy()
    else:
        pol = random_tutor_policy(seed=seed)

    buf = UniformReplayBuffer(buffer_capacity, seed=seed)
    traj: list[Transition] = []

    s, _info0 = env.reset()
    if print_fn:
        print_fn(
            f"[demo] joint concept={joint.concept_id!r} "
            f"dataset_slug={joint.dataset_skill_slug!r} "
            f"n_concepts={n_concepts}"
        )

    for t in range(max_steps):
        action = pol(s)
        s_next, r, done, info = env.step(action)
        r_comp = tuple(info["r_components"])
        tr_info = (
            ("correct", float(info["correct"])),
            ("p_correct", float(info["p_correct"])),
        )
        tr = Transition(
            s=s,
            a=assistments_action_to_composite(action),
            r=r,
            r_components=r_comp,
            s_next=s_next,
            done=done,
            info=tr_info,
        )
        buf.add(tr)
        traj.append(tr)

        if print_fn:
            print_fn(
                f"  t={t} action={action!r} reward={r:.4f} "
                f"mastery={s_next.mastery:.3f} done={done}"
            )

        s = s_next
        if done:
            break

    return traj, buf


def main(argv: Sequence[str] | None = None) -> int:
    project_root = _project_root()
    parser = argparse.ArgumentParser(description="Run one tutoring demo episode (no ML training).")
    parser.add_argument(
        "--teacher",
        type=str,
        default=str(project_root / "configs/curriculum/examples/math_basic.yaml"),
        help="Teacher curriculum YAML or JSON",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(project_root / "configs/curriculum/examples/assistments_synthetic.csv"),
        help="ASSISTments CSV path",
    )
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy", choices=("random", "heuristic"), default="random")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    run_demo_episode(
        teacher_path=args.teacher,
        dataset_path=args.dataset,
        max_steps=args.max_steps,
        seed=args.seed,
        policy=args.policy,
        print_fn=None if args.quiet else print,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
