"""
Main experiment runner: train + evaluate all 5 algorithms x 10 seeds.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from RL_Module import config
from RL_Module.agents.bandit_dqn import BanditDQNAgent
from RL_Module.agents.dqn_agent import DQNAgent, make_env
from RL_Module.agents.ppo_agent import PPOAgent, make_masked_env
from RL_Module.agents.random_agent import RandomAgent
from RL_Module.agents.rule_based import RuleBasedAgent
from RL_Module.environment.student_env import StudentEnv
from RL_Module.evaluation import plots
from RL_Module.evaluation.metrics import aggregate_across_seeds, evaluate, run_ablation
from RL_Module.explainability.explainer import Explainer
from RL_Module.logging_utils.csv_logger import SummaryWriter

AGENT_REGISTRY = {
    "PPO": PPOAgent,
    "DQN": DQNAgent,
    "Bandit_DQN": BanditDQNAgent,
    "Rule": RuleBasedAgent,
    "Random": RandomAgent,
}

ALGO_ORDER = ["PPO", "DQN", "Bandit_DQN", "Rule", "Random"]

AGENT_ALIASES = {
    "Bandit+DQN": "Bandit_DQN",
}


def _resolve_algorithm(algorithm: str) -> str:
    return AGENT_ALIASES.get(algorithm, algorithm)


def model_path(algorithm: str, seed: int) -> Path:
    return config.MODELS_DIR / f"{algorithm}_{seed}"


def train_agent(algorithm: str, seed: int) -> None:
    algorithm = _resolve_algorithm(algorithm)
    config.set_all_seeds(seed)
    cls = AGENT_REGISTRY[algorithm]
    agent = cls() if algorithm != "Random" else RandomAgent(seed)

    if not agent.needs_training():
        return

    if algorithm == "PPO":
        env = make_masked_env(seed, algo_tag=algorithm)
    else:
        env = make_env(seed, algo_tag=algorithm)

    agent.train(env, config.TRAINING_TIMESTEPS, seed)
    agent.save(str(model_path(algorithm, seed)))
    env.close()
    print(f"[train] {algorithm} seed={seed} -> {model_path(algorithm, seed)}")


def load_agent(algorithm: str, seed: int):
    algorithm = _resolve_algorithm(algorithm)
    cls = AGENT_REGISTRY[algorithm]
    agent = cls() if algorithm != "Random" else RandomAgent(seed)
    path = model_path(algorithm, seed)
    if agent.needs_training():
        if path.exists() or Path(f"{path}.zip").exists():
            agent.load(str(path))
    return agent


def eval_agent(algorithm: str, seed: int, n_episodes: Optional[int] = None) -> dict:
    algorithm = _resolve_algorithm(algorithm)
    config.set_all_seeds(seed)
    agent = load_agent(algorithm, seed)
    env = StudentEnv(population_seed=seed)
    env.set_algorithm_name(algorithm)
    explainer = Explainer(algorithm)
    n = n_episodes or config.EVAL_EPISODES
    result = evaluate(
        agent, env, n_episodes=n, seed=seed, algorithm=algorithm, explainer=explainer
    )
    explainer.close()
    env.close()
    print(
        f"[eval] {algorithm} seed={seed}: reward={result['mean_episode_reward']:.2f} "
        f"success={result['success_rate']:.1%}"
    )
    return result


def run_single(
    algorithm: str,
    seed: int,
    eval_episodes: Optional[int] = None,
    train: bool = True,
) -> dict:
    """Train (optional) and evaluate one algorithm/seed pair (Step 4 sanity)."""
    algorithm = _resolve_algorithm(algorithm)
    if train:
        train_agent(algorithm, seed)
    return eval_agent(algorithm, seed, n_episodes=eval_episodes)


def _run_one_worker(algorithm: str, seed: int, train: bool, eval_episodes: int) -> None:
    run_single(algorithm, seed, eval_episodes=eval_episodes, train=train)


def run_algorithm(
    algorithm: str, seeds: List[int], train: bool, eval_only: bool
) -> Dict:
    algorithm = _resolve_algorithm(algorithm)
    per_seed = []
    all_rewards_by_seed: List[List[float]] = []
    for seed in seeds:
        if train and not eval_only:
            train_agent(algorithm, seed)
        r = eval_agent(algorithm, seed)
        per_seed.append(r)
        all_rewards_by_seed.append(r["episode_rewards"])
    agg = aggregate_across_seeds(per_seed)
    agg["algorithm"] = algorithm
    return {"agg": agg, "rewards": all_rewards_by_seed, "per_seed": per_seed}


def print_summary_table(rows: List[Dict]) -> None:
    print("\n" + "=" * 72)
    print(f"{'Algorithm':<14} {'Reward (95% CI)':<22} {'Success':<10} {'LearnGain':<10}")
    print("-" * 72)
    for r in rows:
        mean = r.get("final_reward_mean", r.get("mean_mean_episode_reward", 0))
        lo = r.get("ci_lower", 0)
        hi = r.get("ci_upper", 0)
        ci = f"{mean:.0f} +/- {(hi-mean):.0f}"
        print(
            f"{r['algorithm']:<14} {ci:<22} "
            f"{float(r.get('success_rate', 0)):.0%}      "
            f"{float(r.get('learning_gain_mean', r.get('mean_learning_gain', 0))):.3f}"
        )
    print("=" * 72 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="RL adaptive tutoring experiment")
    parser.add_argument("--algo", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--timesteps", type=int, default=config.TRAINING_TIMESTEPS)
    parser.add_argument("--eval-episodes", type=int, default=config.EVAL_EPISODES)
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run algorithm/seed jobs in parallel (spawn)",
    )
    args = parser.parse_args()

    config.TRAINING_TIMESTEPS = args.timesteps
    config.EVAL_EPISODES = args.eval_episodes
    config.set_all_seeds(config.DEFAULT_SEED)

    if args.algo is not None:
        resolved = _resolve_algorithm(args.algo)
        if resolved not in AGENT_REGISTRY:
            parser.error(
                f"Unknown algorithm '{args.algo}'. "
                f"Valid: {list(AGENT_REGISTRY)}; aliases: {list(AGENT_ALIASES)}"
            )
        algorithms = [resolved]
    else:
        algorithms = ALGO_ORDER
    seeds = [args.seed] if args.seed is not None else config.SEEDS

    summary = SummaryWriter(append=False)
    all_algo_rewards: Dict[str, List[List[float]]] = {}
    summary_rows: List[Dict] = []

    if args.parallel and not args.train_only:
        from multiprocessing import get_context

        ctx = get_context("spawn")
        tasks: List[Tuple[str, int]] = [(a, s) for a in algorithms for s in seeds]
        train_flag = not args.eval_only
        with ctx.Pool(processes=min(6, os.cpu_count() or 1)) as pool:
            pool.starmap(
                _run_one_worker,
                [(a, s, train_flag, args.eval_episodes) for a, s in tasks],
            )

    for algo in algorithms:
        if args.train_only:
            if args.parallel:
                from multiprocessing import get_context

                ctx = get_context("spawn")
                tasks = [(a, s) for a in algorithms for s in seeds]
                with ctx.Pool(processes=min(6, os.cpu_count() or 1)) as pool:
                    pool.starmap(train_agent, tasks)
            else:
                for s in seeds:
                    train_agent(algo, s)
            continue

        if args.parallel:
            per_seed_results = []
            all_rewards_by_seed: List[List[float]] = []
            for s in seeds:
                r = eval_agent(algo, s, n_episodes=args.eval_episodes)
                per_seed_results.append(r)
                all_rewards_by_seed.append(r["episode_rewards"])
            agg = aggregate_across_seeds(per_seed_results)
            agg["algorithm"] = algo
            out = {"agg": agg, "rewards": all_rewards_by_seed, "per_seed": per_seed_results}
        else:
            out = run_algorithm(
                algo, seeds, train=not args.eval_only, eval_only=args.eval_only
            )
        agg = out["agg"]
        all_algo_rewards[algo] = out["rewards"]

        for seed, seed_result in zip(seeds, out["per_seed"]):
            summary.add_row({
                "algorithm": algo,
                "seed": seed,
                "final_reward_mean": seed_result["mean_episode_reward"],
                "final_reward_std": 0.0,
                "ci_lower": seed_result["mean_episode_reward"],
                "ci_upper": seed_result["mean_episode_reward"],
                "convergence_episode": seed_result.get("convergence_episode"),
                "success_rate": seed_result["success_rate"],
                "learning_gain_mean": seed_result["learning_gain"],
                "dropout_rate": seed_result["dropout_rate"],
                "adaptation_accuracy_mean": seed_result["adaptation_accuracy"],
            })

        rewards = [s["mean_episode_reward"] for s in out["per_seed"]]
        summary.add_row({
            "algorithm": algo,
            "seed": "ALL",
            "final_reward_mean": agg.get("mean_mean_episode_reward", agg.get("final_reward_mean", 0)),
            "final_reward_std": (
                float(np.std(rewards, ddof=1)) if len(rewards) > 1 else 0.0
            ),
            "ci_lower": agg.get("ci_lower", 0),
            "ci_upper": agg.get("ci_upper", 0),
            "convergence_episode": agg.get("convergence_episode"),
            "success_rate": float(np.mean([s["success_rate"] for s in out["per_seed"]])),
            "learning_gain_mean": float(np.mean([s["learning_gain"] for s in out["per_seed"]])),
            "dropout_rate": float(np.mean([s["dropout_rate"] for s in out["per_seed"]])),
            "adaptation_accuracy_mean": float(
                np.mean([s["adaptation_accuracy"] for s in out["per_seed"]])
            ),
        })

        summary_rows.append({
            "algorithm": algo,
            "final_reward_mean": agg.get("final_reward_mean", 0),
            "final_reward_std": agg.get("final_reward_std", 0),
            "ci_lower": agg.get("ci_lower", 0),
            "ci_upper": agg.get("ci_upper", 0),
            "convergence_episode": agg.get("convergence_episode"),
            "success_rate": float(np.mean([s["success_rate"] for s in out["per_seed"]])),
            "learning_gain_mean": float(np.mean([s["learning_gain"] for s in out["per_seed"]])),
            "dropout_rate": float(np.mean([s["dropout_rate"] for s in out["per_seed"]])),
            "adaptation_accuracy_mean": float(np.mean([s["adaptation_accuracy"] for s in out["per_seed"]])),
            "mean_mean_episode_reward": agg.get("mean_mean_episode_reward", 0),
        })

    if not args.train_only:
        summary.write()
        ablation_delta = None
        if args.ablation:
            print("\nRunning ablation...")
            ablation = run_ablation(eval_episodes=min(100, args.eval_episodes))
            ablation_delta = ablation["delta"]
            with open(config.LOGS_DIR / "ablation.json", "w") as f:
                json.dump(ablation, f, indent=2, default=str)

        plots.draw_all(summary_rows, all_algo_rewards, ablation_delta)
        print_summary_table(summary_rows)
        print(f"Summary written to {config.LOGS_DIR / 'summary.csv'}")


if __name__ == "__main__":
    main()
