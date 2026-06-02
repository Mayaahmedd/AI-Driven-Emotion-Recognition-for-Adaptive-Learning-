"""
Train and evaluate algorithm comparison + PPO emotion ablation.

Algorithm comparison (default):
  PPO, DQN, Rule, Random

PPO ablation:
  - emotion-aware: full state (knowledge + affect + emotion_id)
  - knowledge-only: ablation_no_emotion (obs[5] = 0, no FER routing in obs)

Usage (from repo root):
  export PYTHONPATH="$(pwd)"
  python -m RL_Module.run_comparison_experiment
  python -m RL_Module.run_comparison_experiment --seed 42 --timesteps 20000 --eval-episodes 200
  python -m RL_Module.run_comparison_experiment --skip-ablation
  python -m RL_Module.run_comparison_experiment --eval-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from RL_Module import config
from RL_Module.agents.dqn_agent import DQNAgent, make_env
from RL_Module.agents.ppo_agent import PPOAgent, make_masked_env
from RL_Module.agents.random_agent import RandomAgent
from RL_Module.agents.rule_based import RuleBasedAgent
from RL_Module.environment.student_env import StudentEnv
from RL_Module.evaluation.metrics import (
    action_frequency_report,
    aggregate_across_seeds,
    evaluate,
)
from RL_Module.main_experiment import eval_agent, print_summary_table, train_agent

COMPARE_ALGOS = ["PPO", "DQN", "Rule", "Random"]


def _metrics_row(algorithm: str, result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "algorithm": algorithm,
        "mean_episode_reward": result["mean_episode_reward"],
        "success_rate": result["success_rate"],
        "learning_gain": result["learning_gain"],
        "dropout_rate": result["dropout_rate"],
        "adaptation_accuracy": result["adaptation_accuracy"],
        "convergence_episode": result.get("convergence_episode"),
    }


def train_and_eval_ppo_ablation(
    seed: int,
    train_timesteps: int,
    eval_episodes: int,
    use_emotion: bool,
    label: str,
    model_suffix: str,
) -> Dict[str, Any]:
    """Train/eval PPO with or without emotion in observation."""
    ablation = not use_emotion
    config.set_all_seeds(seed)

    env = make_masked_env(
        seed,
        use_emotion=use_emotion,
        ablation_no_emotion=ablation,
        algo_tag=f"ppo_{model_suffix}_seed{seed}",
    )
    agent = PPOAgent()
    agent.train(env, train_timesteps, seed)
    save_path = config.MODELS_DIR / f"PPO_{model_suffix}_{seed}"
    agent.save(str(save_path))
    env.close()

    eval_env = StudentEnv(
        population_seed=seed,
        use_emotion=use_emotion,
        ablation_no_emotion=ablation,
    )
    eval_env.set_algorithm_name(label)
    result = evaluate(
        agent,
        eval_env,
        n_episodes=eval_episodes,
        seed=seed,
        algorithm=label,
        log_steps=True,
    )
    eval_env.close()
    return result


def run_algorithm_comparison(
    seeds: List[int],
    train_timesteps: int,
    eval_episodes: int,
    train: bool,
) -> Dict[str, Any]:
    """Train + eval PPO, DQN, Rule, Random."""
    out: Dict[str, Any] = {"per_algorithm": {}, "aggregated": []}

    for algo in COMPARE_ALGOS:
        per_seed: List[Dict[str, Any]] = []
        for seed in seeds:
            if train:
                config.TRAINING_TIMESTEPS = train_timesteps
                train_agent(algo, seed)
            r = eval_agent(algo, seed, n_episodes=eval_episodes)
            per_seed.append(r)
            if algo == "PPO":
                print(f"\n--- PPO action usage (seed={seed}) ---")
                action_frequency_report(r["episode_results"])

        agg = aggregate_across_seeds(per_seed)
        agg["algorithm"] = algo
        out["per_algorithm"][algo] = {
            "per_seed": [_metrics_row(algo, s) for s in per_seed],
            "aggregate": agg,
        }
        out["aggregated"].append({
            "algorithm": algo,
            "final_reward_mean": agg.get("mean_mean_episode_reward", 0),
            "ci_lower": agg.get("ci_lower", 0),
            "ci_upper": agg.get("ci_upper", 0),
            "success_rate": float(sum(s["success_rate"] for s in per_seed) / len(per_seed)),
            "learning_gain_mean": float(sum(s["learning_gain"] for s in per_seed) / len(per_seed)),
            "adaptation_accuracy_mean": float(
                sum(s["adaptation_accuracy"] for s in per_seed) / len(per_seed)
            ),
        })

    return out


def run_ppo_ablation(
    seeds: List[int],
    train_timesteps: int,
    eval_episodes: int,
    train: bool,
) -> Dict[str, Any]:
    """PPO with emotions vs knowledge-only (no emotion in obs)."""
    variants = [
        ("PPO_emotion", "emotion", True),
        ("PPO_knowledge_only", "no_emotion", False),
    ]
    out: Dict[str, Any] = {"variants": {}, "delta": {}}

    for label, suffix, use_emotion in variants:
        per_seed: List[Dict[str, Any]] = []
        for seed in seeds:
            if train:
                r = train_and_eval_ppo_ablation(
                    seed, train_timesteps, eval_episodes, use_emotion, label, suffix
                )
            else:
                agent = PPOAgent()
                path = config.MODELS_DIR / f"PPO_{suffix}_{seed}"
                if path.exists() or Path(f"{path}.zip").exists():
                    agent.load(str(path))
                else:
                    raise FileNotFoundError(
                        f"No model at {path}. Run without --eval-only first."
                    )
                eval_env = StudentEnv(
                    population_seed=seed,
                    use_emotion=use_emotion,
                    ablation_no_emotion=not use_emotion,
                )
                eval_env.set_algorithm_name(label)
                r = evaluate(
                    agent,
                    eval_env,
                    n_episodes=eval_episodes,
                    seed=seed,
                    algorithm=label,
                    log_steps=True,
                )
                eval_env.close()
            per_seed.append(r)
            print(f"\n--- {label} action usage (seed={seed}) ---")
            action_frequency_report(r["episode_results"])

        agg = aggregate_across_seeds(per_seed)
        out["variants"][label] = {
            "per_seed": [_metrics_row(label, s) for s in per_seed],
            "aggregate": agg,
        }

    emo = out["variants"]["PPO_emotion"]["aggregate"]
    kno = out["variants"]["PPO_knowledge_only"]["aggregate"]
    out["delta"] = {
        "reward_emotion_minus_knowledge_only": (
            emo.get("mean_mean_episode_reward", 0) - kno.get("mean_mean_episode_reward", 0)
        ),
        "success_rate_delta": (
            emo.get("mean_success_rate", 0) - kno.get("mean_success_rate", 0)
        ),
        "learning_gain_delta": (
            emo.get("mean_learning_gain", 0) - kno.get("mean_learning_gain", 0)
        ),
        "adaptation_accuracy_delta": (
            emo.get("mean_adaptation_accuracy", 0) - kno.get("mean_adaptation_accuracy", 0)
        ),
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Algorithm comparison + PPO emotion ablation"
    )
    parser.add_argument("--seed", type=int, nargs="+", default=[config.DEFAULT_SEED])
    parser.add_argument("--timesteps", type=int, default=config.TRAINING_TIMESTEPS)
    parser.add_argument("--eval-episodes", type=int, default=config.EVAL_EPISODES)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--skip-comparison", action="store_true")
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Shorthand: 10k timesteps, 100 eval episodes",
    )
    args = parser.parse_args()

    if args.quick:
        args.timesteps = min(args.timesteps, 10_000)
        args.eval_episodes = min(args.eval_episodes, 100)

    config.TRAINING_TIMESTEPS = args.timesteps
    config.EVAL_EPISODES = args.eval_episodes
    seeds = args.seed
    train = not args.eval_only

    results: Dict[str, Any] = {
        "seeds": seeds,
        "train_timesteps": args.timesteps,
        "eval_episodes": args.eval_episodes,
    }

    if not args.skip_comparison:
        print("\n" + "=" * 72)
        print("ALGORITHM COMPARISON: PPO vs DQN vs Rule vs Random")
        print("=" * 72)
        comparison = run_algorithm_comparison(
            seeds, args.timesteps, args.eval_episodes, train=train
        )
        results["comparison"] = comparison
        print_summary_table(comparison["aggregated"])

    if not args.skip_ablation:
        print("\n" + "=" * 72)
        print("PPO ABLATION: emotion-aware vs knowledge-only")
        print("=" * 72)
        ablation = run_ppo_ablation(
            seeds, args.timesteps, args.eval_episodes, train=train
        )
        results["ppo_ablation"] = ablation
        print("\nAblation delta (emotion - knowledge-only):")
        for k, v in ablation["delta"].items():
            print(f"  {k}: {v:+.4f}")

    out_path = config.LOGS_DIR / "comparison_experiment.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
