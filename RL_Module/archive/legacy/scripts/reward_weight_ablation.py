"""
Reward weight ablation: thesis baseline weights vs equal weights.

Compares DQN under two reward weight presets while holding all other
components fixed (architecture, replay, exploration, student model, reward formula).

Usage (from repo root):
  export PYTHONPATH="$(pwd)"
  python -m RL_Module.reward_weight_ablation --phase all
  python -m RL_Module.reward_weight_ablation --phase run --resume
  python -m RL_Module.reward_weight_ablation --phase analyze
  python -m RL_Module.reward_weight_ablation --phase all --quick
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import RL_Module.config as cfg
from RL_Module.agents.dqn_agent import DQNAgent, make_env
from RL_Module.environment.student_env import StudentEnv
from RL_Module.evaluation.metrics import (
    _adaptation_match,
    _is_success,
    action_frequency_report,
    confidence_interval,
    normalized_knowledge_gain,
)
from RL_Module.mdp_definition import ACTION_DIM, ID_TO_ACTION

OUT_DIR = _HERE / "figures" / "reward_weight_ablation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 7, 13, 21, 99, 314]
TRAIN_TS = 50_000
EVAL_EPS = 500

G4_GAIN = {
    "GAIN_CORRECT_FACTOR": 0.1,
    "GAIN_INCORRECT_FACTOR": 1.0,
    "ratio_label": "10:1",
}

# dqn_stability_study baseline hyperparameters
BASELINE_HP: Dict[str, Any] = {
    "learning_rate": 5e-4,
    "batch_size": 64,
    "buffer_size": 100_000,
    "target_update_interval": 2000,
    "exploration_fraction": 0.3,
    "exploration_final_eps": 0.05,
    "net_arch": [64, 64],
}

REWARD_CONFIGS: Dict[str, Dict[str, float]] = {
    "baseline_weights": {
        "wk": 0.50,
        "we": 0.20,
        "wf": 0.15,
        "wb": 0.10,
        "wc": 0.05,
    },
    "equal_weights": {
        "wk": 0.20,
        "we": 0.20,
        "wf": 0.20,
        "wb": 0.20,
        "wc": 0.20,
    },
}

METRICS = [
    "learning_gain",
    "final_knowledge",
    "success_rate",
    "adaptation_accuracy",
    "mean_episode_reward",
    "dropout_rate",
]

AGG_METRICS = [
    "learning_gain",
    "success_rate",
    "adaptation_accuracy",
    "mean_episode_reward",
]


def _patch_gain() -> Dict[str, float]:
    prev = dict(cfg.SIMULATOR_PARAMS)
    cfg.SIMULATOR_PARAMS = dict(G4_GAIN)
    return prev


def _restore_gain(snapshot: Dict[str, float]) -> None:
    cfg.SIMULATOR_PARAMS = snapshot


def _patch_reward(weights: Dict[str, float]) -> Tuple[Dict[str, float], bool, Dict[str, float]]:
    prev_weights = dict(cfg.REWARD_WEIGHTS)
    prev_sens_flag = cfg.USE_SENSITIVITY_WEIGHTS
    prev_sens_weights = dict(cfg.SENSITIVITY_WEIGHTS)
    cfg.REWARD_WEIGHTS = dict(weights)
    cfg.USE_SENSITIVITY_WEIGHTS = False
    cfg.SENSITIVITY_WEIGHTS = {}
    return prev_weights, prev_sens_flag, prev_sens_weights


def _restore_reward(
    snapshot: Tuple[Dict[str, float], bool, Dict[str, float]],
) -> None:
    prev_weights, prev_sens_flag, prev_sens_weights = snapshot
    cfg.REWARD_WEIGHTS = prev_weights
    cfg.USE_SENSITIVITY_WEIGHTS = prev_sens_flag
    cfg.SENSITIVITY_WEIGHTS = prev_sens_weights


def _shannon_entropy(freq: Dict[str, float]) -> float:
    p = np.array([v for v in freq.values() if v > 0], dtype=float)
    return float(-np.sum(p * np.log(p))) if len(p) else 0.0


def _actions_used_at_least(freq: Dict[str, float], threshold: float = 0.01) -> int:
    return sum(1 for v in freq.values() if v >= threshold)


def evaluate_dqn(
    agent: DQNAgent,
    seed: int,
    n_episodes: int = EVAL_EPS,
) -> Dict[str, Any]:
    env = StudentEnv(
        population_seed=seed,
        obs_ablation="full_emotion",
        emotion_dynamics="full",
    )
    cfg.set_all_seeds(seed)
    episode_results: List[Dict[str, Any]] = []

    for ep in range(1, n_episodes + 1):
        obs, info = env.reset(seed=seed + ep)
        mask = info["action_masks"]
        start_k = env._state.knowledge
        total_reward = 0.0
        adapt_hits = adapt_total = 0
        action_counts = {i: 0 for i in range(ACTION_DIM)}
        dropout = False
        done = False

        while not done:
            eid = env._state.emotion_id
            action = agent.predict(obs, mask)
            obs, reward, term, trunc, info = env.step(action)
            adapt_total += 1
            if _adaptation_match(eid, action):
                adapt_hits += 1
            total_reward += reward
            action_counts[action] += 1
            mask = info["action_masks"]
            done = term or trunc
            dropout = info.get("dropout", False)

        final = env._state
        episode_results.append({
            "learning_gain": normalized_knowledge_gain(start_k, final.knowledge),
            "final_knowledge": final.knowledge,
            "success": int(_is_success(final.knowledge, final.frustration, final.confusion)),
            "adaptation_accuracy": adapt_hits / max(adapt_total, 1),
            "total_reward": total_reward,
            "dropout": int(dropout),
            "action_counts": action_counts,
        })

    env.close()
    freq_report = action_frequency_report(episode_results, print_table=False)
    freqs = freq_report["frequencies"]

    return {
        "learning_gain": float(np.mean([r["learning_gain"] for r in episode_results])),
        "final_knowledge": float(np.mean([r["final_knowledge"] for r in episode_results])),
        "success_rate": float(np.mean([r["success"] for r in episode_results])),
        "adaptation_accuracy": float(np.mean([r["adaptation_accuracy"] for r in episode_results])),
        "mean_episode_reward": float(np.mean([r["total_reward"] for r in episode_results])),
        "dropout_rate": float(np.mean([r["dropout"] for r in episode_results])),
        "action_frequencies": {k: round(v, 6) for k, v in freqs.items()},
        "action_entropy": _shannon_entropy(freqs),
        "actions_used": _actions_used_at_least(freqs),
        "dominance_detected": freq_report["dominance_detected"],
    }


def run_single(configuration: str, seed: int) -> Dict[str, Any]:
    weights = REWARD_CONFIGS[configuration]
    gain_snap = _patch_gain()
    reward_snap = _patch_reward(weights)
    tag = f"reward_ablation_{configuration}_s{seed}"
    t0 = time.time()

    try:
        cfg.set_all_seeds(seed)
        train_env = make_env(
            seed=seed,
            obs_ablation="full_emotion",
            emotion_dynamics="full",
            algo_tag=tag,
        )
        agent = DQNAgent()
        agent.train(train_env, TRAIN_TS, seed, hyperparams=BASELINE_HP)
        train_env.close()
        agent.save(str(cfg.MODELS_DIR / tag))

        metrics = evaluate_dqn(agent, seed, EVAL_EPS)
    finally:
        _restore_reward(reward_snap)
        _restore_gain(gain_snap)

    elapsed = time.time() - t0
    row: Dict[str, Any] = {
        "configuration": configuration,
        "seed": seed,
        "train_timesteps": TRAIN_TS,
        "eval_episodes": EVAL_EPS,
        "gain_ratio": G4_GAIN["ratio_label"],
        "elapsed_seconds": round(elapsed, 1),
        **{k: weights[k] for k in ("wk", "we", "wf", "wb", "wc")},
        **{m: metrics[m] for m in METRICS},
        "action_entropy": metrics["action_entropy"],
        "actions_used": metrics["actions_used"],
        "dominance_detected": metrics["dominance_detected"],
    }
    for action_name, freq in metrics["action_frequencies"].items():
        row[f"freq_{action_name}"] = freq

    print(
        f"  {configuration} seed={seed}: lg={row['learning_gain']:.3f} "
        f"succ={row['success_rate']:.3f} adapt={row['adaptation_accuracy']:.3f} "
        f"entropy={row['action_entropy']:.3f} actions_used={row['actions_used']} "
        f"({elapsed:.0f}s)"
    )
    return row


def _load_resume(path: Path, resume: bool) -> List[Dict[str, Any]]:
    if resume and path.exists():
        return pd.read_csv(path).to_dict("records")
    return []


def run_study(seeds: List[int], resume: bool) -> pd.DataFrame:
    csv_path = OUT_DIR / "reward_weight_results.csv"
    rows = _load_resume(csv_path, resume)
    done = {(r["configuration"], int(r["seed"])) for r in rows}

    for configuration in REWARD_CONFIGS:
        print(f"\n--- Reward config: {configuration} ---")
        for seed in seeds:
            if (configuration, seed) in done:
                continue
            rows.append(run_single(configuration, seed))
            pd.DataFrame(rows).to_csv(csv_path, index=False)

    return pd.DataFrame(rows)


def _cv(values: List[float]) -> float:
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return std / mean if abs(mean) > 1e-9 else float("nan")


def aggregate_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for configuration, sub in results_df.groupby("configuration"):
        row: Dict[str, Any] = {"configuration": configuration}
        for metric in AGG_METRICS:
            vals = sub[metric].astype(float).tolist()
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            row[f"{metric}_mean"] = round(mean, 4)
            row[f"{metric}_std"] = round(std, 4)
            row[f"{metric}_cv"] = round(_cv(vals), 4)
        row["n_seeds"] = len(sub)
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_action_diversity(results_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    action_names = list(ID_TO_ACTION.values())

    for configuration, sub in results_df.groupby("configuration"):
        entropies = sub["action_entropy"].astype(float).tolist()
        actions_used = sub["actions_used"].astype(int).tolist()
        row: Dict[str, Any] = {
            "configuration": configuration,
            "action_entropy_mean": round(float(np.mean(entropies)), 4),
            "action_entropy_std": round(float(np.std(entropies, ddof=1)) if len(entropies) > 1 else 0.0, 4),
            "actions_used_mean": round(float(np.mean(actions_used)), 4),
            "actions_used_std": round(float(np.std(actions_used, ddof=1)) if len(actions_used) > 1 else 0.0, 4),
            "dominance_any_seed": bool(sub["dominance_detected"].any()),
        }
        for action_name in action_names:
            col = f"freq_{action_name}"
            if col in sub.columns:
                row[f"{action_name}_mean_freq"] = round(float(sub[col].mean()), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def build_ranking(summary_df: pd.DataFrame, diversity_df: pd.DataFrame) -> pd.DataFrame:
    merged = summary_df.merge(
        diversity_df[["configuration", "action_entropy_mean", "actions_used_mean"]],
        on="configuration",
    )
    merged = merged.rename(columns={
        "action_entropy_mean": "action_entropy",
        "actions_used_mean": "actions_used",
        "adaptation_accuracy_mean": "adaptation_accuracy_mean",
        "mean_episode_reward_mean": "reward_mean",
    })

    rank_cols = {
        "learning_gain_mean": False,
        "learning_gain_std": True,
        "success_rate_mean": False,
        "adaptation_accuracy_mean": False,
        "reward_mean": False,
        "action_entropy": False,
        "actions_used": False,
    }
    score = pd.Series(0.0, index=merged.index)
    for col, ascending in rank_cols.items():
        if col in merged.columns:
            score += merged[col].rank(ascending=ascending, method="average")
    merged["rank"] = score.rank(method="min").astype(int)

    ranking = merged[[
        "configuration",
        "learning_gain_mean",
        "learning_gain_std",
        "success_rate_mean",
        "adaptation_accuracy_mean",
        "reward_mean",
        "action_entropy",
        "actions_used",
        "rank",
    ]].sort_values("rank")
    return ranking


def _plot_comparison(summary_df: pd.DataFrame) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    configs = summary_df["configuration"].tolist()
    colors = {"baseline_weights": "#457B9D", "equal_weights": "#E8A598"}
    metrics_plot = [
        ("learning_gain_mean", "learning_gain_std", "Learning Gain"),
        ("success_rate_mean", "success_rate_std", "Success Rate"),
        ("adaptation_accuracy_mean", "adaptation_accuracy_std", "Adaptation Accuracy"),
        ("mean_episode_reward_mean", "mean_episode_reward_std", "Mean Episode Reward"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    xs = np.arange(len(configs))
    for ax, (mean_col, std_col, title) in zip(axes.flat, metrics_plot):
        means = summary_df[mean_col].tolist()
        stds = summary_df[std_col].tolist()
        ax.bar(
            xs,
            means,
            yerr=stds,
            capsize=5,
            color=[colors.get(c, "#888888") for c in configs],
            edgecolor="#64748B",
        )
        ax.set_xticks(xs)
        ax.set_xticklabels(["Baseline\n(thesis)", "Equal\nweights"], fontsize=9)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Reward Weight Ablation (DQN, G4 {G4_GAIN['ratio_label']}, {TRAIN_TS // 1000}k steps)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "reward_weight_comparison.png", dpi=150)
    plt.close(fig)


def _plot_action_distribution(results_df: pd.DataFrame, diversity_df: pd.DataFrame) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    action_names = list(ID_TO_ACTION.values())
    configs = list(REWARD_CONFIGS.keys())
    x = np.arange(len(action_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, configuration in enumerate(configs):
        div_row = diversity_df[diversity_df["configuration"] == configuration].iloc[0]
        freqs = [div_row.get(f"{name}_mean_freq", 0.0) for name in action_names]
        offset = (i - 0.5) * width
        ax.bar(x + offset, freqs, width, label=configuration.replace("_", " "), edgecolor="#64748B")

    ax.set_xticks(x)
    ax.set_xticklabels(action_names, rotation=35, ha="right")
    ax.set_ylabel("Mean action frequency")
    ax.set_title("Action Distribution by Reward Configuration")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "action_distribution_comparison.png", dpi=150)
    plt.close(fig)


def _print_winner(
    summary_df: pd.DataFrame,
    diversity_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
) -> None:
    best_perf = summary_df.loc[summary_df["learning_gain_mean"].idxmax()]
    most_stable = summary_df.loc[summary_df["learning_gain_std"].idxmin()]
    best_diversity = diversity_df.loc[diversity_df["action_entropy_mean"].idxmax()]
    winner = ranking_df.iloc[0]

    baseline = summary_df[summary_df["configuration"] == "baseline_weights"].iloc[0]
    equal = summary_df[summary_df["configuration"] == "equal_weights"].iloc[0]
    base_div = diversity_df[diversity_df["configuration"] == "baseline_weights"].iloc[0]
    equal_div = diversity_df[diversity_df["configuration"] == "equal_weights"].iloc[0]

    eq_improves_stability = equal["learning_gain_std"] < baseline["learning_gain_std"]
    eq_improves_diversity = equal_div["action_entropy_mean"] > base_div["action_entropy_mean"]
    eq_improves_lg = equal["learning_gain_mean"] > baseline["learning_gain_mean"]
    eq_improves_success = equal["success_rate_mean"] > baseline["success_rate_mean"]

    justified = (
        baseline["learning_gain_mean"] >= equal["learning_gain_mean"]
        and baseline["learning_gain_std"] <= equal["learning_gain_std"]
    ) or winner["configuration"] == "baseline_weights"

    print("\n" + "=" * 72)
    print("WINNER CONFIGURATION")
    print("=" * 72)
    print(f"\nOverall rank #1: {winner['configuration']}")
    print(f"  learning_gain_mean={winner['learning_gain_mean']:.4f} "
          f"(std={winner['learning_gain_std']:.4f})")
    print(f"  success_rate_mean={winner['success_rate_mean']:.4f}")
    print(f"  adaptation_accuracy_mean={winner['adaptation_accuracy_mean']:.4f}")
    print(f"  action_entropy={winner['action_entropy']:.4f}, actions_used={winner['actions_used']:.1f}")

    print(f"\nBest learning gain: {best_perf['configuration']} "
          f"(mean={best_perf['learning_gain_mean']:.4f})")
    print(f"Most stable (lowest LG std): {most_stable['configuration']} "
          f"(std={most_stable['learning_gain_std']:.4f})")
    print(f"Highest action diversity: {best_diversity['configuration']} "
          f"(entropy={best_diversity['action_entropy_mean']:.4f})")

    print("\nEqual weights vs baseline thesis weights:")
    print(f"  Stability (lower std): {'equal_weights' if eq_improves_stability else 'baseline_weights'}")
    print(f"  Action diversity: {'equal_weights' if eq_improves_diversity else 'baseline_weights'}")
    print(f"  Learning gain: {'equal_weights' if eq_improves_lg else 'baseline_weights'}")
    print(f"  Success rate: {'equal_weights' if eq_improves_success else 'baseline_weights'}")

    if justified:
        print(
            "\nThe original thesis reward weights appear justified: they match or exceed "
            "equal weights on learning outcomes and/or seed stability."
        )
    else:
        print(
            "\nEqual weights may be preferable: they improve stability, diversity, or "
            "performance relative to the thesis baseline weights."
        )
    print("=" * 72)


def analyze(results_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    csv_path = OUT_DIR / "reward_weight_results.csv"
    if results_df is None:
        if not csv_path.exists():
            raise FileNotFoundError(f"No results at {csv_path}; run --phase run first.")
        results_df = pd.read_csv(csv_path)

    summary_df = aggregate_summary(results_df)
    diversity_df = aggregate_action_diversity(results_df)
    ranking_df = build_ranking(summary_df, diversity_df)

    summary_df.to_csv(OUT_DIR / "reward_weight_summary.csv", index=False)
    diversity_df.to_csv(OUT_DIR / "action_diversity_summary.csv", index=False)
    ranking_df.to_csv(OUT_DIR / "reward_weight_ranking.csv", index=False)

    _plot_comparison(summary_df)
    _plot_action_distribution(results_df, diversity_df)
    _print_winner(summary_df, diversity_df, ranking_df)

    report = {
        "study": "reward_weight_ablation",
        "gain_ratio": G4_GAIN["ratio_label"],
        "baseline_hp": BASELINE_HP,
        "reward_configs": REWARD_CONFIGS,
        "seeds": SEEDS,
        "summary": summary_df.to_dict("records"),
        "action_diversity": diversity_df.to_dict("records"),
        "ranking": ranking_df.to_dict("records"),
    }
    with open(OUT_DIR / "reward_weight_ablation_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Reward weight ablation for DQN stability")
    parser.add_argument(
        "--phase",
        choices=["run", "analyze", "all"],
        default="all",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    seeds = SEEDS
    if args.quick:
        seeds = [42, 7]
        print(f"QUICK MODE: seeds={seeds}")

    if args.phase in ("run", "all"):
        print("=== Reward Weight Ablation (run) ===")
        run_study(seeds, args.resume)

    if args.phase in ("analyze", "all"):
        print("\n=== Reward Weight Ablation (analyze) ===")
        analyze()


if __name__ == "__main__":
    main()
