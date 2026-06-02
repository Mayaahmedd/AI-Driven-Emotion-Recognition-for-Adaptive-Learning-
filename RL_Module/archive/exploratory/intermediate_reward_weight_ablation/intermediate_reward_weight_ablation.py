"""
Intermediate reward weight ablation: thesis weights vs balanced vs equal.

Evaluates four reward weight configurations on DQN to find the best
performance/stability tradeoff between original thesis weights and equal weights.

Configurations:
  A: Original Thesis   wk=0.50, we=0.20, wf=0.15, wb=0.10, wc=0.05
  B: Balanced-1        wk=0.40, we=0.20, wf=0.15, wb=0.125, wc=0.125
  C: Balanced-2        wk=0.35, we=0.20, wf=0.15, wb=0.15, wc=0.15
  D: Equal             wk=0.20, we=0.20, wf=0.20, wb=0.20, wc=0.20

Protocol: DQN only, G4 gain ratio, full_emotion, mixed population, default transition
noise, checkpoint selection, 50k train, 500 eval, 6 seeds [42, 7, 13, 21, 99, 314],
dqn_stability_study baseline hyperparameters.

Usage (from repo root):
  export PYTHONPATH="$(pwd)"
  python -m RL_Module.intermediate_reward_weight_ablation --phase all
  python -m RL_Module.intermediate_reward_weight_ablation --phase run --resume
  python -m RL_Module.intermediate_reward_weight_ablation --phase analyze
  python -m RL_Module.intermediate_reward_weight_ablation --phase all --quick
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

OUT_DIR = _HERE / "figures" / "intermediate_reward_weight_ablation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 7, 13, 21, 99, 314]
TRAIN_TS = 50_000
EVAL_EPS = 500
VAL_EPS = 100
CHECKPOINT_STEPS = [10_000, 20_000, 30_000, 40_000, 50_000]

G4_GAIN = {
    "GAIN_CORRECT_FACTOR": 0.1,
    "GAIN_INCORRECT_FACTOR": 1.0,
    "ratio_label": "10:1",
}

BASELINE_HP: Dict[str, Any] = {
    "learning_rate": 5e-4,
    "batch_size": 64,
    "buffer_size": 100_000,
    "target_update_interval": 2000,
    "exploration_fraction": 0.3,
    "exploration_final_eps": 0.05,
    "net_arch": [64, 64],
}

REWARD_CONFIGS: Dict[str, Dict[str, Any]] = {
    "A_original_thesis": {
        "label": "A: Original Thesis",
        "wk": 0.50,
        "we": 0.20,
        "wf": 0.15,
        "wb": 0.10,
        "wc": 0.05,
    },
    "B_balanced_1": {
        "label": "B: Balanced-1",
        "wk": 0.40,
        "we": 0.20,
        "wf": 0.15,
        "wb": 0.125,
        "wc": 0.125,
    },
    "C_balanced_2": {
        "label": "C: Balanced-2",
        "wk": 0.35,
        "we": 0.20,
        "wf": 0.15,
        "wb": 0.15,
        "wc": 0.15,
    },
    "D_equal": {
        "label": "D: Equal",
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


def _actions_used_above_1pct(freq: Dict[str, float], threshold: float = 0.01) -> int:
    return sum(1 for v in freq.values() if v >= threshold)


def _dominant_action_pct(freq: Dict[str, float]) -> float:
    return float(max(freq.values())) if freq else 0.0


def quick_eval_lg(agent: DQNAgent, seed: int, n_episodes: int = VAL_EPS) -> float:
    env = StudentEnv(
        population_seed=seed,
        obs_ablation="full_emotion",
        emotion_dynamics="full",
    )
    cfg.set_all_seeds(seed)
    gains: List[float] = []
    for ep in range(1, n_episodes + 1):
        obs, info = env.reset(seed=seed + ep)
        mask = info["action_masks"]
        start_k = env._state.knowledge
        done = False
        while not done:
            action = agent.predict(obs, mask)
            obs, _, term, trunc, info = env.step(action)
            mask = info["action_masks"]
            done = term or trunc
        gains.append(normalized_knowledge_gain(start_k, env._state.knowledge))
    env.close()
    return float(np.mean(gains))


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
        "actions_used_above_1_percent": _actions_used_above_1pct(freqs),
        "dominant_action_pct": _dominant_action_pct(freqs),
        "dominance_detected": freq_report["dominance_detected"],
    }


def run_single(configuration: str, seed: int) -> Dict[str, Any]:
    spec = REWARD_CONFIGS[configuration]
    weights = {k: spec[k] for k in ("wk", "we", "wf", "wb", "wc")}
    gain_snap = _patch_gain()
    reward_snap = _patch_reward(weights)
    tag = f"intermediate_reward_{configuration}_s{seed}"
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
        ckpt_dir = cfg.MODELS_DIR / tag
        ckpt_freq = min(10_000, TRAIN_TS)
        paths = agent.train_with_checkpoints(
            train_env,
            TRAIN_TS,
            seed,
            hyperparams=BASELINE_HP,
            checkpoint_dir=str(ckpt_dir),
            checkpoint_freq=ckpt_freq,
        )
        train_env.close()

        valid_paths = [
            p for p in paths
            if any(f"_{step}_steps" in p for step in CHECKPOINT_STEPS)
        ]
        if not valid_paths:
            valid_paths = paths

        best_path, best_val = agent.select_best_checkpoint(
            valid_paths,
            lambda a, s: quick_eval_lg(a, s, VAL_EPS),
            seed,
        )
        agent.load_checkpoint(best_path)
        metrics = evaluate_dqn(agent, seed, EVAL_EPS)
        selection = f"checkpoint@{best_path}"
    finally:
        _restore_reward(reward_snap)
        _restore_gain(gain_snap)

    elapsed = time.time() - t0
    row: Dict[str, Any] = {
        "configuration": configuration,
        "configuration_label": spec["label"],
        "seed": seed,
        "train_timesteps": TRAIN_TS,
        "eval_episodes": EVAL_EPS,
        "use_checkpoints": True,
        "selection": selection,
        "val_learning_gain": best_val,
        "gain_ratio": G4_GAIN["ratio_label"],
        "elapsed_seconds": round(elapsed, 1),
        **weights,
        **{m: metrics[m] for m in METRICS},
        "action_entropy": metrics["action_entropy"],
        "actions_used_above_1_percent": metrics["actions_used_above_1_percent"],
        "dominant_action_pct": metrics["dominant_action_pct"],
        "dominance_detected": metrics["dominance_detected"],
    }
    for action_name, freq in metrics["action_frequencies"].items():
        row[f"freq_{action_name}"] = freq

    print(
        f"  {configuration} seed={seed}: lg={row['learning_gain']:.3f} "
        f"succ={row['success_rate']:.3f} adapt={row['adaptation_accuracy']:.3f} "
        f"entropy={row['action_entropy']:.3f} dom={row['dominant_action_pct']:.2f} "
        f"({elapsed:.0f}s)"
    )
    return row


def _load_resume(path: Path, resume: bool) -> List[Dict[str, Any]]:
    if resume and path.exists():
        return pd.read_csv(path).to_dict("records")
    return []


def run_study(seeds: List[int], resume: bool) -> pd.DataFrame:
    csv_path = OUT_DIR / "intermediate_reward_weight_results.csv"
    rows = _load_resume(csv_path, resume)
    done = {(r["configuration"], int(r["seed"])) for r in rows}

    for configuration in REWARD_CONFIGS:
        print(f"\n--- {REWARD_CONFIGS[configuration]['label']} ---")
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
        spec = REWARD_CONFIGS[configuration]
        lgs = sub["learning_gain"].astype(float).tolist()
        lg_mean, lg_lo, lg_hi = confidence_interval(lgs)
        lg_std = float(np.std(lgs, ddof=1)) if len(lgs) > 1 else 0.0
        lg_range = float(max(lgs) - min(lgs)) if lgs else 0.0
        best_idx = sub["learning_gain"].astype(float).idxmax()
        worst_idx = sub["learning_gain"].astype(float).idxmin()

        row: Dict[str, Any] = {
            "configuration": configuration,
            "configuration_label": spec["label"],
            "wk": spec["wk"],
            "we": spec["we"],
            "wf": spec["wf"],
            "wb": spec["wb"],
            "wc": spec["wc"],
            "n_seeds": len(sub),
            "learning_gain_mean": round(lg_mean, 4),
            "learning_gain_std": round(lg_std, 4),
            "coefficient_of_variation": round(_cv(lgs), 4),
            "seed_range": round(lg_range, 4),
            "learning_gain_ci": f"[{lg_lo:.3f}, {lg_hi:.3f}]",
            "best_seed": int(sub.loc[best_idx, "seed"]),
            "worst_seed": int(sub.loc[worst_idx, "seed"]),
            "best_seed_lg": round(float(sub.loc[best_idx, "learning_gain"]), 4),
            "worst_seed_lg": round(float(sub.loc[worst_idx, "learning_gain"]), 4),
        }
        for metric in METRICS:
            if metric == "learning_gain":
                continue
            vals = sub[metric].astype(float).tolist()
            row[f"{metric}_mean"] = round(float(np.mean(vals)), 4)
            row[f"{metric}_std"] = round(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, 4)
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_action_diversity(results_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    action_names = list(ID_TO_ACTION.values())

    for configuration, sub in results_df.groupby("configuration"):
        spec = REWARD_CONFIGS[configuration]
        entropies = sub["action_entropy"].astype(float).tolist()
        actions_used = sub["actions_used_above_1_percent"].astype(int).tolist()
        dominant = sub["dominant_action_pct"].astype(float).tolist()
        row: Dict[str, Any] = {
            "configuration": configuration,
            "configuration_label": spec["label"],
            "shannon_entropy_mean": round(float(np.mean(entropies)), 4),
            "shannon_entropy_std": round(float(np.std(entropies, ddof=1)) if len(entropies) > 1 else 0.0, 4),
            "actions_used_above_1_percent_mean": round(float(np.mean(actions_used)), 4),
            "actions_used_above_1_percent_std": round(
                float(np.std(actions_used, ddof=1)) if len(actions_used) > 1 else 0.0, 4
            ),
            "dominant_action_pct_mean": round(float(np.mean(dominant)), 4),
            "dominant_action_pct_std": round(
                float(np.std(dominant, ddof=1)) if len(dominant) > 1 else 0.0, 4
            ),
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
        diversity_df[[
            "configuration",
            "shannon_entropy_mean",
            "dominant_action_pct_mean",
            "actions_used_above_1_percent_mean",
        ]],
        on="configuration",
    )
    merged = merged.rename(columns={
        "shannon_entropy_mean": "action_entropy",
        "dominant_action_pct_mean": "dominant_action_pct",
        "actions_used_above_1_percent_mean": "actions_used_above_1pct",
        "mean_episode_reward_mean": "reward_mean",
    })

    score = pd.Series(0.0, index=merged.index)
    rank_specs = {
        "learning_gain_mean": False,
        "learning_gain_std": True,
        "coefficient_of_variation": True,
        "seed_range": True,
        "success_rate_mean": False,
        "adaptation_accuracy_mean": False,
        "reward_mean": False,
        "action_entropy": False,
        "dominant_action_pct": True,
        "actions_used_above_1pct": False,
    }
    for col, ascending in rank_specs.items():
        if col in merged.columns:
            score += merged[col].rank(ascending=ascending, method="average")
    merged["rank"] = score.rank(method="min").astype(int)

    ranking = merged[[
        "configuration",
        "configuration_label",
        "learning_gain_mean",
        "learning_gain_std",
        "coefficient_of_variation",
        "seed_range",
        "success_rate_mean",
        "adaptation_accuracy_mean",
        "reward_mean",
        "action_entropy",
        "dominant_action_pct",
        "actions_used_above_1pct",
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

    order = list(REWARD_CONFIGS.keys())
    summary_df = summary_df.set_index("configuration").loc[order].reset_index()
    labels = [REWARD_CONFIGS[c]["label"].replace(": ", "\n") for c in order]
    colors = ["#457B9D", "#2A9D8F", "#E9C46A", "#E76F51"]
    xs = np.arange(len(labels))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    panels = [
        ("learning_gain_mean", "learning_gain_std", "Learning Gain"),
        ("success_rate_mean", "success_rate_std", "Success Rate"),
        ("adaptation_accuracy_mean", "adaptation_accuracy_std", "Adaptation Accuracy"),
        ("mean_episode_reward_mean", "mean_episode_reward_std", "Mean Episode Reward"),
    ]
    for ax, (mean_col, std_col, title) in zip(axes.flat, panels):
        means = summary_df[mean_col].tolist()
        stds = summary_df[std_col].tolist() if std_col in summary_df.columns else [0.0] * len(means)
        ax.bar(xs, means, yerr=stds, capsize=5, color=colors, edgecolor="#64748B")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Intermediate Reward Weight Ablation (DQN, G4 {G4_GAIN['ratio_label']}, "
        f"{TRAIN_TS // 1000}k steps, checkpoint selection)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "intermediate_reward_weight_comparison.png", dpi=150)
    plt.close(fig)


def _plot_action_distribution(diversity_df: pd.DataFrame) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    action_names = list(ID_TO_ACTION.values())
    configs = list(REWARD_CONFIGS.keys())
    x = np.arange(len(action_names))
    width = 0.18

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, configuration in enumerate(configs):
        div_row = diversity_df[diversity_df["configuration"] == configuration].iloc[0]
        freqs = [div_row.get(f"{name}_mean_freq", 0.0) for name in action_names]
        offset = (i - 1.5) * width
        ax.bar(
            x + offset,
            freqs,
            width,
            label=REWARD_CONFIGS[configuration]["label"],
            edgecolor="#64748B",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(action_names, rotation=35, ha="right")
    ax.set_ylabel("Mean action frequency")
    ax.set_title("Action Distribution by Reward Configuration")
    ax.legend(fontsize=8)
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
    lowest_cv = summary_df.loc[summary_df["coefficient_of_variation"].idxmin()]
    best_diversity = diversity_df.loc[diversity_df["shannon_entropy_mean"].idxmax()]
    winner = ranking_df.iloc[0]

    print("\n" + "=" * 72)
    print("WINNER REWARD CONFIGURATION")
    print("=" * 72)

    print(f"\nBest overall tradeoff (rank #1): {winner['configuration_label']}")
    print(f"  weights: wk={REWARD_CONFIGS[winner['configuration']]['wk']}, "
          f"we={REWARD_CONFIGS[winner['configuration']]['we']}, "
          f"wf={REWARD_CONFIGS[winner['configuration']]['wf']}, "
          f"wb={REWARD_CONFIGS[winner['configuration']]['wb']}, "
          f"wc={REWARD_CONFIGS[winner['configuration']]['wc']}")
    print(f"  learning_gain_mean={winner['learning_gain_mean']:.4f} "
          f"(std={winner['learning_gain_std']:.4f}, CV={winner['coefficient_of_variation']:.4f})")
    print(f"  success_rate_mean={winner['success_rate_mean']:.4f}")
    print(f"  adaptation_accuracy_mean={winner['adaptation_accuracy_mean']:.4f}")
    print(f"  reward_mean={winner['reward_mean']:.4f}")
    print(f"  action_entropy={winner['action_entropy']:.4f}, "
          f"dominant_action_pct={winner['dominant_action_pct']:.4f}, "
          f"actions_used_above_1%={winner['actions_used_above_1pct']:.1f}")

    print(f"\nBest performance (highest learning gain): {best_perf['configuration_label']}")
    print(f"  learning_gain_mean={best_perf['learning_gain_mean']:.4f} "
          f"(seed_range={best_perf['seed_range']:.4f})")

    print(f"\nBest stability (lowest learning_gain_std): {most_stable['configuration_label']}")
    print(f"  learning_gain_std={most_stable['learning_gain_std']:.4f}, "
          f"CV={most_stable['coefficient_of_variation']:.4f}, "
          f"seed_range={most_stable['seed_range']:.4f}")

    print(f"\nLowest coefficient of variation: {lowest_cv['configuration_label']}")
    print(f"  CV={lowest_cv['coefficient_of_variation']:.4f}")

    print(f"\nHighest action diversity (Shannon entropy): {best_diversity['configuration_label']}")
    print(f"  entropy={best_diversity['shannon_entropy_mean']:.4f}, "
          f"dominant_action_pct={best_diversity['dominant_action_pct_mean']:.4f}")

    print("\n--- Configuration Summary ---")
    for _, row in ranking_df.iterrows():
        print(
            f"  {row['configuration_label']}: "
            f"LG={row['learning_gain_mean']:.4f}+/-{row['learning_gain_std']:.4f}, "
            f"rank={int(row['rank'])}"
        )
    print("=" * 72)


def analyze(results_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    csv_path = OUT_DIR / "intermediate_reward_weight_results.csv"
    if results_df is None:
        if not csv_path.exists():
            raise FileNotFoundError(f"No results at {csv_path}; run --phase run first.")
        results_df = pd.read_csv(csv_path)

    summary_df = aggregate_summary(results_df)
    diversity_df = aggregate_action_diversity(results_df)
    ranking_df = build_ranking(summary_df, diversity_df)

    summary_df.to_csv(OUT_DIR / "intermediate_reward_weight_summary.csv", index=False)
    diversity_df.to_csv(OUT_DIR / "action_diversity_summary.csv", index=False)
    ranking_df.to_csv(OUT_DIR / "intermediate_reward_weight_ranking.csv", index=False)

    _plot_comparison(summary_df)
    _plot_action_distribution(diversity_df)
    _print_winner(summary_df, diversity_df, ranking_df)

    report = {
        "study": "intermediate_reward_weight_ablation",
        "gain_ratio": G4_GAIN["ratio_label"],
        "baseline_hp": BASELINE_HP,
        "reward_configs": {
            k: {kk: vv for kk, vv in v.items() if kk != "label"}
            for k, v in REWARD_CONFIGS.items()
        },
        "seeds": SEEDS,
        "best_performance": summary_df.loc[summary_df["learning_gain_mean"].idxmax()]["configuration"],
        "best_stability": summary_df.loc[summary_df["learning_gain_std"].idxmin()]["configuration"],
        "best_overall_tradeoff": ranking_df.iloc[0]["configuration"],
        "summary": summary_df.to_dict("records"),
        "action_diversity": diversity_df.to_dict("records"),
        "ranking": ranking_df.to_dict("records"),
    }
    with open(OUT_DIR / "intermediate_reward_weight_ablation_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    return report


def main() -> None:
    global TRAIN_TS

    parser = argparse.ArgumentParser(
        description="Intermediate reward weight ablation for DQN performance vs stability"
    )
    parser.add_argument("--phase", choices=["run", "analyze", "all"], default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    seeds = SEEDS
    train_ts = TRAIN_TS
    if args.quick:
        seeds = [42, 7]
        train_ts = 5_000
        print(f"QUICK MODE: seeds={seeds}, train={train_ts}")

    if args.phase in ("run", "all"):
        print("=== Intermediate Reward Weight Ablation (run) ===")
        TRAIN_TS = train_ts
        run_study(seeds, args.resume)

    if args.phase in ("analyze", "all"):
        print("\n=== Intermediate Reward Weight Ablation (analyze) ===")
        analyze()


if __name__ == "__main__":
    main()
