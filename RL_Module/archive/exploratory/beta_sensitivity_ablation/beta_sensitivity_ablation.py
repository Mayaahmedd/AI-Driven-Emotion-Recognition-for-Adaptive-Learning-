"""
Correctness sensitivity (beta) ablation: does IRT correctness stochasticity drive DQN instability?

Compares three IRT_BETA levels (runtime patch via SIMULATOR_PARAMS only):
  A beta_2: beta = 2.0  (flatter / more random student)
  B beta_3: beta = 3.0  (repository default)
  C beta_5: beta = 5.0  (sharper / more deterministic student)

Protocol: DQN only, G4 gain ratio, full_emotion, mixed population, default transition noise,
checkpoint selection, 50k train, 500 eval, 6 seeds [42, 7, 13, 21, 99, 314],
dqn_stability_study baseline hyperparameters.

Usage (from repo root):
  export PYTHONPATH="$(pwd)"
  python -m RL_Module.beta_sensitivity_ablation --phase all
  python -m RL_Module.beta_sensitivity_ablation --phase run --resume
  python -m RL_Module.beta_sensitivity_ablation --phase analyze
  python -m RL_Module.beta_sensitivity_ablation --phase all --quick
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from RL_Module.mdp_definition import ACTION_DIM, ID_TO_ACTION, IRT_BETA

OUT_DIR = _HERE / "figures" / "beta_sensitivity_ablation"
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

DEFAULT_BETA = float(IRT_BETA)

BETA_CONDITIONS: Dict[str, Dict[str, Any]] = {
    "A_beta_2": {
        "label": "Beta 2.0 - More Random (A)",
        "irt_beta": 2.0,
    },
    "B_beta_3": {
        "label": "Beta 3.0 - Current Simulator (B)",
        "irt_beta": 3.0,
    },
    "C_beta_5": {
        "label": "Beta 5.0 - More Deterministic (C)",
        "irt_beta": 5.0,
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


def _patch_simulator(irt_beta: float) -> Dict[str, float]:
    prev = dict(cfg.SIMULATOR_PARAMS)
    cfg.SIMULATOR_PARAMS = {
        "GAIN_CORRECT_FACTOR": G4_GAIN["GAIN_CORRECT_FACTOR"],
        "GAIN_INCORRECT_FACTOR": G4_GAIN["GAIN_INCORRECT_FACTOR"],
        "IRT_BETA": float(irt_beta),
    }
    return prev


def _restore_simulator(snapshot: Dict[str, float]) -> None:
    cfg.SIMULATOR_PARAMS = snapshot


def _shannon_entropy(freq: Dict[str, float]) -> float:
    p = np.array([v for v in freq.values() if v > 0], dtype=float)
    return float(-np.sum(p * np.log(p))) if len(p) else 0.0


def _actions_used_above_1pct(freq: Dict[str, float], threshold: float = 0.01) -> int:
    return sum(1 for v in freq.values() if v >= threshold)


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
        "dominance_detected": freq_report["dominance_detected"],
    }


def run_single(condition_key: str, seed: int) -> Dict[str, Any]:
    spec = BETA_CONDITIONS[condition_key]
    irt_beta = float(spec["irt_beta"])
    sim_snap = _patch_simulator(irt_beta)
    tag = f"beta_sensitivity_{condition_key}_s{seed}"
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
        _restore_simulator(sim_snap)

    elapsed = time.time() - t0
    row: Dict[str, Any] = {
        "condition": condition_key,
        "condition_label": spec["label"],
        "irt_beta": irt_beta,
        "seed": seed,
        "train_timesteps": TRAIN_TS,
        "eval_episodes": EVAL_EPS,
        "use_checkpoints": True,
        "selection": selection,
        "val_learning_gain": best_val,
        "gain_ratio": G4_GAIN["ratio_label"],
        "elapsed_seconds": round(elapsed, 1),
        **{m: metrics[m] for m in METRICS},
        "action_entropy": metrics["action_entropy"],
        "actions_used_above_1_percent": metrics["actions_used_above_1_percent"],
        "dominance_detected": metrics["dominance_detected"],
    }
    for action_name, freq in metrics["action_frequencies"].items():
        row[f"freq_{action_name}"] = freq

    print(
        f"  {condition_key} seed={seed} beta={irt_beta:.1f}: "
        f"lg={row['learning_gain']:.3f} succ={row['success_rate']:.3f} "
        f"adapt={row['adaptation_accuracy']:.3f} ({elapsed:.0f}s)"
    )
    return row


def _load_resume(path: Path, resume: bool) -> List[Dict[str, Any]]:
    if resume and path.exists():
        return pd.read_csv(path).to_dict("records")
    return []


def run_study(seeds: List[int], resume: bool) -> pd.DataFrame:
    csv_path = OUT_DIR / "beta_sensitivity_results.csv"
    rows = _load_resume(csv_path, resume)
    done = {(r["condition"], int(r["seed"])) for r in rows}

    for condition_key in BETA_CONDITIONS:
        print(f"\n--- {BETA_CONDITIONS[condition_key]['label']} ---")
        for seed in seeds:
            if (condition_key, seed) in done:
                continue
            rows.append(run_single(condition_key, seed))
            pd.DataFrame(rows).to_csv(csv_path, index=False)

    return pd.DataFrame(rows)


def _cv(values: List[float]) -> float:
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return std / mean if abs(mean) > 1e-9 else float("nan")


def aggregate_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for condition_key, sub in results_df.groupby("condition"):
        spec = BETA_CONDITIONS[condition_key]
        lgs = sub["learning_gain"].astype(float).tolist()
        lg_mean, lg_lo, lg_hi = confidence_interval(lgs)
        lg_std = float(np.std(lgs, ddof=1)) if len(lgs) > 1 else 0.0
        lg_range = float(max(lgs) - min(lgs)) if lgs else 0.0
        best_idx = sub["learning_gain"].astype(float).idxmax()
        worst_idx = sub["learning_gain"].astype(float).idxmin()

        row: Dict[str, Any] = {
            "condition": condition_key,
            "condition_label": spec["label"],
            "irt_beta": spec["irt_beta"],
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


def build_ranking(summary_df: pd.DataFrame) -> pd.DataFrame:
    score = pd.Series(0.0, index=summary_df.index)
    rank_specs = {
        "learning_gain_mean": False,
        "learning_gain_std": True,
        "coefficient_of_variation": True,
        "seed_range": True,
        "success_rate_mean": False,
        "adaptation_accuracy_mean": False,
        "mean_episode_reward_mean": False,
        "dropout_rate_mean": True,
    }
    for col, ascending in rank_specs.items():
        if col in summary_df.columns:
            score += summary_df[col].rank(ascending=ascending, method="average")
    summary_df = summary_df.copy()
    summary_df["composite_rank_score"] = score
    summary_df["rank"] = score.rank(method="min").astype(int)

    ranking = summary_df[[
        "rank",
        "condition",
        "condition_label",
        "irt_beta",
        "learning_gain_mean",
        "learning_gain_std",
        "coefficient_of_variation",
        "seed_range",
        "success_rate_mean",
        "adaptation_accuracy_mean",
        "mean_episode_reward_mean",
        "dropout_rate_mean",
        "best_seed",
        "worst_seed",
    ]].sort_values("rank")
    return ranking


def _plot_comparison(summary_df: pd.DataFrame) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    summary_df = summary_df.sort_values("irt_beta")
    labels = [
        f"{row['condition_label']}\n(beta={row['irt_beta']:.1f})"
        for _, row in summary_df.iterrows()
    ]
    colors = ["#E76F51", "#457B9D", "#2A9D8F"]
    xs = np.arange(len(labels))

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    panels = [
        ("learning_gain_mean", "learning_gain_std", "Learning Gain"),
        ("success_rate_mean", "success_rate_std", "Success Rate"),
        ("adaptation_accuracy_mean", "adaptation_accuracy_std", "Adaptation Accuracy"),
        ("mean_episode_reward_mean", "mean_episode_reward_std", "Mean Episode Reward"),
        ("dropout_rate_mean", "dropout_rate_std", "Dropout Rate"),
        ("coefficient_of_variation", None, "Learning Gain CV (stability)"),
    ]
    for ax, (mean_col, std_col, title) in zip(axes.flat, panels):
        means = summary_df[mean_col].tolist()
        stds = (
            summary_df[std_col].tolist()
            if std_col and std_col in summary_df.columns
            else [0.0] * len(means)
        )
        ax.bar(xs, means, yerr=stds if std_col else None, capsize=5, color=colors[: len(xs)], edgecolor="#64748B")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Correctness Beta Ablation (DQN, G4 {G4_GAIN['ratio_label']}, "
        f"{TRAIN_TS // 1000}k steps, checkpoint selection, mixed population)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "beta_sensitivity_comparison.png", dpi=150)
    plt.close(fig)


def _print_winner(summary_df: pd.DataFrame, ranking_df: pd.DataFrame) -> None:
    beta2 = summary_df[summary_df["condition"] == "A_beta_2"].iloc[0]
    beta3 = summary_df[summary_df["condition"] == "B_beta_3"].iloc[0]
    beta5 = summary_df[summary_df["condition"] == "C_beta_5"].iloc[0]
    winner = ranking_df.iloc[0]

    most_stable = summary_df.loc[summary_df["learning_gain_std"].idxmin()]
    best_perf = summary_df.loc[summary_df["learning_gain_mean"].idxmax()]
    lowest_cv = summary_df.loc[summary_df["coefficient_of_variation"].idxmin()]

    b3_std = float(beta3["learning_gain_std"])
    b2_std = float(beta2["learning_gain_std"])
    b5_std = float(beta5["learning_gain_std"])
    b3_lg = float(beta3["learning_gain_mean"])
    b2_lg = float(beta2["learning_gain_mean"])
    b5_lg = float(beta5["learning_gain_mean"])
    b3_cv = float(beta3["coefficient_of_variation"])
    b2_cv = float(beta2["coefficient_of_variation"])
    b5_cv = float(beta5["coefficient_of_variation"])

    sharper_more_stable = b5_std < b3_std
    flatter_more_robust = b2_std < b3_std or b2_cv < b3_cv

    print("\n" + "=" * 72)
    print("WINNER BETA CONFIGURATION")
    print("=" * 72)
    print(f"\nOverall rank #1: {winner['condition_label']} (IRT_BETA={winner['irt_beta']:.1f})")
    print(f"  learning_gain_mean={winner['learning_gain_mean']:.4f} "
          f"(std={winner['learning_gain_std']:.4f}, CV={winner['coefficient_of_variation']:.4f})")
    print(f"  seed_range={winner['seed_range']:.4f} "
          f"(best seed {winner['best_seed']}, worst seed {winner['worst_seed']})")
    print(f"  success_rate_mean={winner['success_rate_mean']:.4f}")
    print(f"  adaptation_accuracy_mean={winner['adaptation_accuracy_mean']:.4f}")
    print(f"  mean_episode_reward_mean={winner['mean_episode_reward_mean']:.4f}")
    print(f"  dropout_rate_mean={winner['dropout_rate_mean']:.4f}")

    print(f"\nBest mean learning gain: {best_perf['condition_label']} "
          f"(mean={best_perf['learning_gain_mean']:.4f})")
    print(f"Most stable (lowest LG std): {most_stable['condition_label']} "
          f"(std={most_stable['learning_gain_std']:.4f}, "
          f"CV={most_stable['coefficient_of_variation']:.4f})")
    print(f"Lowest CV: {lowest_cv['condition_label']} "
          f"(CV={lowest_cv['coefficient_of_variation']:.4f})")

    print("\n--- Per-condition summary ---")
    for _, row in summary_df.sort_values("irt_beta").iterrows():
        print(
            f"  beta={row['irt_beta']:.1f}: LG={row['learning_gain_mean']:.4f} "
            f"(std={row['learning_gain_std']:.4f}, CV={row['coefficient_of_variation']:.4f}, "
            f"range={row['seed_range']:.4f}), "
            f"succ={row['success_rate_mean']:.4f}, "
            f"adapt={row['adaptation_accuracy_mean']:.4f}, "
            f"reward={row['mean_episode_reward_mean']:.4f}, "
            f"dropout={row['dropout_rate_mean']:.4f}"
        )

    print("\n--- Findings ---")
    print("\n1. Does a sharper correctness function (beta=5) improve stability?")
    if sharper_more_stable:
        print(
            f"   YES. Beta=5 lowers seed variance vs current beta=3 "
            f"(std: {b5_std:.4f} vs {b3_std:.4f}; CV: {b5_cv:.4f} vs {b3_cv:.4f}). "
            "More deterministic correctness reduces transition stochasticity."
        )
    else:
        print(
            f"   NO. Sharper beta=5 did not reduce learning-gain variance "
            f"(std: {b5_std:.4f} vs baseline {b3_std:.4f}; "
            f"CV: {b5_cv:.4f} vs {b3_cv:.4f}). "
            "Correctness stochasticity is unlikely the primary instability source."
        )

    print("\n2. Does a flatter correctness function (beta=2) improve robustness?")
    if flatter_more_robust and b2_lg >= min(b3_lg, b5_lg) * 0.95:
        print(
            f"   YES (tradeoff). Beta=2 shows lower variance "
            f"(std={b2_std:.4f}, CV={b2_cv:.4f}) with competitive mean LG ({b2_lg:.4f}). "
            "Extra correctness noise may act as regularization."
        )
    elif b2_lg > b3_lg and b2_lg > b5_lg:
        print(
            f"   PARTIALLY. Beta=2 achieves highest mean LG ({b2_lg:.4f}) "
            f"but stability is mixed (std={b2_std:.4f}, CV={b2_cv:.4f})."
        )
    elif flatter_more_robust:
        print(
            f"   PARTIALLY on stability only. Beta=2 reduces variance "
            f"(std={b2_std:.4f} vs {b3_std:.4f}) but mean LG drops ({b2_lg:.4f} vs {b3_lg:.4f})."
        )
    else:
        print(
            f"   NO. Flatter beta=2 does not improve stability or mean performance "
            f"(std={b2_std:.4f}, LG={b2_lg:.4f} vs baseline std={b3_std:.4f}, LG={b3_lg:.4f})."
        )

    print("\n3. Best performance-stability tradeoff:")
    print(
        f"   {winner['condition_label']} - LG={winner['learning_gain_mean']:.4f}, "
        f"std={winner['learning_gain_std']:.4f}, CV={winner['coefficient_of_variation']:.4f}, "
        f"seed_range={winner['seed_range']:.4f}."
    )
    print("=" * 72)


def analyze(results_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    csv_path = OUT_DIR / "beta_sensitivity_results.csv"
    if results_df is None:
        if not csv_path.exists():
            raise FileNotFoundError(f"No results at {csv_path}; run --phase run first.")
        results_df = pd.read_csv(csv_path)

    summary_df = aggregate_summary(results_df)
    ranking_df = build_ranking(summary_df)

    summary_df.to_csv(OUT_DIR / "beta_sensitivity_summary.csv", index=False)
    ranking_df.to_csv(OUT_DIR / "beta_sensitivity_ranking.csv", index=False)

    _plot_comparison(summary_df)
    _print_winner(summary_df, ranking_df)

    report = {
        "study": "beta_sensitivity_ablation",
        "default_irt_beta": DEFAULT_BETA,
        "gain_ratio": G4_GAIN["ratio_label"],
        "baseline_hp": BASELINE_HP,
        "beta_conditions": BETA_CONDITIONS,
        "seeds": SEEDS,
        "summary": summary_df.to_dict("records"),
        "ranking": ranking_df.to_dict("records"),
    }
    with open(OUT_DIR / "beta_sensitivity_ablation_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    return report


def main() -> None:
    global TRAIN_TS

    parser = argparse.ArgumentParser(description="Correctness beta ablation for DQN stability")
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
        print("=== Correctness Beta Ablation (run) ===")
        TRAIN_TS = train_ts
        run_study(seeds, args.resume)

    if args.phase in ("analyze", "all"):
        print("\n=== Correctness Beta Ablation (analyze) ===")
        analyze()


if __name__ == "__main__":
    main()
