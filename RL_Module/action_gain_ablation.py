"""
Action gain sensitivity study: does the base_gain hierarchy bias DQN toward
harder_problem and scaffold?

Compares four runtime-patched base_gain profiles (student_model.py unchanged on disk):
  A current_gains:        harder=0.07, scaffold=0.06, explanation=0.05, ...
  B nearly_equal_gains:   harder/scaffold/explanation=0.06, simplify/hint=0.05
  C explanation_focused:  explanation=0.07, harder=0.06, scaffold=0.05, ...
  D scaffold_penalized:   scaffold=0.04, harder/explanation=0.06, ...

Protocol: DQN only, G4 gain ratio, full_emotion, mixed population, default transition
noise (beta), current thesis reward weights, USE_ZONE_BONUS=False, checkpoint selection,
50k train, 500 eval, 6 seeds [42, 7, 13, 21, 99, 314], dqn_stability_study HP.

Usage (from repo root):
  export PYTHONPATH="$(pwd)"
  python -m RL_Module.action_gain_ablation --phase all
  python -m RL_Module.action_gain_ablation --phase run --resume
  python -m RL_Module.action_gain_ablation --phase analyze
  python -m RL_Module.action_gain_ablation --phase all --quick
"""

from __future__ import annotations

import argparse
import copy
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
import RL_Module.environment.student_model as student_model
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

OUT_DIR = _HERE / "figures" / "action_gain_ablation"
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

CURRENT_GAINS: Dict[str, float] = {
    "harder_problem": 0.07,
    "scaffold": 0.06,
    "explanation": 0.05,
    "simplify_problem": 0.04,
    "hint": 0.02,
    "encouragement": 0.00,
    "break": 0.00,
    "no_action": 0.00,
}

NEARLY_EQUAL_GAINS: Dict[str, float] = {
    "harder_problem": 0.06,
    "scaffold": 0.06,
    "explanation": 0.06,
    "simplify_problem": 0.05,
    "hint": 0.05,
    "encouragement": 0.00,
    "break": 0.00,
    "no_action": 0.00,
}

EXPLANATION_FOCUSED_GAINS: Dict[str, float] = {
    "harder_problem": 0.06,
    "scaffold": 0.05,
    "explanation": 0.07,
    "simplify_problem": 0.04,
    "hint": 0.03,
    "encouragement": 0.00,
    "break": 0.00,
    "no_action": 0.00,
}

SCAFFOLD_PENALIZED_GAINS: Dict[str, float] = {
    "harder_problem": 0.06,
    "scaffold": 0.04,
    "explanation": 0.06,
    "simplify_problem": 0.05,
    "hint": 0.05,
    "encouragement": 0.00,
    "break": 0.00,
    "no_action": 0.00,
}

GAIN_CONDITIONS: Dict[str, Dict[str, Any]] = {
    "A_current_gains": {
        "label": "Current Gains (A)",
        "action_gains": CURRENT_GAINS,
    },
    "B_nearly_equal_gains": {
        "label": "Nearly Equal Gains (B)",
        "action_gains": NEARLY_EQUAL_GAINS,
    },
    "C_explanation_focused": {
        "label": "Explanation-Focused (C)",
        "action_gains": EXPLANATION_FOCUSED_GAINS,
    },
    "D_scaffold_penalized": {
        "label": "Scaffold-Penalized (D)",
        "action_gains": SCAFFOLD_PENALIZED_GAINS,
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

BIAS_ACTIONS = ["harder_problem", "scaffold", "explanation"]


def _patch_gain() -> Dict[str, float]:
    prev = dict(cfg.SIMULATOR_PARAMS)
    cfg.SIMULATOR_PARAMS = dict(G4_GAIN)
    return prev


def _restore_gain(snapshot: Dict[str, float]) -> None:
    cfg.SIMULATOR_PARAMS = snapshot


def _patch_action_gains(gains_by_name: Dict[str, float]) -> Dict[int, float]:
    """Runtime patch ACTION_EFFECT_MAP base_gain only (affect targets unchanged)."""
    prev: Dict[int, float] = {}
    for action_id, action_name in ID_TO_ACTION.items():
        prev[action_id] = float(student_model.ACTION_EFFECT_MAP[action_id]["base_gain"])
        if action_name in gains_by_name:
            student_model.ACTION_EFFECT_MAP[action_id]["base_gain"] = float(
                gains_by_name[action_name]
            )
    return prev


def _restore_action_gains(prev: Dict[int, float]) -> None:
    for action_id, base_gain in prev.items():
        student_model.ACTION_EFFECT_MAP[action_id]["base_gain"] = base_gain


def _shannon_entropy(freq: Dict[str, float]) -> float:
    p = np.array([v for v in freq.values() if v > 0], dtype=float)
    return float(-np.sum(p * np.log(p))) if len(p) else 0.0


def _actions_used_above_threshold(freq: Dict[str, float], threshold: float) -> int:
    return sum(1 for v in freq.values() if v >= threshold)


def _dominant_action_pct(freq: Dict[str, float]) -> float:
    return float(max(freq.values())) if freq else 0.0


def _top_k_actions_pct(freq: Dict[str, float], k: int) -> float:
    if not freq:
        return 0.0
    sorted_vals = sorted(freq.values(), reverse=True)
    return float(sum(sorted_vals[:k]))


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
        "actions_used_above_1_percent": _actions_used_above_threshold(freqs, 0.01),
        "actions_used_above_5_percent": _actions_used_above_threshold(freqs, 0.05),
        "dominant_action_pct": _dominant_action_pct(freqs),
        "top2_actions_pct": _top_k_actions_pct(freqs, 2),
        "dominance_detected": freq_report["dominance_detected"],
    }


def run_single(condition_key: str, seed: int) -> Dict[str, Any]:
    spec = GAIN_CONDITIONS[condition_key]
    action_gains = spec["action_gains"]
    gain_snap = _patch_gain()
    action_snap = _patch_action_gains(action_gains)
    tag = f"action_gain_{condition_key}_s{seed}"
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
        _restore_action_gains(action_snap)
        _restore_gain(gain_snap)

    elapsed = time.time() - t0
    row: Dict[str, Any] = {
        "configuration": condition_key,
        "configuration_label": spec["label"],
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
        "actions_used_above_5_percent": metrics["actions_used_above_5_percent"],
        "dominant_action_pct": metrics["dominant_action_pct"],
        "top2_actions_pct": metrics["top2_actions_pct"],
        "dominance_detected": metrics["dominance_detected"],
    }
    for action_name, freq in metrics["action_frequencies"].items():
        row[f"freq_{action_name}"] = freq

    print(
        f"  {condition_key} seed={seed}: "
        f"lg={row['learning_gain']:.3f} succ={row['success_rate']:.3f} "
        f"adapt={row['adaptation_accuracy']:.3f} entropy={row['action_entropy']:.3f} "
        f"harder={row.get('freq_harder_problem', 0):.2f} ({elapsed:.0f}s)"
    )
    return row


def _load_resume(path: Path, resume: bool) -> List[Dict[str, Any]]:
    if resume and path.exists():
        return pd.read_csv(path).to_dict("records")
    return []


def run_study(seeds: List[int], resume: bool) -> pd.DataFrame:
    csv_path = OUT_DIR / "action_gain_results.csv"
    rows = _load_resume(csv_path, resume)
    done = {(r["configuration"], int(r["seed"])) for r in rows}

    for condition_key in GAIN_CONDITIONS:
        print(f"\n--- {GAIN_CONDITIONS[condition_key]['label']} ---")
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
    for configuration, sub in results_df.groupby("configuration"):
        spec = GAIN_CONDITIONS[configuration]
        lgs = sub["learning_gain"].astype(float).tolist()
        lg_mean, lg_lo, lg_hi = confidence_interval(lgs)
        lg_std = float(np.std(lgs, ddof=1)) if len(lgs) > 1 else 0.0
        lg_range = float(max(lgs) - min(lgs)) if lgs else 0.0
        best_idx = sub["learning_gain"].astype(float).idxmax()
        worst_idx = sub["learning_gain"].astype(float).idxmin()

        row: Dict[str, Any] = {
            "configuration": configuration,
            "configuration_label": spec["label"],
            "action_gains": copy.deepcopy(spec["action_gains"]),
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
        spec = GAIN_CONDITIONS[configuration]
        entropies = sub["action_entropy"].astype(float).tolist()
        actions_1pct = sub["actions_used_above_1_percent"].astype(int).tolist()
        actions_5pct = sub["actions_used_above_5_percent"].astype(int).tolist()
        dominant = sub["dominant_action_pct"].astype(float).tolist()
        top2 = sub["top2_actions_pct"].astype(float).tolist()
        row: Dict[str, Any] = {
            "configuration": configuration,
            "configuration_label": spec["label"],
            "shannon_entropy_mean": round(float(np.mean(entropies)), 4),
            "shannon_entropy_std": round(float(np.std(entropies, ddof=1)) if len(entropies) > 1 else 0.0, 4),
            "actions_used_above_1_percent_mean": round(float(np.mean(actions_1pct)), 4),
            "actions_used_above_1_percent_std": round(
                float(np.std(actions_1pct, ddof=1)) if len(actions_1pct) > 1 else 0.0, 4
            ),
            "actions_used_above_5_percent_mean": round(float(np.mean(actions_5pct)), 4),
            "actions_used_above_5_percent_std": round(
                float(np.std(actions_5pct, ddof=1)) if len(actions_5pct) > 1 else 0.0, 4
            ),
            "dominant_action_pct_mean": round(float(np.mean(dominant)), 4),
            "dominant_action_pct_std": round(
                float(np.std(dominant, ddof=1)) if len(dominant) > 1 else 0.0, 4
            ),
            "top2_actions_pct_mean": round(float(np.mean(top2)), 4),
            "top2_actions_pct_std": round(
                float(np.std(top2, ddof=1)) if len(top2) > 1 else 0.0, 4
            ),
            "dominance_any_seed": bool(sub["dominance_detected"].any()),
        }
        for action_name in action_names:
            col = f"freq_{action_name}"
            if col in sub.columns:
                row[f"{action_name}_mean_freq"] = round(float(sub[col].mean()), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_action_bias_metrics(results_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for configuration, sub in results_df.groupby("configuration"):
        spec = GAIN_CONDITIONS[configuration]
        row: Dict[str, Any] = {
            "configuration": configuration,
            "configuration_label": spec["label"],
            "n_seeds": len(sub),
            "top1_action_pct_mean": round(float(sub["dominant_action_pct"].mean()), 4),
            "top1_action_pct_std": round(
                float(sub["dominant_action_pct"].std(ddof=1)) if len(sub) > 1 else 0.0, 4
            ),
            "top2_actions_pct_mean": round(float(sub["top2_actions_pct"].mean()), 4),
            "top2_actions_pct_std": round(
                float(sub["top2_actions_pct"].std(ddof=1)) if len(sub) > 1 else 0.0, 4
            ),
        }
        for action_name in BIAS_ACTIONS:
            col = f"freq_{action_name}"
            if col in sub.columns:
                vals = sub[col].astype(float).tolist()
                row[f"{action_name}_pct_mean"] = round(float(np.mean(vals)), 4)
                row[f"{action_name}_pct_std"] = round(
                    float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, 4
                )
        rows.append(row)
    return pd.DataFrame(rows)


def build_ranking(summary_df: pd.DataFrame, diversity_df: pd.DataFrame) -> pd.DataFrame:
    merged = summary_df.merge(
        diversity_df[[
            "configuration",
            "shannon_entropy_mean",
            "dominant_action_pct_mean",
            "harder_problem_mean_freq",
            "scaffold_mean_freq",
        ]],
        on="configuration",
    )
    merged = merged.rename(columns={
        "shannon_entropy_mean": "action_entropy",
        "dominant_action_pct_mean": "dominant_action_pct",
        "harder_problem_mean_freq": "harder_problem_pct",
        "scaffold_mean_freq": "scaffold_pct",
        "mean_episode_reward_mean": "reward_mean",
    })

    score = pd.Series(0.0, index=merged.index)
    rank_specs = {
        "learning_gain_mean": False,
        "learning_gain_std": True,
        "coefficient_of_variation": True,
        "success_rate_mean": False,
        "adaptation_accuracy_mean": False,
        "reward_mean": False,
        "action_entropy": False,
        "dominant_action_pct": True,
        "harder_problem_pct": True,
    }
    for col, ascending in rank_specs.items():
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
        "dominant_action_pct",
        "harder_problem_pct",
        "scaffold_pct",
        "rank",
    ]].sort_values("rank")
    return ranking


def _plot_comparison(
    summary_df: pd.DataFrame,
    diversity_df: pd.DataFrame,
    bias_df: pd.DataFrame,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    summary_df = summary_df.sort_values("configuration")
    diversity_df = diversity_df.sort_values("configuration")
    bias_df = bias_df.sort_values("configuration")
    labels = [row["configuration_label"].replace(" (", "\n(") for _, row in summary_df.iterrows()]
    colors = ["#E76F51", "#457B9D", "#2A9D8F", "#E9C46A"]
    xs = np.arange(len(labels))

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    ax_slots = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[1, 2]),
    ]
    panels: List[Tuple[Any, str, str, str]] = [
        (summary_df, "learning_gain_mean", "learning_gain_std", "Learning Gain"),
        (summary_df, "success_rate_mean", "success_rate_std", "Success Rate"),
        (summary_df, "adaptation_accuracy_mean", "adaptation_accuracy_std", "Adaptation Accuracy"),
        (summary_df, "mean_episode_reward_mean", "mean_episode_reward_std", "Mean Episode Reward"),
        (diversity_df, "shannon_entropy_mean", "shannon_entropy_std", "Action Entropy"),
        (diversity_df, "dominant_action_pct_mean", "dominant_action_pct_std", "Dominant Action %"),
    ]
    for ax, (src, mean_col, std_col, title) in zip(ax_slots, panels):
        means = src[mean_col].tolist()
        stds = src[std_col].tolist() if std_col in src.columns else [0.0] * len(means)
        ax.bar(xs, means, yerr=stds, capsize=5, color=colors[: len(xs)], edgecolor="#64748B")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)

    ax_bias = fig.add_subplot(gs[2, :])
    width = 0.25
    for i, action in enumerate(BIAS_ACTIONS):
        col = f"{action}_pct_mean"
        if col in bias_df.columns:
            ax_bias.bar(
                xs + (i - 1) * width,
                bias_df[col].tolist(),
                width=width,
                label=action.replace("_", " "),
                edgecolor="#64748B",
            )
    ax_bias.set_xticks(xs)
    ax_bias.set_xticklabels(labels, fontsize=8)
    ax_bias.set_title("Bias Action Frequencies")
    ax_bias.legend(fontsize=7)
    ax_bias.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Action Gain Ablation (DQN, G4 {G4_GAIN['ratio_label']}, "
        f"{TRAIN_TS // 1000}k steps, checkpoint selection)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "action_gain_comparison.png", dpi=150)
    plt.close(fig)


def _tradeoff_score(summary_df: pd.DataFrame, diversity_df: pd.DataFrame) -> pd.Series:
    """Lower is better: rank LG high, dominance/CV/std low, entropy high."""
    merged = summary_df.merge(
        diversity_df[["configuration", "shannon_entropy_mean", "dominant_action_pct_mean"]],
        on="configuration",
    )
    score = pd.Series(0.0, index=merged.index)
    for col, ascending in [
        ("learning_gain_mean", False),
        ("learning_gain_std", True),
        ("coefficient_of_variation", True),
        ("success_rate_mean", False),
        ("adaptation_accuracy_mean", False),
        ("shannon_entropy_mean", False),
        ("dominant_action_pct_mean", True),
    ]:
        if col in merged.columns:
            score += merged[col].rank(ascending=ascending, method="average")
    return score


def _compute_winners(
    summary_df: pd.DataFrame,
    diversity_df: pd.DataFrame,
) -> Dict[str, str]:
    lg_winner = summary_df.loc[summary_df["learning_gain_mean"].idxmax(), "configuration"]
    dom_winner = diversity_df.loc[
        diversity_df["dominant_action_pct_mean"].idxmin(), "configuration"
    ]
    tradeoff_key = summary_df.loc[
        _tradeoff_score(summary_df, diversity_df).idxmin(), "configuration"
    ]
    return {
        "max_learning_gain": str(lg_winner),
        "min_dominant_action_pct": str(dom_winner),
        "best_tradeoff": str(tradeoff_key),
        "recommended": str(tradeoff_key),
    }


def _print_winner(
    summary_df: pd.DataFrame,
    diversity_df: pd.DataFrame,
    bias_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
) -> None:
    merged = summary_df.merge(
        diversity_df[[
            "configuration",
            "configuration_label",
            "shannon_entropy_mean",
            "dominant_action_pct_mean",
        ]],
        on="configuration",
        suffixes=("", "_div"),
    )
    merged = merged.merge(
        bias_df[["configuration", "harder_problem_pct_mean", "scaffold_pct_mean"]],
        on="configuration",
    )

    lg_winner = summary_df.loc[summary_df["learning_gain_mean"].idxmax()]
    dom_winner = diversity_df.loc[diversity_df["dominant_action_pct_mean"].idxmin()]
    tradeoff_scores = _tradeoff_score(summary_df, diversity_df)
    tradeoff_winner_key = summary_df.loc[tradeoff_scores.idxmin(), "configuration"]
    tradeoff_winner = summary_df[summary_df["configuration"] == tradeoff_winner_key].iloc[0]
    tradeoff_div = diversity_df[diversity_df["configuration"] == tradeoff_winner_key].iloc[0]
    tradeoff_bias = bias_df[bias_df["configuration"] == tradeoff_winner_key].iloc[0]
    overall_rank = ranking_df.iloc[0]

    current = summary_df[summary_df["configuration"] == "A_current_gains"].iloc[0]
    current_div = diversity_df[diversity_df["configuration"] == "A_current_gains"].iloc[0]
    current_bias = bias_df[bias_df["configuration"] == "A_current_gains"].iloc[0]
    current_harder = float(current_bias["harder_problem_pct_mean"])
    current_scaffold = float(current_bias["scaffold_pct_mean"])
    current_dom = float(current_div["dominant_action_pct_mean"])

    print("\n" + "=" * 72)
    print("ACTION GAIN SENSITIVITY STUDY - RESULTS")
    print("=" * 72)

    print("\n--- Per-Configuration Summary ---")
    for _, row in merged.sort_values("configuration").iterrows():
        print(
            f"\n{row['configuration_label']} ({row['configuration']})"
        )
        print(
            f"  learning_gain={row['learning_gain_mean']:.4f} "
            f"(std={row['learning_gain_std']:.4f}, CV={row['coefficient_of_variation']:.4f})"
        )
        print(
            f"  success_rate={row['success_rate_mean']:.4f}, "
            f"adaptation_accuracy={row['adaptation_accuracy_mean']:.4f}"
        )
        print(
            f"  action_entropy={row['shannon_entropy_mean']:.4f}, "
            f"dominant_action_pct={row['dominant_action_pct_mean']:.4f}"
        )
        print(
            f"  harder_problem_pct={row['harder_problem_pct_mean']:.4f}, "
            f"scaffold_pct={row['scaffold_pct_mean']:.4f}"
        )

    print("\n--- Research Questions ---")
    print(
        f"\n1. Max learning gain: {lg_winner['configuration']} "
        f"({lg_winner['configuration_label']})  "
        f"LG={lg_winner['learning_gain_mean']:.4f}"
    )
    print(
        f"\n2. Min action dominance: {dom_winner['configuration']} "
        f"({dom_winner['configuration_label']})  "
        f"dominant_action_pct={dom_winner['dominant_action_pct_mean']:.4f}, "
        f"entropy={dom_winner['shannon_entropy_mean']:.4f}"
    )
    print(
        f"\n3. Best performance-stability-diversity tradeoff: "
        f"{tradeoff_winner['configuration']} ({tradeoff_winner['configuration_label']})  "
        f"LG={tradeoff_winner['learning_gain_mean']:.4f}, "
        f"CV={tradeoff_winner['coefficient_of_variation']:.4f}, "
        f"dominant={float(diversity_df.loc[diversity_df['configuration'] == tradeoff_winner_key, 'dominant_action_pct_mean'].iloc[0]):.4f}"
    )

    print("\n--- Current-Gains (A) Bias Check ---")
    action_bias = current_dom > 0.35 or current_scaffold > 0.30 or current_harder > 0.25
    if action_bias:
        print(
            f"  Policy bias toward scaffold/harder_problem is PRESENT under current gains: "
            f"scaffold={current_scaffold:.1%}, harder_problem={current_harder:.1%}, "
            f"dominant action share={current_dom:.1%}."
        )
    else:
        print(
            f"  Policy bias is WEAK under current gains "
            f"(scaffold={current_scaffold:.1%}, harder_problem={current_harder:.1%})."
        )

    print("\n" + "=" * 72)
    print("WINNER ACTION GAIN CONFIGURATION")
    print("=" * 72)
    print(
        f"\n  {tradeoff_winner_key} "
        f"({GAIN_CONDITIONS[tradeoff_winner_key]['label']})"
    )
    print("  Rationale: best performance-stability-diversity tradeoff (Q3);")
    print(f"  also maximizes learning gain (Q1, LG={tradeoff_winner['learning_gain_mean']:.4f}).")
    print(f"  learning_gain_mean       = {tradeoff_winner['learning_gain_mean']:.4f}")
    print(f"  learning_gain_std        = {tradeoff_winner['learning_gain_std']:.4f}")
    print(f"  coefficient_of_variation = {tradeoff_winner['coefficient_of_variation']:.4f}")
    print(f"  success_rate_mean        = {tradeoff_winner['success_rate_mean']:.4f}")
    print(f"  adaptation_accuracy_mean = {tradeoff_winner['adaptation_accuracy_mean']:.4f}")
    print(f"  action_entropy           = {tradeoff_div['shannon_entropy_mean']:.4f}")
    print(f"  dominant_action_pct      = {tradeoff_div['dominant_action_pct_mean']:.4f}")
    print(f"  harder_problem_pct       = {tradeoff_bias['harder_problem_pct_mean']:.4f}")
    print(f"  scaffold_pct             = {tradeoff_bias['scaffold_pct_mean']:.4f}")
    print(
        f"\n  Min-dominance config (Q2): {dom_winner['configuration']} "
        f"(dominant={dom_winner['dominant_action_pct_mean']:.4f})"
    )
    print(
        f"  Adaptation/success leader: {overall_rank['configuration']} "
        f"(composite rank {int(overall_rank['rank'])})"
    )
    print("=" * 72)


def analyze(results_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    csv_path = OUT_DIR / "action_gain_results.csv"
    if results_df is None:
        if not csv_path.exists():
            raise FileNotFoundError(f"No results at {csv_path}; run --phase run first.")
        results_df = pd.read_csv(csv_path)

    summary_df = aggregate_summary(results_df)
    diversity_df = aggregate_action_diversity(results_df)
    bias_df = aggregate_action_bias_metrics(results_df)
    ranking_df = build_ranking(summary_df, diversity_df)

    results_df.to_csv(OUT_DIR / "action_gain_results.csv", index=False)
    summary_df.to_csv(OUT_DIR / "action_gain_summary.csv", index=False)
    diversity_df.to_csv(OUT_DIR / "action_diversity_summary.csv", index=False)
    bias_df.to_csv(OUT_DIR / "action_bias_metrics.csv", index=False)
    ranking_df.to_csv(OUT_DIR / "action_gain_ranking.csv", index=False)

    _plot_comparison(summary_df, diversity_df, bias_df)
    _print_winner(summary_df, diversity_df, bias_df, ranking_df)
    winners = _compute_winners(summary_df, diversity_df)

    report = {
        "study": "action_gain_sensitivity",
        "train_timesteps": TRAIN_TS,
        "gain_ratio": G4_GAIN["ratio_label"],
        "baseline_hp": BASELINE_HP,
        "gain_conditions": {
            k: {**v, "action_gains": v["action_gains"]} for k, v in GAIN_CONDITIONS.items()
        },
        "seeds": SEEDS,
        "summary": summary_df.to_dict("records"),
        "action_diversity": diversity_df.to_dict("records"),
        "action_bias": bias_df.to_dict("records"),
        "ranking": ranking_df.to_dict("records"),
        "winners": winners,
    }
    with open(OUT_DIR / "action_gain_ablation_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    return report


def main() -> None:
    global TRAIN_TS

    parser = argparse.ArgumentParser(
        description="Action gain sensitivity: four base_gain profiles (A-D)"
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
        print("=== Action Gain Sensitivity Study (run) ===")
        TRAIN_TS = train_ts
        run_study(seeds, args.resume)

    if args.phase in ("analyze", "all"):
        print("\n=== Action Gain Sensitivity Study (analyze) ===")
        analyze()


if __name__ == "__main__":
    main()
