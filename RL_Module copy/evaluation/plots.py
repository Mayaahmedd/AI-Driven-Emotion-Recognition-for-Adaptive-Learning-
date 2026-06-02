"""
Visualization: 5 required plots (spec Phase 5).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from RL_Module import config
from RL_Module.mdp_definition import ID_TO_ACTION


def _ensure_dir(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _read_csv_if_exists(path: Path) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        print(f"WARNING: {path} not found, skipping plot")
        return None
    return pd.read_csv(path)


def _load_episode_csv(algorithm: str, seed: int) -> Optional[pd.DataFrame]:
    path = config.LOGS_DIR / f"episodes_{algorithm}_seed{seed}.csv"
    return _read_csv_if_exists(path)


def _algorithms_from_summary() -> List[str]:
    sum_path = config.LOGS_DIR / "summary.csv"
    if not os.path.exists(sum_path):
        print("WARNING: summary.csv missing, skipping heatmap algorithm list")
        return list(config.ALGORITHMS)
    df = pd.read_csv(sum_path)
    algo_list = df["algorithm"].unique().tolist()
    return [a for a in algo_list if a != "ALL"]


def plot_reward_curves(
    algorithm_rewards: Dict[str, List[List[float]]],
    out_path: Optional[Path] = None,
) -> Optional[Path]:
    """Per-algorithm list of reward series per seed; shaded +/- std."""
    out = _ensure_dir(out_path or config.LOGS_DIR / "reward_curves.png")
    if not algorithm_rewards:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    for algo, seed_series in algorithm_rewards.items():
        if not seed_series:
            continue
        max_len = max(len(s) for s in seed_series)
        mat = np.full((len(seed_series), max_len), np.nan)
        for i, s in enumerate(seed_series):
            mat[i, : len(s)] = s
        mean = np.nanmean(mat, axis=0)
        std = np.nanstd(mat, axis=0)
        x = np.arange(len(mean))
        window = min(50, len(mean))
        mean_s = pd.Series(mean).rolling(window, min_periods=1).mean().values
        std_s = pd.Series(std).rolling(window, min_periods=1).mean().values
        ax.plot(x, mean_s, label=algo)
        ax.fill_between(x, mean_s - std_s, mean_s + std_s, alpha=0.2)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Rolling Mean Reward")
    ax.set_title("Reward Curves by Algorithm (+/- std across seeds)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_convergence_comparison(
    summary_rows: List[Dict[str, Any]],
    out_path: Optional[Path] = None,
) -> Optional[Path]:
    out = _ensure_dir(out_path or config.LOGS_DIR / "convergence_comparison.png")
    if not summary_rows:
        return None
    algos = [r["algorithm"] for r in summary_rows]
    conv = [r.get("convergence_episode") or 0 for r in summary_rows]
    stds = [r.get("final_reward_std", 0) for r in summary_rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(algos, conv, yerr=stds, capsize=4, color="#7858dc")
    ax.set_ylabel("Convergence Episode")
    ax.set_title("Convergence Comparison")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_action_heatmap(
    algorithm: str,
    seeds: Optional[List[int]] = None,
    out_path: Optional[Path] = None,
) -> Optional[Path]:
    out = _ensure_dir(out_path or config.LOGS_DIR / "action_heatmap.png")
    seeds = seeds or config.SEEDS
    algo_list = _algorithms_from_summary()
    if not algo_list:
        return None

    usage = np.zeros((len(algo_list), 10))

    for ai, algo in enumerate(algo_list):
        counts = np.zeros(10)
        for seed in seeds:
            df = _load_episode_csv(algo, seed)
            if df is None:
                continue
            for i in range(8):
                col = f"action_{i}_count"
                if col in df.columns:
                    counts[i] += df[col].sum()
        if counts.sum() > 0:
            usage[ai] = counts / counts.sum()

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(usage, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(8))
    ax.set_xticklabels([ID_TO_ACTION[i][:10] for i in range(8)], rotation=45, ha="right")
    ax.set_yticks(range(len(algo_list)))
    ax.set_yticklabels(algo_list)
    ax.set_title("Action Usage Heatmap (algorithm x action)")
    plt.colorbar(im, ax=ax, label="Usage rate")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_ablation_delta(
    ablation_delta: Dict[str, float],
    out_path: Optional[Path] = None,
) -> Optional[Path]:
    out = _ensure_dir(out_path or config.LOGS_DIR / "ablation_delta.png")
    if not ablation_delta:
        return None
    keys = list(ablation_delta.keys())
    vals = [ablation_delta[k] for k in keys]
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#58a078" if v >= 0 else "#c05858" for v in vals]
    ax.bar(keys, vals, color=colors)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_ylabel("Delta (Full - No Emotion)")
    ax.set_title("Ablation: Gain from Emotion Signals")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_learning_gain_by_type(
    type_results: Dict[str, Dict[str, float]],
    out_path: Optional[Path] = None,
) -> Optional[Path]:
    """type_results[student_type][algorithm] = learning_gain."""
    out = _ensure_dir(out_path or config.LOGS_DIR / "learning_gain_by_type.png")
    if not type_results:
        return None
    types = list(type_results.keys())
    algos = _algorithms_from_summary()[:4] or list(config.ALGORITHMS[:4])
    data = np.array([[type_results[t].get(a, 0) for a in algos] for t in types])
    x = np.arange(len(algos))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, t in enumerate(types):
        ax.bar(x + i * width, data[i], width, label=t)
    ax.set_xticks(x + width * (len(types) - 1) / 2)
    ax.set_xticklabels(algos)
    ax.set_ylabel("Learning Gain (normalized)")
    ax.set_title("Learning Gain by Student Type")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_success_bars(
    summary_rows: List[Dict[str, Any]],
    out_path: Optional[Path] = None,
) -> Optional[Path]:
    out = _ensure_dir(out_path or config.LOGS_DIR / "success_bars.png")
    if not summary_rows:
        return None
    algos = [r["algorithm"] for r in summary_rows]
    rates = [float(r.get("success_rate", 0)) * 100 for r in summary_rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(algos, rates, color="#58a078")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Success Rate by Algorithm")
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def draw_all(
    summary_rows: List[Dict[str, Any]],
    algorithm_rewards: Optional[Dict[str, List[List[float]]]] = None,
    ablation_delta: Optional[Dict[str, float]] = None,
    type_results: Optional[Dict[str, Dict[str, float]]] = None,
) -> None:
    if algorithm_rewards:
        plot_reward_curves(algorithm_rewards)
    if summary_rows:
        plot_success_bars(summary_rows)
        plot_convergence_comparison(summary_rows)
    plot_action_heatmap("PPO")
    if ablation_delta:
        plot_ablation_delta(ablation_delta)
    if type_results:
        plot_learning_gain_by_type(type_results)
