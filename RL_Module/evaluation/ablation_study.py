"""
PPO emotion ablation: publication-quality figures and tables.

Compares PPO with emotion signal vs PPO with emotion ablated (obs[5]=0, no FER routing).

Data sources (in order):
  - Emotion-aware: logs/episodes_PPO_seed{seed}.csv (existing experiment logs)
  - No-emotion: logs/ablation_study/episodes_PPO_no_emotion_seed{seed}.csv
    (generated on first run if missing)

Metrics used (success rate excluded by design):
  - final_reward_mean (mean episode total_reward)
  - learning_gain_mean
  - adaptation_accuracy_mean

Run from repo root:
  python3 -m RL_Module.evaluation.ablation_study
  python3 -m RL_Module.evaluation.ablation_study --timesteps 50000
  python3 -m RL_Module.evaluation.ablation_study --skip-train

Outputs: RL_Module/figures/ablation_study/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from RL_Module import config
from RL_Module.agents.ppo_agent import PPOAgent, make_masked_env
from RL_Module.environment.student_env import StudentEnv
from RL_Module.evaluation.metrics import evaluate

OUT_DIR = config.MODULE_ROOT / "figures" / "ablation_study"
ABLATION_LOG_DIR = config.LOGS_DIR / "ablation_study"
CACHE_CSV = OUT_DIR / "ablation_metrics_detail.csv"
SUMMARY_CSV = OUT_DIR / "ablation_summary.csv"

# Pastel academic palette
COLOR_EMOTION = "#A8DADC"
COLOR_NO_EMOTION = "#FFD6A5"
LABEL_EMOTION = "PPO (emotion-aware)"
LABEL_NO_EMOTION = "PPO (no emotion)"

VARIANT_EMOTION = "emotion_aware"
VARIANT_NO_EMOTION = "no_emotion"


def _setup_plt() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "sans-serif",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "-",
    })


def _episode_path(variant: str, seed: int) -> Path:
    if variant == VARIANT_EMOTION:
        return config.LOGS_DIR / f"episodes_PPO_seed{seed}.csv"
    return ABLATION_LOG_DIR / f"episodes_PPO_no_emotion_seed{seed}.csv"


def _monitor_path(variant: str, seed: int) -> Path:
    tag = "ppo" if variant == VARIANT_EMOTION else "ppo_no_emotion"
    return (
        config.LOGS_DIR / f"monitor_{tag}_seed{seed}.monitor.csv"
        if variant == VARIANT_EMOTION
        else ABLATION_LOG_DIR / f"monitor_{tag}_seed{seed}.monitor.csv"
    )


def _aggregate_episode_file(path: Path) -> Dict[str, float]:
    df = pd.read_csv(path)
    return {
        "final_reward_mean": float(df["total_reward"].mean()),
        "final_reward_std": float(df["total_reward"].std(ddof=1)) if len(df) > 1 else 0.0,
        "learning_gain_mean": float(df["learning_gain"].mean()),
        "learning_gain_std": float(df["learning_gain"].std(ddof=1)) if len(df) > 1 else 0.0,
        "adaptation_accuracy_mean": float(df["adaptation_accuracy"].mean()),
        "adaptation_accuracy_std": float(
            df["adaptation_accuracy"].std(ddof=1)
        ) if len(df) > 1 else 0.0,
        "n_episodes": int(len(df)),
    }


def collect_metrics_from_logs(seeds: List[int]) -> pd.DataFrame:
    """Build per-seed metrics for both variants from episode CSV logs."""
    rows: List[Dict] = []
    for seed in seeds:
        for variant, label in [
            (VARIANT_EMOTION, LABEL_EMOTION),
            (VARIANT_NO_EMOTION, LABEL_NO_EMOTION),
        ]:
            path = _episode_path(variant, seed)
            if not path.exists():
                continue
            agg = _aggregate_episode_file(path)
            rows.append({
                "seed": seed,
                "variant": variant,
                "model": label,
                **agg,
            })
    return pd.DataFrame(rows)


def train_and_eval_no_emotion(
    seed: int,
    train_timesteps: int,
    eval_episodes: int,
) -> Dict[str, float]:
    """Train PPO with emotion ablated; evaluate and write episode + monitor logs."""
    ABLATION_LOG_DIR.mkdir(parents=True, exist_ok=True)
    config.set_all_seeds(seed)

    env = make_masked_env(
        seed,
        use_emotion=False,
        ablation_no_emotion=True,
        algo_tag=f"ppo_no_emotion_seed{seed}",
    )
    agent = PPOAgent()
    agent.train(env, train_timesteps, seed)
    agent.save(str(ABLATION_LOG_DIR / f"PPO_no_emotion_{seed}"))
    env.close()

    eval_env = StudentEnv(
        population_seed=seed,
        use_emotion=False,
        ablation_no_emotion=True,
    )
    result = evaluate(
        agent,
        eval_env,
        n_episodes=eval_episodes,
        seed=seed,
        algorithm="PPO_no_emotion",
        log_steps=True,
    )
    eval_env.close()

    # Move episode + monitor logs into ablation_study/
    default_ep = config.LOGS_DIR / f"episodes_PPO_no_emotion_seed{seed}.csv"
    target_ep = _episode_path(VARIANT_NO_EMOTION, seed)
    if default_ep.exists():
        target_ep.parent.mkdir(parents=True, exist_ok=True)
        default_ep.replace(target_ep)

    default_mon = config.LOGS_DIR / f"monitor_ppo_no_emotion_seed{seed}.monitor.csv"
    target_mon = _monitor_path(VARIANT_NO_EMOTION, seed)
    if default_mon.exists():
        target_mon.parent.mkdir(parents=True, exist_ok=True)
        default_mon.replace(target_mon)

    return {
        "final_reward_mean": result["mean_episode_reward"],
        "learning_gain_mean": result["learning_gain"],
        "adaptation_accuracy_mean": result["adaptation_accuracy"],
    }


def ensure_no_emotion_logs(
    seeds: List[int],
    train_timesteps: int,
    eval_episodes: int,
    skip_train: bool,
) -> None:
    missing = [s for s in seeds if not _episode_path(VARIANT_NO_EMOTION, s).exists()]
    if not missing:
        return
    if skip_train:
        raise FileNotFoundError(
            f"Missing no-emotion logs for seeds {missing}. "
            f"Run without --skip-train to generate them."
        )
    print(f"Training PPO (no emotion) for seeds: {missing} ({train_timesteps} steps each)")
    for seed in missing:
        print(f"  seed={seed} ...")
        train_and_eval_no_emotion(seed, train_timesteps, eval_episodes)


def load_learning_curve_matrix(
    variant: str,
    seeds: List[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns episode indices, mean reward per episode, std across seeds.
    Aligns to minimum episode count across seeds.
    """
    series: List[np.ndarray] = []
    for seed in seeds:
        path = _episode_path(variant, seed)
        if not path.exists():
            continue
        df = pd.read_csv(path)
        series.append(df["total_reward"].to_numpy(dtype=float))

    if not series:
        return np.array([]), np.array([]), np.array([])

    min_len = min(len(s) for s in series)
    mat = np.stack([s[:min_len] for s in series], axis=0)
    episodes = np.arange(1, min_len + 1)
    return episodes, mat.mean(axis=0), mat.std(axis=0, ddof=1) if mat.shape[0] > 1 else np.zeros(min_len)


def load_monitor_curve(
    variant: str,
    seeds: List[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Training-time curves from SB3 Monitor CSV (r = episode return)."""
    series: List[np.ndarray] = []
    for seed in seeds:
        path = _monitor_path(variant, seed)
        if not path.exists():
            continue
        df = pd.read_csv(path, comment="#")
        if "r" not in df.columns:
            continue
        series.append(df["r"].to_numpy(dtype=float))

    if not series:
        return np.array([]), np.array([]), np.array([])

    min_len = min(len(s) for s in series)
    mat = np.stack([s[:min_len] for s in series], axis=0)
    steps = np.arange(1, min_len + 1)
    return steps, mat.mean(axis=0), mat.std(axis=0, ddof=1) if mat.shape[0] > 1 else np.zeros(min_len)


def smooth(y: np.ndarray, window: int) -> np.ndarray:
    if len(y) < window:
        window = max(1, len(y) // 3)
    return pd.Series(y).rolling(window, min_periods=1, center=True).mean().to_numpy()


def plot_learning_curves(
    seeds: List[int],
    window: int,
    out: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6.75))

    for variant, color, label in [
        (VARIANT_EMOTION, COLOR_EMOTION, LABEL_EMOTION),
        (VARIANT_NO_EMOTION, COLOR_NO_EMOTION, LABEL_NO_EMOTION),
    ]:
        x, mean, std = load_learning_curve_matrix(variant, seeds)
        if len(x) == 0:
            x, mean, std = load_monitor_curve(variant, seeds)
            x_label = "Training episode"
        else:
            x_label = "Evaluation episode"

        if len(x) == 0:
            print(f"  Warning: no curve data for {variant}")
            continue

        mean_s = smooth(mean, window)
        std_s = smooth(std, window) if len(std) else np.zeros_like(mean_s)

        ax.plot(x, mean_s, color=color, linewidth=2.5, label=label)
        ax.fill_between(x, mean_s - std_s, mean_s + std_s, color=color, alpha=0.25)

    ax.set_xlabel(x_label)
    ax.set_ylabel("Episode reward")
    ax.set_title("PPO Ablation: Learning Curves (emotion vs no emotion)")
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _bar_chart(
    val_col: str,
    err_col: str,
    ylabel: str,
    title: str,
    summary: pd.DataFrame,
    colors: List[str],
    out: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    vals = summary[val_col].tolist()
    errs = summary[err_col].tolist() if err_col in summary.columns else [0.0] * len(vals)

    x = np.arange(len(summary))
    bars = ax.bar(
        x, vals, yerr=errs, capsize=6,
        color=colors[: len(vals)],
        edgecolor="#94A3B8", linewidth=0.8,
        error_kw={"elinewidth": 1.5, "ecolor": "#475569"},
    )
    ax.set_xticks(x)
    ax.set_xticklabels(["Emotion-aware", "No emotion"], fontsize=11)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (0.02 * abs(max(vals)) if max(vals) else 0.05),
            f"{v:.3f}",
            ha="center", va="bottom", fontsize=11, fontweight="600",
        )
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_summary_table(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, label in [
        (VARIANT_EMOTION, LABEL_EMOTION),
        (VARIANT_NO_EMOTION, LABEL_NO_EMOTION),
    ]:
        sub = detail[detail["variant"] == variant]
        if sub.empty:
            continue
        rows.append({
            "model": label,
            "reward": sub["final_reward_mean"].mean(),
            "reward_std": sub["final_reward_mean"].std(ddof=1) if len(sub) > 1 else 0.0,
            "learning_gain": sub["learning_gain_mean"].mean(),
            "learning_gain_std": sub["learning_gain_mean"].std(ddof=1) if len(sub) > 1 else 0.0,
            "adaptation_accuracy": sub["adaptation_accuracy_mean"].mean(),
            "adaptation_accuracy_std": sub["adaptation_accuracy_mean"].std(ddof=1) if len(sub) > 1 else 0.0,
            "n_seeds": len(sub),
        })
    return pd.DataFrame(rows)


def compute_effect_stats(summary: pd.DataFrame) -> Dict[str, float]:
    if len(summary) < 2:
        return {}
    emo = summary.iloc[0]
    noe = summary.iloc[1]
    stats: Dict[str, float] = {}
    for key, label in [
        ("reward", "final_reward"),
        ("learning_gain", "learning_gain"),
        ("adaptation_accuracy", "adaptation_accuracy"),
    ]:
        base = float(noe[key])
        treat = float(emo[key])
        delta = treat - base
        pct = (delta / abs(base) * 100.0) if abs(base) > 1e-9 else float("nan")
        stats[f"{label}_delta"] = delta
        stats[f"{label}_pct_improvement"] = pct
    return stats


def plot_summary_table_png(summary: pd.DataFrame, stats: Dict[str, float], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")

    headers = ["Model", "Reward", "Learning Gain", "Adaptation Accuracy"]
    table_rows = []
    for _, r in summary.iterrows():
        table_rows.append([
            r["model"].replace("PPO ", ""),
            f"{r['reward']:.3f} +/- {r['reward_std']:.3f}",
            f"{r['learning_gain']:.3f} +/- {r['learning_gain_std']:.3f}",
            f"{r['adaptation_accuracy']:.3f} +/- {r['adaptation_accuracy_std']:.3f}",
        ])

    tbl = ax.table(
        cellText=table_rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.2, 1.8)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#E2E8F0")
            cell.set_text_props(fontweight="bold")
        elif row == 1:
            cell.set_facecolor("#F0FBFC")
        elif row == 2:
            cell.set_facecolor("#FFF8F0")

    if stats:
        pct_r = stats.get("final_reward_pct_improvement", float("nan"))
        delta_r = stats.get("final_reward_delta", float("nan"))
        caption = (
            f"Emotion-aware PPO vs no-emotion: reward delta = {delta_r:+.3f} "
            f"({pct_r:+.1f}% vs baseline). "
            f"Positive delta supports emotion-aware adaptation."
        )
        ax.text(0.5, 0.08, caption, ha="center", va="center", fontsize=10, color="#4A5568",
                transform=ax.transAxes, wrap=True)

    ax.set_title("PPO Emotion Ablation Summary", fontsize=14, fontweight="bold", pad=20)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def generate_all(
    seeds: Optional[List[int]] = None,
    train_timesteps: int = 5_000,
    eval_episodes: int = 1_000,
    smooth_window: int = 50,
    skip_train: bool = False,
) -> None:
    _setup_plt()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ABLATION_LOG_DIR.mkdir(parents=True, exist_ok=True)
    seeds = seeds or list(config.SEEDS)

    ensure_no_emotion_logs(seeds, train_timesteps, eval_episodes, skip_train)

    detail = collect_metrics_from_logs(seeds)
    if detail.empty:
        raise RuntimeError("No episode logs found for PPO ablation.")

    detail.to_csv(CACHE_CSV, index=False)
    summary = build_summary_table(detail)
    summary.to_csv(SUMMARY_CSV, index=False)

    stats = compute_effect_stats(summary)
    with open(OUT_DIR / "ablation_effect_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print("\n--- PPO Emotion Ablation Summary ---")
    print(summary.to_string(index=False))
    if stats:
        print(f"\nReward improvement: {stats.get('final_reward_delta', 0):+.3f} "
              f"({stats.get('final_reward_pct_improvement', 0):+.1f}%)")

    plot_learning_curves(
        seeds, smooth_window,
        OUT_DIR / "ppo_ablation_learning_curves.png",
    )
    colors = [COLOR_EMOTION, COLOR_NO_EMOTION]
    _bar_chart(
        "reward", "reward_std", "Mean episode reward",
        "Ablation: Final Reward (PPO with vs without emotion)",
        summary, colors, OUT_DIR / "ppo_ablation_rewards.png",
    )
    _bar_chart(
        "learning_gain", "learning_gain_std", "Mean learning gain",
        "Ablation: Learning Gain",
        summary, colors, OUT_DIR / "ppo_ablation_learning_gain.png",
    )
    _bar_chart(
        "adaptation_accuracy", "adaptation_accuracy_std",
        "Mean adaptation accuracy",
        "Ablation: Adaptation Accuracy",
        summary, colors, OUT_DIR / "ppo_ablation_accuracy.png",
    )
    plot_summary_table_png(summary, stats, OUT_DIR / "ppo_ablation_table.png")

    print(f"\nAll outputs written to {OUT_DIR.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PPO emotion ablation figures")
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument(
        "--timesteps", type=int, default=5_000,
        help="Training steps for no-emotion PPO when logs are missing (use 50000 for full run)",
    )
    parser.add_argument("--eval-episodes", type=int, default=1_000)
    parser.add_argument("--smooth-window", type=int, default=50)
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Only plot from existing CSV logs (no-emotion logs must exist)",
    )
    args = parser.parse_args()
    generate_all(
        seeds=args.seeds,
        train_timesteps=args.timesteps,
        eval_episodes=args.eval_episodes,
        smooth_window=args.smooth_window,
        skip_train=args.skip_train,
    )


if __name__ == "__main__":
    main()
