"""
Generate publication-ready thesis figures as PNG (matplotlib).

Run from repo root:
  python3 -m RL_Module.evaluation.generate_thesis_figures

Output: RL_Module/figures/*.png (300 DPI, white background, pastel palette)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd

from RL_Module import config

# Pastel academic palette
COLORS = {
    "PPO": "#A8DADC",
    "DQN": "#FFD6A5",
    "Bandit+DQN": "#FDFFB6",
    "Bandit_DQN": "#FDFFB6",
    "Random": "#D3D3D3",
    "Rule": "#FFADAD",
}
HEAT = ["#F87171", "#FCA5A5", "#FDE68A", "#B7E4C7", "#74C69D"]

DISPLAY = {
    "PPO": "PPO",
    "DQN": "DQN",
    "Bandit_DQN": "Bandit+DQN",
    "Bandit+DQN": "Bandit+DQN",
    "Random": "Random",
    "Rule": "Rule",
}

ORDER = ["PPO", "DQN", "Bandit_DQN", "Random", "Rule"]
OUT_DIR = config.MODULE_ROOT / "figures"


def _setup():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
    })


def _algo_color(name: str) -> str:
    return COLORS.get(name, COLORS.get(DISPLAY.get(name, name), "#D3D3D3"))


def _load_summary() -> pd.DataFrame:
    path = config.LOGS_DIR / "summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run main_experiment first: {path} missing")
    df = pd.read_csv(path)
    return df[df["seed"] == "ALL"].copy()


def _load_sensitivity() -> pd.DataFrame:
    path = config.LOGS_DIR / "sensitivity_factorial.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run sensitivity_analysis first: {path} missing")
    return pd.read_csv(path)


def fig1_mean_reward(summary: pd.DataFrame, out: Path) -> None:
    rows = []
    for algo in ORDER:
        r = summary[summary["algorithm"] == algo]
        if r.empty:
            continue
        rows.append({
            "algo": DISPLAY.get(algo, algo),
            "mean": float(r["final_reward_mean"].iloc[0]),
            "std": float(r["final_reward_std"].iloc[0]),
        })
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(df))
    bars = ax.bar(
        x, df["mean"], yerr=df["std"], capsize=5,
        color=[_algo_color(a) for a in ORDER if a in summary["algorithm"].values],
        edgecolor="#94A3B8", linewidth=0.8, error_kw={"elinewidth": 1.5, "ecolor": "#475569"},
    )
    ax.set_xticks(x)
    ax.set_xticklabels(df["algo"], rotation=20, ha="right")
    ax.set_ylabel("Mean Reward")
    ax.set_title("Mean Reward by Algorithm (10 seeds)")
    ax.axhline(0, color="#CBD5E1", linewidth=1)
    ax.grid(axis="y", alpha=0.3)
    for bar, m in zip(bars, df["mean"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08,
            f"{m:.2f}", ha="center", va="bottom", fontsize=10, fontweight="600",
        )
    fig.text(0.5, 0.02, "PPO leads all agents; Rule baseline is negative.", ha="center", fontsize=10, color="#718096")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig2_success_rate(summary: pd.DataFrame, out: Path) -> None:
    rows = []
    for algo in ORDER:
        r = summary[summary["algorithm"] == algo]
        if r.empty:
            continue
        rows.append({
            "algo": DISPLAY.get(algo, algo),
            "rate": float(r["success_rate"].iloc[0]) * 100,
        })
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(df))
    bars = ax.bar(
        x, df["rate"],
        color=[_algo_color(a) for a in ORDER if a in summary["algorithm"].values],
        edgecolor="#94A3B8", linewidth=0.8,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(df["algo"], rotation=20, ha="right")
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(0, 80)
    ax.set_title("Success Rate by Algorithm (10 seeds)")
    ax.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, df["rate"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{v:.1f}%", ha="center", fontsize=10, fontweight="600")
    fig.text(0.5, 0.02, "Random scores high via easy-task sampling, not adaptation.", ha="center", fontsize=10, color="#718096")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig3_heatmap(sens: pd.DataFrame, out: Path) -> None:
    wt_labels = {"equal": "Eq", "knowledge_priority": "Kn", "affective_priority": "Af"}
    th_labels = {"default": "Def", "stricter": "Str", "looser": "Loo"}
    cols = []
    for wt in ["equal", "knowledge_priority", "affective_priority"]:
        for th in ["default", "stricter", "looser"]:
            cols.append(f"{wt_labels[wt]}/{th_labels[th]}")

    algos_plot = ["PPO", "DQN", "Bandit+DQN", "Random", "Rule"]
    mat = np.zeros((len(algos_plot), len(cols)))
    for i, algo in enumerate(algos_plot):
        for j, (wt, th) in enumerate([
            ("equal", "default"), ("equal", "stricter"), ("equal", "looser"),
            ("knowledge_priority", "default"), ("knowledge_priority", "stricter"), ("knowledge_priority", "looser"),
            ("affective_priority", "default"), ("affective_priority", "stricter"), ("affective_priority", "looser"),
        ]):
            row = sens[(sens["algorithm"] == algo) & (sens["weight_config"] == wt) & (sens["threshold_config"] == th)]
            mat[i, j] = float(row["mean_reward"].iloc[0]) if len(row) else np.nan

    fig, ax = plt.subplots(figsize=(12, 4.5))
    im = ax.imshow(mat, aspect="auto", cmap=plt.matplotlib.colors.LinearSegmentedColormap.from_list("pastel", HEAT[::-1]))
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(algos_plot)))
    ax.set_yticklabels(algos_plot, fontsize=11, fontweight="600")
    ax.set_title("Reward Sensitivity Heatmap (seed 42, 9 configurations)")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=9, fontweight="600", color="#1A202C")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Mean Reward")
    fig.text(0.5, 0.01, "Rankings stable across weight and threshold settings (Spearman rho = 1.0).", ha="center", fontsize=10, color="#718096")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig4_spearman(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.text(5, 6.2, "rho = 1.000", ha="center", fontsize=48, fontweight="bold", color="#2D6A6A")
    ax.text(5, 4.8, "Perfect Rank Stability", ha="center", fontsize=16, fontweight="bold", color="#1A202C")
    ax.text(5, 4.0, "36 pairwise comparisons | 9 reward configurations", ha="center", fontsize=11, color="#718096")
    configs = [
        ("Equal", "Default"), ("Equal", "Stricter"), ("Equal", "Looser"),
        ("Knowledge", "Default"), ("Knowledge", "Stricter"), ("Knowledge", "Looser"),
        ("Affective", "Default"), ("Affective", "Stricter"), ("Affective", "Looser"),
    ]
    fills = ["#A8DADC", "#A8DADC", "#A8DADC", "#CDB4DB", "#CDB4DB", "#CDB4DB", "#B8DFDF", "#B8DFDF", "#B8DFDF"]
    for idx, ((w, t), fill) in enumerate(zip(configs, fills)):
        ci, ri = idx % 3, idx // 3
        cx, cy = 2.2 + ci * 2.8, 2.6 - ri * 0.85
        box = FancyBboxPatch((cx - 1.0, cy - 0.35), 2.0, 0.7, boxstyle="round,pad=0.05", facecolor=fill, edgecolor="#E2E8F0", linewidth=1)
        ax.add_patch(box)
        ax.text(cx, cy + 0.08, w, ha="center", fontsize=8, fontweight="bold")
        ax.text(cx, cy - 0.12, t, ha="center", fontsize=8, color="#718096")
    fig.text(0.5, 0.02, "Algorithm ordering unchanged under all tested reward designs.", ha="center", fontsize=10, color="#718096")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig5_architecture(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")
    boxes = [
        (0.3, "FER Camera", "Emotion input", "#EBF8FF"),
        (2.5, "Student Env", "State s_t", "#F0FFF4"),
        (4.7, "RL Agent", "Policy pi", "#A8DADC"),
        (6.9, "Action Select", "Content + hint", "#CDB4DB"),
        (9.1, "Learner", "Adaptive output", "#FFFDE7"),
    ]
    bw, bh = 1.7, 1.0
    centers = []
    for x, title, sub, fill in boxes:
        rect = FancyBboxPatch((x, 1.5), bw, bh, boxstyle="round,pad=0.08", facecolor=fill, edgecolor="#E2E8F0", linewidth=1.5)
        ax.add_patch(rect)
        cx = x + bw / 2
        centers.append(cx)
        ax.text(cx, 2.15, title, ha="center", fontsize=11, fontweight="bold")
        ax.text(cx, 1.85, sub, ha="center", fontsize=9, color="#718096")
    for i in range(len(centers) - 1):
        ax.annotate("", xy=(boxes[i + 1][0] - 0.05, 2.0), xytext=(boxes[i][0] + bw + 0.05, 2.0),
                    arrowprops=dict(arrowstyle="->", color="#94A3B8", lw=1.5))
    ax.annotate("", xy=(centers[1], 1.45), xytext=(centers[4], 1.45),
                arrowprops=dict(arrowstyle="->", color="#F87171", lw=1.2, linestyle="dashed", connectionstyle="arc3,rad=-0.4"))
    ax.text((centers[1] + centers[4]) / 2, 0.55, "Reward r_t / state update", ha="center", fontsize=9, color="#C53030")
    ax.set_title("System Architecture: Emotion-Aware RL Tutoring", pad=12, fontweight="bold")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig6_explainability(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.text(5, 5.5, "Explainability: Adaptive Decision Example", ha="center", fontsize=14, fontweight="bold")

    def box(x, y, w, h, title, lines, face, edge):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06", facecolor=face, edgecolor=edge, linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h - 0.35, title, ha="center", fontsize=9, fontweight="bold", color=edge)
        for i, line in enumerate(lines):
            ax.text(x + 0.15, y + h - 0.75 - i * 0.4, line, fontsize=9, color="#1A202C")

    box(0.4, 3.2, 3.8, 1.6, "STUDENT STATE", [
        "Emotion: Frustrated (0.72)",
        "Knowledge: 40% mastery",
        "Engagement: Low",
    ], "#FFF5F5", "#C53030")
    ax.text(4.5, 3.9, "->", fontsize=22, color="#CBD5E1", ha="center")
    box(5.2, 3.2, 4.4, 1.6, "AGENT DECISION", [
        "Action: Easier content",
        "Hint: Motivational",
        "Difficulty: Reduced -1",
    ], "#F0FFF4", "#276749")

    reason = FancyBboxPatch((0.4, 1.5), 9.2, 1.35, boxstyle="round,pad=0.06", facecolor="#EBF8FF", edgecolor="#BEE3F8", linewidth=1.2)
    ax.add_patch(reason)
    ax.text(5, 2.45, "REASONING", ha="center", fontsize=9, fontweight="bold", color="#2B6CB0")
    ax.text(5, 1.95, '"Frustration exceeded threshold; policy selected remedial content with motivational feedback."',
            ha="center", fontsize=9, style="italic", wrap=True)

    metrics = [("+0.8", "Emotion d", "#A8DADC"), ("+0.5", "Knowledge d", "#B7E4C7"), ("+0.5", "Engagement d", "#CDB4DB"), ("+1.8", "Total r_t", "#74C69D")]
    for i, (val, lbl, col) in enumerate(metrics):
        x = 0.5 + i * 2.35
        rect = FancyBboxPatch((x, 0.25), 2.0, 0.95, boxstyle="round,pad=0.04", facecolor="white", edgecolor=col, linewidth=2)
        ax.add_patch(rect)
        ax.text(x + 1.0, 0.82, val, ha="center", fontsize=14, fontweight="bold", color="#276749" if lbl == "Total r_t" else "#1A202C")
        ax.text(x + 1.0, 0.45, lbl, ha="center", fontsize=8, color="#718096")

    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    _setup()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = _load_summary()
    sens = _load_sensitivity()

    outputs = {
        "fig1_mean_reward.png": lambda p: fig1_mean_reward(summary, p),
        "fig2_success_rate.png": lambda p: fig2_success_rate(summary, p),
        "fig3_heatmap.png": lambda p: fig3_heatmap(sens, p),
        "fig4_spearman.png": lambda p: fig4_spearman(p),
        "fig5_architecture.png": lambda p: fig5_architecture(p),
        "fig6_explainability.png": lambda p: fig6_explainability(p),
    }

    for name, fn in outputs.items():
        path = OUT_DIR / name
        fn(path)
        print(f"Wrote {path}")

    print(f"\nAll figures saved to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
