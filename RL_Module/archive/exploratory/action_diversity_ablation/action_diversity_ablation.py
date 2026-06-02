"""
Action diversity reward ablation: does a small diversity incentive reduce action
dominance and improve DQN policy quality without hurting learning gain?

Configurations (runtime training reward wrapper only; reward_function.py unchanged):
  A baseline:              no diversity shaping
  B diversity_bonus:       +0.02 when action differs from previous step
  C strong_diversity_bonus:+0.05 when action differs from previous step
  D repetition_penalty:    -0.02 when same action repeated consecutively
  E strong_repetition_penalty: -0.05 when same action repeated consecutively

Protocol: DQN only, G4 gain ratio, IRT_BETA=3.0, full_emotion, mixed population,
default transition noise, checkpoint selection, 50k train, 500 eval,
6 seeds [42, 7, 13, 21, 99, 314], dqn_stability_study baseline hyperparameters.

Usage (from repo root):
  export PYTHONPATH="$(pwd)"
  python -m RL_Module.action_diversity_ablation --phase all
  python -m RL_Module.action_diversity_ablation --phase run --resume
  python -m RL_Module.action_diversity_ablation --phase analyze
  python -m RL_Module.action_diversity_ablation --phase all --quick
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import RL_Module.config as cfg
from RL_Module.agents.dqn_agent import DQNAgent
from RL_Module.environment.student_env import DQNSafeEnv, StudentEnv
from RL_Module.evaluation.metrics import (
    _adaptation_match,
    _is_success,
    action_frequency_report,
    confidence_interval,
    normalized_knowledge_gain,
)
from RL_Module.mdp_definition import ACTION_DIM, ID_TO_ACTION

OUT_DIR = _HERE / "figures" / "action_diversity_ablation"
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

IRT_BETA = 3.0

BASELINE_HP: Dict[str, Any] = {
    "learning_rate": 5e-4,
    "batch_size": 64,
    "buffer_size": 100_000,
    "target_update_interval": 2000,
    "exploration_fraction": 0.3,
    "exploration_final_eps": 0.05,
    "net_arch": [64, 64],
}

ACTION_NAMES = list(ID_TO_ACTION.values())

DIVERSITY_CONDITIONS: Dict[str, Dict[str, Any]] = {
    "A_baseline": {
        "label": "Baseline Reward (A)",
        "mode": None,
        "amount": 0.0,
    },
    "B_diversity_bonus": {
        "label": "Diversity Bonus +0.02 (B)",
        "mode": "bonus",
        "amount": 0.02,
    },
    "C_strong_diversity_bonus": {
        "label": "Strong Diversity Bonus +0.05 (C)",
        "mode": "bonus",
        "amount": 0.05,
    },
    "D_repetition_penalty": {
        "label": "Repetition Penalty -0.02 (D)",
        "mode": "penalty",
        "amount": 0.02,
    },
    "E_strong_repetition_penalty": {
        "label": "Strong Repetition Penalty -0.05 (E)",
        "mode": "penalty",
        "amount": 0.05,
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


class DiversityRewardWrapper(gym.Wrapper):
    """Training-only wrapper: adds diversity bonus or repetition penalty to step reward."""

    def __init__(self, env: gym.Env, mode: str, amount: float):
        super().__init__(env)
        self._mode = mode
        self._amount = float(amount)
        self._prev_action: Optional[int] = None

    def reset(self, **kwargs):
        self._prev_action = None
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self._prev_action is not None:
            if self._mode == "bonus" and action != self._prev_action:
                reward = float(reward) + self._amount
            elif self._mode == "penalty" and action == self._prev_action:
                reward = float(reward) - self._amount
        self._prev_action = int(action)
        return obs, reward, terminated, truncated, info


def _patch_simulator() -> Dict[str, float]:
    prev = dict(cfg.SIMULATOR_PARAMS)
    cfg.SIMULATOR_PARAMS = {
        "GAIN_CORRECT_FACTOR": G4_GAIN["GAIN_CORRECT_FACTOR"],
        "GAIN_INCORRECT_FACTOR": G4_GAIN["GAIN_INCORRECT_FACTOR"],
        "IRT_BETA": IRT_BETA,
    }
    return prev


def _restore_simulator(snapshot: Dict[str, float]) -> None:
    cfg.SIMULATOR_PARAMS = snapshot


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


def make_train_env(seed: int, condition_key: str, algo_tag: str) -> gym.Env:
    """Build training env; apply diversity wrapper only when condition requires it."""
    from stable_baselines3.common.monitor import Monitor

    spec = DIVERSITY_CONDITIONS[condition_key]
    core = StudentEnv(
        render_mode=cfg.RENDER_MODE,
        max_episode_steps=cfg.MAX_EPISODE_STEPS,
        population_seed=seed,
        obs_ablation="full_emotion",
        emotion_dynamics="full",
    )
    mode = spec.get("mode")
    amount = float(spec.get("amount", 0.0))
    if mode in ("bonus", "penalty"):
        core = DiversityRewardWrapper(core, mode=mode, amount=amount)
    env = DQNSafeEnv(core)
    return Monitor(
        env,
        filename=str(cfg.LOGS_DIR / f"monitor_{algo_tag}_seed{seed}"),
    )


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
    """Evaluate with baseline reward (no diversity shaping)."""
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
        "shannon_entropy": _shannon_entropy(freqs),
        "actions_used_above_1_percent": _actions_used_above_threshold(freqs, 0.01),
        "actions_used_above_5_percent": _actions_used_above_threshold(freqs, 0.05),
        "dominant_action_pct": _dominant_action_pct(freqs),
        "top2_actions_pct": _top_k_actions_pct(freqs, 2),
        "dominance_detected": freq_report["dominance_detected"],
    }


def run_single(condition_key: str, seed: int) -> Dict[str, Any]:
    spec = DIVERSITY_CONDITIONS[condition_key]
    sim_snap = _patch_simulator()
    tag = f"action_diversity_{condition_key}_s{seed}"
    t0 = time.time()

    try:
        cfg.set_all_seeds(seed)
        train_env = make_train_env(seed, condition_key, tag)
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
        "configuration": condition_key,
        "configuration_label": spec["label"],
        "diversity_mode": spec.get("mode"),
        "diversity_amount": spec.get("amount", 0.0),
        "seed": seed,
        "irt_beta": IRT_BETA,
        "train_timesteps": TRAIN_TS,
        "eval_episodes": EVAL_EPS,
        "use_checkpoints": True,
        "selection": selection,
        "val_learning_gain": best_val,
        "gain_ratio": G4_GAIN["ratio_label"],
        "elapsed_seconds": round(elapsed, 1),
        **{m: metrics[m] for m in METRICS},
        "shannon_entropy": metrics["shannon_entropy"],
        "actions_used_above_1_percent": metrics["actions_used_above_1_percent"],
        "actions_used_above_5_percent": metrics["actions_used_above_5_percent"],
        "dominant_action_pct": metrics["dominant_action_pct"],
        "top2_actions_pct": metrics["top2_actions_pct"],
        "dominance_detected": metrics["dominance_detected"],
    }
    for action_name in ACTION_NAMES:
        row[action_name] = metrics["action_frequencies"].get(action_name, 0.0)

    print(
        f"  {condition_key} seed={seed}: "
        f"lg={row['learning_gain']:.3f} succ={row['success_rate']:.3f} "
        f"adapt={row['adaptation_accuracy']:.3f} entropy={row['shannon_entropy']:.3f} "
        f"dom={row['dominant_action_pct']:.2f} ({elapsed:.0f}s)"
    )
    return row


def _load_resume(path: Path, resume: bool) -> List[Dict[str, Any]]:
    if resume and path.exists():
        return pd.read_csv(path).to_dict("records")
    return []


def run_study(seeds: List[int], resume: bool) -> pd.DataFrame:
    csv_path = OUT_DIR / "action_diversity_results.csv"
    rows = _load_resume(csv_path, resume)
    done = {(r["configuration"], int(r["seed"])) for r in rows}

    for condition_key in DIVERSITY_CONDITIONS:
        print(f"\n--- {DIVERSITY_CONDITIONS[condition_key]['label']} ---")
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
        spec = DIVERSITY_CONDITIONS[configuration]
        lgs = sub["learning_gain"].astype(float).tolist()
        lg_mean, lg_lo, lg_hi = confidence_interval(lgs)
        lg_std = float(np.std(lgs, ddof=1)) if len(lgs) > 1 else 0.0
        lg_range = float(max(lgs) - min(lgs)) if lgs else 0.0
        best_idx = sub["learning_gain"].astype(float).idxmax()
        worst_idx = sub["learning_gain"].astype(float).idxmin()

        row: Dict[str, Any] = {
            "configuration": configuration,
            "configuration_label": spec["label"],
            "diversity_mode": spec.get("mode"),
            "diversity_amount": spec.get("amount", 0.0),
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
            "shannon_entropy_mean": round(float(sub["shannon_entropy"].mean()), 4),
            "shannon_entropy_std": round(
                float(sub["shannon_entropy"].std(ddof=1)) if len(sub) > 1 else 0.0, 4
            ),
            "dominant_action_pct_mean": round(float(sub["dominant_action_pct"].mean()), 4),
            "dominant_action_pct_std": round(
                float(sub["dominant_action_pct"].std(ddof=1)) if len(sub) > 1 else 0.0, 4
            ),
            "top2_actions_pct_mean": round(float(sub["top2_actions_pct"].mean()), 4),
            "top2_actions_pct_std": round(
                float(sub["top2_actions_pct"].std(ddof=1)) if len(sub) > 1 else 0.0, 4
            ),
            "actions_used_above_1_percent_mean": round(
                float(sub["actions_used_above_1_percent"].mean()), 4
            ),
            "actions_used_above_5_percent_mean": round(
                float(sub["actions_used_above_5_percent"].mean()), 4
            ),
        }
        for metric in METRICS:
            if metric == "learning_gain":
                continue
            vals = sub[metric].astype(float).tolist()
            row[f"{metric}_mean"] = round(float(np.mean(vals)), 4)
            row[f"{metric}_std"] = round(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, 4)
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_action_frequency(results_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for configuration, sub in results_df.groupby("configuration"):
        spec = DIVERSITY_CONDITIONS[configuration]
        row: Dict[str, Any] = {
            "configuration": configuration,
            "configuration_label": spec["label"],
            "n_seeds": len(sub),
        }
        for action_name in ACTION_NAMES:
            if action_name in sub.columns:
                vals = sub[action_name].astype(float).tolist()
                row[action_name] = round(float(np.mean(vals)), 4)
                row[f"{action_name}_std"] = round(
                    float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, 4
                )
        rows.append(row)
    return pd.DataFrame(rows)


def build_ranking(summary_df: pd.DataFrame) -> pd.DataFrame:
    merged = summary_df.copy()
    merged = merged.rename(columns={
        "shannon_entropy_mean": "action_entropy",
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
        "dominant_action_pct_mean": True,
        "top2_actions_pct_mean": True,
        "actions_used_above_1_percent_mean": False,
        "actions_used_above_5_percent_mean": False,
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
        "dominant_action_pct_mean",
        "top2_actions_pct_mean",
        "actions_used_above_1_percent_mean",
        "actions_used_above_5_percent_mean",
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

    order = list(DIVERSITY_CONDITIONS.keys())
    summary_df = summary_df.set_index("configuration").loc[order].reset_index()
    labels = [
        DIVERSITY_CONDITIONS[c]["label"].replace(" (", "\n(") for c in order
    ]
    colors = ["#457B9D", "#2A9D8F", "#E9C46A", "#E76F51", "#9B5DE5"]
    xs = np.arange(len(labels))

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    panels = [
        ("learning_gain_mean", "learning_gain_std", "Learning Gain"),
        ("success_rate_mean", "success_rate_std", "Success Rate"),
        ("adaptation_accuracy_mean", "adaptation_accuracy_std", "Adaptation Accuracy"),
        ("mean_episode_reward_mean", "mean_episode_reward_std", "Mean Episode Reward"),
        ("shannon_entropy_mean", "shannon_entropy_std", "Shannon Entropy"),
        ("dominant_action_pct_mean", "dominant_action_pct_std", "Dominant Action %"),
    ]
    for ax, (mean_col, std_col, title) in zip(axes.flat, panels):
        means = summary_df[mean_col].tolist()
        stds = summary_df[std_col].tolist() if std_col in summary_df.columns else [0.0] * len(means)
        ax.bar(xs, means, yerr=stds, capsize=4, color=colors, edgecolor="#64748B")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Action Diversity Reward Ablation (DQN, beta={IRT_BETA}, G4 {G4_GAIN['ratio_label']}, "
        f"{TRAIN_TS // 1000}k steps, checkpoint selection)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "action_diversity_comparison.png", dpi=150)
    plt.close(fig)


def _plot_action_frequency(freq_df: pd.DataFrame) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    configs = list(DIVERSITY_CONDITIONS.keys())
    freq_df = freq_df.set_index("configuration").loc[configs].reset_index()
    x = np.arange(len(ACTION_NAMES))
    width = 0.15

    fig, ax = plt.subplots(figsize=(14, 6))
    palette = ["#457B9D", "#2A9D8F", "#E9C46A", "#E76F51", "#9B5DE5"]
    for i, (_, row) in enumerate(freq_df.iterrows()):
        freqs = [row.get(name, 0.0) for name in ACTION_NAMES]
        offset = (i - 2) * width
        ax.bar(
            x + offset,
            freqs,
            width=width,
            label=row["configuration_label"],
            color=palette[i % len(palette)],
            edgecolor="#64748B",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(ACTION_NAMES, rotation=35, ha="right")
    ax.set_ylabel("Mean action frequency")
    ax.set_title("Action Frequencies by Diversity Configuration")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "action_frequency_comparison.png", dpi=150)
    plt.close(fig)


def _print_winner(
    results_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
) -> None:
    baseline_key = "A_baseline"
    baseline = summary_df[summary_df["configuration"] == baseline_key].iloc[0]
    winner = ranking_df.iloc[0]
    winner_row = summary_df[summary_df["configuration"] == winner["configuration"]].iloc[0]

    best_entropy = summary_df.loc[summary_df["shannon_entropy_mean"].idxmax()]
    lowest_dom = summary_df.loc[summary_df["dominant_action_pct_mean"].idxmin()]
    best_lg = summary_df.loc[summary_df["learning_gain_mean"].idxmax()]
    most_stable = summary_df.loc[summary_df["learning_gain_std"].idxmin()]

    def _vs_baseline(col_mean: str, higher_better: bool) -> Dict[str, Any]:
        b = float(baseline[col_mean])
        improved = []
        hurt = []
        for _, row in summary_df.iterrows():
            if row["configuration"] == baseline_key:
                continue
            v = float(row[col_mean])
            if higher_better and v > b + 1e-4:
                improved.append((row["configuration_label"], v))
            elif not higher_better and v < b - 1e-4:
                improved.append((row["configuration_label"], v))
            elif higher_better and v < b - 0.02:
                hurt.append((row["configuration_label"], v))
            elif not higher_better and v > b + 0.02:
                hurt.append((row["configuration_label"], v))
        return {"improved": improved, "hurt": hurt, "baseline": b}

    dom_cmp = _vs_baseline("dominant_action_pct_mean", higher_better=False)
    lg_cmp = _vs_baseline("learning_gain_mean", higher_better=True)
    entropy_cmp = _vs_baseline("shannon_entropy_mean", higher_better=True)
    std_cmp = _vs_baseline("learning_gain_std", higher_better=False)

    print("\n" + "=" * 72)
    print("WINNER DIVERSITY CONFIGURATION")
    print("=" * 72)
    print(f"\nOverall rank #1: {winner['configuration_label']}")
    print(f"  diversity_mode={DIVERSITY_CONDITIONS[winner['configuration']].get('mode')}, "
          f"amount={DIVERSITY_CONDITIONS[winner['configuration']].get('amount', 0.0)}")
    print(f"  learning_gain_mean={winner['learning_gain_mean']:.4f} "
          f"(std={winner['learning_gain_std']:.4f}, CV={winner['coefficient_of_variation']:.4f})")
    print(f"  success_rate_mean={winner['success_rate_mean']:.4f}")
    print(f"  adaptation_accuracy_mean={winner['adaptation_accuracy_mean']:.4f}")
    print(f"  reward_mean={winner['reward_mean']:.4f}")
    print(f"  action_entropy={winner['action_entropy']:.4f}")
    print(f"  dominant_action_pct={winner['dominant_action_pct_mean']:.4f}")
    print(f"  top2_actions_pct={winner['top2_actions_pct_mean']:.4f}")
    print(f"  actions_above_1%={winner['actions_used_above_1_percent_mean']:.1f}, "
          f"actions_above_5%={winner['actions_used_above_5_percent_mean']:.1f}")

    print("\n--- Research Questions ---")
    print("\n1. Does diversity encouragement reduce action dominance?")
    if dom_cmp["improved"]:
        print(
            f"   YES (partially). vs baseline dominant_action_pct={dom_cmp['baseline']:.3f}:"
        )
        for label, v in dom_cmp["improved"]:
            print(f"     - {label}: {v:.3f}")
    else:
        print(
            f"   NO clear reduction. Baseline dominant_action_pct="
            f"{baseline['dominant_action_pct_mean']:.3f}; "
            f"lowest dominance: {lowest_dom['configuration_label']} "
            f"({lowest_dom['dominant_action_pct_mean']:.3f})."
        )

    print("\n2. Does diversity encouragement improve learning gain?")
    if lg_cmp["improved"]:
        print(f"   YES for: {', '.join(l[0] for l in lg_cmp['improved'])}")
        print(f"   Baseline LG={lg_cmp['baseline']:.4f}; best LG={best_lg['learning_gain_mean']:.4f} "
              f"({best_lg['configuration_label']})")
    else:
        print(
            f"   NO. Baseline LG={baseline['learning_gain_mean']:.4f} remains best or tied; "
            f"highest LG={best_lg['learning_gain_mean']:.4f} ({best_lg['configuration_label']})."
        )
    if lg_cmp["hurt"]:
        print(f"   Configurations with LG penalty vs baseline: "
              f"{', '.join(l[0] for l in lg_cmp['hurt'])}")

    print("\n3. Does diversity encouragement improve stability?")
    if std_cmp["improved"]:
        print(
            f"   YES (lower seed variance). Most stable: {most_stable['configuration_label']} "
            f"(std={most_stable['learning_gain_std']:.4f}, "
            f"CV={most_stable['coefficient_of_variation']:.4f})."
        )
    else:
        print(
            f"   MIXED / NO. Baseline std={baseline['learning_gain_std']:.4f}; "
            f"most stable: {most_stable['configuration_label']} "
            f"(std={most_stable['learning_gain_std']:.4f})."
        )

    print("\n4. Best tradeoff between educational performance and policy diversity?")
    print(
        f"   {winner['configuration_label']} - composite rank balances LG, stability, "
        f"entropy, and reduced dominance."
    )

    print("\n--- Interpretation ---")
    print(
        f"Highest Shannon entropy: {best_entropy['configuration_label']} "
        f"({best_entropy['shannon_entropy_mean']:.4f})."
    )
    print(
        f"Lowest dominant-action share: {lowest_dom['configuration_label']} "
        f"({lowest_dom['dominant_action_pct_mean']:.4f})."
    )

    if winner["configuration"] == baseline_key:
        print(
            "\nThe baseline action-agnostic reward remains optimal: diversity shaping during "
            "training did not improve eval-time learning outcomes enough to justify the added "
            "reward complexity. Action dominance is better addressed via simulator or mask design."
        )
    elif DIVERSITY_CONDITIONS[winner["configuration"]]["mode"] == "bonus":
        print(
            "\nA diversity BONUS outperforms penalties: rewarding action switches nudges exploration "
            "without punishing pedagogically appropriate repetition (e.g. scaffold sequences)."
        )
    else:
        print(
            "\nA repetition PENALTY configuration wins: discouraging consecutive identical actions "
            "reduced collapse while preserving acceptable learning gain."
        )

    if float(winner_row["learning_gain_mean"]) >= float(baseline["learning_gain_mean"]) - 0.02:
        if float(winner_row["shannon_entropy_mean"]) > float(baseline["shannon_entropy_mean"]):
            print(
                "Recommendation: adopt the winning diversity term at training time; it improves "
                "policy breadth with negligible learning-gain cost."
            )
        else:
            print(
                "Recommendation: diversity shaping does not materially broaden the policy; "
                "keep the baseline reward."
            )
    else:
        print(
            "Recommendation: do NOT adopt diversity shaping - learning gain drops outweigh "
            "diversity benefits."
        )
    print("=" * 72)


def analyze(results_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    csv_path = OUT_DIR / "action_diversity_results.csv"
    if results_df is None:
        if not csv_path.exists():
            raise FileNotFoundError(f"No results at {csv_path}; run --phase run first.")
        results_df = pd.read_csv(csv_path)

    summary_df = aggregate_summary(results_df)
    freq_df = aggregate_action_frequency(results_df)
    ranking_df = build_ranking(summary_df)

    results_df.to_csv(OUT_DIR / "action_diversity_results.csv", index=False)
    summary_df.to_csv(OUT_DIR / "action_diversity_summary.csv", index=False)
    freq_df.to_csv(OUT_DIR / "action_frequency_summary.csv", index=False)
    ranking_df.to_csv(OUT_DIR / "action_diversity_ranking.csv", index=False)

    _plot_comparison(summary_df)
    _plot_action_frequency(freq_df)
    _print_winner(results_df, summary_df, ranking_df)

    report = {
        "study": "action_diversity_ablation",
        "irt_beta": IRT_BETA,
        "gain_ratio": G4_GAIN["ratio_label"],
        "baseline_hp": BASELINE_HP,
        "diversity_conditions": DIVERSITY_CONDITIONS,
        "seeds": SEEDS,
        "summary": summary_df.to_dict("records"),
        "action_frequency": freq_df.to_dict("records"),
        "ranking": ranking_df.to_dict("records"),
    }
    with open(OUT_DIR / "action_diversity_ablation_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    return report


def main() -> None:
    global TRAIN_TS

    parser = argparse.ArgumentParser(
        description="Action diversity reward ablation (DQN, runtime training reward wrapper)"
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
        print("=== Action Diversity Reward Ablation (run) ===")
        TRAIN_TS = train_ts
        run_study(seeds, args.resume)

    if args.phase in ("analyze", "all"):
        print("\n=== Action Diversity Reward Ablation (analyze) ===")
        analyze()


if __name__ == "__main__":
    main()
