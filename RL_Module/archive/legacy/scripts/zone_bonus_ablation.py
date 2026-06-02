"""
Zone bonus ablation: does the optional flow-zone reward term drive DQN instability or bias?

Compares DQN under runtime-patched USE_ZONE_BONUS only (reward function unchanged on disk):
  A zone_bonus_on:  USE_ZONE_BONUS = True  (legacy flow-zone bonus active)
  B zone_bonus_off: USE_ZONE_BONUS = False (repository default)

Protocol: DQN only, G4 gain ratio, full_emotion, mixed population, default transition noise,
checkpoint selection, 50k train, 500 eval, 6 seeds [42, 7, 13, 21, 99, 314],
dqn_stability_study baseline hyperparameters.

Usage (from repo root):
  export PYTHONPATH="$(pwd)"
  python -m RL_Module.zone_bonus_ablation --phase all
  python -m RL_Module.zone_bonus_ablation --phase run --resume
  python -m RL_Module.zone_bonus_ablation --phase analyze
  python -m RL_Module.zone_bonus_ablation --phase all --quick
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
from RL_Module.mdp_definition import ACTION_DIM, ID_TO_ACTION, StudentState

OUT_DIR = _HERE / "figures" / "zone_bonus_ablation"
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

ZONE_CONDITIONS: Dict[str, Dict[str, Any]] = {
    "A_zone_bonus_on": {
        "label": "Zone Bonus ON (A)",
        "use_zone_bonus": True,
    },
    "B_zone_bonus_off": {
        "label": "Zone Bonus OFF (B)",
        "use_zone_bonus": False,
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

ZONE_METRICS = [
    "pct_time_in_zone",
    "avg_consecutive_zone_duration",
    "zone_entries_per_episode",
]


def _patch_gain() -> Dict[str, float]:
    prev = dict(cfg.SIMULATOR_PARAMS)
    cfg.SIMULATOR_PARAMS = dict(G4_GAIN)
    return prev


def _restore_gain(snapshot: Dict[str, float]) -> None:
    cfg.SIMULATOR_PARAMS = snapshot


def _patch_zone_bonus(use_zone_bonus: bool) -> bool:
    prev = bool(cfg.USE_ZONE_BONUS)
    cfg.USE_ZONE_BONUS = bool(use_zone_bonus)
    return prev


def _restore_zone_bonus(prev: bool) -> None:
    cfg.USE_ZONE_BONUS = prev


def _state_in_zone(state: StudentState) -> bool:
    return (
        state.knowledge > cfg.ZONE_MIN_KNOWLEDGE
        and state.engagement > cfg.ZONE_MIN_ENGAGEMENT
        and state.frustration < cfg.ZONE_MAX_FRUSTRATION
        and state.confusion < cfg.ZONE_MAX_CONFUSION
        and state.boredom < cfg.ZONE_MAX_BOREDOM
    )


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
    zone_pcts: List[float] = []
    zone_durations: List[float] = []
    zone_entries_list: List[int] = []

    for ep in range(1, n_episodes + 1):
        obs, info = env.reset(seed=seed + ep)
        mask = info["action_masks"]
        start_k = env._state.knowledge
        total_reward = 0.0
        adapt_hits = adapt_total = 0
        action_counts = {i: 0 for i in range(ACTION_DIM)}
        dropout = False
        done = False

        total_steps = 0
        zone_steps = 0
        zone_entries = 0
        in_zone = _state_in_zone(env._state)
        current_run = 1 if in_zone else 0
        run_lengths: List[int] = []

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

            total_steps += 1
            now_in_zone = _state_in_zone(env._state)
            if now_in_zone:
                zone_steps += 1
            if now_in_zone and not in_zone:
                zone_entries += 1
                current_run = 1
            elif now_in_zone and in_zone:
                current_run += 1
            elif not now_in_zone and in_zone:
                if current_run > 0:
                    run_lengths.append(current_run)
                current_run = 0
            in_zone = now_in_zone

        if in_zone and current_run > 0:
            run_lengths.append(current_run)

        pct_in_zone = zone_steps / max(total_steps, 1)
        avg_run = float(np.mean(run_lengths)) if run_lengths else 0.0
        zone_pcts.append(pct_in_zone)
        zone_durations.append(avg_run)
        zone_entries_list.append(zone_entries)

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
        "pct_time_in_zone": float(np.mean(zone_pcts)),
        "avg_consecutive_zone_duration": float(np.mean(zone_durations)),
        "zone_entries_per_episode": float(np.mean(zone_entries_list)),
    }


def run_single(condition_key: str, seed: int) -> Dict[str, Any]:
    spec = ZONE_CONDITIONS[condition_key]
    use_zone_bonus = bool(spec["use_zone_bonus"])
    gain_snap = _patch_gain()
    zone_snap = _patch_zone_bonus(use_zone_bonus)
    tag = f"zone_bonus_{condition_key}_s{seed}"
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
        _restore_zone_bonus(zone_snap)
        _restore_gain(gain_snap)

    elapsed = time.time() - t0
    row: Dict[str, Any] = {
        "configuration": condition_key,
        "configuration_label": spec["label"],
        "use_zone_bonus": use_zone_bonus,
        "seed": seed,
        "train_timesteps": TRAIN_TS,
        "eval_episodes": EVAL_EPS,
        "use_checkpoints": True,
        "selection": selection,
        "val_learning_gain": best_val,
        "gain_ratio": G4_GAIN["ratio_label"],
        "elapsed_seconds": round(elapsed, 1),
        **{m: metrics[m] for m in METRICS},
        **{m: metrics[m] for m in ZONE_METRICS},
        "action_entropy": metrics["action_entropy"],
        "actions_used_above_1_percent": metrics["actions_used_above_1_percent"],
        "dominant_action_pct": metrics["dominant_action_pct"],
        "dominance_detected": metrics["dominance_detected"],
    }
    for action_name, freq in metrics["action_frequencies"].items():
        row[f"freq_{action_name}"] = freq

    print(
        f"  {condition_key} seed={seed} zone_bonus={use_zone_bonus}: "
        f"lg={row['learning_gain']:.3f} succ={row['success_rate']:.3f} "
        f"reward={row['mean_episode_reward']:.1f} entropy={row['action_entropy']:.3f} "
        f"dom={row['dominant_action_pct']:.2f} ({elapsed:.0f}s)"
    )
    return row


def _load_resume(path: Path, resume: bool) -> List[Dict[str, Any]]:
    if resume and path.exists():
        return pd.read_csv(path).to_dict("records")
    return []


def run_study(seeds: List[int], resume: bool) -> pd.DataFrame:
    csv_path = OUT_DIR / "zone_bonus_results.csv"
    rows = _load_resume(csv_path, resume)
    done = {(r["configuration"], int(r["seed"])) for r in rows}

    for condition_key in ZONE_CONDITIONS:
        print(f"\n--- {ZONE_CONDITIONS[condition_key]['label']} ---")
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
        spec = ZONE_CONDITIONS[configuration]
        lgs = sub["learning_gain"].astype(float).tolist()
        lg_mean, lg_lo, lg_hi = confidence_interval(lgs)
        lg_std = float(np.std(lgs, ddof=1)) if len(lgs) > 1 else 0.0
        lg_range = float(max(lgs) - min(lgs)) if lgs else 0.0
        best_idx = sub["learning_gain"].astype(float).idxmax()
        worst_idx = sub["learning_gain"].astype(float).idxmin()

        row: Dict[str, Any] = {
            "configuration": configuration,
            "configuration_label": spec["label"],
            "use_zone_bonus": spec["use_zone_bonus"],
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
        spec = ZONE_CONDITIONS[configuration]
        entropies = sub["action_entropy"].astype(float).tolist()
        actions_used = sub["actions_used_above_1_percent"].astype(int).tolist()
        dominant = sub["dominant_action_pct"].astype(float).tolist()
        row: Dict[str, Any] = {
            "configuration": configuration,
            "configuration_label": spec["label"],
            "use_zone_bonus": spec["use_zone_bonus"],
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


def aggregate_zone_statistics(results_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for configuration, sub in results_df.groupby("configuration"):
        spec = ZONE_CONDITIONS[configuration]
        row: Dict[str, Any] = {
            "configuration": configuration,
            "configuration_label": spec["label"],
            "use_zone_bonus": spec["use_zone_bonus"],
            "n_seeds": len(sub),
        }
        for metric in ZONE_METRICS:
            vals = sub[metric].astype(float).tolist()
            row[f"{metric}_mean"] = round(float(np.mean(vals)), 4)
            row[f"{metric}_std"] = round(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, 4)
        rows.append(row)
    return pd.DataFrame(rows)


def build_ranking(summary_df: pd.DataFrame, diversity_df: pd.DataFrame) -> pd.DataFrame:
    merged = summary_df.merge(
        diversity_df[[
            "configuration",
            "shannon_entropy_mean",
            "dominant_action_pct_mean",
        ]],
        on="configuration",
    )
    merged = merged.rename(columns={
        "shannon_entropy_mean": "action_entropy",
        "dominant_action_pct_mean": "dominant_action_pct",
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

    summary_df = summary_df.sort_values("use_zone_bonus", ascending=False)
    labels = [row["configuration_label"].replace(" (", "\n(") for _, row in summary_df.iterrows()]
    colors = ["#E76F51", "#457B9D"]
    xs = np.arange(len(labels))

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    panels = [
        ("learning_gain_mean", "learning_gain_std", "Learning Gain"),
        ("success_rate_mean", "success_rate_std", "Success Rate"),
        ("adaptation_accuracy_mean", "adaptation_accuracy_std", "Adaptation Accuracy"),
        ("mean_episode_reward_mean", "mean_episode_reward_std", "Mean Episode Reward"),
    ]
    for ax, (mean_col, std_col, title) in zip(axes.flat, panels):
        means = summary_df[mean_col].tolist()
        stds = summary_df[std_col].tolist() if std_col in summary_df.columns else [0.0] * len(means)
        ax.bar(xs, means, yerr=stds, capsize=5, color=colors[: len(xs)], edgecolor="#64748B")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Zone Bonus Ablation (DQN, G4 {G4_GAIN['ratio_label']}, "
        f"{TRAIN_TS // 1000}k steps, checkpoint selection)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "zone_bonus_comparison.png", dpi=150)
    plt.close(fig)


def _print_winner(
    summary_df: pd.DataFrame,
    diversity_df: pd.DataFrame,
    zone_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
) -> None:
    on_key = "A_zone_bonus_on"
    off_key = "B_zone_bonus_off"
    on = summary_df[summary_df["configuration"] == on_key].iloc[0]
    off = summary_df[summary_df["configuration"] == off_key].iloc[0]
    on_div = diversity_df[diversity_df["configuration"] == on_key].iloc[0]
    off_div = diversity_df[diversity_df["configuration"] == off_key].iloc[0]
    on_zone = zone_df[zone_df["configuration"] == on_key].iloc[0]
    off_zone = zone_df[zone_df["configuration"] == off_key].iloc[0]
    winner = ranking_df.iloc[0]

    off_more_stable = float(off["learning_gain_std"]) < float(on["learning_gain_std"])
    off_less_bias = float(off_div["dominant_action_pct_mean"]) < float(on_div["dominant_action_pct_mean"])
    off_higher_entropy = float(off_div["shannon_entropy_mean"]) > float(on_div["shannon_entropy_mean"])
    off_lower_lg = float(off["learning_gain_mean"]) < float(on["learning_gain_mean"])
    off_lower_reward = float(off["mean_episode_reward_mean"]) < float(on["mean_episode_reward_mean"])
    reward_inflated = (
        float(on["mean_episode_reward_mean"]) > float(off["mean_episode_reward_mean"])
        and float(on["learning_gain_mean"]) <= float(off["learning_gain_mean"])
    )

    print("\n" + "=" * 72)
    print("WINNER ZONE BONUS CONFIGURATION")
    print("=" * 72)
    print(f"\nOverall rank #1: {winner['configuration']} "
          f"(USE_ZONE_BONUS={ZONE_CONDITIONS[winner['configuration']]['use_zone_bonus']})")
    print(f"  learning_gain_mean={winner['learning_gain_mean']:.4f} "
          f"(std={winner['learning_gain_std']:.4f})")
    print(f"  success_rate_mean={winner['success_rate_mean']:.4f}")
    print(f"  adaptation_accuracy_mean={winner['adaptation_accuracy_mean']:.4f}")
    print(f"  reward_mean={winner['reward_mean']:.4f}")
    print(f"  action_entropy={winner['action_entropy']:.4f}, "
          f"dominant_action_pct={winner['dominant_action_pct']:.4f}")

    print("\n--- Hypothesis Tests ---")
    print("\nA. Does removing the zone bonus reduce seed variance?")
    if off_more_stable:
        print(
            f"   YES. OFF std={off['learning_gain_std']:.4f}, CV={off['coefficient_of_variation']:.4f} "
            f"vs ON std={on['learning_gain_std']:.4f}, CV={on['coefficient_of_variation']:.4f}."
        )
    else:
        print(
            f"   NO. Zone bonus OFF did not reduce variance "
            f"(ON std={on['learning_gain_std']:.4f}, OFF std={off['learning_gain_std']:.4f})."
        )

    print("\nB. Does removing the zone bonus reduce action bias?")
    if off_less_bias or off_higher_entropy:
        print(
            f"   YES. OFF dominant_action={off_div['dominant_action_pct_mean']:.3f}, "
            f"entropy={off_div['shannon_entropy_mean']:.3f} vs "
            f"ON dominant={on_div['dominant_action_pct_mean']:.3f}, "
            f"entropy={on_div['shannon_entropy_mean']:.3f}."
        )
    else:
        print(
            f"   NO. Action bias was not clearly reduced without the zone bonus "
            f"(ON dom={on_div['dominant_action_pct_mean']:.3f}, "
            f"OFF dom={off_div['dominant_action_pct_mean']:.3f})."
        )

    print("\nC. Does removing the zone bonus reduce learning gain?")
    if off_lower_lg:
        print(
            f"   YES. Mean LG drops without bonus "
            f"(ON={on['learning_gain_mean']:.4f}, OFF={off['learning_gain_mean']:.4f})."
        )
    else:
        print(
            f"   NO. Learning gain is equal or higher without bonus "
            f"(ON={on['learning_gain_mean']:.4f}, OFF={off['learning_gain_mean']:.4f})."
        )

    print("\nD. Is the zone bonus responsible for observed instability?")
    instability_from_bonus = (
        float(on["learning_gain_std"]) > float(off["learning_gain_std"])
        or float(on["coefficient_of_variation"]) > float(off["coefficient_of_variation"])
    )
    if instability_from_bonus:
        print(
            "   PARTIALLY YES. ON shows higher seed spread "
            f"(range ON={on['seed_range']:.4f}, OFF={off['seed_range']:.4f}; "
            f"best/worst seeds ON={on['best_seed']}/{on['worst_seed']}, "
            f"OFF={off['best_seed']}/{off['worst_seed']})."
        )
    else:
        print(
            "   NO. Seed variance is not clearly driven by the zone bonus term alone."
        )

    print("\nE. Is the zone bonus helping learning or merely increasing reward?")
    if reward_inflated:
        print(
            f"   MERELY INFLATING REWARD. ON reward={on['mean_episode_reward_mean']:.2f} vs "
            f"OFF={off['mean_episode_reward_mean']:.2f} without higher LG "
            f"(ON LG={on['learning_gain_mean']:.4f}, OFF LG={off['learning_gain_mean']:.4f})."
        )
    elif float(on["learning_gain_mean"]) > float(off["learning_gain_mean"]) and not off_lower_reward:
        print(
            f"   HELPING LEARNING. ON improves LG ({on['learning_gain_mean']:.4f} vs "
            f"{off['learning_gain_mean']:.4f}) with higher zone occupancy "
            f"({on_zone['pct_time_in_zone_mean']:.3f} vs {off_zone['pct_time_in_zone_mean']:.3f})."
        )
    else:
        print(
            f"   MIXED / UNCLEAR. Reward delta "
            f"{float(on['mean_episode_reward_mean']) - float(off['mean_episode_reward_mean']):+.2f}; "
            f"LG delta {float(on['learning_gain_mean']) - float(off['learning_gain_mean']):+.4f}."
        )

    print("\n--- Thesis Recommendation ---")
    keep_bonus = (
        float(on["learning_gain_mean"]) > float(off["learning_gain_mean"])
        and not instability_from_bonus
        and not reward_inflated
    )
    if keep_bonus:
        print(
            "The zone bonus appears justified: it improves educational outcomes without "
            "clear harm to stability. Consider enabling USE_ZONE_BONUS in the thesis simulator."
        )
    else:
        print(
            "The zone bonus is NOT justified for the thesis simulator: it either inflates "
            "reward without learning benefit, increases seed variance, or biases action selection. "
            "Keep USE_ZONE_BONUS=False (repository default)."
        )
    print("=" * 72)


def analyze(results_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    csv_path = OUT_DIR / "zone_bonus_results.csv"
    if results_df is None:
        if not csv_path.exists():
            raise FileNotFoundError(f"No results at {csv_path}; run --phase run first.")
        results_df = pd.read_csv(csv_path)

    summary_df = aggregate_summary(results_df)
    diversity_df = aggregate_action_diversity(results_df)
    zone_df = aggregate_zone_statistics(results_df)
    ranking_df = build_ranking(summary_df, diversity_df)

    summary_df.to_csv(OUT_DIR / "zone_bonus_summary.csv", index=False)
    diversity_df.to_csv(OUT_DIR / "action_diversity_summary.csv", index=False)
    zone_df.to_csv(OUT_DIR / "zone_statistics.csv", index=False)
    ranking_df.to_csv(OUT_DIR / "zone_bonus_ranking.csv", index=False)

    _plot_comparison(summary_df)
    _print_winner(summary_df, diversity_df, zone_df, ranking_df)

    report = {
        "study": "zone_bonus_ablation",
        "gain_ratio": G4_GAIN["ratio_label"],
        "baseline_hp": BASELINE_HP,
        "zone_conditions": ZONE_CONDITIONS,
        "seeds": SEEDS,
        "summary": summary_df.to_dict("records"),
        "action_diversity": diversity_df.to_dict("records"),
        "zone_statistics": zone_df.to_dict("records"),
        "ranking": ranking_df.to_dict("records"),
    }
    with open(OUT_DIR / "zone_bonus_ablation_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    return report


def main() -> None:
    global TRAIN_TS

    parser = argparse.ArgumentParser(description="Zone bonus ablation for DQN stability and policy bias")
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
        print("=== Zone Bonus Ablation (run) ===")
        TRAIN_TS = train_ts
        run_study(seeds, args.resume)

    if args.phase in ("analyze", "all"):
        print("\n=== Zone Bonus Ablation (analyze) ===")
        analyze()


if __name__ == "__main__":
    main()
