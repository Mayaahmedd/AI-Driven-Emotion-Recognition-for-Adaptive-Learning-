"""
Student profile ablation: are heterogeneous student profiles a major source of DQN instability?

Compares three population designs (runtime patch only; population.py unchanged):
  A mixed:              current repository behavior (full uniform ranges)
  B single_canonical:   every episode uses the same fixed personality profile
  C low_variance:       sampled students with narrow ranges around population means

Protocol: DQN only, G4 gain ratio, full_emotion, checkpoint selection, 50k train, 500 eval,
6 seeds [42, 7, 13, 21, 99, 314], dqn_stability_study baseline hyperparameters.

Robustness: train on each condition, evaluate on same population and on mixed population.

Usage (from repo root):
  export PYTHONPATH="$(pwd)"
  python -m RL_Module.student_profile_ablation --phase all
  python -m RL_Module.student_profile_ablation --phase run --resume
  python -m RL_Module.student_profile_ablation --phase analyze
  python -m RL_Module.student_profile_ablation --phase all --quick
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import RL_Module.config as cfg
import RL_Module.environment.population as pop_module
import RL_Module.environment.student_env as student_env_module
from RL_Module.agents.dqn_agent import DQNAgent, make_env
from RL_Module.environment.population import generate_population as _original_generate_population
from RL_Module.environment.student_model import SyntheticStudent
from RL_Module.environment.student_env import StudentEnv
from RL_Module.evaluation.metrics import (
    _adaptation_match,
    _is_success,
    action_frequency_report,
    confidence_interval,
    normalized_knowledge_gain,
)
from RL_Module.mdp_definition import ACTION_DIM, ID_TO_ACTION

OUT_DIR = _HERE / "figures" / "student_profile_ablation"
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
    "learning_rate": 0.0005,
    "batch_size": 64,
    "buffer_size": 100_000,
    "target_update_interval": 2000,
    "exploration_fraction": 0.3,
    "exploration_final_eps": 0.05,
    "net_arch": [64, 64],
}

# Personality parameter ranges from population.generate_population (repository default).
PERSONALITY_SPECS: Dict[str, Tuple[float, float]] = {
    "gamma_s": (0.1, 1.0),
    "beta_s": (0.1, 1.0),
    "lambda_s": (0.01, 0.3),
    "rho_s": (0.1, 1.0),
    "frustration_tolerance": (0.2, 1.0),
    "boredom_sensitivity": (0.2, 1.0),
    "engagement_recovery": (0.2, 0.9),
    "confidence": (-0.2, 0.3),
    "persistence": (0.1, 0.9),
}

CANONICAL_PARAMS: Dict[str, float] = {
    name: (lo + hi) / 2.0 for name, (lo, hi) in PERSONALITY_SPECS.items()
}

# Low-variance: sample within 10% of full range width on each side of the mean.
LOW_VARIANCE_FRAC = 0.10

POPULATION_CONDITIONS: Dict[str, Dict[str, Any]] = {
    "A_mixed": {
        "label": "Mixed Population (A)",
        "mode": "mixed",
    },
    "B_single_canonical": {
        "label": "Single Canonical Student (B)",
        "mode": "single_canonical",
    },
    "C_low_variance": {
        "label": "Low Variance Population (C)",
        "mode": "low_variance",
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


def _make_canonical_population(n: int) -> List[SyntheticStudent]:
    population: List[SyntheticStudent] = []
    for i in range(n):
        student = SyntheticStudent(
            gamma_s=CANONICAL_PARAMS["gamma_s"],
            beta_s=CANONICAL_PARAMS["beta_s"],
            lambda_s=CANONICAL_PARAMS["lambda_s"],
            rho_s=CANONICAL_PARAMS["rho_s"],
            frustration_tolerance=CANONICAL_PARAMS["frustration_tolerance"],
            boredom_sensitivity=CANONICAL_PARAMS["boredom_sensitivity"],
            engagement_recovery=CANONICAL_PARAMS["engagement_recovery"],
            confidence=CANONICAL_PARAMS["confidence"],
            persistence=CANONICAL_PARAMS["persistence"],
            student_id=i,
        )
        student.student_type = student.classify_type()
        population.append(student)
    return population


def _sample_low_variance(rng: np.random.Generator, lo: float, hi: float) -> float:
    mean = (lo + hi) / 2.0
    half_width = (hi - lo) * LOW_VARIANCE_FRAC
    return float(rng.uniform(max(lo, mean - half_width), min(hi, mean + half_width)))


def _make_low_variance_population(n: int, seed: Optional[int]) -> List[SyntheticStudent]:
    rng = np.random.default_rng(seed)
    population: List[SyntheticStudent] = []
    for i in range(n):
        student = SyntheticStudent(
            gamma_s=_sample_low_variance(rng, *PERSONALITY_SPECS["gamma_s"]),
            beta_s=_sample_low_variance(rng, *PERSONALITY_SPECS["beta_s"]),
            lambda_s=_sample_low_variance(rng, *PERSONALITY_SPECS["lambda_s"]),
            rho_s=_sample_low_variance(rng, *PERSONALITY_SPECS["rho_s"]),
            frustration_tolerance=_sample_low_variance(
                rng, *PERSONALITY_SPECS["frustration_tolerance"]
            ),
            boredom_sensitivity=_sample_low_variance(
                rng, *PERSONALITY_SPECS["boredom_sensitivity"]
            ),
            engagement_recovery=_sample_low_variance(
                rng, *PERSONALITY_SPECS["engagement_recovery"]
            ),
            confidence=_sample_low_variance(rng, *PERSONALITY_SPECS["confidence"]),
            persistence=_sample_low_variance(rng, *PERSONALITY_SPECS["persistence"]),
            student_id=i,
        )
        student.student_type = student.classify_type()
        population.append(student)
    return population


def _population_factory(mode: str):
    def _generate(n: int = cfg.POPULATION_SIZE, seed: Optional[int] = None) -> List[SyntheticStudent]:
        if mode == "mixed":
            return _original_generate_population(n=n, seed=seed)
        if mode == "single_canonical":
            return _make_canonical_population(n)
        if mode == "low_variance":
            return _make_low_variance_population(n, seed)
        raise ValueError(f"Unknown population mode: {mode!r}")

    return _generate


@contextmanager
def _population_patch(mode: str) -> Iterator[None]:
    """Temporarily override generate_population in population + student_env modules."""
    generator = _population_factory(mode)
    prev_pop = pop_module.generate_population
    prev_env = student_env_module.generate_population
    pop_module.generate_population = generator
    student_env_module.generate_population = generator
    try:
        yield
    finally:
        pop_module.generate_population = prev_pop
        student_env_module.generate_population = prev_env


def _patch_gain() -> Dict[str, float]:
    prev = dict(cfg.SIMULATOR_PARAMS)
    cfg.SIMULATOR_PARAMS = dict(G4_GAIN)
    return prev


def _restore_gain(snapshot: Dict[str, float]) -> None:
    cfg.SIMULATOR_PARAMS = snapshot


def _shannon_entropy(freq: Dict[str, float]) -> float:
    p = np.array([v for v in freq.values() if v > 0], dtype=float)
    return float(-np.sum(p * np.log(p))) if len(p) else 0.0


def _actions_used_above_1pct(freq: Dict[str, float], threshold: float = 0.01) -> int:
    return sum(1 for v in freq.values() if v >= threshold)


def make_eval_env(seed: int, population_mode: str) -> StudentEnv:
    with _population_patch(population_mode):
        return StudentEnv(
            population_seed=seed,
            obs_ablation="full_emotion",
            emotion_dynamics="full",
        )


def quick_eval_lg(
    agent: DQNAgent,
    seed: int,
    population_mode: str,
    n_episodes: int = VAL_EPS,
) -> float:
    with _population_patch(population_mode):
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
    population_mode: str,
    n_episodes: int = EVAL_EPS,
    collect_actions: bool = True,
) -> Dict[str, Any]:
    with _population_patch(population_mode):
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

    result: Dict[str, Any] = {
        "learning_gain": float(np.mean([r["learning_gain"] for r in episode_results])),
        "final_knowledge": float(np.mean([r["final_knowledge"] for r in episode_results])),
        "success_rate": float(np.mean([r["success"] for r in episode_results])),
        "adaptation_accuracy": float(np.mean([r["adaptation_accuracy"] for r in episode_results])),
        "mean_episode_reward": float(np.mean([r["total_reward"] for r in episode_results])),
        "dropout_rate": float(np.mean([r["dropout"] for r in episode_results])),
    }

    if collect_actions:
        freq_report = action_frequency_report(episode_results, print_table=False)
        freqs = freq_report["frequencies"]
        result.update({
            "action_frequencies": {k: round(v, 6) for k, v in freqs.items()},
            "action_entropy": _shannon_entropy(freqs),
            "actions_used_above_1_percent": _actions_used_above_1pct(freqs),
            "dominance_detected": freq_report["dominance_detected"],
        })

    return result


def run_single(condition_key: str, seed: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    spec = POPULATION_CONDITIONS[condition_key]
    train_mode = spec["mode"]
    gain_snap = _patch_gain()
    tag = f"student_profile_{condition_key}_s{seed}"
    t0 = time.time()
    robustness_rows: List[Dict[str, Any]] = []

    try:
        with _population_patch(train_mode):
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
            lambda a, s: quick_eval_lg(a, s, train_mode, VAL_EPS),
            seed,
        )
        agent.load_checkpoint(best_path)
        metrics = evaluate_dqn(agent, seed, train_mode, EVAL_EPS, collect_actions=True)
        selection = f"checkpoint@{best_path}"

        for eval_key, eval_spec in POPULATION_CONDITIONS.items():
            eval_mode = eval_spec["mode"]
            if eval_mode not in ("mixed", train_mode):
                continue
            eval_metrics = evaluate_dqn(
                agent, seed, eval_mode, EVAL_EPS, collect_actions=False
            )
            robustness_rows.append({
                "train_condition": condition_key,
                "train_condition_label": spec["label"],
                "eval_condition": eval_key,
                "eval_condition_label": eval_spec["label"],
                "eval_population_mode": eval_mode,
                "seed": seed,
                "checkpoint": best_path,
                **{m: eval_metrics[m] for m in METRICS},
            })
    finally:
        _restore_gain(gain_snap)

    elapsed = time.time() - t0
    row: Dict[str, Any] = {
        "condition": condition_key,
        "condition_label": spec["label"],
        "population_mode": train_mode,
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
        f"  {condition_key} seed={seed}: "
        f"lg={row['learning_gain']:.3f} succ={row['success_rate']:.3f} "
        f"entropy={row['action_entropy']:.3f} ({elapsed:.0f}s)"
    )
    return row, robustness_rows


def _load_resume(path: Path, resume: bool) -> List[Dict[str, Any]]:
    if resume and path.exists():
        return pd.read_csv(path).to_dict("records")
    return []


def run_study(seeds: List[int], resume: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    results_path = OUT_DIR / "student_profile_results.csv"
    robustness_path = OUT_DIR / "robustness_results.csv"
    rows = _load_resume(results_path, resume)
    robustness_rows = _load_resume(robustness_path, resume)
    done = {(r["condition"], int(r["seed"])) for r in rows}

    for condition_key in POPULATION_CONDITIONS:
        print(f"\n--- {POPULATION_CONDITIONS[condition_key]['label']} ---")
        for seed in seeds:
            if (condition_key, seed) in done:
                continue
            row, rob_rows = run_single(condition_key, seed)
            rows.append(row)
            robustness_rows.extend(rob_rows)
            pd.DataFrame(rows).to_csv(results_path, index=False)
            pd.DataFrame(robustness_rows).to_csv(robustness_path, index=False)

    return pd.DataFrame(rows), pd.DataFrame(robustness_rows)


def _cv(values: List[float]) -> float:
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return std / mean if abs(mean) > 1e-9 else float("nan")


def aggregate_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for condition_key, sub in results_df.groupby("condition"):
        spec = POPULATION_CONDITIONS[condition_key]
        lgs = sub["learning_gain"].astype(float).tolist()
        lg_mean, lg_lo, lg_hi = confidence_interval(lgs)
        lg_std = float(np.std(lgs, ddof=1)) if len(lgs) > 1 else 0.0
        lg_range = float(max(lgs) - min(lgs)) if lgs else 0.0
        best_idx = sub["learning_gain"].astype(float).idxmax()
        worst_idx = sub["learning_gain"].astype(float).idxmin()

        row: Dict[str, Any] = {
            "condition": condition_key,
            "condition_label": spec["label"],
            "population_mode": spec["mode"],
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

    for condition_key, sub in results_df.groupby("condition"):
        spec = POPULATION_CONDITIONS[condition_key]
        entropies = sub["action_entropy"].astype(float).tolist()
        actions_used = sub["actions_used_above_1_percent"].astype(int).tolist()
        row: Dict[str, Any] = {
            "condition": condition_key,
            "condition_label": spec["label"],
            "population_mode": spec["mode"],
            "shannon_entropy_mean": round(float(np.mean(entropies)), 4),
            "shannon_entropy_std": round(float(np.std(entropies, ddof=1)) if len(entropies) > 1 else 0.0, 4),
            "actions_used_above_1_percent_mean": round(float(np.mean(actions_used)), 4),
            "actions_used_above_1_percent_std": round(
                float(np.std(actions_used, ddof=1)) if len(actions_used) > 1 else 0.0, 4
            ),
            "dominance_any_seed": bool(sub["dominance_detected"].any()),
        }
        for action_name in action_names:
            col = f"freq_{action_name}"
            if col in sub.columns:
                row[f"{action_name}_mean_freq"] = round(float(sub[col].mean()), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_robustness(robustness_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize same-vs-mixed eval gaps per train condition."""
    rows: List[Dict[str, Any]] = []
    for train_key, sub in robustness_df.groupby("train_condition"):
        same = sub[sub["eval_population_mode"] == POPULATION_CONDITIONS[train_key]["mode"]]
        mixed = sub[sub["eval_population_mode"] == "mixed"]
        if same.empty or mixed.empty:
            continue
        merged = same.merge(
            mixed,
            on=["train_condition", "seed"],
            suffixes=("_same", "_mixed"),
        )
        lg_same = merged["learning_gain_same"].astype(float).tolist()
        lg_mixed = merged["learning_gain_mixed"].astype(float).tolist()
        gaps = [m - s for s, m in zip(lg_same, lg_mixed)]
        rows.append({
            "train_condition": train_key,
            "train_condition_label": POPULATION_CONDITIONS[train_key]["label"],
            "learning_gain_same_mean": round(float(np.mean(lg_same)), 4),
            "learning_gain_mixed_mean": round(float(np.mean(lg_mixed)), 4),
            "mixed_minus_same_mean": round(float(np.mean(gaps)), 4),
            "mixed_minus_same_std": round(
                float(np.std(gaps, ddof=1)) if len(gaps) > 1 else 0.0, 4
            ),
            "overfitting_signal": bool(np.mean(gaps) < -0.02),
        })
    return pd.DataFrame(rows)


def build_ranking(summary_df: pd.DataFrame, diversity_df: pd.DataFrame) -> pd.DataFrame:
    merged = summary_df.merge(
        diversity_df[[
            "condition",
            "shannon_entropy_mean",
            "actions_used_above_1_percent_mean",
        ]],
        on="condition",
    )
    score = pd.Series(0.0, index=merged.index)
    rank_specs = {
        "learning_gain_mean": False,
        "learning_gain_std": True,
        "coefficient_of_variation": True,
        "success_rate_mean": False,
        "adaptation_accuracy_mean": False,
        "mean_episode_reward_mean": False,
        "shannon_entropy_mean": False,
        "actions_used_above_1_percent_mean": False,
    }
    for col, ascending in rank_specs.items():
        if col in merged.columns:
            score += merged[col].rank(ascending=ascending, method="average")
    merged["composite_rank_score"] = score
    merged["rank"] = score.rank(method="min").astype(int)

    ranking = merged[[
        "rank",
        "condition",
        "condition_label",
        "population_mode",
        "learning_gain_mean",
        "learning_gain_std",
        "coefficient_of_variation",
        "seed_range",
        "success_rate_mean",
        "adaptation_accuracy_mean",
        "mean_episode_reward_mean",
        "shannon_entropy_mean",
        "actions_used_above_1_percent_mean",
        "best_seed",
        "worst_seed",
    ]].sort_values("rank")
    return ranking


def _plot_comparison(summary_df: pd.DataFrame, robustness_df: pd.DataFrame) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    order = ["A_mixed", "B_single_canonical", "C_low_variance"]
    summary_df = summary_df.set_index("condition").reindex(order).reset_index()
    labels = [row["condition_label"] for _, row in summary_df.iterrows()]
    colors = ["#457B9D", "#E9C46A", "#2A9D8F"]
    xs = np.arange(len(labels))

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
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
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Student Profile Ablation (DQN, G4 {G4_GAIN['ratio_label']}, "
        f"{TRAIN_TS // 1000}k steps, checkpoint selection)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "student_profile_comparison.png", dpi=150)
    plt.close(fig)

    if robustness_df.empty:
        return

    rob_summary = aggregate_robustness(robustness_df)
    if rob_summary.empty:
        return

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    rob_summary = rob_summary.set_index("train_condition").reindex(order).reset_index()
    width = 0.35
    ax2.bar(
        xs - width / 2,
        rob_summary["learning_gain_same_mean"],
        width,
        label="Eval: same population",
        color="#457B9D",
    )
    ax2.bar(
        xs + width / 2,
        rob_summary["learning_gain_mixed_mean"],
        width,
        label="Eval: mixed population",
        color="#E76F51",
    )
    ax2.set_xticks(xs)
    ax2.set_xticklabels([POPULATION_CONDITIONS[k]["label"] for k in order], fontsize=8)
    ax2.set_ylabel("Learning Gain")
    ax2.set_title("Robustness: train population vs eval population")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(OUT_DIR / "student_profile_robustness.png", dpi=150)
    plt.close(fig2)


def _print_winner(
    summary_df: pd.DataFrame,
    diversity_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    robustness_df: pd.DataFrame,
) -> None:
    mixed = summary_df[summary_df["condition"] == "A_mixed"].iloc[0]
    single = summary_df[summary_df["condition"] == "B_single_canonical"].iloc[0]
    lowvar = summary_df[summary_df["condition"] == "C_low_variance"].iloc[0]
    winner = ranking_df.iloc[0]

    most_stable = summary_df.loc[summary_df["learning_gain_std"].idxmin()]
    best_perf = summary_df.loc[summary_df["learning_gain_mean"].idxmax()]
    best_diversity = diversity_df.loc[diversity_df["shannon_entropy_mean"].idxmax()]

    mixed_std = float(mixed["learning_gain_std"])
    single_std = float(single["learning_gain_std"])
    lowvar_std = float(lowvar["learning_gain_std"])
    mixed_lg = float(mixed["learning_gain_mean"])
    single_lg = float(single["learning_gain_mean"])
    lowvar_lg = float(lowvar["learning_gain_mean"])

    single_more_stable = single_std < mixed_std
    lowvar_more_stable = lowvar_std < mixed_std
    profiles_reduce_std = single_more_stable or lowvar_more_stable

    rob_summary = aggregate_robustness(robustness_df)
    single_rob = rob_summary[rob_summary["train_condition"] == "B_single_canonical"]
    single_overfits = (
        not single_rob.empty and bool(single_rob.iloc[0]["overfitting_signal"])
    )

    print("\n" + "=" * 72)
    print("WINNER POPULATION CONFIGURATION")
    print("=" * 72)
    print(f"\nOverall rank #1: {winner['condition_label']} (mode={winner['population_mode']})")
    print(f"  learning_gain_mean={winner['learning_gain_mean']:.4f} "
          f"(std={winner['learning_gain_std']:.4f}, CV={winner['coefficient_of_variation']:.4f})")
    print(f"  seed_range={winner['seed_range']:.4f} "
          f"(best seed {winner['best_seed']}, worst seed {winner['worst_seed']})")
    print(f"  success_rate_mean={winner['success_rate_mean']:.4f}")
    print(f"  adaptation_accuracy_mean={winner['adaptation_accuracy_mean']:.4f}")
    print(f"  shannon_entropy={winner['shannon_entropy_mean']:.4f}, "
          f"actions_used_above_1%={winner['actions_used_above_1_percent_mean']:.1f}")

    print(f"\nBest mean learning gain: {best_perf['condition_label']} "
          f"(mean={best_perf['learning_gain_mean']:.4f})")
    print(f"Most stable (lowest LG std): {most_stable['condition_label']} "
          f"(std={most_stable['learning_gain_std']:.4f}, "
          f"CV={most_stable['coefficient_of_variation']:.4f})")
    print(f"Highest action diversity: {best_diversity['condition_label']} "
          f"(entropy={best_diversity['shannon_entropy_mean']:.4f})")

    print("\n--- Findings ---")
    print("\n1. Are student profiles a major source of DQN instability?")
    if profiles_reduce_std and (single_std < mixed_std * 0.7 or lowvar_std < mixed_std * 0.7):
        print(
            f"   YES. Heterogeneous profiles materially increase seed variance "
            f"(mixed std={mixed_std:.4f}, single={single_std:.4f}, low-variance={lowvar_std:.4f}). "
            "Different students reward different strategies, amplifying run-to-run spread."
        )
    elif profiles_reduce_std:
        print(
            f"   PARTIALLY. Restricting student diversity lowers seed variance modestly "
            f"(mixed std={mixed_std:.4f}, single={single_std:.4f}, low-variance={lowvar_std:.4f}). "
            "Profiles contribute to instability but are not the sole driver."
        )
    else:
        print(
            f"   NO. Population heterogeneity did not reduce seed variance "
            f"(mixed std={mixed_std:.4f}, single={single_std:.4f}, low-variance={lowvar_std:.4f}). "
            "Instability likely stems from other sources (noise, exploration, Q-learning)."
        )

    print("\n2. Does a single canonical student improve stability?")
    if single_more_stable:
        print(
            f"   YES. Single canonical training cuts learning-gain std "
            f"({single_std:.4f} vs mixed {mixed_std:.4f})."
        )
    else:
        print(
            f"   NO. Fixed-profile training did not lower variance "
            f"(single std={single_std:.4f} vs mixed {mixed_std:.4f})."
        )

    print("\n3. Does a single canonical student reduce generalization?")
    if not single_rob.empty:
        gap = float(single_rob.iloc[0]["mixed_minus_same_mean"])
        same_lg = float(single_rob.iloc[0]["learning_gain_same_mean"])
        mixed_eval_lg = float(single_rob.iloc[0]["learning_gain_mixed_mean"])
        if single_overfits:
            print(
                f"   YES. Single-student policies drop on mixed eval "
                f"(same={same_lg:.4f}, mixed={mixed_eval_lg:.4f}, gap={gap:.4f}). "
                "Lower training variance may reflect overfitting to one profile."
            )
        elif gap < -0.01:
            print(
                f"   SLIGHTLY. Mixed eval is lower but gap is small "
                f"(same={same_lg:.4f}, mixed={mixed_eval_lg:.4f}, gap={gap:.4f})."
            )
        else:
            print(
                f"   NO. Single-student policies transfer to mixed population "
                f"(same={same_lg:.4f}, mixed={mixed_eval_lg:.4f}, gap={gap:.4f})."
            )
    else:
        print("   (robustness data unavailable)")

    print("\n4. Best population design for the thesis:")
    print(
        f"   {winner['condition_label']} - balances mean LG ({winner['learning_gain_mean']:.4f}), "
        f"seed stability (std={winner['learning_gain_std']:.4f}), and action diversity "
        f"(entropy={winner['shannon_entropy_mean']:.4f})."
    )
    if mixed_std <= min(single_std, lowvar_std):
        print(
            "   Mixed students are NOT the primary instability source in this study; "
            "heterogeneous profiles match or beat restricted populations on stability "
            "while supporting realistic tutoring diversity."
        )
    elif lowvar_std < mixed_std and lowvar_lg >= mixed_lg * 0.95:
        print(
            "   Mixed students are HURTING stability; low-variance sampling preserves "
            "heterogeneity while reducing strategy-conflict variance - a practical thesis compromise."
        )
    elif single_std < mixed_std:
        print(
            "   Mixed students are HURTING stability; a canonical profile stabilizes training "
            "but verify mixed-population eval before claiming generalization."
        )
    else:
        print(
            "   Population design has mixed effects on stability; prioritize the winner's "
            "mean performance and robustness gap when choosing a thesis configuration."
        )
    print("=" * 72)


def analyze(
    results_df: Optional[pd.DataFrame] = None,
    robustness_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    results_path = OUT_DIR / "student_profile_results.csv"
    robustness_path = OUT_DIR / "robustness_results.csv"
    if results_df is None:
        if not results_path.exists():
            raise FileNotFoundError(f"No results at {results_path}; run --phase run first.")
        results_df = pd.read_csv(results_path)
    if robustness_df is None:
        if robustness_path.exists():
            robustness_df = pd.read_csv(robustness_path)
        else:
            robustness_df = pd.DataFrame()

    summary_df = aggregate_summary(results_df)
    diversity_df = aggregate_action_diversity(results_df)
    ranking_df = build_ranking(summary_df, diversity_df)

    summary_df.to_csv(OUT_DIR / "student_profile_summary.csv", index=False)
    diversity_df.to_csv(OUT_DIR / "action_diversity_summary.csv", index=False)
    ranking_df.to_csv(OUT_DIR / "student_profile_ranking.csv", index=False)
    if not robustness_df.empty:
        robustness_df.to_csv(robustness_path, index=False)

    _plot_comparison(summary_df, robustness_df)
    _print_winner(summary_df, diversity_df, ranking_df, robustness_df)

    report = {
        "study": "student_profile_ablation",
        "canonical_params": CANONICAL_PARAMS,
        "low_variance_frac": LOW_VARIANCE_FRAC,
        "gain_ratio": G4_GAIN["ratio_label"],
        "baseline_hp": BASELINE_HP,
        "population_conditions": POPULATION_CONDITIONS,
        "seeds": SEEDS,
        "summary": summary_df.to_dict("records"),
        "action_diversity": diversity_df.to_dict("records"),
        "robustness_summary": aggregate_robustness(robustness_df).to_dict("records"),
        "ranking": ranking_df.to_dict("records"),
    }
    with open(OUT_DIR / "student_profile_ablation_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    return report


def main() -> None:
    global TRAIN_TS

    parser = argparse.ArgumentParser(
        description="Student profile ablation for DQN stability"
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
        print("=== Student Profile Ablation (run) ===")
        TRAIN_TS = train_ts
        run_study(seeds, args.resume)

    if args.phase in ("analyze", "all"):
        print("\n=== Student Profile Ablation (analyze) ===")
        analyze()


if __name__ == "__main__":
    main()
