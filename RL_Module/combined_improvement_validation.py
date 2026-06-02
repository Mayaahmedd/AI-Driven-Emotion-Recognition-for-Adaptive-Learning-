"""
Combined Improvement Validation Study
======================================
Tests whether Balanced-2 rewards, Nearly Equal Gains, and Single Canonical Student
combine additively or interact when applied together (Configurations A-E).

Usage (from repo root):
  export PYTHONPATH="$(pwd)"
  python -m RL_Module.combined_improvement_validation --phase all
  python -m RL_Module.combined_improvement_validation --phase run --resume
  python -m RL_Module.combined_improvement_validation --phase analyze
  python -m RL_Module.combined_improvement_validation --phase all --quick
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
import RL_Module.environment.student_model as student_model
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

OUT_DIR = _HERE / "figures" / "combined_improvement_validation"
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

THESIS_REWARD = {"wk": 0.50, "we": 0.20, "wf": 0.15, "wb": 0.10, "wc": 0.05}
BALANCED_2_REWARD = {"wk": 0.35, "we": 0.20, "wf": 0.15, "wb": 0.15, "wc": 0.15}

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

METRICS = [
    "learning_gain",
    "final_knowledge",
    "success_rate",
    "adaptation_accuracy",
    "mean_episode_reward",
    "dropout_rate",
]

ACTION_NAMES = list(ID_TO_ACTION.values())

COMBINED_CONFIGS: Dict[str, Dict[str, Any]] = {
    "A_thesis_baseline": {
        "label": "A: Thesis Baseline",
        "reward": THESIS_REWARD,
        "action_gains": CURRENT_GAINS,
        "population_mode": "mixed",
    },
    "B_reward_only": {
        "label": "B: Reward Only",
        "reward": BALANCED_2_REWARD,
        "action_gains": CURRENT_GAINS,
        "population_mode": "mixed",
    },
    "C_gains_only": {
        "label": "C: Gains Only",
        "reward": THESIS_REWARD,
        "action_gains": NEARLY_EQUAL_GAINS,
        "population_mode": "mixed",
    },
    "D_reward_and_gains": {
        "label": "D: Reward + Gains",
        "reward": BALANCED_2_REWARD,
        "action_gains": NEARLY_EQUAL_GAINS,
        "population_mode": "mixed",
    },
    "E_full_stability": {
        "label": "E: Full Stability Configuration",
        "reward": BALANCED_2_REWARD,
        "action_gains": NEARLY_EQUAL_GAINS,
        "population_mode": "single_canonical",
    },
}


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


def _population_factory(mode: str):
    def _generate(n: int = cfg.POPULATION_SIZE, seed: Optional[int] = None) -> List[SyntheticStudent]:
        if mode == "mixed":
            return _original_generate_population(n=n, seed=seed)
        if mode == "single_canonical":
            return _make_canonical_population(n)
        raise ValueError(f"Unknown population mode: {mode!r}")

    return _generate


@contextmanager
def _population_patch(mode: str) -> Iterator[None]:
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


def _patch_action_gains(gains_by_name: Dict[str, float]) -> Dict[int, float]:
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
    return float(sum(sorted(freq.values(), reverse=True)[:k]))


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


def run_single(
    configuration: str,
    seed: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    spec = COMBINED_CONFIGS[configuration]
    train_mode = spec["population_mode"]
    gain_snap = _patch_gain()
    reward_snap = _patch_reward(spec["reward"])
    action_snap = _patch_action_gains(spec["action_gains"])
    tag = f"combined_{configuration}_s{seed}"
    t0 = time.time()
    generalization_rows: List[Dict[str, Any]] = []

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
        metrics = evaluate_dqn(agent, seed, train_mode, EVAL_EPS)
        selection = f"checkpoint@{best_path}"

        if configuration == "E_full_stability":
            for eval_mode, eval_label in [
                ("single_canonical", "Single Canonical Student"),
                ("mixed", "Mixed Population"),
            ]:
                eval_metrics = evaluate_dqn(agent, seed, eval_mode, EVAL_EPS)
                generalization_rows.append({
                    "configuration": configuration,
                    "configuration_label": spec["label"],
                    "seed": seed,
                    "train_population": train_mode,
                    "eval_population": eval_mode,
                    "eval_population_label": eval_label,
                    "checkpoint": best_path,
                    **{m: eval_metrics[m] for m in METRICS},
                })
    finally:
        _restore_action_gains(action_snap)
        _restore_reward(reward_snap)
        _restore_gain(gain_snap)

    elapsed = time.time() - t0
    row: Dict[str, Any] = {
        "configuration": configuration,
        "configuration_label": spec["label"],
        "population_mode": train_mode,
        "seed": seed,
        "train_timesteps": TRAIN_TS,
        "eval_episodes": EVAL_EPS,
        "use_checkpoints": True,
        "selection": selection,
        "val_learning_gain": best_val,
        "gain_ratio": G4_GAIN["ratio_label"],
        "elapsed_seconds": round(elapsed, 1),
        **spec["reward"],
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
        f"  {configuration} seed={seed}: "
        f"lg={row['learning_gain']:.3f} succ={row['success_rate']:.3f} "
        f"adapt={row['adaptation_accuracy']:.3f} entropy={row['action_entropy']:.3f} "
        f"dom={row['dominant_action_pct']:.2f} ({elapsed:.0f}s)"
    )
    return row, generalization_rows


def _load_resume(path: Path, resume: bool) -> List[Dict[str, Any]]:
    if resume and path.exists():
        return pd.read_csv(path).to_dict("records")
    return []


def run_study(seeds: List[int], resume: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    results_path = OUT_DIR / "combined_results.csv"
    gen_path = OUT_DIR / "generalization_results.csv"
    rows = _load_resume(results_path, resume)
    gen_rows = _load_resume(gen_path, resume)
    done = {(r["configuration"], int(r["seed"])) for r in rows}

    for configuration in COMBINED_CONFIGS:
        print(f"\n--- {COMBINED_CONFIGS[configuration]['label']} ---")
        for seed in seeds:
            if (configuration, seed) in done:
                continue
            row, gen = run_single(configuration, seed)
            rows.append(row)
            gen_rows.extend(gen)
            pd.DataFrame(rows).to_csv(results_path, index=False)
            if gen_rows:
                pd.DataFrame(gen_rows).to_csv(gen_path, index=False)

    return pd.DataFrame(rows), pd.DataFrame(gen_rows) if gen_rows else pd.DataFrame()


def _cv(values: List[float]) -> float:
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return std / mean if abs(mean) > 1e-9 else float("nan")


def aggregate_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for configuration, sub in results_df.groupby("configuration"):
        spec = COMBINED_CONFIGS[configuration]
        lgs = sub["learning_gain"].astype(float).tolist()
        lg_mean, lg_lo, lg_hi = confidence_interval(lgs)
        lg_std = float(np.std(lgs, ddof=1)) if len(lgs) > 1 else 0.0
        lg_range = float(max(lgs) - min(lgs)) if lgs else 0.0
        best_idx = sub["learning_gain"].astype(float).idxmax()
        worst_idx = sub["learning_gain"].astype(float).idxmin()

        row: Dict[str, Any] = {
            "configuration": configuration,
            "configuration_label": spec["label"],
            "population_mode": spec["population_mode"],
            "reward_profile": "balanced_2" if spec["reward"] == BALANCED_2_REWARD else "thesis",
            "gain_profile": "nearly_equal" if spec["action_gains"] == NEARLY_EQUAL_GAINS else "current",
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
            "shannon_entropy_mean": round(float(sub["action_entropy"].mean()), 4),
            "shannon_entropy_std": round(
                float(sub["action_entropy"].std(ddof=1)) if len(sub) > 1 else 0.0, 4
            ),
            "dominant_action_pct_mean": round(float(sub["dominant_action_pct"].mean()), 4),
            "dominant_action_pct_std": round(
                float(sub["dominant_action_pct"].std(ddof=1)) if len(sub) > 1 else 0.0, 4
            ),
            "top2_actions_pct_mean": round(float(sub["top2_actions_pct"].mean()), 4),
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
        for action_name in ACTION_NAMES:
            col = f"freq_{action_name}"
            if col in sub.columns:
                row[f"freq_{action_name}_mean"] = round(float(sub[col].mean()), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_generalization(gen_df: pd.DataFrame) -> pd.DataFrame:
    if gen_df.empty:
        return gen_df

    rows: List[Dict[str, Any]] = []
    for seed, sub in gen_df.groupby("seed"):
        canonical = sub[sub["eval_population"] == "single_canonical"]
        mixed = sub[sub["eval_population"] == "mixed"]
        if canonical.empty or mixed.empty:
            continue
        lg_canon = float(canonical.iloc[0]["learning_gain"])
        lg_mixed = float(mixed.iloc[0]["learning_gain"])
        rows.append({
            "configuration": "E_full_stability",
            "seed": int(seed),
            "learning_gain_canonical": round(lg_canon, 4),
            "learning_gain_mixed": round(lg_mixed, 4),
            "generalization_drop": round(lg_canon - lg_mixed, 4),
            "relative_drop_pct": round(
                100.0 * (lg_canon - lg_mixed) / lg_canon if lg_canon > 1e-9 else float("nan"), 2
            ),
        })

    per_seed = pd.DataFrame(rows)
    if per_seed.empty:
        return per_seed

    summary_row = {
        "configuration": "E_full_stability",
        "seed": "mean",
        "learning_gain_canonical": round(float(per_seed["learning_gain_canonical"].mean()), 4),
        "learning_gain_mixed": round(float(per_seed["learning_gain_mixed"].mean()), 4),
        "generalization_drop": round(float(per_seed["generalization_drop"].mean()), 4),
        "relative_drop_pct": round(float(per_seed["relative_drop_pct"].mean()), 2),
    }
    return pd.concat([per_seed, pd.DataFrame([summary_row])], ignore_index=True)


def build_ranking(summary_df: pd.DataFrame) -> pd.DataFrame:
    merged = summary_df.copy()
    merged = merged.rename(columns={
        "mean_episode_reward_mean": "reward_mean",
        "shannon_entropy_mean": "action_entropy",
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
        "dominant_action_pct_mean": True,
        "seed_range": True,
    }
    for col, ascending in rank_specs.items():
        if col in merged.columns:
            score += merged[col].rank(ascending=ascending, method="average")
    merged["composite_score"] = score
    merged["rank"] = score.rank(method="min").astype(int)

    cols = [
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
        "rank",
    ]
    return merged[[c for c in cols if c in merged.columns]].sort_values("rank")


def _expected_additive_lg(summary_df: pd.DataFrame) -> Dict[str, float]:
    """Estimate additive LG from single-factor deltas vs baseline."""
    baseline = summary_df[summary_df["configuration"] == "A_thesis_baseline"]
    reward_only = summary_df[summary_df["configuration"] == "B_reward_only"]
    gains_only = summary_df[summary_df["configuration"] == "C_gains_only"]
    combined = summary_df[summary_df["configuration"] == "D_reward_and_gains"]
    if baseline.empty or reward_only.empty or gains_only.empty or combined.empty:
        return {}

    lg_a = float(baseline.iloc[0]["learning_gain_mean"])
    delta_reward = float(reward_only.iloc[0]["learning_gain_mean"]) - lg_a
    delta_gains = float(gains_only.iloc[0]["learning_gain_mean"]) - lg_a
    expected_d = lg_a + delta_reward + delta_gains
    actual_d = float(combined.iloc[0]["learning_gain_mean"])
    return {
        "baseline_lg": lg_a,
        "delta_reward_only": delta_reward,
        "delta_gains_only": delta_gains,
        "expected_additive_lg_D": expected_d,
        "actual_lg_D": actual_d,
        "interaction_residual_D": actual_d - expected_d,
    }


def _plot_comparison(summary_df: pd.DataFrame) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    summary_df = summary_df.sort_values("configuration")
    labels = [
        row["configuration_label"].replace(": ", "\n").replace(" Configuration", "")
        for _, row in summary_df.iterrows()
    ]
    colors = ["#6C757D", "#457B9D", "#2A9D8F", "#E9C46A", "#E76F51"]
    xs = np.arange(len(labels))

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    panels = [
        ("learning_gain_mean", "learning_gain_std", "Learning Gain"),
        ("coefficient_of_variation", None, "Coefficient of Variation"),
        ("success_rate_mean", "success_rate_std", "Success Rate"),
        ("shannon_entropy_mean", "shannon_entropy_std", "Action Entropy (Shannon)"),
    ]
    for ax, (mean_col, std_col, title) in zip(axes.flat, panels):
        means = summary_df[mean_col].tolist()
        stds = (
            summary_df[std_col].tolist()
            if std_col and std_col in summary_df.columns
            else [0.0] * len(means)
        )
        if std_col:
            ax.bar(xs, means, yerr=stds, capsize=5, color=colors[: len(xs)], edgecolor="#64748B")
        else:
            ax.bar(xs, means, color=colors[: len(xs)], edgecolor="#64748B")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Combined Improvement Validation (DQN, {TRAIN_TS // 1000}k steps, checkpoint selection)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "combined_comparison.png", dpi=150)
    plt.close(fig)


def _plot_action_distribution(summary_df: pd.DataFrame) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    summary_df = summary_df.sort_values("configuration")
    freq_cols = [f"freq_{a}_mean" for a in ACTION_NAMES]
    if not all(c in summary_df.columns for c in freq_cols):
        return

    labels = [row["configuration"] for _, row in summary_df.iterrows()]
    data = summary_df[freq_cols].values
    x = np.arange(len(ACTION_NAMES))
    width = 0.15
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (label, row_vals) in enumerate(zip(labels, data)):
        ax.bar(x + (i - 2) * width, row_vals, width=width, label=label, edgecolor="#64748B")
    ax.set_xticks(x)
    ax.set_xticklabels([a.replace("_", "\n") for a in ACTION_NAMES], fontsize=8)
    ax.set_ylabel("Mean action frequency")
    ax.set_title("Action Distribution by Configuration")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "action_distribution_comparison.png", dpi=150)
    plt.close(fig)


def _interaction_analysis(summary_df: pd.DataFrame, additive: Dict[str, float]) -> str:
    lines: List[str] = []
    a = summary_df[summary_df["configuration"] == "A_thesis_baseline"].iloc[0]
    b = summary_df[summary_df["configuration"] == "B_reward_only"].iloc[0]
    c = summary_df[summary_df["configuration"] == "C_gains_only"].iloc[0]
    d = summary_df[summary_df["configuration"] == "D_reward_and_gains"].iloc[0]
    e = summary_df[summary_df["configuration"] == "E_full_stability"].iloc[0]

    lines.append("Reward design (Balanced-2 vs thesis weights):")
    lines.append(
        f"  B vs A: LG {b['learning_gain_mean']:.4f} vs {a['learning_gain_mean']:.4f} "
        f"(delta={b['learning_gain_mean'] - a['learning_gain_mean']:+.4f}), "
        f"CV {b['coefficient_of_variation']:.4f} vs {a['coefficient_of_variation']:.4f}."
    )
    lines.append(
        "  Raising wb/wc relative to wk shifts credit from knowledge deltas alone toward "
        "affect regulation; this can stabilize training when frustration/confusion penalties "
        "were under-weighted (thesis wc=0.05)."
    )

    lines.append("\nAction gains (Nearly Equal vs current hierarchy):")
    lines.append(
        f"  C vs A: LG {c['learning_gain_mean']:.4f} vs {a['learning_gain_mean']:.4f} "
        f"(delta={c['learning_gain_mean'] - a['learning_gain_mean']:+.4f}), "
        f"dominant action {c['dominant_action_pct_mean']:.2%} vs {a['dominant_action_pct_mean']:.2%}."
    )
    lines.append(
        "  Flattening base_gain reduces simulator bias toward harder_problem/scaffold, "
        "encouraging broader pedagogical exploration in the policy."
    )

    if additive:
        res = additive["interaction_residual_D"]
        lines.append("\nCombination (D = Reward + Gains):")
        lines.append(
            f"  Expected additive LG ~ {additive['expected_additive_lg_D']:.4f} "
            f"(baseline + reward delta + gains delta); actual D = {additive['actual_lg_D']:.4f}; "
            f"residual = {res:+.4f}."
        )
        if abs(res) < 0.01:
            lines.append("  Effects are approximately additive (little interaction).")
        elif res > 0:
            lines.append("  Positive synergy: combined config exceeds independent sum.")
        else:
            lines.append("  Negative interaction: combined config underperforms additive expectation.")

    lines.append("\nStudent population (E = full stack on canonical student):")
    lines.append(
        f"  E vs D (same reward/gains): LG {e['learning_gain_mean']:.4f} vs {d['learning_gain_mean']:.4f}, "
        f"CV {e['coefficient_of_variation']:.4f} vs {d['coefficient_of_variation']:.4f}."
    )
    lines.append(
        "  Training on a single canonical profile removes cross-student variance during "
        "optimization, often improving seed stability at the cost of heterogeneous generalization."
    )
    return "\n".join(lines)


def _print_winner(
    summary_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    additive: Dict[str, float],
    gen_summary: pd.DataFrame,
) -> None:
    winner = ranking_df.iloc[0]
    interaction_text = _interaction_analysis(summary_df, additive)

    print("\n" + "=" * 72)
    print("COMBINED IMPROVEMENT VALIDATION STUDY - RESULTS")
    print("=" * 72)

    print("\n--- Per-Configuration Summary ---")
    for _, row in summary_df.sort_values("configuration").iterrows():
        print(
            f"\n{row['configuration_label']} ({row['configuration']})"
        )
        print(
            f"  learning_gain={row['learning_gain_mean']:.4f} "
            f"(std={row['learning_gain_std']:.4f}, CV={row['coefficient_of_variation']:.4f}, "
            f"range={row['seed_range']:.4f})"
        )
        print(
            f"  success={row['success_rate_mean']:.4f}, "
            f"adapt={row['adaptation_accuracy_mean']:.4f}, "
            f"entropy={row['shannon_entropy_mean']:.4f}, "
            f"dominant={row['dominant_action_pct_mean']:.4f}"
        )

    print("\n--- Research Questions ---")
    a = summary_df[summary_df["configuration"] == "A_thesis_baseline"].iloc[0]
    d = summary_df[summary_df["configuration"] == "D_reward_and_gains"].iloc[0]
    print(
        f"\n1. Combining Balanced-2 + Nearly Equal Gains (D vs A): "
        f"LG {d['learning_gain_mean']:.4f} vs {a['learning_gain_mean']:.4f} "
        f"({'improves' if d['learning_gain_mean'] > a['learning_gain_mean'] else 'does not improve'})"
    )
    print(
        f"\n2. Stability (D vs A): std {d['learning_gain_std']:.4f} vs {a['learning_gain_std']:.4f}, "
        f"CV {d['coefficient_of_variation']:.4f} vs {a['coefficient_of_variation']:.4f}"
    )
    if additive:
        print(
            f"\n3. Additivity: expected D~{additive['expected_additive_lg_D']:.4f}, "
            f"actual={additive['actual_lg_D']:.4f}, residual={additive['interaction_residual_D']:+.4f}"
        )
    if not gen_summary.empty:
        mean_row = gen_summary[gen_summary["seed"].astype(str) == "mean"]
        if not mean_row.empty:
            mr = mean_row.iloc[0]
            print(
                f"\n4. Canonical overfit (E): train-eval canonical LG={mr['learning_gain_canonical']:.4f}, "
                f"mixed LG={mr['learning_gain_mixed']:.4f}, "
                f"drop={mr['generalization_drop']:.4f} ({mr['relative_drop_pct']:.1f}% relative)"
            )

    print("\n--- Interaction Effects ---")
    print(interaction_text)

    print("\n" + "=" * 72)
    print("WINNER COMBINED CONFIGURATION")
    print("=" * 72)
    wrow = summary_df[summary_df["configuration"] == winner["configuration"]].iloc[0]
    print(f"\n  {winner['configuration']} ({winner['configuration_label']})")
    print(f"  Composite rank = {int(winner['rank'])} (lower is better)")
    print(f"  learning_gain_mean       = {wrow['learning_gain_mean']:.4f}")
    print(f"  learning_gain_std        = {wrow['learning_gain_std']:.4f}")
    print(f"  coefficient_of_variation = {wrow['coefficient_of_variation']:.4f}")
    print(f"  success_rate_mean        = {wrow['success_rate_mean']:.4f}")
    print(f"  adaptation_accuracy_mean = {wrow['adaptation_accuracy_mean']:.4f}")
    print(f"  shannon_entropy          = {wrow['shannon_entropy_mean']:.4f}")
    print(f"  dominant_action_pct      = {wrow['dominant_action_pct_mean']:.4f}")
    print("=" * 72)


def analyze(
    results_df: Optional[pd.DataFrame] = None,
    gen_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    results_path = OUT_DIR / "combined_results.csv"
    gen_path = OUT_DIR / "generalization_results.csv"

    if results_df is None:
        if not results_path.exists():
            raise FileNotFoundError(f"No results at {results_path}; run --phase run first.")
        results_df = pd.read_csv(results_path)
    if gen_df is None and gen_path.exists():
        gen_df = pd.read_csv(gen_path)
    if gen_df is None:
        gen_df = pd.DataFrame()

    summary_df = aggregate_summary(results_df)
    ranking_df = build_ranking(summary_df)
    additive = _expected_additive_lg(summary_df)
    gen_summary = aggregate_generalization(gen_df)

    results_df.to_csv(OUT_DIR / "combined_results.csv", index=False)
    summary_df.to_csv(OUT_DIR / "combined_summary.csv", index=False)
    ranking_df.to_csv(OUT_DIR / "combined_ranking.csv", index=False)
    if not gen_df.empty:
        gen_df.to_csv(OUT_DIR / "generalization_results.csv", index=False)
    if not gen_summary.empty:
        gen_summary.to_csv(OUT_DIR / "generalization_results.csv", index=False)

    _plot_comparison(summary_df)
    _plot_action_distribution(summary_df)
    _print_winner(summary_df, ranking_df, additive, gen_summary)

    winner_key = str(ranking_df.iloc[0]["configuration"])
    report = {
        "study": "combined_improvement_validation",
        "train_timesteps": TRAIN_TS,
        "seeds": SEEDS,
        "configurations": COMBINED_CONFIGS,
        "additive_analysis": additive,
        "summary": summary_df.to_dict("records"),
        "ranking": ranking_df.to_dict("records"),
        "generalization": gen_summary.to_dict("records") if not gen_summary.empty else [],
        "winner": winner_key,
        "interaction_notes": _interaction_analysis(summary_df, additive),
    }
    with open(OUT_DIR / "combined_improvement_validation_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    return report


def main() -> None:
    global TRAIN_TS

    parser = argparse.ArgumentParser(
        description="Combined improvement validation: reward x gains x population (A-E)"
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
        print("=== Combined Improvement Validation (run) ===")
        TRAIN_TS = train_ts
        run_study(seeds, args.resume)

    if args.phase in ("analyze", "all"):
        print("\n=== Combined Improvement Validation (analyze) ===")
        analyze()


if __name__ == "__main__":
    main()
