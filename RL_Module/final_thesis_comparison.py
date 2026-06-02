"""
Final Thesis Configuration Comparison
=====================================
Configuration A (stable): Balanced-2 + Nearly Equal Gains + Single Canonical Student
Configuration B (realistic): Balanced-2 + Nearly Equal Gains + Mixed Student Population

Usage (from repo root):
  export PYTHONPATH="$(pwd)"
  python -m RL_Module.final_thesis_comparison --phase all
  python -m RL_Module.final_thesis_comparison --phase run --resume
  python -m RL_Module.final_thesis_comparison --phase analyze
  python -m RL_Module.final_thesis_comparison --phase all --quick
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

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

OUT_DIR = _HERE / "final_thesis_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 7, 13, 21, 99, 314, 555, 777, 888, 999]
TRAIN_TS = 50_000
EVAL_EPS = 500
VAL_EPS = 100
CHECKPOINT_STEPS = [10_000, 20_000, 30_000, 40_000, 50_000]

G4_GAIN = {
    "GAIN_CORRECT_FACTOR": 0.1,
    "GAIN_INCORRECT_FACTOR": 1.0,
    "ratio_label": "10:1",
    "IRT_BETA": 3.0,
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

BALANCED_2_REWARD = {"wk": 0.35, "we": 0.20, "wf": 0.15, "wb": 0.15, "wc": 0.15}

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

STAT_METRICS = ["learning_gain", "success_rate", "adaptation_accuracy"]

ACTION_NAMES = list(ID_TO_ACTION.values())

THESIS_CONFIGS: Dict[str, Dict[str, Any]] = {
    "A_stable_canonical": {
        "label": "A: Most Stable Thesis System",
        "description": "Balanced-2 + Nearly Equal Gains + Single Canonical Student",
        "reward": BALANCED_2_REWARD,
        "action_gains": NEARLY_EQUAL_GAINS,
        "population_mode": "single_canonical",
        "legacy_combined_key": "E_full_stability",
    },
    "B_realistic_mixed": {
        "label": "B: Best Realistic Thesis System",
        "description": "Balanced-2 + Nearly Equal Gains + Mixed Student Population",
        "reward": BALANCED_2_REWARD,
        "action_gains": NEARLY_EQUAL_GAINS,
        "population_mode": "mixed",
        "legacy_combined_key": "D_reward_and_gains",
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


def _restore_reward(snapshot: Tuple[Dict[str, float], bool, Dict[str, float]]) -> None:
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


def _checkpoint_path_from_selection(selection: str) -> str:
    if selection.startswith("checkpoint@"):
        return selection.split("@", 1)[1]
    return selection


def _legacy_model_tag(configuration: str, seed: int) -> str:
    legacy = THESIS_CONFIGS[configuration]["legacy_combined_key"]
    return f"combined_{legacy}_s{seed}"


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
            "actions_used_above_1_percent": _actions_used_above_threshold(freqs, 0.01),
            "actions_used_above_5_percent": _actions_used_above_threshold(freqs, 0.05),
            "dominant_action_pct": _dominant_action_pct(freqs),
            "top2_actions_pct": _top_k_actions_pct(freqs, 2),
            "dominance_detected": freq_report["dominance_detected"],
        })

    return result


def run_single(configuration: str, seed: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    spec = THESIS_CONFIGS[configuration]
    train_mode = spec["population_mode"]
    gain_snap = _patch_gain()
    reward_snap = _patch_reward(spec["reward"])
    action_snap = _patch_action_gains(spec["action_gains"])
    tag = f"final_thesis_{configuration}_s{seed}"
    t0 = time.time()
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
        generalization_rows = _generalization_rows(agent, configuration, seed, best_path)
        selection = f"checkpoint@{best_path}"
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
        "irt_beta": G4_GAIN["IRT_BETA"],
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


def _generalization_rows(
    agent: DQNAgent,
    configuration: str,
    seed: int,
    checkpoint_path: str,
) -> List[Dict[str, Any]]:
    spec = THESIS_CONFIGS[configuration]
    train_mode = spec["population_mode"]
    rows: List[Dict[str, Any]] = []
    per_eval: Dict[str, Dict[str, Any]] = {}

    for eval_mode, eval_label in [
        ("single_canonical", "Single Canonical Student"),
        ("mixed", "Mixed Student Population"),
    ]:
        per_eval[eval_mode] = evaluate_dqn(
            agent, seed, eval_mode, EVAL_EPS, collect_actions=False
        )
        rows.append({
            "configuration": configuration,
            "configuration_label": spec["label"],
            "seed": seed,
            "train_population": train_mode,
            "eval_population": eval_mode,
            "eval_population_label": eval_label,
            "checkpoint": checkpoint_path,
            **{m: per_eval[eval_mode][m] for m in METRICS},
        })

    train_lg = per_eval[train_mode]["learning_gain"]
    mixed_lg = per_eval["mixed"]["learning_gain"]
    for row in rows:
        row["same_population_learning_gain"] = train_lg
        row["mixed_population_learning_gain"] = mixed_lg
        row["generalization_gap"] = round(train_lg - mixed_lg, 4)
    return rows


def run_generalization_only(
    configuration: str,
    seed: int,
    checkpoint_path: str,
) -> List[Dict[str, Any]]:
    spec = THESIS_CONFIGS[configuration]
    gain_snap = _patch_gain()
    reward_snap = _patch_reward(spec["reward"])
    action_snap = _patch_action_gains(spec["action_gains"])

    try:
        agent = DQNAgent()
        agent.load_checkpoint(checkpoint_path)
        return _generalization_rows(agent, configuration, seed, checkpoint_path)
    finally:
        _restore_action_gains(action_snap)
        _restore_reward(reward_snap)
        _restore_gain(gain_snap)


def _load_resume(path: Path, resume: bool) -> List[Dict[str, Any]]:
    if resume and path.exists():
        return pd.read_csv(path).to_dict("records")
    return []


def _import_legacy_combined(results_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    legacy_path = _HERE / "figures" / "combined_improvement_validation" / "combined_results.csv"
    if not legacy_path.exists():
        return results_rows

    legacy = pd.read_csv(legacy_path)
    done = {(r["configuration"], int(r["seed"])) for r in results_rows}

    for configuration, spec in THESIS_CONFIGS.items():
        legacy_key = spec["legacy_combined_key"]
        sub = legacy[legacy["configuration"] == legacy_key]
        for _, row in sub.iterrows():
            seed = int(row["seed"])
            if (configuration, seed) in done:
                continue
            imported = {
                "configuration": configuration,
                "configuration_label": spec["label"],
                "population_mode": spec["population_mode"],
                "seed": seed,
                "train_timesteps": int(row.get("train_timesteps", TRAIN_TS)),
                "eval_episodes": int(row.get("eval_episodes", EVAL_EPS)),
                "use_checkpoints": True,
                "selection": row["selection"],
                "val_learning_gain": float(row["val_learning_gain"]),
                "gain_ratio": row.get("gain_ratio", G4_GAIN["ratio_label"]),
                "irt_beta": G4_GAIN["IRT_BETA"],
                "elapsed_seconds": float(row.get("elapsed_seconds", 0.0)),
                "imported_from": str(legacy_path),
            }
            for key in ["wk", "we", "wf", "wb", "wc"]:
                if key in row.index:
                    imported[key] = float(row[key])
            for m in METRICS:
                imported[m] = float(row[m])
            for col in [
                "action_entropy",
                "actions_used_above_1_percent",
                "actions_used_above_5_percent",
                "dominant_action_pct",
                "top2_actions_pct",
                "dominance_detected",
            ]:
                if col in row.index:
                    imported[col] = row[col]
            for action_name in ACTION_NAMES:
                col = f"freq_{action_name}"
                if col in row.index:
                    imported[col] = float(row[col])
            results_rows.append(imported)
            done.add((configuration, seed))
            print(f"  imported {configuration} seed={seed} from combined study")

    return results_rows


def run_study(seeds: List[int], resume: bool, import_legacy: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    results_path = OUT_DIR / "final_comparison_results.csv"
    gen_path = OUT_DIR / "generalization_results.csv"
    rows = _load_resume(results_path, resume)
    gen_rows = _load_resume(gen_path, resume)

    if import_legacy:
        rows = _import_legacy_combined(rows)
        if rows:
            pd.DataFrame(rows).to_csv(results_path, index=False)

    done = {(r["configuration"], int(r["seed"])) for r in rows}
    gen_done = {
        (r["configuration"], int(r["seed"]), r["eval_population"])
        for r in gen_rows
    }

    for configuration in THESIS_CONFIGS:
        print(f"\n--- {THESIS_CONFIGS[configuration]['label']} ---")
        for seed in seeds:
            if (configuration, seed) not in done:
                row, gen = run_single(configuration, seed)
                rows.append(row)
                gen_rows = [r for r in gen_rows if not (
                    r["configuration"] == configuration and int(r["seed"]) == seed
                )]
                gen_rows.extend(gen)
                pd.DataFrame(rows).to_csv(results_path, index=False)
                pd.DataFrame(gen_rows).to_csv(gen_path, index=False)
                done.add((configuration, seed))
            else:
                result_row = next(
                    r for r in rows
                    if r["configuration"] == configuration and int(r["seed"]) == seed
                )
                ckpt = _checkpoint_path_from_selection(str(result_row["selection"]))
                if not Path(ckpt).exists():
                    legacy_ckpt = cfg.MODELS_DIR / _legacy_model_tag(configuration, seed)
                    for p in sorted(legacy_ckpt.glob("dqn_*_steps.zip")):
                        ckpt = str(p)
                    if "checkpoint@" in str(result_row["selection"]):
                        legacy_sel = str(result_row["selection"])
                        legacy_path = _checkpoint_path_from_selection(legacy_sel)
                        if Path(legacy_path).exists():
                            ckpt = legacy_path

                need_gen = any(
                    (configuration, seed, mode) not in gen_done
                    for mode in ("single_canonical", "mixed")
                )
                if need_gen and Path(ckpt).exists():
                    print(f"  {configuration} seed={seed}: generalization eval only")
                    new_gen = run_generalization_only(configuration, seed, ckpt)
                    gen_rows = [
                        r for r in gen_rows
                        if not (r["configuration"] == configuration and int(r["seed"]) == seed)
                    ]
                    gen_rows.extend(new_gen)
                    pd.DataFrame(gen_rows).to_csv(gen_path, index=False)
                    for mode in ("single_canonical", "mixed"):
                        gen_done.add((configuration, seed, mode))

    return pd.DataFrame(rows), pd.DataFrame(gen_rows) if gen_rows else pd.DataFrame()


def _cv(values: List[float]) -> float:
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return std / mean if abs(mean) > 1e-9 else float("nan")


def cohens_d(a: List[float], b: List[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = np.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2))
    return float((np.mean(a) - np.mean(b)) / pooled) if pooled > 1e-12 else 0.0


def aggregate_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for configuration, sub in results_df.groupby("configuration"):
        spec = THESIS_CONFIGS[configuration]
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
            "n_seeds": len(sub),
            "learning_gain_mean": round(lg_mean, 4),
            "learning_gain_std": round(lg_std, 4),
            "coefficient_of_variation": round(_cv(lgs), 4),
            "seed_range": round(lg_range, 4),
            "learning_gain_ci_95": f"[{lg_lo:.3f}, {lg_hi:.3f}]",
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
            mean, lo, hi = confidence_interval(vals)
            row[f"{metric}_mean"] = round(float(np.mean(vals)), 4)
            row[f"{metric}_std"] = round(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, 4)
            row[f"{metric}_ci_95"] = f"[{lo:.3f}, {hi:.3f}]"
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_generalization(gen_df: pd.DataFrame) -> pd.DataFrame:
    if gen_df.empty:
        return gen_df

    if "learning_gain_canonical" in gen_df.columns and "eval_population" not in gen_df.columns:
        per_seed = gen_df[gen_df["seed"].astype(str) != "mean"].copy()
        if per_seed.empty:
            return gen_df
        summary_rows: List[Dict[str, Any]] = []
        for configuration, sub in per_seed.groupby("configuration"):
            summary_rows.append({
                "configuration": configuration,
                "configuration_label": THESIS_CONFIGS[configuration]["label"],
                "seed": "mean",
                "train_population": THESIS_CONFIGS[configuration]["population_mode"],
                "learning_gain_canonical": round(
                    float(sub["learning_gain_canonical"].mean()), 4
                ),
                "learning_gain_mixed": round(float(sub["learning_gain_mixed"].mean()), 4),
                "same_population_learning_gain": round(
                    float(sub["same_population_learning_gain"].mean()), 4
                ),
                "mixed_population_learning_gain": round(
                    float(sub["mixed_population_learning_gain"].mean()), 4
                ),
                "generalization_gap": round(float(sub["generalization_gap"].mean()), 4),
                "canonical_transfer_gap": round(
                    float(sub["canonical_transfer_gap"].mean()), 4
                ),
                "overfitting_rate": round(float(sub["overfitting_signal"].mean()), 4),
            })
        return pd.concat([per_seed, pd.DataFrame(summary_rows)], ignore_index=True)

    rows: List[Dict[str, Any]] = []
    for (configuration, seed), sub in gen_df.groupby(["configuration", "seed"]):
        spec = THESIS_CONFIGS[configuration]
        train_mode = spec["population_mode"]
        canonical = sub[sub["eval_population"] == "single_canonical"]
        mixed = sub[sub["eval_population"] == "mixed"]
        if canonical.empty or mixed.empty:
            continue
        lg_canon = float(canonical.iloc[0]["learning_gain"])
        lg_mixed = float(mixed.iloc[0]["learning_gain"])
        train_lg = lg_canon if train_mode == "single_canonical" else lg_mixed
        gap = train_lg - lg_mixed if train_mode == "single_canonical" else lg_mixed - lg_canon
        rows.append({
            "configuration": configuration,
            "configuration_label": spec["label"],
            "seed": int(seed),
            "train_population": train_mode,
            "learning_gain_canonical": round(lg_canon, 4),
            "learning_gain_mixed": round(lg_mixed, 4),
            "same_population_learning_gain": round(train_lg, 4),
            "mixed_population_learning_gain": round(lg_mixed, 4),
            "generalization_gap": round(train_lg - lg_mixed, 4),
            "canonical_transfer_gap": round(lg_mixed - lg_canon, 4),
            "overfitting_signal": bool(train_mode == "single_canonical" and (train_lg - lg_mixed) > 0.02),
        })

    per_seed = pd.DataFrame(rows)
    if per_seed.empty:
        return per_seed

    summary_rows: List[Dict[str, Any]] = []
    for configuration, sub in per_seed.groupby("configuration"):
        summary_rows.append({
            "configuration": configuration,
            "configuration_label": THESIS_CONFIGS[configuration]["label"],
            "seed": "mean",
            "train_population": THESIS_CONFIGS[configuration]["population_mode"],
            "learning_gain_canonical": round(float(sub["learning_gain_canonical"].mean()), 4),
            "learning_gain_mixed": round(float(sub["learning_gain_mixed"].mean()), 4),
            "same_population_learning_gain": round(
                float(sub["same_population_learning_gain"].mean()), 4
            ),
            "mixed_population_learning_gain": round(
                float(sub["mixed_population_learning_gain"].mean()), 4
            ),
            "generalization_gap": round(float(sub["generalization_gap"].mean()), 4),
            "canonical_transfer_gap": round(float(sub["canonical_transfer_gap"].mean()), 4),
            "overfitting_rate": round(float(sub["overfitting_signal"].mean()), 4),
        })
    return pd.concat([per_seed, pd.DataFrame(summary_rows)], ignore_index=True)


def pairwise_statistics(results_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    p_vals: List[float] = []
    sub_a = results_df[results_df["configuration"] == "A_stable_canonical"]
    sub_b = results_df[results_df["configuration"] == "B_realistic_mixed"]

    for metric in STAT_METRICS:
        v_a = sub_a[metric].astype(float).tolist()
        v_b = sub_b[metric].astype(float).tolist()
        if len(v_a) < 2 or len(v_b) < 2:
            continue
        t_stat, p_val = stats.ttest_ind(v_a, v_b, equal_var=False)
        mean_a, lo_a, hi_a = confidence_interval(v_a)
        mean_b, lo_b, hi_b = confidence_interval(v_b)
        rows.append({
            "comparison": "A_stable_canonical_vs_B_realistic_mixed",
            "metric": metric,
            "mean_A": round(mean_a, 4),
            "mean_B": round(mean_b, 4),
            "delta_A_minus_B": round(mean_a - mean_b, 4),
            "ci_95_A": f"[{lo_a:.3f}, {hi_a:.3f}]",
            "ci_95_B": f"[{lo_b:.3f}, {hi_b:.3f}]",
            "std_A": round(float(np.std(v_a, ddof=1)), 4),
            "std_B": round(float(np.std(v_b, ddof=1)), 4),
            "cohens_d": round(cohens_d(v_a, v_b), 4),
            "t": round(float(t_stat), 4),
            "p_raw": round(float(p_val), 6),
            "n_A": len(v_a),
            "n_B": len(v_b),
        })
        p_vals.append(p_val)

    if p_vals:
        order = np.argsort(p_vals)
        holm = [1.0] * len(p_vals)
        for rank, idx in enumerate(order):
            holm[idx] = min(1.0, p_vals[idx] * (len(p_vals) - rank))
        for i, row in enumerate(rows):
            row["p_holm"] = round(holm[i], 6)
    return pd.DataFrame(rows)


def build_ranking(summary_df: pd.DataFrame) -> pd.DataFrame:
    merged = summary_df.copy()
    merged = merged.rename(columns={"mean_episode_reward_mean": "reward_mean"})
    score = pd.Series(0.0, index=merged.index)
    rank_specs = {
        "learning_gain_mean": False,
        "learning_gain_std": True,
        "coefficient_of_variation": True,
        "success_rate_mean": False,
        "adaptation_accuracy_mean": False,
        "reward_mean": False,
        "shannon_entropy_mean": False,
        "dominant_action_pct_mean": True,
        "seed_range": True,
    }
    for col, ascending in rank_specs.items():
        if col in merged.columns:
            score += merged[col].rank(ascending=ascending, method="average")
    merged["composite_score"] = score
    merged["rank"] = score.rank(method="min").astype(int)

    cols = [
        "rank",
        "configuration",
        "configuration_label",
        "population_mode",
        "learning_gain_mean",
        "learning_gain_std",
        "coefficient_of_variation",
        "seed_range",
        "learning_gain_ci_95",
        "success_rate_mean",
        "adaptation_accuracy_mean",
        "shannon_entropy_mean",
        "dominant_action_pct_mean",
        "actions_used_above_1_percent_mean",
        "best_seed",
        "worst_seed",
    ]
    return merged[[c for c in cols if c in merged.columns]].sort_values("rank")


def _plot_comparison(summary_df: pd.DataFrame) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    order = ["A_stable_canonical", "B_realistic_mixed"]
    summary_df = summary_df.set_index("configuration").reindex(order).reset_index()
    labels = ["A: Stable\n(Canonical)", "B: Realistic\n(Mixed)"]
    colors = ["#E76F51", "#457B9D"]
    xs = np.arange(len(labels))

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    panels = [
        ("learning_gain_mean", "learning_gain_std", "Learning Gain"),
        ("coefficient_of_variation", None, "Coefficient of Variation"),
        ("success_rate_mean", "success_rate_std", "Success Rate"),
        ("shannon_entropy_mean", "shannon_entropy_std", "Action Entropy (Shannon)"),
    ]
    for ax, (mean_col, std_col, title) in zip(axes.flat, panels):
        means = summary_df[mean_col].tolist()
        if std_col and std_col in summary_df.columns:
            stds = summary_df[std_col].tolist()
            ax.bar(xs, means, yerr=stds, capsize=5, color=colors, edgecolor="#64748B")
        else:
            ax.bar(xs, means, color=colors, edgecolor="#64748B")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Final Thesis Configuration Comparison (DQN, {TRAIN_TS // 1000}k steps, "
        f"Balanced-2 + Nearly Equal Gains)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "final_comparison.png", dpi=150)
    plt.close(fig)


def _plot_generalization_gap(gen_summary: pd.DataFrame) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    per_seed = gen_summary[gen_summary["seed"].astype(str) != "mean"]
    if per_seed.empty:
        return

    order = ["A_stable_canonical", "B_realistic_mixed"]
    labels = ["A: Stable (Canonical)", "B: Realistic (Mixed)"]
    colors = ["#E76F51", "#457B9D"]
    xs = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.35
    canon_means, mixed_means = [], []
    for key in order:
        sub = per_seed[per_seed["configuration"] == key]
        canon_means.append(float(sub["learning_gain_canonical"].mean()))
        mixed_means.append(float(sub["learning_gain_mixed"].mean()))

    ax.bar(xs - width / 2, canon_means, width, label="Eval: Canonical", color=colors[0], alpha=0.85)
    ax.bar(xs + width / 2, mixed_means, width, label="Eval: Mixed", color=colors[1], alpha=0.85)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Learning Gain")
    ax.set_title("Generalization: Canonical vs Mixed Evaluation")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "generalization_gap.png", dpi=150)
    plt.close(fig)


def _thesis_recommendation(
    summary_df: pd.DataFrame,
    gen_summary: pd.DataFrame,
    stats_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
) -> Dict[str, Any]:
    a = summary_df[summary_df["configuration"] == "A_stable_canonical"].iloc[0]
    b = summary_df[summary_df["configuration"] == "B_realistic_mixed"].iloc[0]
    winner = ranking_df.iloc[0]

    gen_a = gen_summary[
        (gen_summary["configuration"] == "A_stable_canonical")
        & (gen_summary["seed"].astype(str) != "mean")
    ]
    gen_b = gen_summary[
        (gen_summary["configuration"] == "B_realistic_mixed")
        & (gen_summary["seed"].astype(str) != "mean")
    ]

    a_overfits = (
        not gen_a.empty
        and float(gen_a["generalization_gap"].mean()) > 0.03
    )
    b_generalizes_better = (
        not gen_b.empty
        and not gen_a.empty
        and float(gen_b["learning_gain_canonical"].mean())
        >= float(gen_a["learning_gain_mixed"].mean()) * 0.95
    )

    lg_sig = stats_df[stats_df["metric"] == "learning_gain"]
    lg_p = float(lg_sig["p_holm"].iloc[0]) if len(lg_sig) else float("nan")

    if winner["configuration"] == "A_stable_canonical" and a_overfits:
        rec = (
            "Report Configuration A as the stability-optimized system with explicit "
            "generalization caveats; use mixed-population evaluation as a robustness "
            "appendix. Canonical training improves seed stability but shows measurable "
            "mixed-population drop."
        )
        report_config = "A_stable_canonical"
    elif winner["configuration"] == "B_realistic_mixed" or b_generalizes_better:
        rec = (
            "Adopt Configuration B as the final thesis system: it preserves heterogeneous "
            "student realism and generalizes more credibly across evaluation populations."
        )
        report_config = "B_realistic_mixed"
    else:
        rec = (
            "Adopt Configuration A for the primary thesis results (highest stability and "
            "learning gain on the training distribution) while reporting Configuration B "
            "mixed-eval metrics for external validity."
        )
        report_config = str(winner["configuration"])

    return {
        "recommended_configuration": report_config,
        "composite_winner": str(winner["configuration"]),
        "highest_learning_gain": str(
            summary_df.loc[summary_df["learning_gain_mean"].idxmax(), "configuration"]
        ),
        "lowest_variance": str(
            summary_df.loc[summary_df["learning_gain_std"].idxmin(), "configuration"]
        ),
        "canonical_overfitting_detected": a_overfits,
        "learning_gain_p_holm": lg_p,
        "recommendation_text": rec,
    }


def _print_winner(
    summary_df: pd.DataFrame,
    gen_summary: pd.DataFrame,
    stats_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    recommendation: Dict[str, Any],
) -> None:
    a = summary_df[summary_df["configuration"] == "A_stable_canonical"].iloc[0]
    b = summary_df[summary_df["configuration"] == "B_realistic_mixed"].iloc[0]
    winner_key = recommendation["recommended_configuration"]
    winner_row = summary_df[summary_df["configuration"] == winner_key].iloc[0]

    print("\n" + "=" * 72)
    print("WINNER FINAL THESIS CONFIGURATION")
    print("=" * 72)
    print(f"\n  {winner_row['configuration_label']} ({winner_key})")
    print(f"  {recommendation['recommendation_text']}")

    print("\n--- Performance Comparison ---")
    print(
        f"  A learning_gain: {a['learning_gain_mean']:.4f} "
        f"(CI {a['learning_gain_ci_95']})"
    )
    print(
        f"  B learning_gain: {b['learning_gain_mean']:.4f} "
        f"(CI {b['learning_gain_ci_95']})"
    )
    print(f"  A success_rate: {a['success_rate_mean']:.4f}  |  B: {b['success_rate_mean']:.4f}")
    print(
        f"  A adaptation_accuracy: {a['adaptation_accuracy_mean']:.4f}  |  "
        f"B: {b['adaptation_accuracy_mean']:.4f}"
    )

    print("\n--- Stability Comparison ---")
    print(
        f"  A std={a['learning_gain_std']:.4f}, CV={a['coefficient_of_variation']:.4f}, "
        f"range={a['seed_range']:.4f} (best seed {a['best_seed']}, worst {a['worst_seed']})"
    )
    print(
        f"  B std={b['learning_gain_std']:.4f}, CV={b['coefficient_of_variation']:.4f}, "
        f"range={b['seed_range']:.4f} (best seed {b['best_seed']}, worst {b['worst_seed']})"
    )

    print("\n--- Diversity Comparison ---")
    print(
        f"  A entropy={a['shannon_entropy_mean']:.4f}, dominant={a['dominant_action_pct_mean']:.4f}, "
        f"top2={a['top2_actions_pct_mean']:.4f}, actions>1%={a['actions_used_above_1_percent_mean']:.1f}"
    )
    print(
        f"  B entropy={b['shannon_entropy_mean']:.4f}, dominant={b['dominant_action_pct_mean']:.4f}, "
        f"top2={b['top2_actions_pct_mean']:.4f}, actions>1%={b['actions_used_above_1_percent_mean']:.1f}"
    )

    print("\n--- Generalization Comparison ---")
    for key, label in [
        ("A_stable_canonical", "A (trained canonical)"),
        ("B_realistic_mixed", "B (trained mixed)"),
    ]:
        sub = gen_summary[
            (gen_summary["configuration"] == key) & (gen_summary["seed"].astype(str) == "mean")
        ]
        if not sub.empty:
            r = sub.iloc[0]
            print(
                f"  {label}: canonical LG={r['learning_gain_canonical']:.4f}, "
                f"mixed LG={r['learning_gain_mixed']:.4f}, gap={r['generalization_gap']:.4f}"
            )

    print("\n--- Statistical Significance (A vs B) ---")
    for _, row in stats_df.iterrows():
        sig = "yes" if row.get("p_holm", 1.0) < 0.05 else "no"
        print(
            f"  {row['metric']}: delta={row['delta_A_minus_B']:+.4f}, "
            f"d={row['cohens_d']:.3f}, p_holm={row.get('p_holm', row['p_raw']):.4f} "
            f"(significant={sig})"
        )

    print("\n--- Final Thesis Recommendation ---")
    print(f"  Composite rank winner: {ranking_df.iloc[0]['configuration_label']}")
    print(f"  Highest learning gain: {recommendation['highest_learning_gain']}")
    print(f"  Lowest variance: {recommendation['lowest_variance']}")
    print(f"  Canonical overfitting (A): {recommendation['canonical_overfitting_detected']}")
    print("=" * 72)


def analyze(
    results_df: Optional[pd.DataFrame] = None,
    gen_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    results_path = OUT_DIR / "final_comparison_results.csv"
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
    gen_summary = aggregate_generalization(gen_df)
    stats_df = pairwise_statistics(results_df)
    ranking_df = build_ranking(summary_df)
    recommendation = _thesis_recommendation(summary_df, gen_summary, stats_df, ranking_df)

    results_df.to_csv(OUT_DIR / "final_comparison_results.csv", index=False)
    summary_df.to_csv(OUT_DIR / "final_comparison_summary.csv", index=False)
    ranking_df.to_csv(OUT_DIR / "final_comparison_ranking.csv", index=False)
    if not gen_df.empty and "eval_population" in gen_df.columns:
        gen_df.to_csv(OUT_DIR / "generalization_eval_rows.csv", index=False)
    if not gen_summary.empty:
        gen_per_seed = gen_summary[gen_summary["seed"].astype(str) != "mean"]
        gen_per_seed.to_csv(OUT_DIR / "generalization_results.csv", index=False)
        gen_summary.to_csv(OUT_DIR / "generalization_summary.csv", index=False)
    if not stats_df.empty:
        stats_df.to_csv(OUT_DIR / "statistical_tests.csv", index=False)

    _plot_comparison(summary_df)
    _plot_generalization_gap(gen_summary)
    _print_winner(summary_df, gen_summary, stats_df, ranking_df, recommendation)

    report = {
        "study": "final_thesis_configuration_comparison",
        "train_timesteps": TRAIN_TS,
        "seeds": sorted(results_df["seed"].unique().tolist()),
        "configurations": {
            k: {kk: vv for kk, vv in v.items() if kk != "legacy_combined_key"}
            for k, v in THESIS_CONFIGS.items()
        },
        "hyperparameters": BASELINE_HP,
        "reward_weights": BALANCED_2_REWARD,
        "action_gains": NEARLY_EQUAL_GAINS,
        "summary": summary_df.to_dict("records"),
        "ranking": ranking_df.to_dict("records"),
        "statistical_tests": stats_df.to_dict("records"),
        "generalization": gen_summary.to_dict("records"),
        "recommendation": recommendation,
        "winner": recommendation["recommended_configuration"],
    }
    with open(OUT_DIR / "final_thesis_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    return report


def main() -> None:
    global TRAIN_TS, EVAL_EPS, VAL_EPS

    parser = argparse.ArgumentParser(
        description="Final thesis comparison: canonical stable (A) vs mixed realistic (B)"
    )
    parser.add_argument("--phase", choices=["run", "analyze", "all"], default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--no-import-legacy",
        action="store_true",
        help="Do not seed results from combined_improvement_validation CSV",
    )
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    seeds = SEEDS
    train_ts = TRAIN_TS
    if args.quick:
        seeds = [42, 7]
        train_ts = 5_000
        EVAL_EPS = 50
        VAL_EPS = 20
        print(f"QUICK MODE: seeds={seeds}, train={train_ts}, eval={EVAL_EPS}")

    if args.phase in ("run", "all"):
        print("=== Final Thesis Configuration Comparison (run) ===")
        TRAIN_TS = train_ts
        run_study(seeds, args.resume, import_legacy=not args.no_import_legacy)

    if args.phase in ("analyze", "all"):
        print("\n=== Final Thesis Configuration Comparison (analyze) ===")
        analyze()


if __name__ == "__main__":
    main()
