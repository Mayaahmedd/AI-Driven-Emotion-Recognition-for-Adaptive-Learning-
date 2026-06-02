"""
Sensitivity analysis: reward weights and masking thresholds.

Runs a 3x3 factorial (weight config x threshold config) with short experiments
to verify algorithm rankings are stable across parameterizations (Spearman rho).
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Tuple

import pandas as pd
from scipy import stats

import RL_Module.config as cfg
from RL_Module import mdp_definition
from RL_Module.environment import student_env as senv
from RL_Module.main_experiment import run_single

WEIGHT_CONFIGS: Dict[str, Dict[str, Any]] = {
    "equal": {"wk": 0.20, "we": 0.20, "wf": 0.20, "wb": 0.20, "wc": 0.20},
    "knowledge_priority": {"wk": 0.70, "we": 0.15, "wf": 0.05, "wb": 0.05, "wc": 0.05},
    "affective_priority": {"wk": 0.30, "we": 0.25, "wf": 0.20, "wb": 0.15, "wc": 0.10},
}

# Simulator transition hyperparameter grids (literature-inspired defaults as baseline)
SIMULATOR_CONFIGS: Dict[str, Dict[str, float]] = {
    "default": {},
    "low_persistence": {
        "LAMBDA_FRUSTRATION": 0.5,
        "LAMBDA_ENGAGEMENT": 0.5,
        "LAMBDA_CONFUSION": 0.4,
        "LAMBDA_BOREDOM": 0.3,
    },
    "high_persistence": {
        "LAMBDA_FRUSTRATION": 0.8,
        "LAMBDA_ENGAGEMENT": 0.75,
        "LAMBDA_CONFUSION": 0.6,
        "LAMBDA_BOREDOM": 0.5,
    },
    "sharp_irt": {"IRT_BETA": 4.0},
    "flat_irt": {"IRT_BETA": 2.0},
    "wide_mismatch": {"MISMATCH_HIGH": 0.4, "MISMATCH_LOW": -0.4},
    "narrow_mismatch": {"MISMATCH_HIGH": 0.2, "MISMATCH_LOW": -0.2},
    "low_correct_gain": {"GAIN_CORRECT_FACTOR": 0.05},
    "high_correct_gain": {"GAIN_CORRECT_FACTOR": 0.2},
}

THRESHOLD_CONFIGS: Dict[str, Dict[str, float]] = {
    "default": {"FRUSTRATION_BLOCK_HARDER": 0.6},
    "stricter": {"FRUSTRATION_BLOCK_HARDER": 0.5},
    "looser": {"FRUSTRATION_BLOCK_HARDER": 0.7},
}

ALGORITHMS = ["PPO", "DQN", "Bandit+DQN", "Rule", "Random"]
TEST_SEED = 42
TIMESTEPS = 5_000


def _patch_thresholds(params: Dict[str, float]) -> Dict[str, float]:
    """Patch mdp_definition, config, and student_env captured names; return snapshot."""
    snapshot: Dict[str, float] = {}
    for key, val in params.items():
        snapshot[key] = getattr(mdp_definition, key)
        setattr(mdp_definition, key, val)
        setattr(cfg, key, val)
        setattr(senv, key, val)
    return snapshot


def _restore_thresholds(snapshot: Dict[str, float]) -> None:
    for key, val in snapshot.items():
        setattr(mdp_definition, key, val)
        setattr(cfg, key, val)
        setattr(senv, key, val)


def _patch_simulator_params(params: Dict[str, float]) -> Dict[str, float]:
    """Patch config.SIMULATOR_PARAMS; return previous snapshot."""
    prev = dict(cfg.SIMULATOR_PARAMS)
    cfg.SIMULATOR_PARAMS = dict(params)
    return prev


def _restore_simulator_params(snapshot: Dict[str, float]) -> None:
    cfg.SIMULATOR_PARAMS = snapshot


def _print_spearman(
    df: pd.DataFrame,
    group_cols: Tuple[str, str],
) -> None:
    """Spearman rho between every pair of (weight, threshold) config cells."""
    w_col, t_col = group_cols
    cells = df[[w_col, t_col]].drop_duplicates().values.tolist()
    print(f"\nSpearman rank correlation ({w_col} x {t_col}):")
    for i in range(len(cells)):
        for j in range(i + 1, len(cells)):
            w1, t1 = cells[i]
            w2, t2 = cells[j]
            ranks1 = df[(df[w_col] == w1) & (df[t_col] == t1)].set_index("algorithm")["rank"]
            ranks2 = df[(df[w_col] == w2) & (df[t_col] == t2)].set_index("algorithm")["rank"]
            rho, p = stats.spearmanr(ranks1, ranks2)
            print(f"  ({w1},{t1}) vs ({w2},{t2}): rho = {rho:.3f}  (p = {p:.3f})")
            if rho >= 0.9:
                print("    Rankings consistent (rho >= 0.9)")
            else:
                print("    WARNING: rankings differ across configs")


def run_factorial() -> pd.DataFrame:
    """3x3 factorial: 9 (weight, threshold) cells x 5 algorithms = 45 rows."""
    results: Dict[Tuple[str, str], Dict[str, float]] = {}
    orig_timesteps = cfg.TOTAL_TIMESTEPS
    orig_training = cfg.TRAINING_TIMESTEPS

    for w_name, w_params in WEIGHT_CONFIGS.items():
        cfg.SENSITIVITY_WEIGHTS = copy.deepcopy(w_params)
        cfg.USE_SENSITIVITY_WEIGHTS = True

        for t_name, t_params in THRESHOLD_CONFIGS.items():
            cell_key = (w_name, t_name)
            results[cell_key] = {}
            snap = _patch_thresholds(t_params)
            cfg.TOTAL_TIMESTEPS = TIMESTEPS
            cfg.TRAINING_TIMESTEPS = TIMESTEPS

            try:
                for algo in ALGORITHMS:
                    cfg.set_all_seeds(TEST_SEED)
                    result = run_single(algo, seed=TEST_SEED, eval_episodes=50)
                    reward = result["mean_episode_reward"]
                    results[cell_key][algo] = reward
                    print(f"  {w_name} | {t_name} | {algo}: {reward:.3f}")
            finally:
                _restore_thresholds(snap)

        cfg.USE_SENSITIVITY_WEIGHTS = False

    cfg.TOTAL_TIMESTEPS = orig_timesteps
    cfg.TRAINING_TIMESTEPS = orig_training

    rows: List[Dict[str, Any]] = []
    for (w_name, t_name), algo_results in results.items():
        ranked = sorted(algo_results.items(), key=lambda x: x[1], reverse=True)
        for rank, (algo, reward) in enumerate(ranked, 1):
            rows.append({
                "weight_config": w_name,
                "threshold_config": t_name,
                "algorithm": algo,
                "mean_reward": round(reward, 3),
                "rank": rank,
            })

    df = pd.DataFrame(rows)
    out_path = cfg.LOGS_DIR / "sensitivity_factorial.csv"
    df.to_csv(out_path, index=False)
    print(f"\nFactorial sensitivity saved to {out_path} ({len(df)} rows)")
    _print_spearman(df, ("weight_config", "threshold_config"))
    return df


def run_simulator_sensitivity(
    algorithms: List[str] | None = None,
    simulator_configs: Dict[str, Dict[str, float]] | None = None,
) -> pd.DataFrame:
    """
  Sweep SIMULATOR_PARAMS (lambda, IRT, mismatch, gain factors) with short PPO runs.
  """
    algorithms = algorithms or ["PPO"]
    simulator_configs = simulator_configs or SIMULATOR_CONFIGS
    rows: List[Dict[str, Any]] = []
    orig_timesteps = cfg.TOTAL_TIMESTEPS
    orig_training = cfg.TRAINING_TIMESTEPS
    cfg.USE_SENSITIVITY_WEIGHTS = False

    for sim_name, sim_params in simulator_configs.items():
        snap = _patch_simulator_params(sim_params)
        cfg.TOTAL_TIMESTEPS = TIMESTEPS
        cfg.TRAINING_TIMESTEPS = TIMESTEPS
        try:
            for algo in algorithms:
                cfg.set_all_seeds(TEST_SEED)
                result = run_single(algo, seed=TEST_SEED, eval_episodes=50)
                rows.append({
                    "simulator_config": sim_name,
                    "algorithm": algo,
                    "mean_reward": round(result["mean_episode_reward"], 3),
                    "learning_gain": round(result["learning_gain"], 3),
                    "success_rate": round(result["success_rate"], 3),
                })
                print(
                    f"  sim={sim_name} | {algo}: "
                    f"reward={result['mean_episode_reward']:.3f} "
                    f"lg={result['learning_gain']:.3f}"
                )
        finally:
            _restore_simulator_params(snap)

    cfg.TOTAL_TIMESTEPS = orig_timesteps
    cfg.TRAINING_TIMESTEPS = orig_training

    df = pd.DataFrame(rows)
    out_path = cfg.LOGS_DIR / "sensitivity_simulator.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSimulator sensitivity saved to {out_path} ({len(df)} rows)")
    return df


def run_sensitivity() -> pd.DataFrame:
    print("=== 3x3 factorial sensitivity (weights x thresholds) ===")
    df = run_factorial()
    print("\n=== Simulator parameter sensitivity (PPO) ===")
    run_simulator_sensitivity(algorithms=["PPO"])
    return df


if __name__ == "__main__":
    run_sensitivity()
