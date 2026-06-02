"""
Thesis gain-ratio sensitivity: Random vs ERT vs DQN across G0-G5.

Fixed DQN hyperparameters (pre-registered thesis model):
  lr=0.001, batch=64, buffer=100k, target_update=500, expl_frac=0.3, eps=0.05

Usage (from repo root):
  export PYTHONPATH="$(pwd)"
  python -m RL_Module.thesis_gain_sensitivity --phase all
  python -m RL_Module.thesis_gain_sensitivity --phase run --resume
  python -m RL_Module.thesis_gain_sensitivity --phase analyze
  python -m RL_Module.thesis_gain_sensitivity --phase all --quick
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import RL_Module.config as cfg
from RL_Module.agents.dqn_agent import DQNAgent, make_env
from RL_Module.agents.random_agent import RandomAgent
from RL_Module.agents.rule_based import ExpertRuleBasedAgent
from RL_Module.environment.student_env import StudentEnv
from RL_Module.evaluation.metrics import (
    _adaptation_match,
    _is_success,
    action_frequency_report,
    confidence_interval,
    normalized_knowledge_gain,
)
from RL_Module.mdp_definition import ACTIONS, ID_TO_ACTION

OUT_DIR = _HERE / "figures" / "thesis_gain_sensitivity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 7, 13, 21, 99, 314]
TRAIN_TS = 50_000
EVAL_EPS = 500
ALGORITHMS = ["Random", "ERT", "DQN"]

FINAL_DQN_HP: Dict[str, Any] = {
    "learning_rate": 0.001,
    "batch_size": 64,
    "buffer_size": 100_000,
    "target_update_interval": 500,
    "exploration_fraction": 0.3,
    "exploration_final_eps": 0.05,
}

GAIN_CONFIGS: Dict[str, Dict[str, float]] = {
    "G0": {"GAIN_CORRECT_FACTOR": 1.0, "GAIN_INCORRECT_FACTOR": 1.0, "ratio_label": "1:1"},
    "G1": {"GAIN_CORRECT_FACTOR": 0.5, "GAIN_INCORRECT_FACTOR": 1.0, "ratio_label": "2:1"},
    "G2": {"GAIN_CORRECT_FACTOR": 0.3, "GAIN_INCORRECT_FACTOR": 1.0, "ratio_label": "3.3:1"},
    "G3": {"GAIN_CORRECT_FACTOR": 0.2, "GAIN_INCORRECT_FACTOR": 1.0, "ratio_label": "5:1"},
    "G4": {"GAIN_CORRECT_FACTOR": 0.1, "GAIN_INCORRECT_FACTOR": 1.0, "ratio_label": "10:1"},
    "G5": {"GAIN_CORRECT_FACTOR": 0.05, "GAIN_INCORRECT_FACTOR": 1.0, "ratio_label": "20:1"},
}
GAIN_ORDER = list(GAIN_CONFIGS.keys())
PRIMARY_METRICS = [
    "learning_gain",
    "final_knowledge",
    "success_rate",
    "dropout_rate",
]
ALL_METRICS = PRIMARY_METRICS + [
    "adaptation_accuracy",
    "emotional_wellbeing",
    "mean_episode_reward",
]


def emotional_wellbeing(e: float, f: float, c: float, b: float) -> float:
    return float(np.clip(e - 0.35 * f - 0.35 * c - 0.20 * b, -1.0, 1.0))


def _patch_gain(params: Dict[str, float]) -> Dict[str, float]:
    prev = dict(cfg.SIMULATOR_PARAMS)
    cfg.SIMULATOR_PARAMS = {
        "GAIN_CORRECT_FACTOR": float(params["GAIN_CORRECT_FACTOR"]),
        "GAIN_INCORRECT_FACTOR": float(params["GAIN_INCORRECT_FACTOR"]),
    }
    return prev


def _restore_gain(snapshot: Dict[str, float]) -> None:
    cfg.SIMULATOR_PARAMS = snapshot


def _shannon_entropy(freq: Dict[str, float]) -> float:
    p = np.array([v for v in freq.values() if v > 0], dtype=float)
    return float(-np.sum(p * np.log(p))) if len(p) else 0.0


def evaluate_tutor(
    agent,
    env: StudentEnv,
    n_episodes: int,
    seed: int,
    algorithm: str,
) -> Dict[str, Any]:
    cfg.set_all_seeds(seed)
    episode_results: List[Dict[str, Any]] = []
    wellbeing_vals: List[float] = []

    for ep in range(1, n_episodes + 1):
        obs, info = env.reset(seed=seed + ep)
        mask = info["action_masks"]
        total_reward = 0.0
        start_k = env._state.knowledge
        action_counts = {i: 0 for i in range(8)}
        ep_adapt_hits = 0
        ep_adapt_total = 0
        dropout = False
        done = False

        while not done:
            emotion_id = env._state.emotion_id
            if isinstance(agent, ExpertRuleBasedAgent):
                action = agent.predict(
                    obs,
                    mask,
                    persistent_flag=info.get("persistent_frustration_flag", False),
                )
            else:
                action = agent.predict(obs, mask)

            obs, reward, terminated, truncated, info = env.step(action)
            ep_adapt_total += 1
            if _adaptation_match(emotion_id, action):
                ep_adapt_hits += 1

            s = env._state
            wellbeing_vals.append(
                emotional_wellbeing(s.engagement, s.frustration, s.confusion, s.boredom)
            )
            total_reward += reward
            action_counts[action] += 1
            mask = info["action_masks"]
            done = terminated or truncated
            dropout = info.get("dropout", False)

        final = env._state
        lg_norm = normalized_knowledge_gain(start_k, final.knowledge)
        success = _is_success(final.knowledge, final.frustration, final.confusion)
        episode_results.append({
            "total_reward": total_reward,
            "learning_gain": lg_norm,
            "success": int(success),
            "dropout": int(dropout),
            "final_knowledge": final.knowledge,
            "adaptation_accuracy": ep_adapt_hits / max(ep_adapt_total, 1),
            "action_counts": action_counts,
        })

    freq_report = action_frequency_report(episode_results, print_table=False)
    return {
        "mean_episode_reward": float(np.mean([r["total_reward"] for r in episode_results])),
        "success_rate": float(np.mean([r["success"] for r in episode_results])),
        "learning_gain": float(np.mean([r["learning_gain"] for r in episode_results])),
        "final_knowledge": float(np.mean([r["final_knowledge"] for r in episode_results])),
        "dropout_rate": float(np.mean([r["dropout"] for r in episode_results])),
        "adaptation_accuracy": float(np.mean([r["adaptation_accuracy"] for r in episode_results])),
        "emotional_wellbeing": float(np.mean(wellbeing_vals)) if wellbeing_vals else 0.0,
        "action_frequencies": {k: round(v, 4) for k, v in freq_report["frequencies"].items()},
        "action_diversity_shannon": _shannon_entropy(freq_report["frequencies"]),
        "dominance_detected": freq_report["dominance_detected"],
        "dominant_actions": freq_report["dominant_actions"],
    }


def run_single(
    algorithm: str,
    seed: int,
    gain_id: str,
    gain_params: Dict[str, float],
    train_ts: int,
    eval_eps: int,
) -> Dict[str, Any]:
    snap = _patch_gain(gain_params)
    cfg.USE_SENSITIVITY_WEIGHTS = False
    tag = f"{gain_id}_{algorithm}_s{seed}"
    t0 = time.time()
    try:
        cfg.set_all_seeds(seed)
        if algorithm == "DQN":
            env = make_env(seed=seed, algo_tag=f"thesis_{tag}")
            agent = DQNAgent()
            agent.train(env, train_ts, seed, hyperparams=FINAL_DQN_HP)
            env.close()
            model_path = cfg.MODELS_DIR / f"thesis_{tag}"
            agent.save(str(model_path))
        elif algorithm == "ERT":
            agent = ExpertRuleBasedAgent()
        else:
            agent = RandomAgent(seed=seed)

        eval_env = StudentEnv(population_seed=seed)
        eval_env.set_algorithm_name(f"{algorithm}_{gain_id}")
        result = evaluate_tutor(agent, eval_env, eval_eps, seed, tag)
        eval_env.close()
    finally:
        _restore_gain(snap)

    elapsed = time.time() - t0
    row = {
        "gain_id": gain_id,
        "ratio_label": gain_params["ratio_label"],
        "gain_correct": gain_params["GAIN_CORRECT_FACTOR"],
        "algorithm": algorithm,
        "seed": seed,
        "train_timesteps": train_ts if algorithm == "DQN" else 0,
        "eval_episodes": eval_eps,
        "elapsed_seconds": round(elapsed, 1),
        **{k: result[k] for k in ALL_METRICS},
        "action_diversity_shannon": result["action_diversity_shannon"],
        "dominance_detected": result["dominance_detected"],
        "action_frequencies": result["action_frequencies"],
        "dominant_actions": result["dominant_actions"],
    }
    print(
        f"  {tag}: lg={row['learning_gain']:.3f} fk={row['final_knowledge']:.3f} "
        f"succ={row['success_rate']:.3f} drop={row['dropout_rate']:.3f} "
        f"adapt={row['adaptation_accuracy']:.3f} reward={row['mean_episode_reward']:.3f} "
        f"({elapsed:.0f}s)"
    )
    return row


def aggregate_runs(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for m in ALL_METRICS + ["action_diversity_shannon"]:
        vals = [r[m] for r in runs]
        mean, lo, hi = confidence_interval(vals)
        out[f"{m}_mean"] = round(mean, 4)
        out[f"{m}_std"] = round(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, 4)
        out[f"{m}_ci_lo"] = round(lo, 4)
        out[f"{m}_ci_hi"] = round(hi, 4)
    out["n_seeds"] = len(runs)
    out["dominance_any_seed"] = any(r["dominance_detected"] for r in runs)
    return out


def cohens_d(a: List[float], b: List[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va = np.var(a, ddof=1)
    vb = np.var(b, ddof=1)
    pooled = np.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2))
    if pooled < 1e-12:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled)


def holm_correction(p_values: List[float]) -> List[float]:
    n = len(p_values)
    order = np.argsort(p_values)
    corrected = [1.0] * n
    for rank, idx in enumerate(order):
        corrected[idx] = min(1.0, p_values[idx] * (n - rank))
    return corrected


def ranking_for_row(row: Dict[str, float], metric: str = "learning_gain") -> Dict[str, int]:
    scores = {a: row[(a, metric)] for a in ALGORITHMS}
    sorted_algos = sorted(scores, key=lambda a: scores[a], reverse=True)
    return {a: i + 1 for i, a in enumerate(sorted_algos)}


def run_experiment(
    seeds: List[int],
    train_ts: int,
    eval_eps: int,
    resume: bool = False,
) -> pd.DataFrame:
    csv_path = OUT_DIR / "thesis_gain_per_run.csv"
    done_keys: set = set()
    per_run: List[Dict[str, Any]] = []

    if resume and csv_path.exists():
        existing = pd.read_csv(csv_path)
        per_run = existing.to_dict("records")
        for r in per_run:
            done_keys.add((r["gain_id"], r["algorithm"], int(r["seed"])))
        print(f"Resuming: {len(done_keys)} runs already complete.")

    total = len(GAIN_CONFIGS) * len(ALGORITHMS) * len(seeds)
    n_done = len(done_keys)

    for gain_id, gain_params in GAIN_CONFIGS.items():
        print(f"\n--- {gain_id} (ratio {gain_params['ratio_label']}) ---")
        for algorithm in ALGORITHMS:
            for seed in seeds:
                key = (gain_id, algorithm, seed)
                if key in done_keys:
                    continue
                n_done += 1
                print(f"[{n_done}/{total}] {algorithm} seed={seed}")
                row = run_single(algorithm, seed, gain_id, gain_params, train_ts, eval_eps)
                per_run.append(row)
                pd.DataFrame(per_run).to_csv(csv_path, index=False)

    return pd.DataFrame(per_run)


def _comparison_table_all_ratios(
    agg: Dict[Tuple[str, str], Dict[str, Any]],
) -> pd.DataFrame:
