"""
DQN vs Double DQN final algorithm comparison.

Configuration A: Standard DQN (current implementation)
Configuration B: Double DQN (same architecture/hyperparameters; DDQN target only)

Fixed thesis MDP: Balanced-2 reward, nearly equal action gains, mixed population,
IRT_BETA=3.0, default transition noise, G4 gain ratio, checkpoint selection.

Usage (from repo root):
  export PYTHONPATH="$(pwd)"
  python -m RL_Module.dqn_vs_double_dqn --phase all
  python -m RL_Module.dqn_vs_double_dqn --phase run --resume
  python -m RL_Module.dqn_vs_double_dqn --phase analyze
  python -m RL_Module.dqn_vs_double_dqn --phase all --quick
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

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
from RL_Module.agents.double_dqn_agent import DoubleDQNAgent
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

OUT_DIR = _HERE / "figures" / "dqn_vs_double_dqn"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS_10 = [42, 7, 13, 21, 99, 314, 555, 777, 888, 999]
SEEDS_20 = SEEDS_10 + [500, 1234, 2024, 5555, 8888, 9999, 10101, 20202, 30303, 4242]
SEEDS = SEEDS_20

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

THESIS_HP: Dict[str, Any] = {
    "learning_rate": 0.0005,
    "batch_size": 64,
    "buffer_size": 100_000,
    "target_update_interval": 2000,
    "exploration_fraction": 0.3,
    "exploration_final_eps": 0.05,
    "net_arch": [64, 64],
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

ALGORITHM_CONDITIONS: Dict[str, Dict[str, Any]] = {
    "A_standard_dqn": {
        "label": "Standard DQN (A)",
        "agent_class": "dqn",
    },
    "B_double_dqn": {
        "label": "Double DQN (B)",
        "agent_class": "double_dqn",
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

STAT_METRICS = ["learning_gain", "success_rate", "adaptation_accuracy"]
ACTION_NAMES = list(ID_TO_ACTION.values())

AgentType = Union[DQNAgent, DoubleDQNAgent]


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


def _patch_mdp() -> Tuple[Dict[str, float], Dict[int, float]]:
    gain_snap = dict(cfg.SIMULATOR_PARAMS)
    cfg.SIMULATOR_PARAMS = dict(G4_GAIN)
    action_snap: Dict[int, float] = {}
    for action_id, action_name in ID_TO_ACTION.items():
        action_snap[action_id] = float(student_model.ACTION_EFFECT_MAP[action_id]["base_gain"])
        if action_name in NEARLY_EQUAL_GAINS:
            student_model.ACTION_EFFECT_MAP[action_id]["base_gain"] = float(
                NEARLY_EQUAL_GAINS[action_name]
            )
    return gain_snap, action_snap


def _restore_mdp(gain_snap: Dict[str, float], action_snap: Dict[int, float]) -> None:
    cfg.SIMULATOR_PARAMS = gain_snap
    for action_id, base_gain in action_snap.items():
        student_model.ACTION_EFFECT_MAP[action_id]["base_gain"] = base_gain


def _make_agent(agent_class: str) -> AgentType:
    if agent_class == "dqn":
        return DQNAgent()
    if agent_class == "double_dqn":
        return DoubleDQNAgent()
    raise ValueError(agent_class)


def _checkpoint_prefix(agent_class: str) -> str:
    return "dqn" if agent_class == "dqn" else "double_dqn"


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
    agent: AgentType,
    seed: int,
    population_mode: str = "mixed",
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


def measure_q_values(
    agent: AgentType,
    seed: int,
    population_mode: str = "mixed",
    n_episodes: int = 100,
    gamma: float = 0.99,
) -> Dict[str, float]:
    """Collect Q-value statistics and overestimation bias during evaluation rollouts."""
    with _population_patch(population_mode):
        env = StudentEnv(
            population_seed=seed,
            obs_ablation="full_emotion",
            emotion_dynamics="full",
        )
        cfg.set_all_seeds(seed)
        all_q: List[float] = []
        max_q_per_step: List[float] = []
        selected_q: List[float] = []
        overestimation: List[float] = []

        for ep in range(1, n_episodes + 1):
            obs, info = env.reset(seed=seed + ep)
            mask = info["action_masks"]
            rewards: List[float] = []
            q_max_steps: List[float] = []
            done = False

            while not done:
                q_vals = agent._masked_q_values(obs, mask)
                finite = q_vals[np.isfinite(q_vals)]
                if len(finite):
                    q_max = float(np.max(finite))
                    q_max_steps.append(q_max)
                    all_q.extend(finite.tolist())
                    max_q_per_step.append(q_max)

                action = agent.predict(obs, mask)
                sel_q = float(q_vals[action]) if np.isfinite(q_vals[action]) else 0.0
                selected_q.append(sel_q)

                obs, reward, term, trunc, info = env.step(action)
                rewards.append(reward)
                mask = info["action_masks"]
                done = term or trunc

            for t, q_max in enumerate(q_max_steps):
                mc_return = 0.0
                discount = 1.0
                for r in rewards[t:]:
                    mc_return += discount * r
                    discount *= gamma
                overestimation.append(q_max - mc_return)

        env.close()

    return {
        "mean_q_value": float(np.mean(all_q)) if all_q else 0.0,
        "max_q_value": float(np.mean(max_q_per_step)) if max_q_per_step else 0.0,
        "q_value_variance": float(np.var(all_q)) if all_q else 0.0,
        "mean_selected_q": float(np.mean(selected_q)) if selected_q else 0.0,
        "overestimation_bias": float(np.mean(overestimation)) if overestimation else 0.0,
    }


def evaluate_agent(
    agent: AgentType,
    seed: int,
    population_mode: str = "mixed",
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


def run_single(condition_key: str, seed: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    spec = ALGORITHM_CONDITIONS[condition_key]
    agent_class = spec["agent_class"]
    gain_snap, action_snap = _patch_mdp()
    tag = f"dqn_vs_ddqn_{condition_key}_s{seed}"
    prefix = _checkpoint_prefix(agent_class)
    t0 = time.time()
    gen_rows: List[Dict[str, Any]] = []

    try:
        with _population_patch("mixed"):
            cfg.set_all_seeds(seed)
            train_env = make_env(
                seed=seed,
                obs_ablation="full_emotion",
                emotion_dynamics="full",
                algo_tag=tag,
            )
            agent = _make_agent(agent_class)
            ckpt_dir = cfg.MODELS_DIR / tag
            paths = agent.train_with_checkpoints(
                train_env,
                TRAIN_TS,
                seed,
                hyperparams=THESIS_HP,
                checkpoint_dir=str(ckpt_dir),
                checkpoint_freq=min(10_000, TRAIN_TS),
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
            lambda a, s: quick_eval_lg(a, s, "mixed", VAL_EPS),
            seed,
        )
        agent.load_checkpoint(best_path)
        metrics = evaluate_agent(agent, seed, "mixed", EVAL_EPS, collect_actions=True)
        q_stats = measure_q_values(agent, seed, "mixed", n_episodes=100)
        selection = f"checkpoint@{best_path}"

        for eval_mode, eval_label in (
            ("mixed", "Mixed Population"),
            ("single_canonical", "Single Canonical Student"),
        ):
            eval_metrics = evaluate_agent(
                agent, seed, eval_mode, EVAL_EPS, collect_actions=False
            )
            gen_rows.append({
                "configuration": condition_key,
                "configuration_label": spec["label"],
                "seed": seed,
                "train_population": "mixed",
                "eval_population": eval_mode,
                "eval_population_label": eval_label,
                "checkpoint": best_path,
                **{m: eval_metrics[m] for m in METRICS},
            })
    finally:
        _restore_mdp(gain_snap, action_snap)

    elapsed = time.time() - t0
    row: Dict[str, Any] = {
        "configuration": condition_key,
        "configuration_label": spec["label"],
        "algorithm": "DQN" if agent_class == "dqn" else "DoubleDQN",
        "seed": seed,
        "train_timesteps": TRAIN_TS,
        "eval_episodes": EVAL_EPS,
        "use_checkpoints": True,
        "selection": selection,
        "val_learning_gain": best_val,
        "gain_ratio": G4_GAIN["ratio_label"],
        "irt_beta": G4_GAIN["IRT_BETA"],
        "elapsed_seconds": round(elapsed, 1),
        **{m: metrics[m] for m in METRICS},
        "action_entropy": metrics["action_entropy"],
        "actions_used_above_1_percent": metrics["actions_used_above_1_percent"],
        "actions_used_above_5_percent": metrics["actions_used_above_5_percent"],
        "dominant_action_pct": metrics["dominant_action_pct"],
        "top2_actions_pct": metrics["top2_actions_pct"],
        "dominance_detected": metrics["dominance_detected"],
        **q_stats,
    }
    for action_name, freq in metrics["action_frequencies"].items():
        row[f"freq_{action_name}"] = freq

    print(
        f"  {condition_key} seed={seed}: "
        f"lg={row['learning_gain']:.3f} succ={row['success_rate']:.3f} "
        f"adapt={row['adaptation_accuracy']:.3f} entropy={row['action_entropy']:.3f} "
        f"mean_q={row['mean_q_value']:.2f} ({elapsed:.0f}s)"
    )
    return row, gen_rows


def _load_resume(path: Path, resume: bool) -> List[Dict[str, Any]]:
    if resume and path.exists():
        return pd.read_csv(path).to_dict("records")
    return []


def run_study(seeds: List[int], resume: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    results_path = OUT_DIR / "dqn_vs_double_dqn_results.csv"
    gen_path = OUT_DIR / "generalization_results.csv"
    rows = _load_resume(results_path, resume)
    gen_rows = _load_resume(gen_path, resume)
    done = {(r["configuration"], int(r["seed"])) for r in rows}

    for condition_key in ALGORITHM_CONDITIONS:
        print(f"\n--- {ALGORITHM_CONDITIONS[condition_key]['label']} ---")
        for seed in seeds:
            if (condition_key, seed) in done:
                continue
            row, new_gen = run_single(condition_key, seed)
            rows.append(row)
            gen_rows.extend(new_gen)
            pd.DataFrame(rows).to_csv(results_path, index=False)
            pd.DataFrame(gen_rows).to_csv(gen_path, index=False)

    return pd.DataFrame(rows), pd.DataFrame(gen_rows)


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
        spec = ALGORITHM_CONDITIONS[configuration]
        lgs = sub["learning_gain"].astype(float).tolist()
        lg_mean, lg_lo, lg_hi = confidence_interval(lgs)
        lg_std = float(np.std(lgs, ddof=1)) if len(lgs) > 1 else 0.0
        best_idx = sub["learning_gain"].astype(float).idxmax()
        worst_idx = sub["learning_gain"].astype(float).idxmin()

        row: Dict[str, Any] = {
            "configuration": configuration,
            "configuration_label": spec["label"],
            "algorithm": sub["algorithm"].iloc[0],
            "n_seeds": len(sub),
            "learning_gain_mean": round(lg_mean, 4),
            "learning_gain_std": round(lg_std, 4),
            "coefficient_of_variation": round(_cv(lgs), 4),
            "seed_range": round(float(max(lgs) - min(lgs)) if lgs else 0.0, 4),
            "learning_gain_ci_95": f"[{lg_lo:.3f}, {lg_hi:.3f}]",
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

        for qcol in ("mean_q_value", "max_q_value", "q_value_variance", "overestimation_bias"):
            if qcol in sub.columns:
                vals = sub[qcol].astype(float).tolist()
                row[f"{qcol}_mean"] = round(float(np.mean(vals)), 4)
                row[f"{qcol}_std"] = round(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, 4)

        for dcol in (
            "action_entropy",
            "dominant_action_pct",
            "top2_actions_pct",
            "actions_used_above_1_percent",
            "actions_used_above_5_percent",
        ):
            if dcol in sub.columns:
                vals = sub[dcol].astype(float).tolist()
                row[f"{dcol}_mean"] = round(float(np.mean(vals)), 4)
                row[f"{dcol}_std"] = round(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, 4)

        for action_name in ACTION_NAMES:
            col = f"freq_{action_name}"
            if col in sub.columns:
                row[f"{action_name}_mean_freq"] = round(float(sub[col].mean()), 4)

        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_generalization(gen_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for configuration, sub in gen_df.groupby("configuration"):
        spec = ALGORITHM_CONDITIONS[configuration]
        mixed = sub[sub["eval_population"] == "mixed"]
        canonical = sub[sub["eval_population"] == "single_canonical"]
        if mixed.empty or canonical.empty:
            continue
        merged = mixed.merge(
            canonical,
            on="seed",
            suffixes=("_mixed", "_canonical"),
        )
        same_lg = merged["learning_gain_mixed"].astype(float).tolist()
        cross_lg = merged["learning_gain_canonical"].astype(float).tolist()
        gaps = (merged["learning_gain_mixed"] - merged["learning_gain_canonical"]).astype(float).tolist()
        row = {
            "configuration": configuration,
            "configuration_label": spec["label"],
            "n_seeds": len(merged),
            "same_population_learning_gain_mean": round(float(np.mean(same_lg)), 4),
            "cross_population_learning_gain_mean": round(float(np.mean(cross_lg)), 4),
            "generalization_gap_mean": round(float(np.mean(gaps)), 4),
            "generalization_gap_std": round(float(np.std(gaps, ddof=1)) if len(gaps) > 1 else 0.0, 4),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def statistical_tests(results_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    dqn = results_df[results_df["configuration"] == "A_standard_dqn"]
    ddqn = results_df[results_df["configuration"] == "B_double_dqn"]
    for metric in STAT_METRICS:
        v_a = dqn[metric].astype(float).tolist()
        v_b = ddqn[metric].astype(float).tolist()
        if len(v_a) < 2 or len(v_b) < 2:
            continue
        t_stat, p_val = stats.ttest_ind(v_b, v_a, equal_var=False)
        mean_a, lo_a, hi_a = confidence_interval(v_a)
        mean_b, lo_b, hi_b = confidence_interval(v_b)
        pct_change = ((mean_b - mean_a) / mean_a * 100.0) if abs(mean_a) > 1e-9 else float("nan")
        rows.append({
            "comparison": "DoubleDQN_vs_StandardDQN",
            "metric": metric,
            "mean_standard_dqn": round(mean_a, 4),
            "mean_double_dqn": round(mean_b, 4),
            "delta_double_dqn_minus_dqn": round(mean_b - mean_a, 4),
            "pct_change": round(pct_change, 2),
            "ci_95_standard_dqn": f"[{lo_a:.3f}, {hi_a:.3f}]",
            "ci_95_double_dqn": f"[{lo_b:.3f}, {hi_b:.3f}]",
            "cohens_d": round(cohens_d(v_b, v_a), 4),
            "welch_t": round(float(t_stat), 4),
            "p_value": round(float(p_val), 6),
            "significant_005": bool(p_val < 0.05),
        })

    for qcol, label in (
        ("mean_q_value", "mean_q_value"),
        ("max_q_value", "max_q_value"),
        ("q_value_variance", "q_value_variance"),
        ("overestimation_bias", "overestimation_bias"),
    ):
        if qcol not in results_df.columns:
            continue
        v_a = dqn[qcol].astype(float).tolist()
        v_b = ddqn[qcol].astype(float).tolist()
        if len(v_a) < 2 or len(v_b) < 2:
            continue
        t_stat, p_val = stats.ttest_ind(v_b, v_a, equal_var=False)
        rows.append({
            "comparison": "DoubleDQN_vs_StandardDQN",
            "metric": label,
            "mean_standard_dqn": round(float(np.mean(v_a)), 4),
            "mean_double_dqn": round(float(np.mean(v_b)), 4),
            "delta_double_dqn_minus_dqn": round(float(np.mean(v_b) - np.mean(v_a)), 4),
            "pct_change": round(
                (float(np.mean(v_b) - np.mean(v_a)) / float(np.mean(v_a)) * 100.0)
                if abs(np.mean(v_a)) > 1e-9
                else float("nan"),
                2,
            ),
            "ci_95_standard_dqn": "",
            "ci_95_double_dqn": "",
            "cohens_d": round(cohens_d(v_b, v_a), 4),
            "welch_t": round(float(t_stat), 4),
            "p_value": round(float(p_val), 6),
            "significant_005": bool(p_val < 0.05),
        })
    return pd.DataFrame(rows)


def build_ranking(summary_df: pd.DataFrame) -> pd.DataFrame:
    merged = summary_df.copy()
    score = pd.Series(0.0, index=merged.index)
    rank_specs = {
        "learning_gain_mean": False,
        "learning_gain_std": True,
        "coefficient_of_variation": True,
        "success_rate_mean": False,
        "adaptation_accuracy_mean": False,
        "action_entropy_mean": False,
        "dominant_action_pct_mean": True,
        "overestimation_bias_mean": True,
        "mean_q_value_mean": True,
    }
    for col, ascending in rank_specs.items():
        if col in merged.columns:
            score += merged[col].rank(ascending=ascending, method="average")
    merged["rank"] = score.rank(method="min").astype(int)
    cols = [
        "configuration",
        "configuration_label",
        "algorithm",
        "learning_gain_mean",
        "learning_gain_std",
        "coefficient_of_variation",
        "success_rate_mean",
        "adaptation_accuracy_mean",
        "action_entropy_mean",
        "dominant_action_pct_mean",
        "mean_q_value_mean",
        "overestimation_bias_mean",
        "rank",
    ]
    return merged[[c for c in cols if c in merged.columns]].sort_values("rank")


def _determine_winner(summary_df: pd.DataFrame, stats_df: pd.DataFrame) -> Dict[str, Any]:
    dqn = summary_df[summary_df["configuration"] == "A_standard_dqn"].iloc[0]
    ddqn = summary_df[summary_df["configuration"] == "B_double_dqn"].iloc[0]

    lg_improvement = (float(ddqn["learning_gain_mean"]) - float(dqn["learning_gain_mean"])) / float(
        dqn["learning_gain_mean"]
    )
    std_improvement = (
        float(dqn["learning_gain_std"]) - float(ddqn["learning_gain_std"])
    ) / float(dqn["learning_gain_std"]) if float(dqn["learning_gain_std"]) > 1e-9 else 0.0
    cv_improvement = (
        float(dqn["coefficient_of_variation"]) - float(ddqn["coefficient_of_variation"])
    ) / float(dqn["coefficient_of_variation"]) if float(dqn["coefficient_of_variation"]) > 1e-9 else 0.0

    lg_stat = stats_df[stats_df["metric"] == "learning_gain"]
    lg_sig = bool(lg_stat.iloc[0]["significant_005"]) if not lg_stat.empty else False

    meets_threshold = lg_improvement >= 0.05 or std_improvement >= 0.05
    recommend_ddqn = meets_threshold and float(ddqn["learning_gain_mean"]) >= float(dqn["learning_gain_mean"])

    ranking = build_ranking(summary_df)
    winner_key = ranking.iloc[0]["configuration"]
    winner_algo = ranking.iloc[0]["algorithm"]

    return {
        "winner_configuration": winner_key,
        "winner_algorithm": winner_algo,
        "learning_gain_improvement_pct": round(lg_improvement * 100, 2),
        "stability_std_improvement_pct": round(std_improvement * 100, 2),
        "stability_cv_improvement_pct": round(cv_improvement * 100, 2),
        "learning_gain_significant": lg_sig,
        "meets_5pct_threshold": meets_threshold,
        "recommend_replace_dqn": recommend_ddqn,
        "thesis_recommendation": (
            "Replace standard DQN with Double DQN in the final thesis system."
            if recommend_ddqn
            else "Retain standard DQN in the final thesis system."
        ),
    }


def _plot_comparison(summary_df: pd.DataFrame) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    order = ["A_standard_dqn", "B_double_dqn"]
    summary_df = summary_df.set_index("configuration").reindex(order).reset_index()
    labels = ["Standard\nDQN", "Double\nDQN"]
    colors = ["#457B9D", "#2A9D8F"]
    xs = np.arange(2)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    panels = [
        ("learning_gain_mean", "learning_gain_std", "Learning Gain"),
        ("success_rate_mean", "success_rate_std", "Success Rate"),
        ("adaptation_accuracy_mean", "adaptation_accuracy_std", "Adaptation Accuracy"),
        ("action_entropy_mean", "action_entropy_std", "Shannon Entropy"),
        ("dominant_action_pct_mean", "dominant_action_pct_std", "Dominant Action %"),
        ("mean_episode_reward_mean", "mean_episode_reward_std", "Mean Episode Reward"),
    ]
    for ax, (mean_c, std_c, title) in zip(axes.flat, panels):
        means = summary_df[mean_c].tolist()
        stds = summary_df[std_c].tolist() if std_c in summary_df.columns else [0.0, 0.0]
        ax.bar(xs, means, yerr=stds, capsize=5, color=colors, edgecolor="#64748B")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"DQN vs Double DQN (Balanced-2, nearly equal gains, mixed pop, "
        f"n={int(summary_df['n_seeds'].iloc[0])} seeds, {TRAIN_TS // 1000}k steps)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "dqn_vs_double_dqn_comparison.png", dpi=150)
    plt.close(fig)


def _plot_generalization(gen_summary_df: pd.DataFrame) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    order = ["A_standard_dqn", "B_double_dqn"]
    gen_summary_df = gen_summary_df.set_index("configuration").reindex(order).reset_index()
    labels = ["Standard DQN", "Double DQN"]
    xs = np.arange(2)
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    same = gen_summary_df["same_population_learning_gain_mean"].tolist()
    cross = gen_summary_df["cross_population_learning_gain_mean"].tolist()
    ax.bar(xs - width / 2, same, width, label="Mixed (train pop)", color="#457B9D", edgecolor="#64748B")
    ax.bar(xs + width / 2, cross, width, label="Single canonical", color="#E9C46A", edgecolor="#64748B")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Learning Gain")
    ax.set_title("Generalization: Same vs Cross Population Eval")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "generalization_comparison.png", dpi=150)
    plt.close(fig)


def _plot_q_values(summary_df: pd.DataFrame) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    order = ["A_standard_dqn", "B_double_dqn"]
    summary_df = summary_df.set_index("configuration").reindex(order).reset_index()
    labels = ["Standard DQN", "Double DQN"]
    xs = np.arange(2)
    colors = ["#457B9D", "#2A9D8F"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    q_panels = [
        ("mean_q_value_mean", "mean_q_value_std", "Mean Q Value"),
        ("max_q_value_mean", "max_q_value_std", "Mean Max Q / Step"),
        ("overestimation_bias_mean", "overestimation_bias_std", "Overestimation Bias"),
    ]
    for ax, (mean_c, std_c, title) in zip(axes, q_panels):
        if mean_c not in summary_df.columns:
            continue
        means = summary_df[mean_c].tolist()
        stds = summary_df[std_c].tolist()
        ax.bar(xs, means, yerr=stds, capsize=5, color=colors, edgecolor="#64748B")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Q-Value Analysis: Standard DQN vs Double DQN", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "q_value_analysis.png", dpi=150)
    plt.close(fig)


def _print_winner(
    summary_df: pd.DataFrame,
    gen_summary_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    verdict: Dict[str, Any],
) -> None:
    dqn = summary_df[summary_df["configuration"] == "A_standard_dqn"].iloc[0]
    ddqn = summary_df[summary_df["configuration"] == "B_double_dqn"].iloc[0]
    dqn_gen = gen_summary_df[gen_summary_df["configuration"] == "A_standard_dqn"].iloc[0]
    ddqn_gen = gen_summary_df[gen_summary_df["configuration"] == "B_double_dqn"].iloc[0]

    print("\n" + "=" * 72)
    print("DQN vs DOUBLE DQN - FINAL ALGORITHM COMPARISON")
    print("=" * 72)

    print("\n--- Performance Comparison ---")
    for label, row in [("Standard DQN", dqn), ("Double DQN", ddqn)]:
        print(
            f"\n{label}: LG={row['learning_gain_mean']:.4f} "
            f"(std={row['learning_gain_std']:.4f}, CV={row['coefficient_of_variation']:.4f}, "
            f"range={row['seed_range']:.4f}) {row['learning_gain_ci_95']}"
        )
        print(
            f"  success={row['success_rate_mean']:.4f}, "
            f"adapt={row['adaptation_accuracy_mean']:.4f}, "
            f"reward={row['mean_episode_reward_mean']:.2f}, "
            f"dropout={row['dropout_rate_mean']:.4f}"
        )

    print("\n--- Stability Comparison ---")
    print(
        f"  LG std:  DQN={dqn['learning_gain_std']:.4f}  DDQN={ddqn['learning_gain_std']:.4f}  "
        f"({verdict['stability_std_improvement_pct']:+.1f}% DDQN reduction)"
    )
    print(
        f"  CV:      DQN={dqn['coefficient_of_variation']:.4f}  DDQN={ddqn['coefficient_of_variation']:.4f}  "
        f"({verdict['stability_cv_improvement_pct']:+.1f}% DDQN reduction)"
    )
    print(f"  DQN best/worst seed: {dqn['best_seed']} ({dqn['best_seed_lg']:.4f}) / "
          f"{dqn['worst_seed']} ({dqn['worst_seed_lg']:.4f})")
    print(f"  DDQN best/worst seed: {ddqn['best_seed']} ({ddqn['best_seed_lg']:.4f}) / "
          f"{ddqn['worst_seed']} ({ddqn['worst_seed_lg']:.4f})")

    print("\n--- Action Diversity Comparison ---")
    for label, row in [("Standard DQN", dqn), ("Double DQN", ddqn)]:
        print(
            f"  {label}: entropy={row['action_entropy_mean']:.4f}, "
            f"dominant={row['dominant_action_pct_mean']:.4f}, "
            f"top2={row['top2_actions_pct_mean']:.4f}, "
            f"actions>1%={row['actions_used_above_1_percent_mean']:.1f}, "
            f"actions>5%={row['actions_used_above_5_percent_mean']:.1f}"
        )

    print("\n--- Generalization Comparison ---")
    print(
        f"  DQN:  same_pop_LG={dqn_gen['same_population_learning_gain_mean']:.4f}, "
        f"cross_pop_LG={dqn_gen['cross_population_learning_gain_mean']:.4f}, "
        f"gap={dqn_gen['generalization_gap_mean']:.4f}"
    )
    print(
        f"  DDQN: same_pop_LG={ddqn_gen['same_population_learning_gain_mean']:.4f}, "
        f"cross_pop_LG={ddqn_gen['cross_population_learning_gain_mean']:.4f}, "
        f"gap={ddqn_gen['generalization_gap_mean']:.4f}"
    )

    print("\n--- Q-Value Comparison ---")
    print(
        f"  DQN:  mean_q={dqn['mean_q_value_mean']:.4f}, max_q={dqn['max_q_value_mean']:.4f}, "
        f"var={dqn['q_value_variance_mean']:.4f}, overestimation={dqn['overestimation_bias_mean']:.4f}"
    )
    print(
        f"  DDQN: mean_q={ddqn['mean_q_value_mean']:.4f}, max_q={ddqn['max_q_value_mean']:.4f}, "
        f"var={ddqn['q_value_variance_mean']:.4f}, overestimation={ddqn['overestimation_bias_mean']:.4f}"
    )
    ddqn_reduces_overest = float(ddqn["overestimation_bias_mean"]) < float(dqn["overestimation_bias_mean"])
    print(f"  Double DQN reduces Q overestimation: {'YES' if ddqn_reduces_overest else 'NO'}")

    print("\n--- Statistical Significance (Welch t-test, Double DQN vs Standard DQN) ---")
    for _, row in stats_df.iterrows():
        sig = "yes" if row["significant_005"] else "no"
        print(
            f"  {row['metric']}: delta={row['delta_double_dqn_minus_dqn']:+.4f} "
            f"({row['pct_change']:+.1f}%), d={row['cohens_d']:.3f}, p={row['p_value']:.4f} (sig={sig})"
        )

    print("\n--- Research Questions ---")
    print(f"  1. LG improvement? {'YES' if verdict['learning_gain_improvement_pct'] > 0 else 'NO'} "
          f"({verdict['learning_gain_improvement_pct']:+.1f}%)")
    print(f"  2. Stability improvement? {'YES' if verdict['stability_std_improvement_pct'] > 0 else 'NO'} "
          f"(std {verdict['stability_std_improvement_pct']:+.1f}%)")
    print(f"  3. Reduced seed sensitivity? {'YES' if verdict['stability_cv_improvement_pct'] > 0 else 'NO'} "
          f"(CV {verdict['stability_cv_improvement_pct']:+.1f}%)")
    print(f"  4. Reduced action dominance? "
          f"{'YES' if ddqn['dominant_action_pct_mean'] < dqn['dominant_action_pct_mean'] else 'NO'}")
    print(f"  5. Improved generalization (smaller gap)? "
          f"{'YES' if ddqn_gen['generalization_gap_mean'] < dqn_gen['generalization_gap_mean'] else 'NO'}")
    print(f"  6. Reduced Q overestimation? {'YES' if ddqn_reduces_overest else 'NO'}")
    print(f"  7. Statistically significant LG? {'YES' if verdict['learning_gain_significant'] else 'NO'}")
    print(f"  8. Meets 5% improvement threshold? {'YES' if verdict['meets_5pct_threshold'] else 'NO'}")

    print("\n" + "=" * 72)
    print("WINNER ALGORITHM")
    print("=" * 72)
    print(f"\n  {verdict['winner_algorithm']} ({verdict['winner_configuration']})")
    print(f"\n  Final thesis recommendation:")
    print(f"  {verdict['thesis_recommendation']}")
    print("=" * 72)


def analyze(
    results_df: Optional[pd.DataFrame] = None,
    gen_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    results_path = OUT_DIR / "dqn_vs_double_dqn_results.csv"
    gen_path = OUT_DIR / "generalization_results.csv"

    if results_df is None:
        if not results_path.exists():
            raise FileNotFoundError(f"No results at {results_path}; run --phase run first.")
        results_df = pd.read_csv(results_path)
    if gen_df is None:
        if not gen_path.exists():
            raise FileNotFoundError(f"No generalization results at {gen_path}")
        gen_df = pd.read_csv(gen_path)

    summary_df = aggregate_summary(results_df)
    gen_summary_df = aggregate_generalization(gen_df)
    stats_df = statistical_tests(results_df)
    ranking_df = build_ranking(summary_df)
    verdict = _determine_winner(summary_df, stats_df)

    results_df.to_csv(OUT_DIR / "dqn_vs_double_dqn_results.csv", index=False)
    summary_df.to_csv(OUT_DIR / "dqn_vs_double_dqn_summary.csv", index=False)
    ranking_df.to_csv(OUT_DIR / "dqn_vs_double_dqn_ranking.csv", index=False)
    gen_df.to_csv(OUT_DIR / "generalization_results.csv", index=False)
    gen_summary_df.to_csv(OUT_DIR / "generalization_summary.csv", index=False)
    stats_df.to_csv(OUT_DIR / "statistical_tests.csv", index=False)

    _plot_comparison(summary_df)
    _plot_generalization(gen_summary_df)
    _plot_q_values(summary_df)
    _print_winner(summary_df, gen_summary_df, stats_df, verdict)

    report = {
        "study": "dqn_vs_double_dqn",
        "mdp": {
            "reward_preset": cfg.REWARD_PRESET,
            "reward_weights": cfg.REWARD_WEIGHTS,
            "action_gains": NEARLY_EQUAL_GAINS,
            "population": "mixed",
            "irt_beta": G4_GAIN["IRT_BETA"],
            "gain_ratio": G4_GAIN["ratio_label"],
        },
        "hyperparameters": THESIS_HP,
        "train_timesteps": TRAIN_TS,
        "eval_episodes": EVAL_EPS,
        "seeds": sorted(results_df["seed"].unique().tolist()),
        "configurations": ALGORITHM_CONDITIONS,
        "summary": summary_df.to_dict("records"),
        "ranking": ranking_df.to_dict("records"),
        "generalization_summary": gen_summary_df.to_dict("records"),
        "statistical_tests": stats_df.to_dict("records"),
        "verdict": verdict,
        "research_answers": {
            "double_dqn_improves_learning_gain": verdict["learning_gain_improvement_pct"] > 0,
            "double_dqn_improves_stability": verdict["stability_std_improvement_pct"] > 0,
            "double_dqn_reduces_seed_sensitivity": verdict["stability_cv_improvement_pct"] > 0,
            "double_dqn_reduces_action_dominance": float(
                summary_df[summary_df["configuration"] == "B_double_dqn"]["dominant_action_pct_mean"].iloc[0]
            )
            < float(
                summary_df[summary_df["configuration"] == "A_standard_dqn"]["dominant_action_pct_mean"].iloc[0]
            ),
            "double_dqn_improves_generalization": float(
                gen_summary_df[gen_summary_df["configuration"] == "B_double_dqn"]["generalization_gap_mean"].iloc[0]
            )
            < float(
                gen_summary_df[gen_summary_df["configuration"] == "A_standard_dqn"]["generalization_gap_mean"].iloc[0]
            ),
            "double_dqn_reduces_q_overestimation": float(
                summary_df[summary_df["configuration"] == "B_double_dqn"]["overestimation_bias_mean"].iloc[0]
            )
            < float(
                summary_df[summary_df["configuration"] == "A_standard_dqn"]["overestimation_bias_mean"].iloc[0]
            ),
            "learning_gain_statistically_significant": verdict["learning_gain_significant"],
            "meets_5pct_improvement_threshold": verdict["meets_5pct_threshold"],
        },
    }
    with open(OUT_DIR / "dqn_vs_double_dqn_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    return report


def main() -> None:
    global TRAIN_TS, EVAL_EPS, VAL_EPS, SEEDS

    parser = argparse.ArgumentParser(description="DQN vs Double DQN final algorithm comparison")
    parser.add_argument("--phase", choices=["run", "analyze", "all"], default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", type=int, default=0, help="Number of seeds (10 or 20)")
    args = parser.parse_args()

    seeds = SEEDS
    if args.seeds == 10:
        seeds = SEEDS_10
    if args.quick:
        seeds = [42, 7]
        TRAIN_TS = 5_000
        EVAL_EPS = 50
        VAL_EPS = 20
        print(f"QUICK MODE: seeds={seeds}, train={TRAIN_TS}, eval={EVAL_EPS}")

    if args.phase in ("run", "all"):
        print("=== DQN vs Double DQN (run) ===")
        print(f"Seeds: {len(seeds)}, train={TRAIN_TS}, reward={cfg.REWARD_PRESET}")
        run_study(seeds, args.resume)

    if args.phase in ("analyze", "all"):
        print("\n=== DQN vs Double DQN (analyze) ===")
        analyze()


if __name__ == "__main__":
    main()
