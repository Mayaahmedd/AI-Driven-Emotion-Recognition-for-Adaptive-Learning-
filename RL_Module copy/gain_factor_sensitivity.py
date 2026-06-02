"""
Gain-factor sensitivity study (G0-G4): PPO vs DQN across incorrect:correct ratios.

Only GAIN_CORRECT_FACTOR and GAIN_INCORRECT_FACTOR are varied via SIMULATOR_PARAMS.
Reward weights, action effects, emotions, masking, and PPO hyperparameters are fixed.

Run:
  python -m RL_Module.gain_factor_sensitivity
"""

from __future__ import annotations

import copy
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
from RL_Module.agents.ppo_agent import PPOAgent, make_masked_env
from RL_Module.environment.student_env import StudentEnv
from RL_Module.evaluation.metrics import (
    action_frequency_report,
    confidence_interval,
)
from RL_Module.mdp_definition import ACTIONS, ID_TO_ACTION

OUT_DIR = _HERE / "figures" / "gain_factor_sensitivity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 456]
TRAIN_TS = 50_000
EVAL_EPS = 300
ALGORITHMS = ["PPO", "DQN"]

GAIN_CONFIGS: Dict[str, Dict[str, float]] = {
    "G0_symmetric": {"GAIN_CORRECT_FACTOR": 1.0, "GAIN_INCORRECT_FACTOR": 1.0, "ratio_label": "1:1"},
    "G1_ratio_2_1": {"GAIN_CORRECT_FACTOR": 0.5, "GAIN_INCORRECT_FACTOR": 1.0, "ratio_label": "2:1"},
    "G2_ratio_3_1": {"GAIN_CORRECT_FACTOR": 0.3, "GAIN_INCORRECT_FACTOR": 1.0, "ratio_label": "3.3:1"},
    "G3_ratio_5_1": {"GAIN_CORRECT_FACTOR": 0.2, "GAIN_INCORRECT_FACTOR": 1.0, "ratio_label": "5:1"},
    "G4_default_10_1": {"GAIN_CORRECT_FACTOR": 0.1, "GAIN_INCORRECT_FACTOR": 1.0, "ratio_label": "10:1"},
}

ACTION_NAMES = list(ACTIONS)
GAIN_ORDER = list(GAIN_CONFIGS.keys())


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


def evaluate_with_diagnostics(
    agent,
    env: StudentEnv,
    n_episodes: int,
    seed: int,
    algorithm: str,
) -> Dict[str, Any]:
    """Evaluate and collect step-level affect, delta-k, P(correct), action counts."""
    cfg.set_all_seeds(seed)
    episode_results: List[Dict[str, Any]] = []
    step_rows: List[Dict[str, Any]] = []

    for ep in range(1, n_episodes + 1):
        obs, info = env.reset(seed=seed + ep)
        mask = info["action_masks"]
        total_reward = 0.0
        start_k = env._state.knowledge
        action_counts = {i: 0 for i in range(8)}
        ep_adapt_hits = 0
        ep_adapt_total = 0
        prev_k = start_k
        done = False
        step = 0

        while not done:
            emotion_id = env._state.emotion_id
            action = agent.predict(obs, mask)
            obs, reward, terminated, truncated, info = env.step(action)
            if hasattr(agent, "set_last_reward"):
                agent.set_last_reward(reward)

            correct = bool(info.get("last_answer_correct", True))
            k_now = env._state.knowledge
            dk = k_now - prev_k
            step_rows.append({
                "episode": ep,
                "step": step,
                "knowledge": k_now,
                "delta_k": dk,
                "correct": int(correct),
                "engagement": env._state.engagement,
                "frustration": env._state.frustration,
                "confusion": env._state.confusion,
                "boredom": env._state.boredom,
                "action_id": action,
                "action_name": info.get("action_name", ID_TO_ACTION[action]),
                "reward": reward,
            })
            prev_k = k_now

            from RL_Module.evaluation.metrics import _adaptation_match

            ep_adapt_total += 1
            if _adaptation_match(emotion_id, action):
                ep_adapt_hits += 1

            total_reward += reward
            action_counts[action] += 1
            step += 1
            mask = info["action_masks"]
            done = terminated or truncated

        final = env._state
        from RL_Module.evaluation.metrics import _is_success, normalized_knowledge_gain

        lg = final.knowledge - start_k
        lg_norm = normalized_knowledge_gain(start_k, final.knowledge)
        success = _is_success(final.knowledge, final.frustration, final.confusion)
        episode_results.append({
            "episode": ep,
            "total_reward": total_reward,
            "learning_gain": lg,
            "learning_gain_normalized": lg_norm,
            "success": int(success),
            "final_knowledge": final.knowledge,
            "action_counts": action_counts,
            "adaptation_accuracy": ep_adapt_hits / max(ep_adapt_total, 1),
        })

    sdf = pd.DataFrame(step_rows)
    mean_dk = float(sdf["delta_k"].mean()) if len(sdf) else 0.0
    p_correct = float(sdf["correct"].mean()) if len(sdf) else 0.0

    freq_report = action_frequency_report(episode_results, print_table=False)
    action_freq = {k: round(v, 4) for k, v in freq_report["frequencies"].items()}

    return {
        "mean_episode_reward": float(np.mean([r["total_reward"] for r in episode_results])),
        "success_rate": float(np.mean([r["success"] for r in episode_results])),
        "learning_gain": float(np.mean([r["learning_gain_normalized"] for r in episode_results])),
        "final_knowledge": float(np.mean([r["final_knowledge"] for r in episode_results])),
        "adaptation_accuracy": float(np.mean([r["adaptation_accuracy"] for r in episode_results])),
        "mean_delta_k_per_step": mean_dk,
        "p_correct": p_correct,
        "mean_engagement": float(sdf["engagement"].mean()) if len(sdf) else 0.0,
        "mean_frustration": float(sdf["frustration"].mean()) if len(sdf) else 0.0,
        "mean_confusion": float(sdf["confusion"].mean()) if len(sdf) else 0.0,
        "mean_boredom": float(sdf["boredom"].mean()) if len(sdf) else 0.0,
        "action_frequencies": action_freq,
        "action_diversity_shannon": _shannon_entropy(freq_report["frequencies"]),
        "dominance_detected": freq_report["dominance_detected"],
        "dominant_actions": freq_report["dominant_actions"],
        "episode_results": episode_results,
        "n_steps": len(sdf),
    }


def train_and_eval(
    algorithm: str,
    seed: int,
    gain_id: str,
    gain_params: Dict[str, float],
) -> Dict[str, Any]:
    snap = _patch_gain(gain_params)
    cfg.USE_SENSITIVITY_WEIGHTS = False
    tag = f"{gain_id}_{algorithm}_s{seed}"
    t0 = time.time()
    try:
        cfg.set_all_seeds(seed)
        if algorithm == "PPO":
            env = make_masked_env(seed=seed, algo_tag=tag)
            agent = PPOAgent()
            agent.train(env, TRAIN_TS, seed)
            env.close()
            eval_env = StudentEnv(population_seed=seed)
            eval_env.set_algorithm_name(f"PPO_{gain_id}")
        else:
            env = make_env(seed=seed, algo_tag=tag)
            agent = DQNAgent()
            agent.train(env, TRAIN_TS, seed)
            env.close()
            eval_env = StudentEnv(population_seed=seed)
            eval_env.set_algorithm_name(f"DQN_{gain_id}")

        result = evaluate_with_diagnostics(
            agent, eval_env, n_episodes=EVAL_EPS, seed=seed, algorithm=tag
        )
        eval_env.close()
    finally:
        _restore_gain(snap)

    elapsed = time.time() - t0
    row = {
        "gain_id": gain_id,
        "ratio_label": gain_params["ratio_label"],
        "gain_correct": gain_params["GAIN_CORRECT_FACTOR"],
        "gain_incorrect": gain_params["GAIN_INCORRECT_FACTOR"],
        "algorithm": algorithm,
        "seed": seed,
        "train_timesteps": TRAIN_TS,
        "eval_episodes": EVAL_EPS,
        "elapsed_seconds": round(elapsed, 1),
        **{k: result[k] for k in [
            "mean_episode_reward", "success_rate", "learning_gain", "final_knowledge",
            "adaptation_accuracy", "mean_delta_k_per_step", "p_correct",
            "mean_engagement", "mean_frustration", "mean_confusion", "mean_boredom",
            "action_diversity_shannon", "dominance_detected",
        ]},
        "action_frequencies": result["action_frequencies"],
        "dominant_actions": result["dominant_actions"],
    }
    print(
        f"  {tag}: reward={row['mean_episode_reward']:.3f} "
        f"success={row['success_rate']:.3f} lg={row['learning_gain']:.3f} "
        f"dk={row['mean_delta_k_per_step']:.4f} P(c)={row['p_correct']:.3f} ({elapsed:.0f}s)"
    )
    return row


def aggregate_runs(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    metrics = [
        "mean_episode_reward", "success_rate", "learning_gain", "final_knowledge",
        "adaptation_accuracy", "mean_delta_k_per_step", "p_correct",
        "mean_engagement", "mean_frustration", "mean_confusion", "mean_boredom",
        "action_diversity_shannon",
    ]
    out: Dict[str, Any] = {}
    for m in metrics:
        vals = [r[m] for r in runs]
        mean, lo, hi = confidence_interval(vals)
        out[f"{m}_mean"] = round(mean, 4)
        out[f"{m}_std"] = round(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, 4)
        out[f"{m}_ci_lo"] = round(lo, 4)
        out[f"{m}_ci_hi"] = round(hi, 4)
    # Mean action frequencies across seeds
    freq_acc: Dict[str, List[float]] = {a: [] for a in ACTION_NAMES}
    for r in runs:
        for a in ACTION_NAMES:
            freq_acc[a].append(r["action_frequencies"].get(a, 0.0))
    out["action_frequencies_mean"] = {
        a: round(float(np.mean(freq_acc[a])), 4) for a in ACTION_NAMES
    }
    out["dominance_any_seed"] = any(r["dominance_detected"] for r in runs)
    return out


def gap_analysis(agg: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for gain_id in GAIN_ORDER:
        ppo = agg[(gain_id, "PPO")]
        dqn = agg[(gain_id, "DQN")]
        rows.append({
            "gain_id": gain_id,
            "ratio_label": ppo["ratio_label"],
            "gain_correct": ppo["gain_correct"],
            "ppo_success": ppo["success_rate_mean"],
            "dqn_success": dqn["success_rate_mean"],
            "gap_success": round(dqn["success_rate_mean"] - ppo["success_rate_mean"], 4),
            "ppo_learning_gain": ppo["learning_gain_mean"],
            "dqn_learning_gain": dqn["learning_gain_mean"],
            "gap_learning_gain": round(dqn["learning_gain_mean"] - ppo["learning_gain_mean"], 4),
            "ppo_final_k": ppo["final_knowledge_mean"],
            "dqn_final_k": dqn["final_knowledge_mean"],
            "gap_final_knowledge": round(dqn["final_knowledge_mean"] - ppo["final_knowledge_mean"], 4),
            "ppo_mean_dk": ppo["mean_delta_k_per_step_mean"],
            "dqn_mean_dk": dqn["mean_delta_k_per_step_mean"],
            "ppo_p_correct": ppo["p_correct_mean"],
            "dqn_p_correct": dqn["p_correct_mean"],
            "ppo_reward": ppo["mean_episode_reward_mean"],
            "dqn_reward": dqn["mean_episode_reward_mean"],
        })
    return rows


def test_hypotheses(gaps: List[Dict[str, Any]], per_run: pd.DataFrame) -> Dict[str, Any]:
    """Trend tests: asymmetry index vs outcomes (lower gc = higher asymmetry)."""
    asym = [GAIN_CONFIGS[g]["GAIN_CORRECT_FACTOR"] for g in GAIN_ORDER]

    def _spearman(x: List[float], y: List[float]) -> Tuple[float, float]:
        if len(set(x)) < 2:
            return float("nan"), float("nan")
        rho, p = stats.spearmanr(x, y)
        return float(rho), float(p)

    gap_success = [g["gap_success"] for g in gaps]
    ppo_success = [g["ppo_success"] for g in gaps]

    rho_gap, p_gap = _spearman(asym, gap_success)
    rho_ppo, p_ppo = _spearman(asym, ppo_success)

    # Per-run: PPO success vs gain_correct across all configs/seeds
    ppo_df = per_run[per_run["algorithm"] == "PPO"]
    if len(ppo_df["gain_correct"].unique()) > 1:
        rho_ppo_all, p_ppo_all = stats.spearmanr(
            ppo_df["gain_correct"], ppo_df["success_rate"]
        )
    else:
        rho_ppo_all, p_ppo_all = float("nan"), float("nan")

    g0_gap = next(g["gap_success"] for g in gaps if g["gain_id"] == "G0_symmetric")
    g4_gap = next(g["gap_success"] for g in gaps if g["gain_id"] == "G4_default_10_1")

    return {
        "H1_ppo_success_increases_as_asymmetry_decreases": {
            "spearman_gc_vs_ppo_success_across_configs": rho_ppo,
            "p_value_configs_only": p_ppo,
            "supported_at_alpha_05": bool(p_ppo < 0.05 and rho_ppo > 0) if not np.isnan(p_ppo) else None,
            "spearman_all_ppo_runs": float(rho_ppo_all),
            "p_all_ppo_runs": float(p_ppo_all),
        },
        "H2_gap_shrinks_as_asymmetry_decreases": {
            "spearman_gc_vs_gap_success": rho_gap,
            "p_value": p_gap,
            "supported_at_alpha_05": bool(p_gap < 0.05 and rho_gap < 0) if not np.isnan(p_gap) else None,
            "gap_at_G0": g0_gap,
            "gap_at_G4": g4_gap,
            "gap_reduction_G4_to_G0": round(g4_gap - g0_gap, 4),
        },
        "H3_primary_cause_if_gap_near_zero_at_G0_G2": {
            "gap_at_G0": g0_gap,
            "gap_at_G2": next(g["gap_success"] for g in gaps if g["gain_id"] == "G2_ratio_3_1"),
            "interpretation_pending": "See gap table; H3 if |gap| < 0.05 at symmetric configs",
        },
        "H4_other_mechanism_if_gap_large_at_G0": {
            "gap_at_G0": g0_gap,
            "large_gap_threshold": 0.10,
            "supported": bool(abs(g0_gap) >= 0.10),
        },
    }


def plot_results(gaps: List[Dict[str, Any]], agg: Dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    x_labels = [g["ratio_label"] for g in gaps]
    x = np.arange(len(x_labels))

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    ppo_s = [g["ppo_success"] for g in gaps]
    dqn_s = [g["dqn_success"] for g in gaps]
    w = 0.35
    ax.bar(x - w / 2, ppo_s, w, label="PPO", color="#4C72B0")
    ax.bar(x + w / 2, dqn_s, w, label="DQN", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Success rate")
    ax.set_title("Success vs gain ratio")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    ax = axes[0, 1]
    gaps_y = [g["gap_success"] for g in gaps]
    ax.plot(x, gaps_y, "o-", color="#55A868", linewidth=2, markersize=8)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("DQN success - PPO success")
    ax.set_title("PPO-DQN success gap")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(x, [g["ppo_mean_dk"] for g in gaps], "o-", label="PPO mean dk/step")
    ax.plot(x, [g["dqn_mean_dk"] for g in gaps], "s-", label="DQN mean dk/step")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Mean delta knowledge")
    ax.set_title("Learning rate per step")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(x, [g["ppo_p_correct"] for g in gaps], "o-", label="PPO P(correct)")
    ax.plot(x, [g["dqn_p_correct"] for g in gaps], "s-", label="DQN P(correct)")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("P(correct)")
    ax.set_title("Empirical correctness rate")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "gain_factor_overview.png", dpi=150)
    plt.close(fig)


def main() -> None:
    print("=== Gain-factor sensitivity (G0-G4 x PPO/DQN x seeds) ===")
    print(f"Seeds={SEEDS} train={TRAIN_TS} eval={EVAL_EPS}")
    cfg.USE_SENSITIVITY_WEIGHTS = False

    per_run: List[Dict[str, Any]] = []
    total = len(GAIN_CONFIGS) * len(ALGORITHMS) * len(SEEDS)
    n_done = 0

    for gain_id, gain_params in GAIN_CONFIGS.items():
        print(f"\n--- {gain_id} (ratio {gain_params['ratio_label']}) ---")
        for algorithm in ALGORITHMS:
            for seed in SEEDS:
                n_done += 1
                print(f"[{n_done}/{total}] {algorithm} seed={seed}")
                row = train_and_eval(algorithm, seed, gain_id, gain_params)
                per_run.append(row)

    df = pd.DataFrame(per_run)
    df.to_csv(OUT_DIR / "gain_factor_per_run.csv", index=False)

    agg: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for gain_id in GAIN_ORDER:
        for algorithm in ALGORITHMS:
            runs = [r for r in per_run if r["gain_id"] == gain_id and r["algorithm"] == algorithm]
            meta = runs[0]
            a = aggregate_runs(runs)
            a["gain_id"] = gain_id
            a["ratio_label"] = meta["ratio_label"]
            a["gain_correct"] = meta["gain_correct"]
            a["gain_incorrect"] = meta["gain_incorrect"]
            a["algorithm"] = algorithm
            agg[(gain_id, algorithm)] = a

    gaps = gap_analysis(agg)
    hypotheses = test_hypotheses(gaps, df)

    report = {
        "study": "gain_factor_sensitivity",
        "seeds": SEEDS,
        "train_timesteps": TRAIN_TS,
        "eval_episodes": EVAL_EPS,
        "gain_configs": {
            k: {kk: vv for kk, vv in v.items() if kk != "ratio_label"}
            | {"ratio_label": v["ratio_label"]}
            for k, v in GAIN_CONFIGS.items()
        },
        "aggregated": {f"{g}|{a}": agg[(g, a)] for g in GAIN_ORDER for a in ALGORITHMS},
        "ppo_dqn_gaps": gaps,
        "hypothesis_tests": hypotheses,
        "per_run_count": len(per_run),
    }

    with open(OUT_DIR / "gain_factor_sensitivity_report.json", "w") as f:
        json.dump(report, f, indent=2)

    plot_results(gaps, agg)

    print("\n=== PPO-DQN gap by gain config ===")
    for g in gaps:
        print(
            f"  {g['gain_id']:18s} ratio={g['ratio_label']:6s} "
            f"gap_success={g['gap_success']:+.3f} gap_lg={g['gap_learning_gain']:+.3f} "
            f"PPO_succ={g['ppo_success']:.3f} DQN_succ={g['dqn_success']:.3f}"
        )
    print(f"\nReport: {OUT_DIR / 'gain_factor_sensitivity_report.json'}")


if __name__ == "__main__":
    main()
