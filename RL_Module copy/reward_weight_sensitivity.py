"""
Reward Weight Sensitivity Analysis (PPO only)
=============================================
Tests whether wk/we settings cause PPO reward-mastery misalignment.

Configurations (wf=0.15, wb=0.06, wc=0.09 fixed):
  A: wk=0.50, we=0.20  (baseline)
  B: wk=0.60, we=0.20
  C: wk=0.70, we=0.20
  D: wk=0.50, we=0.15
  E: wk=0.60, we=0.15
  F: wk=0.70, we=0.15

Run:
  python -m RL_Module.reward_weight_sensitivity
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
from RL_Module.agents.ppo_agent import PPOAgent, make_masked_env
from RL_Module.evaluation.metrics import (
    action_frequency_report,
    evaluate,
)
from RL_Module.mdp_definition import ACTIONS, ID_TO_ACTION

OUT_DIR = _HERE / "figures" / "reward_weight_sensitivity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
TRAIN_TS = 50_000
EVAL_EPS = 300

WF, WB, WC = 0.15, 0.06, 0.09

CONFIGS: Dict[str, Dict[str, float]] = {
    "A_baseline": {"wk": 0.50, "we": 0.20},
    "B_wk060_we020": {"wk": 0.60, "we": 0.20},
    "C_wk070_we020": {"wk": 0.70, "we": 0.20},
    "D_wk050_we015": {"wk": 0.50, "we": 0.15},
    "E_wk060_we015": {"wk": 0.60, "we": 0.15},
    "F_wk070_we015": {"wk": 0.70, "we": 0.15},
}

STATE_CONDITIONS = {
    "high_confusion":   lambda r: r["confusion"] > 0.28,
    "high_frustration": lambda r: r["frustration"] > 0.28,
    "high_boredom":     lambda r: r["boredom"] > 0.28,
    "flow_state":       lambda r: (r["engagement"] > 0.58)
                                  & (r["frustration"] < 0.25)
                                  & (r["confusion"] < 0.25)
                                  & (r["boredom"] < 0.25)
                                  & (r["knowledge"] > 0.25),
}

ACTION_NAMES = list(ACTIONS)


def _patch_weights(wk: float, we: float) -> Dict[str, Any]:
    """Enable sensitivity override; return snapshot for restore."""
    snapshot = {
        "USE_SENSITIVITY_WEIGHTS": cfg.USE_SENSITIVITY_WEIGHTS,
        "SENSITIVITY_WEIGHTS": copy.deepcopy(cfg.SENSITIVITY_WEIGHTS),
    }
    cfg.SENSITIVITY_WEIGHTS = {
        "wk": wk, "we": we, "wf": WF, "wb": WB, "wc": WC,
    }
    cfg.USE_SENSITIVITY_WEIGHTS = True
    return snapshot


def _restore_weights(snapshot: Dict[str, Any]) -> None:
    cfg.USE_SENSITIVITY_WEIGHTS = snapshot["USE_SENSITIVITY_WEIGHTS"]
    cfg.SENSITIVITY_WEIGHTS = snapshot["SENSITIVITY_WEIGHTS"]


def _shannon_entropy(freq: np.ndarray) -> float:
    p = freq[freq > 0]
    return float(-np.sum(p * np.log(p))) if len(p) else 0.0


def decompose_steps(df: pd.DataFrame, wk: float, we: float) -> pd.DataFrame:
    df = df.sort_values(["episode", "step"]).reset_index(drop=True)
    df["prev_k"] = (
        df.groupby("episode")["knowledge"]
        .shift(1)
        .fillna(df.groupby("episode")["knowledge"].transform("first"))
    )
    denom = (1.0 - df["prev_k"] + 1e-8).clip(lower=1e-8)
    df["delta_k_norm"] = ((df["knowledge"] - df["prev_k"]) / denom).clip(-1, 1)
    df["comp_k"] = wk * df["delta_k_norm"]
    df["comp_e"] = we * df["engagement"]
    df["comp_f"] = -WF * df["frustration"]
    df["comp_b"] = -WB * df["boredom"]
    df["comp_c"] = -WC * df["confusion"]
    return df


def episode_analysis(df: pd.DataFrame, wk: float, we: float) -> Tuple[pd.DataFrame, Dict]:
    df = decompose_steps(df, wk, we)
    ep = df.groupby("episode").agg(
        total_reward=("reward", "sum"),
        comp_k=("comp_k", "sum"),
        comp_e=("comp_e", "sum"),
        comp_f=("comp_f", "sum"),
        comp_b=("comp_b", "sum"),
        comp_c=("comp_c", "sum"),
        final_k=("knowledge", "last"),
        start_k=("knowledge", "first"),
        final_f=("frustration", "last"),
        final_c=("confusion", "last"),
    ).reset_index()
    ep["learning_gain"] = ep["final_k"] - ep["start_k"]
    ep["lg_norm"] = ep["learning_gain"] / (1 - ep["start_k"] + 1e-8)
    ep["success"] = (
        (ep["final_k"] > 0.8) & (ep["final_f"] < 0.3) & (ep["final_c"] < 0.4)
    ).astype(int)

    comp_cols = ["comp_k", "comp_e", "comp_f", "comp_b", "comp_c"]
    means = ep[comp_cols].mean()
    abs_sum = means.abs().sum()
    pct = {c.replace("comp_", ""): float(means[c] / abs_sum * 100) if abs_sum else 0
           for c in comp_cols}

    def _corr(x: str, y: str) -> float:
        v = ep[[x, y]].dropna()
        if len(v) < 5:
            return float("nan")
        r, _ = stats.pearsonr(v[x], v[y])
        return float(r)

    correlations = {
        "total_reward_vs_success": _corr("total_reward", "success"),
        "total_reward_vs_lg_norm": _corr("total_reward", "lg_norm"),
        "comp_k_vs_success": _corr("comp_k", "success"),
        "comp_k_vs_lg_norm": _corr("comp_k", "lg_norm"),
        "comp_e_vs_success": _corr("comp_e", "success"),
    }

    return ep, {"component_pct": pct, "correlations": correlations}


def state_action_probs(df: pd.DataFrame) -> Dict[str, Any]:
    out = {}
    for cond, fn in STATE_CONDITIONS.items():
        sub = df[fn(df)]
        n = len(sub)
        if n < 20:
            out[cond] = {"n": n, "skipped": True}
            continue
        probs = (
            sub["action_name"]
            .value_counts(normalize=True)
            .reindex(ACTION_NAMES, fill_value=0)
            .round(4)
            .to_dict()
        )
        top = max(probs.items(), key=lambda x: x[1])
        out[cond] = {"n": n, "probs": probs, "top_action": top[0], "top_p": top[1]}
    return out


def chi2_context_sensitive(df: pd.DataFrame) -> Dict[str, Any]:
    """Test action independence for high_engagement vs rest."""
    fn = lambda r: (r["engagement"] > 0.58) & (r["frustration"] < 0.25)
    mask = fn(df)
    in_ids = df[mask]["action_id"].values
    out_ids = df[~mask]["action_id"].values
    if len(in_ids) < 30:
        return {"significant": False, "skipped": True}
    c_in = np.bincount(in_ids, minlength=8).astype(float)
    c_out = np.bincount(out_ids, minlength=8).astype(float)
    ctbl = np.vstack([c_in, c_out])
    ctbl = ctbl[:, ctbl.sum(axis=0) > 0]
    if ctbl.shape[1] < 2:
        return {"significant": False, "skipped": True}
    chi2, p, _, _ = stats.chi2_contingency(ctbl)
    return {"chi2": float(chi2), "p": float(p), "significant": bool(p < 0.05)}


def run_config(label: str, wk: float, we: float) -> Dict[str, Any]:
    print(f"\n  [{label}] wk={wk}, we={we} ...", end=" ", flush=True)
    t0 = time.time()
    snap = _patch_weights(wk, we)

    cfg.set_all_seeds(SEED)
    env = make_masked_env(seed=SEED, algo_tag=f"rw_sens_{label}")
    agent = PPOAgent()
    agent.train(env, TRAIN_TS, SEED)
    train_s = time.time() - t0

    # Evaluate (fresh env, same weights still patched)
    eval_env = make_masked_env(seed=SEED, algo_tag=f"rw_sens_eval_{label}").unwrapped
    result = evaluate(
        agent,
        eval_env,
        n_episodes=EVAL_EPS,
        seed=SEED,
        algorithm=f"PPO_{label}",
        log_steps=True,
    )

    # Load steps from logger output
    steps_path = cfg.LOGS_DIR / f"steps_PPO_{label}_seed{SEED}.csv"
    df = pd.read_csv(steps_path)
    df.columns = [c.lower().strip() for c in df.columns]
    if "action_name" not in df.columns:
        df["action_name"] = df["action_id"].map(ID_TO_ACTION)

    ep_df, analysis = episode_analysis(df, wk, we)
    freq_report = action_frequency_report(result["episode_results"], print_table=False)
    freqs = freq_report["frequencies"]
    action_freq = {k: round(v, 4) for k, v in freqs.items()}
    diversity = _shannon_entropy(np.array(list(freqs.values())))

    sa = state_action_probs(df)
    chi2 = chi2_context_sensitive(df)

    _restore_weights(snap)
    elapsed = time.time() - t0
    print(f"train={train_s:.0f}s  reward={result['mean_episode_reward']:.3f}  "
          f"success={result['success_rate']:.3f}  lg={result['learning_gain']:.3f}  ({elapsed:.0f}s total)")

    return {
        "label": label,
        "wk": wk,
        "we": we,
        "wf": WF, "wb": WB, "wc": WC,
        "mean_reward": float(result["mean_episode_reward"]),
        "success_rate": float(result["success_rate"]),
        "learning_gain": float(result["learning_gain"]),
        "adaptation_accuracy": float(result["adaptation_accuracy"]),
        "dropout_rate": float(result["dropout_rate"]),
        "action_diversity_shannon": diversity,
        "dominance_detected": freq_report["dominance_detected"],
        "dominant_actions": freq_report["dominant_actions"],
        "action_frequencies": action_freq,
        "component_pct": analysis["component_pct"],
        "correlations": analysis["correlations"],
        "state_action": sa,
        "context_sensitive": chi2.get("significant", False),
        "chi2_engagement": chi2,
        "train_seconds": train_s,
    }


def compare_to_baseline(all_results: List[Dict]) -> List[Dict]:
    base = next(r for r in all_results if r["label"] == "A_baseline")
    rows = []
    for r in all_results:
        row = {"label": r["label"], "wk": r["wk"], "we": r["we"]}
        for m in ["mean_reward", "success_rate", "learning_gain", "adaptation_accuracy"]:
            delta = r[m] - base[m]
            pct = (delta / base[m] * 100) if base[m] != 0 else float("nan")
            row[f"{m}_delta"] = round(delta, 4)
            row[f"{m}_pct_change"] = round(pct, 2)
        row["reward_success_corr"] = r["correlations"]["total_reward_vs_success"]
        row["k_success_corr"] = r["correlations"]["comp_k_vs_success"]
        row["k_pct"] = r["component_pct"]["k"]
        row["e_pct"] = r["component_pct"]["e"]
        rows.append(row)
    return rows


def score_balance(r: Dict) -> float:
    """Composite: mastery + alignment + adaptation - dominance penalty."""
    align = r["correlations"]["total_reward_vs_success"]
    if np.isnan(align):
        align = 0.0
    dom_pen = 0.2 if r["dominance_detected"] else 0.0
    ctx_bonus = 0.05 if r["context_sensitive"] else 0.0
    return (
        0.35 * r["success_rate"]
        + 0.25 * r["learning_gain"]
        + 0.20 * align
        + 0.15 * r["adaptation_accuracy"]
        + 0.05 * r["action_diversity_shannon"]
        + ctx_bonus
        - dom_pen
    )


def plot_results(all_results: List[Dict], comparison: List[Dict]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    labels = [r["label"].replace("_", "\n") for r in all_results]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # Success vs reward
    ax = axes[0, 0]
    rewards = [r["mean_reward"] for r in all_results]
    successes = [r["success_rate"] for r in all_results]
    ax.scatter(rewards, successes, s=120, c=range(len(labels)), cmap="viridis")
    for i, lb in enumerate([r["label"] for r in all_results]):
        ax.annotate(lb.split("_")[0], (rewards[i], successes[i]), fontsize=8)
    ax.set_xlabel("Mean Episode Reward")
    ax.set_ylabel("Success Rate")
    ax.set_title("Reward vs Mastery Trade-off")
    ax.grid(alpha=0.3)

    # Component percentages
    ax = axes[0, 1]
    k_pcts = [r["component_pct"]["k"] for r in all_results]
    e_pcts = [r["component_pct"]["e"] for r in all_results]
    w = 0.35
    ax.bar(x - w/2, k_pcts, w, label="Knowledge %", color="#4C72B0")
    ax.bar(x + w/2, e_pcts, w, label="Engagement %", color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels([r["label"].split("_")[0] for r in all_results])
    ax.set_ylabel("% of |signal|")
    ax.set_title("Effective Reward Component Share")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Correlations
    ax = axes[0, 2]
    r_succ = [r["correlations"]["total_reward_vs_success"] for r in all_results]
    k_succ = [r["correlations"]["comp_k_vs_success"] for r in all_results]
    ax.bar(x - w/2, r_succ, w, label="total_r vs success", color="#C44E52")
    ax.bar(x + w/2, k_succ, w, label="k_comp vs success", color="#4C72B0")
    ax.set_xticks(x)
    ax.set_xticklabels([r["label"].split("_")[0] for r in all_results])
    ax.set_ylabel("Pearson r")
    ax.set_title("Reward Alignment Correlations")
    ax.axhline(0, color="black", lw=0.5)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Metrics bars
    for ax, metric, title in zip(
        axes[1],
        ["success_rate", "learning_gain", "adaptation_accuracy"],
        ["Success Rate", "Learning Gain (norm)", "Adaptation Accuracy"],
    ):
        vals = [r[metric] for r in all_results]
        colors = ["#4C72B0" if r["label"] == "A_baseline" else "#DD8452" for r in all_results]
        ax.bar(x, vals, color=colors, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([r["label"].split("_")[0] for r in all_results])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle(f"PPO Reward Weight Sensitivity (seed={SEED}, {TRAIN_TS:,} ts, {EVAL_EPS} eval ep)")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "sensitivity_overview.png", dpi=150)
    plt.close(fig)
    print(f"  [saved] sensitivity_overview.png")


def print_comparison_table(comparison: List[Dict]) -> None:
    print("\n  COMPARISON vs BASELINE (A):")
    print(f"  {'Config':16s}  {'dReward':>8s}  {'dSuccess':>9s}  {'dGain':>8s}  "
          f"{'dAdapt':>8s}  {'r(r,s)':>7s}  {'k%':>6s}  {'e%':>6s}")
    print("  " + "-" * 78)
    for row in comparison:
        print(
            f"  {row['label']:16s}  {row['mean_reward_delta']:+8.3f}  "
            f"{row['success_rate_delta']:+9.3f}  {row['learning_gain_delta']:+8.3f}  "
            f"{row['adaptation_accuracy_delta']:+8.3f}  "
            f"{row['reward_success_corr']:7.3f}  {row['k_pct']:6.1f}  {row['e_pct']:6.1f}"
        )


def main() -> None:
    print("=" * 70)
    print("  REWARD WEIGHT SENSITIVITY ANALYSIS")
    print(f"  seed={SEED}, timesteps={TRAIN_TS:,}, eval_episodes={EVAL_EPS}")
    print(f"  fixed: wf={WF}, wb={WB}, wc={WC}")
    print("=" * 70)

    all_results: List[Dict] = []
    for label, weights in CONFIGS.items():
        r = run_config(label, weights["wk"], weights["we"])
        all_results.append(r)

    comparison = compare_to_baseline(all_results)
    print_comparison_table(comparison)

    # Best configs
    best_mastery = max(all_results, key=lambda r: r["success_rate"])
    best_align = max(all_results, key=lambda r: r["correlations"]["total_reward_vs_success"])
    for r in all_results:
        r["balance_score"] = score_balance(r)
    best_balance = max(all_results, key=lambda r: r["balance_score"])

    print("\n  BEST BY CRITERION:")
    print(f"    Mastery:          {best_mastery['label']}  "
          f"(success={best_mastery['success_rate']:.3f}, gain={best_mastery['learning_gain']:.3f})")
    print(f"    Reward alignment: {best_align['label']}  "
          f"(r(total,success)={best_align['correlations']['total_reward_vs_success']:.3f})")
    print(f"    Overall balance:  {best_balance['label']}  "
          f"(score={best_balance['balance_score']:.3f})")

    # Context sensitivity check
    print("\n  CONTEXT SENSITIVITY (chi2 p<0.05 on engagement condition):")
    for r in all_results:
        sig = "YES" if r["context_sensitive"] else "NO"
        print(f"    {r['label']:16s}  {sig}")

    plot_results(all_results, comparison)

    report = {
        "seed": SEED,
        "train_timesteps": TRAIN_TS,
        "eval_episodes": EVAL_EPS,
        "fixed_weights": {"wf": WF, "wb": WB, "wc": WC},
        "configurations": CONFIGS,
        "results": all_results,
        "comparison_vs_baseline": comparison,
        "best_mastery": best_mastery["label"],
        "best_alignment": best_align["label"],
        "best_balance": best_balance["label"],
    }
    out_json = OUT_DIR / "reward_weight_sensitivity_report.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  [saved] {out_json}")
    print("=" * 70)


if __name__ == "__main__":
    main()
