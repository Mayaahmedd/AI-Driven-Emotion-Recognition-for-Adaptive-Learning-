"""
Final Double DQN confirmation vs frozen thesis DQN baseline (20 seeds).

Compares:
  A) Standard DQN - frozen baseline from final_dqn_model_study (no retrain)
  B) Double DQN - same MDP, hyperparameters, checkpoint protocol

Focused metrics: learning gain, CV, success rate, adaptation accuracy.

Usage (from repo root):
  export PYTHONPATH="$(pwd)"
  python -m RL_Module.final_double_dqn_confirmation --phase all
  python -m RL_Module.final_double_dqn_confirmation --phase run --resume
  python -m RL_Module.final_double_dqn_confirmation --phase analyze
  python -m RL_Module.final_double_dqn_confirmation --phase all --quick
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
from RL_Module.agents.double_dqn_agent import DoubleDQNAgent
from RL_Module.agents.dqn_agent import make_env
from RL_Module.evaluation.metrics import confidence_interval
from RL_Module.final_dqn_model_study import (
    CHECKPOINT_STEPS,
    EVAL_EPS,
    FINAL_DQN_HP,
    G4_GAIN,
    METRICS,
    OBS_ABLATION,
    SEEDS_20,
    TRAIN_TS,
    VAL_EPS,
    _patch_simulator,
    _restore_simulator,
    _step_from_path,
    full_eval,
    load_baseline_thesis,
    quick_eval_lg,
)

OUT_DIR = _HERE / "figures" / "final_double_dqn_confirmation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STANDARD_KEY = "standard_dqn"
DOUBLE_KEY = "double_dqn"
FOCUS_METRICS = ["learning_gain", "success_rate", "adaptation_accuracy"]


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


def _baseline_to_comparison_rows(baseline_df: pd.DataFrame) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for _, row in baseline_df.iterrows():
        rows.append({
            "configuration": STANDARD_KEY,
            "configuration_label": "Standard DQN (thesis baseline)",
            "algorithm": "DQN",
            "seed": int(row["seed"]),
            "train_timesteps": int(row.get("train_timesteps", TRAIN_TS)),
            "eval_episodes": int(row.get("eval_episodes", EVAL_EPS)),
            "best_checkpoint_step": int(row["best_checkpoint_step"]),
            "best_checkpoint_path": row.get("best_checkpoint_path", ""),
            "best_val_learning_gain": float(row["best_val_learning_gain"]),
            "elapsed_seconds": float(row.get("elapsed_seconds", 0.0)),
            "source": "baseline_per_run_20.csv",
            **{m: float(row[f"best_{m}"]) for m in METRICS},
        })
    return rows


def train_double_dqn_seed(seed: int) -> Dict[str, Any]:
    snap = _patch_simulator(tier_a=False)
    tag = f"final_ddqn_confirm_s{seed}"
    t0 = time.time()

    try:
        cfg.set_all_seeds(seed)
        train_env = make_env(
            seed=seed,
            obs_ablation=OBS_ABLATION,
            emotion_dynamics="full",
            algo_tag=tag,
        )
        agent = DoubleDQNAgent()
        ckpt_dir = cfg.MODELS_DIR / tag
        paths = agent.train_with_checkpoints(
            train_env,
            TRAIN_TS,
            seed,
            hyperparams=FINAL_DQN_HP,
            checkpoint_dir=str(ckpt_dir),
            checkpoint_freq=min(10_000, TRAIN_TS),
        )
        train_env.close()

        step_paths: Dict[int, str] = {}
        for p in paths:
            for step in CHECKPOINT_STEPS:
                if f"_{step}_steps" in p:
                    step_paths[step] = p

        valid_paths = [step_paths[s] for s in CHECKPOINT_STEPS if s in step_paths]
        if not valid_paths:
            valid_paths = list(paths)

        def _eval_fn(a: DoubleDQNAgent, s: int) -> float:
            return quick_eval_lg(a, s, VAL_EPS)  # type: ignore[arg-type]

        best_path, best_val = agent.select_best_checkpoint(valid_paths, _eval_fn, seed)
        agent.load_checkpoint(best_path)
        metrics = full_eval(agent, seed, EVAL_EPS)  # type: ignore[arg-type]
    finally:
        _restore_simulator(snap)

    elapsed = time.time() - t0
    row = {
        "configuration": DOUBLE_KEY,
        "configuration_label": "Double DQN (confirmation)",
        "algorithm": "DoubleDQN",
        "seed": seed,
        "train_timesteps": TRAIN_TS,
        "eval_episodes": EVAL_EPS,
        "best_checkpoint_step": _step_from_path(best_path),
        "best_checkpoint_path": best_path,
        "best_val_learning_gain": best_val,
        "elapsed_seconds": round(elapsed, 1),
        "source": "trained",
        **metrics,
    }
    print(
        f"  double_dqn seed={seed}: lg={row['learning_gain']:.3f} "
        f"succ={row['success_rate']:.3f} adapt={row['adaptation_accuracy']:.3f} "
        f"({elapsed:.0f}s)"
    )
    return row


def run_study(seeds: List[int], resume: bool) -> pd.DataFrame:
    results_path = OUT_DIR / "confirmation_results.csv"
    baseline_df = load_baseline_thesis(seeds)
    rows = _baseline_to_comparison_rows(baseline_df)

    existing: List[Dict[str, Any]] = []
    if resume and results_path.exists():
        existing = pd.read_csv(results_path).to_dict("records")
    done_ddqn = {
        int(r["seed"])
        for r in existing
        if r.get("configuration") == DOUBLE_KEY
    }

    ddqn_rows = [r for r in existing if r.get("configuration") == DOUBLE_KEY]
    for seed in seeds:
        if seed in done_ddqn:
            continue
        ddqn_rows.append(train_double_dqn_seed(seed))
        all_rows = rows + ddqn_rows
        pd.DataFrame(all_rows).to_csv(results_path, index=False)

    all_rows = rows + ddqn_rows
    df = pd.DataFrame(all_rows)
    df.to_csv(results_path, index=False)
    return df


def _aggregate(sub: pd.DataFrame, configuration: str, label: str, algorithm: str) -> Dict[str, Any]:
    lgs = sub["learning_gain"].astype(float).tolist()
    lg_mean, lg_lo, lg_hi = confidence_interval(lgs)
    lg_std = float(np.std(lgs, ddof=1)) if len(lgs) > 1 else 0.0
    row: Dict[str, Any] = {
        "configuration": configuration,
        "configuration_label": label,
        "algorithm": algorithm,
        "n_seeds": len(sub),
        "learning_gain_mean": round(lg_mean, 4),
        "learning_gain_std": round(lg_std, 4),
        "coefficient_of_variation": round(_cv(lgs), 4),
        "learning_gain_ci_95": f"[{lg_lo:.3f}, {lg_hi:.3f}]",
    }
    for metric in FOCUS_METRICS:
        if metric == "learning_gain":
            continue
        vals = sub[metric].astype(float).tolist()
        mean, lo, hi = confidence_interval(vals)
        row[f"{metric}_mean"] = round(mean, 4)
        row[f"{metric}_std"] = round(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, 4)
        row[f"{metric}_ci_95"] = f"[{lo:.3f}, {hi:.3f}]"
    return row


def statistical_tests(results_df: pd.DataFrame) -> pd.DataFrame:
    std = results_df[results_df["configuration"] == STANDARD_KEY]
    ddqn = results_df[results_df["configuration"] == DOUBLE_KEY]
    rows: List[Dict[str, Any]] = []
    for metric in FOCUS_METRICS:
        v_std = std[metric].astype(float).tolist()
        v_ddqn = ddqn[metric].astype(float).tolist()
        if len(v_std) < 2 or len(v_ddqn) < 2:
            continue
        t_stat, p_val = stats.ttest_ind(v_ddqn, v_std, equal_var=False)
        mean_std, lo_std, hi_std = confidence_interval(v_std)
        mean_ddqn, lo_ddqn, hi_ddqn = confidence_interval(v_ddqn)
        pct = (
            (mean_ddqn - mean_std) / mean_std * 100.0
            if abs(mean_std) > 1e-9
            else float("nan")
        )
        rows.append({
            "comparison": "DoubleDQN_vs_StandardDQN",
            "metric": metric,
            "mean_standard_dqn": round(mean_std, 4),
            "mean_double_dqn": round(mean_ddqn, 4),
            "delta_double_dqn_minus_dqn": round(mean_ddqn - mean_std, 4),
            "pct_change": round(pct, 2),
            "cv_standard_dqn": round(_cv(v_std), 4),
            "cv_double_dqn": round(_cv(v_ddqn), 4),
            "ci_95_standard_dqn": f"[{lo_std:.3f}, {hi_std:.3f}]",
            "ci_95_double_dqn": f"[{lo_ddqn:.3f}, {hi_ddqn:.3f}]",
            "cohens_d": round(cohens_d(v_ddqn, v_std), 4),
            "welch_t": round(float(t_stat), 4),
            "p_value": round(float(p_val), 6),
            "significant_005": bool(p_val < 0.05),
        })
    return pd.DataFrame(rows)


def _verdict(summary_df: pd.DataFrame, stats_df: pd.DataFrame) -> Dict[str, Any]:
    dqn = summary_df[summary_df["configuration"] == STANDARD_KEY].iloc[0]
    ddqn = summary_df[summary_df["configuration"] == DOUBLE_KEY].iloc[0]
    lg_row = stats_df[stats_df["metric"] == "learning_gain"]
    lg_delta = float(lg_row["delta_double_dqn_minus_dqn"].iloc[0]) if len(lg_row) else 0.0
    lg_pct = float(lg_row["pct_change"].iloc[0]) if len(lg_row) else 0.0
    lg_sig = bool(lg_row["significant_005"].iloc[0]) if len(lg_row) else False

    ddqn_better_lg = float(ddqn["learning_gain_mean"]) > float(dqn["learning_gain_mean"])
    ddqn_lower_cv = float(ddqn["coefficient_of_variation"]) < float(dqn["coefficient_of_variation"])
    ddqn_better_success = float(ddqn["success_rate_mean"]) > float(dqn["success_rate_mean"])
    ddqn_better_adapt = float(ddqn["adaptation_accuracy_mean"]) > float(dqn["adaptation_accuracy_mean"])

    wins = sum([ddqn_better_lg, ddqn_lower_cv, ddqn_better_success, ddqn_better_adapt])
    recommend_ddqn = ddqn_better_lg and (wins >= 3 or lg_pct >= 5.0)

    return {
        "double_dqn_better_learning_gain": ddqn_better_lg,
        "double_dqn_lower_cv": ddqn_lower_cv,
        "double_dqn_better_success_rate": ddqn_better_success,
        "double_dqn_better_adaptation_accuracy": ddqn_better_adapt,
        "focus_metric_wins_double_dqn": wins,
        "learning_gain_delta": round(lg_delta, 4),
        "learning_gain_pct_change": round(lg_pct, 2),
        "learning_gain_significant": lg_sig,
        "recommend_replace_with_double_dqn": recommend_ddqn,
        "thesis_recommendation": (
            "Replace standard DQN with Double DQN in the final thesis system."
            if recommend_ddqn
            else "Retain standard DQN as the final thesis system."
        ),
    }


def _write_report(
    summary_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    verdict: Dict[str, Any],
    seeds: List[int],
) -> None:
    dqn = summary_df[summary_df["configuration"] == STANDARD_KEY].iloc[0]
    ddqn = summary_df[summary_df["configuration"] == DOUBLE_KEY].iloc[0]
    lines = [
        "# Final Double DQN Confirmation (20 Seeds)",
        "",
        "## Protocol",
        "",
        "- **Standard DQN:** frozen thesis baseline (`baseline_thesis`, no retrain)",
        "- **Double DQN:** same MDP, hyperparameters, checkpoint selection",
        f"- **Seeds (n={len(seeds)}):** `{seeds}`",
        f"- **Training:** {TRAIN_TS:,} steps | **Eval:** {EVAL_EPS} episodes",
        f"- **Hyperparameters:** `{FINAL_DQN_HP}`",
        f"- **Observation:** `{OBS_ABLATION}` (knowledge + 4 affect dims, no emotion_id)",
        "",
        "## Focus Metrics (Best Checkpoint)",
        "",
        "| System | Learning Gain | CV | Success Rate | Adaptation Accuracy |",
        "| --- | --- | --- | --- | --- |",
        (
            f"| Standard DQN | {dqn['learning_gain_mean']:.4f} +/- {dqn['learning_gain_std']:.4f} "
            f"{dqn['learning_gain_ci_95']} | {dqn['coefficient_of_variation']:.4f} | "
            f"{dqn['success_rate_mean']:.4f} +/- {dqn['success_rate_std']:.4f} | "
            f"{dqn['adaptation_accuracy_mean']:.4f} +/- {dqn['adaptation_accuracy_std']:.4f} |"
        ),
        (
            f"| Double DQN | {ddqn['learning_gain_mean']:.4f} +/- {ddqn['learning_gain_std']:.4f} "
            f"{ddqn['learning_gain_ci_95']} | {ddqn['coefficient_of_variation']:.4f} | "
            f"{ddqn['success_rate_mean']:.4f} +/- {ddqn['success_rate_std']:.4f} | "
            f"{ddqn['adaptation_accuracy_mean']:.4f} +/- {ddqn['adaptation_accuracy_std']:.4f} |"
        ),
        "",
        "## Pairwise Tests (Double DQN vs Standard DQN)",
        "",
        "| Metric | Delta (DDQN-DQN) | % Change | CV DQN | CV DDQN | p-value | Significant |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in stats_df.iterrows():
        sig = "yes" if row["significant_005"] else "no"
        cv_cols = ""
        if row["metric"] == "learning_gain":
            cv_cols = (
                f"{row['cv_standard_dqn']:.4f} | {row['cv_double_dqn']:.4f} | "
            )
        else:
            cv_cols = "- | - | "
        lines.append(
            f"| {row['metric']} | {row['delta_double_dqn_minus_dqn']:+.4f} | "
            f"{row['pct_change']:+.1f}% | {cv_cols}"
            f"{row['p_value']:.4f} | {sig} |"
        )

    lines.extend([
        "",
        "## Verdict",
        "",
        f"- Double DQN higher learning gain: **{verdict['double_dqn_better_learning_gain']}** "
        f"({verdict['learning_gain_pct_change']:+.1f}%)",
        f"- Double DQN lower CV: **{verdict['double_dqn_lower_cv']}**",
        f"- Double DQN higher success rate: **{verdict['double_dqn_better_success_rate']}**",
        f"- Double DQN higher adaptation accuracy: **{verdict['double_dqn_better_adaptation_accuracy']}**",
        f"- Focus-metric wins (DDQN): **{verdict['focus_metric_wins_double_dqn']}/4**",
        "",
        f"**Recommendation:** {verdict['thesis_recommendation']}",
        "",
    ])
    (OUT_DIR / "FINAL_DOUBLE_DQN_CONFIRMATION.md").write_text("\n".join(lines))


def analyze(results_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    path = OUT_DIR / "confirmation_results.csv"
    if results_df is None:
        if not path.exists():
            raise FileNotFoundError(f"No results at {path}; run --phase run first.")
        results_df = pd.read_csv(path)

    summary_rows = []
    for key, label, algo in (
        (STANDARD_KEY, "Standard DQN (thesis baseline)", "DQN"),
        (DOUBLE_KEY, "Double DQN (confirmation)", "DoubleDQN"),
    ):
        sub = results_df[results_df["configuration"] == key]
        if sub.empty:
            raise ValueError(f"Missing configuration {key} in results.")
        summary_rows.append(_aggregate(sub, key, label, algo))

    summary_df = pd.DataFrame(summary_rows)
    stats_df = statistical_tests(results_df)
    verdict = _verdict(summary_df, stats_df)
    seeds = sorted(results_df["seed"].unique().tolist())

    summary_df.to_csv(OUT_DIR / "confirmation_summary.csv", index=False)
    stats_df.to_csv(OUT_DIR / "confirmation_statistical_tests.csv", index=False)
    _write_report(summary_df, stats_df, verdict, seeds)

    report = {
        "study": "final_double_dqn_confirmation",
        "seeds": seeds,
        "hyperparameters": FINAL_DQN_HP,
        "obs_ablation": OBS_ABLATION,
        "train_timesteps": TRAIN_TS,
        "eval_episodes": EVAL_EPS,
        "summary": summary_df.to_dict("records"),
        "statistical_tests": stats_df.to_dict("records"),
        "verdict": verdict,
    }
    with open(OUT_DIR / "confirmation_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print("\n" + "=" * 72)
    print("FINAL DOUBLE DQN CONFIRMATION")
    print("=" * 72)
    for _, row in summary_df.iterrows():
        print(
            f"\n{row['configuration_label']}: "
            f"LG={row['learning_gain_mean']:.4f} (CV={row['coefficient_of_variation']:.4f}), "
            f"success={row['success_rate_mean']:.4f}, "
            f"adapt={row['adaptation_accuracy_mean']:.4f}"
        )
    print(f"\n{verdict['thesis_recommendation']}")
    print("=" * 72)
    return report


def main() -> None:
    global TRAIN_TS, EVAL_EPS, VAL_EPS

    parser = argparse.ArgumentParser(
        description="Final Double DQN confirmation vs thesis DQN baseline (20 seeds)"
    )
    parser.add_argument("--phase", choices=["run", "analyze", "all"], default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    seeds = SEEDS_20
    if args.quick:
        seeds = [42, 7]
        TRAIN_TS = 5_000
        EVAL_EPS = 50
        VAL_EPS = 20
        print(f"QUICK MODE: seeds={seeds}, train={TRAIN_TS}, eval={EVAL_EPS}")

    if args.phase in ("run", "all"):
        print("=== Final Double DQN confirmation (run) ===")
        print(f"Seeds: {len(seeds)}, train={TRAIN_TS}, obs={OBS_ABLATION}")
        run_study(seeds, args.resume)

    if args.phase in ("analyze", "all"):
        print("\n=== Final Double DQN confirmation (analyze) ===")
        analyze()


if __name__ == "__main__":
    main()
