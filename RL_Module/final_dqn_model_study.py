"""
Final thesis DQN model study: Tier A emotion stabilization + 20-seed robustness.

Observation: knowledge, engagement, frustration, confusion, boredom (no emotion_id).
Tier A simulator params (runtime patch only; mdp_definition defaults unchanged):
  LAMBDA_ENGAGEMENT 0.60 -> 0.80
  LAMBDA_FRUSTRATION 0.69 -> 0.845
  LAMBDA_CONFUSION 0.47 -> 0.735
  LAMBDA_BOREDOM 0.36 -> 0.68
  EMOTION_NOISE_STD 0.03 -> 0.015

Compares Tier A (20 seeds) vs baseline thesis model (D_no_emotion_id,
default emotion params, matched 20 seeds, best checkpoint).

Usage (from repo root):
  export PYTHONPATH="$(pwd)"
  python -m RL_Module.final_dqn_model_study --phase all
  python -m RL_Module.final_dqn_model_study --phase run --resume
  python -m RL_Module.final_dqn_model_study --phase baseline_run --resume
  python -m RL_Module.final_dqn_model_study --phase analyze
  python -m RL_Module.final_dqn_model_study --phase all --quick
"""

from __future__ import annotations

import argparse
import hashlib
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
from RL_Module.environment.student_env import StudentEnv
from RL_Module.evaluation.metrics import (
    _adaptation_match,
    _is_success,
    confidence_interval,
    normalized_knowledge_gain,
)

OUT_DIR = _HERE / "figures" / "final_dqn_model"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_DIR = _HERE / "figures" / "emotion_id_ablation"
STRICT_DIR = _HERE / "figures" / "strict_emotion_ablation"

# 20 seeds: thesis 10 + 5 emotion-id extension + 5 final-study extension
SEEDS_20 = list(cfg.SEEDS) + [500, 777, 1234, 2024, 5555, 8888, 9999, 10101, 20202, 30303]
SEEDS_15 = list(cfg.SEEDS) + [500, 777, 1234, 2024, 5555]

TRAIN_TS = 50_000
EVAL_EPS = 500
VAL_EPS = 100
CHECKPOINT_STEPS = [10_000, 20_000, 30_000, 40_000, 50_000]

G4_GAIN = {
    "GAIN_CORRECT_FACTOR": 0.1,
    "GAIN_INCORRECT_FACTOR": 1.0,
    "ratio_label": "10:1",
}

TIER_A_PARAMS = {
    "LAMBDA_ENGAGEMENT": 0.80,
    "LAMBDA_FRUSTRATION": 0.845,
    "LAMBDA_CONFUSION": 0.735,
    "LAMBDA_BOREDOM": 0.68,
    "EMOTION_NOISE_STD": 0.015,
}

FINAL_DQN_HP: Dict[str, Any] = {
    "learning_rate": 0.001,
    "batch_size": 64,
    "buffer_size": 100_000,
    "target_update_interval": 500,
    "exploration_fraction": 0.3,
    "exploration_final_eps": 0.05,
    "net_arch": [64, 64],
}

CONDITION_KEY = "tier_a_no_emotion_id"
BASELINE_KEY = "baseline_thesis"
OBS_ABLATION = "no_emotion_id"

METRICS = [
    "learning_gain",
    "final_knowledge",
    "success_rate",
    "adaptation_accuracy",
    "mean_episode_reward",
]


def _patch_simulator(tier_a: bool = True) -> Dict[str, Any]:
    prev = dict(cfg.SIMULATOR_PARAMS)
    cfg.SIMULATOR_PARAMS = {**G4_GAIN, **(TIER_A_PARAMS if tier_a else {})}
    return prev


def _restore_simulator(snapshot: Dict[str, Any]) -> None:
    cfg.SIMULATOR_PARAMS = snapshot


def config_id() -> str:
    payload = {
        "hp": FINAL_DQN_HP,
        "obs": OBS_ABLATION,
        "tier_a": TIER_A_PARAMS,
        "train_ts": TRAIN_TS,
    }
    return hashlib.md5(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:10]


def make_eval_env(seed: int) -> StudentEnv:
    return StudentEnv(
        population_seed=seed,
        obs_ablation=OBS_ABLATION,
        emotion_dynamics="full",
    )


def quick_eval_lg(
    agent: DQNAgent,
    seed: int,
    n_episodes: int = VAL_EPS,
) -> float:
    env = make_eval_env(seed)
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


def full_eval(
    agent: DQNAgent,
    seed: int,
    n_episodes: int = EVAL_EPS,
) -> Dict[str, float]:
    env = make_eval_env(seed)
    cfg.set_all_seeds(seed)
    results: List[Dict[str, float]] = []

    for ep in range(1, n_episodes + 1):
        obs, info = env.reset(seed=seed + ep)
        mask = info["action_masks"]
        start_k = env._state.knowledge
        total_reward = 0.0
        adapt_hits = adapt_total = 0
        done = False

        while not done:
            eid = env._state.emotion_id
            action = agent.predict(obs, mask)
            obs, reward, term, trunc, info = env.step(action)
            adapt_total += 1
            if _adaptation_match(eid, action):
                adapt_hits += 1
            total_reward += reward
            mask = info["action_masks"]
            done = term or trunc

        final = env._state
        results.append({
            "learning_gain": normalized_knowledge_gain(start_k, final.knowledge),
            "final_knowledge": final.knowledge,
            "success_rate": float(_is_success(final.knowledge, final.frustration, final.confusion)),
            "adaptation_accuracy": adapt_hits / max(adapt_total, 1),
            "mean_episode_reward": total_reward,
        })

    env.close()
    return {k: float(np.mean([r[k] for r in results])) for k in results[0]}


def _step_from_path(path: str) -> int:
    for step in CHECKPOINT_STEPS:
        if f"_{step}_steps" in path:
            return step
    return TRAIN_TS


def train_one_seed(seed: int, tier_a: bool = True) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    snap = _patch_simulator(tier_a=tier_a)
    model_key = CONDITION_KEY if tier_a else BASELINE_KEY
    cid = config_id()
    tag_prefix = "final_dqn_tier_a" if tier_a else "final_dqn_baseline"
    tag = f"{tag_prefix}_{cid}_s{seed}"
    ckpt_rows: List[Dict[str, Any]] = []
    t0 = time.time()

    try:
        cfg.set_all_seeds(seed)
        train_env = make_env(
            seed=seed,
            obs_ablation=OBS_ABLATION,
            emotion_dynamics="full",
            algo_tag=tag,
        )
        agent = DQNAgent()
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

        for step in CHECKPOINT_STEPS:
            path = step_paths.get(step)
            if not path:
                continue
            agent.load_checkpoint(path)
            metrics = full_eval(agent, seed, EVAL_EPS)
            ckpt_rows.append({
                "model": model_key,
                "seed": seed,
                "checkpoint_step": step,
                "checkpoint_path": path,
                **metrics,
            })

        valid_paths = [step_paths[s] for s in CHECKPOINT_STEPS if s in step_paths]
        if not valid_paths:
            valid_paths = list(paths)

        if valid_paths:
            best_path, best_val = agent.select_best_checkpoint(
                valid_paths,
                lambda a, s: quick_eval_lg(a, s, VAL_EPS),
                seed,
            )
            best_step = _step_from_path(best_path)
        else:
            best_path = ""
            best_val = float("nan")
            best_step = TRAIN_TS

        final_step = max(CHECKPOINT_STEPS)
        final_path = step_paths.get(final_step) or (paths[-1] if paths else "")
        if final_path:
            agent.load_checkpoint(final_path)
        final_metrics = full_eval(agent, seed, EVAL_EPS)

        if best_path:
            agent.load_checkpoint(best_path)
        best_metrics = full_eval(agent, seed, EVAL_EPS)

        summary = {
            "model": model_key,
            "model_label": (
                "Tier A: No Emotion ID + Stabilized Affect"
                if tier_a
                else "Baseline Thesis: No Emotion ID (default affect)"
            ),
            "obs_ablation": OBS_ABLATION,
            "emotion_stabilization": "tier_a" if tier_a else "default",
            "seed": seed,
            "gain_ratio": G4_GAIN["ratio_label"],
            "train_timesteps": TRAIN_TS,
            "eval_episodes": EVAL_EPS,
            "best_checkpoint_step": best_step,
            "best_checkpoint_path": best_path,
            "best_val_learning_gain": best_val,
            "final_checkpoint_step": _step_from_path(final_path),
            "elapsed_seconds": round(time.time() - t0, 1),
        }
        for prefix, metrics in [("final", final_metrics), ("best", best_metrics)]:
            for m in METRICS:
                summary[f"{prefix}_{m}"] = metrics[m]

        agent.save(str(cfg.MODELS_DIR / tag))
    finally:
        _restore_simulator(snap)

    return ckpt_rows, summary


def _seed_baseline_from_legacy() -> None:
    """Copy completed D_no_emotion_id runs into baseline_per_run_20.csv."""
    summary_path = OUT_DIR / "baseline_per_run_20.csv"
    existing: List[Dict[str, Any]] = []
    if summary_path.exists():
        existing = pd.read_csv(summary_path).to_dict("records")
    have = {int(r["seed"]) for r in existing}

    legacy_path = BASELINE_DIR / "emotion_id_ablation_per_run.csv"
    if not legacy_path.exists():
        return

    legacy = pd.read_csv(legacy_path)
    legacy = legacy[legacy["condition"] == "D_no_emotion_id"]
    for _, row in legacy.iterrows():
        seed = int(row["seed"])
        if seed in have or seed not in SEEDS_20:
            continue
        existing.append(_normalize_baseline_row(row))

    if existing:
        pd.DataFrame(existing).sort_values("seed").to_csv(summary_path, index=False)


def run_experiment(
    seeds: List[int],
    resume: bool,
    *,
    tier_a: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if tier_a:
        summary_path = OUT_DIR / "final_dqn_per_run.csv"
        ckpt_path = OUT_DIR / "checkpoint_eval_per_run.csv"
    else:
        summary_path = OUT_DIR / "baseline_per_run_20.csv"
        ckpt_path = OUT_DIR / "baseline_checkpoint_eval_per_run.csv"
        _seed_baseline_from_legacy()

    summaries: List[Dict[str, Any]] = _load_resume(summary_path, resume)
    ckpt_rows: List[Dict[str, Any]] = _load_resume(ckpt_path, resume)

    done = {int(r["seed"]) for r in summaries}
    total = len(seeds)
    label = "Tier A" if tier_a else "Baseline"

    for i, seed in enumerate(seeds):
        if seed in done:
            continue
        print(f"[{label} {len(done) + 1}/{total}] seed={seed}")
        batch_ckpt, summary = train_one_seed(seed, tier_a=tier_a)
        ckpt_rows.extend(batch_ckpt)
        summaries.append(summary)
        done.add(seed)
        pd.DataFrame(summaries).to_csv(summary_path, index=False)
        pd.DataFrame(ckpt_rows).to_csv(ckpt_path, index=False)
        print(
            f"  val_lg={summary['best_val_learning_gain']:.3f} "
            f"final_lg={summary['final_learning_gain']:.3f} "
            f"best_lg={summary['best_learning_gain']:.3f} "
            f"@step={summary['best_checkpoint_step']} "
            f"({summary['elapsed_seconds']}s)"
        )

    return pd.DataFrame(summaries), pd.DataFrame(ckpt_rows)


def _load_resume(path: Path, resume: bool) -> List[Dict[str, Any]]:
    if resume and path.exists():
        return pd.read_csv(path).to_dict("records")
    return []


def cohens_d(a: List[float], b: List[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = np.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2))
    return float((np.mean(a) - np.mean(b)) / pooled) if pooled > 1e-12 else 0.0


def aggregate_runs(
    df: pd.DataFrame,
    model_key: str,
    metric_prefix: str,
    label: str,
) -> Dict[str, Any]:
    sub = df[df["model"] == model_key] if "model" in df.columns else df
    row: Dict[str, Any] = {
        "model": model_key,
        "model_label": label,
        "n_seeds": len(sub),
    }
    for m in METRICS:
        col = f"{metric_prefix}_{m}"
        if col not in sub.columns:
            continue
        vals = sub[col].astype(float).tolist()
        mean, lo, hi = confidence_interval(vals)
        row[f"{m}_mean"] = round(mean, 4)
        row[f"{m}_std"] = round(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, 4)
        row[f"{m}_ci"] = f"[{lo:.3f}, {hi:.3f}]"
    return row


def _normalize_baseline_row(row: pd.Series) -> Dict[str, Any]:
    return {
        "model": BASELINE_KEY,
        "model_label": "Baseline Thesis: No Emotion ID (default affect)",
        "obs_ablation": OBS_ABLATION,
        "emotion_stabilization": "default",
        "seed": int(row["seed"]),
        "gain_ratio": row.get("gain_ratio", G4_GAIN["ratio_label"]),
        "train_timesteps": int(row.get("train_timesteps", TRAIN_TS)),
        "eval_episodes": int(row.get("eval_episodes", EVAL_EPS)),
        "best_checkpoint_step": int(row["best_checkpoint_step"]),
        "best_checkpoint_path": row.get("best_checkpoint_path", ""),
        "best_val_learning_gain": float(row["best_val_learning_gain"]),
        "final_checkpoint_step": int(row.get("final_checkpoint_step", TRAIN_TS)),
        "elapsed_seconds": float(row.get("elapsed_seconds", 0.0)),
        **{
            f"{prefix}_{m}": float(row[f"{prefix}_{m}"])
            for prefix in ("final", "best")
            for m in METRICS
            if f"{prefix}_{m}" in row.index
        },
    }


def load_baseline_thesis(seeds: Optional[List[int]] = None) -> pd.DataFrame:
    """Load baseline D_no_emotion_id runs; prefer matched 20-seed study CSV."""
    target_seeds = seeds or SEEDS_20
    local_path = OUT_DIR / "baseline_per_run_20.csv"
    rows: List[Dict[str, Any]] = []

    if local_path.exists():
        local = pd.read_csv(local_path)
        local = local[local["seed"].isin(target_seeds)]
        rows.extend(local.to_dict("records"))

    have = {int(r["seed"]) for r in rows}
    missing = [s for s in target_seeds if s not in have]
    if missing:
        legacy_path = BASELINE_DIR / "emotion_id_ablation_per_run.csv"
        if not legacy_path.exists():
            raise FileNotFoundError(
                f"Baseline results missing for seeds {missing}. "
                f"Run --phase baseline_run first."
            )
        legacy = pd.read_csv(legacy_path)
        legacy = legacy[
            (legacy["condition"] == "D_no_emotion_id")
            & (legacy["seed"].isin(missing))
        ]
        for _, row in legacy.iterrows():
            rows.append(_normalize_baseline_row(row))

    if len(rows) < len(target_seeds):
        still_missing = sorted(set(target_seeds) - {int(r["seed"]) for r in rows})
        raise FileNotFoundError(
            f"Baseline incomplete: missing seeds {still_missing}. "
            "Run --phase baseline_run --resume."
        )

    df = pd.DataFrame(rows)
    df = df[df["seed"].isin(target_seeds)].drop_duplicates(subset=["seed"], keep="last")
    df["model"] = BASELINE_KEY
    return df.sort_values("seed").reset_index(drop=True)


def pairwise_compare(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    model_a: str,
    model_b: str,
    metric_prefix: str,
    label_a: str,
    label_b: str,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    p_vals: List[float] = []
    sub_a = df_a[df_a["model"] == model_a] if "model" in df_a.columns else df_a
    sub_b = df_b[df_b["model"] == model_b] if "model" in df_b.columns else df_b

    for m in METRICS:
        col = f"{metric_prefix}_{m}"
        v_a = sub_a[col].astype(float).tolist()
        v_b = sub_b[col].astype(float).tolist()
        if len(v_a) < 2 or len(v_b) < 2:
            continue
        t_stat, p_val = stats.ttest_ind(v_a, v_b, equal_var=False)
        rows.append({
            "comparison": f"{model_a}_vs_{model_b}",
            "eval_type": metric_prefix,
            "metric": m,
            "label_A": label_a,
            "label_B": label_b,
            "n_A": len(v_a),
            "n_B": len(v_b),
            "mean_A": round(float(np.mean(v_a)), 4),
            "mean_B": round(float(np.mean(v_b)), 4),
            "delta_A_minus_B": round(float(np.mean(v_a) - np.mean(v_b)), 4),
            "std_A": round(float(np.std(v_a, ddof=1)), 4),
            "std_B": round(float(np.std(v_b, ddof=1)), 4),
            "cohens_d": round(cohens_d(v_a, v_b), 4),
            "t": round(float(t_stat), 4),
            "p_raw": round(float(p_val), 6),
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


def checkpoint_selection_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "seed",
        "best_checkpoint_step",
        "best_val_learning_gain",
        "final_learning_gain",
        "best_learning_gain",
    ]
    out = df[cols].copy()
    out = out.rename(columns={
        "best_checkpoint_step": "selected_checkpoint",
        "best_val_learning_gain": "validation_learning_gain",
        "final_learning_gain": "final_eval_learning_gain",
        "best_learning_gain": "selected_eval_learning_gain",
    })
    return out.round(4)


def _strict_no_emotion_lg() -> Optional[float]:
    path = STRICT_DIR / "strict_ablation_per_run.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    sub = df[df["condition"] == "C_strict_no_emotion"]
    if "learning_gain" in sub.columns and len(sub):
        return float(sub["learning_gain"].mean())
    return None


def final_recommendation(
    tier_summary: Dict[str, Any],
    baseline_summary: Dict[str, Any],
    pairwise: pd.DataFrame,
    strict_lg: Optional[float],
) -> Dict[str, Any]:
    lg_row = pairwise[pairwise["metric"] == "learning_gain"]
    lg_p = float(lg_row["p_holm"].iloc[0]) if len(lg_row) else 1.0
    lg_delta = float(lg_row["delta_A_minus_B"].iloc[0]) if len(lg_row) else 0.0
    lg_d = float(lg_row["cohens_d"].iloc[0]) if len(lg_row) else float("nan")

    tier_lg = tier_summary["learning_gain_mean"]
    tier_std = tier_summary["learning_gain_std"]
    base_lg = baseline_summary["learning_gain_mean"]
    base_std = baseline_summary["learning_gain_std"]

    baseline_outperforms = base_lg > tier_lg
    baseline_highest_lg = base_lg >= tier_lg
    tier_lowest_var = tier_std < base_std
    baseline_lowest_var = base_std < tier_std
    sig_diff = lg_p < 0.05

    gap_strict = None
    closes_strict_gap = None
    if strict_lg is not None:
        gap_strict = round(tier_lg - strict_lg, 4)
        closes_strict_gap = tier_lg >= strict_lg

    pick_baseline = (base_lg, -base_std) >= (tier_lg, -tier_std)
    winner = BASELINE_KEY if pick_baseline else "tier_a"
    winner_summary = baseline_summary if pick_baseline else tier_summary
    winner_emotion = (
        "default (mdp_definition literature defaults)"
        if pick_baseline
        else "tier_a (stabilized affect lambdas + reduced noise)"
    )

    return {
        "Q1_baseline_outperforms_tier_a": baseline_outperforms,
        "Q1_baseline_lg_minus_tier_a": round(base_lg - tier_lg, 4),
        "Q2_statistically_significant": sig_diff,
        "Q2_p_holm_learning_gain": lg_p,
        "Q2_cohens_d_learning_gain": lg_d,
        "Q3_baseline_highest_learning_gain": baseline_highest_lg,
        "Q4_lowest_variance_model": (
            BASELINE_KEY if baseline_lowest_var else "tier_a"
        ),
        "Q4_tier_a_std": tier_std,
        "Q4_baseline_std": base_std,
        "Q5_recommended_model": winner,
        "tier_a_learning_gain_mean": tier_lg,
        "tier_a_learning_gain_std": tier_std,
        "tier_a_learning_gain_ci": tier_summary["learning_gain_ci"],
        "baseline_learning_gain_mean": base_lg,
        "baseline_learning_gain_std": base_std,
        "baseline_learning_gain_ci": baseline_summary["learning_gain_ci"],
        "strict_no_emotion_reference_lg": strict_lg,
        "gap_to_strict_no_emotion": gap_strict,
        "recommended_model": winner,
        "recommended_observation": (
            "knowledge, engagement, frustration, confusion, boredom"
        ),
        "recommended_emotion_params": winner_emotion,
        "recommended_checkpoint_strategy": (
            "best checkpoint selected by validation learning gain "
            f"(checkpoints {CHECKPOINT_STEPS})"
        ),
        "recommended_learning_gain_mean": winner_summary["learning_gain_mean"],
        "recommended_learning_gain_std": winner_summary["learning_gain_std"],
        "recommended_learning_gain_ci": winner_summary["learning_gain_ci"],
        "thesis_freeze_statement": (
            "This is the model that should be used for all remaining thesis "
            "experiments and thesis reporting."
        ),
        "justification": _build_justification(
            winner, tier_lg, base_lg, tier_std, base_std, lg_p, strict_lg, closes_strict_gap
        ),
    }


def _metric_improved(pairwise: pd.DataFrame, metric: str) -> bool:
    row = pairwise[pairwise["metric"] == metric]
    if row.empty:
        return False
    return float(row["delta_A_minus_B"].iloc[0]) > 0


def _build_justification(
    winner: str,
    tier_lg: float,
    base_lg: float,
    tier_std: float,
    base_std: float,
    lg_p: float,
    strict_lg: Optional[float],
    closes_strict: Optional[bool],
) -> str:
    parts = []
    if winner == BASELINE_KEY:
        parts.append(
            f"Baseline retains higher mean learning gain on matched 20 seeds "
            f"({base_lg:.3f} vs Tier A {tier_lg:.3f})"
        )
        if base_std <= tier_std:
            parts.append(
                f"with equal or lower variance (std {base_std:.3f} vs {tier_std:.3f})"
            )
    else:
        parts.append(
            f"Tier A achieves higher mean LG ({tier_lg:.3f} vs baseline {base_lg:.3f})"
        )
        if tier_std < base_std:
            parts.append(f"with lower cross-seed variance (std {tier_std:.3f} vs {base_std:.3f})")

    if lg_p >= 0.05:
        parts.append(
            f"the LG difference is not statistically significant after Holm correction "
            f"(p={lg_p:.3f})"
        )
    else:
        parts.append(f"Holm-corrected p={lg_p:.4f} for learning gain")

    if strict_lg is not None and winner == "tier_a":
        if closes_strict:
            parts.append(
                f"Tier A meets or exceeds strict no-emotion LG ({tier_lg:.3f} vs {strict_lg:.3f})"
            )
    return "; ".join(parts)


def analyze_results(tier_df: pd.DataFrame) -> Dict[str, Any]:
    baseline_df = load_baseline_thesis(SEEDS_20)

    tier_best = aggregate_runs(
        tier_df, CONDITION_KEY, "best", "Tier A: No Emotion ID + Stabilized Affect"
    )
    tier_final = aggregate_runs(
        tier_df, CONDITION_KEY, "final", "Tier A: No Emotion ID + Stabilized Affect"
    )
    baseline_best = aggregate_runs(
        baseline_df, BASELINE_KEY, "best", "Baseline Thesis: No Emotion ID"
    )

    pairwise_20 = pairwise_compare(
        tier_df, baseline_df,
        CONDITION_KEY, BASELINE_KEY, "best",
        "Tier A (20 seeds)", "Baseline (20 seeds)",
    )
    pairwise_15 = pairwise_compare(
        tier_df[tier_df["seed"].isin(SEEDS_15)].copy(),
        baseline_df[baseline_df["seed"].isin(SEEDS_15)].copy(),
        CONDITION_KEY, BASELINE_KEY, "best",
        "Tier A (15 seeds)", "Baseline (15 seeds)",
    )

    ckpt_table = checkpoint_selection_table(tier_df)
    baseline_ckpt = checkpoint_selection_table(baseline_df)
    strict_lg = _strict_no_emotion_lg()
    recommendation = final_recommendation(
        tier_best, baseline_best, pairwise_20, strict_lg
    )

    metrics_table = _build_metrics_table(tier_best, baseline_best)

    report = {
        "study": "final_dqn_matched_20_seed_confirmation",
        "observation_space": [
            "knowledge", "engagement", "frustration", "confusion", "boredom",
        ],
        "excluded_observation": ["emotion_id"],
        "gain_ratio": G4_GAIN["ratio_label"],
        "tier_a_params": TIER_A_PARAMS,
        "baseline_emotion_params": "default (mdp_definition literature defaults)",
        "dqn_hyperparameters": FINAL_DQN_HP,
        "checkpoint_steps": CHECKPOINT_STEPS,
        "seeds_20": SEEDS_20,
        "tier_a_summary_best_20": tier_best,
        "tier_a_summary_final_20": tier_final,
        "baseline_summary_best_20": baseline_best,
        "metrics_table_best_checkpoint": metrics_table,
        "pairwise_tier_vs_baseline_20v20": pairwise_20.to_dict("records"),
        "pairwise_tier_vs_baseline_15v15": pairwise_15.to_dict("records"),
        "final_recommendation": recommendation,
        "strict_no_emotion_reference_lg": strict_lg,
    }

    pd.DataFrame([tier_best]).to_csv(OUT_DIR / "summary_tier_a_best_20.csv", index=False)
    pd.DataFrame([baseline_best]).to_csv(OUT_DIR / "summary_baseline_best_20.csv", index=False)
    pd.DataFrame([baseline_best]).to_csv(OUT_DIR / "summary_baseline_best_15.csv", index=False)
    metrics_table.to_csv(OUT_DIR / "metrics_table_matched_20.csv", index=False)
    ckpt_table.to_csv(OUT_DIR / "checkpoint_selection_per_seed.csv", index=False)
    baseline_ckpt.to_csv(OUT_DIR / "baseline_checkpoint_selection_per_seed.csv", index=False)
    pairwise_20.to_csv(OUT_DIR / "pairwise_tier_vs_baseline_20seed.csv", index=False)
    pairwise_20.to_csv(OUT_DIR / "pairwise_tier_vs_baseline.csv", index=False)
    pairwise_15.to_csv(OUT_DIR / "pairwise_tier_vs_baseline_15seed.csv", index=False)
    tier_df.to_csv(OUT_DIR / "final_dqn_per_run.csv", index=False)
    baseline_df.to_csv(OUT_DIR / "baseline_per_run_20.csv", index=False)

    with open(OUT_DIR / "final_dqn_model_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    _write_markdown_report(
        report, tier_best, tier_final, baseline_best,
        pairwise_20, pairwise_15, ckpt_table, baseline_ckpt,
        metrics_table, recommendation,
    )
    _plot_overview(tier_df, baseline_df, tier_best, baseline_best)
    return report


def _build_metrics_table(
    tier_best: Dict[str, Any],
    baseline_best: Dict[str, Any],
) -> pd.DataFrame:
    rows = []
    for label, summary in [
        ("Tier A (20 seeds)", tier_best),
        ("Baseline (20 seeds)", baseline_best),
    ]:
        row = {"model": label, "n_seeds": summary["n_seeds"]}
        for m in METRICS:
            row[f"{m}_mean"] = summary.get(f"{m}_mean")
            row[f"{m}_std"] = summary.get(f"{m}_std")
            row[f"{m}_ci"] = summary.get(f"{m}_ci")
        rows.append(row)
    return pd.DataFrame(rows)


def _df_to_md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def _write_markdown_report(
    report: Dict[str, Any],
    tier_best: Dict[str, Any],
    tier_final: Dict[str, Any],
    baseline_best: Dict[str, Any],
    pairwise_20: pd.DataFrame,
    pairwise_15: pd.DataFrame,
    ckpt_table: pd.DataFrame,
    baseline_ckpt: pd.DataFrame,
    metrics_table: pd.DataFrame,
    rec: Dict[str, Any],
) -> None:
    lines = [
        "# Final DQN Thesis Model - Matched 20-Seed Confirmation",
        "",
        "## Configuration",
        "",
        "**Observation (agent-facing):** knowledge, engagement, frustration, confusion, boredom",
        "",
        "**Removed:** emotion_id",
        "",
        "**Baseline:** default emotional dynamics, default reward, default transitions",
        "",
        "**Tier A emotion stabilization** (runtime `SIMULATOR_PARAMS` patch only):",
        "",
        "| Parameter | Default | Tier A |",
        "|-----------|---------|--------|",
        "| LAMBDA_ENGAGEMENT | 0.60 | 0.80 |",
        "| LAMBDA_FRUSTRATION | 0.69 | 0.845 |",
        "| LAMBDA_CONFUSION | 0.47 | 0.735 |",
        "| LAMBDA_BOREDOM | 0.36 | 0.68 |",
        "| EMOTION_NOISE_STD | 0.03 | 0.015 |",
        "",
        f"Gain ratio: **{G4_GAIN['ratio_label']}** | Training: **{TRAIN_TS:,}** steps | "
        f"Eval: **{EVAL_EPS}** episodes | Checkpoints: {CHECKPOINT_STEPS}",
        "",
        f"Seeds (matched n=20): `{SEEDS_20}`",
        "",
        "## Table 1 - Aggregate Metrics (Best Checkpoint, n=20)",
        "",
        _df_to_md(metrics_table[[
            "model", "n_seeds",
            "learning_gain_mean", "learning_gain_std", "learning_gain_ci",
            "final_knowledge_mean", "success_rate_mean",
            "adaptation_accuracy_mean", "mean_episode_reward_mean",
        ]]),
        "",
        "## Table 2 - Pairwise Comparison (Tier A vs Baseline, n=20)",
        "",
        _df_to_md(pairwise_20[[
            "metric", "mean_A", "mean_B", "delta_A_minus_B",
            "std_A", "std_B", "cohens_d", "p_holm",
        ]].rename(columns={
            "mean_A": "tier_a_mean",
            "mean_B": "baseline_mean",
            "delta_A_minus_B": "tier_minus_baseline",
            "std_A": "tier_a_std",
            "std_B": "baseline_std",
        })),
        "",
        "## Checkpoint Selection (Tier A, per seed)",
        "",
        _df_to_md(ckpt_table.head(20)),
        "",
        "## Checkpoint Selection (Baseline, per seed)",
        "",
        _df_to_md(baseline_ckpt.head(20)),
        "",
        "## Research Questions (Matched 20-Seed Protocol)",
        "",
        f"1. Does Baseline still outperform Tier A? **{rec['Q1_baseline_outperforms_tier_a']}** "
        f"(baseline - tier A LG = {rec['Q1_baseline_lg_minus_tier_a']:+.4f})",
        f"2. Are differences statistically significant? **{rec['Q2_statistically_significant']}** "
        f"(learning gain p_holm={rec['Q2_p_holm_learning_gain']:.4f}, "
        f"Cohen's d={rec['Q2_cohens_d_learning_gain']:.3f})",
        f"3. Does Baseline still have the highest learning gain? "
        f"**{rec['Q3_baseline_highest_learning_gain']}**",
        f"4. Which model has the lowest variance? **`{rec['Q4_lowest_variance_model']}`** "
        f"(Tier A std={rec['Q4_tier_a_std']:.3f}, Baseline std={rec['Q4_baseline_std']:.3f})",
        f"5. Which model should be frozen as the final thesis model? **`{rec['Q5_recommended_model']}`**",
        "",
        "## Final Thesis Recommendation",
        "",
        f"**Selected model:** `{rec['recommended_model']}`",
        "",
        f"- Observation: {rec['recommended_observation']}",
        f"- Emotion configuration: {rec['recommended_emotion_params']}",
        f"- Checkpoint strategy: {rec['recommended_checkpoint_strategy']}",
        f"- Mean learning gain: **{rec['recommended_learning_gain_mean']:.3f}**",
        f"- 95% CI: {rec['recommended_learning_gain_ci']}",
        f"- Std: {rec['recommended_learning_gain_std']:.3f}",
        "",
        rec["justification"],
        "",
        f"**{rec['thesis_freeze_statement']}**",
        "",
        "## Reference (15-Seed Overlap)",
        "",
    ]
    for _, row in pairwise_15.iterrows():
        if row["metric"] == "learning_gain":
            lines.append(
                f"- learning_gain: Tier A={row['mean_A']:.4f} vs Baseline={row['mean_B']:.4f}, "
                f"d={row['cohens_d']:.3f}, p_holm={row.get('p_holm', row['p_raw']):.4f}"
            )

    lines.extend([
        "",
        "## Reference Studies",
        "",
        "- Baseline seeds 1-15: `figures/emotion_id_ablation/` (D_no_emotion_id)",
        "- Baseline seeds 16-20: `figures/final_dqn_model/baseline_per_run_20.csv`",
        "- Tier A: `figures/final_dqn_model/final_dqn_per_run.csv`",
        "",
    ])
    (OUT_DIR / "FINAL_DQN_MODEL_REPORT.md").write_text("\n".join(lines))


def _plot_overview(
    tier_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    tier_best: Dict[str, Any],
    baseline_best: Dict[str, Any],
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    data = [
        tier_df["best_learning_gain"].astype(float).tolist(),
        baseline_df["best_learning_gain"].astype(float).tolist(),
    ]
    bp = ax.boxplot(
        data,
        labels=["Tier A\n(20 seeds)", "Baseline\n(20 seeds)"],
        patch_artist=True,
    )
    bp["boxes"][0].set_facecolor("#A8DADC")
    bp["boxes"][1].set_facecolor("#E8A598")
    ax.set_ylabel("Learning Gain (best checkpoint)")
    ax.set_title("Final DQN Model Comparison")
    ax.grid(axis="y", alpha=0.3)

    ax2 = axes[1]
    models = ["Tier A", "Baseline"]
    means = [tier_best["learning_gain_mean"], baseline_best["learning_gain_mean"]]
    stds = [tier_best["learning_gain_std"], baseline_best["learning_gain_std"]]
    ax2.bar(models, means, yerr=stds, capsize=6, color=["#A8DADC", "#E8A598"], edgecolor="#64748B")
    ax2.set_ylabel("Mean Learning Gain")
    ax2.set_title("Aggregate (best checkpoint)")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Final DQN Thesis Model (G4 {G4_GAIN['ratio_label']})", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "final_dqn_model_overview.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Final DQN thesis model study")
    parser.add_argument(
        "--phase",
        choices=["run", "baseline_run", "analyze", "all"],
        default="all",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    seeds = SEEDS_20
    global TRAIN_TS, EVAL_EPS, CHECKPOINT_STEPS, VAL_EPS

    if args.quick:
        seeds = [42, 7, 13]
        TRAIN_TS = 5_000
        EVAL_EPS = 50
        VAL_EPS = 20
        CHECKPOINT_STEPS = [5_000]
        print("QUICK MODE: 3 seeds, 5k train")

    summary_path = OUT_DIR / "final_dqn_per_run.csv"
    baseline_path = OUT_DIR / "baseline_per_run_20.csv"
    if args.fresh and args.phase in ("run", "all"):
        if summary_path.exists():
            summary_path.unlink()
        ckpt_path = OUT_DIR / "checkpoint_eval_per_run.csv"
        if ckpt_path.exists():
            ckpt_path.unlink()
    if args.fresh and args.phase in ("baseline_run", "all"):
        if baseline_path.exists():
            baseline_path.unlink()
        bckpt = OUT_DIR / "baseline_checkpoint_eval_per_run.csv"
        if bckpt.exists():
            bckpt.unlink()

    if args.phase in ("run", "all"):
        print("=== Final DQN Tier A: RUN ===")
        print(f"Seeds: {len(seeds)} | Tier A params: {TIER_A_PARAMS}")
        run_experiment(seeds, resume=args.resume, tier_a=True)

    if args.phase in ("baseline_run", "all"):
        print("=== Final DQN Baseline (default affect): RUN ===")
        print(f"Seeds: {len(seeds)} | default emotion params")
        run_experiment(seeds, resume=args.resume, tier_a=False)

    if args.phase in ("analyze", "all"):
        if not summary_path.exists():
            raise SystemExit(f"No Tier A results at {summary_path}. Run with --phase run first.")
        tier_df = pd.read_csv(summary_path)
        tier_df["model"] = CONDITION_KEY
        print("\n=== ANALYZE (matched 20-seed) ===")
        report = analyze_results(tier_df)
        print(json.dumps(report["final_recommendation"], indent=2))


if __name__ == "__main__":
    main()
