"""
Emotion-ID ablation + DQN stability investigation (thesis study).

Compares:
  Condition A (full_emotion):     knowledge, engagement, frustration, confusion, boredom, emotion_id
  Condition D (no_emotion_id):    knowledge, engagement, frustration, confusion, boredom

Protocol:
  - 15 seeds (10 thesis seeds + 5 extension seeds)
  - G4 gain ratio (10:1), FINAL_DQN_HP, 50k train, 500 eval episodes
  - Checkpoints at 10k-50k; evaluate all; compare final vs best checkpoint
  - Emotion volatility correlations (confusion, boredom, emotion_id switches vs learning gain)

Usage (from repo root):
  export PYTHONPATH="$(pwd)"
  python -m RL_Module.emotion_id_ablation_study --phase all
  python -m RL_Module.emotion_id_ablation_study --phase run --resume
  python -m RL_Module.emotion_id_ablation_study --phase analyze
  python -m RL_Module.emotion_id_ablation_study --phase all --quick
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

OUT_DIR = _HERE / "figures" / "emotion_id_ablation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 15 seeds: thesis 10 + 5 extension for robustness
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

FINAL_DQN_HP: Dict[str, Any] = {
    "learning_rate": 0.001,
    "batch_size": 64,
    "buffer_size": 100_000,
    "target_update_interval": 500,
    "exploration_fraction": 0.3,
    "exploration_final_eps": 0.05,
    "net_arch": [64, 64],
}

CONDITIONS: Dict[str, Dict[str, str]] = {
    "A_full_emotion": {
        "obs_ablation": "full_emotion",
        "emotion_dynamics": "full",
        "label": "A: Full Emotion",
    },
    "D_no_emotion_id": {
        "obs_ablation": "no_emotion_id",
        "emotion_dynamics": "full",
        "label": "D: No Emotion ID",
    },
}

METRICS = [
    "learning_gain",
    "final_knowledge",
    "success_rate",
    "adaptation_accuracy",
    "mean_episode_reward",
]


def _patch_gain() -> Dict[str, float]:
    prev = dict(cfg.SIMULATOR_PARAMS)
    cfg.SIMULATOR_PARAMS = dict(G4_GAIN)
    return prev


def _restore_gain(snapshot: Dict[str, float]) -> None:
    cfg.SIMULATOR_PARAMS = snapshot


def config_id(obs_ablation: str) -> str:
    payload = {"hp": FINAL_DQN_HP, "obs": obs_ablation, "train_ts": TRAIN_TS}
    return hashlib.md5(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:10]


def make_eval_env(seed: int, obs_ablation: str) -> StudentEnv:
    return StudentEnv(
        population_seed=seed,
        obs_ablation=obs_ablation,
        emotion_dynamics="full",
    )


def quick_eval_lg(
    agent: DQNAgent,
    seed: int,
    obs_ablation: str,
    n_episodes: int = VAL_EPS,
) -> float:
    env = make_eval_env(seed, obs_ablation)
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


def full_eval_with_volatility(
    agent: DQNAgent,
    seed: int,
    obs_ablation: str,
    n_episodes: int = EVAL_EPS,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Full metrics plus per-seed emotion volatility during evaluation rollouts."""
    env = make_eval_env(seed, obs_ablation)
    cfg.set_all_seeds(seed)
    results: List[Dict[str, float]] = []
    confusion_vars: List[float] = []
    boredom_vars: List[float] = []
    emotion_id_switches: List[int] = []

    for ep in range(1, n_episodes + 1):
        obs, info = env.reset(seed=seed + ep)
        mask = info["action_masks"]
        start_k = env._state.knowledge
        total_reward = 0.0
        adapt_hits = adapt_total = 0
        confusion_trace: List[float] = []
        boredom_trace: List[float] = []
        eid_trace: List[int] = []
        done = False

        while not done:
            eid = env._state.emotion_id
            confusion_trace.append(env._state.confusion)
            boredom_trace.append(env._state.boredom)
            eid_trace.append(eid)
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
        if len(confusion_trace) > 1:
            confusion_vars.append(float(np.var(confusion_trace)))
            boredom_vars.append(float(np.var(boredom_trace)))
        switches = sum(1 for i in range(1, len(eid_trace)) if eid_trace[i] != eid_trace[i - 1])
        emotion_id_switches.append(switches)

    env.close()
    metrics = {k: float(np.mean([r[k] for r in results])) for k in results[0]}
    volatility = {
        "mean_confusion_variance": float(np.mean(confusion_vars)) if confusion_vars else 0.0,
        "mean_boredom_variance": float(np.mean(boredom_vars)) if boredom_vars else 0.0,
        "mean_emotion_id_switches": float(np.mean(emotion_id_switches)) if emotion_id_switches else 0.0,
        "std_confusion_variance": float(np.std(confusion_vars, ddof=1)) if len(confusion_vars) > 1 else 0.0,
        "std_boredom_variance": float(np.std(boredom_vars, ddof=1)) if len(boredom_vars) > 1 else 0.0,
    }
    return metrics, volatility


def evaluate_checkpoint(
    agent: DQNAgent,
    seed: int,
    obs_ablation: str,
    checkpoint_step: int,
    n_episodes: int = EVAL_EPS,
) -> Dict[str, Any]:
    metrics, _ = full_eval_with_volatility(agent, seed, obs_ablation, n_episodes)
    return {"checkpoint_step": checkpoint_step, **metrics}


def train_with_full_checkpoint_eval(
    condition_key: str,
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, float]]:
    spec = CONDITIONS[condition_key]
    obs_ablation = spec["obs_ablation"]
    snap = _patch_gain()
    cid = config_id(obs_ablation)
    tag = f"emotion_id_ablation_{condition_key}_{cid}_s{seed}"
    ckpt_rows: List[Dict[str, Any]] = []
    t0 = time.time()

    try:
        cfg.set_all_seeds(seed)
        train_env = make_env(
            seed=seed,
            obs_ablation=obs_ablation,
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
            hyperparams=FINAL_DQN_HP,
            checkpoint_dir=str(ckpt_dir),
            checkpoint_freq=ckpt_freq,
        )
        train_env.close()

        # Map step -> path
        step_paths: Dict[int, str] = {}
        for p in paths:
            for step in CHECKPOINT_STEPS:
                if f"_{step}_steps" in p:
                    step_paths[step] = p

        # Evaluate every checkpoint
        for step in CHECKPOINT_STEPS:
            path = step_paths.get(step)
            if not path:
                continue
            agent.load_checkpoint(path)
            metrics, _ = full_eval_with_volatility(agent, seed, obs_ablation, EVAL_EPS)
            ckpt_rows.append({
                "condition": condition_key,
                "condition_label": spec["label"],
                "obs_ablation": obs_ablation,
                "seed": seed,
                "checkpoint_step": step,
                "checkpoint_path": path,
                **metrics,
            })

        # Best checkpoint via validation LG
        valid_paths = [step_paths[s] for s in CHECKPOINT_STEPS if s in step_paths]
        if not valid_paths:
            valid_paths = list(paths)
        if valid_paths:
            best_path, best_val = agent.select_best_checkpoint(
                valid_paths,
                lambda a, s: quick_eval_lg(a, s, obs_ablation, VAL_EPS),
                seed,
            )
            best_step = _step_from_path(best_path)
        else:
            best_path = ""
            best_val = float("nan")
            best_step = TRAIN_TS

        # Final checkpoint (last training step or last saved path)
        final_step = max(CHECKPOINT_STEPS) if CHECKPOINT_STEPS else TRAIN_TS
        final_path = step_paths.get(final_step) or (paths[-1] if paths else "")
        if final_path:
            agent.load_checkpoint(final_path)
        final_metrics, final_volatility = full_eval_with_volatility(
            agent, seed, obs_ablation, EVAL_EPS
        )

        # Best checkpoint full eval + volatility
        if best_path:
            agent.load_checkpoint(best_path)
        best_metrics, best_volatility = full_eval_with_volatility(
            agent, seed, obs_ablation, EVAL_EPS
        )

        summary = {
            "condition": condition_key,
            "condition_label": spec["label"],
            "obs_ablation": obs_ablation,
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
        for prefix, metrics, vol in [
            ("final", final_metrics, final_volatility),
            ("best", best_metrics, best_volatility),
        ]:
            for m in METRICS:
                summary[f"{prefix}_{m}"] = metrics[m]
            for vk, vv in vol.items():
                summary[f"{prefix}_{vk}"] = vv

        agent.save(str(cfg.MODELS_DIR / tag))
    finally:
        _restore_gain(snap)

    return ckpt_rows, summary, best_volatility


def _step_from_path(path: str) -> int:
    for step in CHECKPOINT_STEPS:
        if f"_{step}_steps" in path:
            return step
    return TRAIN_TS


def run_experiment(seeds: List[int], resume: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary_path = OUT_DIR / "emotion_id_ablation_per_run.csv"
    ckpt_path = OUT_DIR / "checkpoint_eval_per_run.csv"

    summaries: List[Dict[str, Any]] = _load_resume(summary_path, resume)
    ckpt_rows: List[Dict[str, Any]] = _load_resume(ckpt_path, resume)

    done = {(r["condition"], int(r["seed"])) for r in summaries}
    total = len(CONDITIONS) * len(seeds)
    n_done = len(done)

    for condition_key in CONDITIONS:
        spec = CONDITIONS[condition_key]
        print(f"\n--- {spec['label']} ---")
        for seed in seeds:
            key = (condition_key, seed)
            if key in done:
                continue
            n_done += 1
            print(f"[{n_done}/{total}] seed={seed}")
            batch_ckpt, summary, _ = train_with_full_checkpoint_eval(condition_key, seed)
            ckpt_rows.extend(batch_ckpt)
            summaries.append(summary)
            pd.DataFrame(summaries).to_csv(summary_path, index=False)
            pd.DataFrame(ckpt_rows).to_csv(ckpt_path, index=False)
            print(
                f"  final_lg={summary['final_learning_gain']:.3f} "
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


def aggregate_condition(
    df: pd.DataFrame,
    condition_key: str,
    metric_prefix: str,
) -> Dict[str, Any]:
    sub = df[df["condition"] == condition_key]
    row: Dict[str, Any] = {
        "condition": condition_key,
        "condition_label": CONDITIONS[condition_key]["label"],
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


def pairwise_a_vs_d(df: pd.DataFrame, metric_prefix: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    p_vals: List[float] = []
    for m in METRICS:
        col = f"{metric_prefix}_{m}"
        v_a = df[df["condition"] == "A_full_emotion"][col].astype(float).tolist()
        v_d = df[df["condition"] == "D_no_emotion_id"][col].astype(float).tolist()
        if len(v_a) < 2 or len(v_d) < 2:
            continue
        t_stat, p_val = stats.ttest_ind(v_a, v_d, equal_var=False)
        row = {
            "comparison": "A_full_emotion_vs_D_no_emotion_id",
            "eval_type": metric_prefix,
            "metric": m,
            "mean_A": round(float(np.mean(v_a)), 4),
            "mean_D": round(float(np.mean(v_d)), 4),
            "delta_A_minus_D": round(float(np.mean(v_a) - np.mean(v_d)), 4),
            "std_A": round(float(np.std(v_a, ddof=1)), 4),
            "std_D": round(float(np.std(v_d, ddof=1)), 4),
            "cohens_d": round(cohens_d(v_a, v_d), 4),
            "t": round(float(t_stat), 4),
            "p_raw": round(float(p_val), 6),
        }
        rows.append(row)
        p_vals.append(p_val)

    if p_vals:
        order = np.argsort(p_vals)
        holm = [1.0] * len(p_vals)
        for rank, idx in enumerate(order):
            holm[idx] = min(1.0, p_vals[idx] * (len(p_vals) - rank))
        for i, row in enumerate(rows):
            row["p_holm"] = round(holm[i], 6)
    return pd.DataFrame(rows)


def checkpoint_selection_analysis(ckpt_df: pd.DataFrame, summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for condition_key in CONDITIONS:
        for eval_type in ("final", "best"):
            agg = aggregate_condition(summary_df, condition_key, eval_type)
            rows.append({"analysis": f"{eval_type}_checkpoint", **agg})
    return pd.DataFrame(rows)


def volatility_correlations(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Correlate emotion volatility with learning gain per condition and eval type."""
    rows: List[Dict[str, Any]] = []
    vol_cols = [
        ("mean_confusion_variance", "confusion_variance"),
        ("mean_boredom_variance", "boredom_variance"),
        ("mean_emotion_id_switches", "emotion_id_switches"),
    ]
    for condition_key in CONDITIONS:
        sub = summary_df[summary_df["condition"] == condition_key]
        for eval_type in ("final", "best"):
            lg_col = f"{eval_type}_learning_gain"
            lgs = sub[lg_col].astype(float).values
            for vol_col, vol_name in vol_cols:
                vc = f"{eval_type}_{vol_col}"
                if vc not in sub.columns:
                    continue
                vols = sub[vc].astype(float).values
                if len(lgs) < 3:
                    continue
                r_pearson, p_pearson = stats.pearsonr(lgs, vols)
                r_spearman, p_spearman = stats.spearmanr(lgs, vols)
                rows.append({
                    "condition": condition_key,
                    "eval_type": eval_type,
                    "volatility_metric": vol_name,
                    "pearson_r": round(float(r_pearson), 4),
                    "pearson_p": round(float(p_pearson), 6),
                    "spearman_r": round(float(r_spearman), 4),
                    "spearman_p": round(float(p_spearman), 6),
                    "n_seeds": len(lgs),
                })
    return pd.DataFrame(rows)


def seed_robustness_10_vs_15(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Compare 10-seed vs 15-seed aggregates for stability check."""
    rows: List[Dict[str, Any]] = []
    base_seeds = set(cfg.SEEDS)
    for condition_key in CONDITIONS:
        sub = summary_df[summary_df["condition"] == condition_key]
        for n_label, seed_set in [("10_seeds", base_seeds), ("15_seeds", set(SEEDS_15))]:
            ssub = sub[sub["seed"].isin(seed_set)]
            for eval_type in ("final", "best"):
                col = f"{eval_type}_learning_gain"
                vals = ssub[col].astype(float).tolist()
                if not vals:
                    continue
                mean, lo, hi = confidence_interval(vals)
                rows.append({
                    "condition": condition_key,
                    "seed_set": n_label,
                    "eval_type": eval_type,
                    "n": len(vals),
                    "learning_gain_mean": round(mean, 4),
                    "learning_gain_std": round(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, 4),
                    "learning_gain_ci": f"[{lo:.3f}, {hi:.3f}]",
                })
    return pd.DataFrame(rows)


def final_recommendation(
    summary_df: pd.DataFrame,
    pairwise_final: pd.DataFrame,
    pairwise_best: pd.DataFrame,
    vol_corr: pd.DataFrame,
    seed_robust: pd.DataFrame,
) -> Dict[str, Any]:
    def _get_agg(condition: str, eval_type: str) -> Dict[str, Any]:
        sub = summary_df[summary_df["condition"] == condition]
        col = f"{eval_type}_learning_gain"
        vals = sub[col].astype(float).tolist()
        mean, lo, hi = confidence_interval(vals)
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        adapt_col = f"{eval_type}_adaptation_accuracy"
        adapt_vals = sub[adapt_col].astype(float).tolist()
        adapt_mean, adapt_lo, adapt_hi = confidence_interval(adapt_vals)
        return {
            "learning_gain_mean": round(mean, 4),
            "learning_gain_std": round(std, 4),
            "learning_gain_ci": f"[{lo:.3f}, {hi:.3f}]",
            "adaptation_mean": round(adapt_mean, 4),
            "adaptation_ci": f"[{adapt_lo:.3f}, {adapt_hi:.3f}]",
        }

    a_final = _get_agg("A_full_emotion", "final")
    a_best = _get_agg("A_full_emotion", "best")
    d_final = _get_agg("D_no_emotion_id", "final")
    d_best = _get_agg("D_no_emotion_id", "best")

    lg_delta_final = a_final["learning_gain_mean"] - d_final["learning_gain_mean"]
    lg_delta_best = a_best["learning_gain_mean"] - d_best["learning_gain_mean"]

    # Pick winner: higher mean LG with checkpoint selection, tie-break lower std
    candidates = [
        ("A_full_emotion", "best", a_best),
        ("A_full_emotion", "final", a_final),
        ("D_no_emotion_id", "best", d_best),
        ("D_no_emotion_id", "final", d_final),
    ]
    best_cand = max(candidates, key=lambda c: (c[2]["learning_gain_mean"], -c[2]["learning_gain_std"]))

    # Q answers
    lg_row = pairwise_best[pairwise_best["metric"] == "learning_gain"]
    adapt_row = pairwise_best[pairwise_best["metric"] == "adaptation_accuracy"]
    lg_p = float(lg_row["p_holm"].iloc[0]) if len(lg_row) else 1.0
    adapt_p = float(adapt_row["p_holm"].iloc[0]) if len(adapt_row) else 1.0

    std_a = a_best["learning_gain_std"]
    std_d = d_best["learning_gain_std"]
    std_a_final = a_final["learning_gain_std"]
    std_d_final = d_final["learning_gain_std"]

    ckpt_reduces_a = std_a < std_a_final
    ckpt_reduces_d = std_d < std_d_final

    # Volatility explains poor seeds?
    conf_corr = vol_corr[
        (vol_corr["condition"] == "A_full_emotion")
        & (vol_corr["eval_type"] == "best")
        & (vol_corr["volatility_metric"] == "confusion_variance")
    ]
    bore_corr = vol_corr[
        (vol_corr["condition"] == "A_full_emotion")
        & (vol_corr["eval_type"] == "best")
        & (vol_corr["volatility_metric"] == "boredom_variance")
    ]
    eid_corr = vol_corr[
        (vol_corr["condition"] == "A_full_emotion")
        & (vol_corr["eval_type"] == "best")
        & (vol_corr["volatility_metric"] == "emotion_id_switches")
    ]

    def _vol_summary(corr_df: pd.DataFrame, expect_negative: bool = False) -> Optional[str]:
        if corr_df.empty:
            return None
        r = float(corr_df["spearman_r"].iloc[0])
        p = float(corr_df["spearman_p"].iloc[0])
        if p >= 0.05:
            return f"not significant (rho={r:.3f}, p={p:.4f})"
        direction = "negative" if r < 0 else "positive"
        return f"significant {direction} (rho={r:.3f}, p={p:.4f})"

    return {
        "Q1_emotion_id_improves_learning_gain": lg_delta_best > 0 and lg_p < 0.05,
        "Q1_delta_lg_best_checkpoint": round(lg_delta_best, 4),
        "Q1_p_holm": lg_p,
        "Q2_emotion_id_improves_adaptation": (
            float(adapt_row["delta_A_minus_D"].iloc[0]) > 0 if len(adapt_row) else False
        ),
        "Q2_adapt_p_holm": adapt_p,
        "Q3_removing_emotion_id_reduces_variance": std_d < std_a,
        "Q3_std_A_best": std_a,
        "Q3_std_D_best": std_d,
        "Q4_checkpoint_improves_stability": ckpt_reduces_a or ckpt_reduces_d,
        "Q4_std_reduction_A": round((1 - std_a / std_a_final) * 100, 1) if std_a_final > 0 else 0,
        "Q4_std_reduction_D": round((1 - std_d / std_d_final) * 100, 1) if std_d_final > 0 else 0,
        "Q5_confusion_volatility": _vol_summary(conf_corr),
        "Q5_boredom_volatility": _vol_summary(bore_corr, expect_negative=True),
        "Q5_emotion_id_switches": _vol_summary(eid_corr),
        "Q5_volatility_explains_poor_seeds": (
            (not bore_corr.empty and float(bore_corr["spearman_p"].iloc[0]) < 0.05
             and float(bore_corr["spearman_r"].iloc[0]) < 0)
        ),
        "recommended_model": {
            "condition": best_cand[0],
            "condition_label": CONDITIONS[best_cand[0]]["label"],
            "checkpoint_strategy": best_cand[1],
            "learning_gain_mean": best_cand[2]["learning_gain_mean"],
            "learning_gain_ci": best_cand[2]["learning_gain_ci"],
            "learning_gain_std": best_cand[2]["learning_gain_std"],
            "adaptation_mean": best_cand[2]["adaptation_mean"],
            "adaptation_ci": best_cand[2]["adaptation_ci"],
        },
        "justification": _build_justification(
            best_cand[0], best_cand[1], lg_delta_best, lg_p,
            std_a, std_d, ckpt_reduces_a, vol_corr,
        ),
    }


def _build_justification(
    condition: str,
    eval_type: str,
    lg_delta: float,
    lg_p: float,
    std_a: float,
    std_d: float,
    ckpt_helps: bool,
    vol_corr: pd.DataFrame,
) -> str:
    parts = []
    if abs(lg_delta) < 0.02 or lg_p >= 0.05:
        parts.append(
            "emotion_id does not significantly improve learning gain "
            f"(delta={lg_delta:+.4f}, p_holm={lg_p:.3f})"
        )
    elif lg_delta > 0:
        parts.append(f"emotion_id improves learning gain (delta={lg_delta:+.4f})")
    else:
        parts.append(f"removing emotion_id improves learning gain (delta={lg_delta:+.4f})")

    if std_d < std_a:
        parts.append(f"no_emotion_id reduces seed variance (std {std_d:.3f} vs {std_a:.3f})")
    else:
        parts.append(f"full_emotion has lower variance (std {std_a:.3f} vs {std_d:.3f})")

    if ckpt_helps:
        parts.append("checkpoint selection reduces variance vs final-50k model")

    conf = vol_corr[
        (vol_corr["condition"] == "A_full_emotion")
        & (vol_corr["volatility_metric"] == "confusion_variance")
    ]
    if not conf.empty and float(conf["spearman_p"].iloc[0]) < 0.05:
        parts.append(
            f"confusion volatility correlates with LG (rho={conf['spearman_r'].iloc[0]:.3f})"
        )

    parts.append(
        f"Final thesis model: {CONDITIONS[condition]['label']} with {eval_type} checkpoint selection"
    )
    return "; ".join(parts)


def analyze_results(
    summary_df: pd.DataFrame,
    ckpt_df: pd.DataFrame,
) -> Dict[str, Any]:
    summary_final = pd.DataFrame([
        aggregate_condition(summary_df, ck, "final") for ck in CONDITIONS
    ])
    summary_best = pd.DataFrame([
        aggregate_condition(summary_df, ck, "best") for ck in CONDITIONS
    ])

    pairwise_final = pairwise_a_vs_d(summary_df, "final")
    pairwise_best = pairwise_a_vs_d(summary_df, "best")
    vol_corr = volatility_correlations(summary_df)
    seed_robust = seed_robustness_10_vs_15(summary_df)
    ckpt_analysis = checkpoint_selection_analysis(ckpt_df, summary_df)
    recommendation = final_recommendation(
        summary_df, pairwise_final, pairwise_best, vol_corr, seed_robust
    )

    # Per-seed volatility table for thesis
    vol_per_seed = summary_df[[
        "condition", "seed",
        "final_learning_gain", "best_learning_gain",
        "final_mean_confusion_variance", "final_mean_boredom_variance", "final_mean_emotion_id_switches",
        "best_mean_confusion_variance", "best_mean_boredom_variance", "best_mean_emotion_id_switches",
        "best_checkpoint_step",
    ]].copy()

    report = {
        "study": "emotion_id_ablation_stability",
        "gain_ratio": G4_GAIN["ratio_label"],
        "seeds_15": SEEDS_15,
        "n_seeds": len(SEEDS_15),
        "dqn_hyperparameters": FINAL_DQN_HP,
        "conditions": CONDITIONS,
        "checkpoint_steps": CHECKPOINT_STEPS,
        "summary_final_checkpoint": summary_final.to_dict("records"),
        "summary_best_checkpoint": summary_best.to_dict("records"),
        "pairwise_final": pairwise_final.to_dict("records"),
        "pairwise_best": pairwise_best.to_dict("records"),
        "volatility_correlations": vol_corr.to_dict("records"),
        "seed_robustness_10_vs_15": seed_robust.to_dict("records"),
        "final_recommendation": recommendation,
    }

    summary_final.to_csv(OUT_DIR / "summary_final_checkpoint.csv", index=False)
    summary_best.to_csv(OUT_DIR / "summary_best_checkpoint.csv", index=False)
    pairwise_final.to_csv(OUT_DIR / "pairwise_final_checkpoint.csv", index=False)
    pairwise_best.to_csv(OUT_DIR / "pairwise_best_checkpoint.csv", index=False)
    vol_corr.to_csv(OUT_DIR / "volatility_correlations.csv", index=False)
    vol_per_seed.to_csv(OUT_DIR / "volatility_per_seed.csv", index=False)
    seed_robust.to_csv(OUT_DIR / "seed_robustness_10_vs_15.csv", index=False)
    ckpt_df.to_csv(OUT_DIR / "checkpoint_eval_per_run.csv", index=False)

    with open(OUT_DIR / "emotion_id_ablation_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    _write_markdown_report(report, summary_final, summary_best, pairwise_best, vol_corr, seed_robust)
    _plot_results(summary_df, summary_final, summary_best, vol_corr)
    return report


def _df_to_md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def _write_markdown_report(
    report: Dict[str, Any],
    summary_final: pd.DataFrame,
    summary_best: pd.DataFrame,
    pairwise_best: pd.DataFrame,
    vol_corr: pd.DataFrame,
    seed_robust: pd.DataFrame,
) -> None:
    rec = report["final_recommendation"]
    rm = rec["recommended_model"]
    lines = [
        "# Emotion-ID Ablation + DQN Stability Report",
        "",
        "## Methodology",
        "",
        "Two-condition DQN study isolating the discrete `emotion_id` observation channel.",
        "",
        "- **Condition A (Full Emotion):** knowledge, engagement, frustration, confusion, boredom, emotion_id",
        "- **Condition D (No Emotion ID):** knowledge, engagement, frustration, confusion, boredom",
        "",
        "Unchanged: reward function, emotional dynamics, action space, transitions, 50k training budget.",
        f"Gain ratio: **{G4_GAIN['ratio_label']}** | Seeds: **15** | Eval: {EVAL_EPS} episodes",
        f"Checkpoints evaluated: {CHECKPOINT_STEPS}",
        "",
        "## Results - Final Checkpoint (50k)",
        "",
    ]
    disp_f = summary_final[[
        "condition_label", "n_seeds",
        "learning_gain_mean", "learning_gain_std", "learning_gain_ci",
        "adaptation_accuracy_mean", "success_rate_mean",
    ]].rename(columns={
        "learning_gain_mean": "LG mean",
        "learning_gain_std": "LG std",
        "learning_gain_ci": "LG 95% CI",
        "adaptation_accuracy_mean": "Adapt acc",
        "success_rate_mean": "Success rate",
    })
    lines.append(_df_to_md(disp_f))

    lines.extend(["", "## Results - Best Checkpoint Selection", ""])
    disp_b = summary_best[[
        "condition_label", "n_seeds",
        "learning_gain_mean", "learning_gain_std", "learning_gain_ci",
        "adaptation_accuracy_mean", "success_rate_mean",
    ]].rename(columns={
        "learning_gain_mean": "LG mean",
        "learning_gain_std": "LG std",
        "learning_gain_ci": "LG 95% CI",
        "adaptation_accuracy_mean": "Adapt acc",
        "success_rate_mean": "Success rate",
    })
    lines.append(_df_to_md(disp_b))

    lines.extend(["", "## Pairwise A vs D (Best Checkpoint)", ""])
    for _, row in pairwise_best.iterrows():
        lines.append(
            f"- **{row['metric']}**: A={row['mean_A']:.4f} vs D={row['mean_D']:.4f}, "
            f"delta={row['delta_A_minus_D']:+.4f}, d={row['cohens_d']:.3f}, p_holm={row.get('p_holm', row['p_raw']):.4f}"
        )

    lines.extend(["", "## Seed Robustness (10 vs 15 seeds)", ""])
    lines.append(_df_to_md(seed_robust))

    lines.extend(["", "## Emotion Volatility Correlations", ""])
    if not vol_corr.empty:
        lines.append(_df_to_md(vol_corr))

    lines.extend([
        "",
        "## Research Questions",
        "",
        f"1. Does emotion_id improve learning gain? **{rec['Q1_emotion_id_improves_learning_gain']}** "
        f"(delta={rec['Q1_delta_lg_best_checkpoint']}, p_holm={rec['Q1_p_holm']:.4f})",
        f"2. Does emotion_id improve adaptation accuracy? **{rec['Q2_emotion_id_improves_adaptation']}** "
        f"(p_holm={rec['Q2_adapt_p_holm']:.4f})",
        f"3. Does removing emotion_id reduce variance? **{rec['Q3_removing_emotion_id_reduces_variance']}** "
        f"(std A={rec['Q3_std_A_best']:.3f}, std D={rec['Q3_std_D_best']:.3f})",
        f"4. Does checkpoint selection improve stability? **{rec['Q4_checkpoint_improves_stability']}** "
        f"(A std reduction {rec['Q4_std_reduction_A']}%, D {rec['Q4_std_reduction_D']}%)",
        f"5. Does emotional volatility explain poor seeds? **{rec['Q5_volatility_explains_poor_seeds']}**",
        f"   - Confusion: {rec['Q5_confusion_volatility']}",
        f"   - Boredom: {rec['Q5_boredom_volatility']}",
        f"   - Emotion ID switches: {rec['Q5_emotion_id_switches']}",
        "",
        "## Final Recommendation",
        "",
        f"**Thesis model:** {rm['condition_label']}",
        f"- Checkpoint strategy: **{rm['checkpoint_strategy']}** checkpoint",
        f"- Learning gain: **{rm['learning_gain_mean']:.3f}** (95% CI {rm['learning_gain_ci']}, std={rm['learning_gain_std']:.3f})",
        f"- Adaptation accuracy: **{rm['adaptation_mean']:.3f}** (95% CI {rm['adaptation_ci']})",
        "",
        rec.get("justification", ""),
        "",
        "## Discussion",
        "",
        "The emotion update audit identified confusion and boredom as the most volatile",
        "continuous channels (low AR persistence, frequent >0.2 step changes). The discrete",
        "`emotion_id` is derived from these channels and may add redundant or noisy information",
        "for value-based learning. This study tests whether removing it improves DQN stability",
        "without altering simulator dynamics.",
        "",
        "Reference: emotion update audit (`figures/emotion_update_audit/EMOTION_UPDATE_AUDIT.md`).",
        "",
    ])
    (OUT_DIR / "EMOTION_ID_ABLATION_REPORT.md").write_text("\n".join(lines))


def _plot_results(
    summary_df: pd.DataFrame,
    summary_final: pd.DataFrame,
    summary_best: pd.DataFrame,
    vol_corr: pd.DataFrame,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    colors = {"A_full_emotion": "#A8DADC", "D_no_emotion_id": "#E8A598"}
    cond_order = list(CONDITIONS.keys())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, sdf, title in zip(
        axes,
        [summary_final, summary_best],
        ["Final Checkpoint (50k)", "Best Checkpoint Selection"],
    ):
        xs = np.arange(len(cond_order))
        means = [sdf[sdf["condition"] == c]["learning_gain_mean"].iloc[0] for c in cond_order]
        stds = [sdf[sdf["condition"] == c]["learning_gain_std"].iloc[0] for c in cond_order]
        ax.bar(xs, means, yerr=stds, capsize=5, color=[colors[c] for c in cond_order], edgecolor="#64748B")
        ax.set_xticks(xs)
        ax.set_xticklabels(["A: Full", "D: No ID"])
        ax.set_ylabel("Learning Gain")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"Emotion-ID Ablation (G4 {G4_GAIN['ratio_label']}, 15 seeds, DQN)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "emotion_id_ablation_overview.png", dpi=150)
    plt.close(fig)

    # Volatility scatter for Condition A best checkpoint
    sub = summary_df[summary_df["condition"] == "A_full_emotion"]
    if len(sub) >= 3:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for ax, xcol, xlabel in zip(
            axes,
            ["best_mean_confusion_variance", "best_mean_boredom_variance", "best_mean_emotion_id_switches"],
            ["Confusion variance", "Boredom variance", "Emotion ID switches"],
        ):
            ax.scatter(sub[xcol], sub["best_learning_gain"], c="#4C72B0", edgecolors="#334155")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Learning Gain")
            ax.grid(alpha=0.3)
        fig.suptitle("Emotion Volatility vs Learning Gain (Condition A, best checkpoint)")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "volatility_vs_learning_gain.png", dpi=150)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Emotion-ID ablation + stability study")
    parser.add_argument("--phase", choices=["run", "analyze", "all"], default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    seeds = SEEDS_15
    if args.quick:
        seeds = [42, 7, 13]
        print("QUICK MODE: 3 seeds, 5k train, 50 eval")

    summary_path = OUT_DIR / "emotion_id_ablation_per_run.csv"
    if args.fresh and summary_path.exists() and args.phase in ("run", "all"):
        summary_path.unlink()
        ckpt_path = OUT_DIR / "checkpoint_eval_per_run.csv"
        if ckpt_path.exists():
            ckpt_path.unlink()

    global TRAIN_TS, EVAL_EPS, CHECKPOINT_STEPS
    if args.quick:
        TRAIN_TS = 5_000
        EVAL_EPS = 50
        CHECKPOINT_STEPS = [5_000]

    if args.phase in ("run", "all"):
        print("=== Emotion-ID Ablation: RUN ===")
        print(f"Seeds: {len(seeds)} | Conditions: {list(CONDITIONS.keys())}")
        run_experiment(seeds, resume=args.resume)

    if args.phase in ("analyze", "all"):
        if not summary_path.exists():
            raise SystemExit(f"No results at {summary_path}. Run with --phase run first.")
        summary_df = pd.read_csv(summary_path)
        ckpt_path = OUT_DIR / "checkpoint_eval_per_run.csv"
        ckpt_df = pd.read_csv(ckpt_path) if ckpt_path.exists() else pd.DataFrame()
        print("\n=== ANALYZE ===")
        report = analyze_results(summary_df, ckpt_df)
        rec = report["final_recommendation"]
        print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
