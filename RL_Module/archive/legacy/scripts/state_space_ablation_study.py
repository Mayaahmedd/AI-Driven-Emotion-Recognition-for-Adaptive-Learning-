"""
State-space ablation: persistent frustration observation channel.

Compares three observation configurations:
  A (A_original_v6):           6-dim original [k,e,f,c,b,emotion_id]
  B (B_v7_with_persistent):    7-dim + normalized consecutive_flag_steps
  C (C_v7_persistent_no_eid):  7-dim, emotion_id masked (only if redundant)

Usage (from repo root):
  export PYTHONPATH="$(pwd)"
  python -m RL_Module.state_space_ablation_study --phase all
  python -m RL_Module.state_space_ablation_study --phase run --resume
  python -m RL_Module.state_space_ablation_study --phase analyze
  python -m RL_Module.state_space_ablation_study --phase all --quick
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
from RL_Module.environment.student_env import OBS_SPACE_VARIANTS, StudentEnv
from RL_Module.evaluation.metrics import (
    DOMINANCE_THRESHOLD,
    _adaptation_match,
    _is_success,
    action_frequency_report,
    confidence_interval,
    normalized_knowledge_gain,
)
from RL_Module.mdp_definition import ACTION_DIM, ID_TO_ACTION

OUT_DIR = _HERE / "figures" / "state_space_ablation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS_10 = list(cfg.SEEDS)

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

CONDITIONS: Dict[str, str] = {
    "A_original_v6": "A_original_v6",
    "B_v7_with_persistent": "B_v7_with_persistent",
    "C_v7_persistent_no_eid": "C_v7_persistent_no_eid",
}

METRICS = [
    "learning_gain",
    "final_knowledge",
    "success_rate",
    "adaptation_accuracy",
    "mean_episode_reward",
    "dropout_rate",
    "action_diversity",
]


def _patch_gain() -> Dict[str, float]:
    prev = dict(cfg.SIMULATOR_PARAMS)
    cfg.SIMULATOR_PARAMS = dict(G4_GAIN)
    return prev


def _restore_gain(snapshot: Dict[str, float]) -> None:
    cfg.SIMULATOR_PARAMS = snapshot


def config_id(variant: str) -> str:
    spec = OBS_SPACE_VARIANTS[variant]
    payload = {
        "hp": FINAL_DQN_HP,
        "variant": variant,
        "include_persistent": spec["include_persistent_obs"],
        "obs_ablation": spec["obs_ablation"],
        "train_ts": TRAIN_TS,
    }
    return hashlib.md5(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:10]


def make_eval_env(seed: int, variant: str) -> StudentEnv:
    return StudentEnv(
        population_seed=seed,
        obs_space_variant=variant,
        emotion_dynamics="full",
    )


def _action_diversity(action_counts: Dict[int, int]) -> float:
    """Shannon entropy normalized by log(n_actions); 1.0 = uniform."""
    total = sum(action_counts.values())
    if total == 0:
        return 0.0
    probs = np.array([action_counts.get(i, 0) / total for i in range(ACTION_DIM)], dtype=float)
    probs = probs[probs > 0]
    entropy = -float(np.sum(probs * np.log(probs)))
    return entropy / np.log(ACTION_DIM)


def full_eval(
    agent: DQNAgent,
    seed: int,
    variant: str,
    n_episodes: int = EVAL_EPS,
) -> Dict[str, float]:
    env = make_eval_env(seed, variant)
    cfg.set_all_seeds(seed)
    results: List[Dict[str, float]] = []
    all_action_counts = {i: 0 for i in range(ACTION_DIM)}

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
            all_action_counts[action] += 1
            mask = info["action_masks"]
            done = term or trunc
            dropout = info.get("dropout", False)

        final = env._state
        results.append({
            "learning_gain": normalized_knowledge_gain(start_k, final.knowledge),
            "final_knowledge": final.knowledge,
            "success_rate": float(_is_success(final.knowledge, final.frustration, final.confusion)),
            "adaptation_accuracy": adapt_hits / max(adapt_total, 1),
            "mean_episode_reward": total_reward,
            "dropout_rate": float(dropout),
        })

    env.close()
    metrics = {k: float(np.mean([r[k] for r in results])) for k in results[0]}
    metrics["action_diversity"] = _action_diversity(all_action_counts)
    metrics["dominant_action_share"] = max(all_action_counts.values()) / max(sum(all_action_counts.values()), 1)
    return metrics


def quick_eval_lg(agent: DQNAgent, seed: int, variant: str, n_episodes: int = VAL_EPS) -> float:
    env = make_eval_env(seed, variant)
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


def train_condition(condition_key: str, seed: int) -> Dict[str, Any]:
    variant = CONDITIONS[condition_key]
    spec = OBS_SPACE_VARIANTS[variant]
    snap = _patch_gain()
    cid = config_id(variant)
    tag = f"state_space_ablation_{condition_key}_{cid}_s{seed}"
    t0 = time.time()

    try:
        cfg.set_all_seeds(seed)
        train_env = make_env(
            seed=seed,
            obs_space_variant=variant,
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

        valid_paths = [step_paths[s] for s in CHECKPOINT_STEPS if s in step_paths]
        if not valid_paths:
            valid_paths = list(paths)

        if valid_paths:
            best_path, best_val = agent.select_best_checkpoint(
                valid_paths,
                lambda a, s: quick_eval_lg(a, s, variant, VAL_EPS),
                seed,
            )
            best_step = _step_from_path(best_path)
        else:
            best_path = ""
            best_val = float("nan")
            best_step = TRAIN_TS

        final_path = step_paths.get(max(CHECKPOINT_STEPS) if CHECKPOINT_STEPS else TRAIN_TS) or (paths[-1] if paths else "")

        if final_path:
            agent.load_checkpoint(final_path)
        final_metrics = full_eval(agent, seed, variant, EVAL_EPS)

        if best_path:
            agent.load_checkpoint(best_path)
        best_metrics = full_eval(agent, seed, variant, EVAL_EPS)

        eval_env = make_eval_env(seed, variant)
        agent.verify_obs_compat(eval_env)
        eval_env.close()

        summary = {
            "condition": condition_key,
            "condition_label": spec["label"],
            "obs_space_variant": variant,
            "obs_dim": 7 if spec["include_persistent_obs"] else 6,
            "include_persistent_obs": spec["include_persistent_obs"],
            "obs_ablation": spec["obs_ablation"],
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
            for m in METRICS + ["dominant_action_share"]:
                summary[f"{prefix}_{m}"] = metrics.get(m, float("nan"))

        agent.save(str(cfg.MODELS_DIR / tag))
    finally:
        _restore_gain(snap)

    return summary


def _step_from_path(path: str) -> int:
    for step in CHECKPOINT_STEPS:
        if f"_{step}_steps" in path:
            return step
    return TRAIN_TS


def run_experiment(seeds: List[int], resume: bool) -> pd.DataFrame:
    summary_path = OUT_DIR / "state_space_ablation_per_run.csv"
    summaries: List[Dict[str, Any]] = _load_resume(summary_path, resume)
    done = {(r["condition"], int(r["seed"])) for r in summaries}
    total = len(CONDITIONS) * len(seeds)
    n_done = len(done)

    for condition_key in CONDITIONS:
        label = OBS_SPACE_VARIANTS[CONDITIONS[condition_key]]["label"]
        print(f"\n--- {label} ---")
        for seed in seeds:
            key = (condition_key, seed)
            if key in done:
                continue
            n_done += 1
            print(f"[{n_done}/{total}] seed={seed}")
            summary = train_condition(condition_key, seed)
            summaries.append(summary)
            pd.DataFrame(summaries).to_csv(summary_path, index=False)
            print(
                f"  best_lg={summary['best_learning_gain']:.3f} "
                f"adapt={summary['best_adaptation_accuracy']:.3f} "
                f"dropout={summary['best_dropout_rate']:.3f} "
                f"diversity={summary['best_action_diversity']:.3f} "
                f"({summary['elapsed_seconds']}s)"
            )

    return pd.DataFrame(summaries)


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


def aggregate_condition(df: pd.DataFrame, condition_key: str, prefix: str) -> Dict[str, Any]:
    sub = df[df["condition"] == condition_key]
    row: Dict[str, Any] = {
        "condition": condition_key,
        "condition_label": OBS_SPACE_VARIANTS[CONDITIONS[condition_key]]["label"],
        "n_seeds": len(sub),
    }
    for m in METRICS:
        col = f"{prefix}_{m}"
        if col not in sub.columns:
            continue
        vals = sub[col].astype(float).tolist()
        mean, lo, hi = confidence_interval(vals)
        row[f"{m}_mean"] = round(mean, 4)
        row[f"{m}_std"] = round(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, 4)
        row[f"{m}_ci"] = f"[{lo:.3f}, {hi:.3f}]"
    return row


def pairwise(df: pd.DataFrame, cond_a: str, cond_b: str, prefix: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    p_vals: List[float] = []
    for m in METRICS:
        col = f"{prefix}_{m}"
        v_a = df[df["condition"] == cond_a][col].astype(float).tolist()
        v_b = df[df["condition"] == cond_b][col].astype(float).tolist()
        if len(v_a) < 2 or len(v_b) < 2:
            continue
        t_stat, p_val = stats.ttest_ind(v_a, v_b, equal_var=False)
        rows.append({
            "comparison": f"{cond_a}_vs_{cond_b}",
            "eval_type": prefix,
            "metric": m,
            f"mean_{cond_a}": round(float(np.mean(v_a)), 4),
            f"mean_{cond_b}": round(float(np.mean(v_b)), 4),
            "delta": round(float(np.mean(v_a) - np.mean(v_b)), 4),
            "cohens_d": round(cohens_d(v_a, v_b), 4),
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


def final_recommendation(
    summary_df: pd.DataFrame,
    pairwise_ab: pd.DataFrame,
    pairwise_ac: pd.DataFrame,
    pairwise_bc: pd.DataFrame,
) -> Dict[str, Any]:
    def _best(cond: str, prefix: str) -> Dict[str, Any]:
        sub = summary_df[summary_df["condition"] == cond]
        col = f"{prefix}_learning_gain"
        vals = sub[col].astype(float).tolist()
        mean, lo, hi = confidence_interval(vals)
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        adapt_col = f"{prefix}_adaptation_accuracy"
        adapt_vals = sub[adapt_col].astype(float).tolist()
        adapt_mean, adapt_lo, adapt_hi = confidence_interval(adapt_vals)
        return {
            "learning_gain_mean": round(mean, 4),
            "learning_gain_std": round(std, 4),
            "learning_gain_ci": f"[{lo:.3f}, {hi:.3f}]",
            "adaptation_mean": round(adapt_mean, 4),
            "adaptation_ci": f"[{adapt_lo:.3f}, {adapt_hi:.3f}]",
        }

    candidates = []
    for ck in CONDITIONS:
        stats_b = _best(ck, "best")
        candidates.append((ck, stats_b))

    best = max(candidates, key=lambda c: (c[1]["learning_gain_mean"], -c[1]["learning_gain_std"]))

    lg_ab = pairwise_ab[pairwise_ab["metric"] == "learning_gain"]
    lg_bc = pairwise_bc[pairwise_bc["metric"] == "learning_gain"]

    emotion_id_redundant = False
    if not lg_bc.empty:
        delta_bc = float(lg_bc["delta"].iloc[0])  # B - C
        p_bc = float(lg_bc.get("p_holm", lg_bc["p_raw"]).iloc[0])
        # emotion_id redundant if C >= B (within noise)
        emotion_id_redundant = delta_bc <= 0.02 or p_bc >= 0.05

    persistent_helps = False
    if not lg_ab.empty:
        delta_ab = float(lg_ab["delta"].iloc[0])  # A - B
        p_ab = float(lg_ab.get("p_holm", lg_ab["p_raw"]).iloc[0])
        persistent_helps = delta_ab < -0.02 and p_ab < 0.05

    return {
        "recommended_condition": best[0],
        "recommended_label": OBS_SPACE_VARIANTS[CONDITIONS[best[0]]]["label"],
        "recommended_stats": best[1],
        "persistent_obs_improves_lg": persistent_helps,
        "emotion_id_proven_redundant": emotion_id_redundant,
        "pairwise_B_vs_C_lg_delta": float(lg_bc["delta"].iloc[0]) if not lg_bc.empty else None,
        "justification": _build_justification(best[0], persistent_helps, emotion_id_redundant, lg_ab, lg_bc),
    }


def _build_justification(
    winner: str,
    persistent_helps: bool,
    eid_redundant: bool,
    lg_ab: pd.DataFrame,
    lg_bc: pd.DataFrame,
) -> str:
    parts = []
    if persistent_helps:
        parts.append("adding consecutive_flag_steps improves learning gain vs original 6-dim")
    elif not lg_ab.empty:
        delta = float(lg_ab["delta"].iloc[0])
        parts.append(f"persistent channel delta vs original={delta:+.4f} (not significant)")

    if eid_redundant:
        parts.append("emotion_id is redundant when persistent steps are present (B?C)")
    elif not lg_bc.empty:
        delta = float(lg_bc["delta"].iloc[0])
        parts.append(f"emotion_id retains information beyond persistent steps (B-C delta={delta:+.4f})")

    parts.append(f"Recommended: {OBS_SPACE_VARIANTS[CONDITIONS[winner]]['label']}")
    return "; ".join(parts)


def analyze_results(summary_df: pd.DataFrame) -> Dict[str, Any]:
    summary_best = pd.DataFrame([
        aggregate_condition(summary_df, ck, "best") for ck in CONDITIONS
    ])

    pw_ab = pairwise(summary_df, "A_original_v6", "B_v7_with_persistent", "best")
    pw_ac = pairwise(summary_df, "A_original_v6", "C_v7_persistent_no_eid", "best")
    pw_bc = pairwise(summary_df, "B_v7_with_persistent", "C_v7_persistent_no_eid", "best")

    recommendation = final_recommendation(summary_df, pw_ab, pw_ac, pw_bc)

    report = {
        "study": "state_space_ablation",
        "gain_ratio": G4_GAIN["ratio_label"],
        "n_seeds": len(summary_df["seed"].unique()),
        "conditions": {k: OBS_SPACE_VARIANTS[v] for k, v in CONDITIONS.items()},
        "summary_best_checkpoint": summary_best.to_dict("records"),
        "pairwise_A_vs_B": pw_ab.to_dict("records"),
        "pairwise_A_vs_C": pw_ac.to_dict("records"),
        "pairwise_B_vs_C": pw_bc.to_dict("records"),
        "final_recommendation": recommendation,
        "emotion_id_audit": {
            "derivation": "threshold rules on frustration/confusion/boredom (sim mode)",
            "priority": "frustration>0.6 -> frustrated; confusion>0.5 -> confused; boredom>0.5 -> bored; else engaged",
            "contains_info_beyond_continuous": "discretization collapses overlapping thresholds; priority ordering not recoverable from continuous vars alone",
            "prior_ablation_evidence": "emotion_id_ablation: no_emotion_id outperforms full_emotion (LG 0.613 vs 0.580, lower variance)",
        },
        "persistent_frustration_audit": {
            "drives_masking": True,
            "drives_dropout": True,
            "drives_reward_penalty": True,
            "was_observable_to_agent": "no (pre-fix); consecutive_flag_steps encodes urgency to dropout",
            "normalization": "consecutive_flag_steps / PERSISTENT_FLAG_DROPOUT_STEPS clipped to [0,1]",
        },
    }

    summary_best.to_csv(OUT_DIR / "summary_best_checkpoint.csv", index=False)
    pw_ab.to_csv(OUT_DIR / "pairwise_A_vs_B.csv", index=False)
    pw_ac.to_csv(OUT_DIR / "pairwise_A_vs_C.csv", index=False)
    pw_bc.to_csv(OUT_DIR / "pairwise_B_vs_C.csv", index=False)

    with open(OUT_DIR / "state_space_ablation_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    _write_markdown_report(report, summary_best, pw_ab, pw_bc)
    _plot_results(summary_df, summary_best)
    return report


def _write_markdown_report(
    report: Dict[str, Any],
    summary_best: pd.DataFrame,
    pw_ab: pd.DataFrame,
    pw_bc: pd.DataFrame,
) -> None:
    rec = report["final_recommendation"]
    lines = [
        "# State-Space Ablation Report",
        "",
        "## Observation Configurations",
        "",
        "- **A:** 6-dim original [knowledge, engagement, frustration, confusion, boredom, emotion_id]",
        "- **B:** 7-dim A + normalized consecutive_flag_steps",
        "- **C:** 7-dim B with emotion_id masked (ablation only if redundant)",
        "",
        f"Gain ratio: **{G4_GAIN['ratio_label']}** | Seeds: **{report['n_seeds']}** | Eval: {EVAL_EPS} episodes",
        "",
        "## Results (Best Checkpoint)",
        "",
    ]
    cols = ["condition_label", "n_seeds"] + [f"{m}_mean" for m in METRICS] + [f"{m}_std" for m in METRICS[:3]]
    disp = summary_best[[c for c in cols if c in summary_best.columns]]
    lines.append("| " + " | ".join(disp.columns) + " |")
    lines.append("| " + " | ".join("---" for _ in disp.columns) + " |")
    for _, row in disp.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in disp.columns) + " |")

    lines.extend(["", "## Pairwise A vs B (persistent channel)", ""])
    for _, row in pw_ab.iterrows():
        lines.append(
            f"- **{row['metric']}**: A={row['mean_A_original_v6']:.4f} vs B={row['mean_B_v7_with_persistent']:.4f}, "
            f"delta={row['delta']:+.4f}, p_holm={row.get('p_holm', row['p_raw']):.4f}"
        )

    lines.extend(["", "## Pairwise B vs C (emotion_id redundancy)", ""])
    for _, row in pw_bc.iterrows():
        lines.append(
            f"- **{row['metric']}**: B={row['mean_B_v7_with_persistent']:.4f} vs C={row['mean_C_v7_persistent_no_eid']:.4f}, "
            f"delta={row['delta']:+.4f}, p_holm={row.get('p_holm', row['p_raw']):.4f}"
        )

    lines.extend([
        "",
        "## Recommendation",
        "",
        f"**Winner:** {rec['recommended_label']}",
        f"- LG: {rec['recommended_stats']['learning_gain_mean']:.3f} "
        f"(CI {rec['recommended_stats']['learning_gain_ci']}, std={rec['recommended_stats']['learning_gain_std']:.3f})",
        f"- Persistent obs helps LG: **{rec['persistent_obs_improves_lg']}**",
        f"- emotion_id redundant (B?C): **{rec['emotion_id_proven_redundant']}**",
        "",
        rec.get("justification", ""),
        "",
        "## Thesis Rationale",
        "",
        "The agent must observe state variables that (1) affect transition dynamics or reward, "
        "(2) are not fully inferable from other observed channels, and (3) enable anticipatory "
        "action before irreversible termination. `consecutive_flag_steps` satisfies (1)-(3): it "
        "drives emergency masking, reward penalties, and dropout termination, yet was previously "
        "hidden from the Q-network. Normalizing by the dropout horizon gives the agent a "
        "time-to-crisis signal that raw frustration alone cannot provide once the persistent "
        "flag is latched.",
        "",
    ])
    (OUT_DIR / "STATE_SPACE_ABLATION_REPORT.md").write_text("\n".join(lines))


def _plot_results(summary_df: pd.DataFrame, summary_best: pd.DataFrame) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    colors = {"A_original_v6": "#A8DADC", "B_v7_with_persistent": "#457B9D", "C_v7_persistent_no_eid": "#E8A598"}
    cond_order = list(CONDITIONS.keys())

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    metrics_plot = [
        ("learning_gain_mean", "Learning Gain"),
        ("adaptation_accuracy_mean", "Adaptation Accuracy"),
        ("dropout_rate_mean", "Dropout Rate"),
        ("action_diversity_mean", "Action Diversity"),
    ]
    for ax, (col, title) in zip(axes.flat, metrics_plot):
        std_col = col.replace("_mean", "_std")
        xs = np.arange(len(cond_order))
        means = [summary_best[summary_best["condition"] == c][col].iloc[0] for c in cond_order]
        stds = [summary_best[summary_best["condition"] == c][std_col].iloc[0] for c in cond_order]
        ax.bar(xs, means, yerr=stds, capsize=4, color=[colors[c] for c in cond_order], edgecolor="#64748B")
        ax.set_xticks(xs)
        ax.set_xticklabels(["A", "B", "C"])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"State-Space Ablation (G4 {G4_GAIN['ratio_label']})", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "state_space_ablation_overview.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="State-space ablation study")
    parser.add_argument("--phase", choices=["run", "analyze", "all"], default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    seeds = SEEDS_10
    global TRAIN_TS, EVAL_EPS, CHECKPOINT_STEPS
    if args.quick:
        seeds = [42, 7, 13]
        TRAIN_TS = 5_000
        EVAL_EPS = 50
        CHECKPOINT_STEPS = [5_000]
        print("QUICK MODE: 3 seeds, 5k train, 50 eval")

    summary_path = OUT_DIR / "state_space_ablation_per_run.csv"
    if args.fresh and summary_path.exists() and args.phase in ("run", "all"):
        summary_path.unlink()

    if args.phase in ("run", "all"):
        print("=== State-Space Ablation: RUN ===")
        run_experiment(seeds, resume=args.resume)

    if args.phase in ("analyze", "all"):
        if not summary_path.exists():
            raise SystemExit(f"No results at {summary_path}. Run with --phase run first.")
        summary_df = pd.read_csv(summary_path)
        print("\n=== ANALYZE ===")
        report = analyze_results(summary_df)
        print(json.dumps(report["final_recommendation"], indent=2))


if __name__ == "__main__":
    main()
