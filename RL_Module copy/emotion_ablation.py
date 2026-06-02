"""
Emotion ablation study: does affect-aware state improve tutoring outcomes?

Compares Random, ERT, and DQN under:
  - full_emotion (Condition A): knowledge + continuous affect + emotion_id
  - knowledge_only (Condition B): knowledge visible; affect channels zeroed in obs
  - optional per-channel ablations (engagement, frustration, confusion, boredom, emotion_id)

Uses thesis protocol: seeds [42,7,13,21,99,314], G4 gain ratio (10:1), 50k DQN train,
500 eval episodes. Reward, action space, and transition dynamics are unchanged; only
agent-facing observations and ERT rule visibility are ablated.

Usage (from repo root):
  export PYTHONPATH="$(pwd)"
  python -m RL_Module.emotion_ablation --phase all
  python -m RL_Module.emotion_ablation --phase run --resume
  python -m RL_Module.emotion_ablation --phase analyze
  python -m RL_Module.emotion_ablation --phase all --include-per-emotion
  python -m RL_Module.emotion_ablation --phase all --quick
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
from RL_Module.environment.student_env import OBS_ABLATION_DIMS, StudentEnv
from RL_Module.evaluation.metrics import (
    _adaptation_match,
    _is_success,
    action_frequency_report,
    confidence_interval,
    normalized_knowledge_gain,
)

OUT_DIR = _HERE / "figures" / "emotion_ablation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Thesis experiment protocol (matches thesis_gain_sensitivity.py)
SEEDS = [42, 7, 13, 21, 99, 314]
TRAIN_TS = 50_000
EVAL_EPS = 500
ALGORITHMS = ["Random", "ERT", "DQN"]

# G4 = 10:1 gain ratio - recommended by thesis gain sensitivity study
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
}

PRIMARY_CONDITIONS = ["full_emotion", "knowledge_only"]
OPTIONAL_CONDITIONS = [
    "no_engagement",
    "no_frustration",
    "no_confusion",
    "no_boredom",
    "no_emotion_id",
    "emotion_id_only",
]
ALL_CONDITIONS = PRIMARY_CONDITIONS + OPTIONAL_CONDITIONS

CONDITION_LABELS = {
    "full_emotion": "Full Emotion (A)",
    "knowledge_only": "Knowledge Only (B)",
    "no_engagement": "Ablate Engagement",
    "no_frustration": "Ablate Frustration",
    "no_confusion": "Ablate Confusion",
    "no_boredom": "Ablate Boredom",
    "no_emotion_id": "Ablate Emotion ID",
    "emotion_id_only": "Emotion ID Only",
}

METRICS = [
    "learning_gain",
    "final_knowledge",
    "success_rate",
    "adaptation_accuracy",
    "emotional_wellbeing",
    "mean_episode_reward",
    "dropout_rate",
]


def emotional_wellbeing(e: float, f: float, c: float, b: float) -> float:
    return float(np.clip(e - 0.35 * f - 0.35 * c - 0.20 * b, -1.0, 1.0))


def _patch_gain() -> Dict[str, float]:
    prev = dict(cfg.SIMULATOR_PARAMS)
    cfg.SIMULATOR_PARAMS = {
        "GAIN_CORRECT_FACTOR": G4_GAIN["GAIN_CORRECT_FACTOR"],
        "GAIN_INCORRECT_FACTOR": G4_GAIN["GAIN_INCORRECT_FACTOR"],
    }
    return prev


def _restore_gain(snapshot: Dict[str, float]) -> None:
    cfg.SIMULATOR_PARAMS = snapshot


def make_eval_env(seed: int, condition: str) -> StudentEnv:
    return StudentEnv(
        population_seed=seed,
        obs_ablation=condition,
    )


def evaluate_tutor(
    agent,
    env: StudentEnv,
    n_episodes: int,
    seed: int,
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
        "action_frequencies": freq_report["frequencies"],
        "dominance_detected": freq_report["dominance_detected"],
    }


def run_single(
    algorithm: str,
    condition: str,
    seed: int,
    train_ts: int,
    eval_eps: int,
) -> Dict[str, Any]:
    snap = _patch_gain()
    tag = f"{condition}_{algorithm}_s{seed}"
    t0 = time.time()
    try:
        cfg.set_all_seeds(seed)
        model_path = cfg.MODELS_DIR / f"emotion_ablation_{tag}"

        if algorithm == "DQN":
            env = make_env(
                seed=seed,
                obs_ablation=condition,
                algo_tag=f"emotion_ablation_{tag}",
            )
            agent = DQNAgent()
            agent.train(env, train_ts, seed, hyperparams=FINAL_DQN_HP)
            env.close()
            agent.save(str(model_path))
        elif algorithm == "ERT":
            agent = ExpertRuleBasedAgent(obs_ablation=condition)
        else:
            agent = RandomAgent(seed=seed)

        eval_env = make_eval_env(seed, condition)
        eval_env.set_algorithm_name(f"{algorithm}_{condition}")
        result = evaluate_tutor(agent, eval_env, eval_eps, seed)
        eval_env.close()
    finally:
        _restore_gain(snap)

    elapsed = time.time() - t0
    row = {
        "condition": condition,
        "condition_label": CONDITION_LABELS.get(condition, condition),
        "masked_dims": list(OBS_ABLATION_DIMS[condition]),
        "algorithm": algorithm,
        "seed": seed,
        "gain_ratio": G4_GAIN["ratio_label"],
        "train_timesteps": train_ts if algorithm == "DQN" else 0,
        "eval_episodes": eval_eps,
        "elapsed_seconds": round(elapsed, 1),
        **{m: result[m] for m in METRICS},
        "dominance_detected": result["dominance_detected"],
    }
    print(
        f"  {tag}: lg={row['learning_gain']:.3f} adapt={row['adaptation_accuracy']:.3f} "
        f"succ={row['success_rate']:.3f} reward={row['mean_episode_reward']:.3f} ({elapsed:.0f}s)"
    )
    return row


def run_experiment(
    conditions: List[str],
    seeds: List[int],
    train_ts: int,
    eval_eps: int,
    resume: bool = False,
) -> pd.DataFrame:
    csv_path = OUT_DIR / "emotion_ablation_per_run.csv"
    done_keys: set = set()
    per_run: List[Dict[str, Any]] = []

    if resume and csv_path.exists():
        existing = pd.read_csv(csv_path)
        per_run = existing.to_dict("records")
        for r in per_run:
            done_keys.add((r["condition"], r["algorithm"], int(r["seed"])))
        print(f"Resuming: {len(done_keys)} runs already complete.")

    total = len(conditions) * len(ALGORITHMS) * len(seeds)
    n_done = len(done_keys)

    for condition in conditions:
        print(f"\n--- {CONDITION_LABELS.get(condition, condition)} ---")
        for algorithm in ALGORITHMS:
            for seed in seeds:
                key = (condition, algorithm, seed)
                if key in done_keys:
                    continue
                n_done += 1
                print(f"[{n_done}/{total}] {algorithm} seed={seed}")
                row = run_single(algorithm, condition, seed, train_ts, eval_eps)
                per_run.append(row)
                pd.DataFrame(per_run).to_csv(csv_path, index=False)

    return pd.DataFrame(per_run)


def aggregate_runs(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"n_seeds": len(runs)}
    for m in METRICS:
        vals = [r[m] for r in runs]
        mean, lo, hi = confidence_interval(vals)
        out[f"{m}_mean"] = round(mean, 4)
        out[f"{m}_std"] = round(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, 4)
        out[f"{m}_ci_lo"] = round(lo, 4)
        out[f"{m}_ci_hi"] = round(hi, 4)
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


def pairwise_ablation_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Full vs knowledge_only per algorithm; Holm within each algorithm block."""
    rows: List[Dict[str, Any]] = []
    for algo in ALGORITHMS:
        block: List[Dict[str, Any]] = []
        p_vals: List[float] = []
        full = df[(df["condition"] == "full_emotion") & (df["algorithm"] == algo)]
        kno = df[(df["condition"] == "knowledge_only") & (df["algorithm"] == algo)]
        if full.empty or kno.empty:
            continue
        for metric in METRICS:
            v_full = full[metric].astype(float).tolist()
            v_kno = kno[metric].astype(float).tolist()
            if len(v_full) < 2 or len(v_kno) < 2:
                continue
            t_stat, p_val = stats.ttest_ind(v_full, v_kno, equal_var=False)
            d = cohens_d(v_full, v_kno)
            row = {
                "algorithm": algo,
                "comparison": "full_emotion vs knowledge_only",
                "metric": metric,
                "mean_full": round(float(np.mean(v_full)), 4),
                "mean_knowledge_only": round(float(np.mean(v_kno)), 4),
                "delta_full_minus_kno": round(float(np.mean(v_full) - np.mean(v_kno)), 4),
                "std_full": round(float(np.std(v_full, ddof=1)), 4),
                "std_kno": round(float(np.std(v_kno, ddof=1)), 4),
                "t": round(float(t_stat), 4),
                "p_raw": round(float(p_val), 6),
                "cohens_d": round(d, 4),
            }
            block.append(row)
            p_vals.append(p_val)
        if p_vals:
            holm = holm_correction(p_vals)
            for i, row in enumerate(block):
                row["p_holm"] = round(holm[i], 6)
            rows.extend(block)
    return pd.DataFrame(rows)


def per_emotion_contribution(df: pd.DataFrame, algo: str = "DQN") -> pd.DataFrame:
    """Compare full_emotion to each single-channel ablation for DQN."""
    rows: List[Dict[str, Any]] = []
    full = df[(df["condition"] == "full_emotion") & (df["algorithm"] == algo)]
    if full.empty:
        return pd.DataFrame(rows)

    channel_map = {
        "no_engagement": "engagement",
        "no_frustration": "frustration",
        "no_confusion": "confusion",
        "no_boredom": "boredom",
        "no_emotion_id": "emotion_id",
    }
    for cond, channel in channel_map.items():
        sub = df[(df["condition"] == cond) & (df["algorithm"] == algo)]
        if sub.empty:
            continue
        for metric in ["learning_gain", "adaptation_accuracy", "success_rate"]:
            v_full = full[metric].astype(float).tolist()
            v_abl = sub[metric].astype(float).tolist()
            if len(v_full) < 2 or len(v_abl) < 2:
                continue
            delta = float(np.mean(v_full) - np.mean(v_abl))
            _, p_val = stats.ttest_ind(v_full, v_abl, equal_var=False)
            rows.append({
                "algorithm": algo,
                "ablated_channel": channel,
                "condition": cond,
                "metric": metric,
                "mean_full": round(float(np.mean(v_full)), 4),
                "mean_ablated": round(float(np.mean(v_abl)), 4),
                "delta_full_minus_ablated": round(delta, 4),
                "cohens_d": round(cohens_d(v_full, v_abl), 4),
                "p_raw": round(float(p_val), 6),
                "interpretation": (
                    "channel contributes" if delta > 0 else "channel negligible or harmful"
                ),
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        p_vals = out["p_raw"].tolist()
        holm = holm_correction(p_vals)
        out["p_holm"] = [round(p, 6) for p in holm]
    return out


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for condition in df["condition"].unique():
        for algo in ALGORITHMS:
            runs = df[(df["condition"] == condition) & (df["algorithm"] == algo)].to_dict("records")
            if not runs:
                continue
            agg = aggregate_runs(runs)
            rows.append({
                "Condition": CONDITION_LABELS.get(condition, condition),
                "condition": condition,
                "Tutor": algo,
                "Learning Gain": agg["learning_gain_mean"],
                "LG_std": agg["learning_gain_std"],
                "LG_ci": f"[{agg['learning_gain_ci_lo']}, {agg['learning_gain_ci_hi']}]",
                "Final Knowledge": agg["final_knowledge_mean"],
                "Success Rate": agg["success_rate_mean"],
                "Adaptation": agg["adaptation_accuracy_mean"],
                "Wellbeing": agg["emotional_wellbeing_mean"],
                "Reward": agg["mean_episode_reward_mean"],
                "Dropout": agg["dropout_rate_mean"],
                "n_seeds": agg["n_seeds"],
            })
    return pd.DataFrame(rows)


def answer_research_questions(
    df: pd.DataFrame,
    pairwise: pd.DataFrame,
    contribution: pd.DataFrame,
) -> Dict[str, Any]:
    answers: Dict[str, Any] = {}

    def _sig(metric: str, algo: str) -> Optional[Dict[str, Any]]:
        sub = pairwise[
            (pairwise["algorithm"] == algo)
            & (pairwise["metric"] == metric)
        ]
        if sub.empty:
            return None
        row = sub.iloc[0]
        return {
            "delta": row["delta_full_minus_kno"],
            "d": row["cohens_d"],
            "p_holm": row.get("p_holm", row["p_raw"]),
            "significant": float(row.get("p_holm", row["p_raw"])) < 0.05,
        }

    for q, metric in [
        ("Q1_learning_gain", "learning_gain"),
        ("Q2_adaptation_accuracy", "adaptation_accuracy"),
        ("Q3_success_rate", "success_rate"),
    ]:
        answers[q] = {
            algo: _sig(metric, algo) for algo in ALGORITHMS
        }

    if not contribution.empty:
        lg = contribution[contribution["metric"] == "learning_gain"].copy()
        lg = lg.sort_values("delta_full_minus_ablated", ascending=False)
        answers["Q4_emotion_contributions"] = lg[
            ["ablated_channel", "delta_full_minus_ablated", "cohens_d", "p_holm"]
        ].to_dict("records")
    else:
        answers["Q4_emotion_contributions"] = []

    # Q5: justified if full beats knowledge_only for DQN or ERT on LG or adaptation
    dqn_lg = answers["Q1_learning_gain"].get("DQN")
    ert_adapt = answers["Q2_adaptation_accuracy"].get("ERT")
    justified = False
    reasons: List[str] = []
    if dqn_lg and dqn_lg["delta"] > 0:
        reasons.append(f"DQN learning gain +{dqn_lg['delta']:.4f} with emotions")
        if dqn_lg.get("significant"):
            justified = True
    if ert_adapt and ert_adapt["delta"] > 0:
        reasons.append(f"ERT adaptation +{ert_adapt['delta']:.4f} with emotions")
        if ert_adapt.get("significant"):
            justified = True
    answers["Q5_affect_aware_justified"] = {
        "justified": justified,
        "narrative": "; ".join(reasons) if reasons else "No consistent emotion benefit detected.",
    }
    return answers


def _df_to_markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(str(c) for c in cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def write_report(
    report: Dict[str, Any],
    summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    contribution: pd.DataFrame,
    answers: Dict[str, Any],
) -> None:
    primary = summary[summary["condition"].isin(PRIMARY_CONDITIONS)]
    display_cols = [
        "Condition", "Tutor", "Learning Gain", "Final Knowledge",
        "Success Rate", "Adaptation", "Wellbeing", "Reward",
    ]

    lines = [
        "# Emotion Ablation Study Report",
        "",
        f"Gain ratio: **{G4_GAIN['ratio_label']}** (G4 - thesis default)",
        f"Seeds: {SEEDS}",
        f"Train timesteps (DQN): {TRAIN_TS}",
        f"Eval episodes: {EVAL_EPS}",
        "",
        "## Methodology (template)",
        "",
        report["methodology_template"],
        "",
        "## Results Summary",
        "",
        _df_to_markdown_table(primary[display_cols]),
        "",
        "## Primary Ablation: Full vs Knowledge Only",
        "",
    ]
    for _, row in pairwise.iterrows():
        lines.append(
            f"- **{row['algorithm']}** [{row['metric']}]: "
            f"full={row['mean_full']:.4f} vs knowledge_only={row['mean_knowledge_only']:.4f}, "
            f"delta={row['delta_full_minus_kno']:+.4f}, d={row['cohens_d']:.3f}, "
            f"p_holm={row.get('p_holm', row['p_raw']):.4f}"
        )

    if not contribution.empty:
        lines.extend(["", "## Per-Emotion Contribution (DQN)", ""])
        lg = contribution[contribution["metric"] == "learning_gain"].sort_values(
            "delta_full_minus_ablated", ascending=False
        )
        for _, row in lg.iterrows():
            lines.append(
                f"- **{row['ablated_channel']}**: dLG={row['delta_full_minus_ablated']:+.4f}, "
                f"d={row['cohens_d']:.3f}, p_holm={row.get('p_holm', row['p_raw']):.4f}"
            )

    lines.extend([
        "",
        "## Research Questions",
        "",
        f"1. **Does emotion information improve learning gain?** "
        f"{_format_answer(answers.get('Q1_learning_gain', {}))}",
        f"2. **Does emotion information improve adaptation accuracy?** "
        f"{_format_answer(answers.get('Q2_adaptation_accuracy', {}))}",
        f"3. **Does emotion information improve success rate?** "
        f"{_format_answer(answers.get('Q3_success_rate', {}))}",
        f"4. **Which emotions contribute most?** See per-channel table / contribution CSV.",
        f"5. **Is affect-aware tutoring justified?** "
        f"{answers.get('Q5_affect_aware_justified', {}).get('narrative', 'N/A')}",
        "",
        "## Discussion (template)",
        "",
        report["discussion_template"],
        "",
        "## Limitations (template)",
        "",
        report["limitations_template"],
        "",
    ])
    (OUT_DIR / "EMOTION_ABLATION_REPORT.md").write_text("\n".join(lines))


def _format_answer(by_algo: Dict[str, Any]) -> str:
    parts = []
    for algo, info in by_algo.items():
        if info is None:
            continue
        sig = "*" if info.get("significant") else ""
        parts.append(f"{algo} d={info['delta']:+.4f}{sig}")
    return "; ".join(parts) if parts else "Insufficient data."


METHODOLOGY_TEMPLATE = """\
We conducted a controlled ablation on **agent-facing observations** only. The simulator
continued to evolve full internal affect (engagement, frustration, confusion, boredom,
emotion_id); rewards, actions, and transition dynamics were unchanged.

**Condition A (full_emotion):** observation vector
`[knowledge, engagement, frustration, confusion, boredom, emotion_id]`.

**Condition B (knowledge_only):** indices 1-5 zeroed; agents see `[knowledge, 0, 0, 0, 0, 0]`.

**Optional single-channel ablations:** one affect dimension zeroed at a time to rank
contributions (DQN retrained per condition; ERT rules tied to that channel disabled).

Three tutors were compared: **Random** (observation-agnostic), **ERT** (rule tiers
masked to match each condition), **DQN** (retrained per condition with identical
hyperparameters). Evaluation used 500 episodes per seed across seeds
{seeds} with gain ratio **{gain_ratio}** (G4).

**Adaptation accuracy** is computed post hoc against `BEST_ACTION_MAP` using the
*true* simulator emotion_id (not the ablated observation), isolating pedagogical
matching from observation availability.
"""

DISCUSSION_TEMPLATE = """\
Interpret full-vs-knowledge-only deltas jointly across tutors. A positive DQN delta on
learning gain supports that learned policies exploit affect channels; ERT deltas isolate
the value of hand-crafted affect rules. Random should show negligible deltas, validating
the ablation plumbing.

If adaptation accuracy drops sharply under knowledge_only while learning gain is stable,
agents may still learn via reward shaping without explicit emotion features - report both
metrics. Per-channel ablations rank which continuous signals DQN uses most; emotion_id-only
vs affect-only conditions separate discrete FER labels from dimensional affect.

Report effect sizes (Cohen's d) alongside p-values; Holm correction controls family-wise
error within each tutor's metric battery.
"""

LIMITATIONS_TEMPLATE = """\
- Ablation masks observations at the environment interface; it does not remove affect from
  latent simulator state or rewards (multi-objective reward still penalizes frustration).
- ERT knowledge_only retains only knowledge-tier rules (T4 + fallback); this is a strong
  baseline, not a naive random policy.
- DQN requires retraining per condition; sample efficiency differs from ERT/Random.
- Adaptation accuracy uses a fixed expert map - high accuracy does not guarantee learning
  gain if the map is misaligned with reward optima.
- Results are simulator-specific; human tutoring studies needed for external validity.
- Single-channel ablations are not fully orthogonal (affect dimensions co-evolve in transitions).
"""


def analyze_results(df: pd.DataFrame) -> Dict[str, Any]:
    summary = summary_table(df)
    pairwise = pairwise_ablation_tests(df)
    contribution = per_emotion_contribution(df, algo="DQN")
    answers = answer_research_questions(df, pairwise, contribution)

    agg: Dict[str, Dict[str, Any]] = {}
    for condition in df["condition"].unique():
        for algo in ALGORITHMS:
            runs = df[(df["condition"] == condition) & (df["algorithm"] == algo)].to_dict("records")
            if runs:
                agg[f"{condition}|{algo}"] = aggregate_runs(runs)

    report = {
        "study": "emotion_ablation",
        "gain_ratio": G4_GAIN["ratio_label"],
        "seeds": sorted(df["seed"].unique().tolist()),
        "conditions": sorted(df["condition"].unique().tolist()),
        "algorithms": ALGORITHMS,
        "dqn_hyperparameters": FINAL_DQN_HP,
        "obs_ablation_dims": {k: list(v) for k, v in OBS_ABLATION_DIMS.items()},
        "aggregated": agg,
        "research_answers": answers,
        "methodology_template": METHODOLOGY_TEMPLATE.format(
            seeds=SEEDS, gain_ratio=G4_GAIN["ratio_label"]
        ),
        "discussion_template": DISCUSSION_TEMPLATE,
        "limitations_template": LIMITATIONS_TEMPLATE,
    }

    summary.to_csv(OUT_DIR / "emotion_ablation_summary.csv", index=False)
    pairwise.to_csv(OUT_DIR / "pairwise_full_vs_knowledge_only.csv", index=False)
    if not contribution.empty:
        contribution.to_csv(OUT_DIR / "per_emotion_contribution_dqn.csv", index=False)

    with open(OUT_DIR / "emotion_ablation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    write_report(report, summary, pairwise, contribution, answers)
    _plot_results(summary, pairwise, contribution, answers)
    return report


def _plot_results(
    summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    contribution: pd.DataFrame,
    answers: Dict[str, Any],
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    colors = {"Random": "#999999", "ERT": "#4C72B0", "DQN": "#DD8452"}
    cond_colors = {"full_emotion": "#A8DADC", "knowledge_only": "#FFD6A5"}

    # 1. Grouped bar: learning gain by condition x tutor
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    primary = summary[summary["condition"].isin(PRIMARY_CONDITIONS)]

    for ax, metric, title in zip(
        axes.flat,
        ["Learning Gain", "Adaptation", "Success Rate", "Reward"],
        ["Learning Gain", "Adaptation Accuracy", "Success Rate", "Mean Episode Reward"],
    ):
        x = np.arange(len(ALGORITHMS))
        width = 0.35
        for i, cond in enumerate(PRIMARY_CONDITIONS):
            sub = primary[primary["condition"] == cond].set_index("Tutor")
            vals = [sub.loc[a, metric] if a in sub.index else 0 for a in ALGORITHMS]
            std_col = {
                "Learning Gain": "LG_std",
            }.get(metric, None)
            errs = None
            if std_col and std_col in sub.columns:
                errs = [sub.loc[a, std_col] if a in sub.index else 0 for a in ALGORITHMS]
            ax.bar(
                x + (i - 0.5) * width,
                vals,
                width,
                yerr=errs,
                capsize=4,
                label=CONDITION_LABELS[cond],
                color=cond_colors[cond],
                edgecolor="#64748B",
            )
        ax.set_xticks(x)
        ax.set_xticklabels(ALGORITHMS)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Emotion Ablation (G4 {G4_GAIN['ratio_label']}): Full vs Knowledge Only",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "emotion_ablation_overview.png", dpi=150)
    plt.close(fig)

    # 2. Delta heatmap: full - knowledge_only per metric x tutor
    if not pairwise.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        metrics_plot = ["learning_gain", "adaptation_accuracy", "success_rate", "mean_episode_reward"]
        mat = []
        for algo in ALGORITHMS:
            row = []
            for m in metrics_plot:
                sub = pairwise[(pairwise["algorithm"] == algo) & (pairwise["metric"] == m)]
                row.append(sub["delta_full_minus_kno"].iloc[0] if len(sub) else 0.0)
            mat.append(row)
        im = ax.imshow(mat, cmap="RdYlGn", aspect="auto", vmin=-0.15, vmax=0.15)
        ax.set_xticks(range(len(metrics_plot)))
        ax.set_xticklabels(["LG", "Adapt", "Success", "Reward"], rotation=15)
        ax.set_yticks(range(len(ALGORITHMS)))
        ax.set_yticklabels(ALGORITHMS)
        ax.set_title("Emotion Benefit (Full minus Knowledge Only)")
        for i in range(len(ALGORITHMS)):
            for j in range(len(metrics_plot)):
                ax.text(j, i, f"{mat[i][j]:+.3f}", ha="center", va="center", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "emotion_benefit_heatmap.png", dpi=150)
        plt.close(fig)

    # 3. Per-emotion contribution (DQN learning gain)
    if not contribution.empty:
        lg = contribution[contribution["metric"] == "learning_gain"].sort_values(
            "delta_full_minus_ablated", ascending=True
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(
            lg["ablated_channel"],
            lg["delta_full_minus_ablated"],
            color="#DD8452",
            edgecolor="#64748B",
        )
        ax.axvline(0, color="#334155", linewidth=0.8)
        ax.set_xlabel("Delta Learning Gain (Full minus Ablated)")
        ax.set_title("DQN: Per-Emotion Channel Contribution")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "per_emotion_contribution.png", dpi=150)
        plt.close(fig)

    # 4. Summary table PNG
    fig, ax = plt.subplots(figsize=(12, 4 + 0.4 * len(primary)))
    ax.axis("off")
    tbl_cols = ["Condition", "Tutor", "Learning Gain", "Adaptation", "Success Rate", "Reward"]
    tbl_data = primary[tbl_cols].values.tolist()
    tbl = ax.table(
        cellText=tbl_data,
        colLabels=tbl_cols,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.1, 1.5)
    q5 = answers.get("Q5_affect_aware_justified", {}).get("narrative", "")
    ax.set_title("Emotion Ablation Summary", fontsize=13, fontweight="bold", pad=16)
    fig.text(0.5, 0.02, q5, ha="center", fontsize=9, wrap=True)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "emotion_ablation_table.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Emotion ablation study (thesis)")
    parser.add_argument("--phase", choices=["run", "analyze", "all"], default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--include-per-emotion",
        action="store_true",
        help="Also run single-channel ablations (6 extra conditions x DQN train)",
    )
    parser.add_argument("--quick", action="store_true", help="2 seeds, 2k train, 50 eval")
    parser.add_argument("--fresh", action="store_true", help="Delete prior per-run CSV")
    args = parser.parse_args()

    conditions = list(PRIMARY_CONDITIONS)
    if args.include_per_emotion:
        conditions.extend(OPTIONAL_CONDITIONS)

    seeds = SEEDS
    train_ts = TRAIN_TS
    eval_eps = EVAL_EPS
    if args.quick:
        seeds = [42, 7]
        train_ts = 2_000
        eval_eps = 50
        print("QUICK MODE: 2 seeds, 2000 train steps, 50 eval episodes")

    csv_path = OUT_DIR / "emotion_ablation_per_run.csv"
    if args.fresh and csv_path.exists() and args.phase in ("run", "all"):
        csv_path.unlink()
        print(f"Removed prior results: {csv_path}")

    df: Optional[pd.DataFrame] = None
    if args.phase in ("run", "all"):
        print("=== Emotion Ablation: RUN ===")
        print(f"Conditions={conditions} seeds={seeds} gain={G4_GAIN['ratio_label']}")
        df = run_experiment(conditions, seeds, train_ts, eval_eps, resume=args.resume)

    if args.phase in ("analyze", "all"):
        if df is None:
            if not csv_path.exists():
                raise SystemExit(f"No results at {csv_path}. Run with --phase run first.")
            df = pd.read_csv(csv_path)
        print("\n=== Emotion Ablation: ANALYZE ===")
        report = analyze_results(df)
        print(f"\nReport: {OUT_DIR / 'EMOTION_ABLATION_REPORT.md'}")
        q = report["research_answers"]
        print("\n--- Research Answers ---")
        print(f"Q5: {q.get('Q5_affect_aware_justified', {})}")


if __name__ == "__main__":
    main()
