"""
Final thesis ablation: Condition A (Emotion-Aware DQN) vs Condition C (Knowledge-Only DQN).

Condition A - Emotion-Aware DQN (frozen thesis model):
  obs: knowledge, engagement, frustration, confusion, boredom (no emotion_id)
  env: default emotional dynamics, reward, transitions

Condition C - Knowledge-Only DQN (strict no-emotion):
  obs: knowledge only
  env: strict_off (validated emotion-leakage-free implementation)

DQN: lr=0.001, batch=64, buffer=100k, net=[64,64], 50k steps
Checkpoints: 10k-50k; best selected by validation learning gain
Seeds: matched 20-seed set from final_dqn_model_study

Usage (from repo root):
  export PYTHONPATH="$(pwd)"
  python -m RL_Module.final_ac_ablation --phase all
  python -m RL_Module.final_ac_ablation --phase run --resume
  python -m RL_Module.final_ac_ablation --phase analyze
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
from RL_Module.environment.student_env import StudentEnv
from RL_Module.evaluation.metrics import (
    _adaptation_match,
    _is_success,
    confidence_interval,
    normalized_knowledge_gain,
)
from RL_Module.final_dqn_model_study import (
    BASELINE_KEY,
    CHECKPOINT_STEPS,
    EVAL_EPS,
    FINAL_DQN_HP,
    G4_GAIN,
    METRICS,
    SEEDS_20,
    TRAIN_TS,
    VAL_EPS,
    aggregate_runs,
    checkpoint_selection_table,
    cohens_d,
    load_baseline_thesis,
    pairwise_compare,
)

OUT_DIR = _HERE / "figures" / "final_ac_ablation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONDITION_A_KEY = "A_emotion_aware"
CONDITION_C_KEY = "C_knowledge_only_strict"
CONDITION_A_LABEL = "A: Emotion-Aware DQN"
CONDITION_C_LABEL = "C: Knowledge-Only DQN (Strict No-Emotion)"

CONDITION_A_OBS = "no_emotion_id"
CONDITION_C_OBS = "knowledge_only"
CONDITION_C_DYNAMICS = "strict_off"


def _patch_gain() -> Dict[str, Any]:
    prev = dict(cfg.SIMULATOR_PARAMS)
    cfg.SIMULATOR_PARAMS = dict(G4_GAIN)
    return prev


def _restore_gain(snapshot: Dict[str, Any]) -> None:
    cfg.SIMULATOR_PARAMS = snapshot


def make_eval_env_a(seed: int) -> StudentEnv:
    return StudentEnv(
        population_seed=seed,
        obs_ablation=CONDITION_A_OBS,
        emotion_dynamics="full",
    )


def make_eval_env_c(seed: int) -> StudentEnv:
    return StudentEnv(
        population_seed=seed,
        obs_ablation=CONDITION_C_OBS,
        emotion_dynamics=CONDITION_C_DYNAMICS,
    )


def quick_eval_lg(
    agent: DQNAgent,
    seed: int,
    env_factory,
    n_episodes: int = VAL_EPS,
) -> float:
    env = env_factory(seed)
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
    env_factory,
    n_episodes: int = EVAL_EPS,
) -> Dict[str, float]:
    env = env_factory(seed)
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


def train_condition_c_seed(seed: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    snap = _patch_gain()
    tag = f"final_ac_C_s{seed}"
    ckpt_rows: List[Dict[str, Any]] = []
    t0 = time.time()

    try:
        cfg.set_all_seeds(seed)
        train_env = make_env(
            seed=seed,
            obs_ablation=CONDITION_C_OBS,
            emotion_dynamics=CONDITION_C_DYNAMICS,
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
            metrics = full_eval(agent, seed, make_eval_env_c)
            ckpt_rows.append({
                "model": CONDITION_C_KEY,
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
                lambda a, s: quick_eval_lg(a, s, make_eval_env_c, VAL_EPS),
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
        final_metrics = full_eval(agent, seed, make_eval_env_c)

        if best_path:
            agent.load_checkpoint(best_path)
        best_metrics = full_eval(agent, seed, make_eval_env_c)

        summary = {
            "model": CONDITION_C_KEY,
            "model_label": CONDITION_C_LABEL,
            "obs_ablation": CONDITION_C_OBS,
            "emotion_dynamics": CONDITION_C_DYNAMICS,
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
        _restore_gain(snap)

    return ckpt_rows, summary


def load_condition_a(seeds: List[int]) -> pd.DataFrame:
    df = load_baseline_thesis(seeds)
    df = df.copy()
    df["model"] = CONDITION_A_KEY
    df["model_label"] = CONDITION_A_LABEL
    return df


def run_condition_c(
    seeds: List[int],
    resume: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary_path = OUT_DIR / "condition_c_per_run.csv"
    ckpt_path = OUT_DIR / "condition_c_checkpoint_eval.csv"

    summaries: List[Dict[str, Any]] = []
    ckpt_rows: List[Dict[str, Any]] = []

    if resume and summary_path.exists():
        summaries = pd.read_csv(summary_path).to_dict("records")
    if resume and ckpt_path.exists():
        ckpt_rows = pd.read_csv(ckpt_path).to_dict("records")

    done = {int(r["seed"]) for r in summaries}
    total = len(seeds)

    for seed in seeds:
        if seed in done:
            continue
        print(f"[C {len(done) + 1}/{total}] seed={seed}")
        batch_ckpt, summary = train_condition_c_seed(seed)
        ckpt_rows.extend(batch_ckpt)
        summaries.append(summary)
        done.add(seed)
        pd.DataFrame(summaries).to_csv(summary_path, index=False)
        pd.DataFrame(ckpt_rows).to_csv(ckpt_path, index=False)
        print(
            f"  val_lg={summary['best_val_learning_gain']:.3f} "
            f"best_lg={summary['best_learning_gain']:.3f} "
            f"@step={summary['best_checkpoint_step']} "
            f"({summary['elapsed_seconds']}s)"
        )

    return pd.DataFrame(summaries), pd.DataFrame(ckpt_rows)


def _normalize_condition_a(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["model"] = CONDITION_A_KEY
    out["model_label"] = CONDITION_A_LABEL
    out["obs_ablation"] = CONDITION_A_OBS
    out["emotion_dynamics"] = "full"
    return out


def thesis_conclusion(
    summary_a: Dict[str, Any],
    summary_c: Dict[str, Any],
    pairwise: pd.DataFrame,
) -> Dict[str, Any]:
    lg_row = pairwise[pairwise["metric"] == "learning_gain"]
    lg_delta = float(lg_row["delta_A_minus_B"].iloc[0]) if len(lg_row) else None
    lg_d = float(lg_row["cohens_d"].iloc[0]) if len(lg_row) else None
    lg_p = float(lg_row["p_holm"].iloc[0]) if len(lg_row) else 1.0

    a_lg = summary_a["learning_gain_mean"]
    c_lg = summary_c["learning_gain_mean"]
    sig = lg_p < 0.05
    a_wins = a_lg > c_lg

    pct = None
    if c_lg > 1e-6:
        pct = round(100.0 * (a_lg - c_lg) / c_lg, 2)

    narrative_parts = [
        f"Emotion-aware tutoring (A) mean LG={a_lg:.4f} vs knowledge-only (C) LG={c_lg:.4f}",
        f"delta={lg_delta:+.4f}" if lg_delta is not None else "",
        f"Cohen's d={lg_d:.3f}" if lg_d is not None else "",
        f"Holm p={lg_p:.4f} ({'significant' if sig else 'not significant'})",
    ]
    if pct is not None:
        narrative_parts.append(f"relative improvement={pct:+.1f}%")

    return {
        "Q1_emotion_aware_outperforms_knowledge_only": a_wins,
        "Q1_delta_learning_gain_A_minus_C": lg_delta,
        "Q2_effect_size_cohens_d": lg_d,
        "Q3_statistically_significant": sig,
        "Q3_p_holm_learning_gain": lg_p,
        "Q4_pct_performance_from_emotion": pct,
        "Q5_emotional_component_justified": a_wins and (lg_delta is not None and lg_delta > 0.01),
        "condition_a_learning_gain_mean": a_lg,
        "condition_a_learning_gain_ci": summary_a["learning_gain_ci"],
        "condition_c_learning_gain_mean": c_lg,
        "condition_c_learning_gain_ci": summary_c["learning_gain_ci"],
        "final_thesis_verdict": (
            "Emotion + Knowledge OUTPERFORMS Knowledge Only"
            if a_wins and sig
            else (
                "Emotion + Knowledge OUTPERFORMS Knowledge Only (not statistically significant)"
                if a_wins
                else "Knowledge Only matches or exceeds Emotion + Knowledge"
            )
        ),
        "thesis_narrative": "; ".join(p for p in narrative_parts if p),
    }


def _plot_results(
    summary_a: Dict[str, Any],
    summary_c: Dict[str, Any],
    df_a: pd.DataFrame,
    df_c: pd.DataFrame,
    pairwise: pd.DataFrame,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    colors = {"A": "#4C72B0", "C": "#DD8452"}
    labels = ["A\nEmotion-Aware", "C\nKnowledge-Only"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    plot_metrics = [
        ("learning_gain", "Learning Gain"),
        ("final_knowledge", "Final Knowledge"),
        ("success_rate", "Success Rate"),
        ("adaptation_accuracy", "Adaptation Accuracy"),
        ("mean_episode_reward", "Mean Episode Reward"),
    ]

    for ax, (metric, title) in zip(axes.flat[:5], plot_metrics):
        means = [summary_a[f"{metric}_mean"], summary_c[f"{metric}_mean"]]
        stds = [summary_a[f"{metric}_std"], summary_c[f"{metric}_std"]]
        xs = np.arange(2)
        ax.bar(xs, means, yerr=stds, capsize=5, color=[colors["A"], colors["C"]], edgecolor="#64748B")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.3)

    ax = axes.flat[5]
    lg_row = pairwise[pairwise["metric"] == "learning_gain"]
    if not lg_row.empty:
        row = lg_row.iloc[0]
        ax.bar(
            ["A - C"],
            [row["delta_A_minus_B"]],
            color="#55A868" if row["delta_A_minus_B"] > 0 else "#C44E52",
            edgecolor="#64748B",
        )
        ax.axhline(0, color="#334155", lw=0.8)
        ax.set_title(f"Learning Gain Delta (d={row['cohens_d']:.2f}, p={row['p_holm']:.3f})")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Final Thesis Ablation: Emotion-Aware vs Knowledge-Only DQN\n"
        f"(n=20 seeds, best checkpoint, G4 {G4_GAIN['ratio_label']})",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "final_ac_ablation_overview.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    seeds = sorted(df_a["seed"].unique())
    a_vals = df_a.set_index("seed")["best_learning_gain"].reindex(seeds).values
    c_vals = df_c.set_index("seed")["best_learning_gain"].reindex(seeds).values
    xs = np.arange(len(seeds))
    w = 0.35
    ax.bar(xs - w / 2, a_vals, w, label="A: Emotion-Aware", color=colors["A"], edgecolor="#64748B")
    ax.bar(xs + w / 2, c_vals, w, label="C: Knowledge-Only", color=colors["C"], edgecolor="#64748B")
    ax.set_xlabel("Seed")
    ax.set_ylabel("Learning Gain (best checkpoint)")
    ax.set_title("Per-Seed Learning Gain Comparison")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(int(s)) for s in seeds], rotation=45, ha="right", fontsize=8)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "final_ac_per_seed_learning_gain.png", dpi=150)
    plt.close(fig)


def _df_to_md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(str(c) for c in cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def analyze_results(
    df_a: pd.DataFrame,
    df_c: pd.DataFrame,
) -> Dict[str, Any]:
    df_a = _normalize_condition_a(df_a)
    summary_a = aggregate_runs(df_a, CONDITION_A_KEY, "best", CONDITION_A_LABEL)
    summary_c = aggregate_runs(df_c, CONDITION_C_KEY, "best", CONDITION_C_LABEL)

    pairwise = pairwise_compare(
        df_a, df_c,
        CONDITION_A_KEY, CONDITION_C_KEY, "best",
        CONDITION_A_LABEL, CONDITION_C_LABEL,
    )
    pairwise["comparison"] = "A_emotion_aware_vs_C_knowledge_only_strict"

    conclusion = thesis_conclusion(summary_a, summary_c, pairwise)

    metrics_table = pd.DataFrame([
        {
            "condition": CONDITION_A_LABEL,
            "model": CONDITION_A_KEY,
            "n_seeds": summary_a["n_seeds"],
            **{f"{m}_mean": summary_a.get(f"{m}_mean") for m in METRICS},
            **{f"{m}_std": summary_a.get(f"{m}_std") for m in METRICS},
            **{f"{m}_ci": summary_a.get(f"{m}_ci") for m in METRICS},
        },
        {
            "condition": CONDITION_C_LABEL,
            "model": CONDITION_C_KEY,
            "n_seeds": summary_c["n_seeds"],
            **{f"{m}_mean": summary_c.get(f"{m}_mean") for m in METRICS},
            **{f"{m}_std": summary_c.get(f"{m}_std") for m in METRICS},
            **{f"{m}_ci": summary_c.get(f"{m}_ci") for m in METRICS},
        },
    ])

    ckpt_a = checkpoint_selection_table(df_a).assign(condition="A")
    ckpt_c = checkpoint_selection_table(df_c).assign(condition="C")

    report = {
        "study": "final_thesis_ac_ablation",
        "condition_A": {
            "label": CONDITION_A_LABEL,
            "observation": ["knowledge", "engagement", "frustration", "confusion", "boredom"],
            "excluded": ["emotion_id"],
            "emotion_dynamics": "full (default)",
            "source": "figures/final_dqn_model/baseline_per_run_20.csv (frozen thesis model)",
        },
        "condition_C": {
            "label": CONDITION_C_LABEL,
            "observation": ["knowledge"],
            "emotion_dynamics": "strict_off (leakage-audited)",
            "source": "trained in this study",
        },
        "dqn_hyperparameters": FINAL_DQN_HP,
        "checkpoint_steps": CHECKPOINT_STEPS,
        "seeds_20": SEEDS_20,
        "gain_ratio": G4_GAIN["ratio_label"],
        "summary_A_best_checkpoint": summary_a,
        "summary_C_best_checkpoint": summary_c,
        "metrics_table": metrics_table.to_dict("records"),
        "pairwise_A_vs_C": pairwise.to_dict("records"),
        "thesis_conclusion": conclusion,
        "leakage_audit_reference": "figures/strict_emotion_ablation/EMOTION_LEAKAGE_AUDIT.md",
    }

    pd.DataFrame([summary_a]).to_csv(OUT_DIR / "summary_condition_a.csv", index=False)
    pd.DataFrame([summary_c]).to_csv(OUT_DIR / "summary_condition_c.csv", index=False)
    metrics_table.to_csv(OUT_DIR / "metrics_table.csv", index=False)
    pairwise.to_csv(OUT_DIR / "pairwise_a_vs_c.csv", index=False)
    ckpt_a.to_csv(OUT_DIR / "checkpoint_selection_a.csv", index=False)
    ckpt_c.to_csv(OUT_DIR / "checkpoint_selection_c.csv", index=False)
    df_a.to_csv(OUT_DIR / "condition_a_per_run.csv", index=False)
    df_c.to_csv(OUT_DIR / "condition_c_per_run.csv", index=False)

    with open(OUT_DIR / "final_ac_ablation_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    _write_markdown_report(report, metrics_table, pairwise, ckpt_a, ckpt_c, conclusion)
    _plot_results(summary_a, summary_c, df_a, df_c, pairwise)
    return report


def _write_markdown_report(
    report: Dict[str, Any],
    metrics_table: pd.DataFrame,
    pairwise: pd.DataFrame,
    ckpt_a: pd.DataFrame,
    ckpt_c: pd.DataFrame,
    conclusion: Dict[str, Any],
) -> None:
    disp_cols = [
        "condition", "n_seeds",
        "learning_gain_mean", "learning_gain_std", "learning_gain_ci",
        "final_knowledge_mean", "success_rate_mean",
        "adaptation_accuracy_mean", "mean_episode_reward_mean",
    ]
    lines = [
        "# Final Thesis Ablation: Emotion-Aware vs Knowledge-Only DQN",
        "",
        "## Experimental Design",
        "",
        "### Condition A - Emotion-Aware DQN (Frozen Thesis Model)",
        "",
        "- **Observation:** knowledge, engagement, frustration, confusion, boredom",
        "- **Excluded:** emotion_id",
        "- **Environment:** default emotional dynamics, default reward, default transitions",
        "",
        "### Condition C - Knowledge-Only DQN (Strict No-Emotion)",
        "",
        "- **Observation:** knowledge only",
        "- **Environment:** strict no-emotion (transitions, reward, masks, dropout disabled)",
        "- **Validation:** emotion-leakage-free implementation (see EMOTION_LEAKAGE_AUDIT.md)",
        "",
        "### DQN Configuration",
        "",
        f"- Learning rate: {FINAL_DQN_HP['learning_rate']}",
        f"- Batch size: {FINAL_DQN_HP['batch_size']}",
        f"- Buffer size: {FINAL_DQN_HP['buffer_size']:,}",
        f"- Network: {FINAL_DQN_HP['net_arch']}",
        f"- Training budget: {TRAIN_TS:,} steps",
        f"- Checkpoints: {CHECKPOINT_STEPS}",
        f"- Seeds (n=20): `{SEEDS_20}`",
        "",
        "## Table 1 - Aggregate Metrics (Best Checkpoint, n=20)",
        "",
        _df_to_md(metrics_table[disp_cols]),
        "",
        "## Table 2 - Pairwise Comparison (A vs C, Welch t-test + Holm)",
        "",
        _df_to_md(pairwise[[
            "metric", "mean_A", "mean_B", "delta_A_minus_B",
            "std_A", "std_B", "cohens_d", "t", "p_raw", "p_holm",
        ]].rename(columns={"mean_A": "A_mean", "mean_B": "C_mean"})),
        "",
        "## Checkpoint Selection (Condition A)",
        "",
        _df_to_md(ckpt_a.head(20)),
        "",
        "## Checkpoint Selection (Condition C)",
        "",
        _df_to_md(ckpt_c.head(20)),
        "",
        "## Research Questions",
        "",
        f"1. Does emotion-aware tutoring outperform knowledge-only? "
        f"**{conclusion['Q1_emotion_aware_outperforms_knowledge_only']}** "
        f"(delta LG = {conclusion['Q1_delta_learning_gain_A_minus_C']:+.4f})",
        f"2. Effect size (Cohen's d): **{conclusion['Q2_effect_size_cohens_d']:.4f}**",
        f"3. Statistically significant (Holm)? **{conclusion['Q3_statistically_significant']}** "
        f"(p = {conclusion['Q3_p_holm_learning_gain']:.4f})",
        f"4. Performance attributable to emotion: **{conclusion['Q4_pct_performance_from_emotion']}%** relative LG gain",
        f"5. Emotional component justified? **{conclusion['Q5_emotional_component_justified']}**",
        "",
        "## Final Thesis Conclusion",
        "",
        f"**{conclusion['final_thesis_verdict']}**",
        "",
        conclusion["thesis_narrative"],
        "",
        "| Metric | A (Emotion-Aware) | C (Knowledge-Only) |",
        "| --- | --- | --- |",
        f"| Learning Gain | {conclusion['condition_a_learning_gain_mean']:.4f} "
        f"{conclusion['condition_a_learning_gain_ci']} | "
        f"{conclusion['condition_c_learning_gain_mean']:.4f} "
        f"{conclusion['condition_c_learning_gain_ci']} |",
        "",
        "## Limitations",
        "",
        "- Condition C uses knowledge-only reward (wk * delta_k); mean reward is not directly comparable.",
        "- Adaptation accuracy in C uses pinned neutral emotion_id (diagnostic metric).",
        "- Results are simulator-specific; 20-seed protocol matches frozen thesis model.",
        "",
    ]
    (OUT_DIR / "FINAL_AC_ABLATION_REPORT.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Final thesis A vs C ablation")
    parser.add_argument("--phase", choices=["run", "analyze", "all"], default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    seeds = SEEDS_20
    if args.quick:
        seeds = [42, 7]
        print("QUICK MODE: 2 seeds only")

    df_a = load_condition_a(seeds)

    if args.phase in ("run", "all"):
        print("=== Condition C: Knowledge-Only Strict (20-seed training) ===")
        df_c, _ = run_condition_c(seeds, resume=args.resume)
    else:
        c_path = OUT_DIR / "condition_c_per_run.csv"
        if not c_path.exists():
            raise SystemExit(f"No Condition C results at {c_path}. Run --phase run first.")
        df_c = pd.read_csv(c_path)

    if args.phase in ("analyze", "all"):
        print("\n=== ANALYZE ===")
        report = analyze_results(df_a, df_c)
        c = report["thesis_conclusion"]
        print(json.dumps(c, indent=2))
        print(f"\nReport: {OUT_DIR / 'FINAL_AC_ABLATION_REPORT.md'}")


if __name__ == "__main__":
    main()
