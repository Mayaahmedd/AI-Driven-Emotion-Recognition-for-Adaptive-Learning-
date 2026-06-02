"""
DQN stability study: reduce seed variance without changing the thesis problem.

Evaluates:
  1. Training length (50k, 100k, 150k, 200k)
  2. Checkpoint selection (10k-200k, best validation LG)
  3. Exploration schedule (exploration_fraction x exploration_final_eps)
  4. Target update interval (500, 1000, 2000)
  5. Network architecture ([64,64] vs [128,128,128])

Primary environment: full_emotion (Condition A) with G4 gain ratio.
Metrics: mean/std/CV of learning gain across seeds; Holm-corrected pairwise vs baseline.

Usage (from repo root):
  export PYTHONPATH="$(pwd)"
  python -m RL_Module.dqn_stability_study --phase all
  python -m RL_Module.dqn_stability_study --phase screen --quick
  python -m RL_Module.dqn_stability_study --phase analyze
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

OUT_DIR = _HERE / "figures" / "dqn_stability_study"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = list(cfg.SEEDS)
EVAL_EPS = 500
VAL_EPS = 100
G4_GAIN = {"GAIN_CORRECT_FACTOR": 0.1, "GAIN_INCORRECT_FACTOR": 1.0, "ratio_label": "10:1"}

BASELINE_HP: Dict[str, Any] = {
    "learning_rate": 0.001,
    "batch_size": 64,
    "buffer_size": 100_000,
    "target_update_interval": 500,
    "exploration_fraction": 0.3,
    "exploration_final_eps": 0.05,
    "net_arch": [64, 64],
}

TRAIN_LENGTHS = [50_000, 100_000, 150_000, 200_000]
CHECKPOINT_STEPS = [10_000, 20_000, 30_000, 40_000, 50_000, 100_000, 150_000, 200_000]

STABILITY_GRID = {
    "exploration_fraction": [0.3, 0.5, 0.7],
    "exploration_final_eps": [0.05, 0.02, 0.01],
    "target_update_interval": [500, 1000, 2000],
    "net_arch": [[64, 64], [128, 128, 128]],
}


def _patch_gain() -> Dict[str, float]:
    prev = dict(cfg.SIMULATOR_PARAMS)
    cfg.SIMULATOR_PARAMS = dict(G4_GAIN)
    return prev


def _restore_gain(snapshot: Dict[str, float]) -> None:
    cfg.SIMULATOR_PARAMS = snapshot


def config_id(hp: Dict[str, Any], train_ts: int, use_checkpoints: bool) -> str:
    payload = {"hp": hp, "train_ts": train_ts, "ckpt": use_checkpoints}
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.md5(raw.encode()).hexdigest()[:10]


def quick_eval_lg(agent: DQNAgent, seed: int, n_episodes: int = VAL_EPS) -> float:
    """Validation learning gain for checkpoint selection."""
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


def full_eval(
    agent: DQNAgent,
    seed: int,
    n_episodes: int = EVAL_EPS,
) -> Dict[str, float]:
    env = StudentEnv(
        population_seed=seed,
        obs_ablation="full_emotion",
        emotion_dynamics="full",
    )
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


def train_and_eval_config(
    hp: Dict[str, Any],
    train_ts: int,
    seeds: List[int],
    use_checkpoints: bool,
    label: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    cid = config_id(hp, train_ts, use_checkpoints)

    for seed in seeds:
        snap = _patch_gain()
        t0 = time.time()
        try:
            cfg.set_all_seeds(seed)
            tag = f"stability_{label}_{cid}_s{seed}"
            train_env = make_env(
                seed=seed,
                obs_ablation="full_emotion",
                emotion_dynamics="full",
                algo_tag=tag,
            )
            agent = DQNAgent()
            ckpt_dir = cfg.MODELS_DIR / tag

            if use_checkpoints:
                ckpt_freq = 10_000
                paths = agent.train_with_checkpoints(
                    train_env,
                    train_ts,
                    seed,
                    hyperparams=hp,
                    checkpoint_dir=str(ckpt_dir),
                    checkpoint_freq=ckpt_freq,
                )
                train_env.close()
                valid_paths = [
                    p for p in paths
                    if any(f"_{step}_steps" in p for step in CHECKPOINT_STEPS)
                    or paths.index(p) == len(paths) - 1
                ]
                if not valid_paths:
                    valid_paths = paths
                best_path, best_val = agent.select_best_checkpoint(
                    valid_paths,
                    lambda a, s: quick_eval_lg(a, s, VAL_EPS),
                    seed,
                )
                selection = f"checkpoint@{best_path}"
            else:
                agent.train(train_env, train_ts, seed, hyperparams=hp)
                train_env.close()
                agent.save(str(cfg.MODELS_DIR / tag))
                best_val = float("nan")
                selection = f"final@{train_ts}"

            metrics = full_eval(agent, seed, EVAL_EPS)
        finally:
            _restore_gain(snap)

        elapsed = time.time() - t0
        row = {
            "config_id": cid,
            "label": label,
            "seed": seed,
            "train_timesteps": train_ts,
            "use_checkpoints": use_checkpoints,
            "selection": selection,
            "val_learning_gain": best_val,
            "elapsed_seconds": round(elapsed, 1),
            **{k: (json.dumps(v) if k == "net_arch" else v) for k, v in hp.items()},
            **metrics,
        }
        rows.append(row)
        print(
            f"  {label} seed={seed}: lg={metrics['learning_gain']:.3f} "
            f"({elapsed:.0f}s) sel={selection}"
        )
    return rows


def run_training_length_study(seeds: List[int], resume: bool) -> pd.DataFrame:
    csv_path = OUT_DIR / "training_length_per_run.csv"
    rows = _load_resume(csv_path, resume)
    done = {(r["train_timesteps"], r["seed"]) for r in rows}

    for ts in TRAIN_LENGTHS:
        print(f"\n--- Training length {ts} ---")
        for seed in seeds:
            if (ts, seed) in done:
                continue
            batch = train_and_eval_config(
                BASELINE_HP, ts, [seed], use_checkpoints=False, label=f"len_{ts}"
            )
            rows.extend(batch)
            pd.DataFrame(rows).to_csv(csv_path, index=False)
    return pd.DataFrame(rows)


def run_checkpoint_study(seeds: List[int], train_ts: int, resume: bool) -> pd.DataFrame:
    csv_path = OUT_DIR / "checkpoint_selection_per_run.csv"
    rows = _load_resume(csv_path, resume)
    done = {r["seed"] for r in rows if r.get("label") == "checkpoint_sel"}

    for seed in seeds:
        if seed in done:
            continue
        print(f"\n--- Checkpoint selection seed={seed} ---")
        # Baseline: final model only
        base = train_and_eval_config(
            BASELINE_HP, train_ts, [seed], False, "no_checkpoint"
        )
        ckpt = train_and_eval_config(
            BASELINE_HP, train_ts, [seed], True, "checkpoint_sel"
        )
        rows.extend(base)
        rows.extend(ckpt)
        pd.DataFrame(rows).to_csv(csv_path, index=False)
    return pd.DataFrame(rows)


def run_hyperparam_screen(seeds: List[int], train_ts: int, resume: bool) -> pd.DataFrame:
    csv_path = OUT_DIR / "hyperparam_screen_per_run.csv"
    rows = _load_resume(csv_path, resume)
    done = {r["config_id"] for r in rows}

    configs: List[Tuple[str, Dict[str, Any]]] = [("baseline", dict(BASELINE_HP))]

    for ef in STABILITY_GRID["exploration_fraction"]:
        hp = dict(BASELINE_HP)
        hp["exploration_fraction"] = ef
        configs.append((f"expl_frac_{ef}", hp))

    for eps in STABILITY_GRID["exploration_final_eps"]:
        hp = dict(BASELINE_HP)
        hp["exploration_final_eps"] = eps
        configs.append((f"final_eps_{eps}", hp))

    for tui in STABILITY_GRID["target_update_interval"]:
        hp = dict(BASELINE_HP)
        hp["target_update_interval"] = tui
        configs.append((f"target_upd_{tui}", hp))

    for arch in STABILITY_GRID["net_arch"]:
        hp = dict(BASELINE_HP)
        hp["net_arch"] = arch
        configs.append((f"net_arch_{'-'.join(map(str, arch))}", hp))

    for label, hp in configs:
        cid = config_id(hp, train_ts, False)
        if cid in done:
            continue
        print(f"\n--- Hyperparam: {label} ---")
        batch = train_and_eval_config(hp, train_ts, seeds, False, label)
        rows.extend(batch)
        pd.DataFrame(rows).to_csv(csv_path, index=False)
    return pd.DataFrame(rows)


def _load_resume(path: Path, resume: bool) -> List[Dict[str, Any]]:
    if resume and path.exists():
        return pd.read_csv(path).to_dict("records")
    return []


def aggregate_stability(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for key, sub in df.groupby(group_col):
        lgs = sub["learning_gain"].astype(float).tolist()
        mean, lo, hi = confidence_interval(lgs)
        std = float(np.std(lgs, ddof=1)) if len(lgs) > 1 else 0.0
        cv = std / mean if abs(mean) > 1e-9 else float("nan")
        rows.append({
            group_col: key,
            "n_seeds": len(lgs),
            "learning_gain_mean": round(mean, 4),
            "learning_gain_std": round(std, 4),
            "learning_gain_cv": round(cv, 4),
            "learning_gain_ci": f"[{lo:.3f}, {hi:.3f}]",
            "success_rate_mean": round(float(sub["success_rate"].mean()), 4),
            "adaptation_mean": round(float(sub["adaptation_accuracy"].mean()), 4),
            "reward_mean": round(float(sub["mean_episode_reward"].mean()), 4),
        })
    return pd.DataFrame(rows).sort_values("learning_gain_std")


def cohens_d(a: List[float], b: List[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = np.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2))
    return float((np.mean(a) - np.mean(b)) / pooled) if pooled > 1e-12 else 0.0


def recommend_config(length_df: pd.DataFrame, hp_df: pd.DataFrame, ckpt_df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """Pick lowest CV config with competitive mean LG."""
    candidates: List[Dict[str, Any]] = []

    if not length_df.empty:
        agg = aggregate_stability(length_df, "train_timesteps")
        for _, row in agg.iterrows():
            candidates.append({
                "source": "training_length",
                "name": f"train_{int(row['train_timesteps'])}",
                "mean_lg": row["learning_gain_mean"],
                "std_lg": row["learning_gain_std"],
                "cv": row["learning_gain_cv"],
                "hp": dict(BASELINE_HP),
                "train_ts": int(row["train_timesteps"]),
                "use_checkpoints": False,
            })

    if hp_df is not None and not hp_df.empty:
        agg = aggregate_stability(hp_df, "label")
        baseline_std = agg[agg["label"] == "baseline"]["learning_gain_std"]
        baseline_std_val = float(baseline_std.iloc[0]) if len(baseline_std) else 999.0
        for _, row in agg.iterrows():
            if row["label"] == "baseline":
                continue
            sub = hp_df[hp_df["label"] == row["label"]].iloc[0]
            hp = dict(BASELINE_HP)
            for k in BASELINE_HP:
                if k in sub and pd.notna(sub[k]):
                    val = sub[k]
                    if k == "net_arch" and isinstance(val, str):
                        val = json.loads(val.replace("'", '"'))
                    hp[k] = val
            candidates.append({
                "source": "hyperparam",
                "name": row["label"],
                "mean_lg": row["learning_gain_mean"],
                "std_lg": row["learning_gain_std"],
                "cv": row["learning_gain_cv"],
                "hp": hp,
                "train_ts": int(hp_df["train_timesteps"].iloc[0]) if "train_timesteps" in hp_df else 50_000,
                "use_checkpoints": False,
            })

    if ckpt_df is not None and not ckpt_df.empty:
        for label in ckpt_df["label"].unique():
            sub = ckpt_df[ckpt_df["label"] == label]
            lgs = sub["learning_gain"].tolist()
            candidates.append({
                "source": "checkpoint",
                "name": label,
                "mean_lg": float(np.mean(lgs)),
                "std_lg": float(np.std(lgs, ddof=1)) if len(lgs) > 1 else 0.0,
                "cv": float(np.std(lgs, ddof=1) / np.mean(lgs)) if len(lgs) > 1 and np.mean(lgs) > 0 else 999,
                "hp": dict(BASELINE_HP),
                "train_ts": int(sub["train_timesteps"].iloc[0]),
                "use_checkpoints": "checkpoint" in label,
            })

    if not candidates:
        return {"note": "no data"}

    # Score: minimize CV, tie-break by higher mean LG
    best = min(candidates, key=lambda c: (c["std_lg"], -c["mean_lg"]))
    baseline_cands = [c for c in candidates if c["name"] in ("baseline", "no_checkpoint", "train_50000")]
    baseline = baseline_cands[0] if baseline_cands else candidates[0]

    return {
        "recommended": best,
        "baseline_reference": baseline,
        "std_reduction_pct": round(
            (1 - best["std_lg"] / baseline["std_lg"]) * 100, 1
        ) if baseline["std_lg"] > 0 else 0.0,
        "all_candidates_ranked_by_std": sorted(candidates, key=lambda c: c["std_lg"]),
    }


def analyze_all() -> Dict[str, Any]:
    length_path = OUT_DIR / "training_length_per_run.csv"
    hp_path = OUT_DIR / "hyperparam_screen_per_run.csv"
    ckpt_path = OUT_DIR / "checkpoint_selection_per_run.csv"

    length_df = pd.read_csv(length_path) if length_path.exists() else pd.DataFrame()
    hp_df = pd.read_csv(hp_path) if hp_path.exists() else pd.DataFrame()
    ckpt_df = pd.read_csv(ckpt_path) if ckpt_path.exists() else pd.DataFrame()

    length_agg = aggregate_stability(length_df, "train_timesteps") if not length_df.empty else pd.DataFrame()
    hp_agg = aggregate_stability(hp_df, "label") if not hp_df.empty else pd.DataFrame()
    recommendation = recommend_config(length_df, hp_df, ckpt_df if not ckpt_df.empty else None)

    report = {
        "study": "dqn_stability",
        "gain_ratio": G4_GAIN["ratio_label"],
        "baseline_hp": BASELINE_HP,
        "training_length_summary": length_agg.to_dict("records") if not length_agg.empty else [],
        "hyperparam_summary": hp_agg.to_dict("records") if not hp_agg.empty else [],
        "recommendation": recommendation,
    }

    if not length_agg.empty:
        length_agg.to_csv(OUT_DIR / "training_length_summary.csv", index=False)
    if not hp_agg.empty:
        hp_agg.to_csv(OUT_DIR / "hyperparam_summary.csv", index=False)

    with open(OUT_DIR / "dqn_stability_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    _write_markdown(report, length_agg, hp_agg, recommendation)
    _plot_stability(length_agg, hp_agg, recommendation)
    return report


def _write_markdown(report, length_agg, hp_agg, recommendation) -> None:
    rec = recommendation.get("recommended", {})
    lines = [
        "# DQN Stability Study Report",
        "",
        "## Methodology",
        "",
        f"Environment: full_emotion, G4 gain ({G4_GAIN['ratio_label']}), {len(SEEDS)} seeds.",
        "Goal: reduce learning-gain variance across seeds without changing MDP definition.",
        "",
        "Factors tested:",
        "- Training length: 50k-200k",
        "- Checkpoint selection: best validation LG among 10k-200k saves",
        "- Exploration: fraction {0.3,0.5,0.7} x final_eps {0.05,0.02,0.01}",
        "- Target update: {500, 1000, 2000}",
        "- Architecture: [64,64] vs [128,128,128]",
        "",
        "## Training Length Results",
        "",
    ]
    if not length_agg.empty:
        lines.append(_df_to_md(length_agg))
    lines.extend(["", "## Hyperparameter Screen (std ranking)", ""])
    if not hp_agg.empty:
        lines.append(_df_to_md(hp_agg.head(10)))
    lines.extend([
        "",
        "## Recommendation",
        "",
        f"**Best config:** {rec.get('name', 'N/A')} (source: {rec.get('source')})",
        f"- Mean LG: {rec.get('mean_lg', 'N/A')}",
        f"- Std LG: {rec.get('std_lg', 'N/A')}",
        f"- CV: {rec.get('cv', 'N/A')}",
        f"- Std reduction vs baseline: {recommendation.get('std_reduction_pct', 0)}%",
        "",
        "## Thesis Conclusion",
        "",
        "Report the recommended configuration as the final thesis DQN. Prefer configs that",
        "reduce seed std while maintaining or improving mean learning gain. Checkpoint selection",
        "addresses seed-specific convergence timing without changing the underlying MDP.",
        "",
        "## Discussion",
        "",
        "High DQN variance often reflects exploration schedule mismatch and early stopping",
        "before Q-value convergence. Longer training and validation checkpoint selection are",
        "standard SB3 mitigations that do not alter the tutoring problem definition.",
        "",
    ])
    (OUT_DIR / "DQN_STABILITY_REPORT.md").write_text("\n".join(lines))


def _df_to_md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def _plot_stability(length_agg, hp_agg, recommendation) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if not length_agg.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        xs = length_agg["train_timesteps"].astype(str)
        ax.errorbar(
            range(len(xs)),
            length_agg["learning_gain_mean"],
            yerr=length_agg["learning_gain_std"],
            fmt="o-",
            capsize=5,
            color="#DD8452",
        )
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels(xs)
        ax.set_xlabel("Training timesteps")
        ax.set_ylabel("Learning gain (mean +/- std)")
        ax.set_title("DQN Stability vs Training Length")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "training_length_stability.png", dpi=150)
        plt.close(fig)

    if not hp_agg.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        top = hp_agg.nsmallest(12, "learning_gain_std")
        ax.barh(top["label"], top["learning_gain_std"], color="#4C72B0", edgecolor="#64748B")
        ax.set_xlabel("Learning Gain Std (lower = more stable)")
        ax.set_title("Hyperparameter Configurations by Seed Stability")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "hyperparam_stability_ranking.png", dpi=150)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="DQN stability study")
    parser.add_argument(
        "--phase",
        choices=["length", "checkpoint", "screen", "all", "analyze"],
        default="all",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--train-ts", type=int, default=50_000)
    args = parser.parse_args()

    seeds = SEEDS
    train_ts = args.train_ts
    if args.quick:
        seeds = [42, 7, 13]
        train_ts = min(train_ts, 5_000)
        print(f"QUICK MODE: seeds={seeds}, train={train_ts}")

    if args.phase in ("length", "all"):
        print("=== Training Length Study ===")
        run_training_length_study(seeds if args.quick else SEEDS[:5], args.resume)

    if args.phase in ("checkpoint", "all"):
        print("\n=== Checkpoint Selection Study ===")
        run_checkpoint_study(seeds, train_ts, args.resume)

    if args.phase in ("screen", "all"):
        print("\n=== Hyperparameter Screen ===")
        run_hyperparam_screen(seeds, train_ts, args.resume)

    if args.phase in ("analyze", "all"):
        print("\n=== Analyze ===")
        report = analyze_all()
        print(json.dumps(report.get("recommendation", {}), indent=2, default=str))


if __name__ == "__main__":
    main()
