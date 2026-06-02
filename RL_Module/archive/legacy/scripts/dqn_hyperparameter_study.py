"""
DQN hyperparameter optimization study (environment fixed).

Phases:
  baseline  - reproduce default DQN across 5 seeds
  grid      - full factorial search (1 screening seed per config; checkpointed)
  robust    - top-5 configs x 5 seeds
  report    - tables, rankings, educational validity, recommendation

Run:
  python -m RL_Module.dqn_hyperparameter_study --phase all
  python -m RL_Module.dqn_hyperparameter_study --phase grid --resume
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    action_frequency_report,
    aggregate_across_seeds,
    confidence_interval,
    normalized_knowledge_gain,
)

OUT_DIR = _HERE / "figures" / "_archive" / "superseded" / "dqn_hyperparameter_study"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Evaluation protocol (matches main_experiment.py / config defaults)
TRAIN_TS = cfg.TOTAL_TIMESTEPS
EVAL_EPS = cfg.EVAL_EPISODES
ROBUST_SEEDS = [42, 0, 1, 7, 13]
SCREEN_SEED = 42

SEARCH_SPACE = {
    "learning_rate": [1e-4, 3e-4, 5e-4, 1e-3],
    "batch_size": [32, 64, 128, 256],
    "buffer_size": [10_000, 50_000, 100_000],
    "target_update_interval": [500, 1000, 2000, 5000],
    "exploration_fraction": [0.1, 0.2, 0.3, 0.5],
    "exploration_final_eps": [0.01, 0.05, 0.10],
}

BASELINE_HP: Dict[str, Any] = {
    "learning_rate": cfg.DQN_LEARNING_RATE,
    "batch_size": cfg.DQN_BATCH_SIZE,
    "buffer_size": cfg.DQN_BUFFER_SIZE,
    "target_update_interval": cfg.DQN_TARGET_UPDATE_INTERVAL,
    "exploration_fraction": cfg.DQN_EXPLORATION_FRACTION,
    "exploration_final_eps": cfg.DQN_EXPLORATION_FINAL_EPS,
}

COMFORT_ACTIONS = {"break", "encouragement", "no_action"}
CHALLENGE_ACTIONS = {"harder_problem", "scaffold", "hint"}


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def config_id(hp: Dict[str, Any]) -> str:
    raw = json.dumps(_json_safe(hp), sort_keys=True)
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _shannon_entropy(freq: Dict[str, float]) -> float:
    p = np.array([v for v in freq.values() if v > 0], dtype=float)
    return float(-np.sum(p * np.log(p))) if len(p) else 0.0


def evaluate_with_diagnostics(
    agent: DQNAgent,
    env: StudentEnv,
    n_episodes: int,
    seed: int,
    algorithm: str,
) -> Dict[str, Any]:
    """Extended evaluate: final knowledge, delta-k, P(correct), action mix."""
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

            correct = bool(info.get("last_answer_correct", True))
            k_now = env._state.knowledge
            step_rows.append({
                "episode": ep,
                "step": step,
                "knowledge": k_now,
                "delta_k": k_now - prev_k,
                "correct": int(correct),
                "engagement": env._state.engagement,
                "frustration": env._state.frustration,
                "confusion": env._state.confusion,
                "boredom": env._state.boredom,
                "action_id": action,
                "reward": reward,
            })
            prev_k = k_now

            ep_adapt_total += 1
            if _adaptation_match(emotion_id, action):
                ep_adapt_hits += 1

            total_reward += reward
            action_counts[action] += 1
            step += 1
            mask = info["action_masks"]
            done = terminated or truncated

        final = env._state
        lg_norm = normalized_knowledge_gain(start_k, final.knowledge)
        success = _is_success(final.knowledge, final.frustration, final.confusion)
        episode_results.append({
            "total_reward": total_reward,
            "learning_gain_normalized": lg_norm,
            "success": int(success),
            "final_knowledge": final.knowledge,
            "adaptation_accuracy": ep_adapt_hits / max(ep_adapt_total, 1),
            "action_counts": action_counts,
        })

    sdf = pd.DataFrame(step_rows)
    freq_report = action_frequency_report(episode_results, print_table=False)
    freqs = freq_report["frequencies"]
    comfort_frac = sum(freqs.get(a, 0.0) for a in COMFORT_ACTIONS)
    challenge_frac = sum(freqs.get(a, 0.0) for a in CHALLENGE_ACTIONS)

    return {
        "mean_episode_reward": float(np.mean([r["total_reward"] for r in episode_results])),
        "success_rate": float(np.mean([r["success"] for r in episode_results])),
        "learning_gain": float(
            np.mean([r["learning_gain_normalized"] for r in episode_results])
        ),
        "final_knowledge": float(np.mean([r["final_knowledge"] for r in episode_results])),
        "adaptation_accuracy": float(
            np.mean([r["adaptation_accuracy"] for r in episode_results])
        ),
        "mean_delta_k_per_step": float(sdf["delta_k"].mean()) if len(sdf) else 0.0,
        "p_correct": float(sdf["correct"].mean()) if len(sdf) else 0.0,
        "mean_engagement": float(sdf["engagement"].mean()) if len(sdf) else 0.0,
        "mean_frustration": float(sdf["frustration"].mean()) if len(sdf) else 0.0,
        "comfort_action_fraction": comfort_frac,
        "challenge_action_fraction": challenge_frac,
        "action_diversity_shannon": _shannon_entropy(freqs),
        "dominance_detected": freq_report["dominance_detected"],
        "dominant_actions": freq_report["dominant_actions"],
        "action_frequencies": {k: round(v, 4) for k, v in freqs.items()},
        "episode_results": episode_results,
    }


def train_and_eval(
    hp: Dict[str, Any],
    seed: int,
    tag: str,
) -> Dict[str, Any]:
    t0 = time.time()
    cfg.set_all_seeds(seed)
    env = make_env(seed=seed, algo_tag=tag)
    agent = DQNAgent()
    agent.train(env, TRAIN_TS, seed, hyperparams=hp)
    env.close()

    eval_env = StudentEnv(population_seed=seed)
    eval_env.set_algorithm_name(tag)
    result = evaluate_with_diagnostics(
        agent, eval_env, n_episodes=EVAL_EPS, seed=seed, algorithm=tag
    )
    eval_env.close()
    elapsed = time.time() - t0

    row = {
        "config_id": config_id(hp),
        "seed": seed,
        "train_timesteps": TRAIN_TS,
        "eval_episodes": EVAL_EPS,
        "elapsed_seconds": round(elapsed, 1),
        **{k: hp[k] for k in SEARCH_SPACE},
        **{k: result[k] for k in [
            "mean_episode_reward",
            "success_rate",
            "learning_gain",
            "final_knowledge",
            "adaptation_accuracy",
            "mean_delta_k_per_step",
            "p_correct",
            "mean_engagement",
            "mean_frustration",
            "comfort_action_fraction",
            "challenge_action_fraction",
            "action_diversity_shannon",
            "dominance_detected",
        ]},
        "action_frequencies": result["action_frequencies"],
        "dominant_actions": result["dominant_actions"],
        "is_baseline": hp == BASELINE_HP,
    }
    print(
        f"  [{tag}] seed={seed} reward={row['mean_episode_reward']:.3f} "
        f"succ={row['success_rate']:.3f} fk={row['final_knowledge']:.3f} "
        f"adapt={row['adaptation_accuracy']:.3f} ({elapsed:.0f}s)"
    )
    return row


def iter_grid() -> List[Dict[str, Any]]:
    keys = list(SEARCH_SPACE.keys())
    combos = []
    for vals in itertools.product(*(SEARCH_SPACE[k] for k in keys)):
        combos.append(dict(zip(keys, vals)))
    return combos


def run_baseline() -> pd.DataFrame:
    print(f"\n=== Section 1: DQN baseline ({len(ROBUST_SEEDS)} seeds) ===")
    rows = []
    for seed in ROBUST_SEEDS:
        rows.append(
            train_and_eval(BASELINE_HP, seed, f"baseline_s{seed}")
        )
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "baseline_per_seed.csv", index=False)
    agg = aggregate_across_seeds(
        [
            {
                "mean_episode_reward": r["mean_episode_reward"],
                "success_rate": r["success_rate"],
                "learning_gain": r["learning_gain"],
                "dropout_rate": 0.0,
                "adaptation_accuracy": r["adaptation_accuracy"],
            }
            for r in rows
        ]
    )
    fk_mean, fk_lo, fk_hi = confidence_interval([r["final_knowledge"] for r in rows])
    summary = {
        "phase": "baseline",
        "n_seeds": len(ROBUST_SEEDS),
        "mean_reward": agg["mean_mean_episode_reward"],
        "reward_std": float(np.std([r["mean_episode_reward"] for r in rows], ddof=1)),
        "reward_ci_95": [agg["ci_lower_mean_episode_reward"], agg["ci_upper_mean_episode_reward"]],
        "success_rate_mean": agg["mean_success_rate"],
        "success_rate_ci_95": [agg["ci_lower_success_rate"], agg["ci_upper_success_rate"]],
        "learning_gain_mean": agg["mean_learning_gain"],
        "learning_gain_ci_95": [agg["ci_lower_learning_gain"], agg["ci_upper_learning_gain"]],
        "final_knowledge_mean": fk_mean,
        "final_knowledge_ci_95": [fk_lo, fk_hi],
        "adaptation_accuracy_mean": agg["mean_adaptation_accuracy"],
        "adaptation_accuracy_ci_95": [
            agg["ci_lower_adaptation_accuracy"],
            agg["ci_upper_adaptation_accuracy"],
        ],
        "hyperparameters": BASELINE_HP,
    }
    with open(OUT_DIR / "baseline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return df


def _grid_worker(hp: Dict[str, Any]) -> Dict[str, Any]:
    """Process-pool entry: train/eval one hyperparameter configuration."""
    cid = config_id(hp)
    return train_and_eval(hp, SCREEN_SEED, f"grid_{cid}")


def run_grid(
    resume: bool = True,
    max_configs: Optional[int] = None,
    workers: int = 1,
) -> pd.DataFrame:
    print("\n=== Section 2-3: factorial grid (screening seed) ===")
    path = OUT_DIR / "grid_screening_results.csv"
    done_ids: set = set()
    existing: List[Dict[str, Any]] = []
    if resume and path.exists():
        prev = pd.read_csv(path)
        existing = prev.to_dict("records")
        done_ids = set(prev["config_id"].astype(str))

    grid = iter_grid()
    if max_configs is not None:
        grid = grid[:max_configs]
    pending = [hp for hp in grid if config_id(hp) not in done_ids]
    print(
        f"  Grid size: {len(grid)} configs | pending: {len(pending)} | "
        f"workers={workers} | screening seed={SCREEN_SEED}"
    )

    rows = list(existing)
    if workers <= 1:
        for i, hp in enumerate(pending):
            cid = config_id(hp)
            print(f"  [{len(done_ids) + i + 1}/{len(grid)}] config_id={cid}")
            row = train_and_eval(hp, SCREEN_SEED, f"grid_{cid}")
            rows.append(row)
            pd.DataFrame(rows).to_csv(path, index=False)
            done_ids.add(cid)
    else:
        completed = 0
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_grid_worker, hp): hp for hp in pending}
            for fut in as_completed(futures):
                row = fut.result()
                rows.append(row)
                done_ids.add(str(row["config_id"]))
                completed += 1
                pd.DataFrame(rows).to_csv(path, index=False)
                if completed % 10 == 0 or completed == len(pending):
                    print(f"  grid progress: {len(done_ids)}/{len(grid)} configs done")

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    rank_and_save_grid(df)
    return df


def _zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if std < 1e-12:
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def rank_and_save_grid(df: pd.DataFrame) -> Dict[str, Any]:
    """Rank configs aggregated by config_id (screening seed)."""
    g = df.groupby("config_id", as_index=False).first()
    metrics = [
        "mean_episode_reward",
        "success_rate",
        "learning_gain",
        "final_knowledge",
        "adaptation_accuracy",
    ]
    for m in metrics:
        g[f"z_{m}"] = _zscore(g[m])

    g["balanced_score"] = g[[f"z_{m}" for m in metrics]].mean(axis=1)
    g = g.sort_values("balanced_score", ascending=False)

    best_reward = g.loc[g["mean_episode_reward"].idxmax()]
    best_mastery = g.loc[g["final_knowledge"].idxmax()]
    best_balanced = g.iloc[0]

    ranking = g.sort_values("mean_episode_reward", ascending=False).reset_index(drop=True)
    ranking["rank_by_reward"] = ranking.index + 1
    ranking = ranking.sort_values("final_knowledge", ascending=False)
    ranking["rank_by_mastery"] = ranking.index + 1
    ranking = ranking.sort_values("balanced_score", ascending=False).reset_index(drop=True)
    ranking["rank_balanced"] = ranking.index + 1
    ranking.to_csv(OUT_DIR / "grid_rankings.csv", index=False)

    leaders = {
        "best_reward": _hp_row(best_reward),
        "best_mastery": _hp_row(best_mastery),
        "best_balanced": _hp_row(best_balanced),
        "n_configs": len(g),
        "screening_seed": SCREEN_SEED,
    }
    with open(OUT_DIR / "grid_leaders.json", "w") as f:
        json.dump(_json_safe(leaders), f, indent=2)
    print("\n  Top by reward:", leaders["best_reward"]["config_id"])
    print("  Top by mastery:", leaders["best_mastery"]["config_id"])
    print("  Top balanced:", leaders["best_balanced"]["config_id"])
    return leaders


def _hp_row(row: pd.Series) -> Dict[str, Any]:
    out = {"config_id": str(row["config_id"])}
    for k in SEARCH_SPACE:
        out[k] = row[k] if k in row else row.get(k)
    for m in [
        "mean_episode_reward",
        "success_rate",
        "learning_gain",
        "final_knowledge",
        "adaptation_accuracy",
        "balanced_score",
    ]:
        if m in row:
            out[m] = float(row[m])
    return out


def _hp_from_row(row: pd.Series) -> Dict[str, Any]:
    float_keys = {"learning_rate", "exploration_fraction", "exploration_final_eps"}
    return {
        k: float(row[k]) if k in float_keys else int(row[k]) for k in SEARCH_SPACE
    }


def top_config_ids(n: int = 5) -> List[str]:
    path = OUT_DIR / "grid_rankings.csv"
    if not path.exists():
        raise FileNotFoundError("Run grid phase first.")
    df = pd.read_csv(path).sort_values("balanced_score", ascending=False)
    return df["config_id"].head(n).tolist()


def run_robust(top_n: int = 5) -> pd.DataFrame:
    print(f"\n=== Section 5: robustness (top {top_n} x {len(ROBUST_SEEDS)} seeds) ===")
    rankings = pd.read_csv(OUT_DIR / "grid_rankings.csv").sort_values(
        "balanced_score", ascending=False
    )
    top_ids: List[str] = []
    seen_hp: set = set()
    for _, row in rankings.iterrows():
        hp_key = json.dumps(_hp_from_row(row), sort_keys=True)
        if hp_key in seen_hp:
            continue
        seen_hp.add(hp_key)
        top_ids.append(str(row["config_id"]))
        if len(top_ids) >= top_n:
            break
    path = OUT_DIR / "robust_top_configs.csv"
    done: set = set()
    rows: List[Dict[str, Any]] = []
    if path.exists():
        prev = pd.read_csv(path)
        rows = prev.to_dict("records")
        done = {(str(r["config_id"]), int(r["seed"])) for r in rows}

    for cid in top_ids:
        cfg_row = rankings[rankings["config_id"] == cid].iloc[0]
        hp = _hp_from_row(cfg_row)
        for seed in ROBUST_SEEDS:
            if (str(cid), seed) in done:
                continue
            print(f"  robust config_id={cid} seed={seed}")
            row = train_and_eval(hp, seed, f"robust_{cid}_s{seed}")
            rows.append(row)
            pd.DataFrame(rows).to_csv(path, index=False)

    df = pd.DataFrame(rows)
    summarize_robust(df)
    return df


def summarize_robust(df: pd.DataFrame) -> None:
    metrics = [
        "mean_episode_reward",
        "success_rate",
        "learning_gain",
        "final_knowledge",
        "adaptation_accuracy",
    ]
    summaries = []
    for cid, grp in df.groupby("config_id"):
        entry: Dict[str, Any] = {"config_id": str(cid)}
        for k in SEARCH_SPACE:
            entry[k] = grp.iloc[0][k]
        stable = True
        for m in metrics:
            vals = grp[m].astype(float).tolist()
            mean, lo, hi = confidence_interval(vals)
            entry[f"{m}_mean"] = mean
            entry[f"{m}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            entry[f"{m}_ci_95"] = [lo, hi]
            if m == "mean_episode_reward":
                baseline_path = OUT_DIR / "baseline_per_seed.csv"
                if baseline_path.exists():
                    bdf = pd.read_csv(baseline_path)
                    b_mean = float(bdf[m].mean())
                    _, p = stats.ttest_rel(
                        np.array(vals),
                        np.array(bdf[m].tolist()[: len(vals)]),
                    )
                    entry["reward_vs_baseline_p"] = float(p) if len(vals) == len(bdf) else None
                    entry["reward_vs_baseline_delta"] = mean - b_mean
                    if entry["reward_vs_baseline_delta"] > 0 and p and p > 0.05:
                        stable = False
        entry["stable_improvement"] = stable
        summaries.append(entry)

    with open(OUT_DIR / "robust_summary.json", "w") as f:
        json.dump(_json_safe(summaries), f, indent=2)
    pd.DataFrame(summaries).to_csv(OUT_DIR / "robust_summary_table.csv", index=False)


def educational_validity() -> Dict[str, Any]:
    """Section 4: compare leaders vs baseline on pedagogy proxies."""
    print("\n=== Section 4: educational validity ===")
    baseline_path = OUT_DIR / "baseline_per_seed.csv"
    grid_path = OUT_DIR / "grid_screening_results.csv"
    if not baseline_path.exists() or not grid_path.exists():
        return {}

    bdf = pd.read_csv(baseline_path)
    gdf = pd.read_csv(grid_path)
    with open(OUT_DIR / "grid_leaders.json") as f:
        leaders = json.load(f)

    b_mean = {
        "adaptation_accuracy": float(bdf["adaptation_accuracy"].mean()),
        "challenge_action_fraction": float(bdf["challenge_action_fraction"].mean()),
        "comfort_action_fraction": float(bdf["comfort_action_fraction"].mean()),
        "mean_delta_k_per_step": float(bdf["mean_delta_k_per_step"].mean()),
        "p_correct": float(bdf["p_correct"].mean()),
        "mean_episode_reward": float(bdf["mean_episode_reward"].mean()),
        "final_knowledge": float(bdf["final_knowledge"].mean()),
    }

    def _leader_metrics(leader_key: str) -> Dict[str, float]:
        cid = leaders[leader_key]["config_id"]
        row = gdf[gdf["config_id"] == cid].iloc[0]
        return {k: float(row[k]) for k in b_mean}

    report: Dict[str, Any] = {"baseline": b_mean, "comparisons": {}}
    for key in ("best_reward", "best_mastery", "best_balanced"):
        m = _leader_metrics(key)
        delta = {k: m[k] - b_mean[k] for k in b_mean}
        reward_up = delta["mean_episode_reward"] > 0
        adapt_up = delta["adaptation_accuracy"] > 0
        challenge_up = delta["challenge_action_fraction"] > 0
        comfort_down = delta["comfort_action_fraction"] < 0
        dk_up = delta["mean_delta_k_per_step"] > 0

        if reward_up and not adapt_up and comfort_down is False:
            mechanism = "reward_exploitation_suspected"
        elif adapt_up and challenge_up:
            mechanism = "better_personalization_and_challenge"
        elif adapt_up:
            mechanism = "better_personalization"
        elif challenge_up and dk_up:
            mechanism = "better_challenge_adaptation"
        else:
            mechanism = "mixed_or_unclear"

        report["comparisons"][key] = {
            "config_id": leaders[key]["config_id"],
            "metrics": m,
            "delta_vs_baseline": delta,
            "interpretation": mechanism,
            "educationally_meaningful": (
                delta["final_knowledge"] > 0
                and (adapt_up or challenge_up)
                and not (reward_up and delta["p_correct"] < -0.02)
            ),
        }

    with open(OUT_DIR / "educational_validity.json", "w") as f:
        json.dump(_json_safe(report), f, indent=2)
    return report


def final_recommendation() -> Dict[str, Any]:
    """Section 6: pick single best config from robust results if available."""
    robust_path = OUT_DIR / "robust_summary_table.csv"
    leaders_path = OUT_DIR / "grid_leaders.json"
    if robust_path.exists():
        rdf = pd.read_csv(robust_path)
        rdf = rdf.sort_values("final_knowledge_mean", ascending=False)
        best = rdf.iloc[0]
        hp = {k: best[k] for k in SEARCH_SPACE}
        rec = {
            "recommended_hyperparameters": hp,
            "config_id": str(best["config_id"]),
            "source": "robust_top5_aggregate",
            "final_knowledge_mean": float(best["final_knowledge_mean"]),
            "success_rate_mean": float(best["success_rate_mean"]),
            "mean_episode_reward_mean": float(best["mean_episode_reward_mean"]),
            "adaptation_accuracy_mean": float(best["adaptation_accuracy_mean"]),
            "stable_improvement": bool(best.get("stable_improvement", False)),
        }
    elif leaders_path.exists():
        with open(leaders_path) as f:
            leaders = json.load(f)
        rec = {
            "recommended_hyperparameters": {
                k: leaders["best_balanced"][k] for k in SEARCH_SPACE
            },
            "config_id": leaders["best_balanced"]["config_id"],
            "source": "grid_balanced_leader_screening_only",
        }
    else:
        rec = {"recommended_hyperparameters": BASELINE_HP, "source": "baseline_default"}

    validity = educational_validity()
    rec["educational_validity"] = validity.get("comparisons", {}).get(
        "best_balanced", validity.get("comparisons", {})
    )
    rec["expected_educational_impact"] = (
        "Higher mastery and emotion-aligned action selection without "
        "increasing comfort-seeking or reward-only exploitation, "
        "provided robust multi-seed confirmation holds."
    )
    rec["note"] = (
        "Environment, reward weights, and simulator dynamics were not modified; "
        "only DQN algorithm hyperparameters were tuned."
    )

    with open(OUT_DIR / "final_recommendation.json", "w") as f:
        json.dump(_json_safe(rec), f, indent=2)

    print("\n=== Section 6: recommendation ===")
    print(json.dumps(_json_safe(rec), indent=2))
    return rec


def build_report() -> None:
    """Merge artifacts into thesis-ready markdown summary."""
    lines = ["# DQN Hyperparameter Optimization Report\n"]
    bpath = OUT_DIR / "baseline_summary.json"
    if bpath.exists():
        with open(bpath) as f:
            b = json.load(f)
        lines.append("## 1. Baseline (5 seeds)\n")
        lines.append("| Metric | Mean | 95% CI |\n|--------|------|--------|\n")
        for name, key, ci_key in [
            ("Mean reward", "mean_reward", "reward_ci_95"),
            ("Success rate", "success_rate_mean", "success_rate_ci_95"),
            ("Learning gain", "learning_gain_mean", "learning_gain_ci_95"),
            ("Final knowledge", "final_knowledge_mean", "final_knowledge_ci_95"),
            ("Adaptation accuracy", "adaptation_accuracy_mean", "adaptation_accuracy_ci_95"),
        ]:
            lo, hi = b[ci_key]
            lines.append(f"| {name} | {b[key]:.4f} | [{lo:.4f}, {hi:.4f}] |\n")

    lpath = OUT_DIR / "grid_leaders.json"
    if lpath.exists():
        with open(lpath) as f:
            L = json.load(f)
        lines.append("\n## 2-3. Grid search leaders (screening seed)\n")
        for title, key in [
            ("Best reward", "best_reward"),
            ("Best mastery", "best_mastery"),
            ("Best balanced", "best_balanced"),
        ]:
            r = L[key]
            lines.append(f"\n### {title}\n")
            lines.append(f"- config_id: `{r['config_id']}`\n")
            lines.append(
                f"- reward={r.get('mean_episode_reward', 0):.3f}, "
                f"mastery={r.get('final_knowledge', 0):.3f}, "
                f"adapt={r.get('adaptation_accuracy', 0):.3f}\n"
            )

    rpath = OUT_DIR / "robust_summary_table.csv"
    if rpath.exists():
        lines.append("\n## 5. Robustness (top 5)\n")
        rdf = pd.read_csv(rpath)
        try:
            lines.append(rdf.to_markdown(index=False))
        except ImportError:
            lines.append("```\n" + rdf.to_string(index=False) + "\n```\n")
        lines.append("\n")

    evpath = OUT_DIR / "educational_validity.json"
    if evpath.exists():
        with open(evpath) as f:
            ev = json.load(f)
        lines.append("\n## 4. Educational validity\n")
        for k, v in ev.get("comparisons", {}).items():
            lines.append(
                f"- **{k}**: {v['interpretation']}, "
                f"meaningful={v['educationally_meaningful']}\n"
            )

    fpath = OUT_DIR / "final_recommendation.json"
    if fpath.exists():
        with open(fpath) as f:
            rec = json.load(f)
        lines.append("\n## 6. Recommendation\n")
        lines.append(f"```json\n{json.dumps(rec['recommended_hyperparameters'], indent=2)}\n```\n")

    (OUT_DIR / "DQN_HYPERPARAMETER_REPORT.md").write_text("".join(lines))
    print(f"Wrote {OUT_DIR / 'DQN_HYPERPARAMETER_REPORT.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="DQN hyperparameter study")
    parser.add_argument(
        "--phase",
        choices=["baseline", "grid", "robust", "report", "all"],
        default="all",
    )
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--max-configs", type=int, default=None, help="Limit grid size (debug)")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel workers for grid phase (default 1)",
    )
    args = parser.parse_args()

    n_grid = int(np.prod([len(v) for v in SEARCH_SPACE.values()]))
    print(f"DQN study | train={TRAIN_TS} eval={EVAL_EPS} | grid={n_grid} configs")

    if args.phase in ("baseline", "all"):
        run_baseline()
    if args.phase in ("grid", "all"):
        run_grid(
            resume=args.resume,
            max_configs=args.max_configs,
            workers=args.workers,
        )
    if args.phase in ("robust", "all"):
        if args.phase == "robust" and not (OUT_DIR / "grid_rankings.csv").exists():
            run_grid(
                resume=args.resume,
                max_configs=args.max_configs,
                workers=args.workers,
            )
        run_robust(top_n=args.top_n)
    if args.phase in ("report", "all"):
        educational_validity()
        final_recommendation()
        build_report()


if __name__ == "__main__":
    main()
