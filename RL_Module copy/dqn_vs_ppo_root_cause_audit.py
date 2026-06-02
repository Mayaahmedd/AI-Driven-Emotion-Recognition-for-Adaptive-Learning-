"""
Master diagnostic audit: why DQN still outperforms PPO under identical dynamics.

Scope:
- Diagnostic only; does not modify environment dynamics or reward/action definitions.
- Uses symmetric gain setting (G0): GAIN_CORRECT_FACTOR=1.0, GAIN_INCORRECT_FACTOR=1.0
  to isolate mechanisms that remain after removing 10:1 asymmetry.

Run:
  python -m RL_Module.dqn_vs_ppo_root_cause_audit
"""

from __future__ import annotations

import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import RL_Module.config as cfg
from RL_Module.agents.dqn_agent import DQNAgent, make_env
from RL_Module.agents.ppo_agent import PPOAgent, make_masked_env
from RL_Module.environment.student_env import StudentEnv
from RL_Module.evaluation.metrics import confidence_interval
from RL_Module.mdp_definition import ACTIONS, ID_TO_ACTION

OUT_DIR = _HERE / "figures" / "dqn_vs_ppo_root_cause"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 456]
TRAIN_TS = 50_000
EVAL_EPS = 300
GAIN_G0 = {"GAIN_CORRECT_FACTOR": 1.0, "GAIN_INCORRECT_FACTOR": 1.0}


def _patch_gain(params: Dict[str, float]) -> Dict[str, float]:
    prev = dict(cfg.SIMULATOR_PARAMS)
    cfg.SIMULATOR_PARAMS = dict(params)
    return prev


def _restore_gain(snapshot: Dict[str, float]) -> None:
    cfg.SIMULATOR_PARAMS = snapshot


def _shannon_entropy(probs: np.ndarray) -> float:
    p = probs[probs > 0]
    if len(p) == 0:
        return 0.0
    return float(-np.sum(p * np.log(p)))


def _action_freq(rows: pd.DataFrame) -> Dict[str, float]:
    counts = (
        rows["action_name"].value_counts(normalize=True).reindex(list(ACTIONS), fill_value=0.0)
    )
    return {k: float(v) for k, v in counts.items()}


def _state_stats(df: pd.DataFrame, col: str) -> Dict[str, float]:
    x = df[col].values
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        "p10": float(np.percentile(x, 10)),
        "p25": float(np.percentile(x, 25)),
        "p75": float(np.percentile(x, 75)),
        "p90": float(np.percentile(x, 90)),
    }


def _collect_eval_trace(agent: Any, seed: int, n_episodes: int, algo_name: str) -> pd.DataFrame:
    cfg.set_all_seeds(seed)
    env = StudentEnv(population_seed=seed)
    env.set_algorithm_name(algo_name)
    rows: List[Dict[str, Any]] = []

    for ep in range(1, n_episodes + 1):
        obs, info = env.reset(seed=seed + ep)
        mask = info["action_masks"]
        step = 0
        done = False
        prev_k = env._state.knowledge
        while not done:
            if algo_name == "PPO":
                action = int(agent.predict(obs, mask))
                probs = agent.get_action_probs(obs, mask)
                policy_entropy = _shannon_entropy(probs)
            else:
                action = int(agent.predict(obs, mask))
                qvals = agent.get_q_values(obs)
                bmask = mask.astype(bool)
                m_q = qvals.copy()
                m_q[~bmask] = -np.inf
                valid = m_q[np.isfinite(m_q)]
                top = float(np.max(valid)) if len(valid) else float("nan")
                second = float(np.partition(valid, -2)[-2]) if len(valid) >= 2 else float("nan")
                policy_entropy = float("nan")

            obs2, reward, terminated, truncated, info = env.step(action)
            k = float(info["knowledge"])
            difficulty = float(info["difficulty"])
            mismatch = difficulty - k
            delta_k = k - prev_k
            prev_k = k

            row = {
                "algorithm": algo_name,
                "seed": seed,
                "episode": ep,
                "step": step,
                "action_id": int(action),
                "action_name": info["action_name"],
                "reward": float(reward),
                "correct": int(bool(info["last_answer_correct"])),
                "knowledge": k,
                "engagement": float(info["engagement"]),
                "frustration": float(info["frustration"]),
                "confusion": float(info["confusion"]),
                "boredom": float(info["boredom"]),
                "difficulty": difficulty,
                "mismatch": mismatch,
                "delta_k": delta_k,
                "valid_actions_count": int(np.sum(mask)),
                "invalid_actions_count": int(len(mask) - np.sum(mask)),
                "policy_entropy": policy_entropy,
            }

            if algo_name == "DQN":
                row["q_top"] = top
                row["q_margin"] = top - second if np.isfinite(top) and np.isfinite(second) else float("nan")
                row["q_harder"] = float(qvals[4])
                row["q_scaffold"] = float(qvals[1])
                row["q_hint"] = float(qvals[0])

            for aid, allowed in enumerate(mask):
                row[f"mask_valid_{ID_TO_ACTION[aid]}"] = int(allowed)

            rows.append(row)
            obs = obs2
            mask = info["action_masks"]
            step += 1
            done = bool(terminated or truncated)

    env.close()
    return pd.DataFrame(rows)


def _episode_summary(trace: pd.DataFrame) -> pd.DataFrame:
    ep = (
        trace.groupby(["algorithm", "seed", "episode"])
        .agg(
            total_reward=("reward", "sum"),
            final_k=("knowledge", "last"),
            start_k=("knowledge", "first"),
            final_f=("frustration", "last"),
            final_c=("confusion", "last"),
        )
        .reset_index()
    )
    ep["success"] = (
        (ep["final_k"] > 0.8) & (ep["final_f"] < 0.3) & (ep["final_c"] < 0.4)
    ).astype(int)
    ep["learning_gain"] = ep["final_k"] - ep["start_k"]
    return ep


def _aggregate_alg_seed_metrics(trace: pd.DataFrame, ep: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (algo, seed), sub in trace.groupby(["algorithm", "seed"]):
        esub = ep[(ep["algorithm"] == algo) & (ep["seed"] == seed)]
        rows.append(
            {
                "algorithm": algo,
                "seed": seed,
                "mean_reward": float(esub["total_reward"].mean()),
                "success_rate": float(esub["success"].mean()),
                "learning_gain": float(esub["learning_gain"].mean()),
                "final_knowledge": float(esub["final_k"].mean()),
                "mean_delta_k_per_step": float(sub["delta_k"].mean()),
                "p_correct": float(sub["correct"].mean()),
                "mean_frustration": float(sub["frustration"].mean()),
                "mean_confusion": float(sub["confusion"].mean()),
                "mean_engagement": float(sub["engagement"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _aggregate_mean_std(df: pd.DataFrame, metric_cols: List[str]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for algo in ["PPO", "DQN"]:
        sub = df[df["algorithm"] == algo]
        out[algo] = {}
        for m in metric_cols:
            vals = sub[m].tolist()
            mean, lo, hi = confidence_interval(vals)
            out[algo][m] = float(mean)
            out[algo][f"{m}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            out[algo][f"{m}_ci_lo"] = float(lo)
            out[algo][f"{m}_ci_hi"] = float(hi)
    return out


def _action_conditional(trace: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for algo in ["PPO", "DQN"]:
        sub = trace[trace["algorithm"] == algo]
        stats_rows = []
        for a in ACTIONS:
            a_sub = sub[sub["action_name"] == a]
            if len(a_sub) == 0:
                stats_rows.append({"action": a, "n": 0})
                continue
            stats_rows.append(
                {
                    "action": a,
                    "n": int(len(a_sub)),
                    "freq": float(len(a_sub) / len(sub)),
                    "E_delta_k": float(a_sub["delta_k"].mean()),
                    "P_correct": float(a_sub["correct"].mean()),
                    "E_reward": float(a_sub["reward"].mean()),
                    "mean_mismatch": float(a_sub["mismatch"].mean()),
                    "mean_difficulty": float(a_sub["difficulty"].mean()),
                }
            )
        out[algo] = stats_rows
    return out


def _state_visitation(trace: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    cols = ["knowledge", "frustration", "confusion", "boredom", "engagement", "mismatch"]
    for algo in ["PPO", "DQN"]:
        sub = trace[trace["algorithm"] == algo]
        out[algo] = {c: _state_stats(sub, c) for c in cols}
    return out


def _challenge_seeking(trace: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for algo in ["PPO", "DQN"]:
        sub = trace[trace["algorithm"] == algo]
        freq = _action_freq(sub)
        out[algo] = {
            "harder_problem_freq": float(freq["harder_problem"]),
            "simplify_problem_freq": float(freq["simplify_problem"]),
            "avg_difficulty": float(sub["difficulty"].mean()),
            "mismatch_stats": _state_stats(sub, "mismatch"),
            "action_frequencies": freq,
        }
    return out


def _mask_analysis(trace: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    mask_cols = [f"mask_valid_{a}" for a in ACTIONS]
    for algo in ["PPO", "DQN"]:
        sub = trace[trace["algorithm"] == algo]
        per_action_valid = {a: float(sub[f"mask_valid_{a}"].mean()) for a in ACTIONS}
        out[algo] = {
            "avg_valid_actions": float(sub["valid_actions_count"].mean()),
            "avg_invalid_actions": float(sub["invalid_actions_count"].mean()),
            "valid_actions_std": float(sub["valid_actions_count"].std(ddof=1)),
            "mask_valid_rate_by_action": per_action_valid,
            "mask_frequency_any_invalid": float((sub["invalid_actions_count"] > 0).mean()),
            "n_steps": int(len(sub)),
            "mask_columns_used": mask_cols,
        }
    return out


def _build_probe_states(seed: int, n_steps: int = 2000) -> List[Dict[str, Any]]:
    """Generate a fixed probe set by random valid-action rollout."""
    cfg.set_all_seeds(seed)
    env = StudentEnv(population_seed=seed)
    probes: List[Dict[str, Any]] = []
    obs, info = env.reset(seed=seed + 10_000)
    mask = info["action_masks"]
    done = False
    step = 0
    while len(probes) < n_steps:
        valid_actions = np.flatnonzero(mask)
        action = int(np.random.choice(valid_actions))
        probes.append({"obs": obs.copy(), "mask": mask.copy(), "state": env._state.copy(), "difficulty": env._current_difficulty})
        obs, _, terminated, truncated, info = env.step(action)
        mask = info["action_masks"]
        step += 1
        done = bool(terminated or truncated)
        if done:
            obs, info = env.reset(seed=seed + 10_000 + step)
            mask = info["action_masks"]
    env.close()
    return probes


def _ppo_entropy_over_training(seed: int, probes: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    """Section 6: entropy/action-diversity trajectory from same policy over checkpoints."""
    cfg.set_all_seeds(seed)
    env = make_masked_env(seed=seed, algo_tag=f"root_audit_entropy_s{seed}")
    agent = PPOAgent()
    agent.train(env, total_timesteps=1, seed=seed)  # initialize model
    checkpoints = [0, 5_000, 10_000, 20_000, 30_000, 40_000, 50_000]
    out = []
    last = 0

    def _probe_stats() -> Tuple[float, float]:
        ent = []
        actions = []
        for p in probes:
            probs = agent.get_action_probs(p["obs"], p["mask"])
            ent.append(_shannon_entropy(probs))
            actions.append(int(np.argmax(probs)))
        counts = np.bincount(actions, minlength=8).astype(float)
        probs_a = counts / max(np.sum(counts), 1.0)
        return float(np.mean(ent)), _shannon_entropy(probs_a)

    e0, d0 = _probe_stats()
    out.append({"timesteps": 0, "policy_entropy": e0, "action_diversity": d0})
    for cp in checkpoints[1:]:
        delta = cp - last
        agent.model.learn(total_timesteps=delta, reset_num_timesteps=False)
        last = cp
        e, d = _probe_stats()
        out.append({"timesteps": cp, "policy_entropy": e, "action_diversity": d})
    env.close()
    return out


def _dqn_q_value_diagnostics(agent: DQNAgent, probes: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows = []
    for p in probes:
        obs = p["obs"]
        mask = p["mask"].astype(bool)
        q = agent.get_q_values(obs)
        mq = q.copy()
        mq[~mask] = -np.inf
        aid = int(np.argmax(mq))
        valid = mq[np.isfinite(mq)]
        top = float(np.max(valid)) if len(valid) else float("nan")
        second = float(np.partition(valid, -2)[-2]) if len(valid) >= 2 else float("nan")
        rows.append(
            {
                "chosen_action": ID_TO_ACTION[aid],
                "q_top": top,
                "q_margin": top - second if np.isfinite(top) and np.isfinite(second) else float("nan"),
                "q_harder_problem": float(q[4]),
                "q_scaffold": float(q[1]),
                "q_hint": float(q[0]),
                "knowledge": float(p["state"].knowledge),
                "frustration": float(p["state"].frustration),
                "confusion": float(p["state"].confusion),
                "engagement": float(p["state"].engagement),
                "difficulty": float(p["difficulty"]),
            }
        )
    df = pd.DataFrame(rows)
    by_action = (
        df.groupby("chosen_action")
        .agg(
            n=("chosen_action", "size"),
            q_top=("q_top", "mean"),
            q_margin=("q_margin", "mean"),
            q_harder_problem=("q_harder_problem", "mean"),
            q_scaffold=("q_scaffold", "mean"),
            q_hint=("q_hint", "mean"),
        )
        .reset_index()
        .to_dict(orient="records")
    )
    return {
        "overall": {
            "q_top_mean": float(df["q_top"].mean()),
            "q_margin_mean": float(df["q_margin"].mean()),
            "q_harder_minus_scaffold_mean": float((df["q_harder_problem"] - df["q_scaffold"]).mean()),
            "q_harder_minus_hint_mean": float((df["q_harder_problem"] - df["q_hint"]).mean()),
        },
        "by_chosen_action": by_action,
    }


def _matched_state_comparison(ppo: PPOAgent, dqn: DQNAgent, probes: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows = []
    for p in probes:
        obs = p["obs"]
        mask = p["mask"]
        ppo_a = int(ppo.predict(obs, mask))
        dqn_a = int(dqn.predict(obs, mask))
        rows.append(
            {
                "knowledge": float(p["state"].knowledge),
                "engagement": float(p["state"].engagement),
                "frustration": float(p["state"].frustration),
                "confusion": float(p["state"].confusion),
                "boredom": float(p["state"].boredom),
                "difficulty": float(p["difficulty"]),
                "mismatch": float(p["difficulty"] - p["state"].knowledge),
                "ppo_action": ID_TO_ACTION[ppo_a],
                "dqn_action": ID_TO_ACTION[dqn_a],
                "same_action": int(ppo_a == dqn_a),
            }
        )
    df = pd.DataFrame(rows)
    diverge = df[df["same_action"] == 0].copy()
    if len(diverge) > 0:
        diverge["abs_mismatch"] = diverge["mismatch"].abs()
        examples = (
            diverge.sort_values("abs_mismatch", ascending=False)
            .head(20)[
                [
                    "knowledge",
                    "engagement",
                    "frustration",
                    "confusion",
                    "boredom",
                    "difficulty",
                    "mismatch",
                    "ppo_action",
                    "dqn_action",
                ]
            ]
            .round(4)
            .to_dict(orient="records")
        )
    else:
        examples = []

    divergence_table = (
        df.groupby(["ppo_action", "dqn_action"]).size().reset_index(name="count").sort_values("count", ascending=False)
    )
    return {
        "divergence_rate": float(1.0 - df["same_action"].mean()),
        "agreement_rate": float(df["same_action"].mean()),
        "top_divergence_pairs": divergence_table.head(20).to_dict(orient="records"),
        "state_action_examples": examples,
    }


def _rank_root_causes(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    gaps = report["performance_summary"]["gap_dqn_minus_ppo"]
    challenge = report["challenge_seeking"]
    matched = report["matched_state_comparison"]
    mask = report["mask_analysis"]
    ppo_ent = report["ppo_exploration_curve_seed42"]

    ppo_entropy_drop = ppo_ent[0]["policy_entropy"] - ppo_ent[-1]["policy_entropy"]
    ppo_div_drop = ppo_ent[0]["action_diversity"] - ppo_ent[-1]["action_diversity"]

    ranking = [
        {
            "rank": 1,
            "cause": "Challenge-seeking action preference under value maximization",
            "evidence": {
                "harder_problem_freq_ppo": challenge["PPO"]["harder_problem_freq"],
                "harder_problem_freq_dqn": challenge["DQN"]["harder_problem_freq"],
                "avg_mismatch_ppo": challenge["PPO"]["mismatch_stats"]["mean"],
                "avg_mismatch_dqn": challenge["DQN"]["mismatch_stats"]["mean"],
            },
        },
        {
            "rank": 2,
            "cause": "State-visitation difference (DQN spends more time in high-learning regions)",
            "evidence": {
                "knowledge_mean_ppo": report["state_visitation"]["PPO"]["knowledge"]["mean"],
                "knowledge_mean_dqn": report["state_visitation"]["DQN"]["knowledge"]["mean"],
                "difficulty_mean_ppo": challenge["PPO"]["avg_difficulty"],
                "difficulty_mean_dqn": challenge["DQN"]["avg_difficulty"],
            },
        },
        {
            "rank": 3,
            "cause": "PPO conservative optimization (high agreement with comfort actions, lower mastery objective realization)",
            "evidence": {
                "divergence_rate_matched_states": matched["divergence_rate"],
                "ppo_entropy_drop": ppo_entropy_drop,
                "ppo_diversity_drop": ppo_div_drop,
                "success_gap": gaps["success_rate"],
            },
        },
        {
            "rank": 4,
            "cause": "DQN action-value exploitation and clearer top-action margins",
            "evidence": report["dqn_value_analysis"]["overall"],
        },
        {
            "rank": 5,
            "cause": "Mask interaction appears secondary",
            "evidence": {
                "avg_valid_actions_ppo": mask["PPO"]["avg_valid_actions"],
                "avg_valid_actions_dqn": mask["DQN"]["avg_valid_actions"],
                "mask_any_invalid_ppo": mask["PPO"]["mask_frequency_any_invalid"],
                "mask_any_invalid_dqn": mask["DQN"]["mask_frequency_any_invalid"],
            },
        },
    ]
    return ranking


def run_audit() -> Dict[str, Any]:
    print("=== Root-cause audit (G0 symmetric gains) ===")
    cfg.USE_SENSITIVITY_WEIGHTS = False
    snap = _patch_gain(GAIN_G0)
    try:
        traces = []
        perf_seed_rows = []
        trained_agents: Dict[Tuple[str, int], Any] = {}

        for seed in SEEDS:
            print(f"\n[seed={seed}] training PPO and DQN ...")
            cfg.set_all_seeds(seed)

            ppo_env = make_masked_env(seed=seed, algo_tag=f"root_audit_ppo_s{seed}")
            ppo = PPOAgent()
            t0 = time.time()
            ppo.train(ppo_env, TRAIN_TS, seed)
            ppo_env.close()
            print(f"  PPO train: {time.time() - t0:.1f}s")

            dqn_env = make_env(seed=seed, algo_tag=f"root_audit_dqn_s{seed}")
            dqn = DQNAgent()
            t0 = time.time()
            dqn.train(dqn_env, TRAIN_TS, seed)
            dqn_env.close()
            print(f"  DQN train: {time.time() - t0:.1f}s")

            trained_agents[("PPO", seed)] = ppo
            trained_agents[("DQN", seed)] = dqn

            ppo_trace = _collect_eval_trace(ppo, seed, EVAL_EPS, "PPO")
            dqn_trace = _collect_eval_trace(dqn, seed, EVAL_EPS, "DQN")
            trace = pd.concat([ppo_trace, dqn_trace], ignore_index=True)
            trace["seed"] = seed
            traces.append(trace)

            ep = _episode_summary(trace)
            perf = _aggregate_alg_seed_metrics(trace, ep)
            perf_seed_rows.append(perf)

        trace_all = pd.concat(traces, ignore_index=True)
        perf_seed = pd.concat(perf_seed_rows, ignore_index=True)

        metric_cols = [
            "mean_reward",
            "success_rate",
            "learning_gain",
            "final_knowledge",
            "mean_delta_k_per_step",
            "p_correct",
            "mean_frustration",
            "mean_confusion",
            "mean_engagement",
        ]
        perf_agg = _aggregate_mean_std(perf_seed, metric_cols)

        gap = {
            m: float(perf_agg["DQN"][m] - perf_agg["PPO"][m]) for m in metric_cols
        }

        state_visitation = _state_visitation(trace_all)
        action_conditional = _action_conditional(trace_all)
        challenge = _challenge_seeking(trace_all)
        masks = _mask_analysis(trace_all)

        # Deep-dive on seed 42 for sections 6, 7, 8
        probes = _build_probe_states(seed=42, n_steps=2000)
        ppo_curve = _ppo_entropy_over_training(seed=42, probes=probes)
        ppo42 = trained_agents[("PPO", 42)]
        dqn42 = trained_agents[("DQN", 42)]
        dqn_values = _dqn_q_value_diagnostics(dqn42, probes)
        matched_cmp = _matched_state_comparison(ppo42, dqn42, probes)

        report = {
            "study": "dqn_vs_ppo_root_cause_audit",
            "config": {
                "gain_correct_factor": 1.0,
                "gain_incorrect_factor": 1.0,
                "seeds": SEEDS,
                "train_timesteps": TRAIN_TS,
                "eval_episodes": EVAL_EPS,
            },
            "performance_summary": {
                "per_seed": perf_seed.to_dict(orient="records"),
                "aggregated": perf_agg,
                "gap_dqn_minus_ppo": gap,
            },
            "state_visitation": state_visitation,
            "action_conditional_knowledge_correctness_reward": action_conditional,
            "challenge_seeking": challenge,
            "mask_analysis": masks,
            "ppo_exploration_curve_seed42": ppo_curve,
            "dqn_value_analysis": dqn_values,
            "matched_state_comparison": matched_cmp,
        }
        report["root_cause_ranking"] = _rank_root_causes(report)

        with open(OUT_DIR / "root_cause_audit_report.json", "w") as f:
            json.dump(report, f, indent=2)
        perf_seed.to_csv(OUT_DIR / "performance_per_seed.csv", index=False)
        trace_all.to_csv(OUT_DIR / "trace_all_steps.csv", index=False)
        pd.DataFrame(matched_cmp["top_divergence_pairs"]).to_csv(
            OUT_DIR / "matched_state_divergence_pairs.csv", index=False
        )
        pd.DataFrame(matched_cmp["state_action_examples"]).to_csv(
            OUT_DIR / "matched_state_examples.csv", index=False
        )

        print("\n=== DQN - PPO gap (G0) ===")
        for k, v in gap.items():
            print(f"  {k:24s}: {v:+.4f}")
        print(f"\nSaved report to {OUT_DIR / 'root_cause_audit_report.json'}")
        return report
    finally:
        _restore_gain(snap)


if __name__ == "__main__":
    run_audit()
