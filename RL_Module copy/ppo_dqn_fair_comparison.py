"""
Fair PPO vs DQN comparison under identical environment, reward, state, action, and masking.

Prerequisites:
  - DQN tuned: RL_Module/figures/dqn_hyperparameter_study/final_recommendation.json
  - PPO tuned: RL_Module/figures/ppo_hyperparameter_study/final_recommendation.json
    (run ppo_hyperparameter_study first, or pass --ppo-hp-json)

Phases:
  train_eval       - train both algorithms at 50k/100k/200k/500k x 5 seeds
  matched_state    - 10k+ identical observations, agreement/disagreement
  disagreement_cf  - future knowledge gain at 5/10/20 steps on disagreement states
  dqn_eval_modes   - greedy vs epsilon=0.05 vs softmax(Q/T)
  diagnosis        - classify DQN superiority mechanism
  ranking          - final algorithm ranking (mastery, validity, adaptation, robustness)
  report           - markdown summary

Run:
  python -m RL_Module.ppo_dqn_fair_comparison --phase all
  python -m RL_Module.ppo_dqn_fair_comparison --phase train_eval --budget 50000 --resume
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from dataclasses import dataclass
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
from RL_Module.agents.ppo_agent import PPOAgent, make_masked_env
from RL_Module.environment.student_env import StudentEnv
from RL_Module.evaluation.metrics import (
    _adaptation_match,
    _is_success,
    action_frequency_report,
    confidence_interval,
    normalized_knowledge_gain,
)
from RL_Module.mdp_definition import ACTIONS, ID_TO_ACTION

OUT_DIR = _HERE / "figures" / "ppo_dqn_fair_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FAIR_SEEDS = [42, 123, 999, 2024, 777]
TRAIN_BUDGETS = [50_000, 100_000, 200_000, 500_000]
EVAL_EPS = cfg.EVAL_EPISODES
N_PROBE = 10_000
DISAGREEMENT_HORIZONS = [5, 10, 20]
PROBE_SEED = 42
DQN_EVAL_MODES = [
    ("greedy", {"eval_mode": "greedy"}),
    ("epsilon_0.05", {"eval_mode": "epsilon", "epsilon": 0.05}),
    ("softmax", {"eval_mode": "softmax", "temperature": 1.0}),
]

COMFORT_ACTIONS = {"encouragement", "simplify_problem", "break"}
CHALLENGE_ACTIONS = {"scaffold", "harder_problem", "explanation"}

DQN_HP_PATH = _HERE / "figures" / "dqn_hyperparameter_study" / "final_recommendation.json"
PPO_HP_PATH = _HERE / "figures" / "ppo_hyperparameter_study" / "final_recommendation.json"


@dataclass
class EnvSnapshot:
    student: Any
    state: Any
    prev_state: Any
    cooldowns: Dict[int, int]
    persistent_flag: bool
    frustration_high_streak: int
    consecutive_flag_steps: int
    frustration_history: List[float]
    last_answer_wrong: bool
    last_answer_correct: bool
    current_difficulty: float
    step_count: int
    cumulative_reward: float
    reward_history: List[float]
    last_action: int
    episode_count: int


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


def _load_hp(path: Path, fallback: Dict[str, Any]) -> Dict[str, Any]:
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        return dict(data["recommended_hyperparameters"])
    print(f"  [warn] {path} missing; using defaults")
    return dict(fallback)


def _shannon_entropy_from_freq(freq: Dict[str, float]) -> float:
    p = np.array([v for v in freq.values() if v > 0], dtype=float)
    return float(-np.sum(p * np.log(p))) if len(p) else 0.0


def _mean_std(vals: List[float]) -> Dict[str, float]:
    arr = np.array(vals, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "n": len(arr),
    }


def _capture_snapshot(env: StudentEnv) -> EnvSnapshot:
    return EnvSnapshot(
        student=copy.deepcopy(env._student),
        state=env._state.copy(),
        prev_state=env._prev_state.copy(),
        cooldowns=dict(env._cooldowns),
        persistent_flag=bool(env._persistent_frustration_flag),
        frustration_high_streak=int(env._frustration_high_streak),
        consecutive_flag_steps=int(env._consecutive_flag_steps),
        frustration_history=list(env._frustration_history),
        last_answer_wrong=bool(env._last_answer_wrong),
        last_answer_correct=bool(env._last_answer_correct),
        current_difficulty=float(env._current_difficulty),
        step_count=int(env._step_count),
        cumulative_reward=float(env._cumulative_reward),
        reward_history=list(env._reward_history),
        last_action=int(env._last_action),
        episode_count=int(env._episode_count),
    )


def _restore_snapshot(env: StudentEnv, snap: EnvSnapshot) -> Tuple[np.ndarray, Dict[str, Any]]:
    env._student = copy.deepcopy(snap.student)
    env._state = snap.state.copy()
    env._prev_state = snap.prev_state.copy()
    env._cooldowns = dict(snap.cooldowns)
    env._persistent_frustration_flag = bool(snap.persistent_flag)
    env._frustration_high_streak = int(snap.frustration_high_streak)
    env._consecutive_flag_steps = int(snap.consecutive_flag_steps)
    env._frustration_history.clear()
    for v in snap.frustration_history:
        env._frustration_history.append(float(v))
    env._last_answer_wrong = bool(snap.last_answer_wrong)
    env._last_answer_correct = bool(snap.last_answer_correct)
    env._current_difficulty = float(snap.current_difficulty)
    env._step_count = int(snap.step_count)
    env._cumulative_reward = float(snap.cumulative_reward)
    env._reward_history = [float(v) for v in snap.reward_history]
    env._last_action = int(snap.last_action)
    env._episode_count = int(snap.episode_count)
    obs = env._obs()
    info = {
        "action_masks": env.get_action_mask(),
        "persistent_frustration_flag": env._persistent_frustration_flag,
    }
    return obs, info


def evaluate_agent(
    agent: Any,
    env: StudentEnv,
    n_episodes: int,
    seed: int,
    algorithm: str,
    dqn_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Extended evaluation with action mix and pedagogical diagnostics."""
    cfg.set_all_seeds(seed)
    dqn_kwargs = dqn_kwargs or {}
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
            if isinstance(agent, DQNAgent) and dqn_kwargs:
                action = agent.predict(obs, mask, **dqn_kwargs)
            else:
                action = agent.predict(obs, mask)

            obs, reward, terminated, truncated, info = env.step(action)
            k_now = env._state.knowledge
            difficulty = float(env._current_difficulty)
            mismatch = difficulty - k_now
            step_rows.append({
                "episode": ep,
                "step": step,
                "knowledge": k_now,
                "difficulty": difficulty,
                "mismatch": mismatch,
                "delta_k": k_now - prev_k,
                "correct": int(bool(info.get("last_answer_correct", True))),
                "action_id": action,
                "action_name": info.get("action_name", ID_TO_ACTION[action]),
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
        "learning_gain": float(np.mean([r["learning_gain_normalized"] for r in episode_results])),
        "final_knowledge": float(np.mean([r["final_knowledge"] for r in episode_results])),
        "adaptation_accuracy": float(np.mean([r["adaptation_accuracy"] for r in episode_results])),
        "comfort_action_fraction": comfort_frac,
        "challenge_action_fraction": challenge_frac,
        "action_diversity_shannon": _shannon_entropy_from_freq(freqs),
        "action_frequencies": {k: round(v, 4) for k, v in freqs.items()},
        "dominance_detected": freq_report["dominance_detected"],
        "dominant_actions": freq_report["dominant_actions"],
        "mean_delta_k_per_step": float(sdf["delta_k"].mean()) if len(sdf) else 0.0,
        "p_correct": float(sdf["correct"].mean()) if len(sdf) else 0.0,
        "mean_difficulty": float(sdf["difficulty"].mean()) if len(sdf) else 0.0,
        "mean_mismatch": float(sdf["mismatch"].mean()) if len(sdf) else 0.0,
        "knowledge_trajectory_mean": float(sdf.groupby("step")["knowledge"].mean().iloc[-1])
        if len(sdf) else 0.0,
    }


def train_one(
    algorithm: str,
    seed: int,
    budget: int,
    ppo_hp: Dict[str, Any],
    dqn_hp: Dict[str, Any],
) -> Any:
    cfg.set_all_seeds(seed)
    tag = f"fair_{algorithm.lower()}_b{budget}_s{seed}"
    if algorithm == "PPO":
        env = make_masked_env(seed=seed, algo_tag=tag)
        agent = PPOAgent()
        agent.train(env, budget, seed, hyperparams=ppo_hp)
        env.close()
    else:
        env = make_env(seed=seed, algo_tag=tag)
        agent = DQNAgent()
        agent.train(env, budget, seed, hyperparams=dqn_hp)
        env.close()
    return agent


def run_train_eval(
    ppo_hp: Dict[str, Any],
    dqn_hp: Dict[str, Any],
    budgets: List[int],
    seeds: List[int],
    resume: bool = True,
    budget_filter: Optional[int] = None,
) -> pd.DataFrame:
    path = OUT_DIR / "train_eval_per_run.csv"
    done: set = set()
    rows: List[Dict[str, Any]] = []
    if resume and path.exists():
        prev = pd.read_csv(path)
        rows = prev.to_dict("records")
        done = {(r["algorithm"], int(r["budget"]), int(r["seed"])) for r in rows}

    active_budgets = [budget_filter] if budget_filter else budgets
    for budget in active_budgets:
        for seed in seeds:
            for algorithm, hp in [("PPO", ppo_hp), ("DQN", dqn_hp)]:
                key = (algorithm, budget, seed)
                if key in done:
                    continue
                print(f"\n[train_eval] {algorithm} budget={budget} seed={seed}")
                t0 = time.time()
                agent = train_one(algorithm, seed, budget, ppo_hp, dqn_hp)
                eval_env = StudentEnv(population_seed=seed)
                eval_env.set_algorithm_name(f"{algorithm}_fair")
                result = evaluate_agent(agent, eval_env, EVAL_EPS, seed, algorithm)
                eval_env.close()
                row = {
                    "algorithm": algorithm,
                    "budget": budget,
                    "seed": seed,
                    "eval_episodes": EVAL_EPS,
                    "elapsed_seconds": round(time.time() - t0, 1),
                    "hyperparameters": json.dumps(_json_safe(hp)),
                    **{k: result[k] for k in [
                        "mean_episode_reward",
                        "success_rate",
                        "learning_gain",
                        "final_knowledge",
                        "adaptation_accuracy",
                        "comfort_action_fraction",
                        "challenge_action_fraction",
                        "action_diversity_shannon",
                        "mean_delta_k_per_step",
                        "p_correct",
                        "dominance_detected",
                    ]},
                    "action_frequencies": json.dumps(result["action_frequencies"]),
                }
                rows.append(row)
                pd.DataFrame(rows).to_csv(path, index=False)
                done.add(key)
                print(
                    f"  reward={row['mean_episode_reward']:.3f} "
                    f"succ={row['success_rate']:.3f} fk={row['final_knowledge']:.3f}"
                )

    df = pd.DataFrame(rows)
    summarize_train_eval(df)
    return df


def summarize_train_eval(df: pd.DataFrame) -> None:
    metrics = [
        "mean_episode_reward",
        "success_rate",
        "learning_gain",
        "final_knowledge",
        "adaptation_accuracy",
        "comfort_action_fraction",
        "challenge_action_fraction",
        "action_diversity_shannon",
    ]
    summaries = []
    for (algo, budget), grp in df.groupby(["algorithm", "budget"]):
        entry: Dict[str, Any] = {"algorithm": algo, "budget": int(budget), "n_seeds": len(grp)}
        for m in metrics:
            stats = _mean_std(grp[m].astype(float).tolist())
            entry[f"{m}_mean"] = stats["mean"]
            entry[f"{m}_std"] = stats["std"]
        summaries.append(entry)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(OUT_DIR / "train_eval_summary.csv", index=False)

    pivot_rows = []
    sig_rows = []
    for budget in sorted(df["budget"].unique()):
        sub = summary_df[summary_df["budget"] == budget]
        row: Dict[str, Any] = {"budget": int(budget)}
        ppo_sub = df[(df["budget"] == budget) & (df["algorithm"] == "PPO")]
        dqn_sub = df[(df["budget"] == budget) & (df["algorithm"] == "DQN")]
        for algo in ["PPO", "DQN"]:
            a = sub[sub["algorithm"] == algo]
            if len(a) == 0:
                continue
            a = a.iloc[0]
            for m in metrics:
                row[f"{algo}_{m}"] = f"{a[f'{m}_mean']:.4f} +/- {a[f'{m}_std']:.4f}"
        if len(ppo_sub) >= 2 and len(dqn_sub) >= 2:
            for m in [
                "mean_episode_reward",
                "success_rate",
                "learning_gain",
                "final_knowledge",
                "adaptation_accuracy",
            ]:
                _, p = stats.ttest_ind(
                    dqn_sub[m].astype(float),
                    ppo_sub[m].astype(float),
                    equal_var=False,
                )
                sig_rows.append({
                    "budget": int(budget),
                    "metric": m,
                    "ppo_mean": float(ppo_sub[m].mean()),
                    "dqn_mean": float(dqn_sub[m].mean()),
                    "delta_dqn_minus_ppo": float(dqn_sub[m].mean() - ppo_sub[m].mean()),
                    "p_value": float(p),
                    "significant_0.05": bool(p < 0.05),
                })
        pivot_rows.append(row)
    pd.DataFrame(pivot_rows).to_csv(OUT_DIR / "train_eval_formatted.csv", index=False)
    if sig_rows:
        pd.DataFrame(sig_rows).to_csv(OUT_DIR / "train_eval_significance.csv", index=False)


def _build_probes(seed: int, n: int) -> List[Dict[str, Any]]:
    cfg.set_all_seeds(seed)
    env = StudentEnv(population_seed=seed)
    probes: List[Dict[str, Any]] = []
    obs, info = env.reset(seed=seed + 20_000)
    mask = info["action_masks"]
    step = 0
    while len(probes) < n:
        probes.append({
            "obs": obs.copy(),
            "mask": mask.copy(),
            "snapshot": _capture_snapshot(env),
            "knowledge": float(env._state.knowledge),
            "emotion_id": int(env._state.emotion_id),
            "difficulty": float(env._current_difficulty),
        })
        valid = np.flatnonzero(mask)
        action = int(np.random.choice(valid))
        obs, _, term, trunc, info = env.step(action)
        mask = info["action_masks"]
        step += 1
        if term or trunc:
            obs, info = env.reset(seed=seed + 20_000 + step)
            mask = info["action_masks"]
    env.close()
    return probes


def run_matched_state(
    ppo_hp: Dict[str, Any],
    dqn_hp: Dict[str, Any],
    budget: int = 500_000,
    seed: int = PROBE_SEED,
) -> Dict[str, Any]:
    """Train best configs at reference budget, compare on 10k+ identical observations."""
    print(f"\n[matched_state] training PPO/DQN at budget={budget} seed={seed}")
    ppo = train_one("PPO", seed, budget, ppo_hp, dqn_hp)
    dqn = train_one("DQN", seed, budget, ppo_hp, dqn_hp)

    probes = _build_probes(seed=seed, n=N_PROBE)
    rows = []
    for i, p in enumerate(probes):
        obs = p["obs"]
        mask = p["mask"]
        ppo_a = int(ppo.predict(obs, mask))
        dqn_a = int(dqn.predict(obs, mask))
        rows.append({
            "probe_id": i,
            "knowledge": p["knowledge"],
            "emotion_id": p["emotion_id"],
            "difficulty": p["difficulty"],
            "mismatch": p["difficulty"] - p["knowledge"],
            "ppo_action": ID_TO_ACTION[ppo_a],
            "dqn_action": ID_TO_ACTION[dqn_a],
            "same_action": int(ppo_a == dqn_a),
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "matched_state_observations.csv", index=False)

    agree = float(df["same_action"].mean())
    diverge = df[df["same_action"] == 0]
    examples = diverge.head(50).copy()
    examples.to_csv(OUT_DIR / "matched_state_examples.csv", index=False)
    pairs = (
        diverge.groupby(["ppo_action", "dqn_action"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    pairs.to_csv(OUT_DIR / "matched_state_divergence_pairs.csv", index=False)

    ppo_freq = df["ppo_action"].value_counts(normalize=True).reindex(list(ACTIONS), fill_value=0.0)
    dqn_freq = df["dqn_action"].value_counts(normalize=True).reindex(list(ACTIONS), fill_value=0.0)

    result = {
        "n_probes": len(df),
        "budget": budget,
        "seed": seed,
        "agreement_rate": agree,
        "disagreement_rate": 1.0 - agree,
        "n_disagreement": int(len(diverge)),
        "ppo_action_distribution": {k: float(v) for k, v in ppo_freq.items()},
        "dqn_action_distribution": {k: float(v) for k, v in dqn_freq.items()},
        "ppo_action_diversity": _shannon_entropy_from_freq(
            {k: float(v) for k, v in ppo_freq.items()}
        ),
        "dqn_action_diversity": _shannon_entropy_from_freq(
            {k: float(v) for k, v in dqn_freq.items()}
        ),
        "ppo_comfort_fraction": float(
            sum(ppo_freq.get(a, 0.0) for a in COMFORT_ACTIONS)
        ),
        "dqn_comfort_fraction": float(
            sum(dqn_freq.get(a, 0.0) for a in COMFORT_ACTIONS)
        ),
        "ppo_challenge_fraction": float(
            sum(ppo_freq.get(a, 0.0) for a in CHALLENGE_ACTIONS)
        ),
        "dqn_challenge_fraction": float(
            sum(dqn_freq.get(a, 0.0) for a in CHALLENGE_ACTIONS)
        ),
        "top_divergence_pairs": pairs.head(20).to_dict(orient="records"),
        "agents": {"ppo": ppo, "dqn": dqn, "probes": probes},
    }

    report_slice = {k: v for k, v in result.items() if k != "agents"}
    with open(OUT_DIR / "matched_state_report.json", "w") as f:
        json.dump(_json_safe(report_slice), f, indent=2)
    return result


def _rollout_future_k(
    env: StudentEnv,
    agent: Any,
    snap: EnvSnapshot,
    first_action: int,
    horizon: int,
) -> float:
    obs, info = _restore_snapshot(env, snap)
    k0 = float(env._state.knowledge)
    obs, _, term, trunc, info = env.step(first_action)
    done = bool(term or trunc)
    t = 1
    while (not done) and t < horizon:
        mask = info["action_masks"]
        a = int(agent.predict(obs, mask))
        obs, _, term, trunc, info = env.step(a)
        done = bool(term or trunc)
        t += 1
    return float(env._state.knowledge - k0)


def run_disagreement_counterfactual(matched: Dict[str, Any]) -> pd.DataFrame:
    """Future knowledge gain at 5/10/20 steps for disagreement states only."""
    print("\n[disagreement_cf] counterfactual rollouts on disagreement states")
    ppo = matched["agents"]["ppo"]
    dqn = matched["agents"]["dqn"]
    probes = matched["agents"]["probes"]

    env = StudentEnv(population_seed=PROBE_SEED)
    rows: List[Dict[str, Any]] = []
    n_disagree = 0

    for i, p in enumerate(probes):
        obs = p["obs"]
        mask = p["mask"]
        snap = p["snapshot"]
        ppo_a = int(ppo.predict(obs, mask))
        dqn_a = int(dqn.predict(obs, mask))
        if ppo_a == dqn_a:
            continue
        n_disagree += 1
        for h in DISAGREEMENT_HORIZONS:
            fk_ppo_action = _rollout_future_k(env, ppo, snap, ppo_a, h)
            fk_dqn_action = _rollout_future_k(env, dqn, snap, dqn_a, h)
            rows.append({
                "probe_id": i,
                "horizon": h,
                "ppo_action": ID_TO_ACTION[ppo_a],
                "dqn_action": ID_TO_ACTION[dqn_a],
                "future_k_gain_if_ppo_action": fk_ppo_action,
                "future_k_gain_if_dqn_action": fk_dqn_action,
                "dqn_action_advantage": fk_dqn_action - fk_ppo_action,
                "knowledge": p["knowledge"],
                "mismatch": p["difficulty"] - p["knowledge"],
            })

    env.close()
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "disagreement_future_knowledge.csv", index=False)

    summary = []
    if len(df) > 0:
        for h in DISAGREEMENT_HORIZONS:
            sub = df[df["horizon"] == h]
            summary.append({
                "horizon": h,
                "n_disagreement_states": int(sub["probe_id"].nunique()),
                "mean_future_k_ppo_action": float(sub["future_k_gain_if_ppo_action"].mean()),
                "mean_future_k_dqn_action": float(sub["future_k_gain_if_dqn_action"].mean()),
                "mean_dqn_advantage": float(sub["dqn_action_advantage"].mean()),
                "pct_dqn_better": float((sub["dqn_action_advantage"] > 0).mean()),
            })
    pd.DataFrame(summary).to_csv(OUT_DIR / "disagreement_future_knowledge_summary.csv", index=False)
    return df


def run_dqn_eval_modes(
    dqn_hp: Dict[str, Any],
    budgets: List[int],
    seeds: List[int],
    resume: bool = True,
) -> pd.DataFrame:
    path = OUT_DIR / "dqn_eval_modes_per_run.csv"
    done: set = set()
    rows: List[Dict[str, Any]] = []
    if resume and path.exists():
        prev = pd.read_csv(path)
        rows = prev.to_dict("records")
        done = {
            (r["eval_mode"], int(r["budget"]), int(r["seed"])) for r in rows
        }

    for budget in budgets:
        for seed in seeds:
            agent = train_one("DQN", seed, budget, {}, dqn_hp)
            for mode_name, kwargs in DQN_EVAL_MODES:
                key = (mode_name, budget, seed)
                if key in done:
                    continue
                print(f"\n[dqn_eval_modes] {mode_name} budget={budget} seed={seed}")
                eval_env = StudentEnv(population_seed=seed)
                eval_env.set_algorithm_name(f"DQN_{mode_name}")
                result = evaluate_agent(
                    agent, eval_env, EVAL_EPS, seed, "DQN", dqn_kwargs=kwargs
                )
                eval_env.close()
                row = {
                    "eval_mode": mode_name,
                    "budget": budget,
                    "seed": seed,
                    **{k: result[k] for k in [
                        "mean_episode_reward",
                        "success_rate",
                        "learning_gain",
                        "final_knowledge",
                        "adaptation_accuracy",
                        "comfort_action_fraction",
                        "challenge_action_fraction",
                        "action_diversity_shannon",
                    ]},
                }
                rows.append(row)
                pd.DataFrame(rows).to_csv(path, index=False)
                done.add(key)

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df

    summaries = []
    for (mode, budget), grp in df.groupby(["eval_mode", "budget"]):
        entry: Dict[str, Any] = {"eval_mode": mode, "budget": int(budget)}
        for m in [
            "mean_episode_reward",
            "success_rate",
            "learning_gain",
            "final_knowledge",
            "adaptation_accuracy",
        ]:
            s = _mean_std(grp[m].astype(float).tolist())
            entry[f"{m}_mean"] = s["mean"]
            entry[f"{m}_std"] = s["std"]
        summaries.append(entry)
    pd.DataFrame(summaries).to_csv(OUT_DIR / "dqn_eval_modes_summary.csv", index=False)
    return df


def diagnose_superiority(
    train_eval_df: pd.DataFrame,
    matched: Dict[str, Any],
    disagreement_summary: pd.DataFrame,
    dqn_modes_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Classify whether DQN superiority is mastery, challenge-seeking, etc."""
    ref_budget = max(TRAIN_BUDGETS)
    ref = train_eval_df[train_eval_df["budget"] == ref_budget]
    ppo_ref = ref[ref["algorithm"] == "PPO"]
    dqn_ref = ref[ref["algorithm"] == "DQN"]

    gaps = {}
    for m in [
        "mean_episode_reward",
        "success_rate",
        "learning_gain",
        "final_knowledge",
        "adaptation_accuracy",
        "comfort_action_fraction",
        "challenge_action_fraction",
        "p_correct",
    ]:
        ppo_m = float(ppo_ref[m].mean()) if len(ppo_ref) else 0.0
        dqn_m = float(dqn_ref[m].mean()) if len(dqn_ref) else 0.0
        gaps[m] = dqn_m - ppo_m

    dqn_wins_mastery = gaps.get("final_knowledge", 0) > 0.01
    dqn_wins_reward = gaps.get("mean_episode_reward", 0) > 0.01
    dqn_more_challenge = gaps.get("challenge_action_fraction", 0) > 0.05
    dqn_less_comfort = gaps.get("comfort_action_fraction", 0) < -0.05
    dqn_wins_adapt = gaps.get("adaptation_accuracy", 0) > 0.01

    cf_advantage = 0.0
    if len(disagreement_summary) > 0:
        h20 = disagreement_summary[disagreement_summary["horizon"] == 20]
        if len(h20) > 0:
            cf_advantage = float(h20.iloc[0]["mean_dqn_advantage"])

    exploration_advantage = False
    if len(dqn_modes_df) > 0:
        ref_modes = dqn_modes_df[dqn_modes_df["budget"] == ref_budget]
        greedy = ref_modes[ref_modes["eval_mode"] == "greedy"]
        eps = ref_modes[ref_modes["eval_mode"] == "epsilon_0.05"]
        if len(greedy) and len(eps):
            exploration_advantage = (
                float(eps["final_knowledge"].mean()) > float(greedy["final_knowledge"].mean()) + 0.01
            )

    hypotheses: List[Dict[str, Any]] = []

    if dqn_wins_mastery and dqn_wins_adapt and dqn_more_challenge:
        hypotheses.append({
            "mechanism": "genuine_mastery_optimization",
            "confidence": "high" if cf_advantage > 0 else "medium",
            "evidence": {
                "final_knowledge_gap": gaps["final_knowledge"],
                "adaptation_gap": gaps["adaptation_accuracy"],
                "challenge_gap": gaps["challenge_action_fraction"],
                "disagreement_h20_dqn_advantage": cf_advantage,
            },
        })

    if dqn_more_challenge and not dqn_wins_mastery:
        hypotheses.append({
            "mechanism": "challenge_seeking_behavior",
            "confidence": "medium",
            "evidence": {
                "challenge_gap": gaps["challenge_action_fraction"],
                "comfort_gap": gaps["comfort_action_fraction"],
                "matched_dqn_challenge": matched.get("dqn_challenge_fraction"),
            },
        })

    if dqn_wins_reward and not dqn_wins_mastery:
        hypotheses.append({
            "mechanism": "reward_exploitation",
            "confidence": "medium",
            "evidence": {"reward_gap": gaps["mean_episode_reward"], "mastery_gap": gaps["final_knowledge"]},
        })

    if dqn_wins_reward and gaps.get("p_correct", 0) < -0.02:
        hypotheses.append({
            "mechanism": "simulator_exploitation",
            "confidence": "low",
            "evidence": gaps,
        })

    if exploration_advantage:
        hypotheses.append({
            "mechanism": "exploration_advantage",
            "confidence": "medium",
            "evidence": "epsilon eval outperforms greedy at same checkpoint",
        })

    if not hypotheses:
        if gaps.get("final_knowledge", 0) <= 0:
            primary = "no_clear_dqn_superiority"
        else:
            primary = "mixed_mechanisms"
    else:
        primary = hypotheses[0]["mechanism"]

    diagnosis = {
        "reference_budget": ref_budget,
        "performance_gaps_dqn_minus_ppo": gaps,
        "matched_state": {
            k: matched[k]
            for k in matched
            if k not in ("agents",)
        },
        "primary_mechanism": primary,
        "hypotheses": hypotheses,
        "exploration_advantage_detected": exploration_advantage,
    }
    with open(OUT_DIR / "superiority_diagnosis.json", "w") as f:
        json.dump(_json_safe(diagnosis), f, indent=2)
    return diagnosis


def compute_final_ranking(train_eval_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Rank PPO vs DQN on mastery, educational validity, adaptation, robustness."""
    ref_budget = max(TRAIN_BUDGETS)
    ref = train_eval_df[train_eval_df["budget"] == ref_budget]

    def _robustness(algo: str) -> float:
        sub = ref[ref["algorithm"] == algo]
        if len(sub) < 2:
            return 0.0
        fk_std = float(sub["final_knowledge"].std(ddof=1))
        reward_std = float(sub["mean_episode_reward"].std(ddof=1))
        return float(1.0 / (1.0 + fk_std + reward_std))

    def _budget_generalization(algo: str) -> float:
        sub = train_eval_df[train_eval_df["algorithm"] == algo]
        if len(sub) == 0:
            return 0.0
        by_budget = sub.groupby("budget")["final_knowledge"].mean()
        return float(by_budget.mean() - 0.25 * by_budget.std(ddof=0))

    scores: Dict[str, Dict[str, float]] = {}
    for algo in ["PPO", "DQN"]:
        sub = ref[ref["algorithm"] == algo]
        scores[algo] = {
            "mastery": float(sub["final_knowledge"].mean()),
            "educational_validity": float(
                0.4 * sub["learning_gain"].mean()
                + 0.3 * sub["challenge_action_fraction"].mean()
                + 0.3 * (1.0 - sub["comfort_action_fraction"].mean())
            ),
            "emotional_adaptation": float(sub["adaptation_accuracy"].mean()),
            "robustness": _robustness(algo),
            "budget_generalization": _budget_generalization(algo),
            "mean_reward": float(sub["mean_episode_reward"].mean()),
            "success_rate": float(sub["success_rate"].mean()),
        }
        z_vals = [
            scores[algo]["mastery"],
            scores[algo]["educational_validity"],
            scores[algo]["emotional_adaptation"],
            scores[algo]["robustness"],
            scores[algo]["budget_generalization"],
        ]
        scores[algo]["composite"] = float(np.mean(z_vals))

    ranking = []
    for rank, algo in enumerate(
        sorted(scores.keys(), key=lambda a: scores[a]["composite"], reverse=True),
        start=1,
    ):
        ranking.append({"rank": rank, "algorithm": algo, **scores[algo]})

    with open(OUT_DIR / "final_algorithm_ranking.json", "w") as f:
        json.dump(_json_safe(ranking), f, indent=2)
    pd.DataFrame(ranking).to_csv(OUT_DIR / "final_algorithm_ranking.csv", index=False)
    return ranking


def compute_thesis_verdict(
    train_eval_df: pd.DataFrame,
    ranking: List[Dict[str, Any]],
    diagnosis: Dict[str, Any],
) -> Dict[str, Any]:
    """Answer: A) PPO, B) DQN, or C) no significant difference."""
    ref_budget = max(TRAIN_BUDGETS)
    sig_path = OUT_DIR / "train_eval_significance.csv"
    mastery_wins = {"PPO": 0, "DQN": 0, "ties": 0}
    sig_metrics = 0

    if sig_path.exists():
        sig = pd.read_csv(sig_path)
        ref_sig = sig[sig["budget"] == ref_budget]
        for _, row in ref_sig.iterrows():
            if not bool(row.get("significant_0.05", False)):
                mastery_wins["ties"] += 1
                continue
            sig_metrics += 1
            if float(row["delta_dqn_minus_ppo"]) > 0:
                mastery_wins["DQN"] += 1
            else:
                mastery_wins["PPO"] += 1

    top = ranking[0]["algorithm"] if ranking else "unknown"
    gaps = diagnosis.get("performance_gaps_dqn_minus_ppo", {})
    fk_gap = float(gaps.get("final_knowledge", 0.0))

    if sig_metrics >= 2 and mastery_wins["DQN"] >= 2 and fk_gap > 0.02:
        answer = "B"
        label = "DQN"
    elif sig_metrics >= 2 and mastery_wins["PPO"] >= 2 and fk_gap < -0.02:
        answer = "A"
        label = "PPO"
    elif abs(fk_gap) <= 0.02 and mastery_wins["ties"] >= mastery_wins["DQN"]:
        answer = "C"
        label = "No significant difference"
    else:
        answer = "B" if fk_gap > 0 and top == "DQN" else (
            "A" if fk_gap < 0 and top == "PPO" else "C"
        )
        label = {"A": "PPO", "B": "DQN", "C": "No significant difference"}[answer]

    verdict = {
        "thesis_answer": answer,
        "thesis_label": label,
        "top_composite_rank": top,
        "final_knowledge_gap_dqn_minus_ppo": fk_gap,
        "significant_metric_wins": mastery_wins,
        "primary_mechanism": diagnosis.get("primary_mechanism"),
        "evidence_summary": (
            f"At {ref_budget:,} timesteps, DQN-PPO final knowledge gap = {fk_gap:+.4f}; "
            f"composite rank leader = {top}."
        ),
    }
    with open(OUT_DIR / "thesis_verdict.json", "w") as f:
        json.dump(_json_safe(verdict), f, indent=2)
    return verdict


def build_report(
    ppo_hp: Dict[str, Any],
    dqn_hp: Dict[str, Any],
    ranking: List[Dict[str, Any]],
    diagnosis: Dict[str, Any],
    verdict: Optional[Dict[str, Any]] = None,
) -> None:
    lines = [
        "# Fair PPO vs DQN Comparison Report\n\n",
        "Environment, reward function, state representation, action definitions, "
        "masking logic, and simulator dynamics were **not modified**.\n\n",
        "## Hyperparameters\n\n",
        "### Tuned PPO\n```json\n",
        json.dumps(ppo_hp, indent=2),
        "\n```\n\n### Tuned DQN\n```json\n",
        json.dumps(dqn_hp, indent=2),
        "\n```\n\n",
        f"## Protocol\n- Seeds: {FAIR_SEEDS}\n",
        f"- Budgets: {TRAIN_BUDGETS}\n",
        f"- Eval episodes: {EVAL_EPS}\n",
        f"- Matched observations: {N_PROBE}+\n\n",
    ]

    summary_path = OUT_DIR / "train_eval_summary.csv"
    if summary_path.exists():
        sdf = pd.read_csv(summary_path)
        lines.append("## Performance (mean +/- std across seeds)\n\n")
        for budget in sorted(sdf["budget"].unique()):
            lines.append(f"### Budget {int(budget):,} timesteps\n\n")
            lines.append("| Algorithm | Reward | Success | Learning Gain | Final Knowledge | Adaptation |\n")
            lines.append("|-----------|--------|---------|---------------|-----------------|------------|\n")
            sub = sdf[sdf["budget"] == budget]
            for algo in ["PPO", "DQN"]:
                row = sub[sub["algorithm"] == algo]
                if len(row) == 0:
                    continue
                r = row.iloc[0]
                lines.append(
                    f"| {algo} | {r['mean_episode_reward_mean']:.3f}+/-{r['mean_episode_reward_std']:.3f} "
                    f"| {r['success_rate_mean']:.3f}+/-{r['success_rate_std']:.3f} "
                    f"| {r['learning_gain_mean']:.3f}+/-{r['learning_gain_std']:.3f} "
                    f"| {r['final_knowledge_mean']:.3f}+/-{r['final_knowledge_std']:.3f} "
                    f"| {r['adaptation_accuracy_mean']:.3f}+/-{r['adaptation_accuracy_std']:.3f} |\n"
                )
            lines.append("\n")

    lines.append("## Final Algorithm Ranking\n\n")
    for entry in ranking:
        lines.append(
            f"{entry['rank']}. **{entry['algorithm']}** "
            f"(composite={entry['composite']:.3f}, mastery={entry['mastery']:.3f}, "
            f"adaptation={entry['emotional_adaptation']:.3f}, robustness={entry['robustness']:.3f})\n"
        )

    lines.append(f"\n## DQN Superiority Diagnosis\n\nPrimary mechanism: **{diagnosis.get('primary_mechanism')}**\n\n")
    for h in diagnosis.get("hypotheses", []):
        lines.append(f"- {h['mechanism']} ({h['confidence']} confidence)\n")

    if verdict:
        lines.append("\n## Final Thesis Question\n\n")
        lines.append(
            f"**Answer ({verdict['thesis_answer']}): {verdict['thesis_label']}**\n\n"
        )
        lines.append(f"{verdict['evidence_summary']}\n")

    (OUT_DIR / "FAIR_COMPARISON_REPORT.md").write_text("".join(lines))
    print(f"Wrote {OUT_DIR / 'FAIR_COMPARISON_REPORT.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fair PPO vs DQN comparison")
    parser.add_argument(
        "--phase",
        choices=[
            "train_eval",
            "matched_state",
            "disagreement_cf",
            "dqn_eval_modes",
            "diagnosis",
            "ranking",
            "report",
            "all",
        ],
        default="all",
    )
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--budget", type=int, default=None, help="Single budget filter")
    parser.add_argument("--ppo-hp-json", type=str, default=None)
    parser.add_argument("--dqn-hp-json", type=str, default=None)
    parser.add_argument("--max-probes", type=int, default=10_000)
    args = parser.parse_args()

    global N_PROBE
    N_PROBE = args.max_probes

    ppo_hp_path = Path(args.ppo_hp_json) if args.ppo_hp_json else PPO_HP_PATH
    dqn_hp_path = Path(args.dqn_hp_json) if args.dqn_hp_json else DQN_HP_PATH

    from RL_Module.ppo_hyperparameter_study import BASELINE_HP as PPO_BASELINE
    from RL_Module.dqn_hyperparameter_study import BASELINE_HP as DQN_BASELINE

    ppo_hp = _load_hp(ppo_hp_path, PPO_BASELINE)
    dqn_hp = _load_hp(dqn_hp_path, DQN_BASELINE)

    print("Fair PPO vs DQN comparison")
    print(f"  PPO HP: {ppo_hp}")
    print(f"  DQN HP: {dqn_hp}")

    train_eval_df = pd.DataFrame()
    matched: Dict[str, Any] = {}
    disagreement_df = pd.DataFrame()
    dqn_modes_df = pd.DataFrame()
    diagnosis: Dict[str, Any] = {}
    ranking: List[Dict[str, Any]] = []
    verdict: Dict[str, Any] = {}

    te_path = OUT_DIR / "train_eval_per_run.csv"
    if args.phase in ("train_eval", "all"):
        train_eval_df = run_train_eval(
            ppo_hp, dqn_hp, TRAIN_BUDGETS, FAIR_SEEDS,
            resume=args.resume, budget_filter=args.budget,
        )
    elif te_path.exists():
        train_eval_df = pd.read_csv(te_path)

    if args.phase in ("matched_state", "disagreement_cf", "diagnosis", "all"):
        ref_budget = args.budget or max(TRAIN_BUDGETS)
        matched = run_matched_state(ppo_hp, dqn_hp, budget=ref_budget, seed=PROBE_SEED)

    if args.phase in ("disagreement_cf", "diagnosis", "all"):
        if not matched:
            ref_budget = args.budget or max(TRAIN_BUDGETS)
            matched = run_matched_state(ppo_hp, dqn_hp, budget=ref_budget, seed=PROBE_SEED)
        disagreement_df = run_disagreement_counterfactual(matched)
        matched.pop("agents", None)

    if args.phase in ("dqn_eval_modes", "diagnosis", "all"):
        dqn_modes_df = run_dqn_eval_modes(
            dqn_hp, TRAIN_BUDGETS, FAIR_SEEDS, resume=args.resume
        )

    disagree_summary_path = OUT_DIR / "disagreement_future_knowledge_summary.csv"
    disagree_summary = (
        pd.read_csv(disagree_summary_path)
        if disagree_summary_path.exists()
        else pd.DataFrame()
    )

    if args.phase in ("diagnosis", "ranking", "report", "all"):
        if len(train_eval_df) == 0 and te_path.exists():
            train_eval_df = pd.read_csv(te_path)
        if not matched:
            mpath = OUT_DIR / "matched_state_report.json"
            if mpath.exists():
                with open(mpath) as f:
                    matched = json.load(f)
        diagnosis = diagnose_superiority(
            train_eval_df, matched, disagree_summary, dqn_modes_df
        )

    if args.phase in ("ranking", "report", "all"):
        if len(train_eval_df) == 0 and te_path.exists():
            train_eval_df = pd.read_csv(te_path)
        ranking = compute_final_ranking(train_eval_df)
        if len(train_eval_df) > 0 and diagnosis:
            verdict = compute_thesis_verdict(train_eval_df, ranking, diagnosis)

    if args.phase in ("report", "all"):
        if not diagnosis:
            dpath = OUT_DIR / "superiority_diagnosis.json"
            if dpath.exists():
                with open(dpath) as f:
                    diagnosis = json.load(f)
        if not ranking:
            rpath = OUT_DIR / "final_algorithm_ranking.json"
            if rpath.exists():
                with open(rpath) as f:
                    ranking = json.load(f)
        if not verdict:
            vpath = OUT_DIR / "thesis_verdict.json"
            if vpath.exists():
                with open(vpath) as f:
                    verdict = json.load(f)
        if len(train_eval_df) > 0 and diagnosis and not verdict:
            verdict = compute_thesis_verdict(train_eval_df, ranking, diagnosis)
        build_report(ppo_hp, dqn_hp, ranking, diagnosis, verdict)


if __name__ == "__main__":
    main()
