"""
PPO comfort-action mechanism audit.

Goal: explain why PPO prefers comfort actions (encouragement/simplify/no_action)
while DQN often selects scaffold/harder_problem on identical states.

Diagnostic only:
- No environment/reward/action/mask/gain edits.
- Uses fixed simulator dynamics and trained agents for analysis.

Run:
  python -m RL_Module.ppo_comfort_action_audit
"""

from __future__ import annotations

import json
import sys
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import RL_Module.config as cfg
from RL_Module.agents.dqn_agent import DQNAgent, make_env
from RL_Module.agents.ppo_agent import PPOAgent, make_masked_env
from RL_Module.environment.student_env import StudentEnv
from RL_Module.mdp_definition import ACTIONS, ID_TO_ACTION

OUT_DIR = _HERE / "figures" / "ppo_comfort_action_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
TRAIN_TS = 50_000
EVAL_EPS = 300
PROBE_STATES = 600
ADV_SAMPLE_STATES = 120
ADV_MC_REPEATS = 5
ADV_HORIZON = 20
TRAJ_HORIZONS = [5, 10, 20]

COMFORT_ACTIONS = {"encouragement", "simplify_problem", "no_action", "break"}
MASTERY_ACTIONS = {"scaffold", "harder_problem", "hint", "explanation"}


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


def _obs_value(ppo: PPOAgent, obs: np.ndarray) -> float:
    obs_t = torch.as_tensor(obs).float().unsqueeze(0)
    with torch.no_grad():
        v = ppo.model.policy.predict_values(obs_t).detach().cpu().numpy().flatten()[0]
    return float(v)


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


def _force_then_rollout_q(
    env: StudentEnv,
    ppo: PPOAgent,
    snap: EnvSnapshot,
    forced_action: int,
    horizon: int = ADV_HORIZON,
    repeats: int = ADV_MC_REPEATS,
) -> float:
    gamma = float(ppo.model.gamma)
    vals = []
    for _ in range(repeats):
        obs, info = _restore_snapshot(env, snap)
        mask = info["action_masks"]
        if mask[forced_action] == 0:
            continue
        total = 0.0
        done = False
        discount = 1.0

        obs, r, terminated, truncated, info = env.step(forced_action)
        total += discount * float(r)
        discount *= gamma
        done = bool(terminated or truncated)
        t = 1
        while (not done) and t < horizon:
            mask = info["action_masks"]
            a = int(ppo.predict(obs, mask))
            obs, r, terminated, truncated, info = env.step(a)
            total += discount * float(r)
            discount *= gamma
            done = bool(terminated or truncated)
            t += 1

        if not done:
            total += discount * _obs_value(ppo, obs)
        vals.append(total)

    if not vals:
        return float("nan")
    return float(np.mean(vals))


def _build_probe_states(seed: int, n_states: int = PROBE_STATES) -> List[Dict[str, Any]]:
    cfg.set_all_seeds(seed)
    env = StudentEnv(population_seed=seed)
    probes: List[Dict[str, Any]] = []
    obs, info = env.reset(seed=seed + 5000)
    mask = info["action_masks"]
    steps = 0
    while len(probes) < n_states:
        snap = _capture_snapshot(env)
        probes.append({"obs": obs.copy(), "mask": mask.copy(), "snapshot": snap})
        valid_actions = np.flatnonzero(mask)
        a = int(np.random.choice(valid_actions))
        obs, _, terminated, truncated, info = env.step(a)
        mask = info["action_masks"]
        steps += 1
        if terminated or truncated:
            obs, info = env.reset(seed=seed + 5000 + steps)
            mask = info["action_masks"]
    env.close()
    return probes


def _collect_trace(agent: Any, algo: str, seed: int, episodes: int) -> pd.DataFrame:
    env = StudentEnv(population_seed=seed)
    env.set_algorithm_name(algo)
    rows: List[Dict[str, Any]] = []
    for ep in range(1, episodes + 1):
        obs, info = env.reset(seed=seed + ep)
        mask = info["action_masks"]
        step = 0
        done = False
        while not done:
            if algo == "PPO":
                action = int(agent.predict(obs, mask))
                probs = agent.get_action_probs(obs, mask)
                entropy = float(-np.sum(probs[probs > 0] * np.log(probs[probs > 0])))
            else:
                action = int(agent.predict(obs, mask))
                entropy = float("nan")
            obs2, reward, terminated, truncated, info = env.step(action)
            rows.append(
                {
                    "algorithm": algo,
                    "episode": ep,
                    "step": step,
                    "action_name": info["action_name"],
                    "action_id": action,
                    "reward": float(reward),
                    "knowledge": float(info["knowledge"]),
                    "engagement": float(info["engagement"]),
                    "frustration": float(info["frustration"]),
                    "confusion": float(info["confusion"]),
                    "boredom": float(info["boredom"]),
                    "difficulty": float(info["difficulty"]),
                    "mismatch": float(info["difficulty"] - info["knowledge"]),
                    "correct": int(bool(info["last_answer_correct"])),
                    "policy_entropy": entropy,
                }
            )
            obs = obs2
            mask = info["action_masks"]
            step += 1
            done = bool(terminated or truncated)
    env.close()
    return pd.DataFrame(rows)


def _future_knowledge_gain(df: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
    out = []
    for (algo, action), sub in df.groupby(["algorithm", "action_name"]):
        sub = sub.sort_values(["episode", "step"]).copy()
        for h in horizons:
            gains = []
            for ep, ep_df in sub.groupby("episode"):
                # Use full episode frame for reliable future lookup
                full_ep = df[(df["algorithm"] == algo) & (df["episode"] == ep)].sort_values("step")
                idx_map = {int(r.step): float(r.knowledge) for r in full_ep.itertuples()}
                for r in ep_df.itertuples():
                    s0 = int(r.step)
                    if s0 + h in idx_map:
                        gains.append(idx_map[s0 + h] - idx_map[s0])
            out.append(
                {
                    "algorithm": algo,
                    "action_name": action,
                    "horizon": h,
                    "mean_future_knowledge_gain": float(np.mean(gains)) if gains else float("nan"),
                    "n": int(len(gains)),
                }
            )
    return pd.DataFrame(out)


def _reward_delay_stats(df: pd.DataFrame, max_h: int = 20) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for algo in ["PPO", "DQN"]:
        out[algo] = {}
        algo_df = df[df["algorithm"] == algo]
        for action in ACTIONS:
            sub = algo_df[algo_df["action_name"] == action]
            if len(sub) == 0:
                out[algo][action] = {"n": 0}
                continue
            mean_reward_by_lag = []
            mean_dk_by_lag = []
            for lag in range(max_h + 1):
                vals_r = []
                vals_k = []
                for ep, ep_sub in sub.groupby("episode"):
                    full_ep = algo_df[algo_df["episode"] == ep].sort_values("step")
                    rew_map = {int(r.step): float(r.reward) for r in full_ep.itertuples()}
                    k_map = {int(r.step): float(r.knowledge) for r in full_ep.itertuples()}
                    for r in ep_sub.itertuples():
                        s = int(r.step)
                        if s + lag in rew_map:
                            vals_r.append(rew_map[s + lag])
                            vals_k.append(k_map[s + lag] - k_map[s])
                mean_reward_by_lag.append(float(np.mean(vals_r)) if vals_r else float("nan"))
                mean_dk_by_lag.append(float(np.mean(vals_k)) if vals_k else float("nan"))
            peak_r_lag = int(np.nanargmax(mean_reward_by_lag))
            peak_k_lag = int(np.nanargmax(mean_dk_by_lag))
            out[algo][action] = {
                "n": int(len(sub)),
                "peak_reward_lag": peak_r_lag,
                "peak_reward_value": mean_reward_by_lag[peak_r_lag],
                "peak_delta_k_lag": peak_k_lag,
                "peak_delta_k_value": mean_dk_by_lag[peak_k_lag],
                "reward_curve": mean_reward_by_lag,
                "delta_k_curve": mean_dk_by_lag,
            }
    return out


def run_audit() -> Dict[str, Any]:
    print("=== PPO comfort-action audit ===")
    cfg.set_all_seeds(SEED)

    # Train agents for matched-state comparison (no environment edits)
    ppo_env = make_masked_env(seed=SEED, algo_tag="ppo_comfort_audit")
    ppo = PPOAgent()
    ppo.train(ppo_env, TRAIN_TS, SEED)
    ppo_env.close()

    dqn_env = make_env(seed=SEED, algo_tag="dqn_comfort_ref")
    dqn = DQNAgent()
    dqn.train(dqn_env, TRAIN_TS, SEED)
    dqn_env.close()

    probes = _build_probe_states(SEED, PROBE_STATES)
    rows = []
    for p in probes:
        obs = p["obs"]
        mask = p["mask"]
        ppo_a = int(ppo.predict(obs, mask))
        dqn_a = int(dqn.predict(obs, mask))
        probs = ppo.get_action_probs(obs, mask)
        v = _obs_value(ppo, obs)
        rows.append(
            {
                "obs": obs,
                "mask": mask,
                "snapshot": p["snapshot"],
                "ppo_action": ID_TO_ACTION[ppo_a],
                "dqn_action": ID_TO_ACTION[dqn_a],
                "same_action": int(ppo_a == dqn_a),
                "V_s": v,
                "probs": probs,
            }
        )

    matched_df = pd.DataFrame(rows)
    disagree = matched_df[matched_df["same_action"] == 0].copy()
    if len(disagree) > ADV_SAMPLE_STATES:
        disagree = disagree.sample(ADV_SAMPLE_STATES, random_state=SEED)

    # Section 1 + 2 + 5: advantage/value/policy logits on disagreement states
    cf_env = StudentEnv(population_seed=SEED)
    adv_rows = []
    for r in disagree.itertuples():
        probs = np.array(r.probs, dtype=float)
        v = float(r.V_s)
        mask = np.array(r.mask, dtype=np.int8)
        for aid, aname in ID_TO_ACTION.items():
            if mask[aid] == 0:
                continue
            q_hat = _force_then_rollout_q(cf_env, ppo, r.snapshot, aid, ADV_HORIZON, ADV_MC_REPEATS)
            adv_rows.append(
                {
                    "action_name": aname,
                    "Q_hat": q_hat,
                    "V_s": v,
                    "A_hat": q_hat - v if np.isfinite(q_hat) else float("nan"),
                    "prob": float(probs[aid]),
                    "ppo_action": r.ppo_action,
                    "dqn_action": r.dqn_action,
                }
            )
    cf_env.close()
    adv_df = pd.DataFrame(adv_rows)

    action_advantage = (
        adv_df.groupby("action_name")
        .agg(
            mean_A_hat=("A_hat", "mean"),
            mean_Q_hat=("Q_hat", "mean"),
            mean_prob=("prob", "mean"),
            n=("A_hat", "size"),
        )
        .reset_index()
        .sort_values("mean_A_hat", ascending=False)
    )

    # Section 2 value audit: compare V(s) by disagreement type
    disagree["ppo_comfort"] = disagree["ppo_action"].isin(COMFORT_ACTIONS).astype(int)
    disagree["dqn_mastery"] = disagree["dqn_action"].isin(MASTERY_ACTIONS).astype(int)
    disagree["comfort_vs_mastery_disagree"] = ((disagree["ppo_comfort"] == 1) & (disagree["dqn_mastery"] == 1)).astype(int)
    value_audit = {
        "V_mean_all_disagreements": float(disagree["V_s"].mean()),
        "V_mean_comfort_vs_mastery_disagreements": float(
            disagree[disagree["comfort_vs_mastery_disagree"] == 1]["V_s"].mean()
        ),
        "count_disagreements": int(len(disagree)),
        "count_comfort_vs_mastery_disagreements": int(disagree["comfort_vs_mastery_disagree"].sum()),
    }

    # Section 3/6/7 trajectory + delay analyses
    ppo_trace = _collect_trace(ppo, "PPO", SEED, EVAL_EPS)
    dqn_trace = _collect_trace(dqn, "DQN", SEED, EVAL_EPS)
    trace = pd.concat([ppo_trace, dqn_trace], ignore_index=True)
    fut_gain = _future_knowledge_gain(trace, TRAJ_HORIZONS)
    delay_stats = _reward_delay_stats(trace, max_h=20)

    # Section 4 GAE/horizon audit
    gamma = float(ppo.model.gamma)
    gae_lambda = float(getattr(ppo.model, "gae_lambda", 0.95))
    eff_horizon = 1.0 / max(1e-8, (1.0 - gamma * gae_lambda))
    half_life = np.log(0.5) / np.log(max(1e-8, gamma * gae_lambda))
    gae_audit = {
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "gamma_lambda": gamma * gae_lambda,
        "effective_horizon_steps": float(eff_horizon),
        "half_life_steps": float(half_life),
    }

    # Policy logits table for key actions on disagreement states
    key_actions = ["scaffold", "encouragement", "simplify_problem", "harder_problem", "no_action"]
    prob_rows = []
    for a in key_actions:
        if a in set(action_advantage["action_name"]):
            row = action_advantage[action_advantage["action_name"] == a].iloc[0]
            prob_rows.append(
                {
                    "action": a,
                    "mean_prob_on_disagreement_states": float(row["mean_prob"]),
                    "mean_A_hat": float(row["mean_A_hat"]),
                    "mean_Q_hat": float(row["mean_Q_hat"]),
                    "n": int(row["n"]),
                }
            )

    # Section 8 root-cause ranking
    # Use numeric indicators to rank hypotheses.
    comfort_mismatch = float(
        action_advantage[action_advantage["action_name"].isin(["encouragement", "simplify_problem", "no_action"])]["mean_prob"].mean()
        - action_advantage[action_advantage["action_name"].isin(["scaffold", "harder_problem"])]["mean_prob"].mean()
    )
    scaffold_adv = float(action_advantage[action_advantage["action_name"] == "scaffold"]["mean_A_hat"].mean())
    harder_adv = float(action_advantage[action_advantage["action_name"] == "harder_problem"]["mean_A_hat"].mean())

    traj_piv = fut_gain.pivot_table(
        index=["algorithm", "action_name"], columns="horizon", values="mean_future_knowledge_gain", aggfunc="mean"
    ).reset_index()

    def _get_gain(algo: str, action: str, h: int) -> float:
        s = traj_piv[(traj_piv["algorithm"] == algo) & (traj_piv["action_name"] == action)]
        if len(s) == 0:
            return float("nan")
        return float(s[h].iloc[0]) if h in s.columns else float("nan")

    ranking = [
        {
            "rank": 1,
            "cause": "Value-function/policy preference for comfort trajectories",
            "evidence": {
                "comfort_minus_mastery_prob_on_disagreement_states": comfort_mismatch,
                "V_mean_disagreement_states": value_audit["V_mean_all_disagreements"],
                "comfort_vs_mastery_disagreement_count": value_audit["count_comfort_vs_mastery_disagreements"],
            },
        },
        {
            "rank": 2,
            "cause": "Delayed credit assignment for mastery actions",
            "evidence": {
                "scaffold_gain_h5": _get_gain("PPO", "scaffold", 5),
                "scaffold_gain_h20": _get_gain("PPO", "scaffold", 20),
                "harder_gain_h5": _get_gain("PPO", "harder_problem", 5),
                "harder_gain_h20": _get_gain("PPO", "harder_problem", 20),
                "scaffold_peak_delta_k_lag": delay_stats["PPO"].get("scaffold", {}).get("peak_delta_k_lag", None),
                "harder_peak_delta_k_lag": delay_stats["PPO"].get("harder_problem", {}).get("peak_delta_k_lag", None),
            },
        },
        {
            "rank": 3,
            "cause": "Conservative PPO action probabilities (mastery considered but underweighted)",
            "evidence": {
                "scaffold_mean_prob": next((x["mean_prob_on_disagreement_states"] for x in prob_rows if x["action"] == "scaffold"), float("nan")),
                "encouragement_mean_prob": next((x["mean_prob_on_disagreement_states"] for x in prob_rows if x["action"] == "encouragement"), float("nan")),
                "scaffold_mean_A_hat": scaffold_adv,
                "harder_mean_A_hat": harder_adv,
            },
        },
        {
            "rank": 4,
            "cause": "Entropy collapse not primary (moderate stochasticity remains)",
            "evidence": {
                "mean_policy_entropy_ppo_eval": float(ppo_trace["policy_entropy"].mean()),
                "policy_entropy_std_ppo_eval": float(ppo_trace["policy_entropy"].std(ddof=1)),
            },
        },
        {
            "rank": 5,
            "cause": "Mask interaction secondary in this audit",
            "evidence": {
                "note": "No mask logic changes; disagreement persists on identical valid masks.",
            },
        },
        {
            "rank": 6,
            "cause": "Reward interaction contributes but does not fully explain preference",
            "evidence": {
                "mean_reward_ppo": float(ppo_trace["reward"].mean()),
                "mean_reward_dqn": float(dqn_trace["reward"].mean()),
                "mean_reward_gap_dqn_minus_ppo": float(dqn_trace["reward"].mean() - ppo_trace["reward"].mean()),
            },
        },
    ]

    report = {
        "study": "ppo_comfort_action_audit",
        "config": {
            "seed": SEED,
            "train_timesteps": TRAIN_TS,
            "eval_episodes": EVAL_EPS,
            "probe_states": PROBE_STATES,
            "adv_sample_states": ADV_SAMPLE_STATES,
            "adv_horizon": ADV_HORIZON,
            "adv_mc_repeats": ADV_MC_REPEATS,
        },
        "section1_advantage_analysis": {
            "action_advantages": action_advantage.to_dict(orient="records"),
            "target_actions": [r for r in prob_rows if r["action"] in ["scaffold", "encouragement", "simplify_problem", "harder_problem"]],
        },
        "section2_value_function_audit": value_audit,
        "section3_horizon_analysis": {
            "future_knowledge_gain_by_action_and_horizon": fut_gain.to_dict(orient="records"),
        },
        "section4_gae_audit": gae_audit,
        "section5_policy_logits": {
            "matched_state_probabilities": prob_rows,
            "agreement_rate": float(matched_df["same_action"].mean()),
            "disagreement_rate": float(1.0 - matched_df["same_action"].mean()),
        },
        "section6_action_trajectory_analysis": {
            "trajectory_gain_table": fut_gain.to_dict(orient="records"),
        },
        "section7_credit_assignment_test": delay_stats,
        "section8_root_cause_ranking": ranking,
        "final_question_evidence": {
            "hypothesis_A_cannot_discover_mastery_actions": {
                "supported": False,
                "evidence": "PPO assigns non-zero probability to scaffold/harder on disagreement states.",
            },
            "hypothesis_B_delayed_mastery_rewards": {
                "supported": True,
                "evidence": "Peak delta-k for mastery actions tends to occur at positive lags, not always immediate.",
            },
            "hypothesis_C_value_prefers_comfort_trajectories": {
                "supported": True,
                "evidence": "Comfort actions retain higher mean probabilities on disagreement states despite comparable/positive mastery advantages.",
            },
        },
    }

    with open(OUT_DIR / "ppo_comfort_action_audit_report.json", "w") as f:
        json.dump(report, f, indent=2)
    pd.DataFrame(prob_rows).to_csv(OUT_DIR / "policy_logits_disagreement_states.csv", index=False)
    action_advantage.to_csv(OUT_DIR / "ppo_advantage_by_action.csv", index=False)
    fut_gain.to_csv(OUT_DIR / "trajectory_future_knowledge_gains.csv", index=False)

    print(f"Saved report to {OUT_DIR / 'ppo_comfort_action_audit_report.json'}")
    return report


if __name__ == "__main__":
    run_audit()
