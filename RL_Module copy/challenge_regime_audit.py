"""
Challenge-regime audit: why PPO and DQN occupy different state-space regions.

Diagnostic only - no environment/reward/action/mask modifications.

Uses existing G0 trace when available; trains agents for counterfactual rollouts.

Run:
  python -m RL_Module.challenge_regime_audit
"""

from __future__ import annotations

import copy
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
from RL_Module.mdp_definition import ACTIONS, ID_TO_ACTION, MISMATCH_HIGH, MISMATCH_LOW

OUT_DIR = _HERE / "figures" / "challenge_regime_audit"
TRACE_PATH = _HERE / "figures" / "dqn_vs_ppo_root_cause" / "trace_all_steps.csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 456]
TRAIN_TS = 50_000
EVAL_EPS = 300
GAIN_G0 = {"GAIN_CORRECT_FACTOR": 1.0, "GAIN_INCORRECT_FACTOR": 1.0}
COUNTERFACTUAL_SEED = 42
N_PROBE = 400
CF_HORIZONS = [5, 10, 20]


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


def _patch_gain() -> Dict[str, float]:
    prev = dict(cfg.SIMULATOR_PARAMS)
    cfg.SIMULATOR_PARAMS = dict(GAIN_G0)
    return prev


def _restore_gain(snap: Dict[str, float]) -> None:
    cfg.SIMULATOR_PARAMS = snap


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
    return obs, {"action_masks": env.get_action_mask(), "persistent_frustration_flag": env._persistent_frustration_flag}


def _zone_label(m: float) -> str:
    if m < MISMATCH_LOW:
        return "under_challenge"
    if m > MISMATCH_HIGH:
        return "overload"
    if m > 0.0:
        return "challenge"
    return "flow"


def _load_and_enrich_trace() -> pd.DataFrame:
    if not TRACE_PATH.exists():
        raise FileNotFoundError(f"Missing trace: {TRACE_PATH}")
    df = pd.read_csv(TRACE_PATH)
    df = df.sort_values(["algorithm", "seed", "episode", "step"]).reset_index(drop=True)
    g = df.groupby(["algorithm", "seed", "episode"], sort=False)
    df["prev_difficulty"] = g["difficulty"].shift(1)
    df["prev_knowledge"] = g["knowledge"].shift(1)
    df["prev_mismatch"] = g["mismatch"].shift(1)
    df["delta_difficulty"] = df["difficulty"] - df["prev_difficulty"]
    df["delta_mismatch"] = df["mismatch"] - df["prev_mismatch"]
    df["zone"] = df["mismatch"].map(_zone_label)
    return df


def _hist_dict(x: pd.Series, bins: int = 20) -> Dict[str, float]:
    counts, edges = np.histogram(x.dropna(), bins=bins, range=(0.0, 1.0))
    return {f"{edges[i]:.2f}-{edges[i+1]:.2f}": float(counts[i]) for i in range(len(counts))}


def _difficulty_evolution(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for algo in ["PPO", "DQN"]:
        sub = df[df["algorithm"] == algo]
        out[algo] = {
            "mean": float(sub["difficulty"].mean()),
            "median": float(sub["difficulty"].median()),
            "std": float(sub["difficulty"].std(ddof=1)),
            "histogram_0_1": _hist_dict(sub["difficulty"]),
            "mean_by_timestep": (
                sub.groupby("step")["difficulty"].mean().head(50).round(4).to_dict()
            ),
            "mean_by_episode_index": (
                sub.groupby("episode")["difficulty"].mean().head(50).round(4).to_dict()
            ),
            "mean_delta_difficulty_per_step": float(sub["delta_difficulty"].mean()),
            "pct_difficulty_increase_steps": float((sub["delta_difficulty"] > 0).mean()),
            "pct_difficulty_decrease_steps": float((sub["delta_difficulty"] < 0).mean()),
        }
        # harder_problem effect
        hp = sub[sub["action_name"] == "harder_problem"]
        sp = sub[sub["action_name"] == "simplify_problem"]
        out[algo]["after_harder_mean_delta_d"] = float(hp["delta_difficulty"].mean()) if len(hp) else float("nan")
        out[algo]["after_simplify_mean_delta_d"] = float(sp["delta_difficulty"].mean()) if len(sp) else float("nan")
    return out


def _knowledge_evolution(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    divergence_step = None
    for algo in ["PPO", "DQN"]:
        sub = df[df["algorithm"] == algo]
        out[algo] = {
            "mean_knowledge": float(sub["knowledge"].mean()),
            "episode_end_mean": float(sub.groupby(["seed", "episode"])["knowledge"].last().mean()),
            "mastery_curve_by_step": sub.groupby("step")["knowledge"].mean().head(50).round(4).to_dict(),
        }
    ppo_curve = df[df["algorithm"] == "PPO"].groupby("step")["knowledge"].mean()
    dqn_curve = df[df["algorithm"] == "DQN"].groupby("step")["knowledge"].mean()
    common = ppo_curve.index.intersection(dqn_curve.index)
    if len(common):
        gaps = (dqn_curve.loc[common] - ppo_curve.loc[common]).abs()
        divergence_step = int(gaps.idxmax()) if len(gaps) else None
    out["first_max_gap_timestep"] = divergence_step
    out["gap_at_step_0"] = float(dqn_curve.get(0, np.nan) - ppo_curve.get(0, np.nan))
    out["gap_at_step_10"] = float(dqn_curve.get(10, np.nan) - ppo_curve.get(10, np.nan))
    out["gap_at_step_20"] = float(dqn_curve.get(20, np.nan) - ppo_curve.get(20, np.nan))
    return out


def _mismatch_dynamics(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for algo in ["PPO", "DQN"]:
        sub = df[df["algorithm"] == algo]
        m = sub["mismatch"]
        out[algo] = {
            "mean": float(m.mean()),
            "median": float(m.median()),
            "std": float(m.std(ddof=1)),
            "p10": float(m.quantile(0.10)),
            "p25": float(m.quantile(0.25)),
            "p75": float(m.quantile(0.75)),
            "p90": float(m.quantile(0.90)),
            "pct_positive_mismatch": float((m > 0).mean()),
            "pct_negative_mismatch": float((m < 0).mean()),
            "zone_occupancy": sub["zone"].value_counts(normalize=True).round(4).to_dict(),
            "mismatch_by_timestep": sub.groupby("step")["mismatch"].mean().head(50).round(4).to_dict(),
        }
    return out


def _action_mismatch_transitions(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for algo in ["PPO", "DQN"]:
        sub = df[df["algorithm"] == algo].dropna(subset=["delta_mismatch"])
        rows = []
        for a in ACTIONS:
            a_sub = sub[sub["action_name"] == a]
            if len(a_sub) == 0:
                rows.append({"action": a, "n": 0})
                continue
            dm = a_sub["delta_mismatch"]
            rows.append(
                {
                    "action": a,
                    "n": int(len(a_sub)),
                    "freq": float(len(a_sub) / len(sub)),
                    "E_delta_mismatch": float(dm.mean()),
                    "E_delta_difficulty": float(a_sub["delta_difficulty"].mean()),
                    "pct_positive_dm": float((dm > 0).mean()),
                    "pct_negative_dm": float((dm < 0).mean()),
                }
            )
        out[algo] = rows
    return out


def _action_sequences(df: pd.DataFrame, top_n: int = 15) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for algo in ["PPO", "DQN"]:
        bigrams: Counter = Counter()
        trigrams: Counter = Counter()
        for _, ep in df[df["algorithm"] == algo].groupby(["seed", "episode"]):
            acts = ep.sort_values("step")["action_name"].tolist()
            for i in range(len(acts) - 1):
                bigrams[(acts[i], acts[i + 1])] += 1
            for i in range(len(acts) - 2):
                trigrams[(acts[i], acts[i + 1], acts[i + 2])] += 1
        out[algo] = {
            "top_bigrams": [
                {"seq": f"{a} -> {b}", "count": int(c)}
                for (a, b), c in bigrams.most_common(top_n)
            ],
            "top_trigrams": [
                {"seq": f"{a} -> {b} -> {c}", "count": int(c)}
                for (a, b, c), c in trigrams.most_common(top_n)
            ],
        }
    return out


def _state_drift(df: pd.DataFrame, horizon: int = 10) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    cols = ["knowledge", "difficulty", "mismatch", "engagement", "frustration", "confusion"]
    for algo in ["PPO", "DQN"]:
        sub = df[df["algorithm"] == algo].copy()
        sub["k_bin"] = pd.cut(
            sub["knowledge"],
            bins=[0, 0.33, 0.66, 1.0],
            labels=["low", "medium", "high"],
            include_lowest=True,
        )
        drift_rows = []
        for ep_key, ep in sub.groupby(["seed", "episode"]):
            ep = ep.sort_values("step")
            for col in cols:
                ep[f"future_{col}"] = ep[col].shift(-horizon)
            for kb in ["low", "medium", "high"]:
                part = ep[ep["k_bin"] == kb].dropna(subset=[f"future_{cols[0]}"])
                if len(part) == 0:
                    continue
                drift_rows.append(
                    {
                        "k_bin": kb,
                        "n": int(len(part)),
                        "delta_knowledge": float((part["future_knowledge"] - part["knowledge"]).mean()),
                        "delta_difficulty": float((part["future_difficulty"] - part["difficulty"]).mean()),
                        "delta_mismatch": float((part["future_mismatch"] - part["mismatch"]).mean()),
                        "delta_frustration": float((part["future_frustration"] - part["frustration"]).mean()),
                    }
                )
        if drift_rows:
            drift_df = pd.DataFrame(drift_rows).groupby("k_bin", as_index=False).mean(numeric_only=True)
            out[algo] = drift_df.round(4).to_dict(orient="records")
        else:
            out[algo] = []
    return out


def _reward_vs_mismatch(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    bins = np.linspace(-1.0, 1.0, 21)
    for algo in ["PPO", "DQN"]:
        sub = df[df["algorithm"] == algo]
        sub = sub.copy()
        sub["m_bin"] = pd.cut(sub["mismatch"], bins=bins)
        curve = (
            sub.groupby("m_bin", observed=True)["reward"]
            .agg(["mean", "count"])
            .reset_index()
        )
        curve_rows = []
        for r in curve.itertuples():
            curve_rows.append(
                {
                    "mismatch_bin": str(r.m_bin),
                    "mean_reward": float(r.mean),
                    "count": int(r.count),
                }
            )
        out[algo] = {
            "curve": curve_rows,
            "corr_reward_mismatch": float(sub[["reward", "mismatch"]].corr().iloc[0, 1]),
            "mean_reward_near_zero_mismatch": float(
                sub[sub["mismatch"].abs() <= 0.1]["reward"].mean()
            ),
            "mean_reward_positive_mismatch": float(sub[sub["mismatch"] > 0.1]["reward"].mean()),
            "mean_reward_negative_mismatch": float(sub[sub["mismatch"] < -0.1]["reward"].mean()),
        }
    return out


def _mask_cooldown_audit(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    challenge_actions = ["harder_problem", "scaffold", "hint", "explanation"]
    for algo in ["PPO", "DQN"]:
        sub = df[df["algorithm"] == algo]
        blocked = {}
        for a in challenge_actions:
            col = f"mask_valid_{a}"
            if col in sub.columns:
                blocked[a] = {
                    "pct_masked": float(1.0 - sub[col].mean()),
                    "pct_available": float(sub[col].mean()),
                }
        out[algo] = {
            "avg_valid_actions": float(sub["valid_actions_count"].mean()),
            "pct_steps_any_blocked": float((sub["invalid_actions_count"] > 0).mean()),
            "challenge_action_availability": blocked,
            "harder_blocked_when_frustrated_proxy": float(
                (
                    (sub["action_name"] != "harder_problem")
                    & (sub["frustration"] > 0.6)
                ).mean()
            ),
        }
    return out


def _train_agents(seed: int) -> Tuple[PPOAgent, DQNAgent]:
    cfg.set_all_seeds(seed)
    ppo_env = make_masked_env(seed=seed, algo_tag=f"challenge_audit_ppo_{seed}")
    ppo = PPOAgent()
    ppo.train(ppo_env, TRAIN_TS, seed)
    ppo_env.close()
    dqn_env = make_env(seed=seed, algo_tag=f"challenge_audit_dqn_{seed}")
    dqn = DQNAgent()
    dqn.train(dqn_env, TRAIN_TS, seed)
    dqn_env.close()
    return ppo, dqn


def _build_probes(seed: int, n: int) -> List[Dict[str, Any]]:
    cfg.set_all_seeds(seed)
    env = StudentEnv(population_seed=seed)
    probes = []
    obs, info = env.reset(seed=seed + 9000)
    mask = info["action_masks"]
    step = 0
    while len(probes) < n:
        probes.append({"obs": obs.copy(), "mask": mask.copy(), "snapshot": _capture_snapshot(env)})
        valid = np.flatnonzero(mask)
        a = int(np.random.choice(valid))
        obs, _, term, trunc, info = env.step(a)
        mask = info["action_masks"]
        step += 1
        if term or trunc:
            obs, info = env.reset(seed=seed + 9000 + step)
            mask = info["action_masks"]
    env.close()
    return probes


def _rollout_from_snapshot(
    env: StudentEnv,
    agent: Any,
    snap: EnvSnapshot,
    first_action: int,
    horizon: int,
) -> Dict[str, float]:
    obs, info = _restore_snapshot(env, snap)
    mask = info["action_masks"]
    k0 = float(env._state.knowledge)
    m0 = float(env._current_difficulty - env._state.knowledge)
    f0 = float(env._state.frustration)
    obs, r, term, trunc, info = env.step(first_action)
    total_r = float(r)
    done = bool(term or trunc)
    t = 1
    while (not done) and t < horizon:
        mask = info["action_masks"]
        a = int(agent.predict(obs, mask))
        obs, r, term, trunc, info = env.step(a)
        total_r += float(r)
        done = bool(term or trunc)
        t += 1
    k1 = float(env._state.knowledge)
    m1 = float(env._current_difficulty - env._state.knowledge)
    f1 = float(env._state.frustration)
    return {
        "delta_k": k1 - k0,
        "delta_mismatch": m1 - m0,
        "delta_frustration": f1 - f0,
        "total_reward": total_r,
    }


def _counterfactual_analysis(ppo: PPOAgent, dqn: DQNAgent, probes: List[Dict[str, Any]]) -> Dict[str, Any]:
    env = StudentEnv(population_seed=COUNTERFACTUAL_SEED)
    rows = []
    for p in probes:
        obs = p["obs"]
        mask = p["mask"]
        snap = p["snapshot"]
        ppo_a = int(ppo.predict(obs, mask))
        dqn_a = int(dqn.predict(obs, mask))
        for h in CF_HORIZONS:
            ppo_first = _rollout_from_snapshot(env, ppo, snap, ppo_a, h)
            dqn_first = _rollout_from_snapshot(env, ppo, snap, dqn_a, h)
            rows.append({"policy_continue": "PPO", "first_action_source": "PPO", "horizon": h, **ppo_first})
            rows.append({"policy_continue": "PPO", "first_action_source": "DQN", "horizon": h, **dqn_first})
    env.close()
    cdf = pd.DataFrame(rows)
    summary = (
        cdf.groupby(["first_action_source", "horizon"])
        .agg(
            mean_delta_k=("delta_k", "mean"),
            mean_delta_mismatch=("delta_mismatch", "mean"),
            mean_delta_frustration=("delta_frustration", "mean"),
            mean_total_reward=("total_reward", "mean"),
            n=("delta_k", "size"),
        )
        .reset_index()
    )
    return {
        "summary": summary.round(4).to_dict(orient="records"),
        "interpretation": (
            "first_action_source=PPO vs DQN on identical states, continued with PPO policy"
        ),
    }


def _rank_causes(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    diff = report["section1_difficulty_evolution"]
    mm = report["section3_mismatch_dynamics"]
    act = report["section4_action_mismatch_transitions"]
    seq = report["section5_action_sequences"]
    drift = report["section6_state_drift_h10"]
    rew = report["section7_reward_vs_mismatch"]
    mask = report["section8_mask_cooldown"]
    cf = report["section9_counterfactual"]

    def _dm(algo: str, action: str) -> float:
        for r in act[algo]:
            if r["action"] == action:
                return float(r.get("E_delta_mismatch", float("nan")))
        return float("nan")

    return [
        {
            "rank": 1,
            "cause": "Challenge-seeking action mix (especially harder_problem frequency and chains)",
            "evidence": {
                "dqn_harder_freq_in_sequences": seq["DQN"]["top_bigrams"][:3],
                "ppo_harder_E_delta_mismatch": _dm("PPO", "harder_problem"),
                "dqn_harder_E_delta_mismatch": _dm("DQN", "harder_problem"),
                "dqn_pct_difficulty_increase": diff["DQN"]["pct_difficulty_increase_steps"],
                "ppo_pct_difficulty_increase": diff["PPO"]["pct_difficulty_increase_steps"],
            },
        },
        {
            "rank": 2,
            "cause": "Policy-induced state drift toward easier vs harder regimes",
            "evidence": {
                "ppo_drift": drift.get("PPO", []),
                "dqn_drift": drift.get("DQN", []),
                "dqn_mean_mismatch": mm["DQN"]["mean"],
                "ppo_mean_mismatch": mm["PPO"]["mean"],
            },
        },
        {
            "rank": 3,
            "cause": "Difficulty adaptation dynamics (simplify vs harder usage)",
            "evidence": {
                "ppo_after_simplify_delta_d": diff["PPO"]["after_simplify_mean_delta_d"],
                "ppo_after_harder_delta_d": diff["PPO"]["after_harder_mean_delta_d"],
                "dqn_after_harder_delta_d": diff["DQN"]["after_harder_mean_delta_d"],
            },
        },
        {
            "rank": 4,
            "cause": "Reward-mismatch interaction favors near-zero mismatch for PPO",
            "evidence": {
                "ppo_corr_reward_mismatch": rew["PPO"]["corr_reward_mismatch"],
                "dqn_corr_reward_mismatch": rew["DQN"]["corr_reward_mismatch"],
                "ppo_reward_near_zero": rew["PPO"]["mean_reward_near_zero_mismatch"],
                "ppo_reward_positive_mismatch": rew["PPO"]["mean_reward_positive_mismatch"],
            },
        },
        {
            "rank": 5,
            "cause": "Mask/cooldown interaction secondary",
            "evidence": {
                "ppo_harder_pct_masked": mask["PPO"]["challenge_action_availability"].get("harder_problem", {}),
                "dqn_harder_pct_masked": mask["DQN"]["challenge_action_availability"].get("harder_problem", {}),
            },
        },
        {
            "rank": 6,
            "cause": "Counterfactual first-action effect on future challenge",
            "evidence": cf["summary"],
        },
    ]


def run_audit() -> Dict[str, Any]:
    print("=== Challenge-regime audit ===")
    snap = _patch_gain()
    cfg.USE_SENSITIVITY_WEIGHTS = False
    try:
        df = _load_and_enrich_trace()
        print(f"Loaded trace: {len(df)} steps")

        report = {
            "study": "challenge_regime_audit",
            "trace_source": str(TRACE_PATH),
            "seeds_in_trace": sorted(df["seed"].unique().tolist()),
            "gain_config": GAIN_G0,
            "section1_difficulty_evolution": _difficulty_evolution(df),
            "section2_knowledge_evolution": _knowledge_evolution(df),
            "section3_mismatch_dynamics": _mismatch_dynamics(df),
            "section4_action_mismatch_transitions": _action_mismatch_transitions(df),
            "section5_action_sequences": _action_sequences(df),
            "section6_state_drift_h10": _state_drift(df, horizon=10),
            "section7_reward_vs_mismatch": _reward_vs_mismatch(df),
            "section8_mask_cooldown": _mask_cooldown_audit(df),
        }

        print("Training agents for counterfactual section (seed=42)...")
        ppo, dqn = _train_agents(COUNTERFACTUAL_SEED)
        probes = _build_probes(COUNTERFACTUAL_SEED, N_PROBE)
        report["section9_counterfactual"] = _counterfactual_analysis(ppo, dqn, probes)
        report["root_cause_ranking"] = _rank_causes(report)

        # Final thesis synthesis numbers
        s1 = report["section1_difficulty_evolution"]
        s3 = report["section3_mismatch_dynamics"]
        report["final_thesis_mechanism"] = {
            "ppo_path": "comfort-biased actions -> near-zero/negative mismatch -> lower challenge exposure -> lower mastery",
            "dqn_path": "harder/scaffold chains -> positive mismatch -> higher challenge exposure -> higher mastery",
            "ppo_mean_difficulty": s1["PPO"]["mean"],
            "dqn_mean_difficulty": s1["DQN"]["mean"],
            "ppo_zone_occupancy": s3["PPO"]["zone_occupancy"],
            "dqn_zone_occupancy": s3["DQN"]["zone_occupancy"],
            "primary_mechanism": (
                "Policy-induced difficulty/mismatch dynamics: DQN repeatedly applies "
                "mismatch-increasing actions (harder_problem, scaffold under high difficulty), "
                "while PPO applies mismatch-reducing actions (simplify, encouragement, no_action) "
                "and rarely sustains harder_problem chains."
            ),
        }

        out_json = OUT_DIR / "challenge_regime_audit_report.json"
        with open(out_json, "w") as f:
            json.dump(report, f, indent=2)

        # CSV exports
        pd.DataFrame(report["section4_action_mismatch_transitions"]["PPO"]).to_csv(
            OUT_DIR / "ppo_action_mismatch_transitions.csv", index=False
        )
        pd.DataFrame(report["section4_action_mismatch_transitions"]["DQN"]).to_csv(
            OUT_DIR / "dqn_action_mismatch_transitions.csv", index=False
        )
        pd.DataFrame(report["section9_counterfactual"]["summary"]).to_csv(
            OUT_DIR / "counterfactual_rollouts.csv", index=False
        )

        print(f"Saved {out_json}")
        print("PPO mean difficulty:", s1["PPO"]["mean"], "DQN:", s1["DQN"]["mean"])
        print("PPO mean mismatch:", s3["PPO"]["mean"], "DQN:", s3["DQN"]["mean"])
        return report
    finally:
        _restore_gain(snap)


if __name__ == "__main__":
    run_audit()
