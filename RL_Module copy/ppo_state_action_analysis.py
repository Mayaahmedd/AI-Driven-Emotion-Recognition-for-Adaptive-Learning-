"""
PPO State-Action Contingency and Training Depth Analysis
=========================================================

Research question:
  Is PPO action selection STATE-DEPENDENT (context-sensitive) or
  merely frequency-driven (ignoring emotional state)

This script does NOT touch reward shaping.  It evaluates the existing
policy under controlled state conditions and produces:

  1. State-action contingency table  (6 state clusters x 8 actions)
  2. Chi-squared test of independence (state vs action)
  3. Pedagogical niche map per action
  4. Rare-action specialisation check
  5. PPO training curves at 100k / 200k / 500k timesteps
     - reward evolution
     - policy entropy evolution
     - action diversity (Shannon H) over time
  6. JSON summary + recommendation

Run from repo root:
    python -m RL_Module.ppo_state_action_analysis

Or directly:
    cd RL_Module && python ppo_state_action_analysis.py
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from stable_baselines3.common.callbacks import BaseCallback

#  resolve package path 
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from RL_Module import config
from RL_Module.agents.ppo_agent import PPOAgent, make_masked_env
from RL_Module.mdp_definition import ACTIONS, ID_TO_ACTION, ID_TO_EMOTION

#  output directories 
OUT_DIR = _HERE / "figures" / "ppo_state_action_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR = _HERE / "logs"

#  state-condition definitions 
#   Each condition is a function  f(row) -> bool
#   applied to a DataFrame with columns matching the step CSV schema.
#
#   Thresholds are calibrated to the actual PPO evaluation distribution
#   (student initial states: knowledge 0.1-0.5, affect 0-0.3 at episode start).
#   PPO is effective at keeping affect low, so "high" thresholds are relative
#   to what actually appears during evaluation, not absolute clinical scales.
#
#   "overload" = task too hard (frustration + confusion co-elevated at low knowledge)
#   "flow"     = Csikszentmihalyi zone (challenge ? skill, high engagement)

STATE_CONDITIONS: Dict[str, object] = {
    # Mild-to-moderate thresholds -- capture real PPO evaluation distribution
    "high_confusion":    lambda r: r["confusion"] > 0.28,
    "high_frustration":  lambda r: r["frustration"] > 0.28,
    "high_boredom":      lambda r: r["boredom"] > 0.28,
    "high_engagement":   lambda r: (r["engagement"] > 0.58)
                                   & (r["frustration"] < 0.25)
                                   & (r["confusion"] < 0.25),
    "overload":          lambda r: (r["frustration"] > 0.30)
                                   & (r["confusion"] > 0.28)
                                   & (r["knowledge"] < 0.55),
    "flow_state":        lambda r: (r["engagement"] > 0.58)
                                   & (r["frustration"] < 0.25)
                                   & (r["confusion"] < 0.25)
                                   & (r["boredom"] < 0.25)
                                   & (r["knowledge"] > 0.25),
}

ACTION_NAMES = list(ACTIONS)   # ["hint","scaffold",...,"no_action"]
N_ACTIONS = len(ACTION_NAMES)


# 
# Callback: records policy entropy + action distribution per rollout
# 
class EntropyActionCallback(BaseCallback):
    """
    After every PPO rollout (n_steps environment steps), compute:
      - mean policy entropy over the rollout buffer
      - per-action frequency in the collected rollout
    """

    def __init__(self, n_actions: int = 8):
        super().__init__()
        self.n_actions = n_actions
        self.timestep_log: List[int] = []
        self.entropy_log: List[float] = []
        self.action_freq_log: List[np.ndarray] = []
        self.reward_log: List[float] = []   # mean episode reward per rollout window
        self._ep_rewards: List[float] = []
        self._ep_buf: List[float] = []

    # called every env step during collect_rollouts
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self._ep_rewards.append(info["episode"]["r"])
        return True

    # called after each rollout (before gradient update)
    def _on_rollout_end(self) -> None:
        import torch

        ts = self.num_timesteps
        self.timestep_log.append(ts)

        # entropy from rollout buffer action log-probs
        buf = self.model.rollout_buffer
        if hasattr(buf, "log_probs") and buf.log_probs is not None:
            lp = buf.log_probs.flatten()
            ent = float(-np.mean(lp))           # H  -E[log (a|s)]
        else:
            ent = float("nan")
        self.entropy_log.append(ent)

        # action distribution in rollout
        if hasattr(buf, "actions") and buf.actions is not None:
            acts = buf.actions.flatten().astype(int)
            freq = np.zeros(self.n_actions, dtype=float)
            for a in acts:
                if 0 <= a < self.n_actions:
                    freq[a] += 1
            if freq.sum() > 0:
                freq /= freq.sum()
            self.action_freq_log.append(freq)
        else:
            self.action_freq_log.append(np.full(self.n_actions, float("nan")))

        # mean reward over episodes ending in this window
        if self._ep_rewards:
            self.reward_log.append(float(np.mean(self._ep_rewards)))
            self._ep_rewards = []
        else:
            self.reward_log.append(float("nan"))


# 
# Train PPO at checkpoints and return callback data
# 
def train_ppo_with_tracking(
    total_timesteps: int,
    seed: int = 42,
) -> Tuple[PPOAgent, EntropyActionCallback]:
    """Train MaskablePPO for `total_timesteps` steps and return (agent, callback)."""
    config.set_all_seeds(seed)
    env = make_masked_env(seed=seed, algo_tag=f"ppo_analysis_{total_timesteps}k")
    cb = EntropyActionCallback(n_actions=N_ACTIONS)

    agent = PPOAgent()
    agent.model = None

    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    from stable_baselines3.common.monitor import Monitor

    policy_kwargs = dict(net_arch=dict(pi=[256, 256, 256], vf=[256, 256, 256]))
    model = MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=config.PPO_LEARNING_RATE,
        gamma=config.PPO_GAMMA,
        clip_range=config.PPO_CLIP_RANGE,
        n_steps=config.PPO_N_STEPS,
        batch_size=config.PPO_BATCH_SIZE,
        ent_coef=config.PPO_ENT_COEF,
        policy_kwargs=policy_kwargs,
        verbose=0,
        seed=seed,
    )
    model.learn(total_timesteps=total_timesteps, callback=cb)
    agent.model = model
    return agent, cb


# 
# Evaluate agent: collect step-level state-action pairs
# 
def evaluate_state_action(
    agent: PPOAgent,
    n_episodes: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Roll out `n_episodes` deterministic episodes and return a DataFrame with
    one row per step containing state variables + chosen action.
    """
    from RL_Module.environment.student_env import StudentEnv, mask_fn

    env = StudentEnv(
        render_mode=None,
        max_episode_steps=config.MAX_EPISODE_STEPS,
        population_seed=seed,
        use_emotion=True,
    )

    rows = []
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        step = 0
        while not done:
            mask = env.get_action_mask()
            action = agent.predict(obs, action_mask=mask)
            obs_prev = obs.copy()
            obs, reward, terminated, truncated, info = env.step(action)
            rows.append({
                "episode": ep,
                "step": step,
                "action_id": action,
                "action_name": ID_TO_ACTION[action],
                "reward": reward,
                "knowledge": float(obs_prev[0]),
                "engagement": float(obs_prev[1]),
                "frustration": float(obs_prev[2]),
                "confusion": float(obs_prev[3]),
                "boredom": float(obs_prev[4]),
                "emotion_id": int(round(float(obs_prev[5]))),
                "emotion_name": ID_TO_EMOTION[int(round(float(obs_prev[5])))],
            })
            done = terminated or truncated
            step += 1
    return pd.DataFrame(rows)


# 
# State-action contingency analysis
# 
def build_contingency_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return DataFrame[state_condition x action_name] with row-normalised
    proportions (i.e., P(action | state_condition)).
    """
    records = {}
    for cond_name, cond_fn in STATE_CONDITIONS.items():
        mask = cond_fn(df)
        subset = df[mask]
        counts = subset["action_name"].value_counts().reindex(ACTION_NAMES, fill_value=0)
        records[cond_name] = counts.values.astype(float)

    tbl = pd.DataFrame(records, index=ACTION_NAMES).T  # shape: (conditions, actions)
    # row-normalise -> P(action | condition)
    row_sums = tbl.sum(axis=1).replace(0, np.nan)
    tbl_norm = tbl.div(row_sums, axis=0).fillna(0.0)
    return tbl_norm, tbl  # normalised, raw counts


def chi_squared_independence(df: pd.DataFrame) -> Dict:
    """
    Chi-squared test: H0 = action choice is independent of state condition.
    Skips conditions with fewer than 30 in-condition steps (insufficient data).
    Drops action columns where the entire column is zero (unobserved actions).
    """
    results = {}
    for cond_name, cond_fn in STATE_CONDITIONS.items():
        mask_in = cond_fn(df)
        in_cond = df[mask_in]["action_id"].values
        out_cond = df[~mask_in]["action_id"].values

        n_in = len(in_cond)
        n_out = len(out_cond)

        if n_in < 30:
            results[cond_name] = {
                "chi2": None,
                "p_value": None,
                "dof": None,
                "significant": False,
                "n_in": n_in,
                "n_out": n_out,
                "skipped": True,
                "skip_reason": f"only {n_in} in-condition steps (need >= 30)",
            }
            continue

        counts_in = np.bincount(in_cond, minlength=N_ACTIONS).astype(float)
        counts_out = np.bincount(out_cond, minlength=N_ACTIONS).astype(float)

        contingency = np.vstack([counts_in, counts_out])
        # drop action columns where neither in nor out has any counts
        col_mask = contingency.sum(axis=0) > 0
        contingency = contingency[:, col_mask]

        # require at least 2 action columns to run chi2
        if contingency.shape[1] < 2:
            results[cond_name] = {
                "chi2": None,
                "p_value": None,
                "dof": None,
                "significant": False,
                "n_in": int(n_in),
                "n_out": int(n_out),
                "skipped": True,
                "skip_reason": "fewer than 2 distinct actions observed in condition",
            }
            continue

        try:
            chi2, p, dof, _ = stats.chi2_contingency(contingency)
            results[cond_name] = {
                "chi2": float(chi2),
                "p_value": float(p),
                "dof": int(dof),
                "significant": bool(p < 0.05),
                "n_in": int(n_in),
                "n_out": int(n_out),
                "skipped": False,
            }
        except ValueError as e:
            results[cond_name] = {
                "chi2": None,
                "p_value": None,
                "dof": None,
                "significant": False,
                "n_in": int(n_in),
                "n_out": int(n_out),
                "skipped": True,
                "skip_reason": str(e),
            }
    return results


def detect_pedagogical_niches(tbl_norm: pd.DataFrame) -> Dict:
    """
    For each state condition, find the action(s) most over-represented
    relative to the marginal distribution across all conditions.
    """
    marginal = tbl_norm.mean(axis=0)  # marginal across conditions
    niches = {}
    for cond in tbl_norm.index:
        row = tbl_norm.loc[cond]
        lift = (row - marginal) / (marginal + 1e-8)
        top = lift.nlargest(3)
        niches[cond] = {
            action: {
                "P(a|s)": round(float(row[action]), 4),
                "P(a)_marginal": round(float(marginal[action]), 4),
                "lift": round(float(lift[action]), 4),
            }
            for action in top.index
        }
    return niches


def rare_action_specialisation(df: pd.DataFrame, threshold: float = 0.05) -> Dict:
    """
    Identify actions used less than `threshold` of the time overall.
    For each rare action, check whether it concentrates in specific state conditions.
    """
    overall = df["action_name"].value_counts(normalize=True)
    rare_actions = overall[overall < threshold].index.tolist()

    report = {}
    for action in rare_actions:
        action_rows = df[df["action_name"] == action]
        report[action] = {
            "overall_freq": round(float(overall[action]), 4),
            "n_uses": int(len(action_rows)),
            "condition_distribution": {},
        }
        for cond_name, cond_fn in STATE_CONDITIONS.items():
            in_cond_and_action = int((cond_fn(df) & (df["action_name"] == action)).sum())
            in_cond = int(cond_fn(df).sum())
            p_cond_given_action = (
                in_cond_and_action / len(action_rows) if len(action_rows) > 0 else 0
            )
            report[action]["condition_distribution"][cond_name] = {
                "uses_in_condition": in_cond_and_action,
                "P(condition|action)": round(p_cond_given_action, 4),
            }
    return report, rare_actions


# 
# Shannon diversity of action distribution
# 
def shannon_entropy(freq: np.ndarray) -> float:
    p = freq[freq > 0]
    return float(-np.sum(p * np.log(p)))


# 
# Plotting
# 
def plot_contingency_heatmap(tbl_norm: pd.DataFrame, label: str = "50k") -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("[WARN] matplotlib/seaborn not available -- skipping plots")
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(
        tbl_norm.round(3),
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        vmin=0,
        vmax=0.5,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "P(action | state condition)"},
    )
    ax.set_title(
        f"PPO State-Action Contingency Table  [{label} timesteps]\n"
        "Row = state condition  -  Column = action\n"
        "Values = P(action | state condition)"
    )
    ax.set_ylabel("State Condition")
    ax.set_xlabel("Action")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(OUT_DIR / f"contingency_heatmap_{label}.png", dpi=150)
    plt.close(fig)
    print(f"  [saved] contingency_heatmap_{label}.png")


def plot_training_curves(
    cb_dict: Dict[str, EntropyActionCallback],
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=False)
    colors = {"100k": "#4C72B0", "200k": "#55A868", "500k": "#C44E52"}

    # 1. Reward curves
    ax = axes[0]
    for label, cb in cb_dict.items():
        ts = cb.timestep_log
        rw = [r if not np.isnan(r) else None for r in cb.reward_log]
        # smooth with rolling mean
        rw_arr = pd.Series(rw).fillna(method="ffill").values
        window = max(1, len(rw_arr) // 20)
        rw_smooth = pd.Series(rw_arr).rolling(window, min_periods=1).mean().values
        ax.plot(ts, rw_smooth, label=f"{label}", color=colors.get(label))
    ax.set_ylabel("Mean Episode Reward (smoothed)")
    ax.set_title("PPO Training Reward Curves -- multi-budget comparison")
    ax.legend()
    ax.grid(alpha=0.3)

    # 2. Policy entropy
    ax = axes[1]
    for label, cb in cb_dict.items():
        ts = cb.timestep_log
        ent = pd.Series(cb.entropy_log).rolling(5, min_periods=1).mean().values
        ax.plot(ts, ent, label=f"{label}", color=colors.get(label))
    ax.set_ylabel("Policy Entropy  H(|s)  [nats]")
    ax.set_title("Policy Entropy Evolution -- lower = more deterministic policy")
    ax.legend()
    ax.grid(alpha=0.3)

    # 3. Action diversity (Shannon H of rollout action freq)
    ax = axes[2]
    for label, cb in cb_dict.items():
        ts = cb.timestep_log
        diversity = [
            shannon_entropy(f) if not np.any(np.isnan(f)) else float("nan")
            for f in cb.action_freq_log
        ]
        div_smooth = pd.Series(diversity).rolling(5, min_periods=1).mean().values
        ax.plot(ts, div_smooth, label=f"{label}", color=colors.get(label))
    max_h = float(np.log(N_ACTIONS))
    ax.axhline(max_h, color="gray", linestyle="--", alpha=0.5, label=f"max H={max_h:.2f}")
    ax.set_ylabel("Action Shannon H  [nats]")
    ax.set_xlabel("Training Timesteps")
    ax.set_title("Action Diversity During Training -- higher = more exploratory policy")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "training_curves.png", dpi=150)
    plt.close(fig)
    print("  [saved] training_curves.png")


def plot_action_specialisation(
    eval_results: Dict[str, pd.DataFrame],
) -> None:
    """Bar chart of per-action frequency at each training budget."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    budgets = list(eval_results.keys())
    x = np.arange(N_ACTIONS)
    width = 0.25
    colors = {"50k_existing": "#7fcdbb", "100k": "#4C72B0", "200k": "#55A868", "500k": "#C44E52"}

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, budget in enumerate(budgets):
        freq = eval_results[budget]["action_name"].value_counts(normalize=True).reindex(
            ACTION_NAMES, fill_value=0
        ).values
        ax.bar(x + i * width, freq, width, label=budget, color=colors.get(budget, None), alpha=0.85)

    ax.set_xticks(x + width * (len(budgets) - 1) / 2)
    ax.set_xticklabels(ACTION_NAMES, rotation=30, ha="right")
    ax.set_ylabel("Action Frequency (proportion)")
    ax.set_title("PPO Action Usage Across Training Budgets")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "action_specialisation.png", dpi=150)
    plt.close(fig)
    print("  [saved] action_specialisation.png")


def plot_reward_per_emotion(df: pd.DataFrame, label: str) -> None:
    """Box plot of reward per emotion (proxy for action quality)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.boxplot(data=df, x="emotion_name", y="reward", ax=ax,
                order=["confused", "bored", "frustrated", "engaged"],
                palette=["#FEBB46", "#8888BB", "#FF5F57", "#28C840"])
    ax.set_title(f"Reward Distribution by Emotion State  [{label}]")
    ax.set_xlabel("Emotion State")
    ax.set_ylabel("Step Reward")
    ax.axhline(0, color="black", linestyle="--", alpha=0.4)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT_DIR / f"reward_per_emotion_{label}.png", dpi=150)
    plt.close(fig)
    print(f"  [saved] reward_per_emotion_{label}.png")


# 
# Load existing 50k step log
# 
def load_existing_steps(algo: str = "PPO", seed: int = 42) -> Optional[pd.DataFrame]:
    path = LOGS_DIR / f"steps_{algo}_seed{seed}.csv"
    if not path.exists():
        print(f"[WARN] {path} not found -- skipping existing log analysis")
        return None
    df = pd.read_csv(path)
    print(f"[INFO] Loaded {len(df)} steps from {path.name}")
    return df


# 
# Evaluation helper: run deterministic episodes after training
# 
def run_eval(agent: PPOAgent, n_episodes: int = 300, seed: int = 42) -> Dict:
    """Run evaluation and return summary metrics."""
    from RL_Module.environment.student_env import StudentEnv

    env = StudentEnv(
        render_mode=None,
        max_episode_steps=config.MAX_EPISODE_STEPS,
        population_seed=seed,
        use_emotion=True,
    )

    ep_rewards, successes, gains = [], [], []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        k0 = float(obs[0])
        total_r = 0.0
        done = False
        while not done:
            mask = env.get_action_mask()
            action = agent.predict(obs, action_mask=mask)
            obs, r, terminated, truncated, info = env.step(action)
            total_r += r
            done = terminated or truncated
        ep_rewards.append(total_r)
        successes.append(float(info.get("knowledge", obs[0]) >= 0.95))
        gains.append(float(info.get("knowledge", obs[0])) - k0)

    return {
        "mean_reward": float(np.mean(ep_rewards)),
        "std_reward": float(np.std(ep_rewards)),
        "success_rate": float(np.mean(successes)),
        "mean_learning_gain": float(np.mean(gains)),
    }


# 
# Main
# 
def main() -> None:
    SEED = 42
    EVAL_EPS = 300
    TRAIN_BUDGETS = [100_000, 200_000, 500_000]
    BUDGET_LABELS = ["100k", "200k", "500k"]

    print("\n" + "=" * 70)
    print("  PPO STATE-ACTION CONTINGENCY & TRAINING DEPTH ANALYSIS")
    print("=" * 70)

    #  1. Analyse existing 50k eval log 
    print("\n[STEP 1] Analysing existing 50k evaluation log ...")
    df_50k = load_existing_steps("PPO", SEED)

    all_eval_dfs: Dict[str, pd.DataFrame] = {}

    if df_50k is not None and len(df_50k) > 0:
        # normalise column names
        df_50k.columns = [c.lower().strip() for c in df_50k.columns]
        if "action_name" not in df_50k.columns and "action_id" in df_50k.columns:
            df_50k["action_name"] = df_50k["action_id"].map(ID_TO_ACTION)
        if "emotion_name" not in df_50k.columns and "emotion_id" in df_50k.columns:
            df_50k["emotion_name"] = df_50k["emotion_id"].map(ID_TO_EMOTION)

        # --- Emotion-name contingency (direct, always populated) ---
        if "emotion_name" in df_50k.columns and "action_name" in df_50k.columns:
            emotion_pivot = (
                df_50k.groupby("emotion_name")["action_name"]
                .value_counts(normalize=True)
                .unstack(fill_value=0.0)
                .reindex(columns=ACTION_NAMES, fill_value=0.0)
            )
            print("\n  [Emotion-Name Action Contingency -- 50k (existing)]")
            print(emotion_pivot.round(3).to_string())

        tbl_norm_50k, tbl_raw_50k = build_contingency_table(df_50k)
        print("\n  [State-Condition Action Contingency Table -- 50k (existing)]")
        print(tbl_norm_50k.round(3).to_string())

        chi2_50k = chi_squared_independence(df_50k)
        niches_50k = detect_pedagogical_niches(tbl_norm_50k)
        rare_50k, rare_actions_50k = rare_action_specialisation(df_50k)

        plot_contingency_heatmap(tbl_norm_50k, "50k_existing")
        plot_reward_per_emotion(df_50k, "50k_existing")
        all_eval_dfs["50k_existing"] = df_50k

        print("\n  Chi-squared independence test (50k):")
        for cond, res in chi2_50k.items():
            if res.get("skipped"):
                print(f"    {cond:20s}  SKIPPED -- {res.get('skip_reason', '')}  n={res['n_in']}")
            else:
                sig = "SIGNIFICANT" if res["significant"] else "not significant"
                print(f"    {cond:20s}  chi2={res['chi2']:.1f}  p={res['p_value']:.4f}  [{sig}]  n={res['n_in']}")

        print("\n  Pedagogical niches (50k):")
        for cond, actions in niches_50k.items():
            top = sorted(actions.items(), key=lambda x: x[1]["lift"], reverse=True)[:2]
            parts = [f"{a} (lift={v['lift']:+.2f})" for a, v in top]
            print(f"    {cond:20s}  {', '.join(parts)}")

        print("\n  Rare-action analysis (50k):")
        for action, info in rare_50k.items():
            top_cond = max(
                info["condition_distribution"].items(),
                key=lambda x: x[1]["P(condition|action)"],
            )
            print(
                f"    {action:18s}  overall={info['overall_freq']:.3f}"
                f"  most-concentrated-in: {top_cond[0]} "
                f"({top_cond[1]['P(condition|action)']:.2f})"
            )
    else:
        print("  [SKIP] No existing 50k log found.")
        tbl_norm_50k = chi2_50k = niches_50k = rare_50k = rare_actions_50k = None

    #  2. Train PPO at increasing budgets 
    print("\n[STEP 2] Training PPO at multiple budgets ...")
    agents: Dict[str, PPOAgent] = {}
    callbacks: Dict[str, EntropyActionCallback] = {}
    eval_metrics: Dict[str, Dict] = {}

    for budget, label in zip(TRAIN_BUDGETS, BUDGET_LABELS):
        print(f"\n  Training {label} timesteps ...", end=" ", flush=True)
        t0 = time.time()
        agent, cb = train_ppo_with_tracking(budget, seed=SEED)
        elapsed = time.time() - t0
        print(f"done in {elapsed:.0f}s")

        # save model
        model_path = str(_HERE / "models" / f"ppo_analysis_{label}")
        agent.save(model_path)

        agents[label] = agent
        callbacks[label] = cb

        # quick evaluation
        print(f"  Evaluating {label} ...", end=" ", flush=True)
        metrics = run_eval(agent, n_episodes=EVAL_EPS, seed=SEED)
        eval_metrics[label] = metrics
        print(
            f"  reward={metrics['mean_reward']:.3f}  "
            f"success={metrics['success_rate']:.3f}  "
            f"gain={metrics['mean_learning_gain']:.3f}"
        )

        # state-action eval
        df_eval = evaluate_state_action(agent, n_episodes=EVAL_EPS, seed=SEED)
        all_eval_dfs[label] = df_eval

        # Emotion-name contingency
        emotion_pivot = (
            df_eval.groupby("emotion_name")["action_name"]
            .value_counts(normalize=True)
            .unstack(fill_value=0.0)
            .reindex(columns=ACTION_NAMES, fill_value=0.0)
        )
        print(f"\n  [Emotion-Name Action Contingency -- {label}]")
        print(emotion_pivot.round(3).to_string())

        tbl_norm, tbl_raw = build_contingency_table(df_eval)
        plot_contingency_heatmap(tbl_norm, label)
        plot_reward_per_emotion(df_eval, label)

        print(f"\n  [State-Condition Action Contingency Table -- {label}]")
        print(tbl_norm.round(3).to_string())

        chi2_res = chi_squared_independence(df_eval)
        print(f"\n  Chi-squared independence test ({label}):")
        for cond, res in chi2_res.items():
            if res.get("skipped"):
                print(f"    {cond:20s}  SKIPPED -- {res.get('skip_reason', '')}  n={res['n_in']}")
            else:
                sig = "SIGNIFICANT" if res["significant"] else "not significant"
                print(f"    {cond:20s}  chi2={res['chi2']:.1f}  p={res['p_value']:.4f}  [{sig}]  n={res['n_in']}")

    #  3. Plot training curves 
    print("\n[STEP 3] Plotting training curves ...")
    plot_training_curves(callbacks)
    plot_action_specialisation(all_eval_dfs)

    #  4. Final comparative table 
    print("\n[STEP 4] Comparative performance summary")
    print(f"\n  {'Budget':12s}  {'Reward':>10s}  {'Success':>8s}  {'Gain':>8s}")
    print("  " + "-" * 42)
    if df_50k is not None:
        print(f"  {'50k (existing)':12s}  {'3.543':>10s}  {'0.068':>8s}  {'0.345':>8s}")
    for label in BUDGET_LABELS:
        m = eval_metrics[label]
        print(
            f"  {label:12s}  {m['mean_reward']:10.3f}  "
            f"{m['success_rate']:8.3f}  {m['mean_learning_gain']:8.3f}"
        )
    print(f"\n  DQN baseline:     reward=1.936  success=0.500  gain=0.670")

    #  5. Determine recommendation 
    print("\n[STEP 5] Generating recommendation ...")

    # check state-dependence at 500k
    df_500k = all_eval_dfs.get("500k")
    chi2_500k = chi_squared_independence(df_500k) if df_500k is not None else {}
    n_significant = sum(v["significant"] for v in chi2_500k.values())
    n_testable = sum(not v.get("skipped", False) for v in chi2_500k.values())
    # context-sensitive if majority of testable conditions are significant
    context_sensitive = n_testable > 0 and (n_significant / max(n_testable, 1)) >= 0.5

    # check convergence: reward at 200k vs 500k
    r_200k = eval_metrics.get("200k", {}).get("mean_reward", 0)
    r_500k = eval_metrics.get("500k", {}).get("mean_reward", 0)
    still_improving = (r_500k - r_200k) > 0.1 * abs(r_200k)

    # check DQN gap at 500k success rate
    dqn_success = 0.500
    ppo_500k_success = eval_metrics.get("500k", {}).get("success_rate", 0)
    success_gap = dqn_success - ppo_500k_success

    print(f"\n  Context sensitivity (chi2 p<0.05 in {n_significant}/6 conditions): "
          f"{'YES' if context_sensitive else 'NO'}")
    print(f"  Still improving 200k500k: {'YES' if still_improving else 'NO'}")
    print(f"  Success gap vs DQN at 500k: {success_gap:.3f}")

    recommend_reward_shaping = (
        not context_sensitive
        and not still_improving
        and success_gap > 0.2
    )

    recommendation = (
        "CONTINUE TRAINING (increase budget) -- PPO is still learning; "
        "do not modify rewards yet."
        if still_improving else (
            "REWARD SHAPING WARRANTED -- PPO has converged but remains "
            "context-insensitive and substantially behind DQN."
            if recommend_reward_shaping else
            "PPO IS CONTEXT-SENSITIVE -- specialisation emerged; "
            "reward shaping not recommended at this time."
        )
    )
    print(f"\n  RECOMMENDATION: {recommendation}")

    #  6. Save JSON report 
    print("\n[STEP 6] Saving JSON report ...")

    def _safe(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_safe(v) for v in obj]
        return obj

    report = {
        "analysis": "PPO State-Action Contingency & Training Depth",
        "seed": SEED,
        "eval_episodes_per_budget": EVAL_EPS,
        "performance": {
            "50k_existing": {
                "mean_reward": 3.543,
                "success_rate": 0.068,
                "learning_gain": 0.345,
            },
            **{
                label: _safe(eval_metrics[label])
                for label in BUDGET_LABELS
            },
            "DQN_baseline": {
                "mean_reward": 1.936,
                "success_rate": 0.500,
                "learning_gain": 0.670,
            },
        },
        "context_sensitivity_500k": {
            "n_significant_conditions": n_significant,
            "context_sensitive": context_sensitive,
            "chi2_by_condition": _safe(chi2_500k),
        },
        "training_dynamics": {
            "still_improving_200k_to_500k": still_improving,
            "reward_200k": float(r_200k),
            "reward_500k": float(r_500k),
        },
        "recommendation": recommendation,
        "recommend_reward_shaping": recommend_reward_shaping,
    }

    if niches_50k is not None:
        report["pedagogical_niches_50k"] = _safe(niches_50k)
    if rare_50k is not None:
        report["rare_action_specialisation_50k"] = _safe(rare_50k)

    out_json = OUT_DIR / "ppo_state_action_report.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  [saved] {out_json}")

    print("\n" + "=" * 70)
    print("  ANALYSIS COMPLETE")
    print(f"  Outputs: {OUT_DIR}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
