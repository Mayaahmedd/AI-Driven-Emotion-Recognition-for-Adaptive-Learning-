"""
Evaluation metrics with 95% confidence intervals and ablation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from RL_Module import config
from RL_Module.agents.base_agent import BaseAgent
from RL_Module.agents.bandit_dqn import BanditDQNAgent
from RL_Module.agents.dqn_agent import DQNAgent, make_env
from RL_Module.agents.ppo_agent import PPOAgent, make_masked_env
from RL_Module.agents.ppo_dqn_hybrid import PPODQNHybridAgent
from RL_Module.environment.student_env import StudentEnv
from RL_Module.explainability.explainer import Explainer
from RL_Module.logging_utils.csv_logger import EpisodeLogger, StepLogger
from RL_Module.mdp_definition import BEST_ACTION_MAP, ID_TO_EMOTION, normalized_knowledge_gain

RL_ALGORITHMS = ("PPO", "DQN", "PPO_DQN", "Bandit_DQN")


def confidence_interval(values: List[float], z: float = 1.96) -> Tuple[float, float, float]:
    arr = np.array(values, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    margin = z * std / np.sqrt(len(arr)) if len(arr) > 0 else 0.0
    return mean, mean - margin, mean + margin


def convergence_episode(rewards: List[float], window: int = 50, threshold: float = 0.05) -> Optional[int]:
    if len(rewards) < window:
        return None
    rolling_std = pd.Series(rewards).rolling(window).std()
    for i in range(window, len(rewards)):
        if rolling_std.iloc[i] < threshold:
            return i - window
    return None


def _is_success(final_knowledge: float, final_frustration: float, final_confusion: float) -> bool:
    return final_knowledge > 0.8 and final_frustration < 0.3 and final_confusion < 0.4


def _adaptation_match(emotion_id: int, action: int) -> bool:
    return action in BEST_ACTION_MAP.get(emotion_id, set())


def evaluate(
    agent: BaseAgent,
    env: StudentEnv,
    n_episodes: int = config.EVAL_EPISODES,
    seed: int = config.DEFAULT_SEED,
    algorithm: str = "Agent",
    log_steps: bool = True,
    explainer: Optional[Explainer] = None,
) -> Dict[str, Any]:
    config.set_all_seeds(seed)
    episode_results: List[Dict[str, Any]] = []
    step_logger = StepLogger(algorithm, seed) if log_steps else None
    ep_logger = EpisodeLogger(algorithm, seed) if log_steps else None

    all_episode_rewards: List[float] = []
    adapt_hits = 0
    adapt_total = 0
    rolling_for_conv: List[float] = []

    for ep in range(1, n_episodes + 1):
        obs, info = env.reset(seed=seed + ep)
        env._episode_count = ep
        mask = info["action_masks"]
        total_reward = 0.0
        start_k = env._state.knowledge
        action_counts = {i: 0 for i in range(10)}
        ep_adapt_hits = 0
        ep_adapt_total = 0
        step = 0
        dropout = False
        done = False

        while not done:
            emotion_id = env._state.emotion_id

            if isinstance(agent, BanditDQNAgent):
                action = agent.predict(
                    obs,
                    mask,
                    persistent_flag=info.get("persistent_frustration_flag", False),
                )
            else:
                action = agent.predict(obs, mask)

            if explainer:
                record = explainer.explain(
                    step,
                    ep,
                    env._state,
                    action,
                    env._persistent_frustration_flag,
                    env._fer.confidence,
                )
                env.set_explainer_reason(record["reason"])

            obs, reward, terminated, truncated, info = env.step(action)
            if hasattr(agent, "set_last_reward"):
                agent.set_last_reward(reward)

            ep_adapt_total += 1
            if _adaptation_match(emotion_id, action):
                ep_adapt_hits += 1

            total_reward += reward
            action_counts[action] += 1
            step += 1
            mask = info["action_masks"]
            done = terminated or truncated
            dropout = info.get("dropout", False)

            if step_logger:
                step_logger.log_step({
                    "step": step,
                    "episode": ep,
                    "algorithm": algorithm,
                    "seed": seed,
                    "action_id": action,
                    "action_name": info.get("action_name", ""),
                    "reward": round(reward, 4),
                    "cumulative_reward": round(total_reward, 4),
                    "knowledge": round(env._state.knowledge, 4),
                    "engagement": round(env._state.engagement, 4),
                    "frustration": round(env._state.frustration, 4),
                    "confusion": round(env._state.confusion, 4),
                    "boredom": round(env._state.boredom, 4),
                    "emotion_id": env._state.emotion_id,
                    "emotion_name": ID_TO_EMOTION[env._state.emotion_id],
                    "persistent_flag": info.get("persistent_frustration_flag", False),
                    "terminated": terminated,
                    "truncated": truncated,
                    "explainer_reason": env._explainer_reason,
                })

        final = env._state
        lg = final.knowledge - start_k
        lg_norm = normalized_knowledge_gain(start_k, final.knowledge)
        success = _is_success(final.knowledge, final.frustration, final.confusion)
        adapt_acc = ep_adapt_hits / max(ep_adapt_total, 1)

        adapt_hits += ep_adapt_hits
        adapt_total += ep_adapt_total
        all_episode_rewards.append(total_reward)
        rolling_for_conv.append(total_reward)

        conv_flag = False
        if len(rolling_for_conv) >= 50:
            if np.std(rolling_for_conv[-50:]) < 0.05:
                conv_flag = True

        row = {
            "episode": ep,
            "algorithm": algorithm,
            "seed": seed,
            "total_reward": round(total_reward, 4),
            "learning_gain": round(lg, 4),
            "learning_gain_normalized": round(lg_norm, 4),
            "success": int(success),
            "dropout": int(dropout),
            "steps_taken": step,
            "adaptation_accuracy": round(adapt_acc, 4),
            "convergence_flag": int(conv_flag),
            "action_counts": action_counts,
        }
        episode_results.append(row)
        if ep_logger:
            ep_logger.log_episode(row)

    if step_logger:
        step_logger.close()
    if ep_logger:
        ep_logger.close()

    conv_ep = convergence_episode(all_episode_rewards)
    return {
        "episode_results": episode_results,
        "cumulative_reward": float(np.sum(all_episode_rewards)),
        "mean_episode_reward": float(np.mean(all_episode_rewards)),
        "convergence_episode": conv_ep,
        "success_rate": float(np.mean([r["success"] for r in episode_results])),
        "learning_gain": float(np.mean([r["learning_gain_normalized"] for r in episode_results])),
        "dropout_rate": float(np.mean([r["dropout"] for r in episode_results])),
        "adaptation_accuracy": adapt_hits / max(adapt_total, 1),
        "episode_rewards": all_episode_rewards,
        "student_type": getattr(env._student, "student_type", "general"),
    }


def aggregate_across_seeds(per_seed_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = [
        "mean_episode_reward",
        "success_rate",
        "learning_gain",
        "dropout_rate",
        "adaptation_accuracy",
    ]
    agg: Dict[str, Any] = {}
    for key in keys:
        vals = [r[key] for r in per_seed_results]
        mean, lo, hi = confidence_interval(vals)
        agg[f"mean_{key}"] = mean
        agg[f"ci_lower_{key}"] = lo
        agg[f"ci_upper_{key}"] = hi

    conv_vals = [r["convergence_episode"] for r in per_seed_results if r.get("convergence_episode") is not None]
    agg["convergence_episode"] = int(np.mean(conv_vals)) if conv_vals else None
    agg["final_reward_mean"] = agg["mean_mean_episode_reward"]
    reward_vals = [r["mean_episode_reward"] for r in per_seed_results]
    agg["final_reward_std"] = (
        float(np.std(reward_vals, ddof=1)) if len(reward_vals) > 1 else 0.0
    )
    agg["ci_lower"] = agg["ci_lower_mean_episode_reward"]
    agg["ci_upper"] = agg["ci_upper_mean_episode_reward"]
    agg["final_reward"] = f"{agg['final_reward_mean']:.0f} +/- {(agg['ci_upper']-agg['final_reward_mean']):.0f} (95% CI)"
    return agg


def run_ablation(
    train_timesteps: int = config.TRAINING_TIMESTEPS,
    eval_episodes: int = 100,
    seed: int = config.DEFAULT_SEED,
) -> Dict[str, Any]:
    """Version A: no emotion. Version B: full. All 4 RL algorithms."""
    results: Dict[str, Any] = {"A": {}, "B": {}, "delta": {}}

    for label, ablation in [("A", True), ("B", False)]:
        for algo_name, agent_cls in [
            ("PPO", PPOAgent),
            ("DQN", DQNAgent),
            ("PPO_DQN", PPODQNHybridAgent),
            ("Bandit_DQN", BanditDQNAgent),
        ]:
            if algo_name == "PPO":
                train_env = make_masked_env(seed, ablation_no_emotion=ablation)
            else:
                train_env = make_env(seed, ablation_no_emotion=ablation)

            agent = agent_cls()
            agent.train(train_env, train_timesteps, seed)
            train_env.close()

            eval_env = StudentEnv(
                population_seed=seed,
                use_emotion=not ablation,
                use_emotion_bonuses=not ablation,
                ablation_no_emotion=ablation,
            )
            key = f"{algo_name}_{label}"
            results[label][algo_name] = evaluate(
                agent,
                eval_env,
                n_episodes=eval_episodes,
                seed=seed,
                algorithm=key,
                log_steps=False,
            )
            eval_env.close()

    for algo in RL_ALGORITHMS:
        for metric in ["mean_episode_reward", "success_rate", "learning_gain"]:
            a_val = results["A"][algo].get(metric, 0)
            b_val = results["B"][algo].get(metric, 0)
            results["delta"][f"{algo}_{metric}"] = b_val - a_val

    return results
