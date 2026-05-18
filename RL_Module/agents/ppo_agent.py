"""ALG 1 - MaskablePPO with action masking."""

from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor

from RL_Module import config
from RL_Module.agents.base_agent import BaseAgent
from RL_Module.environment.student_env import StudentEnv, mask_fn


def make_masked_env(
    seed: int,
    use_emotion: bool = True,
    use_emotion_bonuses: bool = True,
    ablation_no_emotion: bool = False,
    algo_tag: str = "ppo",
) -> gym.Env:
    env = StudentEnv(
        render_mode=config.RENDER_MODE,
        max_episode_steps=config.MAX_EPISODE_STEPS,
        population_seed=seed,
        use_emotion=use_emotion,
        use_emotion_bonuses=use_emotion_bonuses,
        ablation_no_emotion=ablation_no_emotion,
    )
    env = Monitor(
        env,
        filename=str(config.LOGS_DIR / f"monitor_{algo_tag}_seed{seed}"),
    )
    env = ActionMasker(env, mask_fn)
    return env


class PPOAgent(BaseAgent):
    name = "PPO"

    def __init__(self):
        self.model: Optional[MaskablePPO] = None

    def train(self, env: gym.Env, total_timesteps: int, seed: int) -> None:
        config.set_all_seeds(seed)
        policy_kwargs = dict(net_arch=dict(pi=[256, 256, 256], vf=[256, 256, 256]))
        self.model = MaskablePPO(
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
        self.model.learn(total_timesteps=total_timesteps)

    def predict(self, obs: np.ndarray, action_mask: Optional[np.ndarray] = None) -> int:
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() or load() first.")
        if action_mask is not None and action_mask.dtype != bool:
            action_mask = action_mask.astype(bool)
        action, _ = self.model.predict(obs, action_masks=action_mask, deterministic=True)
        return int(action)

    def get_action_probs(self, obs: np.ndarray, action_mask: np.ndarray) -> np.ndarray:
        obs_t = torch.as_tensor(obs).float().unsqueeze(0)
        mask_t = torch.as_tensor(action_mask).bool().unsqueeze(0)
        with torch.no_grad():
            dist = self.model.policy.get_distribution(obs_t, action_masks=mask_t)
            probs = dist.distribution.probs.detach().cpu().numpy().flatten()
        return probs

    def save(self, path: str) -> None:
        if self.model:
            self.model.save(path)

    def load(self, path: str) -> None:
        self.model = MaskablePPO.load(path)
