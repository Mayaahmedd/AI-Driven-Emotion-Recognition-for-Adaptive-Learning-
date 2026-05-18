"""ALG 2 - DQN with masked action selection at predict time."""

from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from RL_Module import config
from RL_Module.agents.base_agent import BaseAgent
from RL_Module.environment.student_env import DQNSafeEnv, StudentEnv


def make_env(
    seed: int,
    use_emotion: bool = True,
    use_emotion_bonuses: bool = True,
    ablation_no_emotion: bool = False,
    algo_tag: str = "dqn",
) -> gym.Env:
    env = StudentEnv(
        render_mode=config.RENDER_MODE,
        max_episode_steps=config.MAX_EPISODE_STEPS,
        population_seed=seed,
        use_emotion=use_emotion,
        use_emotion_bonuses=use_emotion_bonuses,
        ablation_no_emotion=ablation_no_emotion,
    )
    env = DQNSafeEnv(env)
    return Monitor(
        env,
        filename=str(config.LOGS_DIR / f"monitor_{algo_tag}_seed{seed}"),
    )


class DQNAgent(BaseAgent):
    name = "DQN"

    def __init__(self):
        self.model: Optional[DQN] = None

    def train(self, env: gym.Env, total_timesteps: int, seed: int) -> None:
        config.set_all_seeds(seed)
        policy_kwargs = dict(net_arch=config.DQN_NET_ARCH)
        self.model = DQN(
            "MlpPolicy",
            env,
            learning_rate=config.DQN_LEARNING_RATE,
            batch_size=config.DQN_BATCH_SIZE,
            gamma=config.DQN_GAMMA,
            train_freq=config.DQN_TRAIN_FREQ,
            target_update_interval=config.DQN_TARGET_UPDATE_INTERVAL,
            exploration_fraction=config.DQN_EXPLORATION_FRACTION,
            exploration_final_eps=config.DQN_EXPLORATION_FINAL_EPS,
            policy_kwargs=policy_kwargs,
            verbose=0,
            seed=seed,
        )
        self.model.learn(total_timesteps=total_timesteps)

    def predict(self, obs: np.ndarray, action_mask: Optional[np.ndarray] = None) -> int:
        if self.model is None:
            raise RuntimeError("Model not trained.")
        if action_mask is None:
            action, _ = self.model.predict(obs, deterministic=True)
            return int(action)

        with torch.no_grad():
            q_values = (
                self.model.q_net(torch.as_tensor(obs).float().unsqueeze(0))
                .detach()
                .cpu()
                .numpy()
                .flatten()
            )
        masked_q = q_values.copy()
        blocked = action_mask == 0 if action_mask.dtype != bool else ~action_mask
        masked_q[blocked] = -np.inf
        return int(np.argmax(masked_q))

    def get_q_values(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            q = (
                self.model.q_net(torch.as_tensor(obs).float().unsqueeze(0))
                .detach()
                .cpu()
                .numpy()
                .flatten()
            )
        return q

    def save(self, path: str) -> None:
        if self.model:
            self.model.save(path)

    def load(self, path: str) -> None:
        self.model = DQN.load(path)
