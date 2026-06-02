"""Double DQN with masked action selection at predict time."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch as th
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from torch.nn import functional as F

from RL_Module import config
from RL_Module.agents.base_agent import BaseAgent
from RL_Module.agents.dqn_agent import make_env
from RL_Module.environment.student_env import DQNSafeEnv, StudentEnv


class DoubleDQN(DQN):
    """
    Double DQN: online network selects the greedy action; target network evaluates it.
    Only the TD target computation differs from standard SB3 DQN.
    """

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            with th.no_grad():
                next_q_online = self.q_net(replay_data.next_observations)
                next_actions = next_q_online.argmax(dim=1, keepdim=True)
                next_q_target = self.q_net_target(replay_data.next_observations)
                next_q_values = th.gather(next_q_target, dim=1, index=next_actions.long())
                next_q_values = next_q_values.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            current_q_values = self.q_net(replay_data.observations)
            current_q_values = th.gather(
                current_q_values, dim=1, index=replay_data.actions.long()
            )

            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))


class DoubleDQNAgent(BaseAgent):
    name = "DoubleDQN"

    def __init__(self):
        self.model: Optional[DoubleDQN] = None
        self._checkpoint_dir: Optional[Path] = None
        self._best_checkpoint: Optional[str] = None

    def _build_model(
        self,
        env: gym.Env,
        seed: int,
        hp: Dict[str, Any],
    ) -> DoubleDQN:
        policy_kwargs = dict(net_arch=hp.get("net_arch", config.DQN_NET_ARCH))
        return DoubleDQN(
            "MlpPolicy",
            env,
            learning_rate=hp.get("learning_rate", config.DQN_LEARNING_RATE),
            batch_size=hp.get("batch_size", config.DQN_BATCH_SIZE),
            buffer_size=hp.get("buffer_size", config.DQN_BUFFER_SIZE),
            gamma=hp.get("gamma", config.DQN_GAMMA),
            train_freq=hp.get("train_freq", config.DQN_TRAIN_FREQ),
            target_update_interval=hp.get(
                "target_update_interval", config.DQN_TARGET_UPDATE_INTERVAL
            ),
            exploration_fraction=hp.get(
                "exploration_fraction", config.DQN_EXPLORATION_FRACTION
            ),
            exploration_final_eps=hp.get(
                "exploration_final_eps", config.DQN_EXPLORATION_FINAL_EPS
            ),
            policy_kwargs=policy_kwargs,
            verbose=0,
            seed=seed,
        )

    def train(
        self,
        env: gym.Env,
        total_timesteps: int,
        seed: int,
        hyperparams: Optional[Dict[str, Any]] = None,
    ) -> None:
        config.set_all_seeds(seed)
        hp = hyperparams or {}
        self.model = self._build_model(env, seed, hp)
        self.model.learn(total_timesteps=total_timesteps)

    def train_with_checkpoints(
        self,
        env: gym.Env,
        total_timesteps: int,
        seed: int,
        hyperparams: Optional[Dict[str, Any]] = None,
        checkpoint_dir: Optional[str] = None,
        checkpoint_freq: int = 10_000,
    ) -> List[str]:
        config.set_all_seeds(seed)
        hp = hyperparams or {}

        ckpt_path = Path(checkpoint_dir or config.MODELS_DIR / f"double_dqn_ckpts_seed{seed}")
        ckpt_path.mkdir(parents=True, exist_ok=True)
        self._checkpoint_dir = ckpt_path

        self.model = self._build_model(env, seed, hp)
        callback = CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=str(ckpt_path),
            name_prefix="double_dqn",
            save_replay_buffer=False,
        )
        self.model.learn(total_timesteps=total_timesteps, callback=callback)

        paths = sorted(ckpt_path.glob("double_dqn_*_steps.zip"))
        return [str(p) for p in paths]

    def load_checkpoint(self, path: str) -> None:
        self.model = DoubleDQN.load(path)
        self._best_checkpoint = path

    def select_best_checkpoint(
        self,
        checkpoint_paths: List[str],
        eval_fn: Callable[["DoubleDQNAgent", int], float],
        seed: int,
    ) -> Tuple[str, float]:
        best_path = checkpoint_paths[-1] if checkpoint_paths else ""
        best_score = -float("inf")
        for path in checkpoint_paths:
            self.load_checkpoint(path)
            score = eval_fn(self, seed)
            if score > best_score:
                best_score = score
                best_path = path
        if best_path:
            self.load_checkpoint(best_path)
            self._best_checkpoint = best_path
        return best_path, best_score

    def _masked_q_values(
        self, obs: np.ndarray, action_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        with torch.no_grad():
            q_values = (
                self.model.q_net(torch.as_tensor(obs).float().unsqueeze(0))
                .detach()
                .cpu()
                .numpy()
                .flatten()
            )
        if action_mask is None:
            return q_values
        masked_q = q_values.copy()
        blocked = action_mask == 0 if action_mask.dtype != bool else ~action_mask
        masked_q[blocked] = -np.inf
        return masked_q

    def predict(
        self,
        obs: np.ndarray,
        action_mask: Optional[np.ndarray] = None,
        eval_mode: str = "greedy",
        epsilon: float = 0.05,
        temperature: float = 1.0,
    ) -> int:
        if self.model is None:
            raise RuntimeError("Model not trained.")
        if action_mask is None:
            action, _ = self.model.predict(obs, deterministic=(eval_mode == "greedy"))
            return int(action)

        masked_q = self._masked_q_values(obs, action_mask)
        valid = np.flatnonzero(np.isfinite(masked_q))
        if len(valid) == 0:
            return 0

        if eval_mode == "epsilon":
            if np.random.random() < epsilon:
                return int(np.random.choice(valid))
            return int(valid[np.argmax(masked_q[valid])])

        if eval_mode == "softmax":
            qv = masked_q[valid].astype(float)
            qv = qv - np.max(qv)
            temp = max(temperature, 1e-6)
            probs = np.exp(qv / temp)
            probs /= probs.sum()
            return int(np.random.choice(valid, p=probs))

        return int(valid[np.argmax(masked_q[valid])])

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
        self.model = DoubleDQN.load(path)

    def verify_obs_compat(self, env: gym.Env) -> None:
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        expected = int(env.observation_space.shape[0])
        saved = int(self.model.observation_space.shape[0])
        if expected != saved:
            raise ValueError(
                f"Observation dimension mismatch: env expects {expected}, "
                f"model was trained with {saved}."
            )


__all__ = ["DoubleDQN", "DoubleDQNAgent", "make_env"]
