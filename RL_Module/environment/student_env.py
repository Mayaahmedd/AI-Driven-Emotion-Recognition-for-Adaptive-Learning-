"""
Gymnasium environment for the adaptive tutoring MDP.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from RL_Module import config
from RL_Module.environment.population import generate_population
from RL_Module.environment.student_model import SyntheticStudent
from RL_Module.fer_interface.fer_adapter import FERAdapter, emotion_from_state
from RL_Module.mdp_definition import (
    ACTIONS,
    COOLDOWNS,
    EMERGENCY_ALLOWED,
    ENGAGEMENT_MIN_REFLECTION,
    FRUSTRATION_BLOCK_HARDER,
    FRUSTRATION_BLOCK_STRATEGY,
    FRUSTRATION_PERSISTENT_OFF,
    FRUSTRATION_PERSISTENT_ON,
    FRUSTRATION_PERSISTENT_STEPS,
    ID_TO_ACTION,
    ID_TO_EMOTION,
    KNOWLEDGE_MAX_SCAFFOLD,
    KNOWLEDGE_MIN_AUTONOMY,
    EMOTION_TO_ID,
    StudentState,
)
from RL_Module.explainability.explainer import reason_for
from RL_Module.reward.reward_function import compute_reward

OBS_HIGH = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 3.0], dtype=np.float32)


class StudentEnv(gym.Env):
    """
    observation_space: Box(6,) - knowledge, engagement, frustration, confusion, boredom, emotion_id
    action_space: Discrete(10)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_episode_steps: int = config.MAX_EPISODE_STEPS,
        population_seed: Optional[int] = None,
        use_emotion: bool = True,
        ablation_no_emotion: bool = False,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.use_emotion = use_emotion
        self.ablation_no_emotion = ablation_no_emotion

        self.observation_space = spaces.Box(
            low=0.0, high=OBS_HIGH, shape=(6,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(10)

        self._population: List[SyntheticStudent] = generate_population(
            seed=population_seed or config.DEFAULT_SEED
        )
        self._rng = np.random.default_rng(config.DEFAULT_SEED)
        self._student: Optional[SyntheticStudent] = None
        self._fer = FERAdapter()
        self._state = StudentState(0.3, 0.6, 0.1, 0.1, 0.1, 3)
        self._prev_state = self._state.copy()

        self._step_count = 0
        self._episode_count = 0
        self._cooldowns: Dict[int, int] = {i: 0 for i in range(10)}
        self._persistent_frustration_flag = False
        self._frustration_high_streak = 0
        self._consecutive_flag_steps = 0
        self._frustration_history: deque = deque(maxlen=3)
        self._last_answer_wrong = False
        self._cumulative_reward = 0.0
        self._reward_history: List[float] = []
        self._last_action: int = 0
        self._last_reward: float = 0.0
        self._explainer_reason: str = ""
        self._should_quit = False

        self._screen = None
        self._clock = None
        self._font = None
        self._algorithm_name = "Tutor"

    @property
    def frustration(self) -> float:
        return self._state.frustration

    @frustration.setter
    def frustration(self, v: float) -> None:
        self._state.frustration = float(v)

    @property
    def emotion_id(self) -> int:
        return self._state.emotion_id

    @emotion_id.setter
    def emotion_id(self, v: int) -> None:
        self._state.emotion_id = int(v)

    @property
    def persistent_flag(self) -> bool:
        return self._persistent_frustration_flag

    @persistent_flag.setter
    def persistent_flag(self, v: bool) -> None:
        self._persistent_frustration_flag = bool(v)

    @property
    def knowledge(self) -> float:
        return self._state.knowledge

    @property
    def engagement(self) -> float:
        return self._state.engagement

    @property
    def confusion(self) -> float:
        return self._state.confusion

    @property
    def boredom(self) -> float:
        return self._state.boredom

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def consecutive_flag_steps(self) -> int:
        return self._consecutive_flag_steps

    def _sample_student(self) -> SyntheticStudent:
        idx = int(self._rng.integers(0, len(self._population)))
        student = self._population[idx]
        student.sample_initial_state(self._rng)
        return student

    def get_action_mask(self) -> np.ndarray:
        """Return int8 mask: 1 = available, 0 = blocked."""
        mask = np.ones(10, dtype=np.int8)
        s = self._state

        if self._persistent_frustration_flag:
            mask[:] = 0
            for a in EMERGENCY_ALLOWED:
                mask[a] = 1
            return mask

        if s.frustration > FRUSTRATION_BLOCK_HARDER or s.emotion_id == EMOTION_TO_ID["frustrated"]:
            mask[1] = 0
        if s.engagement < ENGAGEMENT_MIN_REFLECTION:
            mask[6] = 0
        if s.frustration > FRUSTRATION_BLOCK_STRATEGY:
            mask[7] = 0
        if s.knowledge < KNOWLEDGE_MIN_AUTONOMY:
            mask[8] = 0
        if s.knowledge >= KNOWLEDGE_MAX_SCAFFOLD:
            mask[4] = 0

        for action_id, remaining in self._cooldowns.items():
            if remaining > 0:
                mask[action_id] = 0

        return mask

    def action_masks(self) -> np.ndarray:
        """Alias for MaskablePPO (bool mask)."""
        return self.get_action_mask().astype(bool)

    def _derive_emotion_id(self) -> int:
        eid, _ = emotion_from_state(self._state)
        return eid

    def _obs(self) -> np.ndarray:
        vec = self._state.as_vec().astype(np.float32)
        if self.ablation_no_emotion or not self.use_emotion:
            vec[5] = 0.0
        return vec

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._student = self._sample_student()
        self._fer.reset()
        self._state = self._student.state.copy()
        eid, _ = self._fer.get_emotion(self._state)
        self._state.emotion_id = eid
        self._prev_state = self._state.copy()

        self._step_count = 0
        self._cooldowns = {i: 0 for i in range(10)}
        self._persistent_frustration_flag = False
        self._frustration_high_streak = 0
        self._consecutive_flag_steps = 0
        self._frustration_history.clear()
        self._last_answer_wrong = bool(self._rng.random() < 0.3)
        self._cumulative_reward = 0.0
        self._reward_history = []
        self._should_quit = False

        info = {
            "action_masks": self.get_action_mask(),
            "persistent_frustration_flag": self._persistent_frustration_flag,
        }
        return self._obs(), info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        mask = self.get_action_mask()
        if mask[action] == 0:
            raise ValueError(
                f"Action {action} ({ID_TO_ACTION[action]}) is masked. "
                f"Allowed: {[ID_TO_ACTION[i] for i in range(10) if mask[i]]}"
            )

        self._prev_state = self._state.copy()
        self._last_action = action

        self._state = self._student.apply_action(action, self._prev_state)

        if config.USE_REAL_FER:
            eid, conf = self._fer.get_emotion(self._state)
        else:
            eid = self._derive_emotion_id()
            conf = 0.85
            self._fer._current_emotion_id = eid
            self._fer._confidence = conf
        self._state.emotion_id = eid

        self._explainer_reason = reason_for(
            self._state.as_vec(),
            action,
            persistent_flag=self._persistent_frustration_flag,
            confidence=conf,
        )

        self._frustration_history.append(self._state.frustration)

        if self._state.frustration > FRUSTRATION_PERSISTENT_ON:
            self._frustration_high_streak += 1
        else:
            self._frustration_high_streak = 0

        if self._frustration_high_streak >= FRUSTRATION_PERSISTENT_STEPS:
            self._persistent_frustration_flag = True
        if self._state.frustration < FRUSTRATION_PERSISTENT_OFF:
            self._persistent_frustration_flag = False
            self._frustration_high_streak = 0
            self._consecutive_flag_steps = 0

        if self._persistent_frustration_flag:
            self._consecutive_flag_steps += 1
        else:
            self._consecutive_flag_steps = 0

        self._last_answer_wrong = bool(
            self._rng.random() < max(0.2, 1.0 - self._state.knowledge)
        )

        reward = compute_reward(
            self._prev_state,
            action,
            self._state,
            persistent_flag=self._persistent_frustration_flag,
            last_answer_wrong=self._last_answer_wrong,
        )
        self._last_reward = reward
        self._cumulative_reward += reward
        self._reward_history.append(reward)
        if len(self._reward_history) > 60:
            self._reward_history.pop(0)

        for k in self._cooldowns:
            if self._cooldowns[k] > 0:
                self._cooldowns[k] -= 1
        if action in COOLDOWNS:
            self._cooldowns[action] = COOLDOWNS[action]

        self._step_count += 1

        terminated = False
        truncated = False
        dropout = False

        if self._state.knowledge >= 0.95:
            terminated = True
        elif (
            self._persistent_frustration_flag
            and self._consecutive_flag_steps >= config.PERSISTENT_FLAG_DROPOUT_STEPS
        ):
            terminated = True
            dropout = True
        elif self._step_count >= self.max_episode_steps:
            truncated = True

        info = {
            "action_masks": self.get_action_mask(),
            "persistent_frustration_flag": self._persistent_frustration_flag,
            "persistent_flag": self._persistent_frustration_flag,
            "emotion": ID_TO_EMOTION[self._state.emotion_id],
            "emotion_name": ID_TO_EMOTION[self._state.emotion_id],
            "emotion_id": self._state.emotion_id,
            "action_name": ID_TO_ACTION[action],
            "knowledge": self._state.knowledge,
            "engagement": self._state.engagement,
            "frustration": self._state.frustration,
            "confusion": self._state.confusion,
            "boredom": self._state.boredom,
            "fer_confidence": conf,
            "last_answer_wrong": self._last_answer_wrong,
            "last_answer_correct": not self._last_answer_wrong,
            "explainer_reason": self._explainer_reason,
            "terminated": terminated,
            "truncated": truncated,
            "dropout": dropout,
        }
        return self._obs(), float(reward), terminated, truncated, info

    def set_algorithm_name(self, name: str) -> None:
        self._algorithm_name = name

    def set_explainer_reason(self, reason: str) -> None:
        self._explainer_reason = reason

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode is None:
            return None

        try:
            import pygame
        except ImportError:
            return None

        W, H = 660, 580
        try:
            if self._screen is None:
                pygame.init()
                if self.render_mode == "human":
                    self._screen = pygame.display.set_mode((W, H))
                    pygame.display.set_caption("Adaptive Tutor")
                else:
                    self._screen = pygame.Surface((W, H))
                self._clock = pygame.time.Clock()
                self._font = pygame.font.SysFont("monospace", 14)
                self._font_lg = pygame.font.SysFont("monospace", 16, bold=True)
        except pygame.error:
            return None

        screen = self._screen
        screen.fill((20, 22, 30))

        emotion = ID_TO_EMOTION[self._state.emotion_id]
        emotion_colors = {
            "confused": (254, 188, 46),
            "frustrated": (255, 95, 87),
            "bored": (136, 136, 187),
            "engaged": (40, 200, 64),
        }
        em_color = emotion_colors.get(emotion, (220, 220, 230))

        def text(s, x, y, color=(220, 220, 230), big=False):
            f = self._font_lg if big else self._font
            surf = f.render(s, True, color)
            screen.blit(surf, (x, y))

        def bar(label, val, x, y, w=280):
            text(f"{label:<14} {val:.2f}", x, y)
            import pygame as pg
            pg.draw.rect(screen, (50, 55, 70), (x, y + 18, w, 12))
            fill = int(w * np.clip(val, 0, 1))
            pg.draw.rect(screen, (120, 90, 220), (x, y + 18, fill, 12))

        text(
            f"{self._algorithm_name}  |  Ep {self._episode_count}  |  Step {self._step_count}",
            12, 8, big=True,
        )
        text(f"Emotion: {emotion}", 12, 36, em_color)
        text(f"Confidence: {self._fer.confidence:.2f}", 12, 56)

        y0 = 90
        for i, (lbl, val) in enumerate([
            ("knowledge", self._state.knowledge),
            ("engagement", self._state.engagement),
            ("frustration", self._state.frustration),
            ("confusion", self._state.confusion),
            ("boredom", self._state.boredom),
        ]):
            bar(lbl, val, 12, y0 + i * 36)

        text("Actions:", 12, 280)
        mask = self.get_action_mask()
        chip_x = 12
        import pygame as pg
        for i, name in enumerate(ACTIONS):
            short = name[:8]
            if i == self._last_action:
                color = (42, 31, 90)
                border = (124, 106, 247)
            elif mask[i] == 0:
                color = (40, 40, 45)
                border = color
            else:
                color = (60, 65, 80)
                border = color
            pg.draw.rect(screen, color, (chip_x, 300, 58, 22), border_radius=4)
            if i == self._last_action:
                pg.draw.rect(screen, border, (chip_x, 300, 58, 22), width=2, border_radius=4)
            surf = self._font.render(short, True, (230, 230, 240))
            screen.blit(surf, (chip_x + 4, 304))
            chip_x += 62
            if chip_x > 600:
                chip_x = 12

        text(f"Step reward: {self._last_reward:+.3f}", 12, 360)
        text(f"Cumulative:  {self._cumulative_reward:.3f}", 200, 360)
        text(f"Episode:     {self._episode_count}", 400, 360)

        reason = self._explainer_reason[:80]
        text(f"Explainer: {reason}", 12, 390, (160, 180, 200))

        if self._reward_history:
            pg.draw.rect(screen, (35, 38, 50), (12, 420, 636, 140))
            pts = self._reward_history
            mx = max(abs(max(pts)), abs(min(pts)), 0.01)
            for j in range(1, len(pts)):
                x1 = 12 + int((j - 1) * 630 / max(len(pts) - 1, 1))
                x2 = 12 + int(j * 630 / max(len(pts) - 1, 1))
                y1 = 500 - int(pts[j - 1] / mx * 50)
                y2 = 500 - int(pts[j] / mx * 50)
                pg.draw.line(screen, (120, 200, 140), (x1, y1), (x2, y2), 2)

        if self.render_mode == "human":
            pygame.display.flip()
            self._clock.tick(self.metadata["render_fps"])
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._should_quit = True

        if self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(screen)), axes=(1, 0, 2)
            )
        return None

    def close(self) -> None:
        if self._screen is not None:
            try:
                import pygame
                pygame.display.quit()
                pygame.quit()
            except Exception:
                pass
            self._screen = None


class DQNSafeEnv(gym.Wrapper):
    """Resample masked actions for DQN training (SB3 has no native masking)."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._rng = np.random.default_rng(config.DEFAULT_SEED)

    def step(self, action):
        unwrapped = self.env.unwrapped
        mask = unwrapped.get_action_mask()
        attempts = 0
        while mask[action] == 0 and attempts < 100:
            allowed = np.where(mask == 1)[0]
            if len(allowed) == 0:
                break
            action = int(self._rng.choice(allowed))
            attempts += 1
        return self.env.step(action)


def mask_fn(env: gym.Env) -> np.ndarray:
    """Callback for sb3_contrib ActionMasker."""
    return env.unwrapped.get_action_mask()
