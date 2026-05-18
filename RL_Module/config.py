"""
Central configuration - seeds, hyperparams, paths, flags.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np

# Paths
MODULE_ROOT = Path(__file__).resolve().parent
LOGS_DIR = MODULE_ROOT / "logs"
MODELS_DIR = MODULE_ROOT / "models"
VIDEOS_DIR = MODULE_ROOT / "videos"
LOG_DIR = "logs/"
MODEL_DIR = "models/"

LOGS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

# Experiment
SEEDS = [42, 0, 1, 7, 13, 21, 99, 100, 200, 314]
DEFAULT_SEED = 42
TOTAL_TIMESTEPS = 50_000
TRAINING_TIMESTEPS = TOTAL_TIMESTEPS
EVAL_EPISODES = 1_000
MAX_EPISODE_STEPS = 50
N_STUDENTS = 10_000
POPULATION_SIZE = N_STUDENTS

# FER
USE_REAL_FER = False
FER_CONFIDENCE_THRESHOLD = 0.5
FER_PERSISTENT_EMOTION_STEPS = 3

# Rendering
RENDER_MODE = None  # None | "human" | "rgb_array"

# PPO (MaskablePPO)
PPO_LEARNING_RATE = 3e-4
PPO_GAMMA = 0.99
PPO_CLIP_RANGE = 0.2
PPO_N_STEPS = 2048
PPO_BATCH_SIZE = 64
PPO_ENT_COEF = 0.01
PPO_NET_ARCH = [256, 256, 256]

# DQN
DQN_LEARNING_RATE = 1e-3
DQN_BATCH_SIZE = 32
DQN_GAMMA = 0.99
DQN_TRAIN_FREQ = 4
DQN_TARGET_UPDATE_INTERVAL = 500
DQN_EXPLORATION_FRACTION = 0.1
DQN_EXPLORATION_FINAL_EPS = 0.05
DQN_NET_ARCH = [64, 64]

# Hybrid
HYBRID_PPO_WEIGHT = 0.6
HYBRID_DQN_WEIGHT = 0.4

# Bandit
BANDIT_ALPHA = 1.0
BANDIT_EMOTION_DELTA_THRESHOLD = 0.3

# Algorithms
ALGORITHMS = ("PPO", "DQN", "PPO_DQN", "Bandit_DQN", "Rule", "Random")

# Action and emotion names (spec Phase 6)
ACTION_NAMES = [
    "easier_question",
    "harder_question",
    "give_hint",
    "motivational_message",
    "scaffold",
    "change_pacing",
    "reflection_prompt",
    "strategy_guidance",
    "autonomy_support",
    "explanation",
]

EMOTION_NAMES = ["confused", "bored", "frustrated", "engaged"]

# Best actions per emotion_id (spec Phase 6)
BEST_ACTION_MAP = {
    0: [2, 9, 7],   # confused -> hint, explanation, strategy
    3: [1, 4, 6],   # engaged -> harder, scaffold, reflection
    2: [5, 0, 8],   # frustrated -> pacing, easier, autonomy
    1: [3, 8, 4],   # bored -> motivation, autonomy, scaffold
}

# Dropout: consecutive steps with persistent_flag before terminated
PERSISTENT_FLAG_DROPOUT_STEPS = 5


def set_all_seeds(seed: int) -> None:
    """Set all random seeds before train/eval (spec Phase 6)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
