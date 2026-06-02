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
LOG_DIR = "logs/"  # DEFAULT: legacy string path for scripts expecting relative logs/
MODEL_DIR = "models/"  # DEFAULT: legacy string path

LOGS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

# Experiment
SEEDS = [42, 0, 1, 7, 13, 21, 99, 100, 200, 314]  # DEFAULT: 10 seeds for CI reporting
DEFAULT_SEED = 42  # DEFAULT: reproducible baseline
TOTAL_TIMESTEPS = 50_000  # DEFAULT: full training budget per algorithm/seed
TRAINING_TIMESTEPS = TOTAL_TIMESTEPS
EVAL_EPISODES = 1_000  # DEFAULT: post-training evaluation episodes
MAX_EPISODE_STEPS = 50  # DEFAULT: matches MDP horizon in spec
N_STUDENTS = 10_000  # DEFAULT: synthetic population size
POPULATION_SIZE = N_STUDENTS

# FER
USE_REAL_FER = False  # DEFAULT: simulation mode until FER_Module is wired
FER_CONFIDENCE_THRESHOLD = 0.5  # DEFAULT: minimum confidence to accept live FER label
FER_PERSISTENT_EMOTION_STEPS = 3  # SOURCE: MaTHiSiS Paper 9 - 3-step persistence window

# Rendering
RENDER_MODE = None  # DEFAULT: None | "human" | "rgb_array"

# DQN
DQN_LEARNING_RATE = 5e-4  #  1e-4 DEFAULT: SB3 DQN default
DQN_BATCH_SIZE = 64  # DEFAULT: thesis final DQN (hyperparameter study)
DQN_BUFFER_SIZE = 100_000  # DEFAULT: replay capacity (SB3 default is 1e6)
DQN_GAMMA = 0.99  # DEFAULT: discounted return
DQN_TRAIN_FREQ = 4  # DEFAULT: gradient steps per env step
DQN_TARGET_UPDATE_INTERVAL = 2000  # DEFAULT: target network sync period
DQN_EXPLORATION_FRACTION = 0.3  # DEFAULT: epsilon decay over first 10% of training
DQN_EXPLORATION_FINAL_EPS = 0.05  # DEFAULT: residual exploration
DQN_NET_ARCH = [64, 64]  # DEFAULT: smaller net for reactive Q-learning

# Algorithms (thesis tutors: DQN, Rule/ERT, Random)
ALGORITHMS = ("DQN", "Rule", "Random")

# Action and emotion names
ACTION_NAMES = [
    "hint",
    "scaffold",
    "encouragement",
    "simplify_problem",
    "harder_problem",
    "break",
    "explanation",
    "no_action",
]

EMOTION_NAMES = ["confused", "bored", "frustrated", "engaged"]

# BEST_ACTION_MAP: used ONLY in evaluation/metrics.py and explainability/explainer.py
# as a post-training measurement tool. NOT imported by reward_function.py.
# NOT used during training in any way.
BEST_ACTION_MAP = {
    0: [0, 6, 1],   # confused -> hint, explanation, scaffold
    3: [4, 1],      # engaged -> harder_problem, scaffold
    2: [5, 3, 2],   # frustrated -> break, simplify_problem, encouragement
    1: [2, 4],      # bored -> encouragement, harder_problem
}

# Dropout: consecutive steps with persistent_flag before terminated
PERSISTENT_FLAG_DROPOUT_STEPS = 5  # SOURCE: MaTHiSiS Paper 9 - persistent affect dropout

# Masking / persistent-frustration thresholds live in mdp_definition.py (single source
# of truth). Re-exported here for ergonomic config.X access. sensitivity_analysis.py
# patches mdp_definition, config, and student_env module attributes together.
from RL_Module.mdp_definition import (
    DIFFICULTY_INIT_HIGH,
    DIFFICULTY_INIT_LOW,
    DIFFICULTY_STEP,
    EMOTION_NOISE_STD,
    FRUSTRATION_PERSISTENT_ON,
    FRUSTRATION_PERSISTENT_OFF,
    FRUSTRATION_PERSISTENT_STEPS,
    FRUSTRATION_BLOCK_HARDER,
    GAIN_CORRECT_FACTOR,
    GAIN_INCORRECT_FACTOR,
    IRT_ALPHA,
    IRT_BETA,
    LAMBDA_BOREDOM,
    LAMBDA_CONFUSION,
    LAMBDA_ENGAGEMENT,
    LAMBDA_FRUSTRATION,
    MISMATCH_HIGH,
    MISMATCH_LOW,
)

# Simulator transition hyperparameters (literature-inspired defaults; tunable via SIMULATOR_PARAMS)
# Structure is theory-backed; magnitudes are calibrated via sensitivity analysis.

# Reward presets: r_t = wk·Δk + we·e - wf·f - wb·b - wc·c
REWARD_PRESETS: dict = {
    "EQUAL": {"wk": 0.20, "we": 0.20, "wf": 0.20, "wb": 0.20, "wc": 0.20},
    "LEARNING_FOCUSED": {"wk": 0.70, "we": 0.15, "wf": 0.05, "wb": 0.05, "wc": 0.05},
    "AFFECT_FOCUSED": {"wk": 0.30, "we": 0.25, "wf": 0.20, "wb": 0.15, "wc": 0.10},
    "BALANCED": {"wk": 0.50, "we": 0.20, "wf": 0.15, "wb": 0.06, "wc": 0.09},
    "BALANCED_2": {"wk": 0.35, "we": 0.20, "wf": 0.15, "wb": 0.15, "wc": 0.15},
}
REWARD_PRESET = "BALANCED_2"  # thesis final: reward weight ablation winner
REWARD_WEIGHTS: dict = dict(REWARD_PRESETS[REWARD_PRESET])

# no_action opportunity cost during struggle (flow states unpenalized)
NO_ACTION_STRUGGLE_PENALTY: float = 0.02
NO_ACTION_CONFUSION_THRESH: float = 0.40
NO_ACTION_FRUSTRATION_THRESH: float = 0.40

# Optional legacy reward terms (off by default for clean multi-objective formula)
USE_ZONE_BONUS = False
USE_WRONG_ANSWER_PENALTY = False

# Reward internal thresholds
# SOURCE: D'Mello & Graesser (2012) - frustration above moderate disrupts learning
FRUSTRATION_HIGH_THRESHOLD = 0.6  # DEFAULT: upper tertile
FRUSTRATION_PENALTY_HIGH = 0.5  # magnitude subtracted via W * frustration_term
PERSISTENT_PENALTY = 1.0  # magnitude when persistent_flag is True
WRONG_ANSWER_PENALTY = 0.5  # magnitude subtracted when last answer wrong

# Optimal zone (flow) - SOURCE: Csikszentmihalyi (1990)
ZONE_BONUS = 1.0
ZONE_MIN_KNOWLEDGE = 0.5  # DEFAULT: midpoint
ZONE_MIN_ENGAGEMENT = 0.6  # DEFAULT: above moderate engagement
ZONE_MAX_FRUSTRATION = 0.3
ZONE_MAX_CONFUSION = 0.4
ZONE_MAX_BOREDOM = 0.3

# Sensitivity analysis switch - True only during sensitivity_analysis.py runs
USE_SENSITIVITY_WEIGHTS = False
SENSITIVITY_WEIGHTS: dict = {}

# Patched at runtime by sensitivity_analysis.py (transition params, not reward weights)
SIMULATOR_PARAMS: dict = {}


def get_simulator_param(name: str, default: float | None = None) -> float:
    """Read tunable simulator coefficient (patch dict overrides module defaults)."""
    if name in SIMULATOR_PARAMS:
        return float(SIMULATOR_PARAMS[name])
    if default is not None:
        return float(default)
    return float(globals().get(name, 0.0))


def get_reward_weights() -> dict:
    """Active reward weights: sensitivity override > preset > BALANCED."""
    if USE_SENSITIVITY_WEIGHTS and SENSITIVITY_WEIGHTS:
        base = dict(REWARD_PRESETS[REWARD_PRESET])
        base.update(
            {
                k: float(v)
                for k, v in SENSITIVITY_WEIGHTS.items()
                if k in ("wk", "we", "wf", "wb", "wc")
            }
        )
        return base
    return dict(REWARD_WEIGHTS)


def set_all_seeds(seed: int) -> None:
    """Set all random seeds before train/eval."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
