from dataclasses import dataclass


EMOTIONS = ("engaged", "confused", "frustrated", "bored")

EXPLANATION_ACTIONS = (
    "keep_pace",
    "slow_down",
    "give_micro_example",
    "motivate",
    "quick_check",
)

PRACTICE_ACTIONS = (
    "question_easy",
    "question_medium",
    "question_hard",
    "question_review",
)

ARCHETYPES = (
    "balanced",
    "anxious",
    "boredom_prone",
    "struggling",
)


@dataclass(frozen=True)
class SimConfig:
    steps_per_episode: int = 30
    explanation_steps: int = 12
    base_correct_prob: float = 0.65
    train_archetypes: tuple[str, ...] = ("balanced", "anxious", "boredom_prone", "struggling")
    eval_archetypes: tuple[str, ...] = ("balanced", "anxious", "boredom_prone", "struggling")


@dataclass(frozen=True)
class DQNConfig:
    lr: float = 1e-3
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    replay_size: int = 5000
    batch_size: int = 64
    target_update_steps: int = 100
