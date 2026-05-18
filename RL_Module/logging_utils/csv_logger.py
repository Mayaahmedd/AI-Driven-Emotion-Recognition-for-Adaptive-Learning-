"""
CSV logging: step log, episode log, summary (spec Phase 4).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

from RL_Module import config
from RL_Module.mdp_definition import ID_TO_ACTION, ID_TO_EMOTION

STEP_COLUMNS = [
    "step",
    "episode",
    "algorithm",
    "seed",
    "action_id",
    "action_name",
    "reward",
    "cumulative_reward",
    "knowledge",
    "engagement",
    "frustration",
    "confusion",
    "boredom",
    "emotion_id",
    "emotion_name",
    "persistent_flag",
    "terminated",
    "truncated",
    "explainer_reason",
]

EPISODE_COLUMNS = [
    "episode",
    "algorithm",
    "seed",
    "total_reward",
    "learning_gain",
    "learning_gain_normalized",
    "success",
    "dropout",
    "steps_taken",
    "adaptation_accuracy",
    "convergence_flag",
] + [f"action_{i}_count" for i in range(10)]


class CSVLogger:
    """Unified logger for step + episode (spec smoke test name)."""

    def __init__(
        self,
        log_dir: Optional[Path] = None,
        algorithm: str = "default",
        seed: int = 0,
        run_name: Optional[str] = None,
    ):
        self.algorithm = algorithm
        self.seed = seed
        self.run_name = run_name or f"{algorithm}_seed{seed}"
        base = log_dir or config.LOGS_DIR
        self.step_logger = StepLogger(algorithm, seed, base)
        self.episode_logger = EpisodeLogger(algorithm, seed, base)

    def log_episode(
        self,
        episode: int,
        total_reward: float,
        k_start: float,
        k_final: float,
        success: bool,
        dropout: bool,
        steps: int,
        adaptation_accuracy: float,
        convergence_flag: bool,
        action_counts: List[int],
    ) -> None:
        lg = k_final - k_start
        lg_norm = (k_final - k_start) / (1.0 - k_start + 1e-8)
        counts_dict = {i: int(action_counts[i]) for i in range(len(action_counts))}
        self.episode_logger.log_episode({
            "episode": episode,
            "algorithm": self.algorithm,
            "seed": self.seed,
            "total_reward": total_reward,
            "learning_gain": round(lg, 4),
            "learning_gain_normalized": round(lg_norm, 4),
            "success": int(bool(success)),
            "dropout": int(bool(dropout)),
            "steps_taken": int(steps),
            "adaptation_accuracy": round(float(adaptation_accuracy), 4),
            "convergence_flag": int(bool(convergence_flag)),
            "action_counts": counts_dict,
        })

    def log_step(
        self,
        step: int,
        episode: int,
        action: int,
        reward: float,
        cumulative_reward: float,
        obs: Any,
        info: Dict[str, Any],
        explainer_reason: str = "",
    ) -> None:
        emotion_id = int(round(float(obs[5]))) if len(obs) > 5 else 3
        emotion_id = max(0, min(3, emotion_id))
        self.step_logger.log_step({
            "step": step,
            "episode": episode,
            "algorithm": self.algorithm,
            "seed": self.seed,
            "action_id": action,
            "action_name": ID_TO_ACTION[action],
            "reward": round(reward, 4),
            "cumulative_reward": round(cumulative_reward, 4),
            "knowledge": round(float(obs[0]), 4),
            "engagement": round(float(obs[1]), 4),
            "frustration": round(float(obs[2]), 4),
            "confusion": round(float(obs[3]), 4),
            "boredom": round(float(obs[4]), 4),
            "emotion_id": emotion_id,
            "emotion_name": ID_TO_EMOTION[emotion_id],
            "persistent_flag": info.get("persistent_frustration_flag", False),
            "terminated": info.get("terminated", False),
            "truncated": info.get("truncated", False),
            "explainer_reason": explainer_reason or info.get("explainer_reason", ""),
        })

    def close(self) -> None:
        self.step_logger.close()
        self.episode_logger.close()


class StepLogger:
    def __init__(self, algorithm: str, seed: int, log_dir: Optional[Path] = None):
        self.path = (log_dir or config.LOGS_DIR) / f"steps_{algorithm}_seed{seed}.csv"
        self._file = open(self.path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=STEP_COLUMNS)
        self._writer.writeheader()

    def log_step(self, row: Dict[str, Any]) -> None:
        self._writer.writerow({k: row.get(k, "") for k in STEP_COLUMNS})
        self._file.flush()

    def close(self) -> None:
        self._file.close()


class EpisodeLogger:
    def __init__(self, algorithm: str, seed: int, log_dir: Optional[Path] = None):
        self.path = (log_dir or config.LOGS_DIR) / f"episodes_{algorithm}_seed{seed}.csv"
        self._file = open(self.path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=EPISODE_COLUMNS)
        self._writer.writeheader()

    def log_episode(self, row: Dict[str, Any]) -> None:
        out: Dict[str, Any] = {}
        for k in EPISODE_COLUMNS:
            if k.startswith("action_") and k.endswith("_count"):
                idx = int(k.split("_")[1])
                counts = row.get("action_counts", {})
                out[k] = counts.get(idx, counts.get(str(idx), 0))
            else:
                out[k] = row.get(k, "")
        self._writer.writerow(out)
        self._file.flush()

    def close(self) -> None:
        self._file.close()


class SummaryWriter:
    SUMMARY_COLUMNS = [
        "algorithm",
        "seed",
        "final_reward_mean",
        "final_reward_std",
        "ci_lower",
        "ci_upper",
        "convergence_episode",
        "success_rate",
        "learning_gain_mean",
        "dropout_rate",
        "adaptation_accuracy_mean",
        "delta_reward",
        "delta_success",
        "delta_learning_gain",
    ]

    def __init__(self, path: Optional[Path] = None, append: bool = True):
        self.path = path or (config.LOGS_DIR / "summary.csv")
        self._rows: List[Dict[str, Any]] = []
        self._append = append
        if self.path.exists() and not append:
            self.path.unlink()

    def add_row(self, row: Dict[str, Any]) -> None:
        self._rows.append(row)

    def write(self) -> None:
        write_header = not self.path.exists() or not self._append
        mode = "a" if self._append and self.path.exists() else "w"
        with open(self.path, mode, newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.SUMMARY_COLUMNS, extrasaction="ignore")
            if write_header:
                w.writeheader()
            for row in self._rows:
                w.writerow({k: row.get(k, "") for k in self.SUMMARY_COLUMNS})
