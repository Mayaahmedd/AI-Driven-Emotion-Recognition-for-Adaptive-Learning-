"""Rollout-level metrics from synthetic simulator trajectories (Phase 11)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def compute_frustration_rate(trajectory: Sequence[Mapping[str, object]]) -> float:
    """Fraction of steps with frustration proxy above a fixed thesis threshold."""
    if not trajectory:
        return 0.0
    high = 0
    for step in trajectory:
        f = float(step.get("frustrated", 0.0))
        if f >= 0.55:
            high += 1
    return high / len(trajectory)


def compute_mastery_gain(trajectory: Sequence[Mapping[str, object]]) -> float:
    """End minus start mastery along the logged trajectory."""
    if not trajectory:
        return 0.0
    m0 = float(trajectory[0].get("mastery", 0.0))
    m1 = float(trajectory[-1].get("mastery", m0))
    return m1 - m0


def compute_hint_efficiency(trajectory: Sequence[Mapping[str, object]]) -> float:
    """Correct responses per hint use (hints are capped at denominator +1)."""
    if not trajectory:
        return 0.0
    hints = sum(1 for t in trajectory if bool(t.get("hint", False)))
    correct = sum(1 for t in trajectory if int(t.get("correct", 0)) == 1)
    return float(correct) / float(hints + 1)


def compute_learning_rate(mastery_curve: list[float] | Sequence[float]) -> float:
    """Average mastery rise per step (stable, low-noise thesis metric)."""
    if len(mastery_curve) < 2:
        return 0.0
    lo = float(mastery_curve[0])
    hi = float(mastery_curve[-1])
    return (hi - lo) / (len(mastery_curve) - 1)


def compute_episode_metrics(
    trajectory: Sequence[Mapping[str, object]],
    *,
    mastery_threshold: float = 0.85,
) -> dict[str, float]:
    """Aggregate scalar summary for one episode."""
    if not trajectory:
        return {
            "mastery_gain": 0.0,
            "learning_rate": 0.0,
            "frustration_rate": 0.0,
            "hint_efficiency": 0.0,
            "episode_length": 0.0,
            "success_rate": 0.0,
            "convergence_steps": 0.0,
            "total_reward": 0.0,
        }
    curve = [float(s.get("mastery", 0.0)) for s in trajectory]
    conv = 0.0
    for i, m in enumerate(curve):
        if m >= mastery_threshold:
            conv = float(i + 1)
            break
    if conv == 0.0:
        conv = float(len(trajectory))
    success = sum(1 for t in trajectory if int(t.get("correct", 0)) == 1)
    return {
        "mastery_gain": compute_mastery_gain(trajectory),
        "learning_rate": compute_learning_rate(curve),
        "frustration_rate": compute_frustration_rate(trajectory),
        "hint_efficiency": compute_hint_efficiency(trajectory),
        "episode_length": float(len(trajectory)),
        "success_rate": success / len(trajectory),
        "convergence_steps": conv,
        "total_reward": sum(float(t.get("reward", 0.0)) for t in trajectory),
    }
