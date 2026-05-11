"""ASSISTments offline statistics via :class:`DatasetCurriculumProvider` (Phase 11)."""

from __future__ import annotations

from collections.abc import Mapping

from adaptive_tutor.memory.providers.dataset_provider import DatasetCurriculumProvider


def compute_dataset_metrics(provider: DatasetCurriculumProvider) -> dict[str, object]:
    """Deterministic cohort summary + per-skill breakdown."""
    stats = provider.all_stats()

    total_n = sum(st.n_records for st in stats.values())
    if total_n <= 0:
        return {
            "baseline_correctness": 0.0,
            "mean_historical_hint_rate": 0.0,
            "per_skill": {},
            "difficulty_adjusted_success": 0.0,
        }

    baseline = (
        sum(st.mean_correctness * st.n_records for st in stats.values()) / total_n
    )
    hint_rate = (
        sum(st.mean_hint_count * st.n_records for st in stats.values()) / total_n
    )

    per_skill: dict[str, dict[str, float]] = {}
    adj_num = 0.0
    for slug, st in sorted(stats.items()):
        difficulty = max(0.05, min(0.95, 1.0 - st.mean_correctness))
        adjusted = st.mean_correctness / (0.15 + difficulty)
        per_skill[slug] = {
            "n_records": float(st.n_records),
            "mean_correctness": float(st.mean_correctness),
            "mean_hint_count": float(st.mean_hint_count),
            "mean_attempts": float(st.mean_attempts),
            "difficulty_proxy": float(difficulty),
            "difficulty_adjusted_success": float(adjusted),
        }
        adj_num += adjusted * st.n_records

    return {
        "baseline_correctness": float(baseline),
        "mean_historical_hint_rate": float(hint_rate),
        "per_skill": per_skill,
        "difficulty_adjusted_success": float(adj_num / total_n),
        "n_rows": float(total_n),
    }
