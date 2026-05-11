"""Teacher lookups for PPO segments (no hybrid provider)."""

from __future__ import annotations

from adaptive_tutor.memory.providers.base import BaseCurriculumProvider


def dataset_skill_slug_for_concept(
    teacher: BaseCurriculumProvider, concept_id: str
) -> str:
    """First skill under ``concept_id`` in teacher YAML, else ``concept_id``.

    Keeps Phase 7 strict-dataset masks aligned with ASSISTments slugs.
    """
    for c in teacher.get_concepts():
        if str(c["concept_id"]) == concept_id:
            skills = c.get("skills") or []
            if skills:
                return str(skills[0])
            return str(concept_id)
    return str(concept_id)
