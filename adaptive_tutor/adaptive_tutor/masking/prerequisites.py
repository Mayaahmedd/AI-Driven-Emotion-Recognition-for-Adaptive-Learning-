"""Prerequisite mastery gate for escalation actions (Phase 7).

Uses the teacher DAG only. Mastery per concept is supplied explicitly
(``mastery_by_slug``): the simulator updates the **current** concept each
step; prerequisite slugs retain their last recorded value until the
orchestrator changes them.
"""

from __future__ import annotations

from collections.abc import Mapping

from adaptive_tutor.memory.providers.base import BaseCurriculumProvider


def prerequisites_satisfied(
    teacher: BaseCurriculumProvider,
    concept_slug: str,
    mastery_by_slug: Mapping[str, float],
    *,
    threshold: float,
) -> bool:
    """All immediate prerequisites of ``concept_slug`` have mastery >= ``threshold``."""
    prereqs: list[str] = []
    for c in teacher.get_concepts():
        if str(c["concept_id"]) == concept_slug:
            prereqs = [str(p) for p in (c.get("prerequisites") or [])]
            break
    if not prereqs:
        return True
    return all(float(mastery_by_slug.get(p, 0.0)) >= threshold for p in prereqs)


def filter_by_prerequisites(
    actions: frozenset[str],
    *,
    teacher: BaseCurriculumProvider | None,
    concept_slug: str,
    mastery_by_slug: Mapping[str, float],
    threshold: float,
) -> tuple[frozenset[str], tuple[str, ...]]:
    """Remove escalation actions until prerequisites are mastered.

    When prerequisites are **not** satisfied, drop ``harder_problem`` and
    ``advance_to_next_skill`` (cannot increase challenge or skip ahead).

    Unknown prerequisite mastery defaults to ``0.0`` (conservative).
    """
    if teacher is None:
        return actions, ()

    if prerequisites_satisfied(teacher, concept_slug, mastery_by_slug, threshold=threshold):
        return actions, ()

    block = frozenset({"harder_problem", "advance_to_next_skill"})
    dropped = sorted(actions & block)
    allowed = actions - block
    if dropped:
        return allowed, (f"prerequisites_not_mastered:blocked_{dropped}",)
    return allowed, ()
