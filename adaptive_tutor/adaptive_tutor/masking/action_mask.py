"""Curriculum + dataset aware action allow-list (Phase 7).

The runtime uses ASSISTMENTS **string** labels (:data:`ASSISTMENTS_ACTIONS`).
``CompositeAction`` encoding is optional and done only when writing
:class:`~adaptive_tutor.state.state.Transition`; this module never invents
new action types.
"""

from __future__ import annotations

from collections.abc import Iterable

from adaptive_tutor.memory.providers.base import BaseCurriculumProvider
from adaptive_tutor.memory.providers.dataset_provider import ASSISTMENTS_ACTIONS


def concept_in_teacher_graph(teacher: BaseCurriculumProvider, concept_slug: str) -> bool:
    """Return True if ``concept_slug`` is a loaded teacher concept id."""
    try:
        teacher.concept_index(concept_slug)
    except KeyError:
        return False
    return True


def mask_by_curriculum_and_dataset(
    actions: tuple[str, ...],
    *,
    concept_slug: str,
    teacher: BaseCurriculumProvider | None,
    dataset_slugs: frozenset[str] | None,
    dataset_skill_slug: str | None,
    strict_dataset: bool,
) -> tuple[frozenset[str], tuple[str, ...]]:
    """Return allowed actions and human-readable drop reasons.

    Rules (deterministic, in order):

    1. **Unknown vocabulary** - anything not in ``ASSISTMENTS_ACTIONS`` is dropped.
    2. **Teacher membership** - if ``teacher`` is provided, ``concept_slug`` must
       exist; otherwise only ``retry_current_skill`` and ``encouragement`` remain
       (minimal safe tutoring).
    3. **Strict dataset** - if ``strict_dataset`` and we have ``dataset_slugs``,
       and ``dataset_skill_slug not in dataset_slugs``, remove escalation actions
       ``harder_problem`` and ``advance_to_next_skill`` (no empirical support).

    Parameters
    ----------
    dataset_skill_slug:
        Slug used for ASSISTments stats join (may match ``concept_slug``).
    """
    allowed = frozenset(a for a in actions if a in ASSISTMENTS_ACTIONS)
    reasons: list[str] = []

    if teacher is not None:
        if not concept_in_teacher_graph(teacher, concept_slug):
            safe = frozenset({"retry_current_skill", "encouragement"})
            dropped = allowed - safe
            allowed = allowed & safe
            if dropped:
                reasons.append(f"concept_not_in_teacher_curriculum:{concept_slug}")

    if strict_dataset and dataset_slugs is not None and dataset_skill_slug is not None:
        if dataset_skill_slug not in dataset_slugs:
            block = frozenset({"harder_problem", "advance_to_next_skill"})
            dropped = allowed & block
            allowed -= block
            if dropped:
                reasons.append(
                    f"strict_dataset_no_stats:{dataset_skill_slug!r} "
                    f"(removed {sorted(dropped)})"
                )

    return allowed, tuple(reasons)


def as_composite_actions(action_strings: Iterable[str]):
    """Optional: map allowed strings to :class:`~adaptive_tutor.core.types.CompositeAction`."""
    from adaptive_tutor.integration.actions import assistments_action_to_composite

    return [assistments_action_to_composite(a) for a in sorted(action_strings)]
