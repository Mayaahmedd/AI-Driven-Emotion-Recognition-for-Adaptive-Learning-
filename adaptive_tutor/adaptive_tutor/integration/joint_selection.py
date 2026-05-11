"""Runtime-only join between teacher curriculum and ASSISTments statistics.

No merged provider: callers pick one :class:`~adaptive_tutor.memory.providers.base.ConceptNode`
from the teacher graph and attach :class:`~adaptive_tutor.memory.providers.dataset_provider.SkillStats`
when a skill slug appears in both sources.
"""

from __future__ import annotations

from dataclasses import dataclass

from adaptive_tutor.core.exceptions import AdaptiveTutorError
from adaptive_tutor.memory.providers.base import BaseCurriculumProvider
from adaptive_tutor.memory.providers.dataset_provider import DatasetCurriculumProvider, SkillStats


@dataclass(frozen=True, slots=True)
class JointConceptSelection:
    """One teacher concept plus the dataset slice used for calibration / state stats."""

    concept_id: str
    dataset_skill_slug: str
    concept_index: int
    skill_stats: SkillStats


def collect_teacher_skill_slugs(teacher: BaseCurriculumProvider) -> set[str]:
    """All ``concept_id`` values and entries in each concept's ``skills`` list."""
    out: set[str] = set()
    for c in teacher.get_concepts():
        out.add(str(c["concept_id"]))
        for s in c.get("skills") or []:
            out.add(str(s))
    return out


def select_first_joint_concept(
    teacher: BaseCurriculumProvider,
    dataset: DatasetCurriculumProvider,
) -> JointConceptSelection:
    """First teacher concept (sorted provider order) that overlaps any dataset skill slug.

    Raises
    ------
    AdaptiveTutorError
        If no slug appears in both the teacher graph and ``dataset.all_stats()``.
    """
    dataset_slugs = set(dataset.all_stats().keys())
    for c in teacher.get_concepts():
        cid = str(c["concept_id"])
        candidates = [cid, *[str(s) for s in (c.get("skills") or [])]]
        for slug in candidates:
            if slug in dataset_slugs:
                idx = teacher.concept_index(cid)
                st = dataset.stats(slug)
                return JointConceptSelection(cid, slug, idx, st)
    raise AdaptiveTutorError(
        "no overlapping skill/concept slug between teacher curriculum and ASSISTments dataset; "
        "align YAML ``concept_id`` / ``skills`` with slugified CSV ``skill`` labels"
    )
