"""Concrete curriculum providers.

Public surface:

    from adaptive_tutor.memory.providers import (
        BaseCurriculumProvider, ConceptNode, CurriculumIntegrityError,
        TeacherCurriculumProvider,
        DatasetCurriculumProvider,
    )

The previous ``HybridCurriculumProvider`` was removed: dataset statistics
are now looked up directly by the state builder and the simulator, with
no extra merging class. See ADR-006 for the rationale.
"""

from adaptive_tutor.memory.providers.base import (
    BaseCurriculumProvider,
    ConceptNode,
    CurriculumIntegrityError,
)
from adaptive_tutor.memory.providers.dataset_provider import DatasetCurriculumProvider
from adaptive_tutor.memory.providers.teacher_provider import TeacherCurriculumProvider

__all__ = [
    "BaseCurriculumProvider",
    "ConceptNode",
    "CurriculumIntegrityError",
    "DatasetCurriculumProvider",
    "TeacherCurriculumProvider",
]
