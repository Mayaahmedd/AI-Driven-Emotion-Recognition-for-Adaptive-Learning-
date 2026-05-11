"""adaptive_tutor.memory - curriculum memory and provider implementations.

This package owns the *knowledge* side of the system: concepts, skills,
prerequisites, instructional content, and the validation logic that
keeps the curriculum graph well-formed (no duplicate ids, no cycles,
valid prerequisite references).

It is intentionally separate from any RL or simulator code. The RL
stack reads the curriculum graph through the abstract Protocol in
``adaptive_tutor.core.protocols.CurriculumProvider``; it never imports
concrete provider classes.
"""

from adaptive_tutor.memory.providers.base import (
    BaseCurriculumProvider,
    ConceptNode,
    CurriculumIntegrityError,
)

__all__ = ["BaseCurriculumProvider", "ConceptNode", "CurriculumIntegrityError"]
