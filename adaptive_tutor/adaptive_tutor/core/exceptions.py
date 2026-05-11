"""Custom exception hierarchy for the adaptive_tutor package.

We use a small hierarchy with a single root so callers can::

    try:
        ...
    except AdaptiveTutorError as e:
        ...

without listing every subtype. Each subtype carries an explicit semantic
meaning so test failures and bug reports are easy to triage.

Adding new exception types is fine; renaming or removing existing ones is
a breaking change and must bump the package MAJOR version.
"""

from __future__ import annotations


class AdaptiveTutorError(Exception):
    """Root for every error raised by this package."""


class ContractError(AdaptiveTutorError):
    """An object claimed a protocol it does not actually satisfy.

    Examples: a ``Policy`` returning a ``CompositeAction`` whose ``meso``
    field is ``None``; a ``CurriculumProvider`` returning duplicate
    concept ids; a ``Replay`` returning a batch of wrong size.
    """


class MaskError(AdaptiveTutorError):
    """An action chosen despite being masked, or a malformed mask shape.

    Raised by safety / action-mask code paths (Phase 7). Action-masking
    bugs are *critical* — they imply the agent picked an action it should
    have been structurally prevented from picking, which would invalidate
    safety claims of the system.
    """


class SchemaError(AdaptiveTutorError):
    """A persisted artifact does not match its declared schema version.

    Raised by the explanation/trace readers when they encounter an
    unsupported MAJOR version (see ADR-005).
    """
