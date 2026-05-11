"""Offline scaffold for ScienceQA -> teacher curriculum YAML.

Status
------
**Scaffold only.** The actual ScienceQA dataset reader is a
``NotImplementedError`` (see :func:`load_raw_scienceqa`). What this
file *does* implement is the full surrounding pipeline:

1. CLI entrypoint with deterministic arguments.
2. The inference-policy contract from ADR-006 section 2.4.
3. The validation step (snake_case ids, DAG, 1-5 prerequisites)
   - re-using :class:`BaseCurriculumProvider`'s validator, not a
   reimplementation.
4. YAML output in the exact shape that
   :class:`TeacherCurriculumProvider` reads.

When the ScienceQA JSON is available, only ``load_raw_scienceqa`` and
``infer_prerequisites`` need to change. Everything else - validation,
serialisation, CLI - is already correct.

Why this scaffold exists
------------------------
The "future work" entry in the planning narrative would otherwise be a
vague paragraph. Having a CLI with the exact rules encoded in a
``ScienceQAExtractionPolicy`` constant turns "future work" into
"future patch": the rules cannot drift because the validator is the
same one the runtime uses.

Prerequisite inference rules (ADR-006 section 2.4)
--------------------------------------------------
Allowed signals - dataset fields ONLY:

* ``skill``        - skill tag string
* ``topic``        - topic tag string
* ``category``     - category tag string
* ``subject``      - subject tag string
* ``grade``        - integer grade level
* ``lecture``      - explanatory text shipped with each question
* ``solution``     - step-by-step solution text

Disallowed signals:

* External knowledge graphs, world knowledge calls, LLM completions
  that depend on knowledge outside the dataset.

This script will assert the policy at runtime by recording which
fields each prerequisite was derived from in the ``provenance`` field
of each :class:`ConceptNode`. The validator (see ``validate``) then
checks that every prerequisite has at least one provenance entry from
the allowed set.

Difficulty heuristic (ADR-006 section 2.4)
------------------------------------------
When the raw record does not provide difficulty, estimate it from:

1. Grade (primary signal).
2. Number of prerequisites.
3. Reasoning complexity (proxied by solution length).

The exact mapping lives in :func:`estimate_difficulty` and is
deliberately simple.

CLI usage
---------

.. code-block:: bash

    python -m adaptive_tutor.tools.extract_scienceqa_curriculum \\
        --input  data/scienceqa/problems.json \\
        --output configs/curriculum/examples/scienceqa_extracted.yaml \\
        --max-prereqs 5

The output file can be loaded immediately by
``TeacherCurriculumProvider``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from adaptive_tutor.core.exceptions import AdaptiveTutorError
from adaptive_tutor.memory.providers.base import (
    MAX_PREREQUISITES,
    BaseCurriculumProvider,
    ConceptNode,
)

_LOG = logging.getLogger(__name__)


# ---- Frozen policy record (rules encoded as data) -------------------------

ALLOWED_INFERENCE_FIELDS: tuple[str, ...] = (
    "skill",
    "topic",
    "category",
    "subject",
    "grade",
    "lecture",
    "solution",
)


@dataclass(frozen=True)
class ScienceQAExtractionPolicy:
    """The full set of constraints, as a single inspectable object."""

    allowed_fields: tuple[str, ...] = ALLOWED_INFERENCE_FIELDS
    max_prereqs: int = MAX_PREREQUISITES
    min_prereqs_for_advanced: int = 1
    grade_range: tuple[int, int] = (1, 12)
    forbid_external_knowledge: bool = True


# ---- Small intermediate types ---------------------------------------------


@dataclass
class _ExtractedConcept:
    """In-flight concept assembled by the extractor before validation."""

    concept_id: str
    name: str
    subject: str | None = None
    grade: int | None = None
    skills: list[str] = field(default_factory=list)
    instruction: str | None = None
    prerequisites: list[str] = field(default_factory=list)
    difficulty: float | None = None
    # Which dataset field(s) each prerequisite was derived from.
    # Keyed by prerequisite slug. Must be non-empty after inference.
    provenance: dict[str, list[str]] = field(default_factory=dict)


# ---- The provider used to run the validator --------------------------------


class _ExtractedScienceQAProvider(BaseCurriculumProvider):
    """Validation-only provider.

    Used inside the extractor to push the assembled concept list
    through the same validator the runtime uses
    (:class:`BaseCurriculumProvider`). This avoids any drift between
    "what the extractor accepts" and "what the runtime accepts".
    """

    PROVIDER_NAME = "scienceqa_extract"

    def __init__(self, nodes: list[ConceptNode]) -> None:
        super().__init__()
        self._raw_nodes = nodes

    def _load(self) -> Iterable[ConceptNode]:
        return self._raw_nodes


# ---- Stubs for the real implementation -------------------------------------


def load_raw_scienceqa(path: Path) -> list[dict[str, Any]]:
    """Read raw ScienceQA records.

    **Not implemented.** This is where the dataset reader will go.
    Expected output: a list of dicts with at least the seven allowed
    inference fields (see :data:`ALLOWED_INFERENCE_FIELDS`).

    Tests inject a small handcrafted list and skip this function.
    """
    raise NotImplementedError(
        "load_raw_scienceqa: real ScienceQA reader pending. Replace this "
        "stub with a JSON or HuggingFace-datasets reader when the data "
        "lands in data/scienceqa/."
    )


def infer_prerequisites(
    records: list[dict[str, Any]],
    policy: ScienceQAExtractionPolicy,
) -> list[_ExtractedConcept]:
    """Build concept nodes with inferred prerequisites.

    **Not implemented.** Pending the real reader. The signature is
    final: a list of in-flight :class:`_ExtractedConcept` objects with
    provenance populated for every prerequisite.

    Allowed signals: only the seven fields in
    :data:`ALLOWED_INFERENCE_FIELDS`. Any prerequisite without
    provenance will be rejected by :func:`validate`.
    """
    del records, policy
    raise NotImplementedError(
        "infer_prerequisites: real prerequisite inference pending. "
        "Implement using only the seven fields in "
        "ALLOWED_INFERENCE_FIELDS. See ADR-006 section 2.4."
    )


def estimate_difficulty(
    concept: _ExtractedConcept,
    solution_text: str,
    policy: ScienceQAExtractionPolicy,
) -> float:
    """Coarse difficulty in ``[0, 1]``.

    Heuristic (ADR-006 section 2.4):

    * 0.5 * grade_norm
    * 0.3 * prereq_norm
    * 0.2 * reasoning_norm  (proxied by solution length)

    Each component is clamped to ``[0, 1]`` and the weighted sum is
    then clamped again. Deliberately simple; the goal is interpretable
    defaults, not a calibrated psychometric model.
    """
    g_lo, g_hi = policy.grade_range
    grade = (concept.grade if concept.grade is not None else (g_lo + g_hi) // 2)
    grade_norm = max(0.0, min((grade - g_lo) / max(1, g_hi - g_lo), 1.0))
    prereq_norm = min(len(concept.prerequisites), policy.max_prereqs) / float(
        policy.max_prereqs
    )
    # Solution length is a coarse but cheap proxy. Cap at 500 chars so
    # an extremely verbose solution does not dominate.
    reasoning_norm = min(len(solution_text or ""), 500) / 500.0

    score = 0.5 * grade_norm + 0.3 * prereq_norm + 0.2 * reasoning_norm
    return float(round(max(0.0, min(score, 1.0)), 4))


# ---- Validation helpers ----------------------------------------------------


def _check_provenance(
    concepts: list[_ExtractedConcept],
    policy: ScienceQAExtractionPolicy,
) -> list[str]:
    """Return a list of provenance violations.

    Each prerequisite must record at least one source field from
    :attr:`ScienceQAExtractionPolicy.allowed_fields`. This makes the
    "no external knowledge" rule from ADR-006 mechanically checkable
    instead of relying on reviewer trust.
    """
    problems: list[str] = []
    allowed = set(policy.allowed_fields)
    for c in concepts:
        for p in c.prerequisites:
            sources = c.provenance.get(p, [])
            if not sources:
                problems.append(
                    f"concept {c.concept_id!r}: prerequisite {p!r} has no "
                    f"provenance; every prereq must be traceable to one of "
                    f"{sorted(allowed)}"
                )
            else:
                bad = [s for s in sources if s not in allowed]
                if bad:
                    problems.append(
                        f"concept {c.concept_id!r}: prerequisite {p!r} cites "
                        f"disallowed sources {bad}; allowed: {sorted(allowed)}"
                    )
    return problems


def validate(
    concepts: list[_ExtractedConcept],
    policy: ScienceQAExtractionPolicy,
) -> None:
    """Validate extracted concepts against the policy + DAG rules.

    Runs provenance checks plus the standard graph validation through
    :class:`_ExtractedScienceQAProvider`. Raises a single
    :class:`AdaptiveTutorError` listing every problem if any are found.
    """
    problems = _check_provenance(concepts, policy)

    # Push through the same validator the runtime uses.
    nodes = [_to_concept_node(c) for c in concepts]
    try:
        _ExtractedScienceQAProvider(nodes).load()
    except AdaptiveTutorError as e:
        problems.append(str(e))

    if problems:
        raise AdaptiveTutorError(
            "ScienceQA extraction failed validation:\n  - " + "\n  - ".join(problems)
        )


def _to_concept_node(c: _ExtractedConcept) -> ConceptNode:
    return ConceptNode(
        concept_id=c.concept_id,
        name=c.name,
        subject=c.subject,
        grade=c.grade,
        skills=list(c.skills),
        instruction=c.instruction,
        prerequisites=list(c.prerequisites),
        difficulty=(c.difficulty if c.difficulty is not None else 0.5),
        extras={"provenance": c.provenance} if c.provenance else {},
    )


# ---- Serialisation ---------------------------------------------------------


def write_teacher_yaml(
    concepts: list[_ExtractedConcept],
    output_path: Path,
    version: str = "0.1.0-scienceqa-extract",
) -> None:
    """Write a teacher-compatible curriculum YAML.

    The output matches the shape consumed by
    :class:`TeacherCurriculumProvider`: a top-level dict with
    ``version`` and ``concepts``. Provenance is preserved under each
    concept's ``provenance`` key so reviewers can audit how each
    prerequisite was inferred.
    """
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as e:
        raise AdaptiveTutorError(
            "PyYAML is required to write the extracted curriculum file."
        ) from e

    payload = {
        "version": version,
        "concepts": [
            {
                "concept_id": c.concept_id,
                "name": c.name,
                "subject": c.subject,
                "grade": c.grade,
                "skills": list(c.skills),
                "instruction": c.instruction,
                "prerequisites": list(c.prerequisites),
                "difficulty": c.difficulty,
                "provenance": c.provenance,
            }
            for c in concepts
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


# ---- CLI -------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="extract-scienceqa-curriculum",
        description="Build a teacher-compatible curriculum YAML from ScienceQA.",
    )
    parser.add_argument(
        "--input", required=True, type=Path, help="Path to the raw ScienceQA file."
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination YAML to write. Loadable by TeacherCurriculumProvider.",
    )
    parser.add_argument(
        "--max-prereqs",
        type=int,
        default=MAX_PREREQUISITES,
        help=f"Max prerequisites per concept (default {MAX_PREREQUISITES}).",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    policy = ScienceQAExtractionPolicy(max_prereqs=args.max_prereqs)

    _LOG.info("Reading raw ScienceQA from %s", args.input)
    raw = load_raw_scienceqa(args.input)  # raises NotImplementedError until wired
    _LOG.info("Inferring prerequisites with policy %s", policy)
    concepts = infer_prerequisites(raw, policy)
    _LOG.info("Validating %d extracted concepts", len(concepts))
    validate(concepts, policy)
    _LOG.info("Writing teacher YAML to %s", args.output)
    write_teacher_yaml(concepts, args.output)
    _LOG.info("Done.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
