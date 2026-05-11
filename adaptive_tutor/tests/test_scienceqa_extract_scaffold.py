"""Scaffolding tests for ScienceQA extraction.

These tests pin the *contracts* of the extraction script: provenance
requirements, DAG validation, difficulty heuristic, and YAML output
shape. They do not exercise the real ScienceQA reader - that lands
when the data does.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from adaptive_tutor.core.exceptions import AdaptiveTutorError
from adaptive_tutor.memory.providers.teacher_provider import (
    TeacherCurriculumProvider,
)
from adaptive_tutor.tools.extract_scienceqa_curriculum import (
    ScienceQAExtractionPolicy,
    _ExtractedConcept,
    estimate_difficulty,
    validate,
    write_teacher_yaml,
)


# ---- Helper to build an in-memory extracted set --------------------------


def _ok_concepts() -> list[_ExtractedConcept]:
    return [
        _ExtractedConcept(
            concept_id="basic_probability",
            name="Basic probability",
            subject="mathematics",
            grade=6,
            skills=["event_probability"],
            instruction="Probability is favourable / total.",
            prerequisites=[],
        ),
        _ExtractedConcept(
            concept_id="dominant_recessive_alleles",
            name="Dominant and recessive alleles",
            subject="natural_science",
            grade=7,
            skills=["allele_dominance"],
            instruction="Dominant alleles mask recessive ones.",
            prerequisites=[],
        ),
        _ExtractedConcept(
            concept_id="genotype_vs_phenotype",
            name="Genotype vs phenotype",
            subject="natural_science",
            grade=7,
            skills=["genotype_phenotype_distinction"],
            instruction="Genotype is alleles; phenotype is observable trait.",
            prerequisites=["dominant_recessive_alleles"],
            provenance={"dominant_recessive_alleles": ["skill", "lecture"]},
        ),
        _ExtractedConcept(
            concept_id="biology_punnett_square_probability",
            name="Punnett square probability",
            subject="natural_science",
            grade=8,
            skills=["punnett_square_construction"],
            instruction="Use Punnett squares to calculate probabilities.",
            prerequisites=[
                "genotype_vs_phenotype",
                "dominant_recessive_alleles",
                "basic_probability",
            ],
            provenance={
                "genotype_vs_phenotype": ["skill", "lecture"],
                "dominant_recessive_alleles": ["skill"],
                "basic_probability": ["skill", "topic"],
            },
        ),
    ]


# ---- Validation contract -------------------------------------------------


def test_valid_extraction_passes() -> None:
    validate(_ok_concepts(), ScienceQAExtractionPolicy())


def test_prerequisite_without_provenance_rejected() -> None:
    bad = _ok_concepts()
    # Remove the provenance for one prerequisite.
    bad[3].provenance = {}
    with pytest.raises(AdaptiveTutorError, match="no provenance"):
        validate(bad, ScienceQAExtractionPolicy())


def test_disallowed_provenance_source_rejected() -> None:
    bad = _ok_concepts()
    bad[3].provenance = {
        "genotype_vs_phenotype": ["wikipedia"],  # not allowed
        "dominant_recessive_alleles": ["skill"],
        "basic_probability": ["skill"],
    }
    with pytest.raises(AdaptiveTutorError, match="disallowed sources"):
        validate(bad, ScienceQAExtractionPolicy())


def test_cycle_in_extracted_graph_rejected() -> None:
    bad = _ok_concepts()
    # Make basic_probability depend on biology_punnett to introduce a cycle.
    bad[0].prerequisites = ["biology_punnett_square_probability"]
    bad[0].provenance = {"biology_punnett_square_probability": ["topic"]}
    with pytest.raises(AdaptiveTutorError, match="cycle"):
        validate(bad, ScienceQAExtractionPolicy())


# ---- Difficulty heuristic ------------------------------------------------


def test_difficulty_increases_with_grade() -> None:
    p = ScienceQAExtractionPolicy()
    lo = estimate_difficulty(
        _ExtractedConcept(concept_id="a", name="A", grade=2), "short", p
    )
    hi = estimate_difficulty(
        _ExtractedConcept(concept_id="a", name="A", grade=10), "short", p
    )
    assert hi > lo
    assert 0.0 <= lo <= 1.0
    assert 0.0 <= hi <= 1.0


def test_difficulty_increases_with_prereqs() -> None:
    p = ScienceQAExtractionPolicy()
    lo = estimate_difficulty(
        _ExtractedConcept(concept_id="a", name="A", grade=5), "x", p
    )
    hi = estimate_difficulty(
        _ExtractedConcept(
            concept_id="a", name="A", grade=5, prerequisites=["b", "c", "d", "e", "f"]
        ),
        "x",
        p,
    )
    assert hi > lo


def test_difficulty_bounded() -> None:
    p = ScienceQAExtractionPolicy()
    d = estimate_difficulty(
        _ExtractedConcept(
            concept_id="a",
            name="A",
            grade=99,  # well outside the [1, 12] range
            prerequisites=["x"] * 100,
        ),
        "y" * 100000,
        p,
    )
    assert 0.0 <= d <= 1.0


# ---- Round-trip via TeacherCurriculumProvider ----------------------------


def test_yaml_output_is_loadable_by_teacher_provider(tmp_path: Path) -> None:
    """The end-to-end contract: extracted YAML must load cleanly into
    the runtime ``TeacherCurriculumProvider``. Guards against any
    drift between the extractor's output shape and the loader's
    input shape."""
    concepts = _ok_concepts()
    out = tmp_path / "extracted.yaml"
    write_teacher_yaml(concepts, out)
    p = TeacherCurriculumProvider(out)
    assert p.num_concepts() == 4
    assert "biology_punnett_square_probability" in [
        c["concept_id"] for c in p.get_concepts()
    ]
