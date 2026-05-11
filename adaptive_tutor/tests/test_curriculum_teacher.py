"""TeacherCurriculumProvider tests (YAML + JSON loaders + ScienceQA example)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from adaptive_tutor.core.exceptions import AdaptiveTutorError
from adaptive_tutor.memory.providers.teacher_provider import (
    TeacherCurriculumProvider,
)

# Resolve the shipped example fixtures relative to the package root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_MATH_EXAMPLE = _REPO_ROOT / "configs" / "curriculum" / "examples" / "math_basic.yaml"
_SCIENCEQA_EXAMPLE = (
    _REPO_ROOT / "configs" / "curriculum" / "examples" / "scienceqa_excerpt.yaml"
)


# ---- shipped math example ------------------------------------------------


def test_math_example_loads_cleanly() -> None:
    p = TeacherCurriculumProvider(_MATH_EXAMPLE)
    assert p.num_concepts() == 6
    assert "addition_whole_numbers" in [
        c["concept_id"] for c in p.get_concepts()
    ]


def test_math_example_topological_order_is_valid() -> None:
    p = TeacherCurriculumProvider(_MATH_EXAMPLE)
    order = p.topological_order()
    pos = {slug: i for i, slug in enumerate(order)}
    # subtraction depends on addition, division on multiplication + subtraction
    assert pos["addition_whole_numbers"] < pos["subtraction_whole_numbers"]
    assert pos["multiplication_whole_numbers"] < pos["division_whole_numbers"]
    assert pos["subtraction_whole_numbers"] < pos["division_whole_numbers"]


def test_math_example_version_is_sha256() -> None:
    p = TeacherCurriculumProvider(_MATH_EXAMPLE)
    v = p.version()
    assert isinstance(v, str) and len(v) == 64


# ---- shipped ScienceQA example -------------------------------------------


def test_scienceqa_example_loads_cleanly() -> None:
    p = TeacherCurriculumProvider(_SCIENCEQA_EXAMPLE)
    # The Punnett-square concept has three prerequisites; all three
    # must be present in the loaded graph.
    slugs = {c["concept_id"] for c in p.get_concepts()}
    assert {
        "biology_punnett_square_probability",
        "genotype_vs_phenotype",
        "dominant_recessive_alleles",
        "basic_probability",
    }.issubset(slugs)


def test_scienceqa_example_topological_order() -> None:
    p = TeacherCurriculumProvider(_SCIENCEQA_EXAMPLE)
    order = p.topological_order()
    pos = {slug: i for i, slug in enumerate(order)}
    assert pos["dominant_recessive_alleles"] < pos["genotype_vs_phenotype"]
    assert pos["genotype_vs_phenotype"] < pos["biology_punnett_square_probability"]
    assert pos["basic_probability"] < pos["biology_punnett_square_probability"]


# ---- JSON loader ---------------------------------------------------------


def test_json_loader_works(tmp_path: Path) -> None:
    obj = {
        "version": "tjson-1",
        "concepts": [
            {"concept_id": "a", "name": "A"},
            {"concept_id": "b", "name": "B", "prerequisites": ["a"]},
        ],
    }
    path = tmp_path / "curriculum.json"
    path.write_text(json.dumps(obj), encoding="utf-8")
    p = TeacherCurriculumProvider(path)
    assert p.num_concepts() == 2
    assert sorted(p.concept_index(s) for s in ("a", "b")) == [0, 1]


# ---- Error paths ---------------------------------------------------------


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(AdaptiveTutorError, match="file not found"):
        TeacherCurriculumProvider(tmp_path / "missing.yaml")


def test_unsupported_extension_raises(tmp_path: Path) -> None:
    bad = tmp_path / "curriculum.txt"
    bad.write_text("anything", encoding="utf-8")
    with pytest.raises(AdaptiveTutorError, match="unsupported extension"):
        TeacherCurriculumProvider(bad)


def test_missing_concepts_key_raises(tmp_path: Path) -> None:
    path = tmp_path / "curriculum.json"
    path.write_text(json.dumps({"version": "1"}), encoding="utf-8")
    with pytest.raises(AdaptiveTutorError, match="'concepts' key"):
        TeacherCurriculumProvider(path)


def test_missing_concept_id_raises(tmp_path: Path) -> None:
    obj = {"concepts": [{"name": "no id"}]}
    path = tmp_path / "curriculum.json"
    path.write_text(json.dumps(obj), encoding="utf-8")
    with pytest.raises(AdaptiveTutorError, match="missing 'concept_id'"):
        TeacherCurriculumProvider(path)
