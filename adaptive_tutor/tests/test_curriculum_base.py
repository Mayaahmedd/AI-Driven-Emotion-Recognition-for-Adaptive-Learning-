"""Validation contracts for BaseCurriculumProvider.

These tests exercise the well-formedness rules - duplicate ids,
snake_case, cycles, missing references, MAX_PREREQUISITES, version
hashing - using an in-memory provider so the tests do not depend on
the filesystem.
"""

from __future__ import annotations

from collections.abc import Iterable

import pytest

from adaptive_tutor.memory.providers.base import (
    MAX_PREREQUISITES,
    BaseCurriculumProvider,
    ConceptNode,
    CurriculumIntegrityError,
)


class _InMemoryProvider(BaseCurriculumProvider):
    """Tiny provider that returns whatever nodes the test gave it."""

    PROVIDER_NAME = "test"

    def __init__(self, nodes: list[ConceptNode]) -> None:
        super().__init__()
        self._test_nodes = nodes

    def _load(self) -> Iterable[ConceptNode]:
        return self._test_nodes


# ---- Happy path ----------------------------------------------------------


def _simple_dag() -> list[ConceptNode]:
    return [
        ConceptNode(concept_id="a", name="A", grade=1, prerequisites=[]),
        ConceptNode(concept_id="b", name="B", grade=2, prerequisites=["a"]),
        ConceptNode(concept_id="c", name="C", grade=3, prerequisites=["a", "b"]),
    ]


def test_load_assigns_deterministic_index() -> None:
    p = _InMemoryProvider(_simple_dag())
    p.load()
    assert p.num_concepts() == 3
    # Sort key is (subject, grade, concept_id) -> all subjects are None
    # so the effective sort is by grade then by slug.
    assert p.concept_slug(0) == "a"
    assert p.concept_slug(1) == "b"
    assert p.concept_slug(2) == "c"
    assert p.concept_index("c") == 2


def test_prerequisites_returned_as_index_edges() -> None:
    p = _InMemoryProvider(_simple_dag())
    p.load()
    edges = sorted(p.get_prerequisites())
    # (from_idx, to_idx) for a -> b, a -> c, b -> c
    assert edges == [(0, 1), (0, 2), (1, 2)]


def test_topological_order_is_valid() -> None:
    p = _InMemoryProvider(_simple_dag())
    p.load()
    order = p.topological_order()
    pos = {slug: i for i, slug in enumerate(order)}
    # Every prereq must appear before its dependant.
    assert pos["a"] < pos["b"]
    assert pos["a"] < pos["c"]
    assert pos["b"] < pos["c"]


def test_version_hash_changes_when_graph_changes() -> None:
    p1 = _InMemoryProvider(_simple_dag())
    p1.load()
    p2 = _InMemoryProvider(_simple_dag())
    p2.load()
    assert p1.version() == p2.version()

    # Add one extra concept and the hash must change.
    p3 = _InMemoryProvider(
        [
            *_simple_dag(),
            ConceptNode(concept_id="d", name="D", grade=4, prerequisites=["c"]),
        ]
    )
    p3.load()
    assert p3.version() != p1.version()


def test_get_learning_material_and_skill_metadata() -> None:
    p = _InMemoryProvider(_simple_dag())
    p.load()
    mat = p.get_learning_material(1)
    assert mat["concept_id"] == "b"
    meta = p.get_skill_metadata(1)
    assert meta["grade"] == 2
    assert 0.0 <= meta["difficulty"] <= 1.0


# ---- Snake_case enforcement ----------------------------------------------


@pytest.mark.parametrize(
    "bad_slug",
    [
        "BadSlug",      # uppercase
        "bad-slug",     # hyphen
        "bad slug",     # space
        "__bad",        # leading underscores
        "bad__slug",    # double underscore (illegal per regex)
        "1leading",     # leading digit
        "naive",        # actually fine; included to verify the regex is not over-broad
    ],
)
def test_snake_case_rule(bad_slug: str) -> None:
    nodes = [ConceptNode(concept_id=bad_slug, name=bad_slug)]
    if bad_slug == "naive":
        # Valid slug; should load cleanly.
        _InMemoryProvider(nodes).load()
        return
    with pytest.raises(CurriculumIntegrityError, match="invalid snake_case"):
        _InMemoryProvider(nodes).load()


# ---- Duplicate ids -------------------------------------------------------


def test_duplicate_ids_rejected() -> None:
    nodes = [
        ConceptNode(concept_id="a", name="A"),
        ConceptNode(concept_id="a", name="A2"),
    ]
    with pytest.raises(CurriculumIntegrityError, match="duplicate concept ids"):
        _InMemoryProvider(nodes).load()


# ---- Missing prerequisite references -------------------------------------


def test_missing_prerequisite_reference_rejected() -> None:
    nodes = [
        ConceptNode(concept_id="a", name="A", prerequisites=["nonexistent"]),
    ]
    with pytest.raises(CurriculumIntegrityError, match="unknown prerequisite"):
        _InMemoryProvider(nodes).load()


def test_self_prerequisite_rejected() -> None:
    nodes = [
        ConceptNode(concept_id="a", name="A", prerequisites=["a"]),
    ]
    with pytest.raises(CurriculumIntegrityError, match="lists itself"):
        _InMemoryProvider(nodes).load()


def test_duplicate_prerequisites_within_node_rejected() -> None:
    nodes = [
        ConceptNode(concept_id="a", name="A"),
        ConceptNode(concept_id="b", name="B", prerequisites=["a", "a"]),
    ]
    with pytest.raises(CurriculumIntegrityError, match="duplicate prerequisites"):
        _InMemoryProvider(nodes).load()


# ---- Max prerequisites ---------------------------------------------------


def test_max_prerequisites_enforced() -> None:
    n = MAX_PREREQUISITES + 1
    base = [ConceptNode(concept_id=f"a{i}", name=f"A{i}") for i in range(n)]
    target = ConceptNode(
        concept_id="target",
        name="Target",
        prerequisites=[c.concept_id for c in base],
    )
    with pytest.raises(CurriculumIntegrityError, match="prerequisites"):
        _InMemoryProvider([*base, target]).load()


# ---- Cycles --------------------------------------------------------------


def test_two_node_cycle_detected() -> None:
    nodes = [
        ConceptNode(concept_id="a", name="A", prerequisites=["b"]),
        ConceptNode(concept_id="b", name="B", prerequisites=["a"]),
    ]
    with pytest.raises(CurriculumIntegrityError, match="cycle in prerequisite graph"):
        _InMemoryProvider(nodes).load()


def test_three_node_cycle_detected() -> None:
    nodes = [
        ConceptNode(concept_id="a", name="A", prerequisites=["c"]),
        ConceptNode(concept_id="b", name="B", prerequisites=["a"]),
        ConceptNode(concept_id="c", name="C", prerequisites=["b"]),
    ]
    with pytest.raises(CurriculumIntegrityError, match="cycle"):
        _InMemoryProvider(nodes).load()


# ---- Lazy/eager + access-before-load --------------------------------------


def test_access_before_load_raises() -> None:
    p = _InMemoryProvider(_simple_dag())
    # No load() call.
    with pytest.raises(CurriculumIntegrityError, match="before load"):
        p.num_concepts()


def test_is_loaded_property_tracks_load() -> None:
    # Public load-state property; replaces all direct access to the
    # private ``_loaded`` attribute. Off before load, on after.
    p = _InMemoryProvider(_simple_dag())
    assert p.is_loaded is False
    p.load()
    assert p.is_loaded is True
