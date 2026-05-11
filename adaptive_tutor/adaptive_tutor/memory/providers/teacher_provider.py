"""Teacher-authored curriculum loader.

Loads a curriculum definition from a YAML or JSON file. Used for both
manually written syllabi (e.g. a small math example for the thesis
demo) and for pre-extracted ScienceQA-derived curricula produced by a
future offline script.

Expected file shape (YAML)
--------------------------

.. code-block:: yaml

    version: "1.0.0"           # optional; provider tags hash with this
    concepts:
      - concept_id: variables
        name: "Variables"
        subject: programming
        grade: 6
        skills: [assign_variable, read_variable]
        instruction: "A variable is a named storage location."
        prerequisites: []
        difficulty: 0.10
      - concept_id: loops
        name: "Loops"
        subject: programming
        grade: 6
        skills: [for_loop, while_loop]
        instruction: "A loop repeats a block of code."
        prerequisites: [variables]
        difficulty: 0.30

The same shape is supported in JSON (top-level object with the same
keys). The loader is intentionally simple - one read, one parse, no
streaming. Curricula in this project are at most a few hundred
concepts.

Validation, indexing, version hashing, slug-to-index helpers all come
from :class:`BaseCurriculumProvider`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from adaptive_tutor.core.exceptions import AdaptiveTutorError
from adaptive_tutor.memory.providers.base import (
    BaseCurriculumProvider,
    ConceptNode,
)


class TeacherCurriculumProvider(BaseCurriculumProvider):
    """Loads a curriculum from a YAML or JSON file.

    Parameters
    ----------
    path:
        Filesystem path to a ``.yaml`` / ``.yml`` / ``.json`` file with
        the structure documented in the module docstring.
    eager:
        If ``True`` (default), call :meth:`load` from the constructor.
        Set to ``False`` if the caller wants to inspect ``self.path``
        before loading.

    Example
    -------

        provider = TeacherCurriculumProvider("configs/curriculum/examples/math_basic.yaml")
        provider.num_concepts()                # 5
        provider.concept_slug(0)               # 'variables'
        provider.version()                     # 'sha256-...'
    """

    PROVIDER_NAME = "teacher"

    def __init__(self, path: str | Path, *, eager: bool = True) -> None:
        super().__init__()
        self.path = Path(path)
        if eager:
            self.load()

    # ---- BaseCurriculumProvider hook --------------------------------------

    def _load(self) -> Iterable[ConceptNode]:
        if not self.path.exists():
            raise AdaptiveTutorError(
                f"TeacherCurriculumProvider: file not found: {self.path}"
            )
        raw = _read_structured_file(self.path)
        if not isinstance(raw, dict) or "concepts" not in raw:
            raise AdaptiveTutorError(
                f"TeacherCurriculumProvider: expected top-level dict with a "
                f"'concepts' key in {self.path}"
            )

        # The file may set a version string; we mix it into the
        # provider spec version so two curricula with the same shape
        # but different content versions hash differently.
        file_version = str(raw.get("version", "0"))
        self.PROVIDER_SPEC_VERSION = f"teacher::{file_version}"

        concepts_raw = raw["concepts"]
        if not isinstance(concepts_raw, list):
            raise AdaptiveTutorError(
                "TeacherCurriculumProvider: 'concepts' must be a list"
            )

        nodes: list[ConceptNode] = []
        for i, entry in enumerate(concepts_raw):
            if not isinstance(entry, dict):
                raise AdaptiveTutorError(
                    f"TeacherCurriculumProvider: concepts[{i}] is not an object"
                )
            nodes.append(_concept_from_dict(entry, source=str(self.path), index=i))
        return nodes


# ---- helpers ---------------------------------------------------------------


def _read_structured_file(path: Path) -> Any:
    """Read a YAML or JSON file.

    YAML is loaded lazily so that running tests without PyYAML
    installed still works as long as the test uses a JSON fixture.
    """
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as e:
            raise AdaptiveTutorError(
                "PyYAML is required to load .yaml files; install with "
                "`pip install pyyaml`, or convert the file to JSON."
            ) from e
        return yaml.safe_load(text)
    if suffix == ".json":
        return json.loads(text)
    raise AdaptiveTutorError(
        f"TeacherCurriculumProvider: unsupported extension {suffix!r}; "
        f"use .yaml, .yml or .json"
    )


def _concept_from_dict(entry: dict[str, Any], *, source: str, index: int) -> ConceptNode:
    """Build a :class:`ConceptNode` from a parsed dict entry.

    Validation of types is defensive: the base provider's validator
    will catch most issues, but we surface clear errors here for the
    most common authoring mistakes (missing concept_id, wrong types).
    """
    if "concept_id" not in entry:
        raise AdaptiveTutorError(
            f"{source}: concepts[{index}] is missing 'concept_id'"
        )
    raw_prereqs = entry.get("prerequisites", []) or []
    if not isinstance(raw_prereqs, list):
        raise AdaptiveTutorError(
            f"{source}: concepts[{index}].prerequisites must be a list"
        )
    raw_skills = entry.get("skills", []) or []
    if not isinstance(raw_skills, list):
        raise AdaptiveTutorError(
            f"{source}: concepts[{index}].skills must be a list"
        )

    raw_diff = entry.get("difficulty")
    difficulty = float(raw_diff) if raw_diff is not None else 0.5

    return ConceptNode(
        concept_id=str(entry["concept_id"]),
        name=str(entry.get("name", entry["concept_id"])),
        subject=(str(entry["subject"]) if entry.get("subject") is not None else None),
        grade=(int(entry["grade"]) if entry.get("grade") is not None else None),
        skills=[str(s) for s in raw_skills],
        instruction=(
            str(entry["instruction"]) if entry.get("instruction") is not None else None
        ),
        prerequisites=[str(p) for p in raw_prereqs],
        difficulty=difficulty,
        extras={
            k: v
            for k, v in entry.items()
            if k
            not in {
                "concept_id",
                "name",
                "subject",
                "grade",
                "skills",
                "instruction",
                "prerequisites",
                "difficulty",
            }
        },
    )
