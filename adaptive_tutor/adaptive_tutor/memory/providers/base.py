"""Base class and validation logic for curriculum providers.

Why a concrete base class in addition to the Protocol
-----------------------------------------------------
``adaptive_tutor.core.protocols.CurriculumProvider`` is a structural
type (PEP 544 Protocol). It tells the RL stack *what methods to call*
on a curriculum source without constraining how the source is built.

This module adds a small *concrete* base class,
:class:`BaseCurriculumProvider`, that:

* holds the validated concept dictionary in memory,
* exposes the slug-to-index mapping the RL stack uses for tensor
  indexing,
* enforces the bachelor-thesis-level invariants on the curriculum
  graph:

    1. concept ids are unique snake_case strings,
    2. every prerequisite refers to an existing concept,
    3. the graph is a DAG (no cycles),
    4. each node has at most ``MAX_PREREQUISITES`` prerequisites,
    5. ``difficulty`` is in ``[0, 1]`` if present.

* computes a stable SHA-256 version hash over the canonical
  serialisation of the graph so the explainer can cite a specific
  curriculum revision (see ADR-005 ``feature_schema_sha``).

Concrete providers inherit from :class:`BaseCurriculumProvider` and
implement only ``_load() -> Iterable[ConceptNode]``. Everything else
(validation, indexing, version hashing, the
:class:`~adaptive_tutor.core.protocols.CurriculumProvider` Protocol
methods) is handled here.

This deliberately avoids the heavier abstractions described in the
earlier maximalist plan (see ADR-006 - Scope Reset).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from adaptive_tutor.core.exceptions import AdaptiveTutorError

# ---- Module-level constants -----------------------------------------------

# Hard cap recommended in the project spec. Beyond five prerequisites a
# concept node becomes hard to teach and explain in a thesis defence.
MAX_PREREQUISITES = 5

# snake_case validator. Lowercase ascii letters and digits, separated by
# single underscores. Must start with a letter. This rejects spaces,
# uppercase, hyphens, double underscores, and non-ascii characters.
_SNAKE_CASE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")


# ---- Errors ---------------------------------------------------------------


class CurriculumIntegrityError(AdaptiveTutorError):
    """Raised when a curriculum graph violates the well-formedness rules.

    Subclassing :class:`AdaptiveTutorError` keeps the exception
    discoverable via a single ``except`` clause at the top of the
    orchestrator.
    """


# ---- Concept dataclass ----------------------------------------------------


@dataclass
class ConceptNode:
    """One node of the curriculum graph.

    The exact field set matches the example concept objects in the
    project spec. ``extras`` is the open-ended slot for provider
    specific data (e.g. empirical ``historical_success_rate`` from the
    ASSISTments dataset provider or custom teacher notes).

    Attributes
    ----------
    concept_id:
        Unique snake_case slug. The canonical identifier in the
        graph. The RL stack receives this as an integer *index* into
        the concept list - the slug-to-index mapping is built by
        :class:`BaseCurriculumProvider`.
    name:
        Human-readable name. Used by the explainer.
    subject:
        Optional. Free-form domain string such as ``"mathematics"``,
        ``"natural_science"``, ``"social_science"``, ``"programming"``.
    grade:
        Optional integer grade level. Used by the difficulty heuristic.
    skills:
        List of finer-grained skill slugs the concept exercises. May
        be empty.
    instruction:
        Optional short instructional text shown to the learner.
    prerequisites:
        List of *concept_id* slugs that must be mastered before this
        one. May be empty. Maximum length :data:`MAX_PREREQUISITES`.
    difficulty:
        Float in ``[0, 1]``. If a provider does not supply one we
        estimate it from ``grade`` and ``len(prerequisites)``.
    extras:
        Provider specific extension dict. Examples:
        ``{"author": "thesis-demo"}``. ASSISTments-derived aggregates
        belong on :class:`SkillStats`, not inside teacher nodes.
    """

    concept_id: str
    name: str = ""
    subject: str | None = None
    grade: int | None = None
    skills: list[str] = field(default_factory=list)
    instruction: str | None = None
    prerequisites: list[str] = field(default_factory=list)
    difficulty: float = 0.5
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Provider-agnostic serialisation used by the version hash."""
        return asdict(self)


# ---- Base provider --------------------------------------------------------


class BaseCurriculumProvider:
    """Concrete base class for curriculum providers.

    Subclass and implement :meth:`_load`. Everything else is inherited.

    Lifecycle
    ---------
    ``__init__`` does *not* load the curriculum. Call :meth:`load`
    explicitly, or rely on a subclass that calls it from its own
    constructor. Lazy loading is intentional: tests can construct a
    provider with synthetic data without touching the filesystem.
    """

    # Subclasses override these to participate in the version hash.
    PROVIDER_NAME: str = "base"
    PROVIDER_SPEC_VERSION: str = "1.0.0"

    def __init__(self) -> None:
        self._concepts: dict[str, ConceptNode] = {}
        self._order: list[str] = []  # deterministic iteration order
        self._index: dict[str, int] = {}
        self._version_hash: str = ""
        self._loaded: bool = False

    # ---- public load-state flag ------------------------------------------

    @property
    def is_loaded(self) -> bool:
        """Whether :meth:`load` has been called successfully.

        The boolean ``self._loaded`` is private; outside callers (state
        builder, simulator, trainers) must read this property instead.
        Gives us one stable, documented attribute to gate against
        before calling methods that require a loaded graph.
        """
        return self._loaded

    # ---- subclass hook ----------------------------------------------------

    def _load(self) -> Iterable[ConceptNode]:
        """Return the raw concept nodes for this provider.

        Subclasses MUST implement this and return an iterable of
        :class:`ConceptNode` instances. Validation, indexing and
        hashing happen inside :meth:`load` after this returns.
        """
        raise NotImplementedError

    # ---- public API -------------------------------------------------------

    def load(self) -> None:
        """Run ``_load`` and validate the result.

        Raises:
            CurriculumIntegrityError: if any well-formedness rule is
                violated. The error message lists the failing concept
                ids so the user can fix the source curriculum.
        """
        raw = list(self._load())
        self._validate(raw)
        # Deterministic ordering: subjects first, then by grade, then
        # by concept_id. This makes the slug-to-index mapping stable
        # across runs as long as the input set is the same.
        raw.sort(key=lambda c: (str(c.subject or ""), int(c.grade or 0), c.concept_id))

        self._concepts = {c.concept_id: c for c in raw}
        self._order = [c.concept_id for c in raw]
        self._index = {slug: i for i, slug in enumerate(self._order)}

        # Compute difficulty defaults *after* indexing so the heuristic
        # has the prerequisite count available.
        for c in raw:
            if c.difficulty is None or not (0.0 <= c.difficulty <= 1.0):
                c.difficulty = self._estimate_difficulty(c)

        self._version_hash = self._compute_version_hash()
        self._loaded = True

    # ---- CurriculumProvider Protocol implementation -----------------------

    def get_concepts(self) -> Sequence[Mapping[str, Any]]:
        self._require_loaded()
        return [self._concepts[s].to_dict() for s in self._order]

    def get_prerequisites(self) -> Sequence[tuple[int, int]]:
        """Directed edges ``(from_index, to_index)``.

        Convention: ``(prereq, concept)``. The RL stack reads this as
        ``"to teach concept ``to``, the learner must have mastered
        ``from``"``.
        """
        self._require_loaded()
        edges: list[tuple[int, int]] = []
        for c in self._concepts.values():
            j = self._index[c.concept_id]
            for p in c.prerequisites:
                i = self._index[p]
                edges.append((i, j))
        return edges

    def get_learning_material(self, concept_id: int) -> Mapping[str, Any]:
        c = self._concept_by_index(concept_id)
        return {
            "concept_id": c.concept_id,
            "name": c.name,
            "instruction": c.instruction or "",
            "extras": dict(c.extras),
        }

    def get_skill_metadata(self, concept_id: int) -> Mapping[str, Any]:
        c = self._concept_by_index(concept_id)
        return {
            "concept_id": c.concept_id,
            "subject": c.subject,
            "grade": c.grade,
            "skills": list(c.skills),
            "difficulty": c.difficulty,
        }

    def version(self) -> str:
        self._require_loaded()
        return self._version_hash

    # ---- Slug <-> index helpers -------------------------------------------

    def num_concepts(self) -> int:
        self._require_loaded()
        return len(self._order)

    def concept_slug(self, idx: int) -> str:
        self._require_loaded()
        if not (0 <= idx < len(self._order)):
            raise IndexError(
                f"concept index {idx} out of range [0, {len(self._order)})"
            )
        return self._order[idx]

    def concept_index(self, slug: str) -> int:
        self._require_loaded()
        if slug not in self._index:
            raise KeyError(f"unknown concept slug: {slug!r}")
        return self._index[slug]

    def topological_order(self) -> list[str]:
        """Return concept slugs in a topological order (prereqs first).

        Uses Kahn's algorithm. The order is deterministic because we
        break ties using the canonical slug ordering established at
        load time.
        """
        self._require_loaded()
        indeg: dict[str, int] = {slug: 0 for slug in self._order}
        adj: dict[str, list[str]] = defaultdict(list)
        for c in self._concepts.values():
            for p in c.prerequisites:
                adj[p].append(c.concept_id)
                indeg[c.concept_id] += 1
        ready: deque[str] = deque(sorted(s for s, d in indeg.items() if d == 0))
        out: list[str] = []
        while ready:
            s = ready.popleft()
            out.append(s)
            for n in sorted(adj[s]):
                indeg[n] -= 1
                if indeg[n] == 0:
                    ready.append(n)
        # The validate step rejects cycles, so the sum length always
        # matches. Defence-in-depth: assert anyway.
        if len(out) != len(self._order):
            raise CurriculumIntegrityError(
                "topological_order: cycle detected; this indicates a "
                "load-time validation bug"
            )
        return out

    # ---- Internals --------------------------------------------------------

    def _require_loaded(self) -> None:
        if not self._loaded:
            raise CurriculumIntegrityError(
                f"{type(self).__name__} used before load(); call .load() first"
            )

    def _concept_by_index(self, idx: int) -> ConceptNode:
        slug = self.concept_slug(idx)
        return self._concepts[slug]

    def _validate(self, raw: Sequence[ConceptNode]) -> None:
        """Enforce the five well-formedness rules.

        Raises a single :class:`CurriculumIntegrityError` listing
        *all* offending concept ids, so the user can fix everything
        in one pass rather than chasing errors one at a time.
        """
        problems: list[str] = []

        # Rule 1 - unique snake_case ids.
        seen: dict[str, int] = defaultdict(int)
        for c in raw:
            seen[c.concept_id] += 1
            if not isinstance(c.concept_id, str) or not _SNAKE_CASE.match(c.concept_id):
                problems.append(f"invalid snake_case concept_id: {c.concept_id!r}")
        dupes = sorted(s for s, n in seen.items() if n > 1)
        if dupes:
            problems.append(f"duplicate concept ids: {dupes}")

        slugs = {c.concept_id for c in raw}

        # Rules 2 + 4 - prerequisite references and max length.
        for c in raw:
            if len(c.prerequisites) > MAX_PREREQUISITES:
                problems.append(
                    f"concept {c.concept_id!r} has {len(c.prerequisites)} "
                    f"prerequisites (max {MAX_PREREQUISITES})"
                )
            if len(set(c.prerequisites)) != len(c.prerequisites):
                problems.append(
                    f"concept {c.concept_id!r} has duplicate prerequisites"
                )
            for p in c.prerequisites:
                if p == c.concept_id:
                    problems.append(
                        f"concept {c.concept_id!r} lists itself as a prerequisite"
                    )
                elif p not in slugs:
                    problems.append(
                        f"concept {c.concept_id!r} references unknown "
                        f"prerequisite {p!r}"
                    )

        # Rule 5 - difficulty range. Out-of-range values get defaulted
        # later, but we still warn on obviously bad inputs.
        for c in raw:
            if c.difficulty is not None and not (0.0 <= c.difficulty <= 1.0):
                problems.append(
                    f"concept {c.concept_id!r} difficulty={c.difficulty} "
                    f"not in [0, 1] - will be replaced by heuristic estimate"
                )

        # Rule 3 - DAG. Run cycle detection only if earlier rules pass
        # enough that the graph is at least well-typed. Otherwise the
        # cycle algorithm could trip over unknown ids.
        critical = [p for p in problems if "references unknown" in p or "duplicate concept ids" in p]
        if not critical:
            cycle = self._find_cycle(raw)
            if cycle:
                problems.append("cycle in prerequisite graph: " + " -> ".join(cycle))

        if problems:
            raise CurriculumIntegrityError(
                "curriculum failed validation:\n  - " + "\n  - ".join(problems)
            )

    @staticmethod
    def _find_cycle(raw: Sequence[ConceptNode]) -> list[str] | None:
        """Return the first cycle found, or ``None`` if the graph is acyclic.

        Iterative DFS with a colour map: ``0 = unseen``, ``1 = on stack``,
        ``2 = done``. We reconstruct the cycle by walking back through
        the parent map when an on-stack neighbour is encountered.
        """
        adj: dict[str, list[str]] = {c.concept_id: list(c.prerequisites) for c in raw}
        colour: dict[str, int] = dict.fromkeys(adj, 0)
        parent: dict[str, str | None] = dict.fromkeys(adj, None)

        for start in sorted(adj):
            if colour[start] != 0:
                continue
            stack: list[tuple[str, int]] = [(start, 0)]
            while stack:
                node, i = stack[-1]
                if i == 0:
                    colour[node] = 1
                if i < len(adj[node]):
                    nbr = adj[node][i]
                    stack[-1] = (node, i + 1)
                    if colour.get(nbr, 2) == 1:
                        # Reconstruct cycle: nbr ... node nbr.
                        cycle = [nbr]
                        cur: str | None = node
                        while cur is not None and cur != nbr:
                            cycle.append(cur)
                            cur = parent[cur]
                        cycle.append(nbr)
                        return list(reversed(cycle))
                    if colour.get(nbr, 0) == 0:
                        parent[nbr] = node
                        stack.append((nbr, 0))
                else:
                    colour[node] = 2
                    stack.pop()
        return None

    def _estimate_difficulty(self, c: ConceptNode) -> float:
        """Heuristic difficulty in ``[0, 1]``.

        Combines two signals (per ADR-006 section 2.4):

        * normalised grade level (assume grades 1 to 12),
        * normalised number of prerequisites (0 to MAX_PREREQUISITES).

        With weights 0.7 / 0.3 - grade is the stronger signal. The
        heuristic is intentionally simple; concrete providers can
        override or pre-set ``difficulty`` to skip it entirely.
        """
        grade_norm = 0.5 if c.grade is None else min(max(c.grade, 1), 12) / 12.0
        prereq_norm = min(len(c.prerequisites), MAX_PREREQUISITES) / float(
            MAX_PREREQUISITES
        )
        return float(round(0.7 * grade_norm + 0.3 * prereq_norm, 4))

    def _compute_version_hash(self) -> str:
        """SHA-256 over the canonical JSON of the loaded graph.

        Canonicalisation: deterministic key order, deterministic
        concept order (``self._order``), with the provider name and
        spec version mixed in so two providers with structurally
        identical graphs but different semantics produce different
        hashes (e.g. teacher-authored vs dataset-derived).
        """
        payload = {
            "provider": self.PROVIDER_NAME,
            "spec_version": self.PROVIDER_SPEC_VERSION,
            "concepts": [self._concepts[s].to_dict() for s in self._order],
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()
