"""ASSISTments dataset adapter.

Role in the system
------------------
This adapter is the **behaviour** source: it converts ASSISTments
interaction logs into RL transitions and per-skill statistics. It is
**not** curriculum memory in the knowledge sense. Per ADR-006 the role
separation is strict:

* ASSISTments via :class:`DatasetCurriculumProvider`
    -> per-skill statistics, transition iterator (for replay buffer
       seeding and simulator parameter fitting).
* ScienceQA via :class:`TeacherCurriculumProvider` (after offline
  extraction) -> concept graph + instructional content.

Why this adapter still implements ``CurriculumProvider``
-------------------------------------------------------
ASSISTments has *skills* (e.g., ``"Addition Whole Numbers"``,
``"Equation Solving Two or Fewer Steps"``). Treating each unique skill
as a concept with **no** prerequisites gives us a flat, valid curriculum
graph that satisfies :class:`BaseCurriculumProvider` validation only
exists so that ASSISTments *skills* (which are flat and have no
prerequisites declared in the CSV) can still be named in
:class:`DatasetCurriculumProvider` and correlated with teacher YAML
concept slugs by string equality at **runtime** in the state builder
and simulator - not via a merged provider.

Expected CSV columns
--------------------
The adapter accepts a permissive subset of ASSISTments columns. The
absolute minimum is ``skill`` and ``correct``. The following columns
are used when present:

================================================  ==================
``skill`` (str)                                   required - the skill tag
``correct`` (0/1)                                 required - correctness
``hint_count`` (int)                              optional - hints used
``attempt_count`` (int)                           optional - attempts
``ms_first_response`` (int)                       optional - response time (ms)
``Average_confidence(FRUSTRATED)`` (0-1)          optional - emotion proxy
``Average_confidence(CONFUSED)`` (0-1)            optional
``Average_confidence(CONCENTRATING)`` (0-1)       optional - mapped to engaged
``Average_confidence(BORED)`` (0-1)               optional
``user_id`` (any)                                 optional - groups episodes
================================================  ==================

Any extra columns are ignored. Missing optional columns default to
``0``.

What this adapter is **not**
----------------------------
* It does not attempt to fit a sophisticated mastery model. Mastery is
  estimated by a simple running mean of correctness per (user, skill).
* It does not learn pedagogical actions. The historical action is
  *labelled by a deterministic rule* based on hint and attempt counts;
  this is the same approach used in most ASSISTments OPE papers and is
  the "DEFINE and SIMULATE" pattern called out in the project spec.

The rule based action labelling is:

* ``give_hint``         if ``hint_count > 0`` and ``correct == 1``
* ``retry_current_skill`` if ``attempt_count > 1`` and ``correct == 1``
* ``easier_problem``    if ``correct == 0`` and ``hint_count > 1``
* ``harder_problem``    if ``correct == 1`` and ``hint_count == 0``
                          and ``attempt_count == 1``
* ``encouragement``     if frustration proxy > 0.5
* ``advance_to_next_skill`` otherwise

This labelling is intentionally simple. It is documented here so the
thesis defence can cite the exact rule.

Scalar rewards for each transition come from :class:`~adaptive_tutor.rewards.RewardEngine`
(instantiated via ``reward_engine`` on this provider, defaulting to the
process-wide bachelor coefficients). That keeps offline CSV iteration
and the synthetic environment on the same formula without embedding
reward arithmetic in this adapter.
"""

from __future__ import annotations

import csv
import logging
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from adaptive_tutor.core.exceptions import AdaptiveTutorError
from adaptive_tutor.memory.providers.base import (
    BaseCurriculumProvider,
    ConceptNode,
)
from adaptive_tutor.rewards import RewardEngine, default_reward_engine

_LOG = logging.getLogger(__name__)

# Module-level dedup set for malformed-value warnings. Keyed by column
# name so each column produces at most one WARNING per Python process.
# This keeps logs readable on noisy real-world ASSISTments exports while
# still surfacing the first occurrence.
_WARNED_MALFORMED_COLUMNS: set[str] = set()

# ----- Action vocabulary used by the rule-based labeller -------------------

ASSISTMENTS_ACTIONS: tuple[str, ...] = (
    "give_hint",
    "retry_current_skill",
    "easier_problem",
    "harder_problem",
    "encouragement",
    "advance_to_next_skill",
)


# ----- Per-skill aggregate -------------------------------------------------


@dataclass(frozen=True)
class SkillStats:
    """Summary statistics for one ASSISTments skill.

    All fields are simple, interpretable averages computed by one pass
    over the CSV. They feed:

    * the :class:`StateBuilder` / simulator (optional lookup tuple via
      :func:`skill_stats_as_tuple`),
    * cohort calibration in :mod:`adaptive_tutor.simulator.calibration`.
    """

    skill: str
    n_records: int
    mean_correctness: float
    mean_hint_count: float
    mean_attempts: float
    mean_response_time_ms: float
    mean_frustrated: float
    mean_engaged: float


# ----- Simple transition record --------------------------------------------


@dataclass(frozen=True)
class RawTransition:
    """A single ASSISTments-derived transition.

    Field names match the project spec's example state/action/reward
    dicts so the thesis text can quote one transition verbatim.

    Notes:

    * ``state`` and ``next_state`` are simple ``dict`` for legibility;
      Phase 4's state builder converts them into tensors.
    * ``action`` is one of :data:`ASSISTMENTS_ACTIONS`.
    * ``done`` ends a transition sequence when the user moves to a
      different skill (one user, one skill, one mini-episode).
    """

    user_id: str
    skill: str
    state: dict[str, Any]
    action: str
    reward: float
    next_state: dict[str, Any]
    done: bool
    correct: int = 0
    hint_count: int = 0
    attempt_count: int = 1


# ----- The provider --------------------------------------------------------


class DatasetCurriculumProvider(BaseCurriculumProvider):
    """ASSISTments adapter.

    Parameters
    ----------
    path:
        Path to a CSV file with the documented column set.
    eager:
        If ``True`` (default), load and validate immediately.
    reward_engine:
        Optional :class:`~adaptive_tutor.rewards.RewardEngine` for
        ``iter_transitions`` rewards. Defaults to the process-wide
        bachelor-thesis engine.
    """

    PROVIDER_NAME = "assistments"

    def __init__(
        self,
        path: str | Path,
        *,
        eager: bool = True,
        reward_engine: RewardEngine | None = None,
    ) -> None:
        super().__init__()
        self.path = Path(path)
        self._rows: list[dict[str, Any]] = []
        self._stats: dict[str, SkillStats] = {}
        self._reward_engine = reward_engine or default_reward_engine()
        if eager:
            self.load()

    # ---- BaseCurriculumProvider hook --------------------------------------

    def _load(self) -> Iterable[ConceptNode]:
        if not self.path.exists():
            raise AdaptiveTutorError(
                f"DatasetCurriculumProvider: file not found: {self.path}"
            )
        self._rows = _read_csv(self.path)
        if not self._rows:
            raise AdaptiveTutorError(
                f"DatasetCurriculumProvider: {self.path} contains no rows"
            )
        self._stats = _aggregate_stats(self._rows)

        # One concept per unique skill, with no prerequisites. Slugs
        # are produced by a deterministic slugifier so the dataset's
        # free-form skill strings become valid snake_case ids.
        nodes: list[ConceptNode] = []
        for skill, stats in sorted(self._stats.items()):
            slug = _slugify(skill)
            nodes.append(
                ConceptNode(
                    concept_id=slug,
                    name=skill,
                    subject="assistments",
                    grade=None,
                    skills=[slug],
                    instruction=None,
                    prerequisites=[],
                    difficulty=_difficulty_from_correctness(stats.mean_correctness),
                    extras={
                        "historical_success_rate": stats.mean_correctness,
                        "mean_hint_count": stats.mean_hint_count,
                        "mean_attempts": stats.mean_attempts,
                        "mean_response_time_ms": stats.mean_response_time_ms,
                        "mean_frustrated": stats.mean_frustrated,
                        "mean_engaged": stats.mean_engaged,
                        "n_records": stats.n_records,
                    },
                )
            )
        # The provider spec version mixes in the row count so a
        # different cut of ASSISTments hashes differently.
        self.PROVIDER_SPEC_VERSION = f"assistments::rows={len(self._rows)}"
        return nodes

    # ---- Provider-specific public API -------------------------------------

    def stats(self, skill_or_slug: str) -> SkillStats:
        """Return :class:`SkillStats` for the given skill (raw or slugified)."""
        self._require_loaded()
        if skill_or_slug in self._stats:
            return self._stats[skill_or_slug]
        for skill, st in self._stats.items():
            if _slugify(skill) == skill_or_slug:
                return st
        raise KeyError(f"unknown skill or slug: {skill_or_slug!r}")

    def all_stats(self) -> dict[str, SkillStats]:
        """Mapping ``slug -> SkillStats``."""
        self._require_loaded()
        return {_slugify(s): st for s, st in self._stats.items()}

    def iter_transitions(self) -> Iterator[RawTransition]:
        """Yield :class:`RawTransition` rows grouped by ``(user_id, skill)``.

        Each (user, skill) group becomes a mini-episode. Within the
        group, transitions are emitted in original row order. ``done``
        is set on the final row of the group.

        State at row *t*:

        * ``mastery``: running mean correctness over the first *t-1*
          rows of this (user, skill) group, clamped to ``[0, 1]``.
        * ``recent_accuracy``: same as ``mastery`` for the v1 provider.
        * ``hint_usage``: fraction of *t-1* rows where ``hint_count > 0``.
        * ``attempts``: ``attempt_count`` on row *t*.
        * ``emotion``: dict with the four FER-aligned channels derived
          from the dataset's ``Average_confidence`` columns.
        * ``current_skill``: the skill of this group.
        """
        self._require_loaded()
        groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in self._rows:
            key = (str(row.get("user_id", "anon")), str(row["skill"]))
            groups[key].append(row)

        for (user_id, skill), rows in groups.items():
            slug = _slugify(skill)
            running_correct = 0
            running_hinted = 0
            prev_state = _initial_state_dict(slug)
            for i, row in enumerate(rows):
                correct_raw = float(row.get("correct", 0))

# Convert ASSISTMENTS correctness into binary success
                correct = 1 if correct_raw >= 0.5 else 0
                hint = int(row.get("hint_count", 0))
                attempts = int(row.get("attempt_count", 1))
                emotion = _emotion_from_row(row)

                # Action label derives from the *current* row.
                action = _label_action(correct, hint, attempts, emotion)

                # Compute reward from the spec's simple formula.
                reward = self._reward_engine.compute_scalar(
                    correct=correct, hint_count=hint, emotion=emotion
                )

                next_state = {
                    "mastery": _clip01((running_correct + correct) / (i + 1)),
                    "recent_accuracy": _clip01((running_correct + correct) / (i + 1)),
                    "hint_usage": _clip01(
                        (running_hinted + (1 if hint > 0 else 0)) / (i + 1)
                    ),
                    "attempts": attempts,
                    "emotion": emotion,
                    "current_skill": slug,
                }

                yield RawTransition(
                    user_id=user_id,
                    skill=slug,
                    state=prev_state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=(i == len(rows) - 1),
                    correct=correct,
                    hint_count=hint,
                    attempt_count=attempts,
                )

                running_correct += correct
                running_hinted += 1 if hint > 0 else 0
                prev_state = next_state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_csv(path: Path) -> list[dict[str, Any]]:
    """Read an ASSISTments-shaped CSV into a list of dicts.

    Missing required columns raise. Empty cells are treated as 0 for
    numeric columns and as the empty string for textual ones.
    """
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows: list[dict[str, Any]] = []
        if reader.fieldnames is None or "skill" not in reader.fieldnames:
            raise AdaptiveTutorError(
                f"DatasetCurriculumProvider: {path} has no 'skill' column"
            )
        if "correct" not in reader.fieldnames:
            raise AdaptiveTutorError(
                f"DatasetCurriculumProvider: {path} has no 'correct' column"
            )
        for raw in reader:
            row: dict[str, Any] = {}
            for k, v in raw.items():
                if v is None or v == "":
                    row[k] = 0 if k != "skill" and k != "user_id" else ""
                else:
                    row[k] = v
            rows.append(row)
    return rows


def _aggregate_stats(rows: list[dict[str, Any]]) -> dict[str, SkillStats]:
    """One-pass aggregation per skill.

    Complexity: O(n_rows). Numeric coercion is defensive: malformed
    cells default to 0 rather than crashing the whole load.
    """
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row["skill"])].append(row)

    out: dict[str, SkillStats] = {}
    for skill, group in buckets.items():
        n = len(group)
        c_sum = h_sum = a_sum = t_sum = 0.0
        frust_sum = eng_sum = 0.0
        for r in group:
            c_sum += _float(r.get("correct", 0), column="correct")
            h_sum += _float(r.get("hint_count", 0), column="hint_count")
            a_sum += _float(r.get("attempt_count", 1), column="attempt_count")
            t_sum += _float(r.get("ms_first_response", 0), column="ms_first_response")
            frust_sum += _float(
                r.get("Average_confidence(FRUSTRATED)", 0),
                column="Average_confidence(FRUSTRATED)",
            )
            eng_sum += _float(
                r.get("Average_confidence(CONCENTRATING)", 0),
                column="Average_confidence(CONCENTRATING)",
            )
        out[skill] = SkillStats(
            skill=skill,
            n_records=n,
            mean_correctness=c_sum / n,
            mean_hint_count=h_sum / n,
            mean_attempts=a_sum / n,
            mean_response_time_ms=t_sum / n,
            mean_frustrated=frust_sum / n,
            mean_engaged=eng_sum / n,
        )
    return out


def _emotion_from_row(row: dict[str, Any]) -> dict[str, float]:
    """Map ASSISTments confidence columns to the FER channel names.

    The ASSISTments columns we care about are:

    * ``CONCENTRATING`` -> ``engaged``
    * ``CONFUSED``      -> ``confused``
    * ``BORED``         -> ``bored``
    * ``FRUSTRATED``    -> ``frustrated``

    Each value is already a probability in ``[0, 1]``; we clip
    defensively.
    """
    return {
        "engaged": _clip01(
            _float(
                row.get("Average_confidence(CONCENTRATING)", 0),
                column="Average_confidence(CONCENTRATING)",
            )
        ),
        "confused": _clip01(
            _float(
                row.get("Average_confidence(CONFUSED)", 0),
                column="Average_confidence(CONFUSED)",
            )
        ),
        "bored": _clip01(
            _float(row.get("Average_confidence(BORED)", 0), column="Average_confidence(BORED)")
        ),
        "frustrated": _clip01(
            _float(
                row.get("Average_confidence(FRUSTRATED)", 0),
                column="Average_confidence(FRUSTRATED)",
            )
        ),
    }


def _label_action(
    correct: int,
    hint_count: int,
    attempt_count: int,
    emotion: dict[str, float],
) -> str:
    """Deterministic rule-based pedagogical action labeller.

    Order matters: earlier rules win. The order matches the project
    spec's rationale - emotion overrides only when no clear
    behavioural pattern dominates.
    """
    if correct == 1 and hint_count > 0:
        return "give_hint"
    if correct == 1 and attempt_count > 1:
        return "retry_current_skill"
    if correct == 0 and hint_count > 1:
        return "easier_problem"
    if correct == 1 and hint_count == 0 and attempt_count == 1:
        return "harder_problem"
    if emotion.get("frustrated", 0.0) > 0.5:
        return "encouragement"
    return "advance_to_next_skill"


def skill_stats_as_tuple(st: SkillStats) -> tuple[tuple[str, float], ...]:
    """Pack :class:`SkillStats` for :class:`~adaptive_tutor.state.state.LearnerState.dataset_stats`.

    Keys are stable strings the state tensor / explainer can document.
    ``n_records`` is stored as float for a uniform tuple type; cast to
    ``int`` when needed.
    """
    return (
        ("historical_success_rate", float(st.mean_correctness)),
        ("mean_hint_count", float(st.mean_hint_count)),
        ("mean_attempts", float(st.mean_attempts)),
        ("mean_response_time_ms", float(st.mean_response_time_ms)),
        ("mean_frustrated", float(st.mean_frustrated)),
        ("mean_engaged", float(st.mean_engaged)),
        ("n_records", float(st.n_records)),
    )


def _initial_state_dict(slug: str) -> dict[str, Any]:
    """Cold-start state at the first row of a user-skill group."""
    return {
        "mastery": 0.0,
        "recent_accuracy": 0.0,
        "hint_usage": 0.0,
        "attempts": 0,
        "emotion": {
            "engaged": 0.0,
            "confused": 0.0,
            "bored": 0.0,
            "frustrated": 0.0,
        },
        "current_skill": slug,
    }


def _difficulty_from_correctness(mean_correctness: float) -> float:
    """Map mean correctness to a difficulty in ``[0, 1]``.

    Convention: higher correctness -> easier skill, so difficulty is
    ``1 - mean_correctness``. Clamped defensively.
    """
    return _clip01(1.0 - float(mean_correctness))


def _slugify(name: str) -> str:
    """Lossy snake_case slugifier matching the base provider's regex.

    Rules:

    * lower-case ascii letters and digits only
    * runs of non-alphanumeric characters collapse to one underscore
    * leading/trailing underscores stripped
    * if the result is empty or starts with a digit, prefix with ``"skill_"``
    """
    out: list[str] = []
    last_under = False
    for ch in name.lower():
        if ch.isalnum() and ch.isascii():
            out.append(ch)
            last_under = False
        else:
            if not last_under:
                out.append("_")
                last_under = True
    slug = "".join(out).strip("_")
    if not slug:
        return "skill_unknown"
    if slug[0].isdigit():
        return f"skill_{slug}"
    return slug


def assistments_skill_slug(skill_label: str) -> str:
    """Public wrapper for joining teacher ``skills`` entries to dataset keys."""
    return _slugify(skill_label)


def _float(v: Any, *, column: str | None = None) -> float:
    """Best-effort float coercion that never raises.

    Args:
        v:      The value to coerce.
        column: Optional CSV column name. When provided, the first
                malformed value for that column produces a single
                ``WARNING`` log entry; subsequent malformed values in
                the same column are coerced silently. This keeps the
                provider robust to dirty CSVs without sacrificing
                visibility.

    Returns:
        ``float(v)`` on success; ``0.0`` on ``TypeError``/``ValueError``.
        The return value is **deterministic** regardless of logging.
    """
    try:
        return float(v)
    except (TypeError, ValueError):
        if column is not None and column not in _WARNED_MALFORMED_COLUMNS:
            _WARNED_MALFORMED_COLUMNS.add(column)
            _LOG.warning(
                "malformed numeric value %r in column %s -> coerced to 0",
                v,
                column,
            )
        return 0.0


def _reset_malformed_warnings() -> None:
    """Test hook: clear the module-level dedup set so unit tests can
    observe the warning for a freshly-named column. Safe to call from
    application code too; the worst case is one extra WARNING per
    column."""
    _WARNED_MALFORMED_COLUMNS.clear()


def _clip01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))
