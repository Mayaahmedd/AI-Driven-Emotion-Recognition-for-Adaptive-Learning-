"""Sequential action filter orchestrator (Phase 7).

Order (**fixed**, thesis-documented):

1. :mod:`adaptive_tutor.masking.action_mask` - curriculum + dataset awareness
2. :mod:`adaptive_tutor.masking.cooldown` - anti-spam delays
3. :mod:`adaptive_tutor.masking.prerequisites` - DAG mastery gate
4. :mod:`adaptive_tutor.masking.whipsaw` - pacing stability (two passes)

If the candidate action drops out, we deterministically pick ``sorted(allowed)[0]``.
If ``allowed`` is empty, fall back to ``encouragement`` then ``retry_current_skill``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from adaptive_tutor.memory.providers.base import BaseCurriculumProvider
from adaptive_tutor.memory.providers.dataset_provider import ASSISTMENTS_ACTIONS
from adaptive_tutor.masking.action_mask import mask_by_curriculum_and_dataset
from adaptive_tutor.masking.cooldown import CooldownTracker
from adaptive_tutor.masking.prerequisites import filter_by_prerequisites
from adaptive_tutor.masking.whipsaw import WhipsawTracker
from adaptive_tutor.state.state import LearnerState


@dataclass(frozen=True, slots=True)
class ActionFilterContext:
    """Immutable inputs for one filter call (except ``mastery_by_slug`` is read-only view)."""

    teacher: BaseCurriculumProvider | None
    concept_slug: str
    dataset_skill_slug: str | None
    dataset_slugs: frozenset[str] | None
    strict_dataset: bool
    mastery_by_slug: Mapping[str, float]
    prereq_mastery_threshold: float = 0.6


def _project_candidate(candidate: str, allowed: frozenset[str]) -> tuple[str, tuple[str, ...]]:
    """Pick executable action and note if we deviated from the candidate."""
    if candidate in allowed:
        return candidate, ()
    if not allowed:
        for fallback in ("encouragement", "retry_current_skill", ASSISTMENTS_ACTIONS[0]):
            if fallback in ASSISTMENTS_ACTIONS:
                return fallback, ("filter_pipeline:emergency_fallback",)
        return ASSISTMENTS_ACTIONS[0], ("filter_pipeline:emergency_fallback",)
    chosen = sorted(allowed)[0]
    return chosen, (f"filter_pipeline:clamped_to:{chosen}",)


@dataclass(frozen=True, slots=True)
class ActionFilterTrace:
    """Structured Phase 7 audit flags for one filter call (before emergency recovery).

    Booleans refer to the **requested** ``candidate`` action. ``prerequisites_met``
    means the prerequisite stage did not remove the candidate (given the
    candidate survived earlier stages).
    """

    mask_passed: bool
    cooldown_blocked: bool
    prerequisites_met: bool
    whipsaw_blocked: bool

    def to_json_dict(self) -> dict[str, bool]:
        return {
            "mask_passed": self.mask_passed,
            "cooldown_blocked": self.cooldown_blocked,
            "prerequisites_met": self.prerequisites_met,
            "whipsaw_blocked": self.whipsaw_blocked,
        }


def run_filter_pipeline(
    candidate: str,
    state_pre: LearnerState,
    *,
    context: ActionFilterContext,
    cooldown: CooldownTracker,
    whipsaw: WhipsawTracker,
) -> tuple[str, frozenset[str], tuple[str, ...], ActionFilterTrace]:
    """Return ``(chosen_action, final_allowed_set, audit_trail, trace)``."""
    reasons: list[str] = []

    allowed_m, rs = mask_by_curriculum_and_dataset(
        ASSISTMENTS_ACTIONS,
        concept_slug=context.concept_slug,
        teacher=context.teacher,
        dataset_slugs=context.dataset_slugs,
        dataset_skill_slug=context.dataset_skill_slug,
        strict_dataset=context.strict_dataset,
    )
    reasons.extend(rs)
    mask_passed = candidate in allowed_m

    allowed_cd, rs = cooldown.filter(allowed_m, state_pre.timestep)
    reasons.extend(rs)
    cooldown_blocked = candidate in allowed_m and candidate not in allowed_cd

    allowed_pr, rs = filter_by_prerequisites(
        allowed_cd,
        teacher=context.teacher,
        concept_slug=context.concept_slug,
        mastery_by_slug=context.mastery_by_slug,
        threshold=context.prereq_mastery_threshold,
    )
    reasons.extend(rs)
    prerequisites_met = not (candidate in allowed_cd and candidate not in allowed_pr)

    allowed_w1, rs = whipsaw.opposite_pair_filter(allowed_pr)
    reasons.extend(rs)

    allowed_w2, rs = whipsaw.filter(allowed_w1)
    reasons.extend(rs)

    whipsaw_blocked = candidate in allowed_pr and candidate not in allowed_w2

    trace = ActionFilterTrace(
        mask_passed=mask_passed,
        cooldown_blocked=cooldown_blocked,
        prerequisites_met=prerequisites_met,
        whipsaw_blocked=whipsaw_blocked,
    )

    allowed = allowed_w2
    if not allowed:
        reasons.append("filter_pipeline:allowed_empty_recover")
        allowed = frozenset({"encouragement", "retry_current_skill"}) & frozenset(
            ASSISTMENTS_ACTIONS
        )
        if not allowed:
            allowed = frozenset(ASSISTMENTS_ACTIONS[:1])

    chosen, extra = _project_candidate(candidate, allowed)
    reasons.extend(extra)
    return chosen, allowed, tuple(reasons), trace
