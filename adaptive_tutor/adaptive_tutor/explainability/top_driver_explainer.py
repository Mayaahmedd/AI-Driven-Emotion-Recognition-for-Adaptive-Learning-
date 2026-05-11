"""Rule-based top-driver explainer (Phase 10).

No learned attribution: fixed transparent scoring from :class:`LearnerState`,
Phase 7 flags, and :class:`~adaptive_tutor.rewards.RewardEngine` components.
"""

from __future__ import annotations

from collections.abc import Mapping

from adaptive_tutor.explainability.schema import ExplanationRecord, Phase7Checks, TopDriver
from adaptive_tutor.state.state import LearnerState


def _reward_signal(
    r_components: tuple[tuple[str, float], ...] | None,
    *,
    correct: int | None,
) -> dict[str, float]:
    comp = dict(r_components or ())
    out = {
        "correctness": float(comp.get("correctness", 0.0)),
        "frustration": float(comp.get("frustration", 0.0)),
        "hint_penalty": float(comp.get("hint_penalty", 0.0)),
        "engagement": float(comp.get("engagement", 0.0)),
    }
    if correct is not None:
        out["correct"] = float(correct)
    return out


def _phase7(phase7: Mapping[str, bool] | None) -> Phase7Checks:
    p = dict(phase7 or {})
    return Phase7Checks(
        mask_passed=bool(p.get("mask_passed", True)),
        cooldown_blocked=bool(p.get("cooldown_blocked", False)),
        prerequisites_met=bool(p.get("prerequisites_met", True)),
        whipsaw_blocked=bool(p.get("whipsaw_blocked", False)),
    )


def _driver_scores(
    state: LearnerState,
    action_executed: str,
    phase7: Mapping[str, bool] | None,
) -> dict[str, float]:
    """Unsigned rule strengths (interpretable, deterministic)."""
    s: dict[str, float] = {}
    s["low_mastery"] = max(0.0, 0.5 - float(state.mastery))
    s["high_confusion"] = float(state.rolling_emotions.confused)
    s["high_frustration"] = float(state.rolling_emotions.frustrated)
    s["low_engagement"] = max(0.0, 0.45 - float(state.rolling_emotions.engaged))

    p = phase7 or {}
    if not p.get("mask_passed", True):
        s["mask_or_curriculum_constraint"] = 0.45
    if p.get("cooldown_blocked"):
        s["cooldown_active"] = 0.35
    if not p.get("prerequisites_met", True):
        s["prerequisite_failure"] = 0.55
    if p.get("whipsaw_blocked"):
        s["anti_whipsaw_block"] = 0.28

    # Tie-breakers: why this opcode is coherent with state (rules, not Q).
    if action_executed == "give_hint":
        s["hint_addressing_struggle"] = 0.2 + s["low_mastery"] + s["high_confusion"]
    elif action_executed == "encouragement":
        s["encouragement_for_affect"] = 0.25 + s["high_frustration"]
    elif action_executed in ("harder_problem", "advance_to_next_skill"):
        s["advance_when_ready"] = float(state.mastery) + 0.1 * (
            1.0 - s["high_frustration"]
        )
    elif action_executed in ("easier_problem", "retry_current_skill"):
        s["support_after_failure"] = 0.2 + s["high_confusion"]
    return s


def _normalize_top_drivers(raw: dict[str, float], *, k: int = 5) -> list[TopDriver]:
    items = sorted(raw.items(), key=lambda kv: kv[1], reverse=True)[:k]
    total = sum(v for _, v in items)
    if total <= 1e-9:
        return [TopDriver(feature="flat_state", weight=1.0)]
    return [
        TopDriver(feature=name, weight=round(float(val) / total, 4)) for name, val in items
    ]


def _compose_text(
    *,
    action_executed: str,
    action_requested: str,
    phase7: Phase7Checks,
    drivers: list[TopDriver],
) -> str:
    parts: list[str] = []
    if action_executed != action_requested:
        parts.append(
            f"Executed '{action_executed}' instead of '{action_requested}' after Phase 7."
        )
    if not phase7.mask_passed:
        parts.append("The curriculum or dataset mask rejected the request.")
    if phase7.cooldown_blocked:
        parts.append("Cooldown blocked an immediate repeat.")
    if not phase7.prerequisites_met:
        parts.append("Prerequisite mastery blocked escalation-like moves.")
    if phase7.whipsaw_blocked:
        parts.append("Anti-whipsaw pacing blocked a rapid pace reversal.")
    d0 = drivers[0].feature if drivers else ""
    if d0 in ("low_mastery", "hint_addressing_struggle"):
        parts.append("Support-heavy action aligns with weak mastery.")
    if d0 in ("high_confusion", "support_after_failure"):
        parts.append("Learner confusion supports explanatory or easier pacing.")
    if d0 == "high_frustration":
        parts.append("Frustration supports calming or encouragement.")
    if not parts:
        parts.append(
            f"Action '{action_executed}' matches stable state and Phase 7 checks."
        )
    return " ".join(parts)


def explain_action(
    state: LearnerState,
    action_executed: str,
    *,
    action_requested: str | None = None,
    phase7: Mapping[str, bool] | None = None,
    r_components: tuple[tuple[str, float], ...] | None = None,
    correct: int | None = None,
) -> dict[str, object]:
    """Build a validated JSON-serialisable explanation (policy-agnostic).

    ``state`` should be the learner snapshot **before** the env transition
    (the same view the policy used).
    """
    req = action_requested if action_requested is not None else action_executed
    p7 = _phase7(phase7)
    raw = _driver_scores(state, action_executed, phase7)
    drivers = _normalize_top_drivers(raw)
    text = _compose_text(
        action_executed=action_executed,
        action_requested=req,
        phase7=p7,
        drivers=drivers,
    )
    rec = ExplanationRecord(
        action=action_executed,
        action_requested=req,
        top_drivers=drivers,
        phase7_checks=p7,
        reward_signal=_reward_signal(r_components, correct=correct),
        explanation_text=text,
    )
    return rec.model_dump()


def explain_action_from_env_info(
    state_pre_action: LearnerState,
    info: Mapping[str, object],
) -> dict[str, object]:
    """Convenience when reusing :class:`TutoringEnvironment.step` ``info``."""
    comp = info.get("r_components")
    tpl: tuple[tuple[str, float], ...] = ()
    if isinstance(comp, tuple):
        tpl = tuple((str(a), float(b)) for a, b in comp)
    p7 = info.get("action_filter_trace")
    p7_map = p7 if isinstance(p7, dict) else None
    cr = info.get("correct")
    cor: int | None = None
    if isinstance(cr, bool):
        cor = int(cr)
    elif isinstance(cr, int):
        cor = cr
    return explain_action(
        state_pre_action,
        str(info["action"]),
        action_requested=str(info.get("action_requested", info["action"])),
        phase7=p7_map,
        r_components=tpl,
        correct=cor,
    )
