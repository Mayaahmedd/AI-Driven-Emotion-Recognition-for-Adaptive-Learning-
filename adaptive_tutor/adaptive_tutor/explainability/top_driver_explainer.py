"""Dataset-grounded + rule-based explainer (Phase 10 / thesis).

When :class:`~adaptive_tutor.memory.providers.dataset_provider.SkillStats` is
supplied, top drivers and narrative **must** cite cohort frequencies from the
ASSISTments slice. Rolling :class:`LearnerState` and Phase 7 flags refine the
story; they never replace missing dataset evidence.

When ``skill_stats`` is omitted, the explainer falls back to the legacy
state-only heuristic (demos / dry runs without a CSV join).
"""

from __future__ import annotations

from collections.abc import Mapping

from adaptive_tutor.explainability.schema import ExplanationRecord, Phase7Checks, TopDriver
from adaptive_tutor.memory.providers.dataset_provider import SkillStats
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


def _dataset_evidence(st: SkillStats) -> dict[str, float]:
    """Numeric facts attached to every grounded explanation."""
    mc = float(st.mean_correctness)
    diff = max(0.05, min(0.95, 1.0 - mc))
    return {
        "skill_correctness": mc,
        "skill_error_rate": 1.0 - mc,
        "hint_rate": float(st.mean_hint_count),
        "mean_attempts": float(st.mean_attempts),
        "n_records": float(st.n_records),
        "difficulty_proxy": float(diff),
        "mean_frustration_proxy": float(st.mean_frustrated),
        "mean_engagement_proxy": float(st.mean_engaged),
    }


def _dataset_driver_messages(
    st: SkillStats,
    state: LearnerState,
    action_executed: str,
) -> list[tuple[str, float]]:
    """Human-readable strings with unsigned strengths (weighted later)."""
    ev = _dataset_evidence(st)
    lines: list[tuple[str, float]] = []

    lines.append(
        (
            f"ASSISTments historical correctness for this skill: {ev['skill_correctness']:.2f} "
            f"(error rate {ev['skill_error_rate']:.2f}, n={int(ev['n_records'])} rows).",
            0.35 + 0.5 * ev["skill_error_rate"],
        )
    )
    lines.append(
        (
            f"Students in this cohort average {ev['hint_rate']:.2f} hints and "
            f"{ev['mean_attempts']:.2f} attempts per logged interaction on this skill.",
            0.2 + 0.25 * min(1.0, ev["hint_rate"] / 3.0),
        )
    )
    lines.append(
        (
            f"Skill difficulty proxy from data (1 - mean correctness): {ev['difficulty_proxy']:.2f}.",
            0.15 + 0.35 * ev["difficulty_proxy"],
        )
    )
    if st.mean_frustrated > 0.25:
        lines.append(
            (
                f"Historical frustration channel mean in logs: {st.mean_frustrated:.2f}.",
                0.2 + st.mean_frustrated,
            )
        )

    m = float(state.mastery)
    if m + 1e-6 < st.mean_correctness:
        lines.append(
            (
                f"Current rolling mastery ({m:.2f}) sits below the cohort mean ({st.mean_correctness:.2f}), "
                "so support-heavy moves match the data profile.",
                0.25 + (st.mean_correctness - m),
            )
        )
    elif action_executed in ("harder_problem", "advance_to_next_skill") and m >= st.mean_correctness:
        lines.append(
            (
                f"Rolling mastery ({m:.2f}) meets or exceeds cohort mean ({st.mean_correctness:.2f}); "
                "advance-style actions align with students who succeeded in logs.",
                0.2 + (m - st.mean_correctness),
            )
        )

    return lines


def _normalize_message_drivers(
    rows: list[tuple[str, float]],
    *,
    k: int = 6,
) -> list[TopDriver]:
    items = sorted(rows, key=lambda kv: kv[1], reverse=True)[:k]
    total = sum(w for _, w in items)
    if total <= 1e-9:
        return [TopDriver(feature="Insufficient dataset signal for weighted drivers.", weight=1.0)]
    return [
        TopDriver(feature=msg, weight=round(float(w) / total, 4)) for msg, w in items
    ]


def _driver_scores(
    state: LearnerState,
    action_executed: str,
    phase7: Mapping[str, bool] | None,
) -> dict[str, float]:
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

    if action_executed == "give_hint":
        s["hint_addressing_struggle"] = 0.2 + s["low_mastery"] + s["high_confusion"]
    elif action_executed == "encouragement":
        s["encouragement_for_affect"] = 0.25 + s["high_frustration"]
    elif action_executed in ("harder_problem", "advance_to_next_skill"):
        s["advance_when_ready"] = float(state.mastery) + 0.1 * (1.0 - s["high_frustration"])
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
    dataset_grounded: bool,
) -> str:
    parts: list[str] = []
    if dataset_grounded:
        parts.append(
            "Explanation ties claims to ASSISTments cohort statistics "
            "(see dataset_evidence and top_drivers)."
        )
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

    if not dataset_grounded and drivers:
        d0 = drivers[0].feature
        if d0 in ("low_mastery", "hint_addressing_struggle"):
            parts.append("Support-heavy action aligns with weak mastery (state heuristic).")
        if d0 in ("high_confusion", "support_after_failure"):
            parts.append("Learner confusion supports explanatory or easier pacing (state heuristic).")
        if d0 == "high_frustration":
            parts.append("Frustration supports calming or encouragement (state heuristic).")

    if not parts:
        parts.append(f"Action '{action_executed}' is consistent with Phase 7 checks.")
    return " ".join(parts)


def explain_action(
    state: LearnerState,
    action_executed: str,
    *,
    action_requested: str | None = None,
    phase7: Mapping[str, bool] | None = None,
    r_components: tuple[tuple[str, float], ...] | None = None,
    correct: int | None = None,
    skill_stats: SkillStats | None = None,
) -> dict[str, object]:
    """JSON-serialisable explanation; grounded in ``skill_stats`` when provided."""
    req = action_requested if action_requested is not None else action_executed
    p7 = _phase7(phase7)
    dataset_evidence: dict[str, float] = {}
    dataset_grounded = skill_stats is not None

    if skill_stats is not None:
        dataset_evidence = _dataset_evidence(skill_stats)
        msgs = _dataset_driver_messages(skill_stats, state, action_executed)
        raw = _driver_scores(state, action_executed, phase7)
        for k, v in raw.items():
            if v <= 0:
                continue
            msgs.append((f"Observed live state / Phase 7 cue: {k.replace('_', ' ')}.", 0.08 * v))
        drivers = _normalize_message_drivers(msgs, k=6)
    else:
        raw = _driver_scores(state, action_executed, phase7)
        drivers = _normalize_top_drivers(raw)

    text = _compose_text(
        action_executed=action_executed,
        action_requested=req,
        phase7=p7,
        drivers=drivers,
        dataset_grounded=dataset_grounded,
    )
    rec = ExplanationRecord(
        action=action_executed,
        action_requested=req,
        top_drivers=drivers,
        phase7_checks=p7,
        reward_signal=_reward_signal(r_components, correct=correct),
        explanation_text=text,
        dataset_evidence=dataset_evidence,
    )
    out: dict[str, object] = dict(rec.model_dump())
    out["phase7_flags"] = dict(p7.model_dump())
    out["reward_components"] = dict(rec.reward_signal)
    return out


def explain_action_from_env_info(
    state_pre_action: LearnerState,
    info: Mapping[str, object],
    *,
    skill_stats: SkillStats | None = None,
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
        skill_stats=skill_stats,
    )
