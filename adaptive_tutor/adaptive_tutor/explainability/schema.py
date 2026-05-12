"""Frozen Pydantic models for Phase 10 explanation JSON (thesis-facing)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TopDriver(BaseModel):
    """One ranked rule-based driver."""

    model_config = ConfigDict(frozen=True)

    feature: str
    weight: float = Field(..., ge=-1.0, le=1.0)


class Phase7Checks(BaseModel):
    model_config = ConfigDict(frozen=True)

    mask_passed: bool
    cooldown_blocked: bool
    prerequisites_met: bool
    whipsaw_blocked: bool


class ExplanationRecord(BaseModel):
    """Validated explanation object returned by :func:`explain_action`."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    action: str
    action_requested: str
    top_drivers: list[TopDriver]
    phase7_checks: Phase7Checks
    reward_signal: dict[str, float]
    explanation_text: str
    dataset_evidence: dict[str, float] = Field(default_factory=dict)


def validate_explanation_dict(data: dict[str, Any]) -> ExplanationRecord:
    """Strict parse for saved artefacts and tests (ignores API aliases)."""
    keys = (
        "action",
        "action_requested",
        "top_drivers",
        "phase7_checks",
        "reward_signal",
        "explanation_text",
        "dataset_evidence",
    )
    slim = {k: data[k] for k in keys if k in data}
    return ExplanationRecord.model_validate(slim)
