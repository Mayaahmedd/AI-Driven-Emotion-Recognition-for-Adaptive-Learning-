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

    model_config = ConfigDict(frozen=True)

    action: str
    action_requested: str
    top_drivers: list[TopDriver]
    phase7_checks: Phase7Checks
    reward_signal: dict[str, float]
    explanation_text: str


def validate_explanation_dict(data: dict[str, Any]) -> ExplanationRecord:
    """Strict parse for saved artefacts and tests."""
    return ExplanationRecord.model_validate(data)
