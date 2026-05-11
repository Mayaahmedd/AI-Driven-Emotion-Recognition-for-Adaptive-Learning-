"""Explainability package (Phase 10)."""

from adaptive_tutor.explainability.schema import (
    ExplanationRecord,
    Phase7Checks,
    TopDriver,
    validate_explanation_dict,
)
from adaptive_tutor.explainability.top_driver_explainer import (
    explain_action,
    explain_action_from_env_info,
)

__all__ = [
    "ExplanationRecord",
    "Phase7Checks",
    "TopDriver",
    "explain_action",
    "explain_action_from_env_info",
    "validate_explanation_dict",
]
