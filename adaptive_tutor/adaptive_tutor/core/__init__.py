"""adaptive_tutor.core - frozen contracts everyone else codes against.

This package is the *bottom* of the project dependency DAG. It MUST NOT
import from any other ``adaptive_tutor.*`` package. CI enforces this rule.

Public surface:

    from adaptive_tutor.core import (
        EmotionVector,
        MacroAction, MesoAction, CompositeAction,
        Policy, Critic, Replay, RewardComponent,
        CurriculumProvider, ActionMaskBuilder, Explainer,
        Registry, register, build_from_config,
        AdaptiveTutorError, ContractError, MaskError,
    )

Note: ``LearnerState``, ``PerformanceFeatures`` and ``Transition`` now
live in ``adaptive_tutor.state.state`` (see ADR-006 / Phase 3). Import
them from there.
"""

from adaptive_tutor.core.exceptions import (
    AdaptiveTutorError,
    ContractError,
    MaskError,
    SchemaError,
)
from adaptive_tutor.core.protocols import (
    ActionMaskBuilder,
    Critic,
    CurriculumProvider,
    Explainer,
    Policy,
    Replay,
    RewardComponent,
)
from adaptive_tutor.core.registry import Registry, build_from_config, register
from adaptive_tutor.core.types import (
    CompositeAction,
    EmotionVector,
    MacroAction,
    MesoAction,
)

__all__ = [
    "ActionMaskBuilder",
    "AdaptiveTutorError",
    "CompositeAction",
    "ContractError",
    "Critic",
    "CurriculumProvider",
    "EmotionVector",
    "Explainer",
    "MacroAction",
    "MaskError",
    "MesoAction",
    "Policy",
    "Registry",
    "Replay",
    "RewardComponent",
    "SchemaError",
    "build_from_config",
    "register",
]
