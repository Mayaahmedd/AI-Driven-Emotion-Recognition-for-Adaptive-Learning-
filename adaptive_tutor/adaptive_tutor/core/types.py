"""Frozen core dataclasses shared by every sub-system.

This module is intentionally tiny. It now holds only:

* :class:`EmotionVector` - the multilabel FER output (the one piece of
  state that is genuinely a primitive, shared by FER cache, state
  builder, simulator, and explainer);
* :class:`MacroAction`, :class:`MesoAction`, and
  :class:`CompositeAction` (PPO macro + DQN meso; no separate bandit
  action type in the bachelor-thesis stack).

Everything *learner-state-shaped* (``LearnerState``,
``PerformanceFeatures``, ``Transition``) lives in
``adaptive_tutor.state.state`` so the state's shape can evolve without
forcing core to follow.

Design rules (in force across the codebase)
-------------------------------------------
* **Frozen.** Every dataclass here is immutable (``frozen=True``).
* **Slots.** ``slots=True`` cuts per-instance memory.
* **No torch.** Zero PyTorch imports.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Emotion vector — matches FER multilabel output order exactly.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EmotionVector:
    """Multilabel FER output. Each component is a probability in ``[0, 1]``.

    The four components are independent (multilabel, not multiclass), so
    they need not sum to 1. The order here matches the trained FER head:
    ``ResNet50BiLSTM(num_classes=4)`` outputs logits in this order:

        index 0: engaged
        index 1: confused
        index 2: bored
        index 3: frustrated

    See ``FER_Module/src/models/model.py`` (the model used in the existing
    pipeline) and ``adaptive_tutor.fer.client`` for the wrapper that
    applies the sigmoid + cache.
    """

    engaged: float
    confused: float
    bored: float
    frustrated: float

    def __post_init__(self) -> None:
        for name, v in (
            ("engaged", self.engaged),
            ("confused", self.confused),
            ("bored", self.bored),
            ("frustrated", self.frustrated),
        ):
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"EmotionVector.{name}={v} out of [0,1]")

    def as_tuple(self) -> tuple[float, float, float, float]:
        """Convenience: ``(engaged, confused, bored, frustrated)``."""
        return (self.engaged, self.confused, self.bored, self.frustrated)

    @staticmethod
    def neutral() -> EmotionVector:
        """A zero vector — used as the cold-start FER reading."""
        return EmotionVector(0.0, 0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Action types - factorized per controller.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MacroAction:
    """Output of the PPO curriculum manager.

    Fires only at concept boundaries (see ADR-003). ``instruction_type``
    and ``concept_id`` are integer ids whose semantics are defined by the
    active ``CurriculumProvider``.
    """

    instruction_type: int
    concept_id: int


@dataclass(frozen=True, slots=True)
class MesoAction:
    """Factorised DQN tutoring action (Phase 8).

    * ``content_variant`` - instruction strategy head: 0 explain, 1 hint,
      2 encourage, 3 skip.
    * ``pace`` - difficulty head: 0 easier, 1 same, 2 harder.
    * ``ui_variant`` - progression head: 0 stay, 1 advance.
    * ``emotional_support`` - reserved (0) for future affect head.
    """

    content_variant: int
    pace: int
    ui_variant: int
    emotional_support: int


@dataclass(frozen=True, slots=True)
class CompositeAction:
    """Joint action applied at one environment step.

    * ``macro`` - optional PPO curriculum decision (concept boundary).
    * ``meso`` - required DQN / local adaptation decision (always present).

    Bandit-style ``micro`` actions were removed: local adaptation is
    handled entirely through :class:`MesoAction` and the shared action
    vocabulary used by the simulator and dataset labeller.
    """

    macro: MacroAction | None
    meso: MesoAction

    def __post_init__(self) -> None:
        if self.meso is None:  # pragma: no cover - dataclass would already error
            raise ValueError("CompositeAction.meso is required")
