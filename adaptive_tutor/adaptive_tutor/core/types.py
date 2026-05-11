"""Frozen core dataclasses shared by every sub-system.

Design rules (in force across the codebase)
-------------------------------------------
* **Frozen.** Every dataclass here is immutable (``frozen=True``). The
  state representation passed to policies, critics, replay buffers, and
  loggers is *never* mutated in place. Producers create a new instance;
  consumers must not assume reference equality across producers.
* **Slots.** ``slots=True`` cuts per-instance memory by ~40% and makes
  these dataclasses friendly to vectorized batching later.
* **No torch.** This module has *zero* PyTorch imports so it can be used
  from offline-eval scripts and unit tests in minimal environments. Tensor
  conversion happens at the boundary (in ``state/builder.py``, Phase 4).
* **Explicit shapes.** ``mastery`` is a tuple of floats whose length is
  ``|C|`` (number of concepts in the active curriculum). It is *not* a
  NumPy array, again to keep this module dep-light. The state builder
  vectorises it.
* **Sequence vs Tuple.** We use ``tuple[T, ...]`` rather than
  ``Sequence[T]`` so that ``frozen=True`` + ``__hash__`` works on
  ``LearnerState`` (a Sequence is not hashable). This matters because
  some explainability caches key on the state.

Naming
------
We deliberately use *short* attribute names (``s``, ``a``, ``r``,
``s_next``, ``done``) in ``Transition`` because they appear thousands of
times per second in the inner training loop and longer names hurt
readability of the call sites.

Validation
----------
Each dataclass exposes a ``__post_init__`` that validates ranges. This is
not free, but it has caught bugs in earlier prototypes (e.g., FER vectors
escaping ``[0, 1]`` after a normalization regression). For performance-
critical inner loops, callers can use the lower-level numeric pipeline
in ``state/builder.py`` which bypasses construction of these objects.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

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
# Performance features — short-horizon interpretable signals.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PerformanceFeatures:
    """Performance-side features at the current learner state.

    Attributes
    ----------
    correctness_rolling:
        Rolling-window mean correctness in ``[0, 1]``. Window length is a
        config knob (default 8, see ``configs/base.yaml``).
    attempts_last_item:
        Number of attempts the learner made on the most recently completed
        item. ``0`` means the item was answered on the first try.
    hint_level_last_item:
        Hint level (0 = none, 1 = nudge, 2 = scaffold, 3 = worked example)
        revealed on the last item.
    response_time_z:
        Z-scored response time on the last item. Standardization is done
        per-cohort by the simulator in v1 and per-learner online later.
    streak_correct:
        Length of the current consecutive-correct streak.

    Why these specific fields
    -------------------------
    We deliberately keep this set small and interpretable. The deep
    network in ``state/builder.py`` can add richer learned features on top
    of these, but the human-readable rolling stats are kept here so the
    explainer (Phase 12) can quote them in natural-language rationales.
    """

    correctness_rolling: float
    attempts_last_item: int
    hint_level_last_item: int
    response_time_z: float
    streak_correct: int

    def __post_init__(self) -> None:
        if not (0.0 <= self.correctness_rolling <= 1.0):
            raise ValueError(
                f"correctness_rolling={self.correctness_rolling} out of [0,1]"
            )
        if self.attempts_last_item < 0:
            raise ValueError(f"attempts_last_item must be >= 0, got {self.attempts_last_item}")
        if not (0 <= self.hint_level_last_item <= 3):
            raise ValueError(
                f"hint_level_last_item must be in [0,3], got {self.hint_level_last_item}"
            )
        if self.streak_correct < 0:
            raise ValueError(f"streak_correct must be >= 0, got {self.streak_correct}")


# ---------------------------------------------------------------------------
# Learner state — the input to every policy/critic.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LearnerState:
    """Full state vector for the adaptation engine.

    This is what every controller (PPO / DQN / bandit) sees as input. The
    ``state/builder.py`` module (Phase 4) is responsible for converting
    this object to a flat tensor — but the object itself is the
    interpretable, *interchange* representation.

    Attributes
    ----------
    emotion:
        Current FER output (single frame's worth, post-sigmoid).
    emotion_summary:
        Rolling-window summary statistics over the last W FER readings.
        Tuple of two :class:`EmotionVector` instances: ``(mean, slope)``.
        Always present (decision 20.10 — both raw and summary).
    perf:
        Interpretable performance features at this step.
    mastery:
        ``|C|``-long tuple of per-concept mastery in ``[0, 1]``. Built
        from the memory subsystem (Phase 3).
    current_concept_id:
        Concept the learner is currently working on. ``-1`` when the
        session has not yet started.
    session_step:
        Zero-indexed env step within the current session.
    cooldowns:
        Mapping ``action_name -> steps_remaining``. Read by the safety /
        action-mask layer (Phase 7). Frozen as a ``tuple`` of pairs so
        the dataclass is hashable.
    extras:
        Open-ended extension slot for experimental features. Keep keys
        short and prefix with the originating subsystem
        (e.g. ``"sim.fatigue"``).
    """

    emotion: EmotionVector
    emotion_summary: tuple[EmotionVector, EmotionVector]
    perf: PerformanceFeatures
    mastery: tuple[float, ...]
    current_concept_id: int
    session_step: int
    cooldowns: tuple[tuple[str, int], ...] = ()
    extras: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if self.session_step < 0:
            raise ValueError(f"session_step must be >= 0, got {self.session_step}")
        if any(not (0.0 <= m <= 1.0) for m in self.mastery):
            bad = next(i for i, m in enumerate(self.mastery) if not (0.0 <= m <= 1.0))
            raise ValueError(f"mastery[{bad}]={self.mastery[bad]} out of [0,1]")
        for k, v in self.cooldowns:
            if v < 0:
                raise ValueError(f"cooldown {k}={v} must be >= 0")

    @property
    def num_concepts(self) -> int:
        return len(self.mastery)

    def cooldowns_dict(self) -> Mapping[str, int]:
        return dict(self.cooldowns)


# ---------------------------------------------------------------------------
# Action types — factorized per controller.
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
    """Output of the DQN tutoring policy.

    Factorized into four sub-actions (multi-head DQN, Phase 9). Each int
    indexes into the discrete option set for that head defined in the
    config.
    """

    content_variant: int
    pace: int
    ui_variant: int
    emotional_support: int


@dataclass(frozen=True, slots=True)
class MicroAction:
    """Output of the contextual Thompson-sampling bandit.

    Fires only when the bandit-drift trigger is hit (see master plan §10).
    """

    intervention_id: int


@dataclass(frozen=True, slots=True)
class CompositeAction:
    """The action actually applied to the env at one meso step.

    ``macro`` and ``micro`` may be ``None`` because they fire on coarser
    or event-driven schedules. ``meso`` is always present.

    Invariants (enforced in ``__post_init__``):
        * ``meso is not None``.
        * If ``macro is None`` and ``micro is None``, then this is a pure
          meso step — that is *valid* (most steps look like that).
    """

    macro: MacroAction | None
    meso: MesoAction
    micro: MicroAction | None

    def __post_init__(self) -> None:
        if self.meso is None:  # pragma: no cover - dataclass would already error
            raise ValueError("CompositeAction.meso is required")


# ---------------------------------------------------------------------------
# Transition — what a replay buffer stores.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Transition:
    """One env step seen by a controller.

    Attributes
    ----------
    s, a, s_next:
        Standard ``(s, a, s')`` transition tuple.
    r:
        The composite (scalar) reward. This is the sum of components
        weighted by ``rewards.lambda_*`` from the config.
    r_components:
        Per-component breakdown, e.g. ``{"performance": 0.4, "emotion":
        -0.1, "cost": -0.05}``. Used for attribution + ablations. Frozen
        as a tuple of pairs so :class:`Transition` is hashable; convert
        to a dict on read.
    done:
        Episode termination flag at ``s_next``.
    info:
        Free-form extras (e.g., ``"segment_id": 7`` for PPO macro
        bookkeeping). Always a tuple of (key, float-or-int) pairs.

    Sizing
    ------
    A typical training run produces 10^4-10^6 of these. Storing them as
    Python ``Transition`` objects is fine for development; the replay
    buffer in Phase 5 will provide a vectorized columnar variant for the
    hot path.
    """

    s: LearnerState
    a: CompositeAction
    r: float
    r_components: tuple[tuple[str, float], ...]
    s_next: LearnerState
    done: bool
    info: tuple[tuple[str, float], ...] = field(default_factory=tuple)

    def r_components_dict(self) -> Mapping[str, float]:
        return dict(self.r_components)

    def info_dict(self) -> Mapping[str, float]:
        return dict(self.info)
