"""Frozen state dataclasses for the RL stack.

What lives here
---------------
* :class:`PerformanceFeatures` - rolling performance summary
  (accuracy, hint rate, attempts).
* :class:`LearnerState` - the full interchange state passed to every
  controller and explainer.
* :class:`Transition` - one env step (``s, a, r, s', done, info``).

All three are frozen, slotted, hashable, validated. No torch import at
module level (the tensor conversion in :meth:`LearnerState.to_tensor`
imports torch lazily so the dataclass module remains usable in dep-
light environments).

Why these were moved out of ``core/types.py``
----------------------------------------------
Their *shape* is allowed to evolve as the thesis progresses (mastery
went from vector to scalar in the v1 reset, the rolling-window summary
expanded, dataset stats became optional). ``core/types.py`` should
contain only the genuinely stable primitives (``EmotionVector``,
action dataclasses). See ADR-006 section 3.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

from adaptive_tutor.core.types import CompositeAction, EmotionVector

if TYPE_CHECKING:
    import torch  # imported lazily inside to_tensor()


# ---------------------------------------------------------------------------
# Performance features - small, interpretable summary of the perf window.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PerformanceFeatures:
    """Rolling-window performance summary.

    Attributes
    ----------
    recent_accuracy:
        Mean correctness over the window, in ``[0, 1]``.
    hint_usage:
        Fraction of recent steps that used >= 1 hint, in ``[0, 1]``.
    attempts:
        Number of attempts the learner made on the most recent item
        (not a window mean - the latest count). Non-negative integer.

    Why only three fields
    ---------------------
    Anything else (response time, streak length, etc.) can be added to
    the optional ``dataset_stats`` slot on :class:`LearnerState` without
    forcing every downstream consumer to handle a new field. We keep
    this dataclass small for thesis clarity.
    """

    recent_accuracy: float
    hint_usage: float
    attempts: int

    def __post_init__(self) -> None:
        if not (0.0 <= self.recent_accuracy <= 1.0):
            raise ValueError(
                f"recent_accuracy={self.recent_accuracy} out of [0,1]"
            )
        if not (0.0 <= self.hint_usage <= 1.0):
            raise ValueError(f"hint_usage={self.hint_usage} out of [0,1]")
        if self.attempts < 0:
            raise ValueError(f"attempts must be >= 0, got {self.attempts}")


# ---------------------------------------------------------------------------
# LearnerState - the interchange state the RL stack consumes.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LearnerState:
    """The state the RL stack reads at every step.

    Shape rationale
    ---------------
    * **Scalar mastery.** Running mean correctness for the *current*
      concept only. The curriculum manager (PPO) gets a curriculum-wide
      view by querying the StateBuilder separately, not by inflating
      the state.
    * **EmotionVector mean** as the rolling emotion summary, plus four
      slope scalars for trend. This keeps the emotion side small (5
      numbers in the state, 8 in the tensor with the current emotion
      separated).
    * **Optional ``dataset_stats``.** Stored as a tuple of
      ``(key, value)`` pairs so the dataclass stays hashable. The
      orchestrator populates it from
      :meth:`DatasetCurriculumProvider.all_stats`; if empty, the agent
      simply does not use those features.

    Attributes
    ----------
    current_concept_id:
        Slug string (``"basic_probability"``). Stable across runs.
    current_concept_index:
        Integer index into the curriculum's topological order.
        Convenient for tensor encoding.
    mastery:
        Running mean correctness for ``current_concept_id`` in
        ``[0, 1]``.
    perf:
        Rolling performance summary (see :class:`PerformanceFeatures`).
    rolling_emotions:
        EmotionVector window mean.
    engagement_trend, confusion_trend, frustration_trend,
    boredom_trend:
        Signed OLS slope of each emotion channel, in ``[-1, 1]``.
        ``+1`` = perfect monotone rise over the window, ``-1`` = fall.
    dataset_stats:
        Optional per-concept behavioural lookup, e.g.
        ``(("mean_correctness", 0.62), ("mean_hint_count", 0.4))``.
        Empty tuple when no dataset is wired in.
    timestep:
        Zero-indexed env step in the current episode.
    """

    current_concept_id: str
    current_concept_index: int
    mastery: float
    perf: PerformanceFeatures
    rolling_emotions: EmotionVector
    engagement_trend: float
    confusion_trend: float
    frustration_trend: float
    boredom_trend: float
    dataset_stats: tuple[tuple[str, float], ...] = ()
    timestep: int = 0

    # ---- Validation ----------------------------------------------------

    def __post_init__(self) -> None:
        if not (0.0 <= self.mastery <= 1.0):
            raise ValueError(f"mastery={self.mastery} out of [0,1]")
        if self.current_concept_index < 0:
            raise ValueError(
                f"current_concept_index must be >= 0, got {self.current_concept_index}"
            )
        if self.timestep < 0:
            raise ValueError(f"timestep must be >= 0, got {self.timestep}")
        for name, v in (
            ("engagement_trend", self.engagement_trend),
            ("confusion_trend", self.confusion_trend),
            ("frustration_trend", self.frustration_trend),
            ("boredom_trend", self.boredom_trend),
        ):
            if not (-1.0 <= v <= 1.0):
                raise ValueError(f"{name}={v} out of [-1,1]")

    # ---- Human-readable views -----------------------------------------

    def dataset_stats_dict(self) -> Mapping[str, float]:
        """Convenience accessor for the optional dataset stats lookup."""
        return dict(self.dataset_stats)

    def to_dict(self) -> dict[str, Any]:
        """A flat, human-readable dict for the explainer / dashboard.

        Used by the rule-based explainer (Phase 10) and the read-only
        dashboard hooks (Phase 13) to surface what the agent saw. All
        values are pure-Python types (no tensors), so the explainer
        does not need torch.
        """
        return {
            "current_concept_id": self.current_concept_id,
            "current_concept_index": self.current_concept_index,
            "mastery": self.mastery,
            "recent_accuracy": self.perf.recent_accuracy,
            "hint_usage": self.perf.hint_usage,
            "attempts": self.perf.attempts,
            "rolling_emotions": {
                "engaged": self.rolling_emotions.engaged,
                "confused": self.rolling_emotions.confused,
                "bored": self.rolling_emotions.bored,
                "frustrated": self.rolling_emotions.frustrated,
            },
            "trends": {
                "engagement": self.engagement_trend,
                "confusion": self.confusion_trend,
                "frustration": self.frustration_trend,
                "boredom": self.boredom_trend,
            },
            "dataset_stats": self.dataset_stats_dict(),
            "timestep": self.timestep,
        }

    # ---- Tensor conversion --------------------------------------------

    #: Fixed feature order. Documenting this here means anyone reading
    #: a model checkpoint can map column indices back to features.
    TENSOR_FEATURE_NAMES: ClassVar[tuple[str, ...]] = (
        "mastery",
        "recent_accuracy",
        "hint_usage",
        "attempts_norm",
        "engaged_mean",
        "confused_mean",
        "bored_mean",
        "frustrated_mean",
        "engagement_trend_pos",
        "confusion_trend_pos",
        "frustration_trend_pos",
        "boredom_trend_pos",
        "concept_index_norm",
    )

    #: Used to normalise ``attempts`` into ``[0, 1]``. ``attempts >= 5``
    #: saturates - in practice 5 attempts on the same item is already
    #: a strong "stuck" signal, so we lose nothing by clipping.
    MAX_ATTEMPTS_FOR_NORM: ClassVar[int] = 5

    def to_tensor(
        self,
        *,
        num_concepts: int = 1,
        dtype: str = "float32",
    ) -> torch.Tensor:
        """Flatten the state to a small, normalised feature vector.

        Returns
        -------
        ``torch.Tensor`` of shape ``(len(TENSOR_FEATURE_NAMES),)`` (=
        13) with values in ``[0, 1]``.

        Args
        ----
        num_concepts:
            Total curriculum size, used to normalise
            ``current_concept_index`` to ``[0, 1]``. Must be >= 1.
        dtype:
            Either ``"float32"`` (default, RL standard) or ``"float64"``.

        Feature schema
        --------------
        See :attr:`TENSOR_FEATURE_NAMES`. All trend features are shifted
        from ``[-1, 1]`` to ``[0, 1]`` via ``0.5 * (t + 1)`` so the
        whole tensor stays in ``[0, 1]`` - the easiest range for a
        neural network to handle without a learnable normalisation
        layer.

        Why lazy torch import
        ---------------------
        The dataclass file is intentionally torch-free. Importing torch
        only inside this method preserves that property for offline
        scripts and unit tests that never call ``to_tensor``.
        """
        import torch  # noqa: PLC0415 - lazy by design

        if num_concepts < 1:
            raise ValueError(f"num_concepts must be >= 1, got {num_concepts}")
        denom = max(1, num_concepts - 1)
        concept_index_norm = self.current_concept_index / denom
        # Saturating normalisation; matches MAX_ATTEMPTS_FOR_NORM doc.
        attempts_norm = min(self.attempts_for_tensor, self.MAX_ATTEMPTS_FOR_NORM) / float(
            self.MAX_ATTEMPTS_FOR_NORM
        )
        values: list[float] = [
            self.mastery,
            self.perf.recent_accuracy,
            self.perf.hint_usage,
            attempts_norm,
            self.rolling_emotions.engaged,
            self.rolling_emotions.confused,
            self.rolling_emotions.bored,
            self.rolling_emotions.frustrated,
            0.5 * (self.engagement_trend + 1.0),
            0.5 * (self.confusion_trend + 1.0),
            0.5 * (self.frustration_trend + 1.0),
            0.5 * (self.boredom_trend + 1.0),
            min(1.0, max(0.0, concept_index_norm)),
        ]
        torch_dtype = torch.float64 if dtype == "float64" else torch.float32
        return torch.tensor(values, dtype=torch_dtype)

    # Helper kept as a property for to_dict + to_tensor reuse.
    @property
    def attempts_for_tensor(self) -> int:
        return self.perf.attempts


# ---------------------------------------------------------------------------
# Transition - one env step, stored by replay / rollout buffers.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Transition:
    """One env step seen by a controller.

    Same semantics as before, but now references the new
    :class:`LearnerState` shape.

    Attributes
    ----------
    s, a, s_next:
        Standard ``(s, a, s')`` tuple.
    r:
        Composite scalar reward (sum of weighted components).
    r_components:
        Per-component breakdown, e.g.
        ``(("performance", 0.4), ("emotion", -0.1))``. Frozen as a
        tuple of pairs so :class:`Transition` is hashable.
    done:
        Episode termination flag at ``s_next``.
    info:
        Free-form extras as a tuple of ``(key, value)`` pairs.
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
