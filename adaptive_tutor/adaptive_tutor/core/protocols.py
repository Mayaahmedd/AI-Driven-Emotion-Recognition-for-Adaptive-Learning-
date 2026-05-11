"""Structural type contracts (PEP-544 :class:`Protocol`).

Why ``Protocol`` and not ``abc.ABC``
-----------------------------------
Structural subtyping ("duck typing with type checks") lets us:

1. Treat *any* object exposing the right methods as a valid implementer
   without forcing it to inherit a particular base class. This keeps
   third-party objects (e.g., a research baseline imported from another
   package) usable without wrappers.
2. Compose providers cheaply. The ``HybridCurriculumProvider`` (Phase 2)
   does not subclass ``CurriculumProvider``; it just exposes the right
   methods. mypy still verifies the contract.
3. Mock cleanly in tests. ``unittest.mock.MagicMock(spec=Policy)``
   produces a checked mock without any ABC dance.

What you DO NOT get from Protocols
----------------------------------
* No automatic default method implementations. Add a small concrete base
  class alongside the Protocol if you need shared default behavior
  (we do this only when truly needed — most sub-systems have nothing
  meaningful to share).
* No ``isinstance(obj, Policy)`` magic unless we decorate the protocol
  with ``@runtime_checkable``. We do decorate them so that callers can
  runtime-check, *but only on method names* — Protocols cannot verify
  method signatures at runtime.

Stability
---------
Every signature in this file is part of the *public contract*. Adding a
new method is a MINOR change (existing implementers break), so we use
extension via separate protocols (see ``RewardComponent`` vs
``ConditionalRewardComponent`` in later phases) rather than inflating
existing ones.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from adaptive_tutor.core.types import CompositeAction, LearnerState, Transition

# ---------------------------------------------------------------------------
# Policy & critic
# ---------------------------------------------------------------------------


@runtime_checkable
class Policy(Protocol):
    """Anything that can produce a :class:`CompositeAction` from a state.

    The exact action shape depends on the controller (PPO sets ``macro``,
    DQN sets ``meso``, bandit sets ``micro``). A "controller" object may
    set only a *subset* of fields and leave others ``None``; the
    orchestrator composes the final action across controllers.

    The optional ``mask`` argument is the per-head action mask from the
    safety layer (see Phase 7). Implementations MUST respect it; any
    chosen action that has a zero-mask entry is a critical bug and the
    safety layer will raise :class:`MaskError` downstream.

    ``update`` returns a dict of metrics that the trainer logs.
    ``save``/``load`` use file paths so implementations choose their own
    serialization format (torch ``.pt``, parquet, json, ...).
    """

    def act(self, s: LearnerState, mask: Mapping[str, Any] | None = None) -> CompositeAction: ...

    def update(self, batch: Any) -> Mapping[str, float]: ...

    def save(self, path: str) -> None: ...

    def load(self, path: str) -> None: ...


@runtime_checkable
class Critic(Protocol):
    """A value function ``V(s) -> R`` or ``Q(s, a) -> R``.

    We deliberately collapse V and Q under one protocol; the
    ``action`` argument may be ``None`` for value functions. The dual
    critics (Q_perf, Q_flow) each implement this protocol independently;
    the composite weighting lives in
    ``rl/critics/weighting.py`` (Phase 11).
    """

    def value(self, s: LearnerState, action: CompositeAction | None = None) -> float: ...


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


@runtime_checkable
class Replay(Protocol):
    """Off-policy replay buffer or on-policy rollout buffer.

    ``push`` stores a transition. ``sample`` returns a batch in whatever
    format the trainer expects (controller-specific; we type as ``Any``
    to avoid coupling). For PER-style buffers, implementations also
    expose ``update_priorities(idxs, priorities)`` (see ADR-002), but
    that is not part of the base protocol because not every replay
    needs it.
    """

    def push(self, t: Transition) -> None: ...

    def sample(self, batch_size: int) -> Any: ...

    def __len__(self) -> int: ...


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------


@runtime_checkable
class RewardComponent(Protocol):
    """A single, named reward signal (e.g., performance, emotion, cost).

    The reward engine sums weighted components from a registry. Every
    component carries a ``name`` so the engine can build the per-
    component attribution dict for explainability + ablations.
    """

    name: str

    def __call__(
        self,
        s: LearnerState,
        a: CompositeAction,
        s_next: LearnerState,
        info: Mapping[str, float],
    ) -> float: ...


# ---------------------------------------------------------------------------
# Curriculum
# ---------------------------------------------------------------------------


@runtime_checkable
class CurriculumProvider(Protocol):
    """Abstract source of concepts / prerequisites / learning material.

    See ADR section *"Curriculum / memory provider design"* in the master
    plan for the rationale. Implementations may be backed by a public
    dataset, teacher YAML, AI generation, textbook ingestion, or hybrid
    compositions thereof.

    Required semantics:

    * ``get_concepts()`` returns an iterable of dicts with at least
      ``{"id": int, "name": str, "difficulty": float}``. The exact set
      of optional fields is provider-specific and consumed by the
      explainer's rationale templates.
    * ``get_prerequisites()`` returns an iterable of ``(from_id, to_id)``
      directed edges. Cycles are an error and providers should validate
      this on load.
    * ``version()`` returns a stable, opaque identifier. Two providers
      with the same ``version()`` MUST return identical concept and
      prerequisite sets — this is used for caching and for the
      explainer's ``feature_schema_sha``.
    """

    def get_concepts(self) -> Sequence[Mapping[str, Any]]: ...

    def get_prerequisites(self) -> Sequence[tuple[int, int]]: ...

    def get_learning_material(self, concept_id: int) -> Mapping[str, Any]: ...

    def get_skill_metadata(self, concept_id: int) -> Mapping[str, Any]: ...

    def version(self) -> str: ...


# ---------------------------------------------------------------------------
# Action masking
# ---------------------------------------------------------------------------


@runtime_checkable
class ActionMaskBuilder(Protocol):
    """Builds the per-head action mask given the current state.

    Returns a dict like::

        {
          "macro": {"instruction_type": np.ndarray[bool],
                    "concept_id":       np.ndarray[bool]},
          "meso":  {"content_variant":  np.ndarray[bool],
                    "pace":             np.ndarray[bool],
                    ...},
          "micro": {"intervention_id":  np.ndarray[bool]},
        }

    ``True`` = allowed, ``False`` = blocked. The trainer multiplies the
    mask into the policy logits before the softmax/argmax.

    The data type (``np.ndarray`` of bool) is convention; Phase 7 may
    return torch tensors when integrated into the RL inner loop. The
    Protocol uses ``Mapping[str, Any]`` so both are allowed.
    """

    def build(self, s: LearnerState) -> Mapping[str, Any]: ...


# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------


@runtime_checkable
class Explainer(Protocol):
    """Produces an explanation JSON for one decision (see ADR-005).

    The return shape is the validated v1.0.0 schema; implementations are
    free to attach extra optional fields as long as they remain
    backward-compatible. Versioning is the explainer's responsibility,
    not the caller's.
    """

    def explain(
        self,
        s: LearnerState,
        a: CompositeAction,
        q_perf: float,
        q_flow: float,
        w_flow: float,
    ) -> Mapping[str, Any]: ...


# ---------------------------------------------------------------------------
# Iterable helpers — re-exported for convenience.
# ---------------------------------------------------------------------------

__all__ = [
    "ActionMaskBuilder",
    "Critic",
    "CurriculumProvider",
    "Explainer",
    "Iterable",
    "Policy",
    "Replay",
    "RewardComponent",
]
