"""Map ASSISTments :class:`SkillStats` to :class:`LearnerParams`.

Why this module exists
----------------------
The simulator must stay **grounded** in real logs without merging
curriculum YAML into the dataset adapter. :class:`LearnerParams` is a
small, hand-tunable struct that the thesis can cite as ``human-
interpretable priors from empirical cohort statistics``.

Design
------
Every field is a direct function of observable dataset means. No
neural fitting, no latent-variable inference - that keeps the
``adaptive_tutor`` stack honest in a viva.

Trade-off
---------
These formulas are crude; they are intentionally so. A more
sophisticated calibration would need held-out validation data. For a
bachelor thesis, transparency beats marginal realism.
"""

from __future__ import annotations

from dataclasses import dataclass

from adaptive_tutor.memory.providers.dataset_provider import SkillStats


@dataclass(frozen=True, slots=True)
class LearnerParams:
    """Scalar knobs for :class:`~adaptive_tutor.simulator.learner.SyntheticLearner`.

    ``learning_rate`` is not used for mastery arithmetic directly -
    mastery is updated only through :class:`~adaptive_tutor.state.builder.StateBuilder`
    running means - but it is kept for future extensions (e.g., latent
    ``knowledge`` separate from observed correctness).

    ``persistence`` dampens how harshly failed attempts shift emotions
    (higher ``persistence`` -> smaller frustration bump on ``correct=0``).
    """

    learning_rate: float
    persistence: float
    engaged_bias: float
    confused_bias: float
    bored_bias: float
    frustrated_bias: float


def calibrate_from_stats(st: SkillStats) -> LearnerParams:
    """Derive ``LearnerParams`` from one skill's aggregate ASSISTments stats.

    Signal usage (all from ``SkillStats`` only):

    * ``mean_correctness``: harder skills get slightly higher
      ``confused_bias`` and lower ``persistence``.
    * ``mean_engaged`` / ``mean_frustrated``: initialise FER-channel
      proxies before the first interaction.
    * ``mean_hint_count``: proxy for boredom / disengagement when
      problems drag.
    """
    mc = float(st.mean_correctness)
    mc = max(0.0, min(1.0, mc))
    hint_rate = float(st.mean_hint_count)
    return LearnerParams(
        learning_rate=0.04 + 0.08 * (1.0 - mc),
        persistence=0.4 + 0.6 * mc,
        engaged_bias=_clip_unit(st.mean_engaged),
        confused_bias=_clip_unit(0.1 + 0.25 * (1.0 - mc)),
        bored_bias=_clip_unit(0.05 + 0.12 * min(1.0, hint_rate / 3.0)),
        frustrated_bias=_clip_unit(st.mean_frustrated),
    )


def default_learner_params() -> LearnerParams:
    """Neutral cohort when no ASSISTments slice is available (demos / unit tests)."""
    return LearnerParams(
        learning_rate=0.08,
        persistence=0.65,
        engaged_bias=0.55,
        confused_bias=0.15,
        bored_bias=0.10,
        frustrated_bias=0.12,
    )


def _clip_unit(x: float) -> float:
    return max(0.0, min(1.0, float(x)))
