# ADR-006 - Scope Reset to Bachelor-Thesis Level

**Status:** Accepted (supersedes parts of ADR-001 and ADR-004; trims ADR-003)
**Owner:** Lead Architect
**Date:** 2026-05-11

---

## 1. Context

The first planning round produced an ambitious research-grade design with
dual critics, a contextual Thompson-sampling bandit, federated-ready
seams, full offline-policy-evaluation (FQE / WIS / DR), and a
SHAP-backed schema-versioned explainer. After review, the scope was
reset to a **bachelor-thesis-implementable system** that remains
academically credible but is realistic for one student to build and
defend.

This ADR records the cuts, the kept components, and the rationale.

## 2. Decision

### 2.1 Cuts (formally removed from v1)

- **Dual critics** `Q_perf` / `Q_flow` with adaptive weighting are
  removed. We use a single critic on a composite reward whose
  components are still tracked and logged for explainability.
- **Contextual Thompson-sampling bandit** is removed. All local
  adaptation goes through the DQN.
- **Federation seams, DP noise, secure-aggregation stubs** are removed.
  Federation is a future-work bullet in the thesis, not a deliverable.
- **FQE / WIS / DR offline-policy-evaluation harness** is removed.
  Replaced with simulator-based evaluation plus simple replay statistics
  on the ASSISTments logs.
- **KernelSHAP + counterfactual generation + schema migrations** are
  removed. Replaced with a rule-based top-driver explainer that emits a
  single frozen JSON shape.

### 2.2 Kept (unchanged in spirit)

- **PPO curriculum manager** chooses `(strategy, next_concept)` at
  concept boundaries. Strategy set:
  `teach_new_concept, review_prerequisite, give_worked_example,
   guided_practice, assessment`.
- **DQN local adaptation** chooses meso actions during interaction:
  `easier_problem, harder_problem, give_hint, slow_pacing,
   encouragement, retry_current_skill`.
- **FER adapter** + rolling-window cache (Phase 1, already shipped).
- **Curriculum memory** behind an abstract provider interface.
- **PER sum-tree** remains available; uniform replay is the acceptable
  fallback for thesis simplicity.
- **PPO concept-boundary transitions** stay - rollouts close at concept
  boundaries - but the heavier semi-Markov bootstrap math from ADR-003
  is optional. A plain n-step return with `gamma^segment_length`
  bootstrapping is acceptable.

### 2.3 New explicit constraint - dataset role separation

Two datasets play two non-overlapping roles. This is a hard project
invariant.

| Dataset      | Role               | Consumed by                                      |
|--------------|--------------------|--------------------------------------------------|
| ASSISTments  | Student *behaviour*| `dataset_provider.py` -> transitions + per-skill stats |
| ScienceQA    | *Knowledge*        | `teacher_provider.py` after offline extraction  |

ASSISTments is **never** used as curriculum memory.
ScienceQA is **never** used as an RL training set.

There is **no** ``HybridCurriculumProvider``: curriculum structure is
loaded only from ``TeacherCurriculumProvider`` (including ScienceQA-derived
YAML), while ASSISTments statistics feed the behaviour layer
(``DatasetCurriculumProvider``). RL state construction and the simulator
join the two **at call sites** (e.g. slug-aligned ``SkillStats`` lookups)
rather than merging providers.

### 2.4 ScienceQA -> curriculum extraction constraints (future script)

When (and only when) we add a ScienceQA preprocessing script in a later
phase, it MUST obey:

- `concept_id` is snake_case.
- Prerequisites are inferred **only** from dataset fields:
  `skill, topic, category, subject, grade, lecture, solution`. No
  external knowledge graphs, no LLM world-knowledge calls.
- 1 to 5 prerequisites per node.
- Prerequisite ids must already exist as concept ids in the same graph.
- The graph must be a DAG.
- Difficulty defaults derive from `grade` plus the number of
  prerequisites plus a coarse reasoning-complexity estimate from the
  solution length, when not explicitly provided.

These constraints live here so the future script writer cannot
plausibly miss them.

### 2.5 Trimmed architecture (final module list)

```
adaptive_tutor/
  core/             - frozen types, protocols, registry          (Phase 1, done)
  utils/, logging/  - seeding, unified logger                    (Phase 0, done)
  fer/              - FER adapter and rolling cache              (Phase 1, done)
  memory/providers/ - curriculum memory + providers              (Phase 2, this turn)
  state/            - learner state assembly
  simulator/        - ASSISTments-grounded synthetic learner
  replay/           - uniform replay + optional PER
  rewards/          - decomposed reward + attribution
  constraints/      - action mask + prerequisite enforcement
  rl/dqn/           - Double DQN, single critic, factorised meso
  rl/ppo/           - PPO, single critic, concept-level rollouts
  explainability/   - rule-based top-driver JSON
  evaluation/       - simulator metrics + offline replay stats
  orchestration/    - episode loop, trainer coordinator, runner
  dashboard/        - read-only FastAPI hooks
```

11 modules. Federation, bandit, dual critics, FQE/WIS, schema
migrations are all explicitly **out of v1**.

## 3. Rationale

A bachelor thesis needs to be defendable in a 30 minute presentation
with a small examiner panel. Every additional layer of theoretical
machinery is something that has to be explained, justified, and shown
to actually contribute to results. Dual critics and a separate bandit
are *not* the headline of the thesis - the headline is

> "Emotion-aware adaptive tutoring built from a hierarchical PPO + DQN
> stack over an interpretable curriculum memory."

That headline survives every cut above.

## 4. Consequences

- Total Python LOC for v1 drops by roughly 40 to 50 percent.
- Total number of hyperparameters to tune drops by roughly half.
- ADR-001 (dual critic stability) is **superseded** for v1; the
  document remains in the repository as historical context and as a
  future-work pointer.
- ADR-004 (FQE/WIS/DR) is **deferred** to future work.
- ADR-003 (PPO concept-level credit) is **trimmed**: the simpler
  n-step-with-segment-discount variant is the v1 default; the full
  semi-Markov variant becomes an optional flag.
- The thesis chapter "Limitations and Future Work" gets a clean,
  well-motivated three-item list (dual critics, bandit-driven micro
  interventions, federated learning).

## 5. References

- ADR-001 - Dual-Critic Gradient Detachment (superseded)
- ADR-002 - PER Sum-Tree (kept)
- ADR-003 - PPO Concept-Level Credit (trimmed)
- ADR-004 - FQE / WIS / DR (deferred)
- ADR-005 - Explainer Schema Versioning (simplified - single frozen v1)
