# ADR-005 — Explainer JSON Schema Versioning

**Status:** Accepted
**Owners:** Explainability Engineer, API Architect
**Date:** 2026-05-11

---

## 1. Context

Every action chosen by the system is paired with an explanation JSON object (drivers, Q-values, weights, counterfactual, rationale). This object is consumed by:

- the **teacher dashboard** (read-only display),
- the **research report generator** (markdown export),
- **offline replay evaluation** (replayability of historical decisions),
- **external auditors** (compliance/IRB reviews of an educational deployment).

Once an explanation has been written to disk it becomes a **historical artifact** of a particular trained policy at a particular timestamp. We MUST be able to read explanations produced months ago without breaking, even after the policy network, the feature schema, the reward components, or the action space have changed.

That is a schema-versioning problem with the additional twist that the *meaning* of fields (not just their names) can shift between versions.

## 2. Decision

We adopt a strict three-part discipline: **monotonic version, frozen schema files, and forward/backward migrations**.

### 2.1 Every explanation carries a version triple

```json
{
  "schema": {
    "name": "adaptive_tutor.explanation",
    "version": "1.0.0",
    "policy_artifact_sha": "8a4f...e2",
    "feature_schema_sha": "c19a...90"
  },
  "explanation": { ... }
}
```

- `version` follows **semantic versioning**:
  - **MAJOR** bump ? consumers MUST update; old consumers cannot read new explanations.
  - **MINOR** bump ? backward-compatible additive change (new optional fields).
  - **PATCH** bump ? bug-fix only (no field changes).
- `policy_artifact_sha` is the SHA-256 of the policy checkpoint that produced the explanation. This lets us replay an old decision with the *exact same* policy parameters.
- `feature_schema_sha` is the SHA-256 of `state/features.py` at the time the explanation was produced. SHAP attributions are over a specific feature schema; losing that link makes them uninterpretable.

### 2.2 Schema is a frozen JSON-Schema file

Every version is a file:

```
adaptive_tutor/adaptive_tutor/explainability/schema/
    v1.0.0.json        # frozen on Phase 12 ship
    v1.1.0.json        # adds optional `policy_attention` field
    v2.0.0.json        # breaks `drivers` shape; requires migration
```

CI test: every released schema must validate at least one example explanation in `tests/data/explanations/`. Schemas, once a version is tagged, **must not be edited in-place**.

### 2.3 Migrations are explicit and tested

For every MAJOR jump, a migration function lives in:

```
adaptive_tutor/adaptive_tutor/explainability/migrations/v1_to_v2.py
```

Two functions:

- `upgrade(old: dict) -> dict` (forward migration).
- `downgrade(new: dict) -> dict | None` (optional; returns `None` if the downgrade is lossy).

The migration runner walks a sequence of migrations: a `v1.0.0` explanation read into a system at `v2.3.1` is upgraded `v1.x ? v2.0 ? v2.3`.

### 2.4 Required fields (v1.0.0)

```yaml
schema:
  name: "adaptive_tutor.explanation"
  version: "1.0.0"
  policy_artifact_sha: <hex64>
  feature_schema_sha:  <hex64>

experiment_id: <str>
step:          <int>                 # env step index
timestamp:     <ISO-8601 UTC>

action:
  macro:   {instruction_type: <int>, concept_id: <int>} | null
  meso:    {content_variant: <int>, pace: <int>, ui_variant: <int>, emotional_support: <int>}
  micro:   {intervention_id: <int>} | null
  mask_rejected:
    - {head: <str>, action_id: <int>, reason: <str>}   # what was masked off

values:
  Q_perf:      <float>
  Q_flow:      <float>
  Q_composite: <float>
  weight_flow: <float in [0,1]>
  weight_perf: <float in [0,1]>

drivers:                              # SHAP attributions (top-K)
  - {feature: <str>, value: <float>, attribution: <float>}

counterfactual:                       # optional
  feature:    <str>
  delta:      <float>
  alt_action: <action object>
  alt_Q_composite: <float>

rationale:
  template_id: <str>                  # e.g., "low_mastery_high_confusion"
  text:        <str>                  # filled template

provenance:
  fer_window:  {len: <int>, mean: <EmotionVector>, slope: <EmotionVector>}
  mastery_at_concept: <float>
  cooldowns_active: {action_id: steps_remaining, ...}
```

### 2.5 Privacy-sensitive fields

Two explicit categories:

- **`provenance.raw_fer_frames`** — raw FER inputs MAY be included only when the consumer is the dashboard *running on the same device*. Server-side persistence of raw FER frames is **forbidden** by default. The field is therefore absent in v1.0.0 and only added (as opt-in) in a later MINOR version.
- **`learner_id`** — explanations are pseudonymous; the real learner ID is stored separately and joined only by authorized consumers.

### 2.6 Storage format

- One JSON object per line (JSONL) in `runs/<experiment_id>/explanations/<episode_id>.jsonl`.
- Compression: `gzip` for files older than 7 days (configurable).
- Index file: `runs/<experiment_id>/explanations/_index.parquet` with `(episode_id, step, schema_version, policy_sha, ts)` for fast lookup.

### 2.7 SDK contract

The Python API:

```python
from adaptive_tutor.explainability import (
    Explanation, ExplanationReader, ExplanationWriter,
    schema_version, latest_schema_version, migrate,
)
```

Readers automatically migrate to the latest schema by default; the call site can opt out with `migrate=False` to inspect raw legacy data.

## 3. Rationale

- **Semver + frozen schema files** is the cheapest discipline that prevents historical artifacts from rotting. Engineers must consciously cut a new schema version rather than silently mutating fields.
- **`policy_artifact_sha`** is the single most important addition. Without it, two explanations with the same shape but produced by different policy checkpoints look identical, which makes ablation studies impossible.
- **Explicit migrations** with `upgrade/downgrade` make schema evolution testable. Without them, every schema change becomes a flag day.
- **JSONL + parquet index** balances human-readability (jsonl) with fast retrieval (parquet index) for the dashboard.
- **Privacy by default** for FER frames is non-negotiable for an educational deployment.

## 4. Consequences

- Bumping the schema requires: (a) a new JSON-Schema file, (b) a migration if MAJOR, (c) updated example JSONs, (d) a CI run that proves migration round-trips don't lose information.
- The dashboard, reporter, and offline-eval consumers all pin to the latest schema version they understand and refuse to load newer MAJOR versions until updated. This is enforced via a small `assert_compatible` helper.
- Schema files are part of the package and shipped with it; an explanation produced by a pip-installed `adaptive_tutor==1.0.3` can be parsed by `adaptive_tutor==1.4.0` without external schema fetches.

## 5. Alternatives considered

- **Protobuf with FileDescriptors.** Stricter and faster but adds toolchain weight and is harder for a teacher dashboard to inspect by hand. Rejected for v1.
- **JSON without explicit version.** Rejected — guarantees silent breakage.
- **A database table per schema.** Rejected — couples evaluation to a DB at a time when the rest of the system is file-based.

## 6. Test plan (referenced from `tests/test_explanation_schema.py` when Phase 12 lands)

- Round-trip a v1.0.0 explanation through every defined migration; assert no information loss.
- Validate a malformed explanation raises `SchemaValidationError`.
- Reader at version `v2.x` refuses to load a `v3.x` artifact and raises `FutureSchemaError`.
- `policy_artifact_sha` correctness: writer computes the hash from the in-memory checkpoint and matches the on-disk checkpoint's hash.

## 7. References

- Tom Preston-Werner, *Semantic Versioning 2.0.0*.
- JSON Schema Draft 2020-12.
- Datadog / Sentry blog posts on long-lived event schema evolution.
