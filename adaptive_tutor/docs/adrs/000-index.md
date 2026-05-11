# Architecture Decision Records (ADRs)

This directory contains the canonical, versioned design notes for the
`adaptive_tutor` system.

## Status legend

- `Accepted` - currently in force
- `Superseded` - replaced by a later ADR (kept for historical context)
- `Deferred`  - intentionally not implemented in v1; future-work pointer
- `Trimmed`   - kept in force but with explicit simplifications recorded
                in a later ADR

## Index

| #   | Title                                                       | Status                           | Notes                                      |
|-----|-------------------------------------------------------------|----------------------------------|--------------------------------------------|
| 001 | Dual-Critic Gradient Detachment and Adaptive Weighting      | **Superseded by ADR-006**        | Single critic in v1                        |
| 002 | Prioritized Experience Replay - Sum-Tree                    | Accepted                         | Optional in v1; uniform replay also OK     |
| 003 | PPO Concept-Level Credit Assignment                         | **Trimmed by ADR-006**           | Simple n-step default; full SMDP optional  |
| 004 | Offline Policy Evaluation - FQE / WIS / DR                  | **Deferred by ADR-006**          | Replaced by simulator eval in v1           |
| 005 | Explainer JSON Schema Versioning                            | **Simplified by ADR-006**        | Single frozen v1 shape; no migrations      |
| 006 | Scope Reset to Bachelor-Thesis Level                        | Accepted                         | Master scope cut record                    |

## Conventions

- File names: `NNN-kebab-case-title.md`.
- Sections: `Context`, `Decision`, `Rationale`, `Consequences`,
  `Alternatives Considered`, `References`.
- Cross-reference inside code via docstrings of the form `See ADR-NNN`.
