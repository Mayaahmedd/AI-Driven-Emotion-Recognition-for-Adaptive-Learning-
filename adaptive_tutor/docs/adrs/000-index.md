# Architecture Decision Records (ADRs)

This directory contains the canonical, versioned design notes for the `adaptive_tutor` system. Every non-trivial architectural decision MUST have an ADR. ADRs are immutable once accepted; new decisions supersede old ones with explicit "Supersedes" / "Superseded-by" links.

## Status legend

- `Proposed` — under discussion
- `Accepted` — currently in force
- `Superseded` — replaced by a later ADR

## Index

| # | Title | Status | Owner |
|---|---|---|---|
| 001 | Dual-Critic Gradient Detachment & Adaptive-Weighting Stability | Accepted | RL Architect + Research Auditor |
| 002 | Prioritized Experience Replay: Sum-Tree Implementation Choices | Accepted | Infrastructure Engineer |
| 003 | PPO Concept-Level Credit Assignment | Accepted | RL Architect |
| 004 | Offline Policy Evaluation: FQE / WIS Test Harness | Accepted | Evaluation Scientist |
| 005 | Explainer JSON Schema Versioning | Accepted | Explainability Engineer + API Architect |

## Conventions

- ADR file names: `NNN-kebab-case-title.md`.
- Each ADR has sections: `Context`, `Decision`, `Rationale`, `Consequences`, `Alternatives Considered`, `References`.
- Math is rendered with LaTeX (`$...$` for inline, `$$...$$` for block) and cross-referenced from code via docstrings of the form `See ADR-001 §3`.
