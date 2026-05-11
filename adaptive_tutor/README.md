# adaptive_tutor

Hybrid Adaptive Tutoring System with Multilabel FER-Based Emotional Modeling.

This package implements the RL + bandit + dual-critic adaptation engine that consumes the existing `FER_Module/` multilabel model and produces tutoring actions. The architecture and implementation roadmap live in [`docs/adrs/`](docs/adrs/) and the master planning document is in the parent thesis.

## Current status

| Phase | Description | Status |
|---|---|---|
| 0 | Project skeleton, configs, seeding, logger | **In progress** (this commit) |
| 1 | Core types, protocols, registry, FER adapter | **In progress** (this commit) |
| 2+ | Curriculum / memory / replay / agents / etc. | Planned |

See [`docs/adrs/000-index.md`](docs/adrs/000-index.md) for the architectural decision records.

## Install (development)

```bash
# from repository root
cd adaptive_tutor
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # core + dev tooling
# optional groups, install as needed:
pip install -e ".[dev,wandb,graph,sim,explain,api]"
```

## Run tests

```bash
pytest -q
```

## Repository layout (current)

```
adaptive_tutor/
??? adaptive_tutor/      # the Python package
?   ??? core/            # frozen types + protocols + registry
?   ??? fer/             # adapter to the existing FER_Module
?   ??? logging/         # unified TB + W&B + JSONL logger
?   ??? utils/           # seeding, hashing, etc.
??? configs/             # Hydra-resolved YAMLs
??? docs/adrs/           # architecture decision records
??? scripts/             # CLI entry points
??? tests/               # pytest suite
??? pyproject.toml
```
