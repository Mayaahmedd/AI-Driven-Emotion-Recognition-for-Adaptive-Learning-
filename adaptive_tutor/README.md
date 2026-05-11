# adaptive_tutor

Bachelor-thesis scope: interpretable RL tutoring with multilabel FER, a **teacher/ScienceQA curriculum layer**, and an **ASSISTments behaviour/statistics layer** joined explicitly in the state builder and simulator (see [ADR-006](docs/adrs/006-scope-reset-bachelor-thesis.md)).

This package wires the existing `FER_Module/` multilabel stack to tutoring actions (macro + meso), curriculum memory, learner-state features, and a lightweight ASSISTments-calibrated simulator. Roadmap: [`docs/adrs/`](docs/adrs/).

## Current status

| Phase | Description | Status |
|---|---|---|
| 0–2 | Skeleton, core, FER, curriculum providers | Shipped / in progress |
| 3–4 | State builder, synthetic environment | Shipped |
| 5+ | Replay, rewards, DQN/PPO, evaluation | Planned |

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
├── adaptive_tutor/      # the Python package
│   ├── core/            # frozen types + protocols + registry
│   ├── fer/             # adapter to the existing FER_Module
│   ├── memory/          # teacher + dataset curriculum providers
│   ├── state/           # learner state + rolling features
│   ├── simulator/       # synthetic student + gym-style env
│   ├── logging/         # unified TB + W&B + JSONL logger
│   └── utils/           # seeding, hashing, etc.
├── configs/             # Hydra-resolved YAMLs
├── docs/adrs/           # architecture decision records
├── scripts/             # CLI entry points
├── tests/               # pytest suite
└── pyproject.toml
```
