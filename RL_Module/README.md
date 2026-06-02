# RL_Module — Adaptive Tutoring via Reinforcement Learning

MDP-based adaptive tutor parallel to `FER_Module/`. The FER module outputs one of four emotions; this module learns which teaching action to take.

## Setup

```bash
cd /path/to/AI-Driven-Emotion-Recognition-for-Adaptive-Learning-
python3 -m venv RL_Module/.venv
source RL_Module/.venv/bin/activate
pip install -r RL_Module/requirements.txt
export PYTHONPATH="$(pwd)"
```

## Final thesis studies

```bash
export PYTHONPATH="$(pwd)"
python -m RL_Module.action_gain_ablation --phase all
python -m RL_Module.combined_improvement_validation --phase all
python -m RL_Module.final_thesis_comparison --phase all
python -m RL_Module.final_thesis_algorithm_comparison --phase all
python -m RL_Module.final_dqn_model_study --phase all
```

Outputs live under `RL_Module/figures/` (four study folders) and `RL_Module/final_thesis_comparison/`.

Superseded experiments, Double DQN work, and historical reports are under `RL_Module/archive/` (see `repository_cleanup_report.txt`).

## Run tests

```bash
pytest RL_Module/tests/ -q
```

## Layout

| Drawer | Purpose |
|--------|---------|
| `mdp_definition.py` | MDP constants — single source of truth |
| `config.py` | Seeds, hyperparams, paths |
| `environment/` | Gymnasium `StudentEnv`, synthetic students |
| `reward/` | Weighted reward with thesis citations |
| `agents/` | DQN, rule (ERT), random |
| `evaluation/` | Metrics, plots, live viewer |
| `fer_interface/` | FER adapter (synthetic or live) |
| `explainability/` | JSONL action explanations |
| `logging_utils/` | CSV step/episode/summary logs |

Note: `logging_utils/` avoids shadowing Python's stdlib `logging` module.

## Outputs

- `RL_Module/logs/steps_{ALGO}_seed{N}.csv`
- `RL_Module/logs/episodes_{ALGO}_seed{N}.csv`
- `RL_Module/logs/summary.csv`
- `RL_Module/logs/explanations_{ALGO}.jsonl`
- `RL_Module/models/{algorithm}_{seed}/`

## Live viewer (during training)

```bash
python -m RL_Module.evaluation.live_viewer --algo DQN --seed 42
```

Reads step CSV logs written during DQN training runs.
