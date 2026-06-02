# RL_Module � Adaptive Tutoring via Reinforcement Learning

MDP-based adaptive tutor parallel to `FER_Module/`. The FER module outputs one of four emotions; this module learns which teaching action to take.

## Setup

```bash
cd /path/to/AI-Driven-Emotion-Recognition-for-Adaptive-Learning-
python3 -m venv RL_Module/.venv
source RL_Module/.venv/bin/activate
pip install -r RL_Module/requirements.txt
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## Run full experiment

```bash
python -m RL_Module.main_experiment
python -m RL_Module.main_experiment --algo PPO --seed 42
python -m RL_Module.main_experiment --train-only --algo DQN --seed 42
python -m RL_Module.main_experiment --eval-only --algo PPO --seed 42
```

## Algorithm comparison + PPO ablation (recommended)

Compare **PPO vs DQN vs Rule vs Random**, then **PPO with emotions vs knowledge-only** (emotion dimension zeroed in observation):

```bash
export PYTHONPATH="$(pwd)"
python -m RL_Module.run_comparison_experiment --quick --seed 42
python -m RL_Module.run_comparison_experiment --seed 42 --timesteps 50000 --eval-episodes 500
python -m RL_Module.run_comparison_experiment --skip-ablation   # comparison only
python -m RL_Module.run_comparison_experiment --skip-comparison # ablation only
```

Outputs:
- `RL_Module/logs/summary.csv` (via per-algo episode logs)
- `RL_Module/logs/comparison_experiment.json`
- PPO action-usage table printed via `action_frequency_report()`
- Ablation figures (optional): `python -m RL_Module.evaluation.ablation_study`

## Run tests

```bash
pytest RL_Module/tests/ -q
```

## Layout

| Drawer | Purpose |
|--------|---------|
| `mdp_definition.py` | MDP constants � single source of truth |
| `config.py` | Seeds, hyperparams, paths |
| `environment/` | Gymnasium `StudentEnv`, synthetic students |
| `reward/` | Weighted reward with thesis citations |
| `agents/` | PPO, DQN, bandit+DQN, rule, random |
| `evaluation/` | Metrics, plots, ablation |
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

## Demo render

Set `RENDER_MODE = "human"` in `config.py` or pass `render_mode="human"` when creating `StudentEnv`.

## Live viewer (during training)

Run training in one terminal and the live viewer in another. The viewer reads the step CSV every 0.5 seconds.

```bash
# Terminal 1 � training
python -m RL_Module.main_experiment --algo PPO --seed 42

# Terminal 2 � live viewer
python -m RL_Module.evaluation.live_viewer --algo PPO --seed 42
```

## Sensitivity analysis

```bash
python -m RL_Module.evaluation.sensitivity_analysis
```

Outputs `RL_Module/logs/sensitivity_analysis.csv` and `RL_Module/logs/threshold_sensitivity.csv`.
