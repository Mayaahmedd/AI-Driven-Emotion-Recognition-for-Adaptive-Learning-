# Step 3 (DQN Version)

This folder is a self-contained Step 3 implementation with:

- emotion-only input (`engaged`, `confused`, `frustrated`, `bored`)
- two adaptation phases:
  - explanation-time tutoring action selection
  - practice-time next-question selection
- DQN agents (instead of tabular Q-learning)
- simulation archetypes + baseline comparisons

## Files

- `config.py`
- `simulator.py`
- `baselines.py`
- `dqn_agents.py`
- `trainer.py`
- `evaluator.py`
- `run_demo.py`


## `How the files connect (execution flow)`
1- run_demo.py reads settings from config.py.
2- It creates StudentSimulator from simulator.py.
3- It creates DQN agents from dqn_agents.py.
4- trainer.py trains agents on simulator episodes.
5- evaluator.py tests trained agents and baselines from baselines.py.
6- Results are printed and saved to results/comparison.csv.


## Run

From repository root:

```bash
python3 -m adaptation.step3.run_demo --compare --csv-out adaptation/step3/results/comparison.csv
```
