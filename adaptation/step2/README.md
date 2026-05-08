# Step 2 Snapshot

This folder stores Step 2 artifacts so you can keep a stable reference before Step 3 changes.

## What Step 2 includes

- Emotion-only adaptation input:
  - engaged, confused, frustrated, bored
- Two adaptation parts:
  - explanation-time tutoring action selection
  - practice-time next-question selection
- Simulation archetypes:
  - balanced, anxious, boredom_prone, struggling
- Baseline comparison:
  - RL Q-learning vs random/fixed/heuristic

## Step 2 code package

The full Step 2 implementation is now inside this folder:

- `__init__.py`
- `config.py`
- `simulator.py`
- `agents.py`
- `baselines.py`
- `trainer.py`
- `evaluator.py`
- `run_demo.py`

Run it from repo root:

```bash
python3 -m adaptation.step2.run_demo --compare --csv-out adaptation/step2/results/comparison.csv
```

## Saved artifact

- `comparison.csv`: most recent comparison output from Step 2.

## Note about archetypes

Archetypes are optional. If you do not want them, set both:

- `train_archetypes = ("balanced",)`
- `eval_archetypes = ("balanced",)`

in `adaptation/config.py`.
