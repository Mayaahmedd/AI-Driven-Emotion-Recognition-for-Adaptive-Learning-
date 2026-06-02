#!/usr/bin/env bash
# Fair PPO vs DQN master study (environment fixed).
# Phase 1: PPO hyperparameter search (2304 configs, ~5h with 8 workers).
# Phase 2-9: fair comparison using tuned PPO + tuned DQN.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true

echo "=== Phase 1: PPO hyperparameter optimization ==="
python -m RL_Module.ppo_hyperparameter_study --phase grid --workers 8 --resume
python -m RL_Module.ppo_hyperparameter_study --phase robust --top-n 5
python -m RL_Module.ppo_hyperparameter_study --phase report

echo "=== Phases 2-9: Fair PPO vs DQN comparison ==="
python -m RL_Module.ppo_dqn_fair_comparison --phase all --resume

echo "Done. See RL_Module/figures/ppo_dqn_fair_comparison/FAIR_COMPARISON_REPORT.md"
