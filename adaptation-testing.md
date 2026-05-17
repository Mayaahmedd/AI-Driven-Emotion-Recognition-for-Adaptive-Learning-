cd /Users/mayoy/Documents/GitHub/AI-Driven-Emotion-Recognition-for-Adaptive-Learning-/adaptive_tutor
source .venv/bin/activate




python3 -u -c "
from adaptive_tutor.experiments import run_experiment
run_experiment({
    'seed': 0,
    'episodes': 20,
    'policy': 'ppo',
    'log_steps': True,
    'max_episode_steps': 40,
})
"












cd /Users/mayoy/Documents/GitHub/AI-Driven-Emotion-Recognition-for-Adaptive-Learning-/adaptive_tutor
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e '.[dev]'








python3 -u -c "
from adaptive_tutor.experiments import run_experiment

run_experiment({
    'seed': 0,
    'episodes': 20,
    'policy': 'dqn',
    'log_steps': True,
    'max_episode_steps': 40
})
"