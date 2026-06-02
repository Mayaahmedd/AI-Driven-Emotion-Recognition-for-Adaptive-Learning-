"""Step 1 integration checks mirroring the spec inline script."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from RL_Module import config
from RL_Module.environment.student_env import DQNSafeEnv, StudentEnv
from RL_Module.explainability.explainer import Explainer, reason_for
from RL_Module.fer_interface.fer_adapter import FERAdapter
from RL_Module.logging_utils.csv_logger import CSVLogger, EPISODE_COLUMNS, STEP_COLUMNS
from RL_Module.mdp_definition import EMERGENCY_ALLOWED, StudentState
from RL_Module.reward.reward_function import compute_reward


def test_step1_full_integration():
    config.set_all_seeds(42)

    env = StudentEnv(render_mode=None)
    obs, info = env.reset(seed=42)
    assert obs.shape == (6,)
    assert obs.dtype == np.float32
    assert 0 <= obs[5] <= 3

    mask = env.get_action_mask()
    assert mask.dtype == np.int8
    assert mask.shape == (10,)

    action = int(np.where(mask == 1)[0][0])
    obs2, reward, terminated, truncated, info2 = env.step(action)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "persistent_flag" in info2
    assert "last_answer_correct" in info2
    assert "emotion_name" in info2
    assert "explainer_reason" in info2
    assert info2["last_answer_correct"] == (not info2["last_answer_wrong"])

    env2 = StudentEnv(render_mode=None)
    env2.reset(seed=1)
    env2.frustration = 0.8
    env2.emotion_id = 2
    env2.persistent_flag = True
    mask_em = env2.get_action_mask()
    allowed = set(np.where(mask_em == 1)[0].tolist())
    assert allowed == set(EMERGENCY_ALLOWED)

    env3 = StudentEnv(render_mode=None)
    env3.reset(seed=2)
    env3.step(2)
    assert env3.get_action_mask()[2] == 0
    env3.reset(seed=3)
    assert env3.get_action_mask()[2] == 1

    safe = DQNSafeEnv(StudentEnv(render_mode=None))
    obs5, _ = safe.reset(seed=42)
    assert obs5.shape == (6,)
    result2 = safe.step(1)
    assert len(result2) == 5

    fer = FERAdapter(use_real_fer=False)
    obs_vec = np.array([0.3, 0.6, 0.1, 0.1, 0.1, 3.0], dtype=np.float32)
    eid, conf = fer.get_emotion(obs_vec)
    assert 0 <= eid <= 3
    assert 0.0 < conf <= 1.0
    state = StudentState(0.3, 0.6, 0.1, 0.1, 0.1, 3)
    eid2, conf2 = fer.get_emotion(state)
    assert eid2 == eid

    for _ in range(5):
        s = np.random.rand(6).astype(np.float32)
        s[5] = float(np.random.randint(0, 4))
        s2 = np.random.rand(6).astype(np.float32)
        s2[5] = float(np.random.randint(0, 4))
        r = compute_reward(s, 0, s2, persistent_flag=False, last_answer_correct=True)
        assert -1.0 <= r <= 1.0
        r2 = compute_reward(
            StudentState.from_vec(s),
            0,
            StudentState.from_vec(s2),
            last_answer_wrong=True,
        )
        assert -1.0 <= r2 <= 1.0

    with tempfile.TemporaryDirectory() as tmp:
        logger = CSVLogger(log_dir=Path(tmp), algorithm="test", seed=0)
        logger.log_episode(
            episode=1,
            total_reward=10.0,
            k_start=0.3,
            k_final=0.5,
            success=True,
            dropout=False,
            steps=20,
            adaptation_accuracy=0.8,
            convergence_flag=False,
            action_counts=[1] * 10,
        )
        logger.close()
        ep_path = Path(tmp) / "episodes_test_seed0.csv"
        assert ep_path.exists()

    exp = Explainer()
    rec = exp.explain(0, 1, obs_vec, action=2, persistent_flag=False)
    assert "reason" in rec
    assert "emotion" in rec
    exp.close()

    assert isinstance(reason_for(obs_vec, 2, False), str)
    assert set(STEP_COLUMNS)
    assert set(EPISODE_COLUMNS)

    env.close()
    env2.close()
    env3.close()
    safe.close()
