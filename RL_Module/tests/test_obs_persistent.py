"""Tests for v7 observation space with persistent-frustration channel."""

import numpy as np
import pytest

from RL_Module import config
from RL_Module.environment.student_env import OBS_SPACE_VARIANTS, StudentEnv
from RL_Module.mdp_definition import (
    OBS_DIM_V6,
    OBS_DIM_V7,
    OBS_IDX_PERSISTENT_STEPS,
    StudentState,
    build_observation,
    normalize_persistent_steps,
)


def test_normalize_persistent_steps():
    assert normalize_persistent_steps(0) == 0.0
    assert normalize_persistent_steps(1, dropout_steps=5) == 0.2
    assert normalize_persistent_steps(5, dropout_steps=5) == 1.0
    assert normalize_persistent_steps(10, dropout_steps=5) == 1.0


def test_build_observation_v6_v7():
    state = StudentState(0.5, 0.6, 0.7, 0.2, 0.1, 2)
    v6 = build_observation(state, include_persistent=False)
    assert v6.shape == (OBS_DIM_V6,)
    v7 = build_observation(state, consecutive_flag_steps=3, include_persistent=True)
    assert v7.shape == (OBS_DIM_V7,)
    assert v7[OBS_IDX_PERSISTENT_STEPS] == pytest.approx(0.6)


def test_from_vec_accepts_v7():
    state = StudentState(0.3, 0.4, 0.5, 0.2, 0.1, 1)
    v7 = build_observation(state, consecutive_flag_steps=4, include_persistent=True)
    parsed = StudentState.from_vec(v7)
    assert parsed.knowledge == pytest.approx(0.3)
    assert parsed.emotion_id == 1


@pytest.mark.parametrize("variant,expected_dim", [
    ("A_original_v6", OBS_DIM_V6),
    ("B_v7_with_persistent", OBS_DIM_V7),
    ("C_v7_persistent_no_eid", OBS_DIM_V7),
])
def test_obs_space_variants(variant, expected_dim):
    env = StudentEnv(population_seed=42, obs_space_variant=variant, max_episode_steps=10)
    obs, info = env.reset(seed=42)
    assert obs.shape == (expected_dim,)
    assert env.observation_space.shape == (expected_dim,)
    mask = info["action_masks"]
    action = int(np.where(mask == 1)[0][0])
    obs2, _, _, _, info2 = env.step(action)
    assert obs2.shape == (expected_dim,)
    assert "consecutive_flag_steps" in info2
    if variant == "C_v7_persistent_no_eid":
        assert obs2[5] == 0.0  # emotion_id masked
    env.close()


def test_persistent_steps_in_obs_when_flag_active():
    env = StudentEnv(
        population_seed=0,
        include_persistent_obs=True,
        max_episode_steps=50,
    )
    env.reset(seed=0)
    env.frustration = 0.85
    env.persistent_flag = True
    env._consecutive_flag_steps = 3
    obs = env._obs()
    assert obs.shape == (OBS_DIM_V7,)
    assert obs[OBS_IDX_PERSISTENT_STEPS] == pytest.approx(
        normalize_persistent_steps(3, config.PERSISTENT_FLAG_DROPOUT_STEPS)
    )
    env.close()


def test_backward_compat_default_v6():
    env = StudentEnv(population_seed=42)
    obs, _ = env.reset(seed=42)
    assert obs.shape == (OBS_DIM_V6,)
    env.close()
