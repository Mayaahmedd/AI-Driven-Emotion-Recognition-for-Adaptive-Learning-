"""Verify strict_no_emotion mode has no affect leakage."""

from __future__ import annotations

import numpy as np

from RL_Module.environment.student_env import StudentEnv
from RL_Module.mdp_definition import ACTION_TO_ID, EMOTION_TO_ID
from RL_Module.reward.reward_function import compute_reward
from RL_Module.mdp_definition import StudentState


def test_strict_mask_ignores_frustration():
    env = StudentEnv(
        max_episode_steps=10,
        population_seed=7,
        obs_ablation="knowledge_only",
        emotion_dynamics="strict_off",
    )
    env.reset(seed=7)
    env._state.frustration = 0.99
    env._state.emotion_id = EMOTION_TO_ID["frustrated"]
    mask = env.get_action_mask()
    assert mask[ACTION_TO_ID["harder_problem"]] == 1
    env.close()


def test_strict_no_dropout():
    env = StudentEnv(
        max_episode_steps=200,
        population_seed=7,
        emotion_dynamics="strict_off",
    )
    env.reset(seed=7)
    env._persistent_frustration_flag = True
    env._consecutive_flag_steps = 100
    done = False
    steps = 0
    while not done and steps < 20:
        mask = env.get_action_mask()
        action = int(np.where(mask == 1)[0][0])
        _, _, term, trunc, info = env.step(action)
        done = term or trunc
        steps += 1
        assert not info.get("dropout", False)
    env.close()


def test_strict_affect_pinned_after_steps():
    env = StudentEnv(
        max_episode_steps=30,
        population_seed=42,
        emotion_dynamics="strict_off",
    )
    env.reset(seed=42)
    for _ in range(15):
        env.step(ACTION_TO_ID["harder_problem"])
    assert env._state.frustration == 0.0
    assert env._state.confusion == 0.0
    assert env._state.boredom == 0.0
    assert env._state.engagement == 0.5
    env.close()


def test_strict_reward_independent_of_affect():
    prev = StudentState(0.4, 0.5, 0.0, 0.0, 0.0, 3)
    new = StudentState(0.5, 0.5, 0.0, 0.0, 0.0, 3)
    r_neutral = compute_reward(prev, 0, new, strict_no_emotion=True)
    new_high_affect = StudentState(0.5, 0.9, 0.8, 0.7, 0.6, 2)
    r_affect = compute_reward(prev, 0, new_high_affect, strict_no_emotion=True)
    assert abs(r_neutral - r_affect) < 1e-9


def test_full_mode_still_blocks_frustrated():
    env = StudentEnv(max_episode_steps=5, population_seed=7, emotion_dynamics="full")
    env.reset(seed=7)
    env._state.frustration = 0.8
    env._state.emotion_id = EMOTION_TO_ID["frustrated"]
    mask = env.get_action_mask()
    assert mask[ACTION_TO_ID["harder_problem"]] == 0
    env.close()
