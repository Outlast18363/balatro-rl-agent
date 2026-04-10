from pathlib import Path

import numpy as np

from cs590_env.schema import GamePhase
from cs590_env.schema import WrapperAction
from cs590_env.rollout import (
    FirstValidPolicy,
    PhaseEnvSpec,
    RandomMaskedPolicy,
    collect_branch_rollout,
    collect_vector_rollout,
    make_phase_env,
    make_vector_env,
    make_vector_env_from_specs,
)


def test_random_masked_policy_respects_batched_masks():
    observation = {
        "action_mask": np.array(
            [
                [0, 1, 0, 1, 0],
                [1, 0, 0, 0, 0],
                [0, 0, 1, 1, 1],
            ],
            dtype=np.int8,
        )
    }

    actions = RandomMaskedPolicy(seed=7)(observation)

    assert actions.shape == (3,)
    assert np.all(observation["action_mask"][np.arange(3), actions] == 1)


def test_collect_vector_rollout_with_sync_backend():
    vec_env = make_vector_env(num_envs=2, seed=11, backend="sync")
    try:
        batch = collect_vector_rollout(vec_env, FirstValidPolicy(), horizon=4)
    finally:
        vec_env.close()

    assert batch.actions.shape == (4, 2)
    assert batch.rewards.shape == (4, 2)
    assert batch.terminations.shape == (4, 2)
    assert batch.observations["action_mask"].shape[:2] == (4, 2)

    for step in range(batch.horizon):
        mask = batch.observations["action_mask"][step]
        chosen = batch.actions[step]
        assert np.all(mask[np.arange(batch.num_envs), chosen] == 1)


def test_make_vector_env_from_specs_applies_snapshot_bootstrap():
    source_env = make_phase_env(seed=31)
    try:
        source_env.reset(seed=31)
        source_env.step(int(WrapperAction.SELECT_BLIND_BASE))
        combat_snapshot = source_env.unwrapped.save_state()
    finally:
        source_env.close()

    specs = [
        PhaseEnvSpec(seed=41),
        PhaseEnvSpec(seed=42, snapshot=combat_snapshot),
    ]

    vec_env = make_vector_env_from_specs(specs, backend="sync")
    try:
        obs, infos = vec_env.reset()
    finally:
        vec_env.close()

    assert obs["phase"].shape == (2,)
    assert int(obs["phase"][0]) == int(GamePhase.TRANSITION)
    assert int(obs["phase"][1]) == int(GamePhase.COMBAT)
    assert list(infos["bootstrap_source"]) == ["", "snapshot"]


def test_make_vector_env_from_specs_applies_save_injection_bootstrap():
    repo_root = Path(__file__).resolve().parents[1]
    save_path = repo_root / "game_files" / "first_blind_combat_save.jkr"

    specs = [
        PhaseEnvSpec(seed=51, save_source=save_path, validate_save=False),
        PhaseEnvSpec(seed=52),
    ]

    vec_env = make_vector_env_from_specs(specs, backend="sync")
    try:
        obs, infos = vec_env.reset()
    finally:
        vec_env.close()

    assert int(obs["phase"][0]) == int(GamePhase.COMBAT)
    assert int(obs["phase"][1]) == int(GamePhase.TRANSITION)
    assert list(infos["bootstrap_source"]) == ["save_injection", ""]


def test_collect_vector_rollout_with_async_backend_and_specs():
    specs = [
        PhaseEnvSpec(seed=61),
        PhaseEnvSpec(seed=62),
    ]
    vec_env = make_vector_env_from_specs(specs, backend="async")
    try:
        batch = collect_vector_rollout(vec_env, FirstValidPolicy(), horizon=2)
    finally:
        vec_env.close()

    assert batch.actions.shape == (2, 2)
    assert batch.observations["action_mask"].shape[:2] == (2, 2)
    for step in range(batch.horizon):
        mask = batch.observations["action_mask"][step]
        chosen = batch.actions[step]
        assert np.all(mask[np.arange(batch.num_envs), chosen] == 1)


def test_collect_branch_rollout_from_snapshot():
    env = make_phase_env(seed=23)
    env.reset(seed=23)
    snapshot = env.unwrapped.save_state()

    rollout = collect_branch_rollout(
        env,
        snapshot=snapshot,
        policy=FirstValidPolicy(),
        max_steps=3,
    )

    assert rollout.length >= 1
    assert rollout.actions.shape == (rollout.length,)
    assert rollout.observations["action_mask"].shape[0] == rollout.length
    assert np.all(
        rollout.observations["action_mask"][np.arange(rollout.length), rollout.actions]
        == 1
    )
