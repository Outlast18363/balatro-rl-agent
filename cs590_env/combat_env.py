"""cs590_env/combat_env.py - Gymnasium env for pooled combat-phase training.

Wraps ``CombatActionWrapper`` as a proper ``gym.Env`` with a flat
``MultiBinary(MAX_HAND_SIZE + 1)`` action space so it can be used with
``gymnasium.vector.AsyncVectorEnv`` for true multiprocess parallelism.

Each ``PooledCombatEnv`` draws starting positions from a shared snapshot
pool, giving the agent diverse combat situations every episode.
"""

from __future__ import annotations

from copy import deepcopy
from functools import partial
from typing import Any, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from balatro_gym.balatro_env_2 import BalatroEnv
from balatro_gym.constants import MAX_HAND_SIZE
from cs590_env.combat_wrapper import CombatActionWrapper
from cs590_env.schema import build_observation_space
from cs590_env.wrapper import BalatroPhaseWrapper


class PooledCombatEnv(gym.Env):
    """Combat-only Gymnasium env that samples starting positions from a pool.

    On every ``reset()``, a random snapshot is drawn from the pool and loaded
    into the underlying ``BalatroEnv`` with a fresh RNG seed so card draws
    diverge across episodes and workers.

    The action is a flat ``MultiBinary(MAX_HAND_SIZE + 1)`` array:
      * ``action[:MAX_HAND_SIZE]``  — per-card binary selection
      * ``action[MAX_HAND_SIZE]``   — execution (0 = play, 1 = discard)

    The internal ``CombatActionWrapper`` translates this into the sequential
    toggle + play/discard calls that the base environment expects.

    Args:
        snapshot_pool: Non-empty list of snapshot dicts produced by
            ``BalatroEnv.save_state()``.  Shared (read-only) across workers.
        pool_seed: Seed for the pool-sampling RNG.  Give each worker a
            unique seed so they don't draw the same sequence of snapshots.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        snapshot_pool: Sequence[dict[str, Any]],
        pool_seed: int = 0,
    ):
        super().__init__()
        if not snapshot_pool:
            raise ValueError("snapshot_pool must be non-empty")

        self._pool = list(snapshot_pool)
        self._rng = np.random.default_rng(pool_seed)

        self.observation_space = build_observation_space()
        self.action_space = spaces.MultiBinary(MAX_HAND_SIZE + 1)

        self._base_env = BalatroEnv()
        self._phase_env = BalatroPhaseWrapper(self._base_env)
        self._combat = CombatActionWrapper(self._phase_env)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        idx = int(self._rng.integers(0, len(self._pool)))
        snapshot = self._pool[idx]
        fresh_seed = int(self._rng.integers(0, 2**31))

        self._base_env.reset(seed=fresh_seed)
        self._base_env.load_state(deepcopy(snapshot))

        self._phase_env._auto_skip_pack_open()
        obs = self._phase_env._get_phase_observation()
        obs = self._combat._advance_to_combat(obs)
        self._combat._last_obs = obs

        return obs, {"snapshot_idx": idx, "env_seed": fresh_seed}

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        card_selections = action[: MAX_HAND_SIZE]
        execution = int(action[MAX_HAND_SIZE])

        obs, reward, done, info = self._combat.step(card_selections, execution)
        return obs, reward, done, False, info

    def close(self) -> None:
        self._combat.close()


def make_pooled_combat_env(
    snapshot_pool: Sequence[dict[str, Any]],
    pool_seed: int = 0,
) -> PooledCombatEnv:
    """Factory callable for use with ``AsyncVectorEnv``.

    Example::

        from functools import partial
        from gymnasium.vector import AsyncVectorEnv

        fns = [partial(make_pooled_combat_env, pool, pool_seed=base + i)
               for i in range(num_envs)]
        vec_env = AsyncVectorEnv(fns)
    """
    return PooledCombatEnv(snapshot_pool, pool_seed=pool_seed)
