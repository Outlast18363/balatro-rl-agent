"""cs590_env/combat_env.py - Parallel combat environment and rollout utilities.

Provides:
* ``PooledCombatEnv`` — a ``gym.Env`` wrapping ``CombatActionWrapper`` with a
  snapshot pool, compatible with ``gymnasium.vector.AsyncVectorEnv``.
* ``VecRolloutBuffer`` — fixed-size ``(T, N)`` buffer for vectorized rollouts.
* ``compute_gae_vectorized`` — per-env GAE over ``(T, N)`` trajectories.
* Observation helpers: ``dict_to_tensors``, ``get_card_mask``, ``mask_logits``.
"""

from __future__ import annotations

from copy import deepcopy
from functools import partial
from typing import Any, Sequence

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from balatro_gym.balatro_env_2 import BalatroEnv, DeterministicRNG
from balatro_gym.constants import MAX_HAND_SIZE
from cs590_env.combat_wrapper import CombatActionWrapper
from cs590_env.schema import build_observation_space
from cs590_env.wrapper import BalatroPhaseWrapper


class PooledCombatEnv(gym.Env):
    """Combat-only Gymnasium env that samples starting positions from a pool.

    On every ``reset()``, a random snapshot is drawn from the pool and loaded
    into the underlying ``BalatroEnv`` with a fresh RNG seed so card draws
    diverge across episodes and workers.

    When used under a vector env, prefer same-step autoreset so the wrapper can
    expose the true terminal/truncation flags without burning an extra step on a
    deferred reset observation.

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
        self._base_env.rng = DeterministicRNG(fresh_seed)

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

        obs, reward, terminated, truncated, info = self._combat.step(
            card_selections, execution
        )
        return obs, reward, terminated, truncated, info

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


# ═══════════════════════════════════════════════════════════════════
# Observation / action helpers
# ═══════════════════════════════════════════════════════════════════

def dict_to_tensors(obs_np: dict, dev: torch.device) -> dict:
    """Convert a vectorized observation dict ``{key: (N, ...)}`` to GPU tensors."""
    return {k: torch.as_tensor(np.asarray(v), device=dev) for k, v in obs_np.items()}


def get_card_mask(obs_t: dict) -> torch.Tensor:
    """``(B, MAX_HAND_SIZE)`` bool — True for real cards, False for padding."""
    return obs_t['hand_card_ids'] >= 0


def mask_logits(sel_logits: torch.Tensor, card_mask: torch.Tensor) -> torch.Tensor:
    """Zero-out the 'select' logit for empty card slots.

    Args:
        sel_logits: ``(B, MAX_HAND_SIZE, 2)``
        card_mask:  ``(B, MAX_HAND_SIZE)`` bool

    Returns:
        Masked copy safe for autograd.
    """
    masked = sel_logits.clone()
    masked[:, :, 1] = masked[:, :, 1].masked_fill(~card_mask, -1e9)
    return masked


# ═══════════════════════════════════════════════════════════════════
# Vectorized rollout buffer
# ═══════════════════════════════════════════════════════════════════

class VecRolloutBuffer:
    """Fixed-size ``(T, N)`` rollout buffer for vectorized environments.

    All tensors live on the target device from the start.  Observations are
    stored as ``dict[str, Tensor]`` with shape ``(T, N, ...)``.
    """

    def __init__(self, T: int, N: int, dev: torch.device):
        self.T, self.N, self.dev = T, N, dev

        self.card_sels  = torch.zeros(T, N, MAX_HAND_SIZE, device=dev, dtype=torch.long)
        self.executions = torch.zeros(T, N, device=dev, dtype=torch.long)
        self.log_probs  = torch.zeros(T, N, device=dev)
        self.values     = torch.zeros(T, N, device=dev)
        self.rewards    = torch.zeros(T, N, device=dev)
        self.dones      = torch.zeros(T, N, device=dev)
        self.obs: dict[str, torch.Tensor] = {}

    def store_step(
        self,
        t: int,
        obs_t: dict[str, torch.Tensor],
        card_sels: torch.Tensor,
        executions: torch.Tensor,
        log_probs: torch.Tensor,
        values: torch.Tensor,
        rewards: np.ndarray,
        dones: np.ndarray,
    ):
        """Write one rollout time-step for all N envs."""
        if not self.obs:
            self.obs = {
                k: torch.zeros(self.T, self.N, *v.shape[1:],
                               device=self.dev, dtype=v.dtype)
                for k, v in obs_t.items()
            }
        for k, v in obs_t.items():
            self.obs[k][t] = v

        self.card_sels[t]  = card_sels.detach()
        self.executions[t] = executions.detach()
        self.log_probs[t]  = log_probs.detach()
        self.values[t]     = values.detach()
        self.rewards[t]    = torch.as_tensor(rewards, device=self.dev)
        self.dones[t]      = torch.as_tensor(dones.astype(np.float32), device=self.dev)

    def flatten(self):
        """Reshape everything from ``(T, N, ...)`` to ``(T*N, ...)`` for PPO."""
        TN = self.T * self.N
        flat_obs = {k: v.reshape(TN, *v.shape[2:]) for k, v in self.obs.items()}
        return (
            flat_obs,
            self.card_sels.reshape(TN, MAX_HAND_SIZE),
            self.executions.reshape(TN),
            self.log_probs.reshape(TN),
        )


# ═══════════════════════════════════════════════════════════════════
# Per-env GAE
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_gae_vectorized(
    rewards: torch.Tensor,      # (T, N)
    values: torch.Tensor,       # (T, N)
    next_values: torch.Tensor,  # (N,)
    dones: torch.Tensor,        # (T, N)
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-env Generalized Advantage Estimation over a ``(T, N)`` rollout.

    The reverse sweep runs over T time-steps; all N envs are handled
    simultaneously (no Python loop over envs).

    Returns:
        advantages: ``(T, N)``  normalized over all T*N entries
        returns:    ``(T, N)``
    """
    T, N = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(N, device=rewards.device)

    for t in reversed(range(T)):
        next_val = next_values if t == T - 1 else values[t + 1]
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_val * not_done - values[t]
        last_gae = delta + gamma * gae_lambda * not_done * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages, returns
