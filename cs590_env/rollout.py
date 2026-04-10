"""Gymnasium-native rollout helpers for BalatroPhaseWrapper.

This module keeps rollout collection independent from Stable-Baselines3 while
remaining compatible with Gymnasium vector environments. The base policy API is
intentionally small: given an observation dict containing ``action_mask``,
return one valid action per environment.

For true concurrent rollout collection, use ``backend="async"`` together with
``make_vector_env_from_specs(...)`` so each worker can bootstrap from its own
seed, snapshot, or save-file injection state.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

import gymnasium as gym
import numpy as np
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv, VectorEnv

from balatro_gym.balatro_env_2 import BalatroEnv
from balatro_gym.save_injection import inject_save_into_balatro_env
from cs590_env.wrapper import BalatroPhaseWrapper

RolloutBackend = Literal["sync", "async"]
ObservationDict = Mapping[str, np.ndarray]
VectorPolicy = Callable[[ObservationDict], int | np.ndarray]


@dataclass(frozen=True)
class PhaseEnvSpec:
    """Bootstrap configuration for one wrapped Balatro environment worker.

    Exactly one of ``snapshot`` or ``save_source`` may be provided.
    ``seed`` is used for the underlying env construction and as the default
    ``reset(seed=...)`` value when the vector env resets the worker.
    """

    seed: int | None = None
    snapshot: dict[str, Any] | None = None
    save_source: str | Path | Mapping[str, Any] | None = None
    reset_options: dict[str, Any] | None = None
    validate_save: bool = True

    def __post_init__(self) -> None:
        if self.snapshot is not None and self.save_source is not None:
            raise ValueError("PhaseEnvSpec accepts at most one of snapshot/save_source")


class ConfiguredPhaseEnv(BalatroPhaseWrapper):
    """Phase wrapper whose reset behavior is driven by a ``PhaseEnvSpec``."""

    def __init__(self, spec: PhaseEnvSpec):
        super().__init__(BalatroEnv(seed=spec.seed))
        self._spec = spec

    def reset(self, *, seed=None, options=None) -> tuple[dict, dict]:
        effective_seed = self._spec.seed if seed is None else seed
        effective_options = _merge_reset_options(self._spec.reset_options, options)

        if self._spec.save_source is not None:
            _, report = inject_save_into_balatro_env(
                self._spec.save_source,
                env=self.env,
                seed=0 if effective_seed is None else int(effective_seed),
                validate=self._spec.validate_save,
            )
            self._auto_skip_pack_open()
            obs = self._get_phase_observation()
            info = _build_reset_info(self._game_phase.name, "save_injection")
            info["save_injection_validated"] = self._spec.validate_save
            info["save_injection_ignored_paths"] = report.get("ignored_from_save_total", 0)
            return obs, info

        if self._spec.snapshot is not None:
            self.env.reset(seed=effective_seed, options=effective_options)
            self.env.load_state(deepcopy(self._spec.snapshot))
            self._auto_skip_pack_open()
            obs = self._get_phase_observation()
            return obs, _build_reset_info(self._game_phase.name, "snapshot")

        obs, info = super().reset(seed=effective_seed, options=effective_options)
        info["bootstrap_source"] = ""
        return obs, info


@dataclass(frozen=True)
class VectorRolloutBatch:
    """Fixed-horizon rollout batch collected from a vector environment."""

    observations: dict[str, np.ndarray]
    actions: np.ndarray
    rewards: np.ndarray
    terminations: np.ndarray
    truncations: np.ndarray
    infos: list[dict[str, Any]]
    next_observation: dict[str, np.ndarray]

    @property
    def dones(self) -> np.ndarray:
        """Boolean done flags with the same shape as ``rewards``."""
        return np.logical_or(self.terminations, self.truncations)

    @property
    def horizon(self) -> int:
        """Number of time steps stored in the batch."""
        return int(self.actions.shape[0])

    @property
    def num_envs(self) -> int:
        """Number of parallel environments represented in the batch."""
        return int(self.actions.shape[1])


@dataclass(frozen=True)
class SingleEnvRollout:
    """Step sequence collected from one wrapped environment."""

    observations: dict[str, np.ndarray]
    actions: np.ndarray
    rewards: np.ndarray
    terminations: np.ndarray
    truncations: np.ndarray
    infos: list[dict[str, Any]]
    next_observation: dict[str, np.ndarray]

    @property
    def dones(self) -> np.ndarray:
        """Boolean done flags for each stored step."""
        return np.logical_or(self.terminations, self.truncations)

    @property
    def length(self) -> int:
        """Number of stored steps."""
        return int(self.actions.shape[0])


class FirstValidPolicy:
    """Deterministic baseline that always selects the first legal action."""

    def __call__(self, observation: ObservationDict) -> int | np.ndarray:
        mask = np.asarray(observation["action_mask"])
        if mask.ndim == 1:
            return _first_valid_action(mask)
        return np.asarray([_first_valid_action(row) for row in mask], dtype=np.int64)


class RandomMaskedPolicy:
    """Uniform random policy over legal actions only."""

    def __init__(self, seed: int | None = None):
        self._rng = np.random.default_rng(seed)

    def __call__(self, observation: ObservationDict) -> int | np.ndarray:
        mask = np.asarray(observation["action_mask"])
        if mask.ndim == 1:
            return _random_valid_action(mask, self._rng)
        return np.asarray(
            [_random_valid_action(row, self._rng) for row in mask],
            dtype=np.int64,
        )


def make_phase_env(seed: int | None = None) -> BalatroPhaseWrapper:
    """Construct the current project-standard single environment."""
    return BalatroPhaseWrapper(BalatroEnv(seed=seed))


def make_phase_env_from_spec(spec: PhaseEnvSpec) -> BalatroPhaseWrapper:
    """Construct one wrapped env whose reset behavior follows ``spec``."""
    return ConfiguredPhaseEnv(spec)


def make_vector_env_from_specs(
    env_specs: Sequence[PhaseEnvSpec],
    *,
    backend: RolloutBackend = "async",
) -> VectorEnv:
    """Create a vector env whose workers can start from different bootstraps.

    This is the preferred entrypoint when the project requirement is "run
    multiple independent environments truly at the same time" because each
    worker can carry its own seed / snapshot / save-injection setup while still
    sharing the same wrapper API.
    """
    if len(env_specs) < 1:
        raise ValueError("env_specs must contain at least one worker spec")

    env_fns = [partial(make_phase_env_from_spec, spec) for spec in env_specs]
    return _build_vector_env(env_fns, backend)


def make_vector_env(
    num_envs: int,
    *,
    seed: int = 0,
    backend: RolloutBackend = "sync",
    env_factory: Callable[[int | None], BalatroPhaseWrapper] = make_phase_env,
) -> VectorEnv:
    """Create a Gymnasium vector env over wrapped Balatro environments.

    The collector below is written against the Gymnasium vector API so that the
    backend can switch from ``SyncVectorEnv`` to ``AsyncVectorEnv`` without
    touching training code.
    """
    if num_envs < 1:
        raise ValueError("num_envs must be at least 1")

    env_fns = [partial(env_factory, seed + rank) for rank in range(num_envs)]
    return _build_vector_env(env_fns, backend)


def collect_vector_rollout(
    vector_env: VectorEnv,
    policy: VectorPolicy,
    horizon: int,
    *,
    observation: ObservationDict | None = None,
    reset: bool = True,
) -> VectorRolloutBatch:
    """Collect a fixed-horizon rollout from a vectorized Balatro setup.

    Args:
        vector_env: Gymnasium vector env with batched ``action_mask`` in the obs.
        policy: Callable that returns one legal action per environment.
        horizon: Number of vector steps to collect.
        observation: Optional already-live observation to start from.
        reset: When True and no observation is supplied, call ``vector_env.reset()``.

    Returns:
        ``VectorRolloutBatch`` with time-major arrays: ``(T, N, ...)``.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1")

    if observation is None:
        if not reset:
            raise ValueError("observation must be provided when reset=False")
        observation, _ = vector_env.reset()

    obs_steps: list[dict[str, np.ndarray]] = []
    action_steps: list[np.ndarray] = []
    reward_steps: list[np.ndarray] = []
    termination_steps: list[np.ndarray] = []
    truncation_steps: list[np.ndarray] = []
    info_steps: list[dict[str, Any]] = []

    current_obs = _copy_observation(observation)
    num_envs = int(vector_env.num_envs)

    for _ in range(horizon):
        actions = np.asarray(policy(current_obs), dtype=np.int64)
        if actions.shape != (num_envs,):
            raise ValueError(
                f"policy must return shape ({num_envs},), got {actions.shape}"
            )
        _validate_masked_actions(current_obs["action_mask"], actions)

        next_obs, rewards, terminations, truncations, infos = vector_env.step(actions)

        obs_steps.append(_copy_observation(current_obs))
        action_steps.append(np.array(actions, copy=True))
        reward_steps.append(np.array(rewards, copy=True))
        termination_steps.append(np.array(terminations, copy=True))
        truncation_steps.append(np.array(truncations, copy=True))
        info_steps.append(deepcopy(infos))

        current_obs = _copy_observation(next_obs)

    return VectorRolloutBatch(
        observations=_stack_observation_sequence(obs_steps),
        actions=np.stack(action_steps, axis=0),
        rewards=np.stack(reward_steps, axis=0),
        terminations=np.stack(termination_steps, axis=0),
        truncations=np.stack(truncation_steps, axis=0),
        infos=info_steps,
        next_observation=current_obs,
    )


def collect_branch_rollout(
    env: BalatroPhaseWrapper,
    snapshot: dict[str, Any],
    policy: VectorPolicy,
    max_steps: int,
) -> SingleEnvRollout:
    """Replay a snapshot from one decision point using a single-env policy.

    This is the reproducible counterpart to parallel online rollout collection:
    load a saved state, try a policy branch, then restore the same snapshot for
    another attempt.
    """
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")

    base_env = env.unwrapped
    if not isinstance(base_env, BalatroEnv):
        raise TypeError("collect_branch_rollout expects a BalatroPhaseWrapper")

    base_env.load_state(snapshot)
    current_obs = _copy_observation(env._get_phase_observation())

    obs_steps: list[dict[str, np.ndarray]] = []
    actions: list[int] = []
    rewards: list[float] = []
    terminations: list[bool] = []
    truncations: list[bool] = []
    info_steps: list[dict[str, Any]] = []

    for _ in range(max_steps):
        action = _coerce_single_action(policy(current_obs))
        _validate_masked_actions(current_obs["action_mask"], action)

        next_obs, reward, terminated, truncated, info = env.step(action)

        obs_steps.append(_copy_observation(current_obs))
        actions.append(action)
        rewards.append(float(reward))
        terminations.append(bool(terminated))
        truncations.append(bool(truncated))
        info_steps.append(deepcopy(info))

        current_obs = _copy_observation(next_obs)
        if terminated or truncated:
            break

    return SingleEnvRollout(
        observations=_stack_observation_sequence(obs_steps),
        actions=np.asarray(actions, dtype=np.int64),
        rewards=np.asarray(rewards, dtype=np.float32),
        terminations=np.asarray(terminations, dtype=bool),
        truncations=np.asarray(truncations, dtype=bool),
        infos=info_steps,
        next_observation=current_obs,
    )


def _copy_observation(observation: ObservationDict) -> dict[str, np.ndarray]:
    """Deep-copy an observation dict into plain numpy arrays."""
    return {key: np.array(value, copy=True) for key, value in observation.items()}


def _stack_observation_sequence(
    observations: list[dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    """Stack a time sequence of observations into a time-major dict."""
    if not observations:
        raise ValueError("cannot stack an empty observation sequence")
    keys = observations[0].keys()
    return {
        key: np.stack([step[key] for step in observations], axis=0)
        for key in keys
    }


def _first_valid_action(mask: np.ndarray) -> int:
    """Return the first valid action from a 1D action mask."""
    valid = np.flatnonzero(mask)
    if len(valid) == 0:
        raise ValueError("received an action_mask with no valid actions")
    return int(valid[0])


def _random_valid_action(mask: np.ndarray, rng: np.random.Generator) -> int:
    """Sample uniformly from the legal actions of a 1D action mask."""
    valid = np.flatnonzero(mask)
    if len(valid) == 0:
        raise ValueError("received an action_mask with no valid actions")
    return int(rng.choice(valid))


def _coerce_single_action(action: int | np.ndarray) -> int:
    """Normalize policy output for the single-env replay path."""
    array = np.asarray(action, dtype=np.int64)
    if array.shape != ():
        raise ValueError(f"single-env policy must return a scalar, got {array.shape}")
    return int(array.item())


def _validate_masked_actions(
    action_mask: np.ndarray,
    actions: int | np.ndarray,
) -> None:
    """Raise if the policy proposes an action masked out by the env."""
    mask = np.asarray(action_mask)

    if mask.ndim == 1:
        action = int(actions)
        if mask[action] != 1:
            raise ValueError(f"action {action} is masked out")
        return

    batched_actions = np.asarray(actions, dtype=np.int64)
    if batched_actions.shape != (mask.shape[0],):
        raise ValueError(
            "batched actions must align with the first dimension of action_mask"
        )
    if not np.all(mask[np.arange(mask.shape[0]), batched_actions] == 1):
        raise ValueError("policy proposed at least one masked-out action")


def _build_vector_env(
    env_fns: Sequence[Callable[[], gym.Env]],
    backend: RolloutBackend,
) -> VectorEnv:
    """Instantiate the requested Gymnasium vector backend."""
    if backend == "sync":
        return SyncVectorEnv(env_fns)
    if backend == "async":
        return AsyncVectorEnv(env_fns)
    raise ValueError("backend must be either 'sync' or 'async'")


def _merge_reset_options(
    base: Mapping[str, Any] | None,
    override: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Merge optional reset option dictionaries."""
    merged: dict[str, Any] = {}
    if base:
        merged.update(base)
    if override:
        merged.update(override)
    return merged or None


def _build_reset_info(phase_name: str, bootstrap_source: str) -> dict[str, Any]:
    """Standard reset info for spec-configured workers."""
    return {
        "phase": phase_name,
        "phase_changed": True,
        "previous_phase": None,
        "bootstrap_source": bootstrap_source,
    }
