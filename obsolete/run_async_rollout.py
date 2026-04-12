#!/usr/bin/env python3
"""Minimal async rollout demo for BalatroPhaseWrapper.

Examples:
    python3 run_async_rollout.py --num-envs 2 --horizon 4
    python3 run_async_rollout.py --num-envs 4 --horizon 8 --policy random
    python3 run_async_rollout.py --num-envs 3 --save-source game_files/first_blind_combat_save.jkr
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Sequence

import numpy as np

from cs590_env import (
    FirstValidPolicy,
    GamePhase,
    PhaseEnvSpec,
    RandomMaskedPolicy,
    WrapperAction,
    collect_vector_rollout,
    make_phase_env,
    make_vector_env_from_specs,
)


def build_demo_specs(
    num_envs: int,
    *,
    seed: int = 0,
    save_source: str | Path | None = None,
    include_snapshot_demo: bool = False,
    validate_save: bool = False,
) -> list[PhaseEnvSpec]:
    """Build a small heterogeneous env-spec list for rollout demos."""
    if num_envs < 1:
        raise ValueError("num_envs must be at least 1")

    specs = [PhaseEnvSpec(seed=seed + idx) for idx in range(num_envs)]

    if include_snapshot_demo and num_envs >= 2:
        specs[1] = PhaseEnvSpec(
            seed=seed + 1,
            snapshot=_build_combat_snapshot(seed + 10_000),
        )

    if save_source is not None:
        specs[-1] = PhaseEnvSpec(
            seed=seed + num_envs - 1,
            save_source=Path(save_source),
            validate_save=validate_save,
        )

    return specs


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the demo runner."""
    parser = argparse.ArgumentParser(description="Async Balatro rollout demo")
    parser.add_argument("--num-envs", type=int, default=2, help="Number of parallel workers")
    parser.add_argument("--horizon", type=int, default=4, help="Number of vector steps to collect")
    parser.add_argument("--seed", type=int, default=0, help="Base seed for worker specs")
    parser.add_argument(
        "--backend",
        choices=("async", "sync"),
        default="async",
        help="Vector env backend; async is the intended concurrent mode",
    )
    parser.add_argument(
        "--policy",
        choices=("first-valid", "random"),
        default="first-valid",
        help="Simple masked policy used for the demo rollout",
    )
    parser.add_argument(
        "--policy-seed",
        type=int,
        default=0,
        help="Random policy seed (ignored by first-valid)",
    )
    parser.add_argument(
        "--save-source",
        type=Path,
        default=None,
        help="Optional .jkr/.json save path to inject into the last worker",
    )
    parser.add_argument(
        "--include-snapshot-demo",
        action="store_true",
        help="Boot worker 1 from a prebuilt combat snapshot",
    )
    parser.add_argument(
        "--validate-save",
        action="store_true",
        help="Run post-injection validation when --save-source is used",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-step phase/action/reward details",
    )
    return parser.parse_args()


def main() -> int:
    """Run a short vectorized rollout and print a compact summary."""
    args = parse_args()
    specs = build_demo_specs(
        args.num_envs,
        seed=args.seed,
        save_source=args.save_source,
        include_snapshot_demo=args.include_snapshot_demo,
        validate_save=args.validate_save,
    )
    policy = _build_policy(args.policy, args.policy_seed)

    vec_env = make_vector_env_from_specs(specs, backend=args.backend)
    try:
        initial_obs, initial_info = vec_env.reset()
        _print_reset_summary(initial_obs, initial_info)

        batch = collect_vector_rollout(
            vec_env,
            policy,
            horizon=args.horizon,
            observation=initial_obs,
            reset=False,
        )
        _print_batch_summary(batch, verbose=args.verbose)
    finally:
        vec_env.close()

    return 0


def _build_combat_snapshot(seed: int) -> dict:
    """Create a deterministic combat-phase snapshot for demo purposes."""
    env = make_phase_env(seed=seed)
    try:
        env.reset(seed=seed)
        env.step(int(WrapperAction.SELECT_BLIND_BASE))
        return deepcopy(env.unwrapped.save_state())
    finally:
        env.close()


def _build_policy(name: str, seed: int):
    """Instantiate one of the small masked-action demo policies."""
    if name == "random":
        return RandomMaskedPolicy(seed=seed)
    return FirstValidPolicy()


def _phase_names(phases: Sequence[int] | np.ndarray) -> list[str]:
    """Convert wrapper phase ids to readable names."""
    phase_ids = np.asarray(phases, dtype=np.int64).reshape(-1)
    return [GamePhase(int(phase_id)).name for phase_id in phase_ids]


def _extract_info_column(info: dict, key: str, width: int) -> list[str]:
    """Read a batched info field from Gymnasium vector env output."""
    if key not in info:
        return [""] * width
    values = np.asarray(info[key], dtype=object).reshape(-1)
    return [str(values[idx]) for idx in range(width)]


def _print_reset_summary(initial_obs: dict[str, np.ndarray], initial_info: dict) -> None:
    """Print the starting state of each worker."""
    phases = _phase_names(initial_obs["phase"])
    valid_counts = np.sum(initial_obs["action_mask"], axis=1).astype(int)
    bootstrap = _extract_info_column(initial_info, "bootstrap_source", len(phases))

    print("Initial worker states")
    for idx, (phase, count, source) in enumerate(zip(phases, valid_counts, bootstrap, strict=True)):
        label = source or "plain_reset"
        print(f"  worker={idx} phase={phase:10s} valid_actions={count:2d} bootstrap={label}")


def _print_batch_summary(batch, *, verbose: bool = False) -> None:
    """Print a compact rollout summary for the collected batch."""
    total_rewards = batch.rewards.sum(axis=0)
    done_counts = batch.dones.sum(axis=0)
    final_phases = _phase_names(batch.next_observation["phase"])

    print("\nRollout summary")
    print(f"  horizon={batch.horizon}")
    print(f"  num_envs={batch.num_envs}")
    print(f"  total_rewards={total_rewards.tolist()}")
    print(f"  done_counts={done_counts.astype(int).tolist()}")
    print(f"  final_phases={final_phases}")

    if not verbose:
        return

    print("\nPer-step details")
    for step in range(batch.horizon):
        phases = _phase_names(batch.observations["phase"][step])
        actions = batch.actions[step].tolist()
        rewards = batch.rewards[step].tolist()
        dones = batch.dones[step].astype(int).tolist()
        print(
            f"  step={step} "
            f"phases={phases} actions={actions} rewards={rewards} dones={dones}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
