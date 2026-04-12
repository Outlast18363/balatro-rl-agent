"""cs590_env/snapshots.py - Snapshot pool construction from .jkr save files."""

from __future__ import annotations

from pathlib import Path

from balatro_gym.save_injection import inject_save_into_balatro_env


def load_snapshot_pool(save_dir: str = "save_files", seed: int = 42) -> list[dict]:
    """Load .jkr save files and convert them into env snapshots.

    Each ``.jkr`` file is injected into a fresh ``BalatroEnv`` and its full
    state is captured via ``save_state()``.  The resulting list of snapshot
    dicts can be passed directly to ``PooledCombatEnv`` or
    ``make_pooled_combat_env``.

    Args:
        save_dir: Directory containing ``.jkr`` files.
        seed: Seed forwarded to ``inject_save_into_balatro_env``.

    Returns:
        Non-empty list of snapshot dicts.

    Raises:
        FileNotFoundError: If *save_dir* contains no ``.jkr`` files.
    """
    pool: list[dict] = []
    for jkr_path in sorted(Path(save_dir).glob("*.jkr")):
        env, _report = inject_save_into_balatro_env(jkr_path, seed=seed)
        pool.append(env.save_state())
    if not pool:
        raise FileNotFoundError(f"No .jkr files found in {save_dir}/")
    return pool
