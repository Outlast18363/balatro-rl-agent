"""cs590_env - Phase-aware Gymnasium wrapper for the Balatro RL environment.

Public API
----------
BalatroPhaseWrapper : gym.Wrapper
    Three-phase wrapper with phase-specific observations and unified masking.
WrapperAction : IntEnum
    Flat action IDs (0-59) including SWAP_JOKER (15-18).
GamePhase : IntEnum
    TRANSITION (blind select), COMBAT (play), SHOP.
build_observation_space, build_action_space : callables
    Gymnasium space constructors for the wrapper's obs/action spaces.
CombatActionWrapper
    Factored action wrapper that bridges per-card binary selections
    + play/discard with the sequential toggle-based BalatroPhaseWrapper.
PooledCombatEnv : gym.Env
    Combat-only env that samples starting positions from a snapshot pool.
    Compatible with ``gymnasium.vector.AsyncVectorEnv`` for true parallelism.
VecRolloutBuffer, compute_gae_vectorized
    Vectorized rollout buffer and per-env GAE for parallel PPO training.
dict_to_tensors, get_card_mask, mask_logits
    Observation/action helpers shared by rollout collection and PPO update.
load_snapshot_pool : callable
    Build a snapshot pool from ``.jkr`` save files on disk.
"""

from cs590_env.schema import (
    build_observation_space,
    build_action_space,
    GamePhase,
    WrapperAction,
    get_wrapper_select_action,
    get_wrapper_select_slot,
)
from cs590_env.wrapper import BalatroPhaseWrapper
from cs590_env.combat_wrapper import CombatActionWrapper
from cs590_env.combat_env import (
    PooledCombatEnv,
    make_pooled_combat_env,
    VecRolloutBuffer,
    compute_gae_vectorized,
    dict_to_tensors,
    get_card_mask,
    mask_logits,
)
from balatro_gym.save_injection import load_snapshot_pool

__all__ = [
    'BalatroPhaseWrapper',
    'WrapperAction',
    'GamePhase',
    'get_wrapper_select_action',
    'get_wrapper_select_slot',
    'build_observation_space',
    'build_action_space',
    'CombatActionWrapper',
    'PooledCombatEnv',
    'make_pooled_combat_env',
    'VecRolloutBuffer',
    'compute_gae_vectorized',
    'dict_to_tensors',
    'get_card_mask',
    'mask_logits',
    'load_snapshot_pool',
]
