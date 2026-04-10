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
"""

from cs590_env.schema import (
    build_observation_space,
    build_action_space,
    GamePhase,
    WrapperAction,
    get_wrapper_select_action,
    get_wrapper_select_slot,
)
from cs590_env.rollout import (
    ConfiguredPhaseEnv,
    FirstValidPolicy,
    PhaseEnvSpec,
    RandomMaskedPolicy,
    VectorRolloutBatch,
    SingleEnvRollout,
    collect_branch_rollout,
    collect_vector_rollout,
    make_phase_env,
    make_phase_env_from_spec,
    make_vector_env,
    make_vector_env_from_specs,
)
from cs590_env.wrapper import BalatroPhaseWrapper

__all__ = [
    'BalatroPhaseWrapper',
    'WrapperAction',
    'GamePhase',
    'get_wrapper_select_action',
    'get_wrapper_select_slot',
    'build_observation_space',
    'build_action_space',
    'ConfiguredPhaseEnv',
    'FirstValidPolicy',
    'PhaseEnvSpec',
    'RandomMaskedPolicy',
    'VectorRolloutBatch',
    'SingleEnvRollout',
    'collect_branch_rollout',
    'collect_vector_rollout',
    'make_phase_env',
    'make_phase_env_from_spec',
    'make_vector_env',
    'make_vector_env_from_specs',
]
