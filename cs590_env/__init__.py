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
    WrapperAction,
    GamePhase,
    build_observation_space,
    build_action_space,
)
from cs590_env.wrapper import BalatroPhaseWrapper

__all__ = [
    'BalatroPhaseWrapper',
    'WrapperAction',
    'GamePhase',
    'build_observation_space',
    'build_action_space',
]
