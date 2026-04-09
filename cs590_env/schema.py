"""cs590_env/schema.py - Single source of truth for wrapper constants, enums, and space builders.

Defines the flat Discrete(60) action space, Dict observation space, and all
constants/enums needed by BalatroPhaseWrapper.
"""

from __future__ import annotations

from enum import IntEnum, unique

import numpy as np
from gymnasium import spaces

# ─── Slot / padding constants ─────────────────────────────────────────────────

ACTION_SPACE_SIZE = 60
MAX_DECK_SIZE = 52            # shared deck-histogram upper bound
MAX_JOKER_DISPLAY = 10       # padded joker slots in observation
MAX_CONSUMABLE_DISPLAY = 5   # padded consumable slots in observation
MAX_HAND_SIZE = 8            # max cards displayed in hand
MAX_SHOP_ITEMS = 10          # max shop inventory slots
NUM_HAND_TYPES = 12          # HandType.HIGH_CARD(0) .. FLUSH_FIVE(11)
NUM_RANKS = 13               # 2 .. Ace
NUM_SUITS = 4
NUM_BOSS_BLINDS = 28         # BossBlindType enum size
NUM_VOUCHERS = 16            # placeholder for voucher binary flags


# ─── WrapperAction enum ──────────────────────────────────────────────────────

@unique
class WrapperAction(IntEnum):
    """Flat action IDs for the wrapper's Discrete(60) space.

    Mirrors the base env Action layout and fills the 15-18 gap with
    SWAP_JOKER actions for in-place joker reordering.
    """
    PLAY_HAND            = 0
    DISCARD              = 1
    SELECT_CARD_BASE     = 2    # 2-9   (8 slots)
    USE_CONSUMABLE_BASE  = 10   # 10-14 (5 slots)
    SWAP_JOKER_BASE      = 15   # 15-18 swap slot i ↔ i+1 [wrapper-only]
    SHOP_BUY_BASE        = 20   # 20-29 (10 slots)
    SHOP_REROLL          = 30
    SHOP_END             = 31
    SELL_JOKER_BASE      = 32   # 32-36 (5 slots)
    SELL_CONSUMABLE_BASE = 37   # 37-41 (5 slots)
    SELECT_BLIND_BASE    = 45   # 45-47 (small / big / boss)
    SKIP_BLIND           = 48
    SELECT_FROM_PACK_BASE = 50  # 50-54 (5 slots)
    SKIP_PACK            = 55


# Range counts for each parameterised action group
SELECT_CARD_COUNT      = 8
USE_CONSUMABLE_COUNT   = 5
SWAP_JOKER_COUNT       = 4   # i ∈ {0,1,2,3}
SHOP_BUY_COUNT         = 10
SELL_JOKER_COUNT       = 5
SELL_CONSUMABLE_COUNT  = 5
SELECT_BLIND_COUNT     = 3
SELECT_FROM_PACK_COUNT = 5


# ─── GamePhase enum ──────────────────────────────────────────────────────────

@unique
class GamePhase(IntEnum):
    """Wrapper-level phase labels mapped from the base env's Phase enum.

    TRANSITION = base Phase.BLIND_SELECT (2),
    COMBAT     = base Phase.PLAY (0),
    SHOP       = base Phase.SHOP (1).
    """
    TRANSITION = 0  # blind selection
    COMBAT     = 1  # play / scoring
    SHOP       = 2  # shop


# ─── Consumable sell-value helper ─────────────────────────────────────────────

_PLANET_NAMES = frozenset({
    'Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn',
    'Uranus', 'Neptune', 'Pluto', 'Planet X', 'Ceres', 'Eris',
})


def consumable_sell_value(name: str) -> int:
    """Return the sell value of a consumable by name.

    Args:
        name: consumable string identifier (e.g. 'Mercury', 'The Fool').

    Returns:
        4 for planet cards, 3 for tarots and spectrals.
    """
    return 4 if name in _PLANET_NAMES else 3


# ─── Space builders ──────────────────────────────────────────────────────────

def build_observation_space() -> spaces.Dict:
    """Construct the unified Dict observation space.

    All fields are present in every observation regardless of phase.
    Phase-irrelevant fields are zero-filled (or -1 for empty card IDs)
    so tensor shapes remain constant across steps.

    Returns:
        gymnasium.spaces.Dict whose keys and shapes match the wrapper's obs dicts.
    """
    return spaces.Dict({
        # ── Global token (all phases) ─────────────────────────
        'ante':                   spaces.Box(0, 1000, (), np.int16),
        'round':                  spaces.Box(0, 3, (), np.int8),
        'phase':                  spaces.Box(0, 2, (), np.int8),
        'money':                  spaces.Box(-1000, 1_000_000, (), np.int32),
        'next_boss_blind_id':     spaces.Box(0, NUM_BOSS_BLINDS + 1, (), np.int8),

        'joker_ids':              spaces.Box(0, 200, (MAX_JOKER_DISPLAY,), np.int16),
        'joker_sell_values':      spaces.Box(0, 200, (MAX_JOKER_DISPLAY,), np.int16),
        'joker_is_empty':         spaces.MultiBinary(MAX_JOKER_DISPLAY),

        'consumable_ids':         spaces.Box(0, 100, (MAX_CONSUMABLE_DISPLAY,), np.int16),
        'consumable_sell_values': spaces.Box(0, 20, (MAX_CONSUMABLE_DISPLAY,), np.int8),
        'consumable_is_empty':    spaces.MultiBinary(MAX_CONSUMABLE_DISPLAY),

        'vouchers_owned':         spaces.MultiBinary(NUM_VOUCHERS),

        # [id, level, chip, mult] per hand type
        'hand_levels':            spaces.Box(0, 500, (NUM_HAND_TYPES, 4), np.int16),

        'action_mask':            spaces.MultiBinary(ACTION_SPACE_SIZE),

        # ── Deck histogram (all phases; content varies) ───────
        'deck_ranks':             spaces.Box(0, MAX_DECK_SIZE, (NUM_RANKS,), np.int8),
        'deck_suits':             spaces.Box(0, MAX_DECK_SIZE, (NUM_SUITS,), np.int8),

        # ── Transition (BLIND_SELECT) fields ──────────────────
        'blind_type':             spaces.Box(0, 3, (), np.int8),
        'target_score':           spaces.Box(0, 1_000_000_000, (), np.int32),
        'blind_reward':           spaces.Box(0, 1_000_000, (), np.int32),

        # ── Combat (PLAY) fields ──────────────────────────────
        'hand_card_ids':          spaces.Box(-1, 51, (MAX_HAND_SIZE,), np.int8),
        'hand_card_enhancements': spaces.Box(0, 10, (MAX_HAND_SIZE,), np.int8),
        'hand_card_editions':     spaces.Box(0, 5, (MAX_HAND_SIZE,), np.int8),
        'hand_card_seals':        spaces.Box(0, 5, (MAX_HAND_SIZE,), np.int8),
        'hand_is_face_down':      spaces.MultiBinary(MAX_HAND_SIZE),
        'hand_is_selected':       spaces.MultiBinary(MAX_HAND_SIZE),
        'hand_is_debuffed':       spaces.MultiBinary(MAX_HAND_SIZE),
        'current_score':          spaces.Box(0, 1_000_000_000, (), np.int32),
        'hand_size':              spaces.Box(0, 12, (), np.int8),
        'hands_remaining':        spaces.Box(0, 12, (), np.int8),
        'discards_remaining':     spaces.Box(0, 12, (), np.int8),
        'hands_played_round':     spaces.Box(0, 100_000, (), np.int32),
        'boss_id':                spaces.Box(0, NUM_BOSS_BLINDS + 1, (), np.int8),
        'boss_is_active':         spaces.Box(0, 1, (), np.int8),

        # ── Shop fields ──────────────────────────────────────
        'shop_item_types':        spaces.Box(0, 10, (MAX_SHOP_ITEMS,), np.int8),
        'shop_item_ids':          spaces.Box(0, 300, (MAX_SHOP_ITEMS,), np.int16),
        'shop_costs':             spaces.Box(0, 5000, (MAX_SHOP_ITEMS,), np.int16),
        'shop_is_empty':          spaces.MultiBinary(MAX_SHOP_ITEMS),
        'reroll_cost':            spaces.Box(0, 1000, (), np.int16),
    })


def build_action_space() -> spaces.Discrete:
    """Flat Discrete(60) action space matching the WrapperAction layout.

    Returns:
        gymnasium.spaces.Discrete(60).
    """
    return spaces.Discrete(ACTION_SPACE_SIZE)
