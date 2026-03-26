"""Shared schema for the architecture-facing Balatro wrapper.

This module centralizes the wrapper contract so the environment wrapper,
tests, and model code all agree on the same payload keys, slot counts, and
padding values.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np
from gymnasium import spaces


MAX_HAND_SIZE = 8
MAX_JOKER_SLOTS = 10
HAND_LEVEL_COUNT = 12
RANK_HISTOGRAM_SIZE = 13
SUIT_HISTOGRAM_SIZE = 4

PAD_CARD_ID = -1
PAD_JOKER_ID = 0
PAD_MODIFIER_ID = 0

HAND_TOKEN_KEYS = (
    "card_id",
    "is_empty",
    "is_face_down",
    "enhancement_id",
    "edition_id",
    "seal_id",
    "is_selected",
)
RUN_TOKEN_KEYS = (
    "money",
    "hands_left",
    "discards_left",
    "chips_needed",
    "round_chips_scored",
    "ante",
    "round",
    "phase",
)
DECK_TOKEN_KEYS = (
    "draw_pile_size",
    "remaining_rank_histogram",
    "remaining_suit_histogram",
)
BOSS_TOKEN_KEYS = (
    "boss_blind_active",
    "boss_blind_type",
)
JOKER_TOKEN_KEYS = (
    "joker_id",
    "is_empty",
    "is_disabled",
)
ACTION_MASK_KEYS = (
    "card_select_mask",
    "play_allowed",
    "discard_allowed",
    "selected_count",
)
ARCH_OBSERVATION_KEYS = (
    "hand_tokens",
    "run_token",
    "deck_token",
    "hand_levels",
    "boss_token",
    "joker_tokens",
    "action_masks",
)


class ArchExecuteAction(IntEnum):
    """Execution choices exposed by the wrapper action contract."""

    PLAY = 0
    DISCARD = 1


def build_action_space() -> spaces.Dict:
    """Return the structured wrapper action space.

    Returns:
        A `spaces.Dict` with a binary hand selection vector and a two-way
        execution choice (`PLAY` or `DISCARD`).
    """

    return spaces.Dict(
        {
            "selection": spaces.MultiBinary(MAX_HAND_SIZE),
            "execute": spaces.Discrete(len(ArchExecuteAction)),
        }
    )


def build_observation_space() -> spaces.Dict:
    """Return the nested observation space for wrapper payloads.

    Returns:
        A `spaces.Dict` mirroring the exact wrapper observation contract used
        by `BalatroArchWrapper`.
    """

    return spaces.Dict(
        {
            "hand_tokens": spaces.Dict(
                {
                    "card_id": spaces.Box(
                        low=PAD_CARD_ID,
                        high=51,
                        shape=(MAX_HAND_SIZE,),
                        dtype=np.int16,
                    ),
                    "is_empty": spaces.MultiBinary(MAX_HAND_SIZE),
                    "is_face_down": spaces.MultiBinary(MAX_HAND_SIZE),
                    "enhancement_id": spaces.Box(
                        low=0,
                        high=8,
                        shape=(MAX_HAND_SIZE,),
                        dtype=np.int8,
                    ),
                    "edition_id": spaces.Box(
                        low=0,
                        high=4,
                        shape=(MAX_HAND_SIZE,),
                        dtype=np.int8,
                    ),
                    "seal_id": spaces.Box(
                        low=0,
                        high=4,
                        shape=(MAX_HAND_SIZE,),
                        dtype=np.int8,
                    ),
                    "is_selected": spaces.MultiBinary(MAX_HAND_SIZE),
                }
            ),
            "run_token": spaces.Dict(
                {
                    "money": spaces.Box(low=-20, high=9999, shape=(), dtype=np.int32),
                    "hands_left": spaces.Box(low=0, high=12, shape=(), dtype=np.int8),
                    "discards_left": spaces.Box(low=0, high=10, shape=(), dtype=np.int8),
                    "chips_needed": spaces.Box(
                        low=0, high=10_000_000, shape=(), dtype=np.int32
                    ),
                    "round_chips_scored": spaces.Box(
                        low=0, high=10_000_000, shape=(), dtype=np.int32
                    ),
                    "ante": spaces.Box(low=1, high=1000, shape=(), dtype=np.int16),
                    "round": spaces.Box(low=1, high=3, shape=(), dtype=np.int8),
                    "phase": spaces.Box(low=0, high=3, shape=(), dtype=np.int8),
                }
            ),
            "deck_token": spaces.Dict(
                {
                    "draw_pile_size": spaces.Box(
                        low=0, high=52, shape=(), dtype=np.int16
                    ),
                    "remaining_rank_histogram": spaces.Box(
                        low=0,
                        high=4,
                        shape=(RANK_HISTOGRAM_SIZE,),
                        dtype=np.int8,
                    ),
                    "remaining_suit_histogram": spaces.Box(
                        low=0,
                        high=13,
                        shape=(SUIT_HISTOGRAM_SIZE,),
                        dtype=np.int8,
                    ),
                }
            ),
            "hand_levels": spaces.Box(
                low=0,
                high=99,
                shape=(HAND_LEVEL_COUNT,),
                dtype=np.int16,
            ),
            "boss_token": spaces.Dict(
                {
                    "boss_blind_active": spaces.MultiBinary(1),
                    "boss_blind_type": spaces.Box(low=0, high=64, shape=(), dtype=np.int8),
                }
            ),
            "joker_tokens": spaces.Dict(
                {
                    "joker_id": spaces.Box(
                        low=PAD_JOKER_ID,
                        high=255,
                        shape=(MAX_JOKER_SLOTS,),
                        dtype=np.int16,
                    ),
                    "is_empty": spaces.MultiBinary(MAX_JOKER_SLOTS),
                    "is_disabled": spaces.MultiBinary(MAX_JOKER_SLOTS),
                }
            ),
            "action_masks": spaces.Dict(
                {
                    "card_select_mask": spaces.MultiBinary(MAX_HAND_SIZE),
                    "play_allowed": spaces.MultiBinary(1),
                    "discard_allowed": spaces.MultiBinary(1),
                    "selected_count": spaces.Box(
                        low=0, high=MAX_HAND_SIZE, shape=(), dtype=np.int8
                    ),
                }
            ),
        }
    )
