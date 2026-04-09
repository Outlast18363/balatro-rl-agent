"""Centralised enumerations and helper utilities for Balatro Gym
================================================================

This module replaces scattered *magic integers* with explicit `IntEnum`
classes. Import these enums everywhere instead of raw numbers to gain:

1. **Safety** – your IDE & `mypy` will scream when you pass the wrong kind of
   integer.
2. **Readability** – `Action.PLAY_HAND` is self‑explanatory; `0` is not.
3. **Refactor‑friendliness** – change a value here, *all* call‑sites update.

Usage (env excerpt)
-------------------
```python
from balatro_gym.constants import (
    Action,
    Phase,
    get_select_card_slot,
)

if Phase(self.state.phase) is Phase.PLAY:
    if Action(action) is Action.PLAY_HAND:
        ...
    else:
        card_idx = get_select_card_slot(action)
        if card_idx is not None:
            ...
```

Tip: wrap the integer `action` into `Action(action)` once at the top of
`step()` and pattern‑match from there.
"""

from __future__ import annotations
from enum import IntEnum, unique


MAX_HAND_SIZE = 10


@unique
class Phase(IntEnum):
    """High‑level game phases."""
    PLAY = 0
    SHOP = 1
    BLIND_SELECT = 2
    PACK_OPEN = 3


@unique
class Action(IntEnum):
    """Flat action space (0‑59) with base offsets for action families.

    Most parameterised action groups still occupy contiguous ranges. Select-card
    actions are the exception: slots 0-7 use historical IDs 2-9, while slots
    8-9 use spare IDs 56-57 so later action families keep their existing values.
    """
    # === Play‑phase basics ===
    PLAY_HAND: int = 0
    DISCARD: int = 1
    
    # --- Card selection ---
    SELECT_CARD_BASE: int = 2
    SELECT_CARD_SLOT_8: int = 56
    SELECT_CARD_SLOT_9: int = 57
    
    # --- Consumables (5 slots) ---
    USE_CONSUMABLE_BASE: int = 10
    # USE_CONSUMABLE_COUNT = 5  # → 10‑14
    
    # === Shop‑phase ===
    SHOP_BUY_BASE: int = 20  # 10 item slots → 20‑29
    # SHOP_BUY_COUNT = 10
    SHOP_REROLL: int = 30
    SHOP_END: int = 31
    
    # Selling
    SELL_JOKER_BASE: int = 32  # 5 joker slots → 32‑36
    # SELL_JOKER_COUNT = 5
    SELL_CONSUMABLE_BASE: int = 37  # 5 consumable slots → 37‑41
    # SELL_CONSUMABLE_COUNT = 5
    
    # === Blind selection ===
    SELECT_BLIND_BASE: int = 45  # small/big/boss → 45‑47
    # SELECT_BLIND_COUNT = 3
    SKIP_BLIND: int = 48
    
    # === Pack opening ===
    SELECT_FROM_PACK_BASE: int = 50  # 5 choices → 50‑54
    # SELECT_FROM_PACK_COUNT = 5
    SKIP_PACK: int = 55
    
    # === Meta ===
    # ACTION_SPACE_SIZE = 60

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def offset(self, index: int) -> int:  # noqa: D401 – simple helper
        """Return the concrete *integer* ID for an indexed variant.
        
        Example::
            Action.SHOP_BUY_BASE.offset(3)  # → 23
        """
        return int(self) + index
    
    @classmethod
    def from_offset(cls, base: "Action", id_: int) -> int:
        """Reverse‑map: given a *concrete* action ID, return its **index**.
        
        Example::
            idx = Action.from_offset(Action.SHOP_BUY_BASE, action_id)
        """
        return id_ - int(base)


SELECT_CARD_ACTION_IDS = (
    int(Action.SELECT_CARD_BASE),
    int(Action.SELECT_CARD_BASE) + 1,
    int(Action.SELECT_CARD_BASE) + 2,
    int(Action.SELECT_CARD_BASE) + 3,
    int(Action.SELECT_CARD_BASE) + 4,
    int(Action.SELECT_CARD_BASE) + 5,
    int(Action.SELECT_CARD_BASE) + 6,
    int(Action.SELECT_CARD_BASE) + 7,
    int(Action.SELECT_CARD_SLOT_8),
    int(Action.SELECT_CARD_SLOT_9),
)
SELECT_CARD_ACTION_TO_SLOT = {
    action_id: slot for slot, action_id in enumerate(SELECT_CARD_ACTION_IDS)
}


def get_select_card_action(slot: int) -> int:
    """Return the concrete select-card action ID for a hand slot.

    Args:
        slot: Zero-based hand slot index in ``[0, MAX_HAND_SIZE)``.

    Returns:
        The flat discrete action ID that toggles that slot.
    """
    return SELECT_CARD_ACTION_IDS[slot]


def get_select_card_slot(action_id: int) -> int | None:
    """Return the hand slot selected by an action ID.

    Args:
        action_id: Flat discrete action ID from the environment.

    Returns:
        The zero-based hand slot index if ``action_id`` is a select-card action,
        otherwise ``None``.
    """
    return SELECT_CARD_ACTION_TO_SLOT.get(action_id)


# Action count constants - not part of the enum!
class ActionCounts:
    """Constants defining how many actions of each type exist."""
    SELECT_CARD_COUNT: int = MAX_HAND_SIZE
    USE_CONSUMABLE_COUNT: int = 5
    SHOP_BUY_COUNT: int = 10
    SELL_JOKER_COUNT: int = 5
    SELL_CONSUMABLE_COUNT: int = 5
    SELECT_BLIND_COUNT: int = 3
    SELECT_FROM_PACK_COUNT: int = 5
    ACTION_SPACE_SIZE: int = 60


# For backward compatibility, also expose as module-level constants
SELECT_CARD_COUNT = ActionCounts.SELECT_CARD_COUNT
USE_CONSUMABLE_COUNT = ActionCounts.USE_CONSUMABLE_COUNT
SHOP_BUY_COUNT = ActionCounts.SHOP_BUY_COUNT
SELL_JOKER_COUNT = ActionCounts.SELL_JOKER_COUNT
SELL_CONSUMABLE_COUNT = ActionCounts.SELL_CONSUMABLE_COUNT
SELECT_BLIND_COUNT = ActionCounts.SELECT_BLIND_COUNT
SELECT_FROM_PACK_COUNT = ActionCounts.SELECT_FROM_PACK_COUNT
ACTION_SPACE_SIZE = ActionCounts.ACTION_SPACE_SIZE

# Alternatively, if you want to keep them on the Action class but not as enum members:
Action.SELECT_CARD_COUNT = SELECT_CARD_COUNT
Action.SELECT_CARD_ACTION_IDS = SELECT_CARD_ACTION_IDS
Action.SELECT_CARD_ACTION_TO_SLOT = SELECT_CARD_ACTION_TO_SLOT
Action.MAX_HAND_SIZE = MAX_HAND_SIZE
Action.USE_CONSUMABLE_COUNT = USE_CONSUMABLE_COUNT
Action.SHOP_BUY_COUNT = SHOP_BUY_COUNT
Action.SELL_JOKER_COUNT = SELL_JOKER_COUNT
Action.SELL_CONSUMABLE_COUNT = SELL_CONSUMABLE_COUNT
Action.SELECT_BLIND_COUNT = SELECT_BLIND_COUNT
Action.SELECT_FROM_PACK_COUNT = SELECT_FROM_PACK_COUNT
Action.ACTION_SPACE_SIZE = ACTION_SPACE_SIZE


__all__ = [
    "Phase",
    "Action",
    "MAX_HAND_SIZE",
    "ActionCounts",
    "SELECT_CARD_ACTION_IDS",
    "SELECT_CARD_ACTION_TO_SLOT",
    "get_select_card_action",
    "get_select_card_slot",
    # Export the constants too for convenience
    "SELECT_CARD_COUNT",
    "USE_CONSUMABLE_COUNT", 
    "SHOP_BUY_COUNT",
    "SELL_JOKER_COUNT",
    "SELL_CONSUMABLE_COUNT",
    "SELECT_BLIND_COUNT",
    "SELECT_FROM_PACK_COUNT",
    "ACTION_SPACE_SIZE",
]