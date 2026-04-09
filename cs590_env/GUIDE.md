# BalatroPhaseWrapper Usage Guide

## Setup

```python
from balatro_gym.balatro_env_2 import BalatroEnv
from cs590_env import BalatroPhaseWrapper, WrapperAction, GamePhase

env = BalatroPhaseWrapper(BalatroEnv(seed=42))
obs, info = env.reset()
```

`obs` is a `dict` of numpy arrays/scalars. `info['phase']` is the phase name string (`'TRANSITION'`, `'COMBAT'`, or `'SHOP'`).

---

## Phase Flow

```
reset() ──► TRANSITION ──► COMBAT ──► SHOP ──► TRANSITION ──► ...
              (blind       (play      (buy/     (next
              select)       hands)     sell)     blind)
```

Each ante has 3 rounds (small → big → boss). Beating a blind sends you to SHOP. Ending the shop sends you back to TRANSITION. Failing a blind terminates the episode.

---

## Action Space — `Discrete(60)` with Sparse IDs

Every `step(action)` takes a single `int` in `[0, 60)`. Use `obs['action_mask']` (int8 array of shape `(60,)`, 1=valid) to filter legal actions.

The wrapper keeps the base env's sparse ID layout, so not every integer in `[0, 60)` is assigned to a named action. Currently used IDs are:

| IDs | Action | Available in |
|---------|-------------------------------------|---------------------------|
| 0 | `PLAY_HAND` | Combat |
| 1 | `DISCARD` | Combat |
| 2–9 | `SELECT_CARD_BASE + i` (i=0..7) | Combat |
| 10–14 | `USE_CONSUMABLE_BASE + i` (i=0..4) | All phases |
| 15–18 | `SWAP_JOKER_BASE + i` (swap i↔i+1) | All phases |
| 20–29 | `SHOP_BUY_BASE + i` (i=0..9) | Shop |
| 30 | `SHOP_REROLL` | Shop |
| 31 | `SHOP_END` | Shop |
| 32–36 | `SELL_JOKER_BASE + i` (i=0..4) | All phases |
| 37–41 | `SELL_CONSUMABLE_BASE + i` (i=0..4) | All phases |
| 45–47 | `SELECT_BLIND_BASE + i` (0=S,1=B,2=Boss) | Transition |
| 48 | `SKIP_BLIND` | Transition (not boss) |
| 50–54 | `SELECT_FROM_PACK_BASE + i` | Reserved for pack-open; never surfaced |
| 55 | `SKIP_PACK` | Reserved for pack-open; never surfaced |

Unused / unassigned IDs: `19`, `42–44`, `49`, `56–59`.

So the action space size is 60, but the wrapper currently names only 51 concrete IDs and only a subset of those are ever valid in any given state. In particular, pack-open actions never become valid because the wrapper auto-skips `PACK_OPEN`.

Use the enum for readability:
```python
action = int(WrapperAction.SELECT_BLIND_BASE)  # 45 — select small blind
obs, reward, terminated, truncated, info = env.step(action)
```

Sending a masked action returns `reward=-1.0` and `info['error']` with no state change.

---

## Observation Space — `Dict`

All keys are present in every step. Phase-irrelevant fields are zeroed. Check `obs['phase']` (int8: 0=TRANSITION, 1=COMBAT, 2=SHOP) to know which fields are populated.

### Global Token (all phases)

| Key | Shape | Dtype | Description |
|-------------------------|---------|---------|-------------|
| `ante` | `()` | int16 | Current ante (1+) |
| `round` | `()` | int8 | Current round (1=small, 2=big, 3=boss) |
| `phase` | `()` | int8 | 0=TRANSITION, 1=COMBAT, 2=SHOP |
| `money` | `()` | int32 | Current money |
| `next_boss_blind_id` | `()` | int8 | Pre-selected boss blind ID (1–28), 0 if none |
| `joker_ids` | `(10,)` | int16 | Joker IDs per slot, 0 = empty |
| `joker_sell_values` | `(10,)` | int16 | Sell value per slot, 0 = empty |
| `joker_is_empty` | `(10,)` | int8 | 1 = slot empty |
| `consumable_ids` | `(5,)` | int16 | Consumable IDs per slot, 0 = empty |
| `consumable_sell_values` | `(5,)` | int8 | 3 (tarot/spectral) or 4 (planet) |
| `consumable_is_empty` | `(5,)` | int8 | 1 = slot empty |
| `vouchers_owned` | `(16,)` | int8 | Binary flags (placeholder, all 0) |
| `hand_levels` | `(12,3)` | int16 | `[level, chip, mult]` per hand type |
| `action_mask` | `(60,)` | int8 | 1 = action legal this step |
| `deck_ranks` | `(13,)` | int8 | Cards per rank (idx 0=Two .. 12=Ace) |
| `deck_suits` | `(4,)` | int8 | Cards per suit (0=♣, 1=♦, 2=♥, 3=♠) |

For `hand_levels`, `chip` and `mult` are the level-scaled hand values returned by the env scoring engine for each hand type. They are not the final scored value after card effects, jokers, or boss modifiers.

### Transition Phase Fields

| Key | Shape | Dtype | Description |
|----------------|-------|-------|-------------|
| `blind_type` | `()` | int8 | Which blind is next (1=small, 2=big, 3=boss) |
| `target_score` | `()` | int32 | Chips required to beat this blind |
| `blind_reward` | `()` | int32 | Estimated money for beating this blind |

Deck histograms count the **current full deck state**. Early in a run this is usually `MAX_DECK_SIZE` cards (`52` in the current schema), but it can change if cards are added or removed. The histogram space in the wrapper schema is bounded by `MAX_DECK_SIZE`.

### Combat Phase Fields

| Key | Shape | Dtype | Description |
|--------------------------|-------|-------|-------------|
| `hand_card_ids` | `(8,)` | int8 | Card ID 0–51 (`(rank-2)*4 + suit`), -1 = empty |
| `hand_card_enhancements` | `(8,)` | int8 | Enhancement enum (0=none, 1=bonus, ..., 8=lucky) |
| `hand_card_editions` | `(8,)` | int8 | Edition enum (0=none, 1=foil, 2=holo, 3=poly, 4=neg) |
| `hand_card_seals` | `(8,)` | int8 | Seal enum (0=none, 1=gold, 2=red, 3=blue, 4=purple) |
| `hand_is_face_down` | `(8,)` | int8 | 1 = face-down (boss blind effect) |
| `hand_is_selected` | `(8,)` | int8 | 1 = currently selected for play/discard |
| `hand_is_debuffed` | `(8,)` | int8 | 1 = debuffed (placeholder, all 0) |
| `current_score` | `()` | int32 | Chips scored so far this round |
| `target_score` | `()` | int32 | Chips needed to beat the blind |
| `hand_size` | `()` | int8 | Number of cards currently in hand |
| `hands_remaining` | `()` | int8 | Hands left to play |
| `discards_remaining` | `()` | int8 | Discards left |
| `hands_played_round` | `()` | int32 | Hands played this ante |
| `boss_id` | `()` | int8 | Active boss blind ID (1–28), 0 = none |
| `boss_is_active` | `()` | int8 | 1 = boss blind is active |

Deck histograms count the **current draw pile** (current deck minus cards in hand).

### Shop Phase Fields

| Key | Shape | Dtype | Description |
|------------------|--------|-------|-------------|
| `shop_item_types` | `(10,)` | int8 | Item type enum per slot |
| `shop_item_ids` | `(10,)` | int16 | Item-specific ID per slot |
| `shop_costs` | `(10,)` | int16 | Cost in money per slot |
| `shop_is_empty` | `(10,)` | int8 | 1 = slot empty |
| `reroll_cost` | `()` | int16 | Cost to reroll the shop |

Deck histograms count the **current full deck state**.

---

## Info Dict

`step()` always returns `info` with:

| Key | Type | Description |
|--------------------|------|-------------|
| `phase` | str | `'TRANSITION'`, `'COMBAT'`, or `'SHOP'` |
| `translated_action` | int | Base-env action ID that was executed; `-1` for wrapper-handled or invalid actions |

`step()` additionally returns these keys on successful valid actions:

| Key | Type | Description |
|--------------------|------|-------------|
| `phase_changed` | bool | Whether this step caused a phase transition |
| `previous_phase` | str | Phase before transition (present only if the phase changed) |

Invalid masked actions return only:

| Key | Type | Description |
|--------------------|------|-------------|
| `phase` | str | Current phase name |
| `translated_action` | int | Always `-1` |
| `error` | str | Explanation of why the action was invalid |

Additional keys are forwarded from the base env (e.g. `sold_joker`, `boss_blind`, `skipped_blind`).

---

## Masking Rules Per Phase

**TRANSITION:** Only `SELECT_BLIND_BASE + (round-1)` is enabled (you must face blinds in order). `SKIP_BLIND` is enabled except for boss (round 3). Sell/swap/use consumable are available if items exist.

**COMBAT:** `SELECT_CARD_BASE + i` for each card in hand. `PLAY_HAND` requires 1–5 cards selected. `DISCARD` requires cards selected and discards > 0. Sell/swap/use consumable available.

**SHOP:** `SHOP_BUY_BASE + i` if `money >= cost`. `SHOP_REROLL` if `money >= reroll_cost`. `SHOP_END` always. Sell/swap/use consumable available.

---

## Minimal Training Loop

```python
from balatro_gym.balatro_env_2 import BalatroEnv
from cs590_env import BalatroPhaseWrapper, WrapperAction

env = BalatroPhaseWrapper(BalatroEnv(seed=0))
obs, info = env.reset()

for _ in range(1000):
    mask = obs['action_mask']
    valid_actions = mask.nonzero()[0]
    action = int(valid_actions[0])              # replace with your policy

    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()
```

## Typical Combat Sequence

```python
# 1. Select cards (toggle selection — call multiple times for multi-card hands)
obs, *_ = env.step(int(WrapperAction.SELECT_CARD_BASE + 0))  # select card 0
obs, *_ = env.step(int(WrapperAction.SELECT_CARD_BASE + 2))  # select card 2

# 2. Play the selected hand
obs, reward, terminated, truncated, info = env.step(int(WrapperAction.PLAY_HAND))
# reward reflects scoring; if current_score >= target_score, blind is beaten
# and phase transitions to SHOP
```
