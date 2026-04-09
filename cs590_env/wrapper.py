"""cs590_env/wrapper.py - BalatroPhaseWrapper: phase-aware Gymnasium wrapper.

Wraps BalatroEnv (balatro_gym) with:
- Three-phase observation dispatch (transition / combat / shop)
- Unified action masking including wrapper-handled actions (SWAP_JOKER, cross-phase sell/use)
- Phase-correct routing of actions to base env or direct state mutation
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import gymnasium as gym

from balatro_gym.balatro_env_2 import BalatroEnv, get_blind_chips
from balatro_gym.constants import Phase, Action
from balatro_gym.scoring_engine import HandType

from cs590_env.schema import (
    WrapperAction, GamePhase,
    ACTION_SPACE_SIZE, MAX_JOKER_DISPLAY, MAX_CONSUMABLE_DISPLAY,
    MAX_HAND_SIZE, MAX_SHOP_ITEMS, NUM_HAND_TYPES, NUM_RANKS, NUM_SUITS,
    NUM_VOUCHERS,
    SELECT_CARD_COUNT, USE_CONSUMABLE_COUNT, SWAP_JOKER_COUNT,
    SHOP_BUY_COUNT, SELL_JOKER_COUNT, SELL_CONSUMABLE_COUNT,
    SELECT_BLIND_COUNT,
    consumable_sell_value,
    build_observation_space, build_action_space,
)

# Base Phase → wrapper GamePhase
_PHASE_MAP = {
    Phase.PLAY: GamePhase.COMBAT,
    Phase.SHOP: GamePhase.SHOP,
    Phase.BLIND_SELECT: GamePhase.TRANSITION,
    Phase.PACK_OPEN: GamePhase.SHOP,
}

_BLIND_KEYS = ('small', 'big', 'boss')


class BalatroPhaseWrapper(gym.Wrapper):
    """Phase-aware wrapper over BalatroEnv.

    Provides phase-specific observations, unified action masking (including
    SWAP_JOKER 15-18), and routes wrapper-handled actions (swap / sell / use
    in non-native phases) via direct state mutation.

    Args:
        env: BalatroEnv instance to wrap.
    """

    def __init__(self, env: BalatroEnv):
        super().__init__(env)
        self.observation_space = build_observation_space()
        self.action_space = build_action_space()

    # ─── Gymnasium API ────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None) -> Tuple[dict, dict]:
        """Reset the base env and return a wrapped observation.

        Returns:
            (obs, info) tuple where obs is a phase-specific observation dict.
        """
        _, base_info = self.env.reset(seed=seed, options=options)
        self._auto_skip_pack_open()
        obs = self._get_phase_observation()
        info = {
            'phase': self._game_phase.name,
            'phase_changed': True,
            'previous_phase': None,
        }
        info.update(base_info)
        return obs, info

    def step(self, action: int) -> Tuple[dict, float, bool, bool, dict]:
        """Execute *action* with phase-aware routing.

        1. Validate against current phase mask.
        2. If wrapper-handled (swap / sell / use in non-native phase), mutate
           state directly.
        3. Otherwise forward to base env.

        Args:
            action: flat action ID in [0, 60).

        Returns:
            (obs, reward, terminated, truncated, info) tuple.
        """
        prev_phase = self._game_phase
        mask = self._build_phase_mask()

        if mask[action] == 0:
            obs = self._get_phase_observation()
            return obs, -1.0, False, False, {
                'error': f'Action {action} invalid in {prev_phase.name}',
                'phase': prev_phase.name,
                'translated_action': -1,
            }

        if self._is_wrapper_action(action):
            reward, info = self._handle_wrapper_action(action)
            terminated = truncated = False
            info['translated_action'] = -1
        else:
            _, reward, terminated, truncated, info = self.env.step(action)
            info['translated_action'] = action
            if not terminated:
                self._auto_skip_pack_open()

        cur_phase = self._game_phase
        info['phase'] = cur_phase.name
        info['phase_changed'] = cur_phase != prev_phase
        if info['phase_changed']:
            info['previous_phase'] = prev_phase.name

        obs = self._get_phase_observation()
        return obs, reward, terminated, truncated, info

    # ─── Phase helpers ────────────────────────────────────────────────────

    @property
    def _game_phase(self) -> GamePhase:
        """Current wrapper-level phase derived from base env state."""
        return _PHASE_MAP.get(self.env.state.phase, GamePhase.SHOP)

    @property
    def _state(self):
        """Shortcut to the base env's UnifiedGameState."""
        return self.env.state

    def _auto_skip_pack_open(self):
        """Silently advance past PACK_OPEN so it never surfaces to the agent."""
        while self._state.phase == Phase.PACK_OPEN:
            self.env.step(int(Action.SKIP_PACK))

    # ─── Observation builders ─────────────────────────────────────────────

    def _get_phase_observation(self) -> dict:
        """Dispatch to the phase-specific observation builder.

        Returns:
            Dict observation conforming to ``self.observation_space``.
        """
        phase = self._game_phase
        if phase == GamePhase.TRANSITION:
            return self._build_transition_obs()
        elif phase == GamePhase.COMBAT:
            return self._build_combat_obs()
        return self._build_shop_obs()

    def _build_global_token(self) -> dict:
        """Build observation fields common to all phases.

        Returns:
            Dict with global keys (ante, round, phase, money, jokers,
            consumables, hand_levels, action_mask, vouchers).
        """
        s = self._state

        # Joker tokens — pad to MAX_JOKER_DISPLAY
        joker_ids = np.zeros(MAX_JOKER_DISPLAY, dtype=np.int16)
        joker_sell = np.zeros(MAX_JOKER_DISPLAY, dtype=np.int16)
        joker_empty = np.ones(MAX_JOKER_DISPLAY, dtype=np.int8)
        for i, j in enumerate(s.jokers[:MAX_JOKER_DISPLAY]):
            joker_ids[i] = j.id
            joker_sell[i] = max(3, j.base_cost // 2)
            joker_empty[i] = 0

        # Consumable tokens — pad to MAX_CONSUMABLE_DISPLAY
        cons_ids = np.array(self.env._get_consumable_ids(), dtype=np.int16)
        cons_sell = np.zeros(MAX_CONSUMABLE_DISPLAY, dtype=np.int8)
        cons_empty = np.ones(MAX_CONSUMABLE_DISPLAY, dtype=np.int8)
        for i, name in enumerate(s.consumables[:MAX_CONSUMABLE_DISPLAY]):
            cons_sell[i] = consumable_sell_value(name)
            cons_empty[i] = 0

        # Hand levels: [level, chip, mult] per HandType at the current engine level
        hand_levels = np.zeros((NUM_HAND_TYPES, 3), dtype=np.int16)
        for ht in HandType:
            if ht.value < NUM_HAND_TYPES:
                level = self.env.engine.get_hand_level(ht)
                chip, mult = self.env.engine.get_hand_chips_mult(ht)
                hand_levels[ht.value] = [level, chip, mult]

        boss_id = s.next_boss_blind.value if s.next_boss_blind else 0

        return {
            'ante':                  np.int16(s.ante),
            'round':                 np.int8(s.round),
            'phase':                 np.int8(self._game_phase),
            'money':                 np.int32(s.money),
            'next_boss_blind_id':    np.int8(boss_id),
            'joker_ids':             joker_ids,
            'joker_sell_values':     joker_sell,
            'joker_is_empty':        joker_empty,
            'consumable_ids':        cons_ids,
            'consumable_sell_values': cons_sell,
            'consumable_is_empty':   cons_empty,
            'vouchers_owned':        np.zeros(NUM_VOUCHERS, dtype=np.int8),
            'hand_levels':           hand_levels,
            'action_mask':           self._build_phase_mask(),
        }

    def _build_deck_histogram(self, exclude_hand: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Compute rank and suit histograms of the deck.

        Args:
            exclude_hand: if True, exclude cards currently in hand (draw-pile only).

        Returns:
            (rank_counts[13], suit_counts[4]) int8 arrays.
        """
        ranks = np.zeros(NUM_RANKS, dtype=np.int8)
        suits = np.zeros(NUM_SUITS, dtype=np.int8)
        hand_set = set(self._state.hand_indexes) if exclude_hand else set()
        for i, card in enumerate(self._state.deck):
            if i not in hand_set:
                ranks[card.rank.value - 2] += 1   # Rank.TWO=2 → idx 0
                suits[card.suit.value] += 1
        return ranks, suits

    def _phase_zeros(self) -> dict:
        """Return zeroed-out phase-specific fields.

        Called by each obs builder to initialise all phase-specific keys before
        overwriting the relevant subset.

        Returns:
            Dict with every phase-specific key set to zeros / -1.
        """
        return {
            # Transition
            'blind_type':             np.int8(0),
            'target_score':           np.int32(0),
            'blind_reward':           np.int32(0),
            # Combat
            'hand_card_ids':          np.full(MAX_HAND_SIZE, -1, dtype=np.int8),
            'hand_card_enhancements': np.zeros(MAX_HAND_SIZE, dtype=np.int8),
            'hand_card_editions':     np.zeros(MAX_HAND_SIZE, dtype=np.int8),
            'hand_card_seals':        np.zeros(MAX_HAND_SIZE, dtype=np.int8),
            'hand_is_face_down':      np.zeros(MAX_HAND_SIZE, dtype=np.int8),
            'hand_is_selected':       np.zeros(MAX_HAND_SIZE, dtype=np.int8),
            'hand_is_debuffed':       np.zeros(MAX_HAND_SIZE, dtype=np.int8),
            'current_score':          np.int32(0),
            'hand_size':              np.int8(0),
            'hands_remaining':        np.int8(0),
            'discards_remaining':     np.int8(0),
            'hands_played_round':     np.int32(0),
            'boss_id':                np.int8(0),
            'boss_is_active':         np.int8(0),
            # Shop
            'shop_item_types':        np.zeros(MAX_SHOP_ITEMS, dtype=np.int8),
            'shop_item_ids':          np.zeros(MAX_SHOP_ITEMS, dtype=np.int16),
            'shop_costs':             np.zeros(MAX_SHOP_ITEMS, dtype=np.int16),
            'shop_is_empty':          np.ones(MAX_SHOP_ITEMS, dtype=np.int8),
            'reroll_cost':            np.int16(0),
            # Deck
            'deck_ranks':             np.zeros(NUM_RANKS, dtype=np.int8),
            'deck_suits':             np.zeros(NUM_SUITS, dtype=np.int8),
        }

    # ── Per-phase builders ────────────────────────────────────────────────

    def _build_transition_obs(self) -> dict:
        """Build observation for the BLIND_SELECT (transition) phase.

        Includes global token plus blind-specific fields and a full-deck histogram.

        Returns:
            Complete observation dict.
        """
        obs = self._build_global_token()
        obs.update(self._phase_zeros())

        s = self._state
        blind_key = _BLIND_KEYS[s.round - 1]
        target = get_blind_chips(s.ante, blind_key)

        # Estimated money reward from _advance_round after beating this blind
        next_round = 1 if s.round == 3 else s.round + 1
        blind_reward = 25 * next_round + (10 if next_round == 3 else 0)

        obs['blind_type']   = np.int8(s.round)
        obs['target_score'] = np.int32(target)
        obs['blind_reward'] = np.int32(blind_reward)

        ranks, suits = self._build_deck_histogram(exclude_hand=False)
        obs['deck_ranks'] = ranks
        obs['deck_suits'] = suits
        return obs

    def _build_combat_obs(self) -> dict:
        """Build observation for the PLAY (combat) phase.

        Includes global token, per-card hand tokens, scoring state, boss info,
        and a draw-pile histogram (deck minus hand).

        Returns:
            Complete observation dict.
        """
        obs = self._build_global_token()
        obs.update(self._phase_zeros())
        s = self._state

        # Hand card tokens
        card_ids = np.full(MAX_HAND_SIZE, -1, dtype=np.int8)
        card_enh = np.zeros(MAX_HAND_SIZE, dtype=np.int8)
        card_ed  = np.zeros(MAX_HAND_SIZE, dtype=np.int8)
        card_seal = np.zeros(MAX_HAND_SIZE, dtype=np.int8)
        for i, idx in enumerate(s.hand_indexes[:MAX_HAND_SIZE]):
            if idx < len(s.deck):
                card_ids[i] = int(s.deck[idx])
                cs = s.card_states.get(idx)
                if cs:
                    card_enh[i] = int(cs.enhancement)
                    card_ed[i]  = int(cs.edition)
                    card_seal[i] = int(cs.seal)

        face_down = np.zeros(MAX_HAND_SIZE, dtype=np.int8)
        selected  = np.zeros(MAX_HAND_SIZE, dtype=np.int8)
        for i in range(MAX_HAND_SIZE):
            if i in s.face_down_cards:
                face_down[i] = 1
            if i in s.selected_cards:
                selected[i] = 1

        obs['hand_card_ids']          = card_ids
        obs['hand_card_enhancements'] = card_enh
        obs['hand_card_editions']     = card_ed
        obs['hand_card_seals']        = card_seal
        obs['hand_is_face_down']      = face_down
        obs['hand_is_selected']       = selected
        # hand_is_debuffed: placeholder zeros – needs boss-blind derivation
        obs['current_score']          = np.int32(s.round_chips_scored)
        obs['target_score']           = np.int32(s.chips_needed)
        obs['hand_size']              = np.int8(len(s.hand_indexes))
        obs['hands_remaining']        = np.int8(s.hands_left)
        obs['discards_remaining']     = np.int8(s.discards_left)
        obs['hands_played_round']     = np.int32(s.hands_played_ante)
        obs['boss_id']                = np.int8(
            s.active_boss_blind.value if s.active_boss_blind else 0)
        obs['boss_is_active']         = np.int8(1 if s.boss_blind_active else 0)

        ranks, suits = self._build_deck_histogram(exclude_hand=True)
        obs['deck_ranks'] = ranks
        obs['deck_suits'] = suits
        return obs

    def _build_shop_obs(self) -> dict:
        """Build observation for the SHOP phase.

        Includes global token, shop inventory tokens, reroll cost, and a
        full-deck histogram.

        Returns:
            Complete observation dict.
        """
        obs = self._build_global_token()
        obs.update(self._phase_zeros())
        s = self._state

        item_types = np.zeros(MAX_SHOP_ITEMS, dtype=np.int8)
        item_ids   = np.zeros(MAX_SHOP_ITEMS, dtype=np.int16)
        costs      = np.zeros(MAX_SHOP_ITEMS, dtype=np.int16)
        empty      = np.ones(MAX_SHOP_ITEMS, dtype=np.int8)

        if hasattr(self.env, 'shop') and self.env.shop:
            for i, item in enumerate(self.env.shop.inventory[:MAX_SHOP_ITEMS]):
                it = item.item_type
                item_types[i] = it.value if hasattr(it, 'value') else int(it)
                item_ids[i]   = getattr(item, 'item_id', 0)
                costs[i]      = item.cost
                empty[i]      = 0

        obs['shop_item_types'] = item_types
        obs['shop_item_ids']   = item_ids
        obs['shop_costs']      = costs
        obs['shop_is_empty']   = empty
        obs['reroll_cost']     = np.int16(s.shop_reroll_cost)

        ranks, suits = self._build_deck_histogram(exclude_hand=False)
        obs['deck_ranks'] = ranks
        obs['deck_suits'] = suits
        return obs

    # ─── Action masking ───────────────────────────────────────────────────

    def _build_phase_mask(self) -> np.ndarray:
        """Build the 60-element action validity mask for the current phase.

        Cross-phase actions (swap joker, sell, use consumable) are enabled in
        every phase where the prerequisite items exist.  Phase-native actions
        (play hand, shop buy, select blind, etc.) are enabled only in their
        respective phase.

        Returns:
            int8 ndarray of shape (60,), 1 = valid, 0 = invalid.
        """
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.int8)
        s = self._state
        phase = self._game_phase

        # ── Cross-phase: swap joker (slot i ↔ i+1 when both occupied) ──
        for i in range(SWAP_JOKER_COUNT):
            if i + 1 < len(s.jokers):
                mask[WrapperAction.SWAP_JOKER_BASE + i] = 1

        # ── Cross-phase: sell joker ──
        for i in range(min(SELL_JOKER_COUNT, len(s.jokers))):
            mask[WrapperAction.SELL_JOKER_BASE + i] = 1

        # ── Cross-phase: sell consumable ──
        for i in range(min(SELL_CONSUMABLE_COUNT, len(s.consumables))):
            mask[WrapperAction.SELL_CONSUMABLE_BASE + i] = 1

        # ── Cross-phase: use consumable ──
        for i in range(min(USE_CONSUMABLE_COUNT, len(s.consumables))):
            mask[WrapperAction.USE_CONSUMABLE_BASE + i] = 1

        # ── Phase-native actions ──
        if phase == GamePhase.TRANSITION:
            blind_idx = s.round - 1  # 0=small, 1=big, 2=boss
            if 0 <= blind_idx < SELECT_BLIND_COUNT:
                mask[WrapperAction.SELECT_BLIND_BASE + blind_idx] = 1
            if s.round < 3:  # cannot skip boss
                mask[WrapperAction.SKIP_BLIND] = 1

        elif phase == GamePhase.COMBAT:
            for i in range(min(SELECT_CARD_COUNT, len(s.hand_indexes))):
                mask[WrapperAction.SELECT_CARD_BASE + i] = 1
            n_sel = len(s.selected_cards)
            if 1 <= n_sel <= 5:
                mask[WrapperAction.PLAY_HAND] = 1
            if 1 <= n_sel <= 5 and s.discards_left > 0:
                mask[WrapperAction.DISCARD] = 1

        elif phase == GamePhase.SHOP:
            if hasattr(self.env, 'shop') and self.env.shop:
                for i, item in enumerate(self.env.shop.inventory[:SHOP_BUY_COUNT]):
                    if s.money >= item.cost:
                        mask[WrapperAction.SHOP_BUY_BASE + i] = 1
                if s.money >= s.shop_reroll_cost:
                    mask[WrapperAction.SHOP_REROLL] = 1
            mask[WrapperAction.SHOP_END] = 1

        return mask

    # ─── Wrapper-action routing ───────────────────────────────────────────

    def _is_wrapper_action(self, action: int) -> bool:
        """Return True if *action* should be handled by the wrapper, not the base env.

        Wrapper handles:
        - SWAP_JOKER (15-18): always (no base env equivalent).
        - SELL_JOKER (32-36): outside SHOP (base env handles it in SHOP).
        - SELL_CONSUMABLE (37-41): outside SHOP.
        - USE_CONSUMABLE (10-14): outside COMBAT (base env handles it in PLAY).

        Args:
            action: flat action ID.

        Returns:
            True if the wrapper should handle this action.
        """
        if WrapperAction.SWAP_JOKER_BASE <= action < WrapperAction.SWAP_JOKER_BASE + SWAP_JOKER_COUNT:
            return True
        phase = self._game_phase
        if WrapperAction.SELL_JOKER_BASE <= action < WrapperAction.SELL_JOKER_BASE + SELL_JOKER_COUNT:
            return phase != GamePhase.SHOP
        if WrapperAction.SELL_CONSUMABLE_BASE <= action < WrapperAction.SELL_CONSUMABLE_BASE + SELL_CONSUMABLE_COUNT:
            return phase != GamePhase.SHOP
        if WrapperAction.USE_CONSUMABLE_BASE <= action < WrapperAction.USE_CONSUMABLE_BASE + USE_CONSUMABLE_COUNT:
            return phase != GamePhase.COMBAT
        return False

    def _handle_wrapper_action(self, action: int) -> Tuple[float, dict]:
        """Execute a wrapper-handled action via direct state mutation.

        Args:
            action: validated action ID (already mask-checked).

        Returns:
            (reward, info) tuple.
        """
        if WrapperAction.SWAP_JOKER_BASE <= action < WrapperAction.SWAP_JOKER_BASE + SWAP_JOKER_COUNT:
            return self._handle_swap_joker(action - WrapperAction.SWAP_JOKER_BASE)
        if WrapperAction.SELL_JOKER_BASE <= action < WrapperAction.SELL_JOKER_BASE + SELL_JOKER_COUNT:
            return self._handle_sell_joker_nonshop(action - WrapperAction.SELL_JOKER_BASE)
        if WrapperAction.SELL_CONSUMABLE_BASE <= action < WrapperAction.SELL_CONSUMABLE_BASE + SELL_CONSUMABLE_COUNT:
            return self._handle_sell_consumable_nonshop(action - WrapperAction.SELL_CONSUMABLE_BASE)
        if WrapperAction.USE_CONSUMABLE_BASE <= action < WrapperAction.USE_CONSUMABLE_BASE + USE_CONSUMABLE_COUNT:
            return self._handle_use_consumable_nonplay(action - WrapperAction.USE_CONSUMABLE_BASE)
        return -1.0, {'error': 'Unrecognised wrapper action'}

    # ── Individual action handlers ────────────────────────────────────────

    def _handle_swap_joker(self, slot: int) -> Tuple[float, dict]:
        """Swap jokers at *slot* and *slot+1* in-place.

        Args:
            slot: lower index of the pair (0-3).

        Returns:
            (0.0 reward, info dict) – pure utility action.
        """
        jokers = self._state.jokers
        if slot + 1 < len(jokers):
            jokers[slot], jokers[slot + 1] = jokers[slot + 1], jokers[slot]
            return 0.0, {'swapped_jokers': (slot, slot + 1)}
        return -1.0, {'error': f'Cannot swap joker slot {slot}'}

    def _handle_sell_joker_nonshop(self, idx: int) -> Tuple[float, dict]:
        """Sell joker at *idx* outside the shop phase (direct state mutation).

        Mirrors the base env's sell-joker logic but works in any phase.

        Args:
            idx: joker slot index (0-4).

        Returns:
            (reward, info) with reward = sell_value / 5.
        """
        s = self._state
        if 0 <= idx < len(s.jokers):
            sold = s.jokers.pop(idx)
            sell_value = max(3, sold.base_cost // 2)
            s.money += sell_value
            s.jokers_sold += 1
            return sell_value / 5.0, {'sold_joker': sold.name, 'sell_value': sell_value}
        return -1.0, {'error': f'Invalid joker index {idx}'}

    def _handle_sell_consumable_nonshop(self, idx: int) -> Tuple[float, dict]:
        """Sell consumable at *idx* outside the shop phase.

        Args:
            idx: consumable slot index (0-4).

        Returns:
            (reward, info) with reward = sell_value / 5.
        """
        s = self._state
        if 0 <= idx < len(s.consumables):
            name = s.consumables.pop(idx)
            value = consumable_sell_value(name)
            s.money += value
            return value / 5.0, {'sold_consumable': name, 'sell_value': value}
        return -1.0, {'error': f'Invalid consumable index {idx}'}

    def _handle_use_consumable_nonplay(self, idx: int) -> Tuple[float, dict]:
        """Use consumable at *idx* outside the play phase.

        Delegates to the base env's ``_use_consumable`` which handles planet /
        tarot / spectral effects. Consumables requiring target cards will
        return an error when no hand cards are selected.

        Args:
            idx: consumable slot index (0-4).

        Returns:
            (reward, info) from the base env's consumable handler.
        """
        return self.env._use_consumable(idx)
