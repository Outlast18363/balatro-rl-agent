"""cs590_env/test_wrapper.py - Tests for BalatroPhaseWrapper.

Covers:
  - Full phase cycle (TRANSITION → COMBAT → SHOP → TRANSITION)
  - Per-phase action masking
  - Wrapper-handled actions (swap joker, cross-phase sell/use)
  - Edge cases (invalid actions, empty slots, observation compliance)

Run with:  pytest cs590_env/test_wrapper.py -v
"""

import pytest
import numpy as np

from balatro_gym.balatro_env_2 import BalatroEnv
from balatro_gym.constants import Action
from balatro_gym.jokers import JokerInfo

from cs590_env.schema import (
    WrapperAction, GamePhase, ACTION_SPACE_SIZE,
    SWAP_JOKER_COUNT, SELL_JOKER_COUNT, SELL_CONSUMABLE_COUNT,
)
from cs590_env.wrapper import BalatroPhaseWrapper


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def env():
    """Wrapped BalatroEnv with fixed seed for determinism."""
    base = BalatroEnv(seed=42)
    return BalatroPhaseWrapper(base)


def _enter_combat(env):
    """Helper: select small blind to transition from TRANSITION → COMBAT."""
    obs, _, _, _, _ = env.step(int(WrapperAction.SELECT_BLIND_BASE))
    assert obs['phase'] == GamePhase.COMBAT, 'Expected COMBAT after selecting blind'
    return obs


def _beat_blind_and_enter_shop(env):
    """Helper: trivially beat current blind and land in SHOP.

    Sets chips_needed to 1, selects a card, and plays the hand.
    Returns the observation after entering SHOP.
    """
    env.env.state.chips_needed = 1
    env.step(int(WrapperAction.SELECT_CARD_BASE))        # select first card
    obs, _, term, _, _ = env.step(int(WrapperAction.PLAY_HAND))
    assert not term, 'Did not expect termination when trivially beating blind'
    assert obs['phase'] == GamePhase.SHOP, 'Expected SHOP after beating blind'
    return obs


# ─── Reset / basic observation ────────────────────────────────────────────────

class TestReset:
    def test_starts_in_transition(self, env):
        """After reset, phase should be TRANSITION (BLIND_SELECT)."""
        obs, info = env.reset()
        assert obs['phase'] == GamePhase.TRANSITION
        assert info['phase'] == 'TRANSITION'

    def test_obs_has_all_keys(self, env):
        """Observation must contain every key declared in observation_space."""
        obs, _ = env.reset()
        space_keys = set(env.observation_space.spaces.keys())
        obs_keys = set(obs.keys())
        assert space_keys == obs_keys, f'Missing: {space_keys - obs_keys}, Extra: {obs_keys - space_keys}'

    def test_obs_space_compliance(self, env):
        """Observation values must lie within the declared space bounds."""
        obs, _ = env.reset()
        assert env.observation_space.contains(obs), 'Observation out of space bounds'

    def test_action_mask_shape(self, env):
        """action_mask must be int8 array of length ACTION_SPACE_SIZE."""
        obs, _ = env.reset()
        assert obs['action_mask'].shape == (ACTION_SPACE_SIZE,)
        assert obs['action_mask'].dtype == np.int8


# ─── Transition (BLIND_SELECT) masking ────────────────────────────────────────

class TestTransitionMask:
    def test_only_current_blind_selectable(self, env):
        """In round 1, only SELECT_BLIND_BASE+0 (small) should be enabled."""
        obs, _ = env.reset()
        mask = obs['action_mask']
        assert mask[WrapperAction.SELECT_BLIND_BASE + 0] == 1   # small
        assert mask[WrapperAction.SELECT_BLIND_BASE + 1] == 0   # big
        assert mask[WrapperAction.SELECT_BLIND_BASE + 2] == 0   # boss

    def test_skip_blind_available_for_nonboss(self, env):
        """SKIP_BLIND should be enabled for small/big blinds."""
        obs, _ = env.reset()
        assert obs['action_mask'][WrapperAction.SKIP_BLIND] == 1

    def test_skip_blind_disabled_for_boss(self, env):
        """SKIP_BLIND must be disabled when facing boss (round 3)."""
        env.reset()
        env.env.state.round = 3
        mask = env._build_phase_mask()
        assert mask[WrapperAction.SKIP_BLIND] == 0
        assert mask[WrapperAction.SELECT_BLIND_BASE + 2] == 1

    def test_combat_actions_masked(self, env):
        """PLAY_HAND, DISCARD, SHOP_END should be disabled in transition."""
        obs, _ = env.reset()
        mask = obs['action_mask']
        assert mask[WrapperAction.PLAY_HAND] == 0
        assert mask[WrapperAction.DISCARD] == 0
        assert mask[WrapperAction.SHOP_END] == 0


# ─── Combat (PLAY) masking ────────────────────────────────────────────────────

class TestCombatMask:
    def test_card_selection_enabled(self, env):
        """Cards in hand should be selectable in combat."""
        env.reset()
        obs = _enter_combat(env)
        mask = obs['action_mask']
        assert mask[WrapperAction.SELECT_CARD_BASE] == 1

    def test_play_hand_disabled_without_selection(self, env):
        """PLAY_HAND should be off when no cards are selected."""
        env.reset()
        obs = _enter_combat(env)
        assert obs['action_mask'][WrapperAction.PLAY_HAND] == 0

    def test_play_hand_enabled_after_selection(self, env):
        """PLAY_HAND should turn on once at least one card is selected."""
        env.reset()
        _enter_combat(env)
        obs, _, _, _, _ = env.step(int(WrapperAction.SELECT_CARD_BASE))
        assert obs['action_mask'][WrapperAction.PLAY_HAND] == 1

    def test_discard_disabled_when_more_than_five_cards_selected(self, env):
        """DISCARD should be masked off when more than five cards are selected."""
        env.reset()
        _enter_combat(env)

        obs = None
        for i in range(6):
            obs, _, _, _, _ = env.step(int(WrapperAction.SELECT_CARD_BASE + i))

        assert obs is not None
        assert obs['action_mask'][WrapperAction.PLAY_HAND] == 0
        assert obs['action_mask'][WrapperAction.DISCARD] == 0

    def test_shop_actions_masked(self, env):
        """SHOP_END, SHOP_REROLL, SHOP_BUY should be disabled in combat."""
        env.reset()
        obs = _enter_combat(env)
        mask = obs['action_mask']
        assert mask[WrapperAction.SHOP_END] == 0
        assert mask[WrapperAction.SHOP_REROLL] == 0
        assert mask[WrapperAction.SHOP_BUY_BASE] == 0

    def test_blind_select_masked(self, env):
        """SELECT_BLIND should be disabled in combat."""
        env.reset()
        obs = _enter_combat(env)
        mask = obs['action_mask']
        for i in range(3):
            assert mask[WrapperAction.SELECT_BLIND_BASE + i] == 0


# ─── Shop masking ─────────────────────────────────────────────────────────────

class TestShopMask:
    def test_shop_end_always_available(self, env):
        """SHOP_END must always be enabled in shop phase."""
        env.reset()
        _enter_combat(env)
        obs = _beat_blind_and_enter_shop(env)
        assert obs['action_mask'][WrapperAction.SHOP_END] == 1

    def test_combat_actions_masked_in_shop(self, env):
        """PLAY_HAND and card selection should be disabled in shop."""
        env.reset()
        _enter_combat(env)
        obs = _beat_blind_and_enter_shop(env)
        mask = obs['action_mask']
        assert mask[WrapperAction.PLAY_HAND] == 0
        assert mask[WrapperAction.SELECT_CARD_BASE] == 0


# ─── Swap joker (wrapper-handled) ────────────────────────────────────────────

class TestSwapJoker:
    def test_swap_reorders_jokers(self, env):
        """Swapping slot 0 ↔ 1 should reverse two jokers."""
        env.reset()
        env.env.state.jokers = [
            JokerInfo(1, 'Joker', 2, '+4 Mult'),
            JokerInfo(2, 'Greedy Joker', 5, '♦ +3 Mult'),
        ]
        obs, reward, _, _, info = env.step(int(WrapperAction.SWAP_JOKER_BASE))
        assert reward == 0.0
        assert env.env.state.jokers[0].id == 2
        assert env.env.state.jokers[1].id == 1
        assert 'swapped_jokers' in info

    def test_swap_mask_requires_adjacent_pair(self, env):
        """SWAP_JOKER should only be masked on when both slots occupied."""
        env.reset()
        env.env.state.jokers = [JokerInfo(1, 'Joker', 2, '+4 Mult')]
        mask = env._build_phase_mask()
        for i in range(SWAP_JOKER_COUNT):
            assert mask[WrapperAction.SWAP_JOKER_BASE + i] == 0

    def test_swap_mask_with_three_jokers(self, env):
        """With 3 jokers, slots 0-1 and 1-2 can swap, but not 2-3."""
        env.reset()
        env.env.state.jokers = [
            JokerInfo(1, 'Joker', 2, ''),
            JokerInfo(2, 'Greedy Joker', 5, ''),
            JokerInfo(3, 'Lusty Joker', 5, ''),
        ]
        mask = env._build_phase_mask()
        assert mask[WrapperAction.SWAP_JOKER_BASE + 0] == 1
        assert mask[WrapperAction.SWAP_JOKER_BASE + 1] == 1
        assert mask[WrapperAction.SWAP_JOKER_BASE + 2] == 0


# ─── Cross-phase sell/use (wrapper-handled) ──────────────────────────────────

class TestCrossPhaseSell:
    def test_sell_joker_in_combat(self, env):
        """Selling a joker during combat should work via wrapper handler."""
        env.reset()
        _enter_combat(env)
        env.env.state.jokers = [JokerInfo(1, 'Joker', 2, '+4 Mult')]
        initial_money = env.env.state.money

        obs, reward, _, _, info = env.step(int(WrapperAction.SELL_JOKER_BASE))
        assert len(env.env.state.jokers) == 0
        assert env.env.state.money > initial_money
        assert 'sold_joker' in info
        assert reward > 0

    def test_sell_consumable_in_combat(self, env):
        """Selling a consumable during combat should work via wrapper handler."""
        env.reset()
        _enter_combat(env)
        env.env.state.consumables = ['Mercury']
        initial_money = env.env.state.money

        obs, reward, _, _, info = env.step(int(WrapperAction.SELL_CONSUMABLE_BASE))
        assert len(env.env.state.consumables) == 0
        assert env.env.state.money == initial_money + 4   # planet → sell for 4
        assert 'sold_consumable' in info

    def test_sell_joker_in_transition(self, env):
        """Selling a joker during transition should also work."""
        env.reset()
        env.env.state.jokers = [JokerInfo(5, 'Gluttonous Joker', 5, '♣ +3 Mult')]
        initial_money = env.env.state.money

        obs, reward, _, _, info = env.step(int(WrapperAction.SELL_JOKER_BASE))
        assert len(env.env.state.jokers) == 0
        assert env.env.state.money > initial_money

    def test_sell_mask_empty_slots(self, env):
        """Sell actions should be masked off when no items to sell."""
        env.reset()
        env.env.state.jokers = []
        env.env.state.consumables = []
        mask = env._build_phase_mask()
        for i in range(SELL_JOKER_COUNT):
            assert mask[WrapperAction.SELL_JOKER_BASE + i] == 0
        for i in range(SELL_CONSUMABLE_COUNT):
            assert mask[WrapperAction.SELL_CONSUMABLE_BASE + i] == 0


# ─── Invalid actions ──────────────────────────────────────────────────────────

class TestInvalidActions:
    def test_masked_action_returns_penalty(self, env):
        """An invalid (masked) action should give -1 reward and error info."""
        env.reset()
        obs, reward, terminated, truncated, info = env.step(int(WrapperAction.PLAY_HAND))
        assert reward == -1.0
        assert 'error' in info
        assert not terminated
        assert not truncated

    def test_shop_action_in_transition_invalid(self, env):
        """SHOP_END during transition should be invalid."""
        env.reset()
        obs, reward, _, _, info = env.step(int(WrapperAction.SHOP_END))
        assert reward == -1.0
        assert 'error' in info

    def test_discard_with_six_selected_cards_invalid(self, env):
        """Discarding six selected cards should be rejected by wrapper and base env."""
        env.reset()
        _enter_combat(env)

        for i in range(6):
            env.step(int(WrapperAction.SELECT_CARD_BASE + i))

        _, wrapper_reward, wrapper_terminated, wrapper_truncated, wrapper_info = env.step(
            int(WrapperAction.DISCARD)
        )
        assert wrapper_reward == -1.0
        assert 'error' in wrapper_info
        assert not wrapper_terminated
        assert not wrapper_truncated

        selected_before = list(env.env.state.selected_cards)
        discards_before = env.env.state.discards_left
        _, base_reward, base_terminated, base_truncated, base_info = env.env.step(int(Action.DISCARD))
        assert base_reward == -1.0
        assert base_info['error'] == 'Invalid action'
        assert not base_terminated
        assert not base_truncated
        assert env.env.state.selected_cards == selected_before
        assert env.env.state.discards_left == discards_before


# ─── Full phase cycle ─────────────────────────────────────────────────────────

class TestFullCycle:
    def test_transition_to_combat_to_shop_to_transition(self, env):
        """Complete phase cycle: TRANSITION → COMBAT → SHOP → TRANSITION."""
        # 1. Reset → TRANSITION
        obs, _ = env.reset()
        assert obs['phase'] == GamePhase.TRANSITION

        # 2. Select small blind → COMBAT
        obs = _enter_combat(env)
        assert obs['phase'] == GamePhase.COMBAT
        assert obs['hand_size'] > 0            # should have cards in hand
        assert obs['target_score'] > 0         # blind target should be set

        # 3. Trivially beat blind → SHOP
        obs = _beat_blind_and_enter_shop(env)
        assert obs['phase'] == GamePhase.SHOP

        # 4. End shop → TRANSITION
        obs, _, _, _, info = env.step(int(WrapperAction.SHOP_END))
        assert obs['phase'] == GamePhase.TRANSITION
        assert info.get('phase_changed', False)

    def test_obs_compliance_across_phases(self, env):
        """Observation must conform to observation_space in every phase."""
        obs, _ = env.reset()
        assert env.observation_space.contains(obs), 'TRANSITION obs out of space'

        obs = _enter_combat(env)
        assert env.observation_space.contains(obs), 'COMBAT obs out of space'

        obs = _beat_blind_and_enter_shop(env)
        assert env.observation_space.contains(obs), 'SHOP obs out of space'

        obs, _, _, _, _ = env.step(int(WrapperAction.SHOP_END))
        assert env.observation_space.contains(obs), 'Second TRANSITION obs out of space'

    def test_global_token_consistent(self, env):
        """Ante, round, and money should be present and reasonable in all phases."""
        obs, _ = env.reset()
        assert obs['ante'] >= 1
        assert obs['round'] >= 1
        assert obs['money'] >= 0

        obs = _enter_combat(env)
        assert obs['ante'] >= 1
        assert obs['money'] >= 0

    def test_deck_histogram_sums(self, env):
        """Rank histogram should sum to deck size in transition (full deck)."""
        obs, _ = env.reset()
        total_ranks = int(obs['deck_ranks'].sum())
        total_suits = int(obs['deck_suits'].sum())
        assert total_ranks == total_suits  # both count same cards
        assert total_ranks == 52           # standard deck

    def test_combat_deck_excludes_hand(self, env):
        """In combat, deck histogram should exclude cards in hand."""
        env.reset()
        obs = _enter_combat(env)
        deck_size = int(obs['deck_ranks'].sum())
        hand_size = int(obs['hand_size'])
        assert deck_size + hand_size == 52
