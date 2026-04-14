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
from balatro_gym.boss_blinds import BOSS_BLINDS, BossBlindType, is_card_debuffed_by_blind
from balatro_gym.cards import Card, Rank, Suit
from balatro_gym.constants import Action
from balatro_gym.jokers import JokerInfo

from cs590_env.schema import (
    ACTION_SPACE_SIZE,
    GamePhase,
    MAX_DECK_SIZE,
    MAX_HAND_SIZE,
    SELL_CONSUMABLE_COUNT,
    SELL_JOKER_COUNT,
    SWAP_JOKER_COUNT,
    WrapperAction,
    get_wrapper_select_action,
)
from cs590_env.wrapper import BalatroPhaseWrapper, compute_combat_pass_reward


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
        """Cards in hand should be selectable across the full 10-slot mapping."""
        env.reset()
        obs = _enter_combat(env)
        mask = obs['action_mask']
        assert obs['hand_size'] == MAX_HAND_SIZE
        assert mask[get_wrapper_select_action(0)] == 1
        assert mask[get_wrapper_select_action(MAX_HAND_SIZE - 1)] == 1

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
            obs, _, _, _, _ = env.step(get_wrapper_select_action(i))

        assert obs is not None
        assert obs['action_mask'][WrapperAction.PLAY_HAND] == 0
        assert obs['action_mask'][WrapperAction.DISCARD] == 0

    def test_extra_select_actions_toggle_ninth_and_tenth_slots(self, env):
        """The spare action IDs should select combat slots 8 and 9."""
        env.reset()
        _enter_combat(env)

        ninth_action = get_wrapper_select_action(8)
        tenth_action = get_wrapper_select_action(9)

        obs, _, _, _, _ = env.step(ninth_action)
        assert obs['hand_is_selected'][8] == 1

        obs, _, _, _, _ = env.step(tenth_action)
        assert obs['hand_is_selected'][9] == 1
        assert obs['action_mask'][WrapperAction.PLAY_HAND] == 1

    def test_eight_card_states_pad_to_ten_and_mask_extra_select_actions(self, env):
        """Shorter hands should still expose 10 slots with the extras padded out."""
        env.reset()
        _enter_combat(env)
        env.env.state.hand_indexes = env.env.state.hand_indexes[: MAX_HAND_SIZE - 2]
        env.env.state.selected_cards = []
        env.env.state.face_down_cards = []

        obs = env._get_phase_observation()

        assert obs['hand_card_ids'].shape == (MAX_HAND_SIZE,)
        assert np.all(obs['hand_card_ids'][8:] == -1)
        assert obs['action_mask'][get_wrapper_select_action(7)] == 1
        assert obs['action_mask'][get_wrapper_select_action(8)] == 0
        assert obs['action_mask'][get_wrapper_select_action(9)] == 0

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


class TestBossBlindDebuffs:
    def test_suit_debuff_helper_matches_card_suit_enum(self):
        """Suit-based boss blinds should work with ``Card.suit`` enum values."""
        club_card = Card(rank=Rank.TWO, suit=Suit.CLUBS)
        spade_card = Card(rank=Rank.THREE, suit=Suit.SPADES)

        assert is_card_debuffed_by_blind(club_card, BossBlindType.THE_CLUB)
        assert not is_card_debuffed_by_blind(spade_card, BossBlindType.THE_CLUB)

    def test_combat_obs_marks_debuffed_hand_slots_for_active_boss(self, env):
        """Combat observations should expose the active boss-debuff mask."""
        env.reset()
        _enter_combat(env)

        first_idx = env.env.state.hand_indexes[0]
        second_idx = env.env.state.hand_indexes[1]
        env.env.state.deck[first_idx] = Card(rank=Rank.TWO, suit=Suit.CLUBS)
        env.env.state.deck[second_idx] = Card(rank=Rank.THREE, suit=Suit.SPADES)
        env.env.state.active_boss_blind = BossBlindType.THE_CLUB
        env.env.state.boss_blind_active = True
        env.env.boss_blind_manager.active_blind = BOSS_BLINDS[BossBlindType.THE_CLUB]
        env.env.boss_blind_manager.blind_state = {}

        obs = env._get_phase_observation()
        expected_mask = np.zeros(MAX_HAND_SIZE, dtype=np.int8)
        for slot, deck_idx in enumerate(env.env.state.hand_indexes[:MAX_HAND_SIZE]):
            if env.env.state.deck[deck_idx].suit == Suit.CLUBS:
                expected_mask[slot] = 1

        assert obs['boss_is_active'] == 1
        assert obs['boss_id'] == BossBlindType.THE_CLUB
        assert obs['hand_is_debuffed'][0] == 1
        assert obs['hand_is_debuffed'][1] == 0
        assert np.array_equal(obs['hand_is_debuffed'], expected_mask)


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


class TestCombatReward:
    def test_combat_pass_reward_helper_matches_formula(self):
        """Helper should implement 2 * (plays_remaining + log10(score)) * 1_pass."""
        reward = compute_combat_pass_reward(
            plays_remaining=3,
            score=1000,
            passed=True,
        )
        assert reward == pytest.approx(12.0)

    def test_combat_pass_reward_helper_zero_when_not_passed(self):
        """Failing to clear the blind should zero-out the pass indicator."""
        reward = compute_combat_pass_reward(
            plays_remaining=3,
            score=1000,
            passed=False,
        )
        assert reward == 0.0

    def test_combat_pass_reward_helper_stays_finite_for_large_scores(self):
        """Large late-game scores should still yield a finite log-scaled reward."""
        reward = compute_combat_pass_reward(
            plays_remaining=3,
            score=10**15,
            passed=True,
        )
        assert np.isfinite(reward)
        assert reward == pytest.approx(36.0)

    def test_play_hand_reward_uses_wrapper_pass_formula(self, env):
        """Winning hands should use the wrapper's explicit pass-reward formula."""
        env.reset()
        _enter_combat(env)
        env.env.state.chips_needed = 1
        env.step(int(WrapperAction.SELECT_CARD_BASE))

        hands_before = int(env.env.state.hands_left)
        obs, reward, terminated, truncated, info = env.step(int(WrapperAction.PLAY_HAND))

        expected_reward = compute_combat_pass_reward(
            plays_remaining=hands_before - 1,
            score=float(info['final_score']),
            passed=True,
        )
        assert obs['phase'] == GamePhase.SHOP
        assert not terminated
        assert not truncated
        assert info['beat_blind'] is True
        assert reward == pytest.approx(expected_reward)
        assert info['base_reward'] != pytest.approx(reward)
        assert info['wrapper_reward_formula'] == '2 * (plays_remaining + log10(score)) * 1_pass'

    def test_play_hand_reward_is_zero_when_blind_not_cleared(self, env):
        """Non-winning plays should return zero combat reward from the wrapper."""
        env.reset()
        _enter_combat(env)
        env.env.state.chips_needed = 10**9
        env.step(int(WrapperAction.SELECT_CARD_BASE))

        _, reward, terminated, truncated, info = env.step(int(WrapperAction.PLAY_HAND))

        assert reward == 0.0
        assert not info.get('beat_blind', False)
        assert not terminated
        assert not truncated
        assert info['wrapper_reward_breakdown']['pass_indicator'] == 0

    def test_discard_reward_is_neutralized_in_wrapper(self, env):
        """Discard actions should not inherit the base env's positive shaping."""
        env.reset()
        _enter_combat(env)
        env.step(int(WrapperAction.SELECT_CARD_BASE))

        obs, reward, terminated, truncated, info = env.step(int(WrapperAction.DISCARD))

        assert obs['phase'] == GamePhase.COMBAT
        assert reward == 0.0
        assert info['base_reward'] > reward
        assert info['wrapper_reward_breakdown']['combat_reward'] == 0.0
        assert not terminated
        assert not truncated


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
            env.step(get_wrapper_select_action(i))

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

    def test_transition_target_score_handles_high_antes(self, env):
        """Transition obs should preserve very large blind targets without overflow."""
        env.reset()
        env.env.state.ante = 50

        obs = env._get_phase_observation()

        assert obs['phase'] == GamePhase.TRANSITION
        assert obs['target_score'] > np.int64(2_147_483_647)
        assert obs['target_score'].dtype == np.int64
        assert env.observation_space.contains(obs), 'High-ante TRANSITION obs out of space'

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
        assert total_ranks == len(env.env.state.deck)

    def test_combat_deck_excludes_hand(self, env):
        """In combat, deck histogram should exclude cards in hand."""
        env.reset()
        obs = _enter_combat(env)
        deck_size = int(obs['deck_ranks'].sum())
        hand_size = int(obs['hand_size'])
        assert deck_size + hand_size == len(env.env.state.deck)

    def test_deck_arrays_are_padded_at_reset(self, env):
        """Reset obs should expose padded deck arrays with masked empty slots."""
        obs, _ = env.reset()
        filled_slots = obs['deck_card_ids'] >= 0
        empty_slots = ~filled_slots
        deck_len = len(env.env.state.deck)

        assert obs['deck_card_ids'].shape == (MAX_DECK_SIZE,)
        assert int(np.count_nonzero(filled_slots)) == deck_len
        assert int(np.count_nonzero(empty_slots)) == MAX_DECK_SIZE - deck_len
        assert np.all(obs['deck_card_enhancements'][filled_slots] == 0)
        assert np.all(obs['deck_card_editions'][filled_slots] == 0)
        assert np.all(obs['deck_card_seals'][filled_slots] == 0)
        assert np.all(obs['deck_card_enhancements'][empty_slots] == -1)
        assert np.all(obs['deck_card_editions'][empty_slots] == -1)
        assert np.all(obs['deck_card_seals'][empty_slots] == -1)

    def test_combat_deck_arrays_exclude_hand(self, env):
        """Combat deck arrays should describe the draw pile, not cards in hand."""
        env.reset()
        obs = _enter_combat(env)
        filled_slots = obs['deck_card_ids'] >= 0
        draw_pile_size = len(env.env.state.deck) - int(obs['hand_size'])

        assert int(np.count_nonzero(filled_slots)) == draw_pile_size
        assert np.all(obs['deck_card_enhancements'][~filled_slots] == -1)
        assert np.all(obs['deck_card_editions'][~filled_slots] == -1)
        assert np.all(obs['deck_card_seals'][~filled_slots] == -1)
