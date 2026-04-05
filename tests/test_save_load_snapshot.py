import unittest

import numpy as np

from balatro_gym.balatro_env_2 import BalatroEnv
from balatro_gym.cards import CardState, Enhancement
from balatro_gym.constants import Action, Phase


class SaveLoadSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.env = BalatroEnv(seed=123)
        self.env.reset(seed=123)

    def _first_valid_action(self, obs):
        valid = np.where(obs["action_mask"])[0]
        self.assertGreater(len(valid), 0, "No valid action available.")
        return int(valid[0])

    def test_roundtrip_state_integrity(self):
        # Move to PLAY and mutate runtime trackers.
        self.env.step(int(Action.SELECT_BLIND_BASE))
        obs = self.env._get_observation()
        if int(obs["action_mask"][int(Action.SELECT_CARD_BASE)]) == 1:
            self.env.step(int(Action.SELECT_CARD_BASE))
        obs = self.env._get_observation()
        if int(obs["action_mask"][int(Action.PLAY_HAND)]) == 1:
            self.env.step(int(Action.PLAY_HAND))

        self.env.boss_blind_manager.blind_state = {
            "played_hand_types": {"Pair"},
            "played_cards": {1, 2},
            "hands_played": 1,
        }
        self.env.joker_effects_engine.joker_states = {"Ride the Bus": {"mult": 3}}

        saved = self.env.save_state()
        expected_money = saved["state"].money
        expected_hand_indexes = saved["state"].hand_indexes.copy()
        expected_game_hand_indexes = saved["game_state"]["hand_indexes"].copy()
        expected_round_hands = saved["game_state"]["round_hands"]

        # Perturb multiple sub-systems after saving.
        self.env.state.money += 99
        self.env.state.hand_indexes = []
        self.env.game.hand_indexes = []
        self.env.game.round_hands = 0
        self.env.boss_blind_manager.blind_state["played_hand_types"].add("Flush")
        self.env.joker_effects_engine.joker_states["Ride the Bus"]["mult"] = 0

        self.env.load_state(saved)

        self.assertEqual(self.env.state.money, expected_money)
        self.assertEqual(self.env.state.hand_indexes, expected_hand_indexes)
        self.assertEqual(self.env.game.hand_indexes, expected_game_hand_indexes)
        self.assertEqual(self.env.game.round_hands, expected_round_hands)
        self.assertEqual(self.env.boss_blind_manager.blind_state["played_hand_types"], {"Pair"})
        self.assertEqual(self.env.joker_effects_engine.joker_states["Ride the Bus"]["mult"], 3)

    def test_card_states_are_isolated(self):
        self.env.state.card_states[0] = CardState(card_id=0, enhancement=Enhancement.BONUS)
        saved = self.env.save_state()

        # Mutate runtime state after save; snapshot must remain unchanged.
        self.env.state.card_states[0].enhancement = Enhancement.MULT
        self.assertEqual(saved["state"].card_states[0].enhancement, Enhancement.BONUS)

        self.env.load_state(saved)
        self.assertEqual(self.env.state.card_states[0].enhancement, Enhancement.BONUS)

    def test_shop_phase_roundtrip(self):
        # Build a live shop snapshot in a controlled way.
        self.env.state.phase = Phase.SHOP
        self.env._generate_shop()
        self.assertIsNotNone(self.env.shop)

        saved = self.env.save_state()
        self.assertIsNotNone(saved.get("shop_state"))

        expected_items = [(item.name, item.cost, int(item.item_type)) for item in self.env.shop.inventory]
        expected_reroll = self.env.shop.reroll_cost
        expected_player_chips = self.env.shop.player.chips

        # Break shop internals.
        self.env.shop.inventory = []
        self.env.shop.reroll_cost = 999
        self.env.shop.player.chips = 0
        self.env.state.money = 0

        self.env.load_state(saved)
        self.assertIsNotNone(self.env.shop)
        restored_items = [(item.name, item.cost, int(item.item_type)) for item in self.env.shop.inventory]
        self.assertEqual(restored_items, expected_items)
        self.assertEqual(self.env.shop.reroll_cost, expected_reroll)
        self.assertEqual(self.env.shop.player.chips, expected_player_chips)
        self.assertEqual(len(self.env.state.shop_inventory), len(expected_items))

    def test_step_after_restore(self):
        saved = self.env.save_state()
        obs = self.env._get_observation()
        action = self._first_valid_action(obs)
        self.env.step(action)

        self.env.load_state(saved)
        restored_obs = self.env._get_observation()
        restored_action = self._first_valid_action(restored_obs)
        out = self.env.step(restored_action)
        self.assertEqual(len(out), 5)


if __name__ == "__main__":
    unittest.main()
