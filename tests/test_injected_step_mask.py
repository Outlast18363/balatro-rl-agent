import unittest
from pathlib import Path

import numpy as np

from balatro_gym.constants import Action, Phase
from balatro_gym.save_injection import inject_save_into_balatro_env
from cs590_env.wrapper import BalatroPhaseWrapper


class InjectedStepMaskTests(unittest.TestCase):
    """Regression tests for action masks after save-file injection."""

    def setUp(self):
        """Locate the bundled save fixtures used by the injection regression.

        Returns:
            None. Stores the repository root and save directory on the test case.
        """
        self.repo_root = Path(__file__).resolve().parents[1]
        self.save_dir = self.repo_root / "game_files"

    def _inject(self, filename: str):
        """Inject a bundled save fixture into a fresh BalatroEnv.

        Args:
            filename: Save fixture filename under ``game_files``.

        Returns:
            ``(env, report)`` from ``inject_save_into_balatro_env``.
        """
        save_path = self.save_dir / filename
        if not save_path.exists():
            self.skipTest(f"Missing save file: {save_path}")
        return inject_save_into_balatro_env(save_path, seed=0, validate=True)

    def _valid_actions(self, obs: dict) -> set[int]:
        """Return the currently legal action ids from an observation mask.

        Args:
            obs: Raw environment observation containing ``action_mask``.

        Returns:
            Set of legal flat action ids.
        """
        return {int(idx) for idx in np.flatnonzero(obs["action_mask"])}

    def _advance_mid_game_injection_to_shop(self, env):
        """Play the injected mid-game state forward until the shop opens.

        Args:
            env: Injected raw ``BalatroEnv`` currently in ``Phase.BLIND_SELECT``.

        Returns:
            Shop-phase observation reached from the injected save.
        """
        blind_action = int(Action.SELECT_BLIND_BASE + (env.state.round - 1))
        obs, _, terminated, truncated, info = env.step(blind_action)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertNotEqual(info.get("error"), "Invalid action")
        self.assertEqual(Phase(int(obs["phase"])), Phase.PLAY)

        # Lower the blind target so a single legal hand ends the round quickly.
        env.state.chips_needed = 1

        obs, _, terminated, truncated, info = env.step(int(Action.SELECT_CARD_BASE))
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertNotEqual(info.get("error"), "Invalid action")
        self.assertEqual(int(obs["action_mask"][int(Action.PLAY_HAND)]), 1)

        obs, _, terminated, truncated, info = env.step(int(Action.PLAY_HAND))
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertNotEqual(info.get("error"), "Invalid action")
        self.assertEqual(Phase(int(obs["phase"])), Phase.SHOP)
        return obs

    def test_combat_injection_masks_and_legal_step(self):
        """Injected combat state should expose only combat actions and remain stepable."""
        env, _ = self._inject("first_blind_combat_save.jkr")
        obs = env._get_observation()
        valid_actions = self._valid_actions(obs)
        allowed_actions = {
            int(Action.PLAY_HAND),
            int(Action.DISCARD),
            *range(int(Action.SELECT_CARD_BASE), int(Action.SELECT_CARD_BASE) + int(Action.SELECT_CARD_COUNT)),
            *range(
                int(Action.USE_CONSUMABLE_BASE),
                int(Action.USE_CONSUMABLE_BASE) + int(Action.USE_CONSUMABLE_COUNT),
            ),
        }

        self.assertEqual(Phase(int(obs["phase"])), Phase.PLAY)
        self.assertTrue(valid_actions)
        self.assertTrue(valid_actions.issubset(allowed_actions))

        mask = obs["action_mask"]
        self.assertEqual(int(mask[int(Action.PLAY_HAND)]), 0)
        self.assertEqual(int(mask[int(Action.DISCARD)]), 0)
        for action in range(int(Action.SHOP_BUY_BASE), int(Action.SHOP_END) + 1):
            self.assertEqual(int(mask[action]), 0)
        for action in range(int(Action.SELECT_BLIND_BASE), int(Action.SKIP_BLIND) + 1):
            self.assertEqual(int(mask[action]), 0)

        obs, _, terminated, truncated, info = env.step(int(Action.SELECT_CARD_BASE))
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertNotEqual(info.get("error"), "Invalid action")

        mask = obs["action_mask"]
        self.assertEqual(int(mask[int(Action.PLAY_HAND)]), 1)
        self.assertEqual(int(mask[int(Action.DISCARD)]), int(env.state.discards_left > 0))

        follow_up_action = int(Action.DISCARD) if int(mask[int(Action.DISCARD)]) == 1 else int(Action.PLAY_HAND)
        step_out = env.step(follow_up_action)
        self.assertEqual(len(step_out), 5)
        self.assertNotEqual(step_out[-1].get("error"), "Invalid action")

    def test_combat_injection_masks_play_and_discard_for_six_selected_cards(self):
        """Raw env mask should stay aligned with the 1-5 card playability rule."""
        env, _ = self._inject("first_blind_combat_save.jkr")

        for offset in range(6):
            obs, _, terminated, truncated, info = env.step(int(Action.SELECT_CARD_BASE + offset))
            self.assertFalse(terminated)
            self.assertFalse(truncated)
            self.assertNotEqual(info.get("error"), "Invalid action")

        mask = obs["action_mask"]
        self.assertEqual(len(env.state.selected_cards), 6)
        self.assertEqual(int(mask[int(Action.PLAY_HAND)]), 0)
        self.assertEqual(int(mask[int(Action.DISCARD)]), 0)

    def test_blind_select_injection_only_enables_current_round_actions(self):
        """Injected blind-select state should expose only its live blind slot and skip rule."""
        env, _ = self._inject("mid_game_save.jkr")
        obs = env._get_observation()
        mask = obs["action_mask"]
        round_index = int(obs["round"]) - 1

        self.assertEqual(Phase(int(obs["phase"])), Phase.BLIND_SELECT)
        for offset in range(int(Action.SELECT_BLIND_COUNT)):
            expected = 1 if offset == round_index else 0
            self.assertEqual(int(mask[int(Action.SELECT_BLIND_BASE) + offset]), expected)
        self.assertEqual(int(mask[int(Action.SKIP_BLIND)]), 1)

        blind_action = int(Action.SELECT_BLIND_BASE + round_index)
        step_out = env.step(blind_action)
        self.assertEqual(len(step_out), 5)
        self.assertNotEqual(step_out[-1].get("error"), "Invalid action")
        self.assertEqual(Phase(int(step_out[0]["phase"])), Phase.PLAY)

    def test_shop_mask_after_injected_round_uses_live_shop_state(self):
        """Shop actions after injected progression should match live inventory and money."""
        env, _ = self._inject("mid_game_save.jkr")
        obs = self._advance_mid_game_injection_to_shop(env)
        mask = obs["action_mask"]

        self.assertEqual(Phase(int(obs["phase"])), Phase.SHOP)
        self.assertEqual(int(mask[int(Action.SHOP_END)]), 1)
        self.assertEqual(int(mask[int(Action.PLAY_HAND)]), 0)
        self.assertEqual(int(mask[int(Action.DISCARD)]), 0)
        for action in range(int(Action.SELECT_BLIND_BASE), int(Action.SKIP_BLIND) + 1):
            self.assertEqual(int(mask[action]), 0)

        inventory = env.shop.inventory if env.shop else []
        for slot in range(int(Action.SHOP_BUY_COUNT)):
            expected = 0
            if slot < len(inventory) and env.state.money >= inventory[slot].cost:
                expected = 1
            self.assertEqual(int(mask[int(Action.SHOP_BUY_BASE) + slot]), expected)

        reroll_expected = int(env.state.money >= env.state.shop_reroll_cost)
        self.assertEqual(int(mask[int(Action.SHOP_REROLL)]), reroll_expected)

    def test_wrapper_smoke_after_injection(self):
        """Injected raw env should still produce a wrapper-compatible observation."""
        env, _ = self._inject("first_blind_combat_save.jkr")
        wrapped = BalatroPhaseWrapper(env)
        wrapped_obs = wrapped._get_phase_observation()

        self.assertEqual(len(wrapped_obs["action_mask"]), int(Action.ACTION_SPACE_SIZE))
        self.assertTrue(wrapped.observation_space.contains(wrapped_obs))


if __name__ == "__main__":
    unittest.main()
