"""Example-driven tests for the architecture-facing Balatro wrapper.

The tests intentionally showcase the expected call pattern so this file also
serves as lightweight usage documentation for future model code.
"""

from __future__ import annotations

import unittest

import numpy as np

from balatro_gym.balatro_env_2 import BalatroEnv
from balatro_gym.constants import Action, Phase
from cs590_env import BalatroArchWrapper
from cs590_env.arch_schema import ARCH_OBSERVATION_KEYS, ArchExecuteAction, MAX_HAND_SIZE, PAD_CARD_ID


def make_wrapper(seed: int = 7) -> BalatroArchWrapper:
    """Create a fresh wrapper instance for one test case.

    Parameters:
        seed: Deterministic seed so payload shapes and translations stay stable.
    Returns:
        A wrapper that auto-advances the base env into the combat phase.
    """

    return BalatroArchWrapper(BalatroEnv(seed=seed))


class BalatroArchWrapperTest(unittest.TestCase):
    """Validate wrapper payloads and demonstrate the intended call pattern."""

    def test_reset_returns_documented_payload(self) -> None:
        """`reset()` should be the main entry point for the model-facing payload."""

        wrapper = make_wrapper()
        observation, info = wrapper.reset(seed=7)

        # The wrapper auto-advances the raw env so the model starts from combat.
        self.assertEqual(int(observation["run_token"]["phase"]), int(Phase.PLAY))
        self.assertEqual(set(observation.keys()), set(ARCH_OBSERVATION_KEYS))
        self.assertIn("auto_advanced", info)

        # These shape checks document what downstream embedding code can expect.
        self.assertEqual(observation["hand_tokens"]["card_id"].shape, (MAX_HAND_SIZE,))
        self.assertEqual(
            observation["deck_token"]["remaining_rank_histogram"].shape,
            (13,),
        )
        self.assertEqual(
            observation["deck_token"]["remaining_suit_histogram"].shape,
            (4,),
        )
        self.assertEqual(observation["hand_levels"].shape, (12,))
        self.assertEqual(observation["joker_tokens"]["joker_id"].shape, (10,))

        wrapper.close()

    def test_padding_and_masks_make_empty_slots_explicit(self) -> None:
        """Empty slots and legality masks should be obvious from the payload."""

        wrapper = make_wrapper()
        wrapper.reset(seed=7)

        # Trim the hand so the payload shows how empty-slot padding is encoded.
        wrapper.base_env.state.hand_indexes = wrapper.base_env.state.hand_indexes[:3]
        wrapper.base_env.state.selected_cards = [0, 2]
        wrapper.base_env.state.discards_left = 1

        observation = wrapper.get_arch_observation()
        hand_tokens = observation["hand_tokens"]
        action_masks = observation["action_masks"]

        np.testing.assert_array_equal(
            hand_tokens["is_empty"][3:],
            np.ones(MAX_HAND_SIZE - 3, dtype=np.int8),
        )
        np.testing.assert_array_equal(
            hand_tokens["card_id"][3:],
            np.full(MAX_HAND_SIZE - 3, PAD_CARD_ID, dtype=np.int16),
        )
        self.assertEqual(int(action_masks["selected_count"]), 2)
        self.assertEqual(int(action_masks["play_allowed"][0]), 1)
        self.assertEqual(int(action_masks["discard_allowed"][0]), 1)

        wrapper.close()

    def test_translate_arch_action_to_env_action_shows_expected_call_shape(self) -> None:
        """Translation should reveal exactly how structured actions hit the env."""

        wrapper = make_wrapper()
        wrapper.reset(seed=7)

        # This is the canonical wrapper action shape: target selection + execute.
        action = {
            "selection": np.asarray([1, 1, 0, 0, 0, 0, 0, 0], dtype=np.int8),
            "execute": ArchExecuteAction.PLAY,
        }
        translated = wrapper.translate_arch_action_to_env_action(action)

        # Two toggles are needed to select slots 0 and 1, then we play the hand.
        self.assertEqual(
            translated,
            [int(Action.SELECT_CARD_BASE) + 0, int(Action.SELECT_CARD_BASE) + 1, int(Action.PLAY_HAND)],
        )

        wrapper.close()

    def test_step_returns_same_schema_after_a_structured_play(self) -> None:
        """`step()` should accept the structured wrapper action and return the same payload schema."""

        wrapper = make_wrapper()
        wrapper.reset(seed=7)

        action = {
            "selection": np.asarray([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8),
            "execute": ArchExecuteAction.PLAY,
        }
        observation, reward, terminated, truncated, info = wrapper.step(action)

        self.assertEqual(set(observation.keys()), set(ARCH_OBSERVATION_KEYS))
        self.assertIn("translated_actions", info)
        self.assertIsInstance(reward, float)
        self.assertFalse(truncated)
        self.assertIsInstance(terminated, bool)

        wrapper.close()


if __name__ == "__main__":
    unittest.main()
