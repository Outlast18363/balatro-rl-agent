"""Smoke tests for the runtime CS590 policy network."""

from __future__ import annotations

import importlib.util
import unittest

import numpy as np


HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_GYMNASIUM = importlib.util.find_spec("gymnasium") is not None


def build_synthetic_observation() -> dict:
    """Return one wrapper-shaped observation for fast policy smoke tests."""

    return {
        "hand_tokens": {
            "card_id": np.asarray([5, 18, 31, -1, -1, -1, -1, -1], dtype=np.int16),
            "is_empty": np.asarray([0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int8),
            "is_face_down": np.asarray([0, 0, 1, 0, 0, 0, 0, 0], dtype=np.int8),
            "enhancement_id": np.asarray([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.int8),
            "edition_id": np.asarray([0, 0, 2, 0, 0, 0, 0, 0], dtype=np.int8),
            "seal_id": np.asarray([0, 0, 1, 0, 0, 0, 0, 0], dtype=np.int8),
            "is_selected": np.asarray([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8),
        },
        "run_token": {
            "money": np.int32(4),
            "hands_left": np.int8(3),
            "discards_left": np.int8(2),
            "chips_needed": np.int32(300),
            "round_chips_scored": np.int32(120),
            "ante": np.int16(1),
            "round": np.int8(1),
            "phase": np.int8(1),
        },
        "deck_token": {
            "draw_pile_size": np.int16(47),
            "remaining_rank_histogram": np.asarray([4, 3, 4, 4, 4, 3, 4, 4, 4, 4, 4, 4, 1], dtype=np.int8),
            "remaining_suit_histogram": np.asarray([12, 12, 11, 12], dtype=np.int8),
        },
        "hand_levels": np.asarray([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int16),
        "boss_token": {
            "boss_blind_active": np.asarray([0], dtype=np.int8),
            "boss_blind_type": np.int8(0),
        },
        "joker_tokens": {
            "joker_id": np.asarray([7, 12, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int16),
            "is_empty": np.asarray([0, 0, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int8),
            "is_disabled": np.asarray([0, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8),
        },
        "action_masks": {
            "card_select_mask": np.asarray([1, 1, 1, 0, 0, 0, 0, 0], dtype=np.int8),
            "play_allowed": np.asarray([1], dtype=np.int8),
            "discard_allowed": np.asarray([1], dtype=np.int8),
            "selected_count": np.int8(1),
        },
    }


@unittest.skipUnless(HAS_TORCH, "requires torch")
class BalatroPolicySyntheticSmokeTest(unittest.TestCase):
    """Validate the minimal runtime model on a hand-crafted observation."""

    def test_forward_returns_expected_shapes(self) -> None:
        from cs590_src import BalatroModelConfig, BalatroPolicyNetwork

        model = BalatroPolicyNetwork(
            BalatroModelConfig(embed_dim=64, num_heads=4, dropout=0.0)
        )
        output = model(build_synthetic_observation())

        self.assertEqual(tuple(output.selection_logits.shape), (1, 8))
        self.assertEqual(tuple(output.masked_selection_logits.shape), (1, 8))
        self.assertEqual(tuple(output.execution_logits.shape), (1, 2))
        self.assertEqual(tuple(output.masked_execution_logits.shape), (1, 2))
        self.assertEqual(tuple(output.value.shape), (1,))

    def test_masking_pushes_empty_slots_to_invalid_logit(self) -> None:
        import torch

        from cs590_src import BalatroModelConfig, BalatroPolicyNetwork

        config = BalatroModelConfig(embed_dim=64, num_heads=4, dropout=0.0)
        model = BalatroPolicyNetwork(config)
        output = model(build_synthetic_observation())

        self.assertTrue(
            torch.all(output.masked_selection_logits[:, 3:] == config.invalid_logit)
        )


@unittest.skipUnless(HAS_TORCH and HAS_GYMNASIUM, "requires torch and gymnasium")
class BalatroPolicyWrapperSmokeTest(unittest.TestCase):
    """Validate that the runtime model can consume a real wrapper observation."""

    def test_forward_on_wrapper_observation(self) -> None:
        from balatro_gym.balatro_env_2 import BalatroEnv
        from cs590_env import BalatroArchWrapper
        from cs590_src import BalatroModelConfig, BalatroPolicyNetwork

        wrapper = BalatroArchWrapper(BalatroEnv(seed=7))
        observation, _ = wrapper.reset(seed=7)

        model = BalatroPolicyNetwork(
            BalatroModelConfig(embed_dim=64, num_heads=4, dropout=0.0)
        )
        output = model(observation)

        self.assertEqual(tuple(output.selection_logits.shape), (1, 8))
        self.assertEqual(tuple(output.execution_logits.shape), (1, 2))
        self.assertEqual(tuple(output.value.shape), (1,))

        wrapper.close()


if __name__ == "__main__":
    unittest.main()
