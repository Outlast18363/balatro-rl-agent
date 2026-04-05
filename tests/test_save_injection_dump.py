import unittest
from pathlib import Path
from pprint import pprint

from balatro_gym.constants import Phase
from balatro_gym.save_injection import inject_save_into_balatro_env


class SaveInjectionDumpTests(unittest.TestCase):
    """Dump injected env information for manual inspection."""

    def setUp(self):
        self.repo_root = Path(__file__).resolve().parents[1]
        self.save_dir = self.repo_root / "game_files"

    def _dump_injected_env(self, label: str, filename: str):
        save_path = self.save_dir / filename
        if not save_path.exists():
            self.skipTest(f"Missing save file: {save_path}")

        env, report = inject_save_into_balatro_env(save_path, seed=0, validate=True)
        obs = env._get_observation()
        snapshot = env.save_state()

        print("\n" + "=" * 90)
        print(f"[SAVE INJECTION DUMP] {label}")
        print(f"source: {save_path}")
        print("-" * 90)
        print("Phase:", Phase(int(obs["phase"])).name)
        print("Money:", int(obs["money"]))
        print("Ante/Round:", int(obs["ante"]), int(obs["round"]))
        print("Hands/Discards left:", int(obs["hands_left"]), int(obs["discards_left"]))
        print("Deck size:", int(obs["deck_size"]), "| Hand size:", int(obs["hand_size"]))
        print("Report summary:")
        pprint(
            {
                "applied_fields_count": len(report.get("applied_fields", [])),
                "missing_in_save_count": len(report.get("missing_in_save", [])),
                "ignored_from_save_total": report.get("ignored_from_save_total"),
                "validation": report.get("validation"),
            }
        )

        print("\n[Injected env.state (dataclass copy)]")
        pprint(snapshot["state"])

        print("\n[Injected observation dict]")
        pprint(obs)

        print("\n[Injected save_state snapshot]")
        pprint(snapshot)
        print("=" * 90 + "\n")

        # Basic sanity checks so this is still a real test.
        self.assertIn("state", snapshot)
        self.assertIn("action_mask", obs)
        self.assertEqual(len(obs["action_mask"]), 60)

    def test_dump_first_combat_injection(self):
        self._dump_injected_env("first_blind_combat", "first_blind_combat_save.jkr")

    def test_dump_mid_game_injection(self):
        self._dump_injected_env("mid_game", "mid_game_save.jkr")


if __name__ == "__main__":
    unittest.main()
