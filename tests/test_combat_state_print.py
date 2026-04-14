"""Tests for ``cs590_env.util.print_combat_state``."""

import io
import os
import sys
import unittest
from pathlib import Path

try:
    from google.colab import drive  # type: ignore[import-not-found,import-untyped]
except ImportError:
    drive = None

# Google Colab: mount Drive and add the repo to ``sys.path`` (matches TrainCombat.ipynb).
_DEFAULT_COLAB_REPO = (
    "/content/drive/MyDrive/Duke/CS 590 RL/Project/balatro-rl-agent"
)
if drive is not None:
    drive.mount("/content/drive")
    _colab_repo = os.environ.get("BALATRO_RL_AGENT_DIR", _DEFAULT_COLAB_REPO).rstrip("/")
    if os.path.isdir(_colab_repo):
        os.chdir(_colab_repo)
        if _colab_repo not in sys.path:
            sys.path.insert(0, _colab_repo)
    else:
        print(
            f"[test_combat_state_print] Skip Colab chdir: not a directory: {_colab_repo!r}. "
            "Set BALATRO_RL_AGENT_DIR to your clone path.",
            file=sys.stderr,
        )

from balatro_gym.save_injection import inject_save_into_balatro_env

from cs590_env.combat_wrapper import CombatActionWrapper
from cs590_env.schema import GamePhase
from cs590_env.util import print_combat_state
from cs590_env.wrapper import BalatroPhaseWrapper


REPO_ROOT = Path(__file__).resolve().parent.parent
COMBAT_JKR = REPO_ROOT / "save_files" / "combat" / "combat_first_round.jkr"


def _combat_obs_from_jkr(jkr_path: Path, seed: int = 42):
    """Build a COMBAT observation from an injected save without ``reset()`` (which would clear injection)."""
    base_env, _ = inject_save_into_balatro_env(jkr_path, seed=seed)
    phase = BalatroPhaseWrapper(base_env)
    phase._auto_skip_pack_open()
    obs = phase._get_phase_observation()
    combat = CombatActionWrapper(phase)
    obs = combat._advance_to_combat(obs)
    return obs


class PrintCombatStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not COMBAT_JKR.is_file():
            raise unittest.SkipTest(f"Missing fixture: {COMBAT_JKR}")

    def test_print_combat_state_from_jkr(self):
        obs = _combat_obs_from_jkr(COMBAT_JKR, seed=42)
        buf = io.StringIO()
        print_combat_state(obs, file=buf)
        text = buf.getvalue()
        # Echo to console (visible with ``pytest -s``; otherwise captured).
        print("\n--- print_combat_state (fixture) ---", flush=True)
        print(text, end="", flush=True)
        print("--- end print_combat_state ---\n", flush=True)
        self.assertIn("=== Run ===", text)
        self.assertIn("=== Blind / score ===", text)
        self.assertIn("=== Hand ===", text)
        self.assertIn("=== Hand levels", text)
        self.assertEqual(int(obs["phase"]), int(GamePhase.COMBAT))
        if int(obs["hand_size"]) > 0:
            # At least one playing-card line uses rank+suit symbols
            self.assertTrue(any(c in text for c in "♣♦♥♠"))

    def test_format_card_id_known(self):
        from cs590_env.util import _format_card_id

        # Two of clubs: rank 2 -> (2-2)*4+0 = 0
        self.assertEqual(_format_card_id(0), "2♣")
        # Ace of spades: (14-2)*4+3 = 51
        self.assertEqual(_format_card_id(51), "A♠")
        self.assertEqual(_format_card_id(-1), "PAD")
