import numpy as np
import pytest

from balatro_gym.balatro_env_2 import BalatroEnv
from cs590_env.combat_env import PooledCombatEnv
from cs590_env.combat_wrapper import CombatActionWrapper
from cs590_env.schema import GamePhase, MAX_HAND_SIZE


def _make_combat_obs(selected_slots: list[int] | None = None) -> dict:
    """Build a minimal combat observation for wrapper-only unit tests.

    Args:
        selected_slots: Hand slot indices marked selected in the returned obs.

    Returns:
        Observation dict containing the phase and selection vector fields that
        ``CombatActionWrapper.step()`` reads.
    """
    selected = np.zeros(MAX_HAND_SIZE, dtype=np.int8)
    for idx in selected_slots or []:
        selected[idx] = 1
    return {
        "phase": np.int8(GamePhase.COMBAT),
        "hand_is_selected": selected,
    }


class _FakePhaseEnv:
    """Minimal phase env stub for unit-testing ``CombatActionWrapper``.

    Args:
        steps: Sequence of pre-baked ``step()`` results returned in order.
    """

    def __init__(self, steps: list[tuple[dict, float, bool, bool, dict]]):
        self._steps = iter(steps)

    @property
    def unwrapped(self):
        """Mirror ``gym.Wrapper.unwrapped`` expected by the wrapper."""
        return self

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        """Return the next scripted transition regardless of ``action``."""
        return next(self._steps)

    def close(self) -> None:
        """No-op close hook for the fake env."""
        return None


def test_combat_action_wrapper_uses_only_execution_reward():
    """Toggle steps should stay reward-neutral and expose only execution reward."""
    toggle_obs = _make_combat_obs([0])
    exec_obs = {
        "phase": np.int8(GamePhase.SHOP),
        "hand_is_selected": np.zeros(MAX_HAND_SIZE, dtype=np.int8),
    }
    wrapper = CombatActionWrapper(
        _FakePhaseEnv(
            [
                (toggle_obs, 0.0, False, False, {}),
                (exec_obs, 0.75, False, False, {}),
            ]
        )
    )
    wrapper._last_obs = _make_combat_obs()

    obs, reward, terminated, truncated, info = wrapper.step(
        np.array([1] + [0] * (MAX_HAND_SIZE - 1), dtype=np.int8),
        execution=0,
    )

    assert reward == pytest.approx(0.75)
    assert terminated
    assert not truncated
    assert obs["phase"] == GamePhase.SHOP
    assert info["combat_ended"] is True


def test_combat_action_wrapper_rejects_nonzero_toggle_reward():
    """Non-zero rewards on internal toggle steps should surface as a bug."""
    wrapper = CombatActionWrapper(
        _FakePhaseEnv(
            [
                (_make_combat_obs([0]), 0.5, False, False, {}),
            ]
        )
    )
    wrapper._last_obs = _make_combat_obs()

    with pytest.raises(RuntimeError, match="SELECT_CARD action"):
        wrapper.step(
            np.array([1] + [0] * (MAX_HAND_SIZE - 1), dtype=np.int8),
            execution=0,
        )


def test_pooled_combat_env_preserves_truncation_without_reset(monkeypatch):
    """Top-level combat env should forward terminal flags without self-resetting."""
    seed_env = BalatroEnv(seed=7)
    seed_env.reset()
    env = PooledCombatEnv([seed_env.save_state()])
    env._combat.step = lambda *_args: (
        {"phase": np.int8(GamePhase.COMBAT)},
        1.25,
        False,
        True,
        {"truncated": "max_ante_reached"},
    )

    def fail_reset(*args, **kwargs):
        raise AssertionError("PooledCombatEnv.step() should not call reset().")

    monkeypatch.setattr(env, "reset", fail_reset)

    obs, reward, terminated, truncated, info = env.step(
        np.zeros(MAX_HAND_SIZE + 1, dtype=np.int8)
    )

    assert reward == pytest.approx(1.25)
    assert not terminated
    assert truncated
    assert obs["phase"] == GamePhase.COMBAT
    assert info["truncated"] == "max_ante_reached"


def test_balatro_env_max_ante_reports_truncation():
    """Ante caps should surface as truncation, not terminal failure."""
    env = BalatroEnv(seed=11)
    env.reset()
    env.state.ante = 101

    _, reward, terminated, truncated, info = env.step(0)

    assert reward == 0.0
    assert not terminated
    assert truncated
    assert info["truncated"] == "max_ante_reached"
