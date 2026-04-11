"""cs590_env/combat_wrapper.py - Factored action wrapper for the combat phase.

Bridges the CombatPPOAgent's factored action space (per-card binary selections
+ play/discard) with the sequential toggle-based BalatroPhaseWrapper.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from cs590_env.schema import (
    GamePhase,
    WrapperAction,
    SELECT_CARD_COUNT,
    get_wrapper_select_action,
)
from cs590_env.wrapper import BalatroPhaseWrapper


class CombatActionWrapper:
    """Bridges the factored action space (binary card selections + play/discard)
    with the sequential toggle-based BalatroPhaseWrapper.

    On reset, auto-advances through non-combat phases (blind select, shop)
    so the agent always sees a COMBAT observation.

    On step, translates a factored action into sequential env steps:
      1. Toggle cards to match the desired selection
      2. Execute play (action 0) or discard (action 1)

    Args:
        env: A ``BalatroPhaseWrapper`` instance (or subclass such as
             ``ConfiguredPhaseEnv``).
    """

    def __init__(self, env: BalatroPhaseWrapper):
        self.env = env
        self._last_obs: dict | None = None

    @property
    def unwrapped(self):
        """Access the underlying base ``BalatroEnv``."""
        return self.env.unwrapped

    def reset(self, **kwargs) -> Tuple[dict, dict]:
        """Reset the wrapped env and advance to the COMBAT phase.

        Returns:
            (obs, info) tuple where obs is a combat-phase observation.
        """
        obs, info = self.env.reset(**kwargs)
        obs = self._advance_to_combat(obs)
        self._last_obs = obs
        return obs, info

    def _advance_to_combat(self, obs: dict) -> dict:
        """Step through non-combat phases with default actions until
        we reach COMBAT."""
        while True:
            phase = GamePhase(int(obs['phase']))
            if phase == GamePhase.COMBAT:
                return obs

            if phase == GamePhase.TRANSITION:
                mask = obs['action_mask']
                for i in range(3):
                    if mask[WrapperAction.SELECT_BLIND_BASE + i]:
                        obs, _, done, _, _ = self.env.step(
                            int(WrapperAction.SELECT_BLIND_BASE + i))
                        break
            elif phase == GamePhase.SHOP:
                obs, _, done, _, _ = self.env.step(int(WrapperAction.SHOP_END))
            else:
                break

            if done:
                obs, _ = self.env.reset()
        return obs

    def step(
        self,
        card_selections: np.ndarray,
        execution: int,
    ) -> Tuple[dict, float, bool, dict]:
        """Execute a factored combat action.

        Internally toggles card selections to match the desired state, then
        fires the play or discard action.  The entire sequence is treated as
        one atomic transition from the agent's perspective.

        Args:
            card_selections: ``(MAX_HAND_SIZE,)`` binary array — 1 = select,
                             0 = don't select.
            execution:       0 = play, 1 = discard.

        Returns:
            ``(obs, reward, done, info)`` tuple.
        """
        n_selected = int(card_selections.sum())
        if n_selected < 1 or n_selected > 5:
            return self._last_obs, -1.0, False, {
                'error': f'Invalid selection count: {n_selected}'}

        current_sel = self._last_obs['hand_is_selected']
        to_toggle = np.where(card_selections != current_sel)[0]

        total_reward = 0.0
        obs = self._last_obs
        done = False
        info: dict = {}

        for idx in to_toggle:
            if idx >= SELECT_CARD_COUNT:
                continue
            action = get_wrapper_select_action(int(idx))
            obs, r, done, _, info = self.env.step(action)
            total_reward += r
            if done:
                self._last_obs = obs
                return obs, total_reward, True, info

        exec_action = (int(WrapperAction.PLAY_HAND) if execution == 0
                       else int(WrapperAction.DISCARD))
        obs, r, done, _, info = self.env.step(exec_action)
        total_reward += r

        if not done:
            phase = GamePhase(int(obs['phase']))
            if phase != GamePhase.COMBAT:
                done = True
                info['combat_ended'] = True

        self._last_obs = obs
        return obs, total_reward, done, info

    def close(self) -> None:
        """Close the underlying environment."""
        self.env.close()
