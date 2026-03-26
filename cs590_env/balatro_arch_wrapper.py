"""Architecture-facing wrapper for the Balatro simulator.

The wrapper exposes a compact tokenized payload tailored to the combat model
while keeping the underlying simulator unchanged.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import gymnasium as gym
import numpy as np

from balatro_gym.balatro_env_2 import BalatroEnv, CardAdapter
from balatro_gym.cards import CardState
from balatro_gym.constants import Action, Phase

from cs590_env.arch_schema import (
    ACTION_MASK_KEYS,
    ARCH_OBSERVATION_KEYS,
    BOSS_TOKEN_KEYS,
    DECK_TOKEN_KEYS,
    HAND_LEVEL_COUNT,
    HAND_TOKEN_KEYS,
    MAX_HAND_SIZE,
    MAX_JOKER_SLOTS,
    PAD_CARD_ID,
    PAD_JOKER_ID,
    PAD_MODIFIER_ID,
    RUN_TOKEN_KEYS,
    SUIT_HISTOGRAM_SIZE,
    RANK_HISTOGRAM_SIZE,
    JOKER_TOKEN_KEYS,
    ArchExecuteAction,
    build_action_space,
    build_observation_space,
)


class BalatroArchWrapper(gym.Wrapper):
    """Expose a model-ready observation and action contract.

    Parameters:
        env: The base `BalatroEnv` simulator instance to wrap.
        auto_advance: When `True`, automatically skip non-combat phases so the
            wrapper always returns combat-phase observations to the model.
        invalid_action_penalty: Reward returned when a structured wrapper
            action cannot be translated into a legal simulator action.
    Returns:
        A wrapper with nested observation payloads and structured actions.
    """

    def __init__(
        self,
        env: BalatroEnv,
        *,
        auto_advance: bool = True,
        invalid_action_penalty: float = -1.0,
    ) -> None:
        super().__init__(env)
        self.auto_advance = auto_advance
        self.invalid_action_penalty = invalid_action_penalty
        self.observation_space = build_observation_space()
        self.action_space = build_action_space()

    @property
    def base_env(self) -> BalatroEnv:
        """Return the underlying simulator with the concrete env type.

        Returns:
            The wrapped `BalatroEnv` instance.
        """

        return self.unwrapped

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the wrapper and return the architecture-facing observation.

        Parameters:
            seed: Optional environment seed forwarded to the base env.
            options: Optional reset options forwarded to the base env.
        Returns:
            A tuple of `(arch_observation, info)` after optional phase
            auto-advancement.
        """

        _, info = self.env.reset(seed=seed, options=options)
        auto_reward, terminated, truncated, auto_info = self._advance_to_play_phase()
        info = dict(info)
        if auto_info["auto_actions"]:
            info["auto_advanced"] = auto_info
            info["auto_advance_reward"] = auto_reward
        if terminated or truncated:
            info["warning"] = "reset ended before reaching play phase"
        return self.get_arch_observation(), info

    def step(
        self, action: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute one structured wrapper action.

        Parameters:
            action: A dict with `selection` and `execute` keys matching the
                wrapper action space.
        Returns:
            The wrapped observation, accumulated reward, termination flags, and
            info including the translated simulator actions.
        """

        try:
            translated_actions = self.translate_arch_action_to_env_action(action)
        except ValueError as exc:
            return (
                self.get_arch_observation(),
                self.invalid_action_penalty,
                False,
                False,
                {"error": str(exc), "translated_actions": []},
            )

        total_reward = 0.0
        terminated = False
        truncated = False
        last_info: Dict[str, Any] = {}

        for flat_action in translated_actions:
            _, reward, terminated, truncated, info = self.env.step(flat_action)
            total_reward += reward
            last_info = info
            if terminated or truncated:
                break

        if not terminated and not truncated:
            auto_reward, auto_terminated, auto_truncated, auto_info = (
                self._advance_to_play_phase()
            )
            total_reward += auto_reward
            terminated = terminated or auto_terminated
            truncated = truncated or auto_truncated
            if auto_info["auto_actions"]:
                last_info = dict(last_info)
                last_info["auto_advanced"] = auto_info

        last_info = dict(last_info)
        last_info["translated_actions"] = translated_actions
        return self.get_arch_observation(), total_reward, terminated, truncated, last_info

    def get_arch_observation(self) -> Dict[str, Any]:
        """Build the wrapper observation from the current simulator state.

        Returns:
            A nested observation dict with token groups aligned to the model
            architecture.
        """

        observation = self._build_arch_observation()
        return observation

    def get_arch_masks(self) -> Dict[str, Any]:
        """Return the factorized legality masks for the current state.

        Returns:
            The `action_masks` section of the wrapper observation.
        """

        return self._build_action_masks()

    def translate_arch_action_to_env_action(self, action: Dict[str, Any]) -> List[int]:
        """Translate one structured wrapper action into flat env actions.

        Parameters:
            action: A dict with a target hand-selection vector and an execution
                choice (`PLAY` or `DISCARD`).
        Returns:
            A deterministic list of simulator action IDs. The list first
            toggles any slots needed to reach the target selection, then emits
            the final execute action.
        """

        selection, execute = self._normalize_arch_action(action)
        hand_size = len(self.base_env.state.hand_indexes)
        invalid_slots = [idx for idx in np.flatnonzero(selection) if idx >= hand_size]
        if invalid_slots:
            raise ValueError(f"selection includes empty hand slots: {invalid_slots}")

        current_selected = set(self.base_env.state.selected_cards)
        target_selected = set(np.flatnonzero(selection))
        translated_actions = [
            int(Action.SELECT_CARD_BASE) + idx
            for idx in sorted(current_selected ^ target_selected)
        ]

        if execute is ArchExecuteAction.PLAY:
            if not self._is_play_allowed(target_selected):
                raise ValueError("play action is not legal for the requested selection")
            translated_actions.append(int(Action.PLAY_HAND))
        else:
            if not self._is_discard_allowed(target_selected):
                raise ValueError(
                    "discard action is not legal for the requested selection"
                )
            translated_actions.append(int(Action.DISCARD))

        return translated_actions

    def _build_arch_observation(self) -> Dict[str, Any]:
        """Compose the full observation payload from wrapper helper blocks.

        Returns:
            A nested observation dict keyed by the schema declared in
            `cs590_env.arch_schema`.
        """

        observation = {
            "hand_tokens": self._build_hand_tokens(),
            "run_token": self._build_run_token(),
            "deck_token": self._build_deck_token(),
            "hand_levels": self._build_hand_levels(),
            "boss_token": self._build_boss_token(),
            "joker_tokens": self._build_joker_tokens(),
            "action_masks": self._build_action_masks(),
        }
        return observation

    def _build_hand_tokens(self) -> Dict[str, np.ndarray]:
        """Convert the current hand into per-slot token fields.

        Returns:
            A dict of parallel arrays for card identity, local modifiers, and
            slot-level flags.
        """

        state = self.base_env.state
        hand_indexes = state.hand_indexes[:MAX_HAND_SIZE]
        face_down_slots = set(state.face_down_cards)
        selected_slots = set(state.selected_cards)

        card_ids = np.full(MAX_HAND_SIZE, PAD_CARD_ID, dtype=np.int16)
        is_empty = np.ones(MAX_HAND_SIZE, dtype=np.int8)
        is_face_down = np.zeros(MAX_HAND_SIZE, dtype=np.int8)
        enhancement_ids = np.full(MAX_HAND_SIZE, PAD_MODIFIER_ID, dtype=np.int8)
        edition_ids = np.full(MAX_HAND_SIZE, PAD_MODIFIER_ID, dtype=np.int8)
        seal_ids = np.full(MAX_HAND_SIZE, PAD_MODIFIER_ID, dtype=np.int8)
        is_selected = np.zeros(MAX_HAND_SIZE, dtype=np.int8)

        for slot_idx, deck_idx in enumerate(hand_indexes):
            card = state.deck[deck_idx]
            card_state = state.card_states.get(deck_idx, CardState(deck_idx))
            face_down = slot_idx in face_down_slots

            is_empty[slot_idx] = 0
            is_face_down[slot_idx] = int(face_down)
            is_selected[slot_idx] = int(slot_idx in selected_slots)

            # Hidden cards should not leak identity or modifier information.
            if not face_down:
                card_ids[slot_idx] = CardAdapter.encode_to_int(card)
                enhancement_ids[slot_idx] = int(card_state.enhancement)
                edition_ids[slot_idx] = int(card_state.edition)
                seal_ids[slot_idx] = int(card_state.seal)

        return {
            "card_id": card_ids,
            "is_empty": is_empty,
            "is_face_down": is_face_down,
            "enhancement_id": enhancement_ids,
            "edition_id": edition_ids,
            "seal_id": seal_ids,
            "is_selected": is_selected,
        }

    def _build_run_token(self) -> Dict[str, np.generic]:
        """Collect the global run-state scalars used by the model.

        Returns:
            A dict of scalar numpy values describing resources, score progress,
            and current phase.
        """

        state = self.base_env.state
        return {
            "money": np.int32(state.money),
            "hands_left": np.int8(state.hands_left),
            "discards_left": np.int8(state.discards_left),
            "chips_needed": np.int32(state.chips_needed),
            "round_chips_scored": np.int32(state.round_chips_scored),
            "ante": np.int16(state.ante),
            "round": np.int8(state.round),
            "phase": np.int8(state.phase),
        }

    def _build_deck_token(self) -> Dict[str, np.ndarray | np.generic]:
        """Summarize cards that are not currently in hand.

        Returns:
            A histogram-based deck token with remaining-card count plus rank and
            suit histograms.
        """

        state = self.base_env.state
        hand_index_set = set(state.hand_indexes)
        rank_histogram = np.zeros(RANK_HISTOGRAM_SIZE, dtype=np.int8)
        suit_histogram = np.zeros(SUIT_HISTOGRAM_SIZE, dtype=np.int8)

        draw_pile_size = 0
        for deck_idx, card in enumerate(state.deck):
            if deck_idx in hand_index_set:
                continue
            draw_pile_size += 1
            rank_histogram[card.rank.value - 2] += 1
            suit_histogram[int(card.suit)] += 1

        return {
            "draw_pile_size": np.int16(draw_pile_size),
            "remaining_rank_histogram": rank_histogram,
            "remaining_suit_histogram": suit_histogram,
        }

    def _build_hand_levels(self) -> np.ndarray:
        """Return the fixed-size hand-level vector.

        Returns:
            A length-12 numpy array ordered exactly like the underlying env's
            current hand-level mapping.
        """

        hand_levels = np.zeros(HAND_LEVEL_COUNT, dtype=np.int16)
        current_levels = list(self.base_env.state.hand_levels.values())[:HAND_LEVEL_COUNT]
        if current_levels:
            hand_levels[: len(current_levels)] = np.asarray(current_levels, dtype=np.int16)
        return hand_levels

    def _build_boss_token(self) -> Dict[str, np.ndarray | np.generic]:
        """Encode boss-blind state into a compact token block.

        Returns:
            A dict with boss-active and boss-type fields.
        """

        state = self.base_env.state
        boss_type = int(state.active_boss_blind.value) if state.active_boss_blind else 0
        return {
            "boss_blind_active": np.asarray([int(state.boss_blind_active)], dtype=np.int8),
            "boss_blind_type": np.int8(boss_type),
        }

    def _build_joker_tokens(self) -> Dict[str, np.ndarray]:
        """Convert the joker row into padded token fields.

        Returns:
            Parallel arrays for joker IDs, empty-slot flags, and disabled-slot
            flags derived from the boss blind manager.
        """

        state = self.base_env.state
        joker_ids = np.full(MAX_JOKER_SLOTS, PAD_JOKER_ID, dtype=np.int16)
        is_empty = np.ones(MAX_JOKER_SLOTS, dtype=np.int8)
        is_disabled = np.zeros(MAX_JOKER_SLOTS, dtype=np.int8)

        disabled_count = 0
        if state.boss_blind_active and self.base_env.boss_blind_manager.active_blind:
            disabled_count = self.base_env.boss_blind_manager.get_disabled_joker_count()

        active_count = min(len(state.jokers), MAX_JOKER_SLOTS)
        for slot_idx, joker in enumerate(state.jokers[:MAX_JOKER_SLOTS]):
            joker_ids[slot_idx] = np.int16(joker.id)
            is_empty[slot_idx] = 0

        # The current simulator only exposes a disabled-count aggregate, so the
        # wrapper maps that count to the right-most occupied joker slots.
        for slot_idx in range(max(0, active_count - disabled_count), active_count):
            is_disabled[slot_idx] = 1

        return {
            "joker_id": joker_ids,
            "is_empty": is_empty,
            "is_disabled": is_disabled,
        }

    def _build_action_masks(self) -> Dict[str, np.ndarray | np.generic]:
        """Build factorized legality masks from the current selection state.

        Returns:
            A dict with card-selection availability, play/discard booleans, and
            current selection count.
        """

        state = self.base_env.state
        selectable_slots = np.zeros(MAX_HAND_SIZE, dtype=np.int8)

        if state.phase == Phase.PLAY:
            selectable_slots[: min(MAX_HAND_SIZE, len(state.hand_indexes))] = 1

        selected_slots = set(state.selected_cards)
        selected_count = len(selected_slots)
        play_allowed = int(self._is_play_allowed(selected_slots))
        discard_allowed = int(self._is_discard_allowed(selected_slots))

        return {
            "card_select_mask": selectable_slots,
            "play_allowed": np.asarray([play_allowed], dtype=np.int8),
            "discard_allowed": np.asarray([discard_allowed], dtype=np.int8),
            "selected_count": np.int8(selected_count),
        }

    def _normalize_arch_action(
        self, action: Dict[str, Any]
    ) -> Tuple[np.ndarray, ArchExecuteAction]:
        """Validate and normalize one structured wrapper action.

        Parameters:
            action: User- or policy-provided wrapper action payload.
        Returns:
            A tuple of `(selection_vector, execute_choice)`.
        """

        if not isinstance(action, dict):
            raise ValueError("wrapper action must be a dict with selection and execute")

        if "selection" not in action or "execute" not in action:
            raise ValueError("wrapper action requires both selection and execute keys")

        selection = np.asarray(action["selection"], dtype=np.int8)
        if selection.shape != (MAX_HAND_SIZE,):
            raise ValueError(f"selection must have shape ({MAX_HAND_SIZE},)")
        if np.any((selection != 0) & (selection != 1)):
            raise ValueError("selection entries must be binary")

        execute_value = action["execute"]
        if isinstance(execute_value, str):
            execute = ArchExecuteAction[execute_value.upper()]
        else:
            execute = ArchExecuteAction(int(execute_value))

        return selection, execute

    def _is_play_allowed(self, selected_slots: Iterable[int]) -> bool:
        """Check wrapper-level legality for `PLAY`.

        Parameters:
            selected_slots: The hand-slot indexes that would be played.
        Returns:
            `True` when the selected cards satisfy count rules and any active
            boss-blind play restrictions.
        """

        state = self.base_env.state
        if state.phase != Phase.PLAY:
            return False

        selected_slots = sorted(set(selected_slots))
        if not (1 <= len(selected_slots) <= 5):
            return False

        if not (state.boss_blind_active and self.base_env.boss_blind_manager.active_blind):
            return True

        selected_cards = []
        for slot_idx in selected_slots:
            if slot_idx >= len(state.hand_indexes):
                return False
            deck_idx = state.hand_indexes[slot_idx]
            selected_cards.append(state.deck[deck_idx])

        if not selected_cards:
            return False

        hand_type, _ = self.base_env.game._classify_hand(selected_cards)
        hand_name = hand_type.name.replace("_", " ").title()
        can_play, _ = self.base_env.boss_blind_manager.can_play_hand(
            selected_cards, hand_name
        )
        return can_play

    def _is_discard_allowed(self, selected_slots: Iterable[int]) -> bool:
        """Check wrapper-level legality for `DISCARD`.

        Parameters:
            selected_slots: The hand-slot indexes that would be discarded.
        Returns:
            `True` when at least one occupied slot is selected and discards
            remain in the current round.
        """

        state = self.base_env.state
        if state.phase != Phase.PLAY:
            return False

        selected_slots = set(selected_slots)
        if not selected_slots:
            return False
        if state.discards_left <= 0:
            return False
        return all(slot_idx < len(state.hand_indexes) for slot_idx in selected_slots)

    def _advance_to_play_phase(self) -> Tuple[float, bool, bool, Dict[str, Any]]:
        """Auto-resolve non-combat phases with deterministic default actions.

        Returns:
            A tuple of accumulated reward, termination flags, and debug info
            describing the automatic simulator actions that were taken.
        """

        if not self.auto_advance:
            return 0.0, False, False, {"auto_actions": [], "phase_infos": []}

        total_reward = 0.0
        terminated = False
        truncated = False
        auto_actions: List[int] = []
        phase_infos: List[Dict[str, Any]] = []

        for _ in range(16):
            if self.base_env.state.phase == Phase.PLAY or terminated or truncated:
                break

            action = self._default_phase_action()
            _, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            auto_actions.append(action)
            phase_infos.append(info)
        else:
            raise RuntimeError("failed to auto-advance wrapper into play phase")

        return total_reward, terminated, truncated, {
            "auto_actions": auto_actions,
            "phase_infos": phase_infos,
        }

    def _default_phase_action(self) -> int:
        """Choose the deterministic action used for non-combat phases.

        Returns:
            The flat simulator action ID used to move the env back toward the
            combat phase.
        """

        phase = self.base_env.state.phase
        if phase == Phase.BLIND_SELECT:
            return int(Action.SELECT_BLIND_BASE)
        if phase == Phase.SHOP:
            return int(Action.SHOP_END)
        if phase == Phase.PACK_OPEN:
            return int(Action.SKIP_PACK)
        raise ValueError(f"no default wrapper action for phase {phase}")

    def close(self) -> None:
        """Close the wrapped environment.

        Returns:
            `None`. The call is forwarded to the base env.
        """

        self.env.close()
