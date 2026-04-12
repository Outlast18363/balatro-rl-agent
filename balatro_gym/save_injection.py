"""Utilities for injecting parsed Balatro save blobs into ``BalatroEnv``.

This module intentionally accepts already-parsed Python dictionaries. Binary
``.jkr`` decoding is out of scope here and should be provided by a caller-side
parser callback.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple
import json
import re
import unicodedata
import zlib

from balatro_gym.balatro_env_2 import BalatroEnv, get_blind_chips
from balatro_gym.cards import Card, CardState, Edition, Enhancement, Rank, Seal, Suit
from balatro_gym.constants import Action, MAX_HAND_SIZE, Phase
from balatro_gym.scoring_engine import HandType
from balatro_gym.boss_blinds import BossBlindType, BOSS_BLINDS
from balatro_gym.jokers import JOKER_LIBRARY


SaveParser = Callable[[Any], Mapping[str, Any]]


RANK_MAP: Dict[str, Rank] = {
    "2": Rank.TWO,
    "3": Rank.THREE,
    "4": Rank.FOUR,
    "5": Rank.FIVE,
    "6": Rank.SIX,
    "7": Rank.SEVEN,
    "8": Rank.EIGHT,
    "9": Rank.NINE,
    "10": Rank.TEN,
    "T": Rank.TEN,
    "J": Rank.JACK,
    "Q": Rank.QUEEN,
    "K": Rank.KING,
    "A": Rank.ACE,
    "Jack": Rank.JACK,
    "Queen": Rank.QUEEN,
    "King": Rank.KING,
    "Ace": Rank.ACE,
}

SUIT_MAP: Dict[str, Suit] = {
    "Clubs": Suit.CLUBS,
    "Diamonds": Suit.DIAMONDS,
    "Hearts": Suit.HEARTS,
    "Spades": Suit.SPADES,
    "C": Suit.CLUBS,
    "D": Suit.DIAMONDS,
    "H": Suit.HEARTS,
    "S": Suit.SPADES,
}

HAND_NAME_TO_TYPE: Dict[str, HandType] = {
    "High Card": HandType.HIGH_CARD,
    "Pair": HandType.ONE_PAIR,
    "Two Pair": HandType.TWO_PAIR,
    "Three of a Kind": HandType.THREE_KIND,
    "Straight": HandType.STRAIGHT,
    "Flush": HandType.FLUSH,
    "Full House": HandType.FULL_HOUSE,
    "Four of a Kind": HandType.FOUR_KIND,
    "Straight Flush": HandType.STRAIGHT_FLUSH,
    "Five of a Kind": HandType.FIVE_KIND,
    "Flush House": HandType.FLUSH_HOUSE,
    "Flush Five": HandType.FLUSH_FIVE,
}

SAVE_BOSS_KEY_TO_ENUM: Dict[str, BossBlindType] = {
    "bl_hook": BossBlindType.THE_HOOK,
    "bl_wall": BossBlindType.THE_WALL,
    "bl_wheel": BossBlindType.THE_WHEEL,
    "bl_house": BossBlindType.THE_HOUSE,
    "bl_mark": BossBlindType.THE_MARK,
    "bl_fish": BossBlindType.THE_FISH,
    "bl_psychic": BossBlindType.THE_PSYCHIC,
    "bl_goad": BossBlindType.THE_GOAD,
    "bl_water": BossBlindType.THE_WATER,
    "bl_window": BossBlindType.THE_WINDOW,
    "bl_manacle": BossBlindType.THE_MANACLE,
    "bl_eye": BossBlindType.THE_EYE,
    "bl_mouth": BossBlindType.THE_MOUTH,
    "bl_plant": BossBlindType.THE_PLANT,
    "bl_serpent": BossBlindType.THE_SERPENT,
    "bl_pillar": BossBlindType.THE_PILLAR,
    "bl_needle": BossBlindType.THE_NEEDLE,
    "bl_head": BossBlindType.THE_HEAD,
    "bl_club": BossBlindType.THE_CLUB,
    "bl_tooth": BossBlindType.THE_TOOTH,
    "bl_flint": BossBlindType.THE_FLINT,
    "bl_oxide": BossBlindType.THE_OXIDE,
    "bl_arm": BossBlindType.THE_ARM,
    "bl_violet_vessel": BossBlindType.THE_VIOLET,
    "bl_verdant_leaf": BossBlindType.THE_VERDANT,
    "bl_amber_acorn": BossBlindType.THE_AMBER,
    "bl_crimson_heart": BossBlindType.THE_CRIMSON,
    "bl_cerulean_bell": BossBlindType.THE_CERULEAN,
}

ROUND_NAME_TO_INDEX = {
    "Small": 1,
    "Big": 2,
    "Boss": 3,
}

CARD_CENTER_TO_ENHANCEMENT: Dict[str, Enhancement] = {
    "c_base": Enhancement.NONE,
    "m_bonus": Enhancement.BONUS,
    "m_mult": Enhancement.MULT,
    "m_wild": Enhancement.WILD,
    "m_glass": Enhancement.GLASS,
    "m_steel": Enhancement.STEEL,
    "m_stone": Enhancement.STONE,
    "m_gold": Enhancement.GOLD,
    "m_lucky": Enhancement.LUCKY,
}

CARD_EDITION_KEY_TO_ENUM: Dict[str, Edition] = {
    "e_base": Edition.NONE,
    "base": Edition.NONE,
    "e_foil": Edition.FOIL,
    "foil": Edition.FOIL,
    "e_holo": Edition.HOLOGRAPHIC,
    "holo": Edition.HOLOGRAPHIC,
    "holographic": Edition.HOLOGRAPHIC,
    "e_polychrome": Edition.POLYCHROME,
    "polychrome": Edition.POLYCHROME,
    "e_negative": Edition.NEGATIVE,
    "negative": Edition.NEGATIVE,
}

CARD_SEAL_TO_ENUM: Dict[str, Seal] = {
    "": Seal.NONE,
    "Gold": Seal.GOLD,
    "Red": Seal.RED,
    "Blue": Seal.BLUE,
    "Purple": Seal.PURPLE,
}

GYM_STATE_FIELDS = (
    "ante",
    "round",
    "phase",
    "chips_needed",
    "chips_scored",
    "round_chips_scored",
    "money",
    "deck",
    "hand_indexes",
    "selected_cards",
    "hands_left",
    "discards_left",
    "hand_size",
    "jokers",
    "consumables",
    "vouchers",
    "joker_slots",
    "consumable_slots",
    "shop_inventory",
    "shop_reroll_cost",
    "hands_played_total",
    "hands_played_ante",
    "best_hand_this_ante",
    "jokers_sold",
    "hand_levels",
    "card_states",
    "next_boss_blind",
    "active_boss_blind",
    "boss_blind_active",
    "face_down_cards",
    "force_draw_count",
)


def inject_save_into_balatro_env(
    save_source: Any,
    *,
    env: Optional[BalatroEnv] = None,
    seed: int = 0,
    parser: Optional[SaveParser] = None,
    validate: bool = True,
) -> Tuple[BalatroEnv, Dict[str, Any]]:
    """Inject a parsed save blob into ``BalatroEnv``.

    Args:
        save_source: Parsed save dict, JSON path, or any payload accepted by
            ``parser``.
        env: Existing environment. If omitted, creates a new one.
        seed: Seed for deterministic reset before applying injected state.
        parser: Optional callback converting ``save_source`` into a dict.
        validate: Whether to run post-injection checks.

    Returns:
        ``(env, report)``.
    """
    save_blob = _resolve_save_blob(save_source, parser)

    if env is None:
        env = BalatroEnv(seed=seed)
    env.reset(seed=seed)

    report: Dict[str, Any] = {
        "applied_fields": [],
        "missing_in_save": [],
        "ignored_from_save": [],
        "ignored_from_save_total": 0,
        "source_paths": {},
        "skipped_fields": {},
        "warnings": [],
        "validation": {},
    }
    consumed_paths: set[str] = set()

    _apply_injection(env, save_blob, report, consumed_paths)
    _sync_runtime_state(env)
    if validate:
        report["validation"] = _validate_injected_env(env)

    all_save_paths = _collect_paths(save_blob, max_depth=3)
    ignored = sorted(p for p in all_save_paths if not _is_consumed_path(p, consumed_paths))
    report["ignored_from_save_total"] = len(ignored)
    report["ignored_from_save"] = ignored[:200]
    report["applied_fields"] = sorted(set(report["applied_fields"]))
    report["missing_in_save"] = sorted(set(report["missing_in_save"]))
    return env, report


def load_save_json(path: str | Path) -> Dict[str, Any]:
    """Load a parsed JSON save file."""
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise TypeError("Save JSON root must be an object/dict.")
    return payload


def load_save_jkr(path: str | Path) -> Dict[str, Any]:
    """Load a Balatro `.jkr` save file (raw-deflate Lua table)."""
    raw = Path(path).read_bytes()
    # Balatro .jkr here is stored as raw deflate (no zlib/gzip headers).
    try:
        decoded = zlib.decompress(raw, -15)
    except zlib.error as exc:
        raise ValueError(f"Failed to decompress jkr save: {exc}") from exc
    text = decoded.decode("utf-8")
    return _parse_lua_return_table(text)


def _resolve_save_blob(save_source: Any, parser: Optional[SaveParser]) -> Dict[str, Any]:
    if parser is not None:
        parsed = parser(save_source)
        if not isinstance(parsed, Mapping):
            raise TypeError("`parser` must return a mapping/dict.")
        return dict(parsed)

    if isinstance(save_source, Mapping):
        return dict(save_source)

    if isinstance(save_source, (str, Path)):
        path = Path(save_source)
        if path.suffix.lower() == ".jkr":
            return load_save_jkr(path)
        return load_save_json(path)

    raise TypeError("Unsupported `save_source`. Provide dict, JSON/.jkr path, or parser.")


_LUA_TOKEN_PATTERN = re.compile(
    r"""
    \s*(?:
        (?P<brace_open>\{)|
        (?P<brace_close>\})|
        (?P<bracket_open>\[)|
        (?P<bracket_close>\])|
        (?P<comma>,)|
        (?P<eq>=)|
        (?P<string>"(?:\\.|[^"\\])*")|
        (?P<number>[+-]?\d+(?:\.\d+)?)|
        (?P<ident>true|false|nil|[A-Za-z_][A-Za-z0-9_]*)
    )
    """,
    re.VERBOSE,
)


class _LuaTableParser:
    """Minimal Lua table parser for Balatro save payloads."""

    def __init__(self, src: str):
        self.src = src.strip()
        if self.src.startswith("return "):
            self.src = self.src[len("return ") :]
        self.tokens = list(self._tokenize(self.src))
        self.idx = 0

    def parse(self) -> Any:
        value = self._parse_value()
        if self.idx != len(self.tokens):
            raise ValueError("Unexpected trailing tokens in Lua save payload.")
        return value

    def _tokenize(self, text: str):
        pos = 0
        while pos < len(text):
            m = _LUA_TOKEN_PATTERN.match(text, pos)
            if not m:
                snippet = text[pos : pos + 80]
                raise ValueError(f"Could not tokenize Lua payload near: {snippet!r}")
            pos = m.end()
            kind = m.lastgroup
            value = m.group(kind)
            yield (kind, value)

    def _peek(self) -> Optional[Tuple[str, str]]:
        if self.idx >= len(self.tokens):
            return None
        return self.tokens[self.idx]

    def _consume(self, expected_kind: Optional[str] = None) -> Tuple[str, str]:
        tok = self._peek()
        if tok is None:
            raise ValueError("Unexpected end of Lua payload.")
        if expected_kind is not None and tok[0] != expected_kind:
            raise ValueError(f"Expected token {expected_kind}, got {tok[0]}:{tok[1]}")
        self.idx += 1
        return tok

    def _parse_value(self) -> Any:
        tok = self._peek()
        if tok is None:
            raise ValueError("Unexpected end while parsing Lua value.")
        kind, value = tok
        if kind == "brace_open":
            return self._parse_table()
        if kind == "string":
            self._consume("string")
            return _decode_lua_string(value)
        if kind == "number":
            self._consume("number")
            return float(value) if "." in value else int(value)
        if kind == "ident":
            self._consume("ident")
            if value == "true":
                return True
            if value == "false":
                return False
            if value == "nil":
                return None
            # Fallback for bare identifiers (rare in this payload style).
            return value
        raise ValueError(f"Unexpected token while parsing value: {kind}:{value}")

    def _parse_table(self) -> Any:
        self._consume("brace_open")
        entries: list[tuple[Any, Any]] = []
        array_values: list[Any] = []
        saw_explicit_key = False
        next_array_index = 1

        while True:
            tok = self._peek()
            if tok is None:
                raise ValueError("Unclosed Lua table.")
            kind, _ = tok
            if kind == "brace_close":
                self._consume("brace_close")
                break
            if kind == "comma":
                self._consume("comma")
                continue

            key, value, had_key = self._parse_field(next_array_index)
            if had_key:
                saw_explicit_key = True
                entries.append((key, value))
            else:
                array_values.append(value)
                next_array_index += 1

            tok = self._peek()
            if tok and tok[0] == "comma":
                self._consume("comma")

        if not saw_explicit_key:
            return array_values

        result: Dict[Any, Any] = {k: v for k, v in entries}
        # Preserve implicit array items if mixed tables appear.
        for idx, value in enumerate(array_values, start=1):
            result[idx] = value
        return result

    def _parse_field(self, next_array_index: int) -> Tuple[Any, Any, bool]:
        tok = self._peek()
        if tok is None:
            raise ValueError("Unexpected end while parsing table field.")
        kind, value = tok

        # [key] = value
        if kind == "bracket_open":
            self._consume("bracket_open")
            key = self._parse_value()
            self._consume("bracket_close")
            self._consume("eq")
            val = self._parse_value()
            return key, val, True

        # bare_ident = value
        if kind == "ident":
            nxt = self.tokens[self.idx + 1] if (self.idx + 1) < len(self.tokens) else None
            if nxt and nxt[0] == "eq":
                self._consume("ident")
                self._consume("eq")
                val = self._parse_value()
                return value, val, True

        # array-style item
        val = self._parse_value()
        return next_array_index, val, False


def _decode_lua_string(token: str) -> str:
    # token includes quotes; json parser safely handles common escapes.
    return json.loads(token)


def _normalize_lua_keys(obj: Any) -> Any:
    if isinstance(obj, dict):
        normalized: Dict[Any, Any] = {}
        for key, value in obj.items():
            if isinstance(key, (int, float)):
                key_out = str(int(key))
            else:
                key_out = str(key)
            normalized[key_out] = _normalize_lua_keys(value)
        return normalized
    if isinstance(obj, list):
        return [_normalize_lua_keys(x) for x in obj]
    return obj


def _parse_lua_return_table(lua_text: str) -> Dict[str, Any]:
    parsed = _LuaTableParser(lua_text).parse()
    normalized = _normalize_lua_keys(parsed)
    if not isinstance(normalized, dict):
        raise TypeError("Parsed Lua payload root is not a dict/table.")
    return normalized


def _apply_injection(
    env: BalatroEnv,
    save_blob: Mapping[str, Any],
    report: Dict[str, Any],
    consumed_paths: set[str],
) -> None:
    hand_card_blobs = _iter_area_cards(save_blob, "hand", consumed_paths, report)
    draw_pile_blobs = _iter_area_cards(save_blob, "deck", consumed_paths, report)

    hand_cards = [c for c in (_parse_playing_card(card, report) for card in hand_card_blobs) if c is not None]
    draw_pile_cards = [c for c in (_parse_playing_card(card, report) for card in draw_pile_blobs) if c is not None]
    full_deck = hand_cards + draw_pile_cards
    hand_indexes = list(range(len(hand_cards)))

    hand_levels, hand_play_counts = _parse_hand_levels(save_blob, report, consumed_paths)
    ante = _read_int(
        save_blob,
        (("GAME", "round_resets", "ante"), ("GAME", "ante")),
        default=1,
        field_name="ante",
        report=report,
        consumed_paths=consumed_paths,
    )
    blind_key, blind_config_key = _infer_blind_keys(save_blob, consumed_paths)

    state = env.state
    state.ante = ante
    _mark_applied(report, "ante")

    state.round = _infer_round_index(save_blob, report, consumed_paths)
    _mark_applied(report, "round")

    state.phase = _infer_phase(save_blob, report, consumed_paths)
    _mark_applied(report, "phase")

    state.chips_needed = _infer_chips_needed(save_blob, ante, consumed_paths)
    _mark_applied(report, "chips_needed")

    state.chips_scored = _read_int(
        save_blob,
        (("GAME", "chips"),),
        default=0,
        field_name="chips_scored",
        report=report,
        consumed_paths=consumed_paths,
    )
    _mark_applied(report, "chips_scored")

    state.round_chips_scored = _read_int(
        save_blob,
        (("GAME", "current_round", "current_hand", "chip_total"),),
        default=0,
        field_name="round_chips_scored",
        report=report,
        consumed_paths=consumed_paths,
    )
    _mark_applied(report, "round_chips_scored")

    state.money = _read_int(
        save_blob,
        (("GAME", "dollars"),),
        default=0,
        field_name="money",
        report=report,
        consumed_paths=consumed_paths,
    )
    _mark_applied(report, "money")

    state.deck = full_deck
    state.hand_indexes = hand_indexes
    state.selected_cards = []
    _mark_applied(report, "deck")
    _mark_applied(report, "hand_indexes")
    _mark_applied(report, "selected_cards")

    state.hands_left = _read_int(
        save_blob,
        (("GAME", "current_round", "hands_left"),),
        default=4,
        field_name="hands_left",
        report=report,
        consumed_paths=consumed_paths,
    )
    _mark_applied(report, "hands_left")

    state.discards_left = _read_int(
        save_blob,
        (("GAME", "current_round", "discards_left"),),
        default=3,
        field_name="discards_left",
        report=report,
        consumed_paths=consumed_paths,
    )
    _mark_applied(report, "discards_left")

    hand_size_fallback = _read_int(
        save_blob,
        (("GAME", "starting_params", "hand_size"),),
        default=MAX_HAND_SIZE,
        field_name="hand_size(starting_params)",
        report=report,
        consumed_paths=consumed_paths,
    )
    state.hand_size = _get_total_slots(save_blob, "hand", hand_size_fallback, consumed_paths, report)
    _mark_applied(report, "hand_size")

    state.jokers = _parse_jokers(save_blob, report, consumed_paths)
    _mark_applied(report, "jokers")

    state.consumables = _parse_named_collection(save_blob, "consumeables", report, consumed_paths)
    _mark_applied(report, "consumables")

    state.vouchers = _parse_named_collection(save_blob, "vouchers", report, consumed_paths)
    _mark_applied(report, "vouchers")

    joker_slot_fallback = _read_int(
        save_blob,
        (("GAME", "max_jokers"), ("GAME", "starting_params", "joker_slots")),
        default=5,
        field_name="joker_slots",
        report=report,
        consumed_paths=consumed_paths,
    )
    state.joker_slots = _get_total_slots(save_blob, "jokers", joker_slot_fallback, consumed_paths, report)
    _mark_applied(report, "joker_slots")

    consumable_slot_fallback = _read_int(
        save_blob,
        (("GAME", "starting_params", "consumable_slots"),),
        default=2,
        field_name="consumable_slots",
        report=report,
        consumed_paths=consumed_paths,
    )
    state.consumable_slots = _get_total_slots(
        save_blob, "consumeables", consumable_slot_fallback, consumed_paths, report
    )
    _mark_applied(report, "consumable_slots")

    state.shop_inventory = []
    _mark_applied(report, "shop_inventory")
    state.shop_reroll_cost = _read_int(
        save_blob,
        (("GAME", "current_round", "reroll_cost"),),
        default=0,
        field_name="shop_reroll_cost",
        report=report,
        consumed_paths=consumed_paths,
    )
    _mark_applied(report, "shop_reroll_cost")

    state.hands_played_total = _read_int(
        save_blob,
        (("GAME", "hands_played"),),
        default=0,
        field_name="hands_played_total",
        report=report,
        consumed_paths=consumed_paths,
    )
    _mark_applied(report, "hands_played_total")
    state.hands_played_ante = _read_int(
        save_blob,
        (("GAME", "current_round", "hands_played"),),
        default=0,
        field_name="hands_played_ante",
        report=report,
        consumed_paths=consumed_paths,
    )
    _mark_applied(report, "hands_played_ante")

    state.best_hand_this_ante = 0
    state.jokers_sold = 0
    _mark_applied(report, "best_hand_this_ante")
    _mark_applied(report, "jokers_sold")

    full_card_blobs = hand_card_blobs + draw_pile_blobs
    state.hand_levels = hand_levels.copy()
    _mark_applied(report, "hand_levels")
    card_states: Dict[int, CardState] = {}
    for deck_idx, card_blob in enumerate(full_card_blobs):
        card_states[deck_idx] = _parse_card_state(card_blob, deck_idx, report)
    state.card_states = card_states
    _mark_applied(report, "card_states")

    state.next_boss_blind = None
    boss_choice_key = _read_value(
        save_blob,
        (("GAME", "round_resets", "blind_choices", "Boss"),),
        field_name="next_boss_blind",
        report=report,
        consumed_paths=consumed_paths,
    )
    if boss_choice_key and boss_choice_key in SAVE_BOSS_KEY_TO_ENUM:
        state.next_boss_blind = SAVE_BOSS_KEY_TO_ENUM[boss_choice_key]
    _mark_applied(report, "next_boss_blind")

    state.active_boss_blind = None
    if _is_in_blind(save_blob) and blind_key == "boss" and blind_config_key in SAVE_BOSS_KEY_TO_ENUM:
        state.active_boss_blind = SAVE_BOSS_KEY_TO_ENUM[blind_config_key]
    state.boss_blind_active = state.active_boss_blind is not None
    _mark_applied(report, "active_boss_blind")
    _mark_applied(report, "boss_blind_active")

    state.face_down_cards = []
    state.force_draw_count = None
    _mark_applied(report, "face_down_cards")
    _mark_applied(report, "force_draw_count")

    env.engine.hand_levels = hand_levels.copy()
    env.engine.hand_play_counts = hand_play_counts.copy()

    env.game.deck = full_deck.copy()
    env.game.hand_indexes = hand_indexes.copy()
    env.game.highlighted_indexes = []
    env.game.round_hands = state.hands_left
    env.game.round_discards = state.discards_left
    env.game.round_score = state.round_chips_scored
    env.game.hand_size = state.hand_size
    env.game.blind_index = max(0, min(2, state.round - 1))
    env.game.blinds[env.game.blind_index] = state.chips_needed
    env.shop = None

    if state.active_boss_blind and state.active_boss_blind in BOSS_BLINDS:
        env.boss_blind_manager.active_blind = BOSS_BLINDS[state.active_boss_blind]
        env.boss_blind_manager.blind_state = {
            "played_hand_types": set(),
            "played_cards": set(),
            "hands_played": 0,
            "cards_required": 5,
            "first_hand": True,
            "disabled_joker_slots": 0,
        }
    else:
        env.boss_blind_manager.deactivate()

    for field_name in GYM_STATE_FIELDS:
        if field_name not in report["applied_fields"]:
            report["missing_in_save"].append(field_name)


def _sync_runtime_state(env: BalatroEnv) -> None:
    env._sync_state_to_game()
    env.game.round_score = env.state.round_chips_scored


def _validate_injected_env(env: BalatroEnv) -> Dict[str, Any]:
    validation: Dict[str, Any] = {"ok": True, "checks": {}}
    try:
        obs = env._get_observation()
        validation["checks"]["base_observation_keys"] = sorted(obs.keys())
        validation["checks"]["action_mask_size"] = int(len(obs["action_mask"]))
        validation["checks"]["action_mask_expected"] = int(Action.ACTION_SPACE_SIZE)
        validation["checks"]["phase"] = int(obs["phase"])
        validation["checks"]["step_ready"] = bool(len(obs["action_mask"]) == Action.ACTION_SPACE_SIZE)
    except Exception as exc:  # pragma: no cover - defensive path
        validation["ok"] = False
        validation["checks"]["base_observation_error"] = str(exc)
        return validation

    try:
        from cs590_env.wrapper import BalatroPhaseWrapper

        wrapped = BalatroPhaseWrapper(env)
        wrapped_obs = wrapped._get_phase_observation()
        validation["checks"]["wrapper_phase"] = int(wrapped_obs["phase"])
        validation["checks"]["wrapper_action_mask_size"] = int(len(wrapped_obs["action_mask"]))
    except Exception as exc:  # pragma: no cover - optional integration check
        validation["ok"] = False
        validation["checks"]["wrapper_read_error"] = str(exc)
    return validation


def _normalize_name(name: Any) -> str:
    if not name:
        return ""
    ascii_name = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    return " ".join(ascii_name.split()).casefold()


JOKER_EXACT_NAME_TO_INFO = {joker.name: joker for joker in JOKER_LIBRARY}
JOKER_NORMALIZED_NAME_TO_INFO = {_normalize_name(joker.name): joker for joker in JOKER_LIBRARY}


def _iter_area_cards(
    save_blob: Mapping[str, Any],
    area_name: str,
    consumed_paths: set[str],
    report: Dict[str, Any],
) -> list[Mapping[str, Any]]:
    cards = _read_value(
        save_blob,
        (("cardAreas", area_name, "cards"),),
        field_name=f"cardAreas.{area_name}.cards",
        report=report,
        consumed_paths=consumed_paths,
    )
    if cards is None:
        return []
    if isinstance(cards, dict):
        out = []
        for payload in cards.values():
            if isinstance(payload, Mapping):
                out.append(payload)
        return out
    if isinstance(cards, list):
        return [item for item in cards if isinstance(item, Mapping)]
    report["warnings"].append(f"Unexpected cards container for area `{area_name}`.")
    return []


def _get_total_slots(
    save_blob: Mapping[str, Any],
    area_name: str,
    fallback: int,
    consumed_paths: set[str],
    report: Dict[str, Any],
) -> int:
    value = _read_value(
        save_blob,
        (("cardAreas", area_name, "config", "card_limits", "total_slots"),),
        field_name=f"cardAreas.{area_name}.config.card_limits.total_slots",
        report=report,
        consumed_paths=consumed_paths,
    )
    if value is None:
        return int(fallback)
    try:
        return int(value)
    except (TypeError, ValueError):
        report["warnings"].append(f"Invalid total_slots for area `{area_name}`; using fallback {fallback}.")
        return int(fallback)


def _parse_playing_card(card_blob: Mapping[str, Any], report: Dict[str, Any]) -> Optional[Card]:
    base = card_blob.get("base")
    if not isinstance(base, Mapping):
        report["warnings"].append("Skipping card without `base` payload.")
        return None
    rank_key = str(base.get("value"))
    suit_key = str(base.get("suit"))
    rank = RANK_MAP.get(rank_key)
    suit = SUIT_MAP.get(suit_key)
    if rank is None or suit is None:
        report["warnings"].append(f"Skipping unrecognized card rank/suit: value={rank_key}, suit={suit_key}")
        return None
    return Card(rank=rank, suit=suit)


def _parse_card_state(card_blob: Mapping[str, Any], card_id: int, report: Dict[str, Any]) -> CardState:
    state = CardState(card_id=card_id)
    center = card_blob.get("save_fields", {}).get("center", "c_base")
    state.enhancement = CARD_CENTER_TO_ENHANCEMENT.get(center, Enhancement.NONE)
    if center not in CARD_CENTER_TO_ENHANCEMENT:
        report["warnings"].append(f"Unknown enhancement center `{center}` -> Enhancement.NONE")

    edition_blob = card_blob.get("edition")
    if edition_blob in (None, {}, "", "e_base"):
        edition_key = "e_base"
    elif isinstance(edition_blob, Mapping):
        edition_key = edition_blob.get("key") or edition_blob.get("type")
    else:
        edition_key = str(edition_blob)
    state.edition = CARD_EDITION_KEY_TO_ENUM.get(str(edition_key), Edition.NONE)
    if str(edition_key) not in CARD_EDITION_KEY_TO_ENUM:
        report["warnings"].append(f"Unknown card edition `{edition_blob}` -> Edition.NONE")

    seal_blob = card_blob.get("seal")
    if seal_blob in (None, {}, ""):
        seal_key = ""
    elif isinstance(seal_blob, Mapping):
        seal_key = seal_blob.get("type") or seal_blob.get("key") or ""
    else:
        seal_key = str(seal_blob)
    state.seal = CARD_SEAL_TO_ENUM.get(str(seal_key), Seal.NONE)
    if str(seal_key) not in CARD_SEAL_TO_ENUM:
        report["warnings"].append(f"Unknown card seal `{seal_blob}` -> Seal.NONE")
    return state


def _parse_hand_levels(
    save_blob: Mapping[str, Any],
    report: Dict[str, Any],
    consumed_paths: set[str],
) -> Tuple[Dict[HandType, int], Dict[HandType, int]]:
    hand_levels: Dict[HandType, int] = {}
    hand_play_counts: Dict[HandType, int] = {}
    for hand_name, hand_type in HAND_NAME_TO_TYPE.items():
        lvl = _read_int(
            save_blob,
            (("GAME", "hands", hand_name, "level"),),
            default=0,
            field_name=f"hand_levels.{hand_name}",
            report=report,
            consumed_paths=consumed_paths,
            missing_to_report=False,
        )
        played = _read_int(
            save_blob,
            (("GAME", "hands", hand_name, "played"),),
            default=0,
            field_name=f"hand_play_counts.{hand_name}",
            report=report,
            consumed_paths=consumed_paths,
            missing_to_report=False,
        )
        hand_levels[hand_type] = lvl
        hand_play_counts[hand_type] = played
    return hand_levels, hand_play_counts


def _parse_jokers(
    save_blob: Mapping[str, Any],
    report: Dict[str, Any],
    consumed_paths: set[str],
) -> list[Any]:
    jokers = []
    for joker_blob in _iter_area_cards(save_blob, "jokers", consumed_paths, report):
        candidates = [
            joker_blob.get("ability", {}).get("name"),
            joker_blob.get("label"),
        ]
        joker_info = None
        for name in candidates:
            if name in JOKER_EXACT_NAME_TO_INFO:
                joker_info = JOKER_EXACT_NAME_TO_INFO[name]
                break
            normalized_name = _normalize_name(name)
            if normalized_name in JOKER_NORMALIZED_NAME_TO_INFO:
                joker_info = JOKER_NORMALIZED_NAME_TO_INFO[normalized_name]
                break
        if joker_info is None:
            report["warnings"].append(f"Skipping unmapped joker from save candidates={candidates}")
            report["skipped_fields"].setdefault("jokers", []).append(str(candidates))
            continue
        jokers.append(joker_info)
    return jokers


def _parse_named_collection(
    save_blob: Mapping[str, Any],
    area_name: str,
    report: Dict[str, Any],
    consumed_paths: set[str],
) -> list[str]:
    names: list[str] = []
    for item_blob in _iter_area_cards(save_blob, area_name, consumed_paths, report):
        name = item_blob.get("ability", {}).get("name") or item_blob.get("label")
        if not name:
            report["warnings"].append(f"Skipping nameless item in area `{area_name}`")
            report["skipped_fields"].setdefault(area_name, []).append("missing_name")
            continue
        names.append(str(name))
    return names


def _infer_round_index(
    save_blob: Mapping[str, Any],
    report: Dict[str, Any],
    consumed_paths: set[str],
) -> int:
    blind_name = _read_value(
        save_blob,
        (("BLIND", "name"),),
        field_name="BLIND.name",
        report=report,
        consumed_paths=consumed_paths,
        missing_to_report=False,
    ) or ""
    if isinstance(blind_name, str):
        if blind_name.startswith("Small"):
            return 1
        if blind_name.startswith("Big"):
            return 2
        if blind_name.startswith("Boss"):
            return 3

    blind_on_deck = _read_value(
        save_blob,
        (("GAME", "blind_on_deck"),),
        field_name="GAME.blind_on_deck",
        report=report,
        consumed_paths=consumed_paths,
        missing_to_report=False,
    )
    if blind_on_deck in ROUND_NAME_TO_INDEX:
        return ROUND_NAME_TO_INDEX[str(blind_on_deck)]

    blind_order = _read_value(
        save_blob,
        (("GAME", "round_resets", "blind", "order"),),
        field_name="GAME.round_resets.blind.order",
        report=report,
        consumed_paths=consumed_paths,
        missing_to_report=False,
    )
    if blind_order in (1, 2, 3):
        return int(blind_order)
    return 1


def _infer_phase(
    save_blob: Mapping[str, Any],
    report: Dict[str, Any],
    consumed_paths: set[str],
) -> Phase:
    if _is_in_blind(save_blob, report=report, consumed_paths=consumed_paths):
        return Phase.PLAY
    # Safe default if current shop inventory cannot be reconstructed.
    return Phase.BLIND_SELECT


def _blind_config_to_key(config_key: str) -> str:
    if config_key == "bl_small":
        return "small"
    if config_key == "bl_big":
        return "big"
    return "boss"


def _infer_blind_keys(save_blob: Mapping[str, Any], consumed_paths: set[str]) -> Tuple[str, str]:
    if _is_in_blind(save_blob):
        config_key = _get_path_value(save_blob, ("BLIND", "config_blind"), consumed_paths=consumed_paths)
        if isinstance(config_key, str) and config_key:
            return _blind_config_to_key(config_key), config_key

    config_key = _get_path_value(save_blob, ("GAME", "round_resets", "blind", "key"), consumed_paths=consumed_paths)
    if isinstance(config_key, str) and config_key:
        return _blind_config_to_key(config_key), config_key

    blind_on_deck = _get_path_value(save_blob, ("GAME", "blind_on_deck"), consumed_paths=consumed_paths)
    if blind_on_deck == "Small":
        return "small", "bl_small"
    if blind_on_deck == "Big":
        return "big", "bl_big"
    if blind_on_deck == "Boss":
        boss_key = _get_path_value(
            save_blob,
            ("GAME", "round_resets", "blind_choices", "Boss"),
            consumed_paths=consumed_paths,
        )
        if isinstance(boss_key, str) and boss_key:
            return "boss", boss_key
        return "boss", "bl_violet_vessel"
    return "small", "bl_small"


def _infer_chips_needed(save_blob: Mapping[str, Any], ante: int, consumed_paths: set[str]) -> int:
    blind_target = _get_path_value(save_blob, ("BLIND", "chips"), consumed_paths=consumed_paths)
    try:
        blind_target_int = int(blind_target or 0)
    except (TypeError, ValueError):
        blind_target_int = 0
    if blind_target_int > 0:
        return blind_target_int
    blind_key, _ = _infer_blind_keys(save_blob, consumed_paths)
    return int(get_blind_chips(ante, blind_key))


def _is_in_blind(
    save_blob: Mapping[str, Any],
    *,
    report: Optional[Dict[str, Any]] = None,
    consumed_paths: Optional[set[str]] = None,
) -> bool:
    value = _get_path_value(save_blob, ("BLIND", "in_blind"), consumed_paths=consumed_paths)
    if value is None and report is not None:
        report["missing_in_save"].append("BLIND.in_blind")
    return bool(value)


def _read_int(
    save_blob: Mapping[str, Any],
    candidate_paths: Sequence[Sequence[str]],
    *,
    default: int,
    field_name: str,
    report: Dict[str, Any],
    consumed_paths: set[str],
    missing_to_report: bool = True,
) -> int:
    value = _read_value(
        save_blob,
        candidate_paths,
        field_name=field_name,
        report=report,
        consumed_paths=consumed_paths,
        missing_to_report=missing_to_report,
    )
    if value is None:
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        report["warnings"].append(f"Invalid integer value for `{field_name}`; using default {default}.")
        report["skipped_fields"][field_name] = f"invalid_int:{value}"
        return int(default)


def _read_value(
    save_blob: Mapping[str, Any],
    candidate_paths: Sequence[Sequence[str]],
    *,
    field_name: str,
    report: Dict[str, Any],
    consumed_paths: set[str],
    missing_to_report: bool = True,
) -> Any:
    for path in candidate_paths:
        found, value = _get_path(save_blob, path)
        if found:
            path_str = ".".join(path)
            consumed_paths.add(path_str)
            report["source_paths"][field_name] = path_str
            return value
    if missing_to_report:
        report["missing_in_save"].append(field_name)
    return None


def _get_path(save_blob: Mapping[str, Any], path: Sequence[str]) -> Tuple[bool, Any]:
    cur: Any = save_blob
    for key in path:
        if not isinstance(cur, Mapping) or key not in cur:
            return False, None
        cur = cur[key]
    return True, cur


def _get_path_value(
    save_blob: Mapping[str, Any],
    path: Sequence[str],
    *,
    consumed_paths: Optional[set[str]] = None,
) -> Any:
    found, value = _get_path(save_blob, path)
    if found and consumed_paths is not None:
        consumed_paths.add(".".join(path))
    return value if found else None


def _mark_applied(report: Dict[str, Any], field_name: str) -> None:
    report["applied_fields"].append(field_name)


def _collect_paths(obj: Any, prefix: str = "", max_depth: int = 3) -> set[str]:
    if max_depth <= 0:
        return {prefix} if prefix else set()
    paths: set[str] = set()
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            key_str = str(key)
            current = key_str if not prefix else f"{prefix}.{key_str}"
            paths.add(current)
            paths.update(_collect_paths(value, current, max_depth - 1))
        return paths
    if isinstance(obj, list):
        for idx, value in enumerate(obj):
            current = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            paths.add(current)
            paths.update(_collect_paths(value, current, max_depth - 1))
    return paths


def _is_consumed_path(path: str, consumed_paths: Iterable[str]) -> bool:
    for consumed in consumed_paths:
        if path == consumed or path.startswith(f"{consumed}."):
            return True
    return False


def load_snapshot_pool(save_dir: str = "save_files", seed: int = 42) -> list[dict]:
    """Load .jkr save files and convert them into env snapshots.

    Each ``.jkr`` file is injected into a fresh ``BalatroEnv`` and its full
    state is captured via ``save_state()``.  The resulting list of snapshot
    dicts can be passed directly to ``PooledCombatEnv`` or
    ``make_pooled_combat_env``.

    Args:
        save_dir: Root save directory. Snapshots are loaded only from the
            ``combat`` subdirectory (``<save_dir>/combat``).
        seed: Seed forwarded to ``inject_save_into_balatro_env``.

    Returns:
        Non-empty list of snapshot dicts.

    Raises:
        FileNotFoundError: If ``<save_dir>/combat`` contains no ``.jkr`` files.
    """
    pool: list[dict] = []
    combat_dir = Path(save_dir) / "combat"
    for jkr_path in sorted(combat_dir.glob("*.jkr")):
        env, _report = inject_save_into_balatro_env(jkr_path, seed=seed)
        pool.append(env.save_state())
    if not pool:
        raise FileNotFoundError(f"No .jkr files found in {combat_dir}/")
    return pool

