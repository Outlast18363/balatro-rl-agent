"""cs590_env/util.py — Human-readable stdout dumps of combat observations."""

from __future__ import annotations

import sys
from enum import IntEnum
from typing import Any, TextIO

import numpy as np

from balatro_gym.cards import Edition, Enhancement, Rank, Seal, Suit
from balatro_gym.scoring_engine import HandType

from cs590_env.schema import (
    GamePhase,
    MAX_HAND_SIZE,
    NUM_HAND_TYPES,
    NUM_RANKS,
)

_HAND_TYPE_NAMES = [HandType(i).name for i in range(NUM_HAND_TYPES)]

# Hand table: 1 → ✅, 0 → ❌ (fixed width for alignment in monospace fonts).
_ON = "✅"
_OFF = "❌"


def _bool_cell(v: int) -> str:
    return _ON if int(v) else _OFF


def _format_card_id(card_id: int) -> str:
    """Decode 0–51 using the same mapping as ``Card.__int__``."""
    if card_id < 0:
        return "PAD"
    if card_id > 51:
        return f"?id={card_id}"
    suit = int(card_id % 4)
    rank = int(card_id // 4 + 2)
    try:
        r = Rank(rank)
        s = Suit(suit)
        return f"{r.short}{s.symbol()}"
    except ValueError:
        return f"?id={card_id}"


def _enum_name(enum_cls: type[IntEnum], value: int) -> str:
    try:
        return enum_cls(int(value)).name
    except ValueError:
        return f"UNKNOWN({int(value)})"


def _batch_size(obs: dict[str, Any]) -> int | None:
    """Return leading batch size ``N`` if obs is vectorized, else ``None``."""
    ph = np.asarray(obs["phase"])
    if ph.ndim == 0:
        return None
    return int(ph.shape[0])


def _get_row(obs: dict[str, Any], key: str, env_index: int) -> np.ndarray:
    """Slice batch dimension ``env_index`` from ``obs[key]`` when present."""
    arr = np.asarray(obs[key])
    n = _batch_size(obs)
    if n is None:
        return arr
    if arr.shape[0] == n:
        return np.asarray(arr[env_index])
    return arr


def _scalar(obs: dict[str, Any], key: str, env_index: int) -> int | float:
    row = _get_row(obs, key, env_index)
    return int(row) if np.issubdtype(row.dtype, np.integer) else float(row.flat[0])


def print_combat_state(
    obs: dict[str, np.ndarray],
    *,
    env_index: int = 0,
    file: TextIO | None = None,
) -> None:
    """Print a human-readable combat snapshot from a wrapper-style observation.

    Works for a single-env observation dict or vectorized observations whose
    first dimension is the number of environments (e.g. from ``AsyncVectorEnv``).

    Args:
        obs: Observation dict matching ``build_observation_space()``.
        env_index: Which environment row to print when batched.
        file: Text stream (default ``sys.stdout``).
    """
    if file is None:
        file = sys.stdout

    n = _batch_size(obs)
    if n is not None and (env_index < 0 or env_index >= n):
        print(f"[print_combat_state] env_index={env_index} out of range (N={n})", file=file)

    phase_val = _scalar(obs, "phase", env_index)
    phase_name = _enum_name(GamePhase, int(phase_val)) if phase_val <= 2 else str(int(phase_val))
    if int(phase_val) != int(GamePhase.COMBAT):
        print(
            f"*** Note: phase is {phase_name} ({phase_val}), not COMBAT — "
            "combat fields may be zeroed or stale. ***",
            file=file,
        )

    ante = _scalar(obs, "ante", env_index)
    rnd = _scalar(obs, "round", env_index)
    money = _scalar(obs, "money", env_index)
    print("=== Run ===", file=file)
    print(f"  ante={ante}  round={rnd}  phase={phase_name}({phase_val})  money={money}", file=file)

    cur = _scalar(obs, "current_score", env_index)
    tgt = _scalar(obs, "target_score", env_index)
    print("\n=== Blind / score ===", file=file)
    print(f"  round chips: {cur} / target {tgt}  (need {max(0, int(tgt) - int(cur))} more)", file=file)

    hs = _scalar(obs, "hand_size", env_index)
    hr = _scalar(obs, "hands_remaining", env_index)
    dr = _scalar(obs, "discards_remaining", env_index)
    # hpr = _scalar(obs, "hands_played_round", env_index)
    print("\n=== Resources ===", file=file)
    print(
        f"  hand_size={hs}  hands_left={hr}  discards_left={dr}", # hands_played_round={hpr}",
        file=file,
    )

    boss_act = _scalar(obs, "boss_is_active", env_index)
    boss_id = _scalar(obs, "boss_id", env_index)
    next_boss = _scalar(obs, "next_boss_blind_id", env_index)
    print("\n=== Boss ===", file=file)
    print(f"  boss_is_active={boss_act}  boss_id={boss_id}  next_boss_blind_id={next_boss}", file=file)

    hand_ids = _get_row(obs, "hand_card_ids", env_index).astype(int).ravel()
    enh = _get_row(obs, "hand_card_enhancements", env_index).astype(int).ravel()
    ed = _get_row(obs, "hand_card_editions", env_index).astype(int).ravel()
    seal = _get_row(obs, "hand_card_seals", env_index).astype(int).ravel()
    sel = _get_row(obs, "hand_is_selected", env_index).astype(int).ravel()
    fd = _get_row(obs, "hand_is_face_down", env_index).astype(int).ravel()
    db = _get_row(obs, "hand_is_debuffed", env_index).astype(int).ravel()

    print("\n=== Hand ===", file=file)

    def _clip(s: str, w: int) -> str:
        if len(s) <= w:
            return s
        return s[: max(0, w - 1)] + "…"

    rows: list[tuple[int, str, str, str, str, str, str, str]] = []
    for i in range(min(MAX_HAND_SIZE, len(hand_ids))):
        cid = int(hand_ids[i])
        if cid < 0:
            continue
        card = _format_card_id(cid)
        en_s = "-" if int(enh[i]) == 0 else _enum_name(Enhancement, enh[i])
        ed_s = "-" if int(ed[i]) == 0 else _enum_name(Edition, ed[i])
        sl_s = "-" if int(seal[i]) == 0 else _enum_name(Seal, seal[i])
        rows.append(
            (
                i,
                card,
                en_s,
                ed_s,
                sl_s,
                _bool_cell(sel[i]),
                _bool_cell(fd[i]),
                _bool_cell(db[i]),
            ),
        )

    if not rows:
        print("  (no cards in hand slots)", file=file)
    else:
        w_idx, w_card, w_en, w_ed, w_seal = 3, 6, 14, 12, 10
        top = (
            f"  {'#':>{w_idx}} │ {'card':^{w_card}} │ "
            f"{'enhancement':^{w_en}} │ {'edition':^{w_ed}} │ {'seal':^{w_seal}} │ "
            f"sel │ F↓ │ db"
        )
        rule = (
            f"  {'─' * w_idx}─┼─{'─' * w_card}─┼─{'─' * w_en}─┼─{'─' * w_ed}─┼─{'─' * w_seal}─┼───┼───┼───"
        )
        print(top, file=file)
        print(rule, file=file)
        for i, card, en_s, ed_s, sl_s, s_cell, f_cell, d_cell in rows:
            print(
                f"  {i:>{w_idx}} │ {_clip(card, w_card):^{w_card}} │ "
                f"{_clip(en_s, w_en):^{w_en}} │ {_clip(ed_s, w_ed):^{w_ed}} │ {_clip(sl_s, w_seal):^{w_seal}} │ "
                f"{s_cell} │ {f_cell} │ {d_cell}",
                file=file,
            )

    joker_ids = _get_row(obs, "joker_ids", env_index).astype(int).ravel()
    joker_empty = _get_row(obs, "joker_is_empty", env_index).astype(int).ravel()
    print("\n=== Jokers ===", file=file)
    shown = False
    for i in range(len(joker_ids)):
        if int(joker_empty[i]) == 1:
            continue
        print(f"  slot {i}: id={int(joker_ids[i])}", file=file)
        shown = True
    if not shown:
        print("  (none)", file=file)

    cons_ids = _get_row(obs, "consumable_ids", env_index).astype(int).ravel()
    cons_empty = _get_row(obs, "consumable_is_empty", env_index).astype(int).ravel()
    print("\n=== Consumables ===", file=file)
    shown = False
    for i in range(len(cons_ids)):
        if int(cons_empty[i]) == 1:
            continue
        print(f"  slot {i}: id={int(cons_ids[i])}", file=file)
        shown = True
    if not shown:
        print("  (none)", file=file)

    hl = _get_row(obs, "hand_levels", env_index)
    if hl.ndim == 1 and hl.size == NUM_HAND_TYPES * 4:
        hl = hl.reshape(NUM_HAND_TYPES, 4)
    print("\n=== Hand levels (id, lvl, chips, mult) ===", file=file)
    for hi in range(NUM_HAND_TYPES):
        row = hl[hi]
        hid, lvl, ch, mult = int(row[0]), int(row[1]), int(row[2]), int(row[3])
        name = _HAND_TYPE_NAMES[hi]
        print(f"  {name:14s}  id={hid}  lvl={lvl:3d}  chips={ch:5d}  mult={mult:3d}", file=file)

    deck_ranks = _get_row(obs, "deck_ranks", env_index).astype(int).ravel()
    deck_suits = _get_row(obs, "deck_suits", env_index).astype(int).ravel()
    deck_ids = _get_row(obs, "deck_card_ids", env_index).astype(int).ravel()
    n_deck_slots = int((deck_ids >= 0).sum())
    rsum, ssum = int(deck_ranks.sum()), int(deck_suits.sum())
    print("\n=== Draw pile (deck minus hand in combat) ===", file=file)
    print(f"  padded deck slots used: {n_deck_slots}  rank_hist_sum={rsum}  suit_hist_sum={ssum}", file=file)
    rank_labels = [Rank(r + 2).short for r in range(NUM_RANKS)]
    parts = [f"{lab}:{int(deck_ranks[i])}" for i, lab in enumerate(rank_labels) if deck_ranks[i]]
    if parts:
        print("  ranks: " + "  ".join(parts), file=file)
    suit_syms = [Suit(s).symbol() for s in range(4)]
    sparts = [f"{sym}:{int(deck_suits[i])}" for i, sym in enumerate(suit_syms) if deck_suits[i]]
    if sparts:
        print("  suits: " + "  ".join(sparts), file=file)

    print("", file=file)
