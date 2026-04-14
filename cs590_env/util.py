"""cs590_env/util.py — Human-readable stdout dumps of combat observations."""

from __future__ import annotations

import sys
from enum import IntEnum
from pathlib import Path
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


def _enum_name_none_as_empty(enum_cls: type[IntEnum], value: int) -> str:
    s = _enum_name(enum_cls, value)
    return "" if s == "NONE" else s


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

    # phase_val = _scalar(obs, "phase", env_index)
    # phase_name = _enum_name(GamePhase, int(phase_val)) if phase_val <= 2 else str(int(phase_val))
    # if int(phase_val) != int(GamePhase.COMBAT):
    #     print(
    #         f"*** Note: phase is {phase_name} ({phase_val}), not COMBAT — "
    #         "combat fields may be zeroed or stale. ***",
    #         file=file,
    #     )

    # ante = _scalar(obs, "ante", env_index)
    # rnd = _scalar(obs, "round", env_index)
    # hpr = _scalar(obs, "hands_played_round", env_index)
    # print("=== Run ===", file=file)
    # print(f"  ante={ante}  round={rnd}  phase={phase_name}({phase_val})  hands_played_round={hpr}", file=file)

    cur = _scalar(obs, "current_score", env_index)
    tgt = _scalar(obs, "target_score", env_index)
    print("\n=== Blind / score ===", file=file)
    print(f"  round chips: {cur} / target {tgt}  (need {max(0, int(tgt) - int(cur))} more)", file=file)

    money = _scalar(obs, "money", env_index)
    hs = _scalar(obs, "hand_size", env_index)
    hr = _scalar(obs, "hands_remaining", env_index)
    dr = _scalar(obs, "discards_remaining", env_index)
    print("\n=== Resources ===", file=file)
    print(f"  money={money}  hand_size={hs}  hands_left={hr}  discards_left={dr}", file=file)

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

    def _cell(s: str, w: int) -> str:
        if len(s) <= w:
            return s.center(w)
        return s[: max(1, w - 1)] + "…"

    cols: list[dict[str, str]] = []
    for i in range(min(MAX_HAND_SIZE, len(hand_ids))):
        cid = int(hand_ids[i])
        if cid < 0:
            continue
        cols.append(
            {
                "slot": str(i),
                "card": _format_card_id(cid),
                "enhancement": _enum_name_none_as_empty(Enhancement, int(enh[i])),
                "edition": _enum_name_none_as_empty(Edition, int(ed[i])),
                "seal": _enum_name_none_as_empty(Seal, int(seal[i])),
                "sel": "✅" if int(sel[i]) else "",
                "face↓": "⬇️" if int(fd[i]) else "",
                "debuff": "❌" if int(db[i]) else "",
            },
        )

    if not cols:
        print("  (no cards in hand slots)", file=file)
    else:
        row_spec: list[tuple[str, str]] = [
            ("slot", "slot"),
            ("card", "card"),
            ("enhancement", "enhancement"),
            ("edition", "edition"),
            ("seal", "seal"),
            ("sel", "selected"),
            ("face↓", "face_down"),
            ("debuff", "debuffed"),
        ]
        w_label = max(len(lbl) for _, lbl in row_spec)
        n = len(cols)
        col_widths: list[int] = []
        for j in range(n):
            w = 2  # ⬇️ (U+2B07 + VS-16) etc.; keep room for single-cell markers
            w = max(w, len(cols[j]["card"]))
            for key, _ in row_spec:
                w = max(w, len(cols[j][key]))
            col_widths.append(w)

        gap = " │ "

        def _hline(left_w: int) -> str:
            s = "  " + "─" * left_w + "─┼"
            for j, cw in enumerate(col_widths):
                s += "─" * (cw + 2)
                if j < n - 1:
                    s += "┼"
            return s

        # Header row: one column per card (card as column title).
        hdr_cells = [_cell(cols[j]["card"], col_widths[j]) for j in range(n)]
        print(f"  {' ':{w_label}}{gap}{gap.join(hdr_cells)}", file=file)
        print(_hline(w_label), file=file)

        for key, lbl in row_spec:
            cells = [_cell(cols[j][key], col_widths[j]) for j in range(n)]
            print(f"  {lbl:{w_label}}{gap}{gap.join(cells)}", file=file)

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

    deck_ids = _get_row(obs, "deck_card_ids", env_index).astype(int).ravel()
    n_deck_slots = int((deck_ids >= 0).sum())
    print("\n=== Draw pile (deck minus hand in combat) ===", file=file)
    print(f"  padded deck slots used: {n_deck_slots}", file=file)

    # Same id layout as hand: suit = id % 4, rank index = id // 4 (13 ranks 2..A).
    grid: list[list[int]] = [[0] * NUM_RANKS for _ in range(4)]
    for cid in deck_ids:
        ci = int(cid)
        if ci < 0:
            continue
        grid[ci % 4][ci // 4] += 1

    # Bordered 4×13 grid: rows = suits, cols = ranks; fixed cell width for alignment.
    inner = 3
    ncols = 1 + NUM_RANKS
    indent = "  "

    def _deck_slot(text: str) -> str:
        t = (text or "")[:inner]
        return f"{t:^{inner}}"

    def _deck_rule(left: str, cross: str, right: str) -> str:
        return indent + left + cross.join("─" * inner for _ in range(ncols)) + right

    print(_deck_rule("┌", "┬", "┐"), file=file)
    hdr_cells = [""] + [Rank(r + 2).short for r in range(NUM_RANKS)]
    print(indent + "│" + "│".join(_deck_slot(x) for x in hdr_cells) + "│", file=file)
    print(_deck_rule("├", "┼", "┤"), file=file)
    for s in range(4):
        sym = Suit(s).symbol()
        row_cells = [_deck_slot(sym)] + [
            _deck_slot("" if grid[s][ri] == 0 else str(grid[s][ri])) for ri in range(NUM_RANKS)
        ]
        print(indent + "│" + "│".join(row_cells) + "│", file=file)
    print(_deck_rule("└", "┴", "┘"), file=file)

    hl = _get_row(obs, "hand_levels", env_index)
    if hl.ndim == 1 and hl.size == NUM_HAND_TYPES * 4:
        hl = hl.reshape(NUM_HAND_TYPES, 4)
    print("\n=== Hand levels (id, lvl, chips, mult) ===", file=file)
    for hi in range(NUM_HAND_TYPES):
        row = hl[hi]
        hid, lvl, ch, mult = int(row[0]), int(row[1]), int(row[2]), int(row[3])
        name = _HAND_TYPE_NAMES[hi]
        print(f"  {name:14s}  id={hid}  lvl={lvl:3d}  chips={ch:5d}  mult={mult:3d}", file=file)

    print("", file=file)


def _ppo_hparams_from_checkpoint(ckpt: dict[str, Any]) -> dict[str, Any]:
    """Read ``d_model`` / ``nhead`` / ``dim_ff`` / ``dropout`` from a TrainCombat-style checkpoint."""
    defaults = {"d_model": 256, "nhead": 8, "dim_ff": 1024, "dropout": 0.1}
    cfg = ckpt.get("config")
    if cfg is None:
        return defaults
    if isinstance(cfg, dict):
        return {k: type(defaults[k])(cfg.get(k, defaults[k])) for k in defaults}
    for k in defaults:
        if hasattr(cfg, k):
            defaults[k] = getattr(cfg, k)
    return {k: int(defaults[k]) if k != "dropout" else float(defaults[k]) for k in defaults}


def load_combat_ppo_agent(
    checkpoint_path: str | Path,
    *,
    device: str | None = None,
    map_location: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Load ``CombatPPOAgent`` weights from a ``.pt`` file (e.g. ``TrainCombat.ipynb`` checkpoints).

    Expected keys: ``model_state_dict``, optionally ``config``, ``iteration``.

    Returns:
        ``(agent, meta)`` where ``meta`` includes ``iteration``, ``config``, and ``path``.
    """
    import torch

    from cs590_src.combat_agent_model import CombatPPOAgent

    path = Path(checkpoint_path)
    loc = map_location or device or ("cuda" if torch.cuda.is_available() else "cpu")

    # TrainCombat saves ``cfg`` as a dataclass defined under ``__main__`` when
    # ``%run BalatroPPO.ipynb`` — unpickling elsewhere fails unless ``__main__``
    # exposes a compatible ``PPOConfig`` (see cs590_src.ppo_config).
    import __main__ as _main

    if not hasattr(_main, "PPOConfig"):
        from cs590_src.ppo_config import PPOConfig as _PPOConfig

        _main.PPOConfig = _PPOConfig

    try:
        ckpt = torch.load(path, map_location=loc, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=loc)

    hp = _ppo_hparams_from_checkpoint(ckpt)
    agent = CombatPPOAgent(
        d_model=hp["d_model"],
        nhead=hp["nhead"],
        dim_ff=hp["dim_ff"],
        dropout=hp["dropout"],
    )
    agent.load_state_dict(ckpt["model_state_dict"])
    dev = torch.device(device or loc)
    agent.to(dev)
    agent.eval()
    meta: dict[str, Any] = {
        "path": path,
        "iteration": ckpt.get("iteration"),
        "config": ckpt.get("config"),
        "device": dev,
        "hparams": hp,
    }
    return agent, meta


def _obs_numpy_to_torch_batch(
    obs: dict[str, np.ndarray],
    device: Any,
) -> dict[str, Any]:
    """Add batch dimension ``B=1`` for every observation tensor."""
    import torch

    out: dict[str, Any] = {}
    for k, v in obs.items():
        t = torch.as_tensor(np.asarray(v), device=device)
        if t.dim() == 0:
            out[k] = t.unsqueeze(0)
        else:
            out[k] = t.unsqueeze(0)
    return out


def combat_obs_policy_action(
    agent: Any,
    obs: dict[str, np.ndarray],
    *,
    device: Any,
    deterministic: bool = True,
) -> tuple[np.ndarray, int]:
    """Map a single-env combat observation to factored action (card mask + execution).

    Returns:
        ``(card_selections, execution)`` with ``card_selections`` shape ``(MAX_HAND_SIZE,)``
        int8 in ``{0,1}``, ``execution`` ``0`` = play, ``1`` = discard.
    """
    import torch
    from torch.distributions import Categorical

    from cs590_env.combat_env import get_card_mask, mask_logits

    obs_t = _obs_numpy_to_torch_batch(obs, device)
    with torch.no_grad():
        sel_logits, exec_logits, _values = agent(obs_t)
    card_mask = get_card_mask(obs_t)
    sel_logits = mask_logits(sel_logits, card_mask)

    if deterministic:
        card_sels = sel_logits.argmax(dim=-1).squeeze(0)
        execution = int(exec_logits.argmax(dim=-1).squeeze(0).item())
    else:
        sel_dist = Categorical(logits=sel_logits)
        exec_dist = Categorical(logits=exec_logits)
        card_sels = sel_dist.sample().squeeze(0)
        execution = int(exec_dist.sample().item())

    card_np = card_sels.cpu().numpy().astype(np.int8)
    return card_np, execution


def run_weight_interpreter(
    combat: Any,
    checkpoint_path: str | Path,
    *,
    device: str | None = None,
    deterministic: bool = True,
    max_steps: int = 512,
    env_index: int = 0,
    initial_obs: dict[str, np.ndarray] | None = None,
    print_policy_line: bool = True,
    file: TextIO | None = None,
) -> dict[str, Any]:
    """Run the loaded combat policy until the combat episode ends, printing each observation.

    Uses the same factored action space as training: ``CombatActionWrapper.step`` with
    1–5 cards selected, then play (0) or discard (1). Stops when the wrapper reports
    ``done`` (blind cleared, run out of hands, or left combat phase).

    Args:
        combat: A ``CombatActionWrapper`` instance, or a ``PooledCombatEnv`` (uses ``._combat``).
        checkpoint_path: ``torch.save`` file with ``model_state_dict`` (and optional ``config``).
        device: Torch device string; default follows CUDA availability.
        deterministic: If True, argmax on policy logits; else sample from the same distributions as PPO.
        max_steps: Safety cap on policy steps (each step = one play or discard attempt after toggles).
        env_index: Row printed when observations are batched.
        initial_obs: If set, assigned to ``combat._last_obs`` before the first printed step.
        print_policy_line: If True, print play/discard and selected slot indices after each state.
        file: Text stream for prints (default ``sys.stdout``).

    Returns:
        Summary dict with ``steps``, ``total_reward``, ``done``, ``reason``, ``last_info``.
    """
    from cs590_env.combat_wrapper import CombatActionWrapper

    if file is None:
        file = sys.stdout

    if isinstance(combat, CombatActionWrapper):
        wrap = combat
    else:
        inner = getattr(combat, "_combat", None)
        if not isinstance(inner, CombatActionWrapper):
            raise TypeError(
                "combat must be CombatActionWrapper or PooledCombatEnv with ._combat wrapper"
            )
        wrap = inner

    if initial_obs is not None:
        wrap._last_obs = initial_obs

    if wrap._last_obs is None:
        raise RuntimeError("No observation to start from: set combat._last_obs or pass initial_obs=")

    agent, meta = load_combat_ppo_agent(checkpoint_path, device=device)
    dev = meta["device"]

    total_reward = 0.0
    last_info: dict[str, Any] = {}
    reason = "max_steps"
    done = False
    obs: dict[str, np.ndarray] = wrap._last_obs
    step_count = 0

    for t in range(max_steps):
        print(f"\n=== weight interpreter step {t} ===", file=file)
        print_combat_state(wrap._last_obs, env_index=env_index, file=file)

        card_sel, execution = combat_obs_policy_action(
            agent, wrap._last_obs, device=dev, deterministic=deterministic,
        )
        if print_policy_line:
            slots = np.flatnonzero(card_sel).tolist()
            mode = "discard" if execution else "play"
            print(f"  policy → {mode}  cards={slots}  (n={len(slots)})", file=file)

        obs, reward, done, info = wrap.step(card_sel, execution)
        step_count += 1
        total_reward += float(reward)
        last_info = dict(info) if info else {}

        if "error" in last_info:
            reason = "invalid_action"
            break
        if done:
            reason = "terminal"
            break

    if reason == "max_steps" and not done:
        print(f"\n[run_weight_interpreter] stopped after {max_steps} steps (not terminal)", file=file)

    if done:
        print(f"\n=== weight interpreter final ({reason}) ===", file=file)
        print_combat_state(obs, env_index=env_index, file=file)

    return {
        "steps": step_count,
        "total_reward": total_reward,
        "done": done,
        "reason": reason,
        "last_info": last_info,
        "checkpoint": str(Path(checkpoint_path)),
        "device": str(dev),
    }
