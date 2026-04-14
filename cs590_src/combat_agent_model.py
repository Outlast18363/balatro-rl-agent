"""Combat PPO policy network — single source of truth for training and inference.

Lives under ``cs590_src/`` next to training notebooks. Import as
``from cs590_src.combat_agent_model import CombatPPOAgent`` (repo root on ``sys.path``).
"""

import torch
import torch.nn as nn
from typing import Tuple

from cs590_env.schema import (
    MAX_JOKER_DISPLAY,
    NUM_HAND_TYPES,
    NUM_RANKS,
    NUM_SUITS,
    NUM_BOSS_BLINDS,
)
from balatro_gym.cards import Enhancement, Edition, Seal
from balatro_gym.jokers import JOKER_LIBRARY

__all__ = [
    "CardEmbedding",
    "RunStateEmbedding",
    "HandLevelEmbedding",
    "ModifierEmbedding",
    "CombatEmbeddings",
    "PreNormBlock",
    "CombatBackbone",
    "CombatHeads",
    "CombatPPOAgent",
]


class CardEmbedding(nn.Module):
    """Embed a sequence of playing cards via additive composition:
    rank + suit + enhancement + edition + seal.

    Each instance owns its own embedding tables, so hand cards and
    deck cards can learn distinct representations for the same card.
    """

    _NUM_RANKS        = NUM_RANKS              # 13: TWO(0) .. ACE(12)
    _NUM_SUITS        = NUM_SUITS              # 4:  CLUBS(0) .. SPADES(3)
    _NUM_ENHANCEMENTS = len(Enhancement)       # 9:  NONE(0) .. LUCKY(8)
    _NUM_EDITIONS     = len(Edition) - 1        # 4:  NONE(0) .. POLYCHROME(3); NEGATIVE is joker-only
    _NUM_SEALS        = len(Seal)              # 5:  NONE(0) .. PURPLE(4)

    def __init__(self, d_model: int):
        super().__init__()
        D = d_model
        self.rank_emb        = nn.Embedding(self._NUM_RANKS, D)
        self.suit_emb        = nn.Embedding(self._NUM_SUITS, D)
        self.enhancement_emb = nn.Embedding(self._NUM_ENHANCEMENTS, D)
        self.edition_emb     = nn.Embedding(self._NUM_EDITIONS, D)
        self.seal_emb        = nn.Embedding(self._NUM_SEALS, D)

    def forward(
        self,
        card_ids: torch.Tensor,        # (B, N) long, -1 = empty
        enhancements: torch.Tensor,    # (B, N) long
        editions: torch.Tensor,        # (B, N) long
        seals: torch.Tensor,           # (B, N) long
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns:
            toks: (B, N, D) — padding positions zeroed out
            mask: (B, N)    — True for real cards
        """
        mask = card_ids >= 0
        safe = card_ids.clamp(min=0)

        toks = (
            self.rank_emb(safe // 4)
            + self.suit_emb(safe % 4)
            + self.enhancement_emb(enhancements)
            + self.edition_emb(editions)
            + self.seal_emb(seals)
        )
        toks = toks * mask.unsqueeze(-1).float()
        return toks, mask

class RunStateEmbedding(nn.Module):
    """Project run-state scalars into a single D-dimensional token.

    Scalars: hands_remaining, discards_remaining, money,
             current_score, target_score.
    Large-magnitude values are log-compressed; money uses a
    sign-preserving variant since it can go negative.
    """

    NUM_SCALARS = 5

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(self.NUM_SCALARS, d_model)
        self.ln   = nn.LayerNorm(d_model)

    @staticmethod
    def _signed_log1p(x: torch.Tensor) -> torch.Tensor:
        return x.sign() * torch.log1p(x.abs())

    def forward(self, obs: dict) -> torch.Tensor:
        """Returns: (B, 1, D)"""
        feats = torch.stack([
            obs['hands_remaining'].float(),
            obs['discards_remaining'].float(),
            self._signed_log1p(obs['money'].float()),
            torch.log1p(obs['current_score'].float()),
            torch.log1p(obs['target_score'].float()),
        ], dim=-1)
        return self.ln(self.proj(feats).unsqueeze(1))


class HandLevelEmbedding(nn.Module):
    """Embed 12 hand-type level tokens via type_emb + stat_proj (summed).

    Input: hand_levels (B, 12, 4) where each row is [hand_type_id, level, chip, mult].
    Column 0 (id) is used for embedding lookup; columns 2-3 (chip, mult) are
    projected through a linear layer.  Column 1 (level) is skipped because
    chip and mult already encode the level-scaled values.
    """

    _NUM_HAND_TYPES = NUM_HAND_TYPES   # 12
    _HL_FEATS       = 2                # chip, mult

    def __init__(self, d_model: int):
        super().__init__()
        self.type_emb   = nn.Embedding(self._NUM_HAND_TYPES, d_model)
        self.level_proj = nn.Linear(self._HL_FEATS, d_model)
        self.ln         = nn.LayerNorm(d_model)

    def forward(self, hand_levels: torch.Tensor) -> torch.Tensor:
        """Returns: (B, 12, D)"""
        ht_ids   = hand_levels[:, :, 0].long()
        hl_feats = hand_levels[:, :, 2:].float()
        return self.ln(self.type_emb(ht_ids) + self.level_proj(hl_feats))

class ModifierEmbedding(nn.Module):
    """Embed boss + jokers into a positional modifier sequence.

    Boss and jokers share a single embedding table.
    ID layout: 0 = padding, 1-150 = jokers, 151-178 = boss blinds.
    Boss occupies slot 0 when active; otherwise jokers shift left.
    """

    _BOSS_ID_OFFSET   = len(JOKER_LIBRARY)                         # 150
    _NUM_MODIFIER_IDS = len(JOKER_LIBRARY) + NUM_BOSS_BLINDS + 1   # 179
    _MAX_MODIFIERS    = 1 + MAX_JOKER_DISPLAY                      # 11

    def __init__(self, d_model: int):
        super().__init__()
        self.emb     = nn.Embedding(self._NUM_MODIFIER_IDS, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(self._MAX_MODIFIERS, d_model)
        self.ln      = nn.LayerNorm(d_model)

    def forward(self, obs: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns: mod_seq (B, 11, D), mod_mask (B, 11)"""
        device = obs['boss_id'].device
        B      = obs['boss_id'].shape[0]

        has_boss    = obs['boss_is_active'].bool()                 # (B,)
        boss_as_mod = (obs['boss_id'].long()
                       + self._BOSS_ID_OFFSET).unsqueeze(1)        # (B, 1)
        joker_ids   = obs['joker_ids'].long()                      # (B, 10)
        pad         = torch.zeros(B, 1, dtype=torch.long, device=device)

        mod_ids = torch.where(
            has_boss[:, None],
            torch.cat([boss_as_mod, joker_ids], 1),
            torch.cat([joker_ids, pad], 1),
        )                                                          # (B, 11)

        pos     = self.pos_emb(torch.arange(self._MAX_MODIFIERS, device=device))
        mod_seq = self.ln(self.emb(mod_ids) + pos)                 # (B, 11, D)

        joker_real = obs['joker_is_empty'].long() == 0             # (B, 10)
        mod_mask = torch.where(
            has_boss[:, None],
            torch.cat([torch.ones(B, 1, dtype=torch.bool, device=device),
                        joker_real], 1),
            torch.cat([joker_real,
                        torch.zeros(B, 1, dtype=torch.bool, device=device)], 1),
        )                                                          # (B, 11)

        # Guarantee at least one unmasked position so attention
        # never softmaxes over all -inf (which produces NaN).
        # Slot 0 uses padding_idx=0 → zero-vector embedding, semantically neutral.
        no_mod = ~mod_mask.any(dim=1)                              # (B,)
        mod_mask[:, 0] = mod_mask[:, 0] | no_mod

        return mod_seq, mod_mask

class CombatEmbeddings(nn.Module):
    """Orchestrator: composes sub-embeddings into the full combat
    observation encoding.

    Delegates to:
        CardEmbedding        — hand cards and deck cards (separate weights)
        RunStateEmbedding    — scalar game-state → single token
        HandLevelEmbedding   — 12 hand-type level tokens
        ModifierEmbedding    — boss + jokers with positional encoding

    Returns:
        hand_toks  (B, MAX_HAND_SIZE, D)                hand_mask  (B, MAX_HAND_SIZE)
        run_tok    (B, 1, D)
        ctx_seq    (B, NUM_HAND_TYPES + MAX_DECK, D)    ctx_mask   (B, NUM_HAND_TYPES + MAX_DECK)
        mod_seq    (B, 1+MAX_JOKER_DISPLAY, D)          mod_mask   (B, 1+MAX_JOKER_DISPLAY)
    """

    NUM_HAND_FLAGS = 2   # face_down, debuffed

    def __init__(self, d_model: int):
        super().__init__()
        D = d_model

        self.hand_card_emb   = CardEmbedding(D)
        self.deck_card_emb   = CardEmbedding(D)
        self.hand_flags_proj = nn.Linear(self.NUM_HAND_FLAGS, D, bias=False)

        self.run_emb = RunStateEmbedding(D)
        self.hl_emb  = HandLevelEmbedding(D)
        self.mod_emb = ModifierEmbedding(D)

        self.hand_ln = nn.LayerNorm(D)
        self.deck_ln = nn.LayerNorm(D)

    def forward(self, obs: dict):
        # ── Hand cards ───────────────────────────────────────────
        hand_toks, hand_mask = self.hand_card_emb(
            obs['hand_card_ids'].long(),
            obs['hand_card_enhancements'].long(),
            obs['hand_card_editions'].long(),
            obs['hand_card_seals'].long(),
        )
        flags = torch.stack([
            obs['hand_is_face_down'].float(),
            obs['hand_is_debuffed'].float(),
        ], dim=-1)
        hand_toks = hand_toks + self.hand_flags_proj(flags) * hand_mask.unsqueeze(-1).float()
        hand_toks = self.hand_ln(hand_toks)

        # ── Deck cards ───────────────────────────────────────────
        deck_toks, deck_mask = self.deck_card_emb(
            obs['deck_card_ids'].long(),
            obs['deck_card_enhancements'].long(),
            obs['deck_card_editions'].long(),
            obs['deck_card_seals'].long(),
        )
        deck_toks = self.deck_ln(deck_toks)

        # ── Scalar state / hand levels / modifiers ───────────────
        run_tok            = self.run_emb(obs)
        hl_toks            = self.hl_emb(obs['hand_levels'])
        mod_seq, mod_mask  = self.mod_emb(obs)

        # ── Context sequence: hand levels ‖ deck cards ───────────
        B = hand_toks.shape[0]
        ctx_seq = torch.cat([hl_toks, deck_toks], dim=1)
        ctx_mask = torch.cat([
            torch.ones(B, hl_toks.shape[1], dtype=torch.bool,
                       device=hand_toks.device),
            deck_mask,
        ], dim=1)

        return hand_toks, hand_mask, run_tok, ctx_seq, ctx_mask, mod_seq, mod_mask

class PreNormBlock(nn.Module):
    """Pre-LayerNorm attention + FFN with residual connections.

    Supports both self-attention (kv=None) and cross-attention (kv given).
    """

    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor = None,
        kv_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            q:       (B, Nq, D)  query sequence
            kv:      (B, Nkv, D) key-value sequence; None → self-attention
            kv_mask: (B, Nkv)    True = real token (inverted internally for PyTorch)
        """
        q_norm = self.norm1(q)

        if kv is None:
            k = v = q_norm
        else:
            k = v = kv

        pad_mask = ~kv_mask if kv_mask is not None else None
        attn_out, _ = self.attn(q_norm, k, v, key_padding_mask=pad_mask)
        q = q + attn_out

        q = q + self.ffn(self.norm2(q))
        return q


class CombatBackbone(nn.Module):
    """4-stage sequential attention backbone.

    Stage 1A: Hand self-attention         (cards learn about each other)
    Stage 1B: Modifier self-attention     (jokers learn about each other + boss)
                                          [parallel with 1A]
    Stage 2:  Modifier × run_state cross  (jokers become combat-aware)
    Stage 3:  Hand × modifier cross       (cards absorb modifier effects)
    Stage 4:  Hand × [hl ‖ deck] cross    (cards reason about hand levels + draw pile)
    Pool:     Mean pool over hand cards   → global context vector

    Args:
        d_model:       token dimension (must match CombatEmbeddings)
        nhead:         number of attention heads
        dim_ff:        FFN hidden dimension
        dropout:       dropout rate
        depth_hand:    layers for Stage 1A  (hand self-attention)
        depth_mod:     layers for Stage 1B  (modifier self-attention)
        depth_mod_run: layers for Stage 2   (modifier × run state cross)
        depth_hm:      layers for Stage 3   (hand × modifier cross)
        depth_hc:      layers for Stage 4   (hand × context cross)
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        dim_ff: int = 1024,
        dropout: float = 0.1,
        depth_hand: int = 4,
        depth_mod: int = 3,
        depth_mod_run: int = 1,
        depth_hm: int = 3,
        depth_hc: int = 3,
    ):
        super().__init__()
        blk = lambda: PreNormBlock(d_model, nhead, dim_ff, dropout)

        self.hand_self_layers = nn.ModuleList([blk() for _ in range(depth_hand)])     # Stage 1A
        self.mod_self_layers  = nn.ModuleList([blk() for _ in range(depth_mod)])      # Stage 1B
        self.mod_run_layers   = nn.ModuleList([blk() for _ in range(depth_mod_run)])  # Stage 2
        self.hand_mod_layers  = nn.ModuleList([blk() for _ in range(depth_hm)])       # Stage 3
        self.hand_ctx_layers  = nn.ModuleList([blk() for _ in range(depth_hc)])       # Stage 4

        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        hand_toks: torch.Tensor,   # (B, MAX_HAND_SIZE, D)
        hand_mask: torch.Tensor,   # (B, MAX_HAND_SIZE)
        run_tok:   torch.Tensor,   # (B, 1, D)
        ctx_seq:   torch.Tensor,   # (B, NUM_HAND_TYPES + MAX_DECK, D)
        ctx_mask:  torch.Tensor,   # (B, NUM_HAND_TYPES + MAX_DECK)
        mod_seq:   torch.Tensor,   # (B, 1+MAX_JOKER_DISPLAY, D)
        mod_mask:  torch.Tensor,   # (B, 1+MAX_JOKER_DISPLAY)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            hand_final: (B, MAX_HAND_SIZE, D)  enriched per-card representations
            global_ctx: (B, D)                 mean-pooled context for execution / value heads
        """

        # ── Stage 1A: Hand self-attention (4 layers) ─────────────
        hand = hand_toks
        for layer in self.hand_self_layers:
            hand = layer(hand, kv_mask=hand_mask)

        # ── Stage 1B: Modifier self-attention (3 layers) ─────────
        mod = mod_seq
        for layer in self.mod_self_layers:
            mod = layer(mod, kv_mask=mod_mask)

        # ── Stage 2: Modifier × run_state cross (1 layer) ────────
        for layer in self.mod_run_layers:
            mod = layer(mod, kv=run_tok)

        # ── Stage 3: Hand × modifier cross (3 layers) ────────────
        for layer in self.hand_mod_layers:
            hand = layer(hand, kv=mod, kv_mask=mod_mask)

        # ── Stage 4: Hand × context cross (3 layers) ─────────────
        for layer in self.hand_ctx_layers:
            hand = layer(hand, kv=ctx_seq, kv_mask=ctx_mask)

        # ── Final norm + pool ────────────────────────────────────
        hand_final = self.final_norm(hand)

        mask_f = hand_mask.unsqueeze(-1).float()                   # (B, MAX_HAND_SIZE, 1)
        global_ctx = (
            (hand_final * mask_f).sum(dim=1)
            / mask_f.sum(dim=1).clamp(min=1)
        )                                                          # (B, D)

        return hand_final, global_ctx

class CombatHeads(nn.Module):
    """Actor-Critic heads for the combat agent.

    Card Selection (actor):  per-card binary logits from hand_final
    Execution (actor):       play-vs-discard logits from global_ctx
    Critic:                  scalar V(s) from global_ctx

    The actor and critic share the embedding + backbone trunk.
    Only the final projections are separate.
    """

    def __init__(self, d_model: int):
        super().__init__()

        # ── Card selection: per-card (B, MAX_HAND_SIZE, D) → (B, MAX_HAND_SIZE, 2)
        self.select_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
        )

        # ── Execution: global (B, D) → (B, 2) ──────────────────
        self.exec_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
        )

        # ── Critic: global (B, D) → (B, 1) ─────────────────────
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        hand_final: torch.Tensor,   # (B, MAX_HAND_SIZE, D)
        global_ctx: torch.Tensor,   # (B, D)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            sel_logits:  (B, MAX_HAND_SIZE, 2)  per-card select / don't-select logits
            exec_logits: (B, 2)                 play / discard logits
            value:       (B, 1)                 state-value estimate
        """
        sel_logits  = self.select_head(hand_final)     # (B, MAX_HAND_SIZE, 2)
        exec_logits = self.exec_head(global_ctx)       # (B, 2)
        value       = self.value_head(global_ctx)      # (B, 1)

        return sel_logits, exec_logits, value

class CombatPPOAgent(nn.Module):
    """Full combat agent: embeddings → backbone → actor-critic heads.

    Args:
        d_model: token / hidden dimension shared across all components
        nhead:   attention heads in the backbone
        dim_ff:  FFN hidden dim in the backbone
        dropout: dropout rate in the backbone
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        dim_ff: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embeddings = CombatEmbeddings(d_model=d_model)
        self.backbone   = CombatBackbone(d_model=d_model, nhead=nhead,
                                         dim_ff=dim_ff, dropout=dropout)
        self.heads      = CombatHeads(d_model=d_model)

    def forward(
        self, obs: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            obs: dict from BalatroPhaseWrapper (combat phase)

        Returns:
            sel_logits:  (B, MAX_HAND_SIZE, 2)  per-card select / don't-select
            exec_logits: (B, 2)                 play / discard
            value:       (B, 1)                 state-value estimate V(s)
        """
        (hand_toks, hand_mask, run_tok,
         ctx_seq, ctx_mask, mod_seq, mod_mask) = self.embeddings(obs)

        hand_final, global_ctx = self.backbone(
            hand_toks, hand_mask, run_tok,
            ctx_seq, ctx_mask, mod_seq, mod_mask,
        )

        return self.heads(hand_final, global_ctx)