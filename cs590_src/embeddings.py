"""Embedding specs and runtime token encoders for the CS590 model stack."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - handled by runtime guards
    torch = None
    nn = None


MAX_HAND_SIZE = 8
MAX_JOKER_SLOTS = 10
HAND_LEVEL_COUNT = 12

CARD_VOCAB_SIZE = 53  # 52 cards + 1 padding slot produced from card_id + 1.
ENHANCEMENT_VOCAB_SIZE = 9
EDITION_VOCAB_SIZE = 5
SEAL_VOCAB_SIZE = 5
JOKER_VOCAB_SIZE = 256
BOSS_VOCAB_SIZE = 65


@dataclass(frozen=True)
class EmbeddingBlockSpec:
    """Describe one embedding block and the wrapper fields it consumes.

    Parameters:
        name: Stable block name used across model modules.
        input_keys: Wrapper payload paths consumed by the block.
        description: Brief explanation of the semantic role of the block.
    Returns:
        A lightweight specification object for the model skeleton.
    """

    name: str
    input_keys: Tuple[str, ...]
    description: str


@dataclass
class EmbeddingOutput:
    """Embedded token sequences consumed by the transformer backbone."""

    card_tokens: Any
    card_padding_mask: Any
    state_tokens: Any
    modifier_tokens: Any
    modifier_padding_mask: Any


CARD_EMBEDDING = EmbeddingBlockSpec(
    name="card_embedding",
    input_keys=(
        "hand_tokens.card_id",
        "hand_tokens.is_empty",
        "hand_tokens.is_face_down",
        "hand_tokens.enhancement_id",
        "hand_tokens.edition_id",
        "hand_tokens.seal_id",
        "hand_tokens.is_selected",
    ),
    description="Per-card token block built from the fixed 8-slot hand view.",
)

RUN_STATE_EMBEDDING = EmbeddingBlockSpec(
    name="run_state_embedding",
    input_keys=(
        "run_token.money",
        "run_token.hands_left",
        "run_token.discards_left",
        "run_token.chips_needed",
        "run_token.round_chips_scored",
        "run_token.ante",
        "run_token.round",
        "run_token.phase",
    ),
    description="Global combat-state block for resources, progress, and phase.",
)

DECK_STATE_EMBEDDING = EmbeddingBlockSpec(
    name="deck_state_embedding",
    input_keys=(
        "deck_token.draw_pile_size",
        "deck_token.remaining_rank_histogram",
        "deck_token.remaining_suit_histogram",
    ),
    description="Draw-pile summary block for future draw potential.",
)

HAND_LEVEL_EMBEDDING = EmbeddingBlockSpec(
    name="hand_level_embedding",
    input_keys=("hand_levels",),
    description="Poker-hand progression block for the 12 hand levels.",
)

BOSS_EMBEDDING = EmbeddingBlockSpec(
    name="boss_embedding",
    input_keys=(
        "boss_token.boss_blind_active",
        "boss_token.boss_blind_type",
    ),
    description="Boss-blind block for legality and scoring modifiers.",
)

JOKER_EMBEDDING = EmbeddingBlockSpec(
    name="joker_embedding",
    input_keys=(
        "joker_tokens.joker_id",
        "joker_tokens.is_empty",
        "joker_tokens.is_disabled",
    ),
    description="Per-slot joker block for active and disabled modifiers.",
)


def get_embedding_block_specs() -> Tuple[EmbeddingBlockSpec, ...]:
    """Return the embedding blocks defined by the current architecture.

    Returns:
        An ordered tuple of high-level embedding block specifications.
    """

    return (
        CARD_EMBEDDING,
        RUN_STATE_EMBEDDING,
        DECK_STATE_EMBEDDING,
        HAND_LEVEL_EMBEDDING,
        BOSS_EMBEDDING,
        JOKER_EMBEDDING,
    )


def split_wrapper_observation(observation: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Group wrapper payload fields into the decided embedding partitions.

    Parameters:
        observation: The nested observation returned by `BalatroArchWrapper`.
    Returns:
        A dict keyed by embedding-block name. Each value contains only the
        wrapper fields assigned to that block.
    """

    return {
        CARD_EMBEDDING.name: {"hand_tokens": observation["hand_tokens"]},
        RUN_STATE_EMBEDDING.name: {"run_token": observation["run_token"]},
        DECK_STATE_EMBEDDING.name: {"deck_token": observation["deck_token"]},
        HAND_LEVEL_EMBEDDING.name: {"hand_levels": observation["hand_levels"]},
        BOSS_EMBEDDING.name: {"boss_token": observation["boss_token"]},
        JOKER_EMBEDDING.name: {"joker_tokens": observation["joker_tokens"]},
    }


def _require_torch() -> None:
    """Raise a friendly error when runtime modules are used without torch."""

    if nn is None:
        raise ModuleNotFoundError(
            "BalatroEmbeddingStack requires PyTorch. Install `torch` to use the "
            "runtime model implementation."
        )


if nn is not None:
    class DenseTokenEncoder(nn.Module):
        """Project one dense feature vector into a single transformer token."""

        def __init__(self, input_dim: int, embed_dim: int, dropout: float) -> None:
            super().__init__()
            hidden_dim = max(embed_dim, input_dim * 2)
            self.net = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, embed_dim),
            )

        def forward(self, features: torch.Tensor) -> torch.Tensor:
            return self.net(features)


    class BalatroEmbeddingStack(nn.Module):
        """Embed wrapper observations into the token groups used by the backbone."""

        def __init__(
            self,
            embed_dim: int,
            *,
            dropout: float = 0.1,
            max_boss_type: int = BOSS_VOCAB_SIZE,
            max_joker_id: int = JOKER_VOCAB_SIZE,
        ) -> None:
            super().__init__()
            self.embed_dim = embed_dim

            self.card_id_embedding = nn.Embedding(
                CARD_VOCAB_SIZE,
                embed_dim,
                padding_idx=0,
            )
            self.card_slot_embedding = nn.Embedding(MAX_HAND_SIZE, embed_dim)
            self.enhancement_embedding = nn.Embedding(ENHANCEMENT_VOCAB_SIZE, embed_dim)
            self.edition_embedding = nn.Embedding(EDITION_VOCAB_SIZE, embed_dim)
            self.seal_embedding = nn.Embedding(SEAL_VOCAB_SIZE, embed_dim)
            self.empty_flag_embedding = nn.Embedding(2, embed_dim)
            self.face_down_flag_embedding = nn.Embedding(2, embed_dim)
            self.selected_flag_embedding = nn.Embedding(2, embed_dim)

            self.state_token_type_embedding = nn.Embedding(3, embed_dim)
            self.run_token_encoder = DenseTokenEncoder(input_dim=8, embed_dim=embed_dim, dropout=dropout)
            self.deck_token_encoder = DenseTokenEncoder(input_dim=18, embed_dim=embed_dim, dropout=dropout)
            self.hand_level_encoder = DenseTokenEncoder(
                input_dim=HAND_LEVEL_COUNT,
                embed_dim=embed_dim,
                dropout=dropout,
            )

            self.boss_type_embedding = nn.Embedding(max_boss_type, embed_dim)
            self.boss_active_embedding = nn.Embedding(2, embed_dim)
            self.modifier_type_embedding = nn.Embedding(2, embed_dim)

            self.joker_id_embedding = nn.Embedding(
                max_joker_id,
                embed_dim,
                padding_idx=0,
            )
            self.joker_slot_embedding = nn.Embedding(MAX_JOKER_SLOTS, embed_dim)
            self.joker_empty_embedding = nn.Embedding(2, embed_dim)
            self.joker_disabled_embedding = nn.Embedding(2, embed_dim)

            self.card_norm = nn.LayerNorm(embed_dim)
            self.state_norm = nn.LayerNorm(embed_dim)
            self.modifier_norm = nn.LayerNorm(embed_dim)
            self.dropout = nn.Dropout(dropout)

        @staticmethod
        def _signed_log1p(values: torch.Tensor) -> torch.Tensor:
            return torch.sign(values) * torch.log1p(values.abs())

        @staticmethod
        def _zero_masked_tokens(tokens: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
            return tokens.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        def _build_card_tokens(
            self,
            hand_tokens: Mapping[str, torch.Tensor],
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            card_ids = hand_tokens["card_id"].long().add(1).clamp(min=0, max=CARD_VOCAB_SIZE - 1)
            is_empty = hand_tokens["is_empty"].long().clamp(0, 1)
            is_face_down = hand_tokens["is_face_down"].long().clamp(0, 1)
            enhancement_id = hand_tokens["enhancement_id"].long().clamp(0, ENHANCEMENT_VOCAB_SIZE - 1)
            edition_id = hand_tokens["edition_id"].long().clamp(0, EDITION_VOCAB_SIZE - 1)
            seal_id = hand_tokens["seal_id"].long().clamp(0, SEAL_VOCAB_SIZE - 1)
            is_selected = hand_tokens["is_selected"].long().clamp(0, 1)

            batch_size = card_ids.shape[0]
            slot_index = torch.arange(MAX_HAND_SIZE, device=card_ids.device).expand(batch_size, -1)

            tokens = self.card_id_embedding(card_ids)
            tokens = tokens + self.card_slot_embedding(slot_index)
            tokens = tokens + self.enhancement_embedding(enhancement_id)
            tokens = tokens + self.edition_embedding(edition_id)
            tokens = tokens + self.seal_embedding(seal_id)
            tokens = tokens + self.empty_flag_embedding(is_empty)
            tokens = tokens + self.face_down_flag_embedding(is_face_down)
            tokens = tokens + self.selected_flag_embedding(is_selected)
            tokens = self.card_norm(self.dropout(tokens))

            padding_mask = is_empty.bool()
            tokens = self._zero_masked_tokens(tokens, padding_mask)
            return tokens, padding_mask

        def _build_state_tokens(
            self,
            observation: Mapping[str, Any],
        ) -> torch.Tensor:
            run_token = observation["run_token"]
            deck_token = observation["deck_token"]
            hand_levels = observation["hand_levels"]

            run_features = torch.stack(
                (
                    self._signed_log1p(run_token["money"].float()),
                    run_token["hands_left"].float() / 12.0,
                    run_token["discards_left"].float() / 10.0,
                    torch.log1p(run_token["chips_needed"].float()) / 16.0,
                    torch.log1p(run_token["round_chips_scored"].float()) / 16.0,
                    run_token["ante"].float() / 20.0,
                    run_token["round"].float() / 3.0,
                    run_token["phase"].float() / 3.0,
                ),
                dim=-1,
            )

            deck_features = torch.cat(
                (
                    deck_token["draw_pile_size"].float().unsqueeze(-1) / 52.0,
                    deck_token["remaining_rank_histogram"].float() / 4.0,
                    deck_token["remaining_suit_histogram"].float() / 13.0,
                ),
                dim=-1,
            )

            hand_level_features = torch.log1p(hand_levels.float()) / 5.0

            run_state = self.run_token_encoder(run_features).unsqueeze(1)
            deck_state = self.deck_token_encoder(deck_features).unsqueeze(1)
            hand_state = self.hand_level_encoder(hand_level_features).unsqueeze(1)

            state_tokens = torch.cat((run_state, deck_state, hand_state), dim=1)
            type_ids = torch.arange(3, device=state_tokens.device).unsqueeze(0)
            state_tokens = state_tokens + self.state_token_type_embedding(type_ids)
            return self.state_norm(self.dropout(state_tokens))

        def _build_modifier_tokens(
            self,
            observation: Mapping[str, Any],
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            boss_token = observation["boss_token"]
            joker_tokens = observation["joker_tokens"]

            boss_active = boss_token["boss_blind_active"].long().squeeze(-1).clamp(0, 1)
            boss_type = boss_token["boss_blind_type"].long().clamp(0, self.boss_type_embedding.num_embeddings - 1)
            boss_embedding = self.boss_type_embedding(boss_type)
            boss_embedding = boss_embedding + self.boss_active_embedding(boss_active)
            boss_embedding = boss_embedding + self.modifier_type_embedding(
                torch.zeros_like(boss_active)
            )
            boss_embedding = boss_embedding.unsqueeze(1)

            joker_ids = joker_tokens["joker_id"].long().clamp(0, self.joker_id_embedding.num_embeddings - 1)
            is_empty = joker_tokens["is_empty"].long().clamp(0, 1)
            is_disabled = joker_tokens["is_disabled"].long().clamp(0, 1)

            batch_size = joker_ids.shape[0]
            slot_index = torch.arange(MAX_JOKER_SLOTS, device=joker_ids.device).expand(batch_size, -1)
            joker_embedding = self.joker_id_embedding(joker_ids)
            joker_embedding = joker_embedding + self.joker_slot_embedding(slot_index)
            joker_embedding = joker_embedding + self.joker_empty_embedding(is_empty)
            joker_embedding = joker_embedding + self.joker_disabled_embedding(is_disabled)
            joker_embedding = joker_embedding + self.modifier_type_embedding(
                torch.ones_like(joker_ids)
            )

            modifier_tokens = torch.cat((boss_embedding, joker_embedding), dim=1)
            modifier_padding_mask = torch.cat(
                (
                    torch.zeros((batch_size, 1), dtype=torch.bool, device=joker_ids.device),
                    is_empty.bool(),
                ),
                dim=1,
            )
            modifier_tokens = self.modifier_norm(self.dropout(modifier_tokens))
            modifier_tokens = self._zero_masked_tokens(modifier_tokens, modifier_padding_mask)
            return modifier_tokens, modifier_padding_mask

        def forward(self, observation: Mapping[str, Any]) -> EmbeddingOutput:
            card_tokens, card_padding_mask = self._build_card_tokens(observation["hand_tokens"])
            state_tokens = self._build_state_tokens(observation)
            modifier_tokens, modifier_padding_mask = self._build_modifier_tokens(observation)
            return EmbeddingOutput(
                card_tokens=card_tokens,
                card_padding_mask=card_padding_mask,
                state_tokens=state_tokens,
                modifier_tokens=modifier_tokens,
                modifier_padding_mask=modifier_padding_mask,
            )


else:
    class DenseTokenEncoder:  # pragma: no cover - exercised only without torch
        """Placeholder used when torch is unavailable."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_torch()


    class BalatroEmbeddingStack:  # pragma: no cover - exercised only without torch
        """Placeholder used when torch is unavailable."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_torch()
