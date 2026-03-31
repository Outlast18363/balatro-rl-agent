"""Backbone specs and runtime transformer modules for the CS590 model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - handled by runtime guards
    torch = None
    nn = None

from cs590_src.embeddings import EmbeddingOutput


@dataclass(frozen=True)
class BackboneStageSpec:
    """Describe one logical stage of the transformer backbone.

    Parameters:
        name: Stable stage name used across documentation and code.
        query_source: Sequence or token group driving the stage query side.
        context_source: Sequence or token group supplying the attention context.
        description: Human-readable summary of the stage role.
    Returns:
        A stage specification for the model skeleton.
    """

    name: str
    query_source: str
    context_source: str
    description: str


@dataclass
class BackboneOutput:
    """Outputs exposed by the runtime transformer backbone."""

    card_tokens: Any
    pooled_state: Any
    state_tokens: Any
    modifier_tokens: Any
    card_padding_mask: Any


CARD_SELF_ATTENTION = BackboneStageSpec(
    name="stage1_card_self_attention",
    query_source="card_embedding",
    context_source="card_embedding",
    description="Cards attend to the rest of the current hand representation.",
)

STATE_CROSS_ATTENTION = BackboneStageSpec(
    name="stage2_state_cross_attention",
    query_source="card_embedding",
    context_source="run_state_embedding + deck_state_embedding + hand_level_embedding",
    description="Card tokens query the combat-state context sequence.",
)

MODIFIER_SELF_ATTENTION = BackboneStageSpec(
    name="stage1_modifier_self_attention",
    query_source="boss_embedding + joker_embedding",
    context_source="boss_embedding + joker_embedding",
    description="Modifier tokens attend within the modifier branch.",
)

MODIFIER_CROSS_ATTENTION = BackboneStageSpec(
    name="stage3_modifier_cross_attention",
    query_source="card_embedding",
    context_source="boss_embedding + joker_embedding",
    description="Card tokens query the modifier branch before pooling.",
)

GLOBAL_POOLING = BackboneStageSpec(
    name="global_pooling",
    query_source="card_embedding",
    context_source="pooled_context",
    description="Aggregate the final card-context sequence into one global state.",
)


def get_backbone_stage_specs() -> Tuple[BackboneStageSpec, ...]:
    """Return the ordered high-level stages of the backbone.

    Returns:
        A tuple describing the current backbone stage boundaries without fixing
        detailed layer internals yet.
    """

    return (
        CARD_SELF_ATTENTION,
        STATE_CROSS_ATTENTION,
        MODIFIER_SELF_ATTENTION,
        MODIFIER_CROSS_ATTENTION,
        GLOBAL_POOLING,
    )


def _require_torch() -> None:
    """Raise a friendly error when runtime modules are used without torch."""

    if nn is None:
        raise ModuleNotFoundError(
            "BalatroTransformerBackbone requires PyTorch. Install `torch` to use "
            "the runtime model implementation."
        )


if nn is not None:
    class FeedForwardBlock(nn.Module):
        """Two-layer MLP used inside each transformer block."""

        def __init__(self, embed_dim: int, hidden_dim: int, dropout: float) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, embed_dim),
                nn.Dropout(dropout),
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.net(inputs)


    class SelfAttentionBlock(nn.Module):
        """A small pre-norm self-attention block."""

        def __init__(self, embed_dim: int, num_heads: int, dropout: float) -> None:
            super().__init__()
            self.attn_norm = nn.LayerNorm(embed_dim)
            self.attn = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.ffn_norm = nn.LayerNorm(embed_dim)
            self.ffn = FeedForwardBlock(
                embed_dim=embed_dim,
                hidden_dim=embed_dim * 4,
                dropout=dropout,
            )

        def forward(
            self,
            tokens: torch.Tensor,
            *,
            padding_mask: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            normalized_tokens = self.attn_norm(tokens)
            attn_output, _ = self.attn(
                normalized_tokens,
                normalized_tokens,
                normalized_tokens,
                key_padding_mask=padding_mask,
                need_weights=False,
            )
            tokens = tokens + attn_output
            tokens = tokens + self.ffn(self.ffn_norm(tokens))
            if padding_mask is not None:
                tokens = tokens.masked_fill(padding_mask.unsqueeze(-1), 0.0)
            return tokens


    class CrossAttentionBlock(nn.Module):
        """A small pre-norm cross-attention block."""

        def __init__(self, embed_dim: int, num_heads: int, dropout: float) -> None:
            super().__init__()
            self.query_norm = nn.LayerNorm(embed_dim)
            self.context_norm = nn.LayerNorm(embed_dim)
            self.attn = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.ffn_norm = nn.LayerNorm(embed_dim)
            self.ffn = FeedForwardBlock(
                embed_dim=embed_dim,
                hidden_dim=embed_dim * 4,
                dropout=dropout,
            )

        def forward(
            self,
            query_tokens: torch.Tensor,
            context_tokens: torch.Tensor,
            *,
            query_padding_mask: Optional[torch.Tensor] = None,
            context_padding_mask: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            normalized_query = self.query_norm(query_tokens)
            normalized_context = self.context_norm(context_tokens)
            attn_output, _ = self.attn(
                normalized_query,
                normalized_context,
                normalized_context,
                key_padding_mask=context_padding_mask,
                need_weights=False,
            )
            query_tokens = query_tokens + attn_output
            query_tokens = query_tokens + self.ffn(self.ffn_norm(query_tokens))
            if query_padding_mask is not None:
                query_tokens = query_tokens.masked_fill(
                    query_padding_mask.unsqueeze(-1),
                    0.0,
                )
            return query_tokens


    class BalatroTransformerBackbone(nn.Module):
        """Minimal 3-stage transformer backbone matching the architecture doc."""

        def __init__(
            self,
            embed_dim: int,
            *,
            num_heads: int = 4,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            self.card_self_attention = SelfAttentionBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            self.state_cross_attention = CrossAttentionBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            self.modifier_self_attention = SelfAttentionBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            self.modifier_cross_attention = CrossAttentionBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            self.output_norm = nn.LayerNorm(embed_dim)

        @staticmethod
        def _masked_mean_pool(
            tokens: torch.Tensor,
            padding_mask: torch.Tensor,
        ) -> torch.Tensor:
            valid_mask = (~padding_mask).unsqueeze(-1).float()
            token_sum = (tokens * valid_mask).sum(dim=1)
            valid_count = valid_mask.sum(dim=1).clamp_min(1.0)
            return token_sum / valid_count

        def forward(self, embedded_observation: EmbeddingOutput) -> BackboneOutput:
            card_tokens = self.card_self_attention(
                embedded_observation.card_tokens,
                padding_mask=embedded_observation.card_padding_mask,
            )
            card_tokens = self.state_cross_attention(
                card_tokens,
                embedded_observation.state_tokens,
                query_padding_mask=embedded_observation.card_padding_mask,
            )

            modifier_tokens = self.modifier_self_attention(
                embedded_observation.modifier_tokens,
                padding_mask=embedded_observation.modifier_padding_mask,
            )
            card_tokens = self.modifier_cross_attention(
                card_tokens,
                modifier_tokens,
                query_padding_mask=embedded_observation.card_padding_mask,
                context_padding_mask=embedded_observation.modifier_padding_mask,
            )
            card_tokens = self.output_norm(card_tokens)
            pooled_state = self._masked_mean_pool(
                card_tokens,
                embedded_observation.card_padding_mask,
            )
            return BackboneOutput(
                card_tokens=card_tokens,
                pooled_state=pooled_state,
                state_tokens=embedded_observation.state_tokens,
                modifier_tokens=modifier_tokens,
                card_padding_mask=embedded_observation.card_padding_mask,
            )


else:
    class FeedForwardBlock:  # pragma: no cover - exercised only without torch
        """Placeholder used when torch is unavailable."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_torch()


    class SelfAttentionBlock:  # pragma: no cover - exercised only without torch
        """Placeholder used when torch is unavailable."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_torch()


    class CrossAttentionBlock:  # pragma: no cover - exercised only without torch
        """Placeholder used when torch is unavailable."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_torch()


    class BalatroTransformerBackbone:  # pragma: no cover - exercised only without torch
        """Placeholder used when torch is unavailable."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_torch()
