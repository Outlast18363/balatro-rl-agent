"""Actor-critic specs and runtime policy network for the CS590 model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, MutableMapping, Tuple

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - handled by runtime guards
    torch = None
    nn = None

from cs590_src.backbone import (
    BackboneOutput,
    BackboneStageSpec,
    BalatroTransformerBackbone,
    get_backbone_stage_specs,
)
from cs590_src.embeddings import (
    EmbeddingBlockSpec,
    BalatroEmbeddingStack,
    get_embedding_block_specs,
)


@dataclass(frozen=True)
class PolicyHeadSpec:
    """Describe one model head attached to the shared backbone.

    Parameters:
        name: Stable head name used across the architecture.
        input_source: Backbone output consumed by the head.
        output_role: Semantic purpose of the head output.
    Returns:
        A lightweight head specification for the model skeleton.
    """

    name: str
    input_source: str
    output_role: str


@dataclass(frozen=True)
class BalatroModelSkeleton:
    """Bundle the high-level building blocks of the planned model.

    Parameters:
        embedding_blocks: The embedding partitions already fixed by the wrapper.
        backbone_stages: The transformer stage boundaries from the architecture.
        policy_heads: The actor-critic heads attached after the backbone.
    Returns:
        A high-level architecture summary object for the current skeleton.
    """

    embedding_blocks: Tuple[EmbeddingBlockSpec, ...]
    backbone_stages: Tuple[BackboneStageSpec, ...]
    policy_heads: Tuple[PolicyHeadSpec, ...]


@dataclass(frozen=True)
class BalatroModelConfig:
    """Hyperparameters for the minimal runnable CS590 policy network."""

    embed_dim: int = 128
    num_heads: int = 4
    dropout: float = 0.1
    max_boss_type: int = 65
    max_joker_id: int = 256
    invalid_logit: float = -1e9


@dataclass
class PolicyOutput:
    """Actor-critic outputs produced by the runtime policy network."""

    selection_logits: Any
    masked_selection_logits: Any
    execution_logits: Any
    masked_execution_logits: Any
    value: Any
    pooled_state: Any
    card_features: Any


CARD_SELECTION_HEAD = PolicyHeadSpec(
    name="card_selection_head",
    input_source="per_card_backbone_output",
    output_role="Per-slot card-selection logits.",
)

EXECUTION_HEAD = PolicyHeadSpec(
    name="execution_head",
    input_source="global_pooled_context",
    output_role="Play-versus-discard execution logits.",
)

CRITIC_HEAD = PolicyHeadSpec(
    name="critic_head",
    input_source="global_pooled_context",
    output_role="Scalar state-value estimate.",
)


def get_policy_head_specs() -> Tuple[PolicyHeadSpec, ...]:
    """Return the current actor-critic head boundaries.

    Returns:
        A tuple containing the card-selection, execution, and critic heads.
    """

    return (
        CARD_SELECTION_HEAD,
        EXECUTION_HEAD,
        CRITIC_HEAD,
    )


def build_model_skeleton() -> BalatroModelSkeleton:
    """Assemble the current high-level model skeleton.

    Returns:
        A `BalatroModelSkeleton` summarizing the embedding blocks, transformer
        stages, and actor-critic heads without fixing fusion details yet.
    """

    return BalatroModelSkeleton(
        embedding_blocks=get_embedding_block_specs(),
        backbone_stages=get_backbone_stage_specs(),
        policy_heads=get_policy_head_specs(),
    )


def _require_torch() -> None:
    """Raise a friendly error when runtime modules are used without torch."""

    if nn is None:
        raise ModuleNotFoundError(
            "BalatroPolicyNetwork requires PyTorch. Install `torch` to use the "
            "runtime model implementation."
        )


if nn is not None:
    OBSERVATION_RANKS = {
        ("hand_tokens", "card_id"): 2,
        ("hand_tokens", "is_empty"): 2,
        ("hand_tokens", "is_face_down"): 2,
        ("hand_tokens", "enhancement_id"): 2,
        ("hand_tokens", "edition_id"): 2,
        ("hand_tokens", "seal_id"): 2,
        ("hand_tokens", "is_selected"): 2,
        ("run_token", "money"): 1,
        ("run_token", "hands_left"): 1,
        ("run_token", "discards_left"): 1,
        ("run_token", "chips_needed"): 1,
        ("run_token", "round_chips_scored"): 1,
        ("run_token", "ante"): 1,
        ("run_token", "round"): 1,
        ("run_token", "phase"): 1,
        ("deck_token", "draw_pile_size"): 1,
        ("deck_token", "remaining_rank_histogram"): 2,
        ("deck_token", "remaining_suit_histogram"): 2,
        ("hand_levels",): 2,
        ("boss_token", "boss_blind_active"): 2,
        ("boss_token", "boss_blind_type"): 1,
        ("joker_tokens", "joker_id"): 2,
        ("joker_tokens", "is_empty"): 2,
        ("joker_tokens", "is_disabled"): 2,
        ("action_masks", "card_select_mask"): 2,
        ("action_masks", "play_allowed"): 2,
        ("action_masks", "discard_allowed"): 2,
        ("action_masks", "selected_count"): 1,
    }

    def _ensure_rank(tensor: torch.Tensor, expected_rank: int) -> torch.Tensor:
        while tensor.ndim < expected_rank:
            tensor = tensor.unsqueeze(0)
        return tensor


    def _leaf_to_tensor(
        value: Any,
        *,
        expected_rank: int,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        if torch.is_tensor(value):
            tensor = value.to(device=device)
        else:
            tensor = torch.as_tensor(value, device=device)
        return _ensure_rank(tensor, expected_rank)


    def prepare_observation_batch(
        observation: Mapping[str, Any],
        *,
        device: torch.device | None = None,
    ) -> Dict[str, Any]:
        """Convert a wrapper observation into a batched tensor tree."""

        prepared: Dict[str, Any] = {}
        for path, expected_rank in OBSERVATION_RANKS.items():
            current_in: Any = observation
            current_out: MutableMapping[str, Any] = prepared
            for key in path[:-1]:
                current_in = current_in[key]
                current_out = current_out.setdefault(key, {})
            leaf_key = path[-1]
            leaf_value = current_in[leaf_key]
            current_out[leaf_key] = _leaf_to_tensor(
                leaf_value,
                expected_rank=expected_rank,
                device=device,
            )
        return prepared


    def _apply_mask_to_logits(
        logits: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        invalid_logit: float,
    ) -> torch.Tensor:
        valid_mask = valid_mask.to(device=logits.device, dtype=torch.bool)
        masked_logits = logits.masked_fill(~valid_mask, invalid_logit)
        if valid_mask.ndim == 2:
            all_invalid = (~valid_mask).all(dim=-1, keepdim=True)
            masked_logits = torch.where(all_invalid, logits, masked_logits)
        return masked_logits


    class PredictionHead(nn.Module):
        """Small MLP head used for policy and value predictions."""

        def __init__(self, input_dim: int, output_dim: int, dropout: float) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, input_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(input_dim, output_dim),
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.net(inputs)


    class BalatroPolicyNetwork(nn.Module):
        """Minimal runnable actor-critic model aligned with the CS590 design."""

        def __init__(self, config: BalatroModelConfig | None = None) -> None:
            super().__init__()
            self.config = config or BalatroModelConfig()
            self.embedding_stack = BalatroEmbeddingStack(
                embed_dim=self.config.embed_dim,
                dropout=self.config.dropout,
                max_boss_type=self.config.max_boss_type,
                max_joker_id=self.config.max_joker_id,
            )
            self.backbone = BalatroTransformerBackbone(
                embed_dim=self.config.embed_dim,
                num_heads=self.config.num_heads,
                dropout=self.config.dropout,
            )
            self.selection_head = PredictionHead(
                input_dim=self.config.embed_dim,
                output_dim=1,
                dropout=self.config.dropout,
            )
            self.execution_head = PredictionHead(
                input_dim=self.config.embed_dim,
                output_dim=2,
                dropout=self.config.dropout,
            )
            self.value_head = PredictionHead(
                input_dim=self.config.embed_dim,
                output_dim=1,
                dropout=self.config.dropout,
            )

        @property
        def device(self) -> torch.device:
            return next(self.parameters()).device

        def _mask_selection_logits(
            self,
            selection_logits: torch.Tensor,
            action_masks: Mapping[str, torch.Tensor],
        ) -> torch.Tensor:
            card_select_mask = action_masks["card_select_mask"].bool()
            return _apply_mask_to_logits(
                selection_logits,
                card_select_mask,
                invalid_logit=self.config.invalid_logit,
            )

        def _mask_execution_logits(
            self,
            execution_logits: torch.Tensor,
            action_masks: Mapping[str, torch.Tensor],
        ) -> torch.Tensor:
            execution_mask = torch.cat(
                (
                    action_masks["play_allowed"].bool(),
                    action_masks["discard_allowed"].bool(),
                ),
                dim=-1,
            )
            return _apply_mask_to_logits(
                execution_logits,
                execution_mask,
                invalid_logit=self.config.invalid_logit,
            )

        def forward(
            self,
            observation: Mapping[str, Any],
            *,
            apply_legality_masks: bool = True,
        ) -> PolicyOutput:
            prepared_observation = prepare_observation_batch(
                observation,
                device=self.device,
            )
            embedded_observation = self.embedding_stack(prepared_observation)
            backbone_output: BackboneOutput = self.backbone(embedded_observation)

            selection_logits = self.selection_head(backbone_output.card_tokens).squeeze(-1)
            execution_logits = self.execution_head(backbone_output.pooled_state)
            value = self.value_head(backbone_output.pooled_state).squeeze(-1)

            if apply_legality_masks:
                masked_selection_logits = self._mask_selection_logits(
                    selection_logits,
                    prepared_observation["action_masks"],
                )
                masked_execution_logits = self._mask_execution_logits(
                    execution_logits,
                    prepared_observation["action_masks"],
                )
            else:
                masked_selection_logits = selection_logits
                masked_execution_logits = execution_logits

            return PolicyOutput(
                selection_logits=selection_logits,
                masked_selection_logits=masked_selection_logits,
                execution_logits=execution_logits,
                masked_execution_logits=masked_execution_logits,
                value=value,
                pooled_state=backbone_output.pooled_state,
                card_features=backbone_output.card_tokens,
            )


    def build_policy_network(
        config: BalatroModelConfig | None = None,
    ) -> BalatroPolicyNetwork:
        """Create the minimal runnable actor-critic policy network."""

        return BalatroPolicyNetwork(config=config)


else:
    def prepare_observation_batch(  # pragma: no cover - exercised only without torch
        observation: Mapping[str, Any],
        *,
        device: Any | None = None,
    ) -> Dict[str, Any]:
        _require_torch()
        return {}


    class PredictionHead:  # pragma: no cover - exercised only without torch
        """Placeholder used when torch is unavailable."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_torch()


    class BalatroPolicyNetwork:  # pragma: no cover - exercised only without torch
        """Placeholder used when torch is unavailable."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_torch()


    def build_policy_network(  # pragma: no cover - exercised only without torch
        config: BalatroModelConfig | None = None,
    ) -> BalatroPolicyNetwork:
        _require_torch()
        raise AssertionError("unreachable")
