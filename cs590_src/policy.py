"""High-level actor-critic skeleton for the CS590 Balatro model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from cs590_src.backbone import BackboneStageSpec, get_backbone_stage_specs
from cs590_src.embeddings import EmbeddingBlockSpec, get_embedding_block_specs


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
