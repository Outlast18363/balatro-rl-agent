"""High-level backbone stage definitions for the CS590 transformer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


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
