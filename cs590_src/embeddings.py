"""High-level embedding block definitions for the CS590 model stack."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple


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
