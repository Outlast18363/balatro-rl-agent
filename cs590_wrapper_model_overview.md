# CS590 Wrapper And Model Overview

This note summarizes the new files added under `cs590_env` and `cs590_src`.

The main design choice is:

- `cs590_env` exposes **embedding-ready token fields**, not learned embeddings
- `cs590_src` defines the **high-level model skeleton**, not a fully fixed neural implementation

This keeps the simulator-facing code stable while leaving the exact embedding fusion and transformer internals flexible.

## High-Level Architecture Mapping

The wrapper outputs the state groups required by the planned model:

- `hand_tokens` -> card embedding block
- `run_token` -> run-state embedding block
- `deck_token` -> deck-state embedding block
- `hand_levels` -> hand-level embedding block
- `boss_token` -> boss embedding block
- `joker_tokens` -> joker embedding block
- `action_masks` -> legality support for policy sampling and execution

The model skeleton then maps those blocks into:

- embedding blocks in `cs590_src/embeddings.py`
- backbone stage definitions in `cs590_src/backbone.py`
- actor-critic head definitions in `cs590_src/policy.py`

## `cs590_env`

### `cs590_env/__init__.py`

Purpose:
- public entry point for the wrapper package
- re-exports the wrapper class and schema helpers

Supports:
- wrapper construction from other modules
- cleaner imports such as `from cs590_env import BalatroArchWrapper`

### `cs590_env/arch_schema.py`

Purpose:
- defines the wrapper contract in one place
- centralizes payload keys, slot counts, padding values, and wrapper action/observation spaces

Supports:
- stable interface between the environment wrapper and model code
- consistent assumptions in tests and future training code

Important contents:
- fixed sizes such as `MAX_HAND_SIZE`, `MAX_JOKER_SLOTS`, and `HAND_LEVEL_COUNT`
- nested wrapper observation keys
- structured wrapper action space:
  - `selection`: 8-slot binary selection vector
  - `execute`: `PLAY` or `DISCARD`

Model correspondence:
- this file is the contract layer between simulator state and all embedding blocks

### `cs590_env/balatro_arch_wrapper.py`

Purpose:
- wraps `BalatroEnv` and exposes the model-facing payload
- translates the structured wrapper action into the flat simulator action sequence
- enforces wrapper-side legality for `PLAY` and `DISCARD`

Supports:
- card token extraction from raw hand state and `CardState`
- run-state packaging
- deck histogram construction
- hand-level export
- boss token export
- joker token export, including disabled-slot handling
- factorized legality masks

Important public methods:
- `reset()`
- `step(action)`
- `get_arch_observation()`
- `get_arch_masks()`
- `translate_arch_action_to_env_action(action)`

Wrapper payload returned by `reset()` and `step()`:

```python
{
    "hand_tokens": {...},
    "run_token": {...},
    "deck_token": {...},
    "hand_levels": ...,
    "boss_token": {...},
    "joker_tokens": {...},
    "action_masks": {...},
}
```

Model correspondence:
- `hand_tokens` supports the card token stream
- `run_token`, `deck_token`, and `hand_levels` support the combat-state sequence
- `boss_token` and `joker_tokens` support the modifier sequence
- `action_masks` supports legality-aware action sampling

Important implementation notes:
- hidden face-down cards do not leak card identity or modifier information
- `PLAY` is only allowed when `1 <= selected_count <= 5`
- `DISCARD` is only allowed when at least one card is selected and discards remain
- the wrapper can auto-advance non-combat phases so model code can stay focused on combat observations

### `cs590_env/test_balatro_arch_wrapper.py`

Purpose:
- example-driven test file for the wrapper
- doubles as lightweight usage documentation

Supports:
- schema validation
- payload shape validation
- legality and padding checks
- action translation checks

What it demonstrates:
- how to construct the wrapper
- how to call `reset()`
- how to call `step()` with a structured action
- how to call `translate_arch_action_to_env_action()`
- what the returned payload shape looks like

Model correspondence:
- validates the environment contract before the model consumes it

## `cs590_src`

### `cs590_src/__init__.py`

Purpose:
- public entry point for the high-level model skeleton
- re-exports embedding, backbone, and policy skeleton helpers

Supports:
- clean imports from one package root

### `cs590_src/embeddings.py`

Purpose:
- defines the high-level embedding blocks already fixed by the architecture
- records which wrapper fields belong to which embedding block

Supports:
- a stable partition between wrapper outputs and future learned embedding modules
- future implementation work without changing the wrapper contract

Defined embedding blocks:
- `card_embedding`
- `run_state_embedding`
- `deck_state_embedding`
- `hand_level_embedding`
- `boss_embedding`
- `joker_embedding`

Helper functions:
- `get_embedding_block_specs()`
- `split_wrapper_observation(observation)`

Model correspondence:
- directly represents the embedding layer in your architecture diagram

### `cs590_src/backbone.py`

Purpose:
- defines the high-level backbone stage boundaries
- keeps the current implementation at the architecture level instead of fixing exact layer details

Supports:
- future transformer implementation while preserving the intended stage structure

Defined stage specs:
- `stage1_card_self_attention`
- `stage2_state_cross_attention`
- `stage1_modifier_self_attention`
- `stage3_modifier_cross_attention`
- `global_pooling`

Model correspondence:
- directly represents the 3-stage transformer backbone in your architecture diagram

### `cs590_src/policy.py`

Purpose:
- defines the high-level actor-critic head structure
- bundles embedding blocks, backbone stages, and policy heads into one model skeleton summary

Supports:
- future implementation of the actual policy/value network without fixing framework-specific code yet

Defined heads:
- `card_selection_head`
- `execution_head`
- `critic_head`

Main helper:
- `build_model_skeleton()`

Model correspondence:
- directly represents the output heads in your architecture diagram

## Current Intended Usage

### Wrapper side

Typical usage:

```python
from balatro_gym.balatro_env_2 import BalatroEnv
from cs590_env import BalatroArchWrapper, ArchExecuteAction

wrapper = BalatroArchWrapper(BalatroEnv(seed=7))
observation, info = wrapper.reset(seed=7)

action = {
    "selection": [1, 0, 0, 0, 0, 0, 0, 0],
    "execute": ArchExecuteAction.PLAY,
}

next_observation, reward, terminated, truncated, info = wrapper.step(action)
```

### Model side

Typical usage today:

```python
from cs590_src import build_model_skeleton, split_wrapper_observation

model_skeleton = build_model_skeleton()
embedding_inputs = split_wrapper_observation(observation)
```

This does **not** run a neural network yet. It only gives the agreed structure for future implementation.

## Scope And Non-Goals

The current implementation intentionally does **not** do the following:

- it does not modify the base simulator into a model-specific env
- it does not add learned embedding modules yet
- it does not commit to a final inter-block fusion strategy
- it does not add legacy training integration
- it does not use `old_train_script`

## Validation

Wrapper tests were added in:

- `cs590_env/test_balatro_arch_wrapper.py`

Run them with:

```bash
python -m unittest cs590_env.test_balatro_arch_wrapper
```
