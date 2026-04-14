# Single-source combat policy (revised)

## User choice: canonical module location

- **Single source of truth:** [`cs590_src/combat_agent_model.py`](cs590_src/combat_agent_model.py) (not under `cs590_env/`).
- **Remove** the duplicate file [`cs590_env/combat_agent_model.py`](cs590_env/combat_agent_model.py) after the move (one copy only).

## Importability

[`cs590_src/`](cs590_src/) currently has only notebooks and no `__init__.py`. To support `from cs590_src.combat_agent_model import CombatPPOAgent` from [`cs590_env/util.py`](cs590_env/util.py), tests, and Colab (with repo root on `sys.path`):

- Add a minimal [`cs590_src/__init__.py`](cs590_src/__init__.py) (empty or docstring-only) so `cs590_src` is a normal package when the repository root is on `PYTHONPATH` (already true in `TrainCombat.ipynb`).

## Steps

### 1. Move and canonicalize

- Move the existing implementation from [`cs590_env/combat_agent_model.py`](cs590_env/combat_agent_model.py) to [`cs590_src/combat_agent_model.py`](cs590_src/combat_agent_model.py).
- Update the module docstring: state this file is the **only** policy implementation.
- Add **`__all__`** for public exports (`CombatPPOAgent`, embeddings, backbone, heads, etc.) for notebook shims and `import *` if desired.
- Delete [`cs590_env/combat_agent_model.py`](cs590_env/combat_agent_model.py).

### 2. Fix all imports

- [`cs590_env/util.py`](cs590_env/util.py): `from cs590_src.combat_agent_model import CombatPPOAgent` inside `load_combat_ppo_agent` (same lazy pattern as today).
- Grep the repo for `cs590_env.combat_agent_model` or `combat_agent_model` and point every reference to `cs590_src.combat_agent_model`.

### 3. Notebooks

- [`cs590_src/BalatroPPO.ipynb`](cs590_src/BalatroPPO.ipynb): remove `%run cs590_src/CombatAgent.ipynb`; add `from cs590_src.combat_agent_model import CombatPPOAgent`.
- [`cs590_src/CombatAgent.ipynb`](cs590_src/CombatAgent.ipynb): strip duplicate model cells; add markdown pointing to `cs590_src/combat_agent_model.py`; one code cell `from cs590_src.combat_agent_model import ...` (re-export for anyone who still `%run`s this notebook).

### 4. [`cs590_src/TrainCombat.ipynb`](cs590_src/TrainCombat.ipynb)

- No change required if it only `%run`s `BalatroPPO.ipynb`; optional one-line comment that the policy lives in `cs590_src/combat_agent_model.py`.

### 5. Sanity

- Grep for `CombatAgent`, `%run.*CombatAgent`, `cs590_env.combat_agent_model`.
- [`cs590_env/__init__.py`](cs590_env/__init__.py): no need to export `CombatPPOAgent` unless you want it; weight interpreter stays on `util`.

## Dependency note

`cs590_env` will **import** `cs590_src` (sibling under repo root). That is fine when the project is run with the repo root on `sys.path` (current training / Colab setup). Avoid circular imports: keep `combat_agent_model.py` depending only on `torch`, `cs590_env.schema`, `balatro_gym.*` — not on `cs590_env.util`.
