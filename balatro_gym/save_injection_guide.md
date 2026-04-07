# Save Injection & Snapshot Replay Guide

This guide covers two distinct mechanisms for restoring a Balatro environment
to a known state, and shows how to combine them for repeated rollouts from
the same starting position.

---

## Two Mechanisms — Don't Confuse Them

| | Save-File Injection | Environment Checkpoint |
|---|---|---|
| **Entry point** | `inject_save_into_balatro_env(...)` in `save_injection.py` | `env.save_state()` / `env.load_state(snap)` in `balatro_env_2.py` |
| **Input** | A `.jkr` save file (or pre-parsed dict) from the real Balatro game | An opaque snapshot dict produced by the env itself |
| **What it does** | Resets the env, then overwrites state fields (hand, deck, jokers, ante, money, …) to match the save file | Captures / restores *every* mutable sub-system (state, RNG, engine, game, boss-blind, joker-effects, shop) |
| **RNG behaviour** | Does **not** restore the original save's RNG stream — the env starts a fresh deterministic RNG seeded by `seed` | Restores the exact RNG position, so identical actions yield identical outcomes |
| **Typical use** | Bootstrap an env into a specific game situation extracted from a real Balatro session | Checkpoint a running env so you can replay the same decision point many times |

**Rule of thumb:** inject a save to *set the scene*, then immediately call
`save_state()` to get a reproducible checkpoint you can reload cheaply.

---

## Minimal End-to-End Example

The script below loads a real save file, checkpoints the injected state, and
plays until the blind is resolved.

```python
from pathlib import Path
import numpy as np
from balatro_gym.save_injection import inject_save_into_balatro_env

save_path = Path("game_files/first_blind_combat_save.jkr")

# 1. Inject the save into a fresh env (resets internally).
env, report = inject_save_into_balatro_env(save_path, seed=42)

# 2. Checkpoint right after injection for later replay.
snapshot = env.save_state()

# 3. Step with a simple first-legal-action policy until the blind ends.
obs = env._get_observation()
done = False
while not done:
    valid = np.where(obs["action_mask"])[0]
    if len(valid) == 0:
        break
    obs, reward, terminated, truncated, info = env.step(int(valid[0]))
    done = terminated or truncated

if info.get("beat_blind"):
    print("Blind cleared!")
elif info.get("failed"):
    print("Blind failed — episode over.")
```

### Replaying From the Same Snapshot

After the episode ends you can reload the checkpoint and try again with a
different policy. Because `save_state()` captured the RNG position, the
initial deal is identical on every reload.

```python
for attempt in range(5):
    env.load_state(snapshot)  # rewind to the post-injection state
    obs = env._get_observation()
    done = False
    while not done:
        valid = np.where(obs["action_mask"])[0]
        if len(valid) == 0:
            break
        action = int(np.random.choice(valid))  # swap in your policy here
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    result = "cleared" if info.get("beat_blind") else "failed"
    print(f"Attempt {attempt}: blind {result}")
```

---

## Replay Is Manual — There Is No Auto-Reset Mode

`BalatroEnv` has no built-in "truncated combat replay" mode.  When a blind
fails the episode ends with `terminated=True`; the env does **not**
automatically reload the injected save or any prior snapshot.

If you want multiple rollouts from the same starting position:

1. Hold the snapshot returned by `save_state()`.
2. After each terminal step, call `env.load_state(snapshot)` yourself.
3. Do **not** call `env.reset()` — that discards the injected state and
   starts a brand-new game from ante 1.

The `truncated` flag is never set by the blind-failure path and should not be
relied on as a replay trigger.

---

## Existing Test References

* **`tests/test_save_injection_dump.py`** — injects `.jkr` saves and dumps
  the resulting observation, state, and snapshot for manual inspection.
* **`tests/test_save_load_snapshot.py`** — round-trips `save_state()` /
  `load_state()`, verifies sub-system isolation, and confirms that stepping
  works correctly after a restore.
