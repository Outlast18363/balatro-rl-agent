# Combat PPO: rollouts and training log

This document describes how **`TrainCombat.ipynb`** collects rollouts and what each field in the periodic training log means. Implementation lives in **`TrainCombat.ipynb`** (loop), **`BalatroPPO.ipynb`** (`PPOConfig`, `ppo_update`, `compute_log_prob_and_entropy`), and **`cs590_env/combat_env.py`** (`PooledCombatEnv`, `VecRolloutBuffer`, `compute_gae_vectorized`, observation helpers).

---

## High-level training iteration

Each **iteration** does two phases:

1. **Rollout collection** — For `rollout_steps` time steps, all `num_envs` workers act in parallel with the **current** policy (eval mode, no gradients). Transitions are written to a `VecRolloutBuffer` of shape `(T, N)` in time × env.
2. **Advantage computation + PPO update** — After the horizon, the policy is run once more on the **post-rollout** observations to get bootstrap values `V(s_{T})`. GAE produces advantages and returns for every stored step. Data is flattened to batch size `T × N`, then **`ppo_update`** runs several epochs of clipped surrogate loss with minibatches.

So one iteration uses **`rollout_steps × num_envs`** environment steps for the buffer (defaults: 16 × 64 = 1024), plus one extra forward pass per env for `next_vals`.

---

## Parallel environments and starting states

- **`AsyncVectorEnv`** runs **`num_envs`** separate **`PooledCombatEnv`** instances (each in its own process when using Gymnasium’s async vector API).
- **`make_pooled_combat_env(snapshot_pool, pool_seed=BASE_SEED + i)`** gives each worker a **different pool RNG seed** so they do not draw the same sequence of snapshots.
- On each **`reset()`**, a worker picks a **new** random snapshot from **`load_snapshot_pool()`**, loads it into **`BalatroEnv`**, and **`CombatActionWrapper`** advances until the observation is **combat-only** (same stack as training: phase wrapper + combat wrapper).

**Snapshot and RNG behavior on every reset**

- **New snapshot each time** — `PooledCombatEnv.reset` draws a fresh pool index every call. The env does **not** stay on the previous snapshot across resets; each reset is an independent draw from the shared pool (uniform over indices, using that worker’s pool RNG).
- **Randomness does not carry across resets** — After choosing the snapshot, `reset` samples a new **`fresh_seed`**, runs **`BalatroEnv.reset(seed=fresh_seed)`**, **`load_state(deepcopy(snapshot))`**, then replaces the base env’s RNG with **`DeterministicRNG(fresh_seed)`**. The next episode therefore does **not** inherit the prior episode’s in-game RNG state; only the saved snapshot contents carry over, and all post-load randomness is keyed off the new seed.

When a combat episode **ends**, `PooledCombatEnv.step` may **auto-reset** (which runs the same logic above), so the next observation always comes from **another** random snapshot and **new** seeds—not a continuation of the same table under the old RNG.

---

## Single rollout step (what the policy does)

For each time index `t` and all envs at once:

1. **Observation** — Vectorized dict `obs_np` (numpy, per-key shape `(N, …)`) is converted with **`dict_to_tensors`** to the training device.
2. **Forward** — **`CombatPPOAgent`** returns:
   - **`sel_logits`**: `(N, MAX_HAND_SIZE, 2)` — per-card binary “select / not select”.
   - **`exec_logits`**: `(N, 2)` — play vs discard.
   - **`value`**: `(N, 1)` state value.
3. **Masking** — **`get_card_mask`** marks real cards (`hand_card_ids >= 0`); **`mask_logits`** prevents selecting empty padding slots.
4. **Sampling** — Two **`Categorical`** distributions sample `card_sels` and `executions`. **Log-prob** is the sum of per-card selection log-probs plus the execution log-prob (this is what PPO treats as a single action’s log-probability).
5. **Environment step** — Actions are packed as **`np.concatenate([card_sels, executions[:, None]], axis=-1)`**, matching **`PooledCombatEnv`**’s **`MultiBinary(MAX_HAND_SIZE + 1)`** layout (first slots: card toggles, last bit: play=0 / discard=1).
6. **Storage** — **`buffer.store_step`** records obs, actions, log-probs, values, rewards, and **dones** (`terminated | truncated` from the vector env).

Invalid selections (e.g. not between 1 and 5 cards) are rejected inside **`CombatActionWrapper`** with reward **-1** and no env transition; those steps still appear in the buffer like any other step.

---

## Rollout buffer and GAE

- **`VecRolloutBuffer(T, N, device)`** holds **`T = rollout_steps`** steps for **`N = num_envs`** envs: rewards, values, dones, log-probs, discrete actions, and a full observation dict replicated over time.
- **`flatten()`** reshapes time × env into a single batch of size **`T × N`** for PPO.
- **`compute_gae_vectorized`** takes **`rewards` `(T, N)`**, **`values` `(T, N)`**, **`next_values` `(N,)`** (bootstrap at the end of the horizon), **`dones` `(T, N)`**, and **`gamma` / `gae_lambda`**. It runs a **reverse** time sweep **per env** (vectorized over `N`), then sets **`returns = advantages + values`** and **normalizes advantages** to zero mean and unit variance **across all `T×N` entries** (with `1e-8` for numerical stability).

`dones` mask the bootstrap term so terminal transitions do not leak value from the next episode.

---

## PPO update (what the logged losses measure)

`ppo_update` (in **`BalatroPPO.ipynb`**) for **`ppo_epochs`** passes over shuffled data in **`num_minibatches`** chunks. For each minibatch it recomputes log-probs and entropy with the **current** policy, then:

- **Policy loss** — Clipped surrogate: `ratio = exp(new_logp - old_logp)`, `pg_loss = -mean(min(ratio * A, clip(ratio) * A))`.
- **Value loss** — **`MSE(V(s), return)`** (PyTorch `F.mse_loss` mean over the minibatch).
- **Entropy** — Mean of **sum of entropies** over per-card categorical heads plus the execution head (same decomposition as at sampling time).

Total optimization objective per minibatch:

`loss = pg_loss + c_value * value_loss - c_entropy * entropy_mean`

Gradients are clipped with **`max_grad_norm`**. The training log prints the **mean** `pg_loss`, `value_loss`, and **entropy** averaged over all minibatch gradient steps in that iteration’s `ppo_update` call (not weighted by `c_value` / `c_entropy`).

---

## Training log line (printed every 5 iterations, and on iteration 1)

Example:

```text
[iter   10]  reward=+4.13  len=5.8  win=86.0%  pg=-0.0003  vf=5.5813  ent=5.5905  episodes=1652
```

| Field | Meaning |
|--------|--------|
| **`iter`** | PPO outer-loop iteration index (one rollout + one GAE/PPO update). |
| **`reward`** | Mean **undiscounted return** of the **last up to 50 finished combat episodes** across all envs (`ep_returns[-50:]`). Each episode return is the **sum of step rewards** from reset until `done` for that worker. |
| **`len`** | Mean **number of agent steps** (factored combat actions) in those same **up to 50** completed episodes (`ep_lengths[-50:]`). |
| **`win`** | Percentage of those episodes where **`infos['combat_ended']`** was true when the episode ended (`ep_wins`). In **`CombatActionWrapper`**, this is set when combat resolves by leaving the **COMBAT** phase after a legal play (e.g. blind cleared), as opposed to dying or other early termination paths that may not set the flag. |
| **`pg`** | Mean **clipped surrogate policy gradient loss** from `ppo_update` (scalar reported as `losses['pg_loss']`). Negative values are normal when advantages and policy updates align favorably with the surrogate. |
| **`vf`** | Mean **value-function MSE** (`losses['value_loss']`) between predicted values and GAE **returns** on the same rollout batch. |
| **`ent`** | Mean **policy entropy** (`losses['entropy']`) across the policy heads on that update; typically **decreases** as the policy becomes sharper. |
| **`episodes`** | **Total** number of completed episodes recorded since training started (length of `ep_returns`), not “episodes this iteration”. |

---

## Default hyperparameters (`PPOConfig` in `BalatroPPO.ipynb`)

| Parameter | Default | Role in rollouts / PPO |
|-----------|---------|-------------------------|
| `num_envs` | 64 | Parallel workers. |
| `rollout_steps` | 16 | Time horizon `T` per iteration per env. |
| `lr` | 2.5e-4 | Adam learning rate. |
| `gamma` | 0.99 | Discount in GAE / bootstrap. |
| `gae_lambda` | 0.95 | Bias–variance tradeoff in GAE. |
| `clip_eps` | 0.2 | PPO trust region clip on probability ratio. |
| `ppo_epochs` | 4 | Full passes over shuffled `T×N` data per iteration. |
| `num_minibatches` | 4 | Minibatch count per epoch (`B // num_minibatches` must divide evenly; partial tail is skipped). |
| `c_value` | 0.5 | Weight on value loss in total loss. |
| `c_entropy` | 0.01 | Weight on entropy **bonus** (encourages exploration). |
| `max_grad_norm` | 0.5 | Global norm clipping after each minibatch step. |

Architecture fields (`d_model`, `nhead`, `dim_ff`, `dropout`) only affect the network, not the log field definitions.

---

## Checkpoints (same notebook)

- **`CKPT_EVERY`** (e.g. 100): periodic saves under `checkpoints/combat_ppo_iter_{iteration}.pt` including model, optimizer, `iteration`, and `config`.
- Final cell: `checkpoints/combat_ppo.pt` after `vec_env.close()`.

For **resume**, `RESUME_CHECKPOINT` reloads weights, optimizer, and sets `start_iteration` from the checkpoint’s `iteration` field.
