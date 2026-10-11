from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from block_blast_jax.env import _step
from block_blast_jax.rl.encode import encode_obs


def compute_value_targets(rewards, dones, last_value, discount: float):
    bootstrap = jnp.where(dones[-1], jnp.float32(0.0), last_value).astype(jnp.float32)

    def step(carry, x):
        reward, done = x
        v_next = jnp.where(done, jnp.float32(0.0), carry)
        v = reward + discount * v_next
        return v, v

    _, returns = jax.lax.scan(step, bootstrap, (rewards, dones), reverse=True)
    return returns


def self_play_iteration(
    run_mcts_jit,
    net_apply_fn,
    params,
    env,
    key: jax.Array,
    num_simulations: int,
    max_depth: int,
    max_moves: int,
    discount: float,
    pb_c_init: float = 1.25,
    pb_c_base: float = 19652.0,
    combo_coef: float = 0.0,
    visit_temperature: float = 1.0,
):
    B = env.num_envs
    P = env.num_pieces

    grids, trays, combos, streaks = [], [], [], []
    visits = []
    rewards_t = []
    dones_t = []
    done_lens_t = []
    done_scs_t = []
    visit_entropies = []
    visits_max_per_step = []
    root_values_per_step = []

    import sys as _sys
    import time as _time
    _t0 = _time.perf_counter()
    for move_step in range(max_moves):
        key, sub = jax.random.split(key)
        out = run_mcts_jit(
            params, env.state, sub,
            num_simulations=num_simulations, max_depth=max_depth,
            dirichlet_fraction=0.0, dirichlet_alpha=0.3,
            pb_c_init=pb_c_init, pb_c_base=pb_c_base, use_dirichlet=False,
        )
        action = out.action
        visit_dist = out.action_weights
        if visit_temperature != 1.0 and visit_temperature > 0.0:
            am = env.state["action_mask"].astype(jnp.float32)
            sharp = jnp.power(jnp.maximum(visit_dist, 1e-9), 1.0 / visit_temperature)
            sharp = sharp * am
            sharp = sharp / jnp.maximum(sharp.sum(axis=-1, keepdims=True), 1e-9)
            visit_dist = sharp
        if (move_step + 1) % 10 == 0:
            jax.block_until_ready(action)
            elapsed = _time.perf_counter() - _t0
            eta = elapsed / (move_step + 1) * (max_moves - move_step - 1)
            print(f"    [self-play] move-batch {move_step+1}/{max_moves} "
                  f"elapsed={elapsed:.0f}s ETA={eta:.0f}s",
                  file=_sys.stderr, flush=True)

        v = jnp.clip(visit_dist, 1e-9, 1.0)
        ent = -(v * jnp.log(v)).sum(axis=-1).mean()
        vmax = visit_dist.max(axis=-1).mean()
        try:
            rv = out.search_tree.summary().value.mean()
        except Exception:
            rv = jnp.float32(0.0)
        visit_entropies.append(ent)
        visits_max_per_step.append(vmax)
        root_values_per_step.append(rv)

        grids.append(env.state["grid"])
        trays.append(env.state["tray_idx"])
        combos.append(env.state["combo_count"])
        streaks.append(env.state["no_clear_streak"])
        visits.append(visit_dist)

        prev_ep_len = env.state["ep_len"]
        prev_score = env.state["score"]

        new_state, _, term, trunc, info = env.step(action)
        terminated = term | trunc
        combo_mult_f = info["combo_mult"].astype(jnp.float32)
        alive_r = jnp.float32(1.0) + jnp.float32(combo_coef) * (combo_mult_f - jnp.float32(1.0))
        reward = jnp.where(terminated, jnp.float32(0.0), alive_r)
        rewards_t.append(reward)
        dones_t.append(terminated)
        valid_int = info["cells_placed"].astype(jnp.int32)
        full_len = prev_ep_len.astype(jnp.int32) + valid_int
        full_sc = prev_score + info["score_delta"]
        done_lens_t.append(jnp.where(terminated, full_len, jnp.int32(-1)))
        done_scs_t.append(jnp.where(terminated, full_sc, jnp.int32(-1)))

    grids   = jnp.stack(grids, axis=0)
    trays   = jnp.stack(trays, axis=0)
    combos  = jnp.stack(combos, axis=0)
    streaks = jnp.stack(streaks, axis=0)
    visits  = jnp.stack(visits, axis=0)
    rewards = jnp.stack(rewards_t, axis=0)
    dones   = jnp.stack(dones_t, axis=0)

    obs_last = encode_obs(env.state, P)
    am_last = env.state["action_mask"]
    _, last_value = net_apply_fn(params, obs_last, am_last)

    returns = compute_value_targets(rewards, dones, last_value, discount)

    if done_lens_t:
        done_lens_arr = jnp.stack(done_lens_t, axis=0)
        done_scs_arr = jnp.stack(done_scs_t, axis=0)
        done_lens_np = np.asarray(done_lens_arr)
        done_scs_np = np.asarray(done_scs_arr)
        mask = done_lens_np >= 0
        ep_lens_collected = done_lens_np[mask]
        ep_scores_collected = done_scs_np[mask]
        n_episodes_done = int(mask.sum())
    else:
        ep_lens_collected = np.array([], dtype=np.int32)
        ep_scores_collected = np.array([], dtype=np.int32)
        n_episodes_done = 0
    ve_mean = float(jnp.stack(visit_entropies).mean()) if visit_entropies else 0.0
    vm_mean = float(jnp.stack(visits_max_per_step).mean()) if visits_max_per_step else 0.0
    rv_mean = float(jnp.stack(root_values_per_step).mean()) if root_values_per_step else 0.0

    return {
        "grids": grids, "trays": trays, "combos": combos, "streaks": streaks,
        "visits": visits, "values": returns,
        "n_episodes_done": n_episodes_done,
        "ep_lens": ep_lens_collected.astype(np.int32) if len(ep_lens_collected) else np.array([0], dtype=np.int32),
        "ep_scores": ep_scores_collected.astype(np.int32) if len(ep_scores_collected) else np.array([0], dtype=np.int32),
        "visit_entropy_mean": ve_mean,
        "visits_max_mean": vm_mean,
        "root_value_mean": rv_mean,
        "n_transitions": int(rewards.shape[0] * rewards.shape[1]),
        "n_simulations_total": int(rewards.shape[0] * rewards.shape[1] * num_simulations),
    }
