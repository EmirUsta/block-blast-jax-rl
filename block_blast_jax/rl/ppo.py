from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state as ts

from block_blast_jax.env import _step
from block_blast_jax.rl.encode import encode_obs, encode_obs_2d


def make_train_state(net, params, lr, max_grad_norm: float):
    tx = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(lr),
    )
    return ts.TrainState.create(apply_fn=net.apply, params=params, tx=tx)


def gae(rewards, values, dones, last_value, gamma: float, gae_lambda: float):
    def body(carry, x):
        next_value, next_adv = carry
        reward_t, value_t, done_t = x
        not_done = 1.0 - done_t.astype(jnp.float32)
        delta = reward_t + gamma * next_value * not_done - value_t
        adv = delta + gamma * gae_lambda * not_done * next_adv
        return (value_t, adv), adv

    init_carry = (last_value, jnp.zeros_like(last_value))
    _, adv = jax.lax.scan(body, init_carry, (rewards, values, dones), reverse=True)
    returns = adv + values
    return adv, returns


@partial(jax.jit, static_argnames=("auto_reset", "max_steps", "num_pieces", "num_actions",
                                     "rollout_len", "ppo_epochs", "num_minibatches",
                                     "reward_mode", "arch_type"))
def train_step(
    train_state: ts.TrainState,
    env_state: dict,
    key: jax.Array,
    pmasks_flat,
    valid_anchor_flat,
    pieces_ncells,
    row_masks,
    col_masks,
    line_bonus_table,
    cascade_table,
    auto_reset: bool,
    max_steps: int,
    num_pieces: int,
    num_actions: int,
    rollout_len: int,
    ppo_epochs: int,
    num_minibatches: int,
    death_penalty: float,
    gamma: float,
    gae_lambda: float,
    clip_eps: float,
    ent_coef: float,
    vf_coef: float,
    hole_penalty: float = 0.0,
    empty_board_reward: float = 0.0,
    roughness_penalty: float = 0.0,
    combo_preserve_bonus: float = 0.0,
    reward_mode: str = "score",
    arch_type: str = "mlp",
):

    def step_env(env_state, action):
        new_state, info = _step(
            env_state, action, pmasks_flat, valid_anchor_flat, pieces_ncells,
            row_masks, col_masks, line_bonus_table, cascade_table,
            auto_reset=auto_reset, max_steps=max_steps, num_pieces=num_pieces,
        )
        return new_state, info

    def _encode_for_arch(env_state):
        if arch_type == "cnn":
            b, t, c, s = encode_obs_2d(env_state, num_pieces)
            return {"board": b, "tray_oh": t, "combo": c, "streak": s}
        else:
            return {"obs": encode_obs(env_state, num_pieces)}

    def _apply_for_arch(params, obs_dict, am):
        if arch_type == "cnn":
            raw, value = train_state.apply_fn(
                params, obs_dict["board"], obs_dict["tray_oh"],
                obs_dict["combo"], obs_dict["streak"],
            )
            logits = jnp.where(am, raw, jnp.float32(-1e9))
        else:
            logits, value = train_state.apply_fn(params, obs_dict["obs"], am)
        return logits, value

    def rollout_body(carry, _):
        env_state, key = carry
        obs_dict = _encode_for_arch(env_state)
        am = env_state["action_mask"]
        logits, value = _apply_for_arch(train_state.params, obs_dict, am)
        key, sub = jax.random.split(key)
        action = jax.random.categorical(sub, logits)
        any_valid = am.any(axis=-1)
        action = jnp.where(any_valid, action, jnp.int32(0))
        log_probs = jax.nn.log_softmax(logits)
        action_logp = jnp.take_along_axis(log_probs, action[:, None], axis=-1).squeeze(-1)

        new_env_state, info = step_env(env_state, action)
        terminated = info["terminated"]
        truncated = info["truncated"]
        done_or_trunc = terminated | truncated
        pure_reward = info["reward"].astype(jnp.float32)

        death_term      = death_penalty        * terminated.astype(jnp.float32)
        hole_term       = hole_penalty         * info["num_holes"].astype(jnp.float32)
        empty_term      = empty_board_reward   * info["num_empty_cells"].astype(jnp.float32)
        rough_term      = roughness_penalty    * info["boundary_edges"].astype(jnp.float32)
        combo_term      = combo_preserve_bonus * info["combo_count"].astype(jnp.float32)

        if reward_mode == "survival":
            reward = jnp.where(done_or_trunc, jnp.float32(0.0), jnp.float32(1.0))
        else:
            reward = (pure_reward
                      - death_term
                      - hole_term
                      + empty_term
                      - rough_term
                      + combo_term)

        traj = {
            "obs_dict": obs_dict,
            "action_mask": am,
            "action": action,
            "value": value,
            "log_prob": action_logp,
            "reward": reward,
            "pure_reward": pure_reward,
            "done": done_or_trunc,
            "death_term": death_term,
            "hole_term": hole_term,
            "empty_term": empty_term,
            "rough_term": rough_term,
            "combo_term": combo_term,
            "ep_len_at_done": jnp.where(
                done_or_trunc, env_state["ep_len"] + info["cells_placed"].astype(jnp.bool_).astype(jnp.int32), 0
            ),
            "ep_score_at_done": jnp.where(
                done_or_trunc, env_state["score"] + info["score_delta"], 0
            ),
        }
        return (new_env_state, key), traj

    (env_state, key), traj = jax.lax.scan(
        rollout_body, (env_state, key), None, length=rollout_len
    )

    obs_dict_last = _encode_for_arch(env_state)
    am_last = env_state["action_mask"]
    _, last_value = _apply_for_arch(train_state.params, obs_dict_last, am_last)

    advantages, returns = gae(
        traj["reward"], traj["value"], traj["done"], last_value, gamma, gae_lambda
    )
    adv_flat = advantages.reshape(-1)
    adv_norm = (advantages - adv_flat.mean()) / (adv_flat.std() + 1e-8)

    T = rollout_len
    N = traj["action"].shape[1]
    total = T * N
    mb_size = total // num_minibatches

    flat_obs_dict = jax.tree_util.tree_map(
        lambda x: x.reshape(total, *x.shape[2:]), traj["obs_dict"]
    )
    flat = {
        "obs_dict": flat_obs_dict,
        "action_mask": traj["action_mask"].reshape(total, -1),
        "action": traj["action"].reshape(total),
        "log_prob_old": traj["log_prob"].reshape(total),
        "value_old": traj["value"].reshape(total),
        "advantage": adv_norm.reshape(total),
        "return": returns.reshape(total),
    }

    def loss_fn(params, batch):
        logits, value = _apply_for_arch(params, batch["obs_dict"], batch["action_mask"])
        log_probs = jax.nn.log_softmax(logits)
        action_logp = jnp.take_along_axis(
            log_probs, batch["action"][:, None], axis=-1
        ).squeeze(-1)
        ratio = jnp.exp(action_logp - batch["log_prob_old"])
        adv = batch["advantage"]
        unclipped = ratio * adv
        clipped = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
        policy_loss = -jnp.minimum(unclipped, clipped).mean()
        value_loss = 0.5 * ((value - batch["return"]) ** 2).mean()
        probs = jnp.exp(log_probs)
        entropy = -(probs * log_probs).sum(axis=-1).mean()
        loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
        approx_kl = ((ratio - 1.0) - jnp.log(ratio + 1e-9)).mean()
        clip_frac = (jnp.abs(ratio - 1.0) > clip_eps).astype(jnp.float32).mean()
        aux = {
            "policy_loss": policy_loss, "value_loss": value_loss,
            "entropy": entropy, "approx_kl": approx_kl, "clip_frac": clip_frac,
        }
        return loss, aux

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    def epoch_body(carry, key_perm):
        train_state = carry
        perm = jax.random.permutation(key_perm, total)
        def mb_body(state, mb_idx):
            start = mb_idx * mb_size
            idx = jax.lax.dynamic_slice_in_dim(perm, start, mb_size)
            batch = {}
            for k, v in flat.items():
                if k == "obs_dict":
                    batch[k] = jax.tree_util.tree_map(lambda x: x[idx], v)
                else:
                    batch[k] = v[idx]
            (loss, aux), grads = grad_fn(state.params, batch)
            state = state.apply_gradients(grads=grads)
            return state, aux
        state, aux = jax.lax.scan(mb_body, train_state, jnp.arange(num_minibatches))
        aux_mean = jax.tree_util.tree_map(lambda x: x.mean(), aux)
        return state, aux_mean

    keys = jax.random.split(key, ppo_epochs + 1)
    key = keys[0]
    perm_keys = keys[1:]
    train_state, epoch_aux = jax.lax.scan(epoch_body, train_state, perm_keys)
    aux_mean = jax.tree_util.tree_map(lambda x: x.mean(), epoch_aux)

    ep_len_arr = traj["ep_len_at_done"].astype(jnp.float32)
    ep_score_arr = traj["ep_score_at_done"].astype(jnp.float32)
    ep_done_mask = traj["done"]

    n_done = ep_done_mask.sum().astype(jnp.float32)
    safe_n = jnp.maximum(n_done, 1.0)

    sum_len = jnp.where(ep_done_mask, ep_len_arr, 0).sum() / safe_n
    sum_score = jnp.where(ep_done_mask, ep_score_arr, 0).sum() / safe_n

    flat_len = jnp.where(ep_done_mask, ep_len_arr, jnp.nan).reshape(-1)
    flat_score = jnp.where(ep_done_mask, ep_score_arr, jnp.nan).reshape(-1)
    sorted_len = jnp.sort(flat_len)
    sorted_score = jnp.sort(flat_score)
    n_done_int = n_done.astype(jnp.int32)
    p50_idx = jnp.maximum(n_done_int // 2, 0)
    p99_idx = jnp.maximum((n_done_int * 99) // 100, 0)
    safe_p50_idx = jnp.minimum(p50_idx, sorted_len.shape[0] - 1)
    safe_p99_idx = jnp.minimum(p99_idx, sorted_len.shape[0] - 1)
    p50_len = jnp.where(n_done > 0, sorted_len[safe_p50_idx], 0.0)
    p99_len = jnp.where(n_done > 0, sorted_len[safe_p99_idx], 0.0)
    p50_sc = jnp.where(n_done > 0, sorted_score[safe_p50_idx], 0.0)
    p99_sc = jnp.where(n_done > 0, sorted_score[safe_p99_idx], 0.0)

    sum_shaped_reward = traj["reward"].sum() / safe_n
    sum_pure_reward   = traj["pure_reward"].sum() / safe_n
    sum_death = traj["death_term"].sum() / safe_n
    sum_hole  = traj["hole_term"].sum() / safe_n
    sum_empty = traj["empty_term"].sum() / safe_n
    sum_rough = traj["rough_term"].sum() / safe_n
    sum_combo = traj["combo_term"].sum() / safe_n

    metrics = {
        "policy_loss": aux_mean["policy_loss"],
        "value_loss": aux_mean["value_loss"],
        "entropy": aux_mean["entropy"],
        "approx_kl": aux_mean["approx_kl"],
        "clip_frac": aux_mean["clip_frac"],
        "n_episodes": n_done,
        "ep_length_mean": sum_len,
        "ep_length_p50": p50_len,
        "ep_length_p99": p99_len,
        "ep_score_mean": sum_score,
        "ep_score_p50": p50_sc,
        "ep_score_p99": p99_sc,
        "ep_shaped_reward_mean": sum_shaped_reward,
        "ep_pure_reward_mean":   sum_pure_reward,
        "shaping/death_pen_per_ep":      sum_death,
        "shaping/hole_pen_per_ep":       sum_hole,
        "shaping/empty_board_rew_per_ep": sum_empty,
        "shaping/roughness_pen_per_ep":  sum_rough,
        "shaping/combo_preserve_per_ep": sum_combo,
    }
    return train_state, env_state, key, metrics
