from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import mctx

from block_blast_jax.env import _step
from block_blast_jax.rl.encode import encode_obs


def _net_apply(net_apply_fn, params, env_state, num_pieces: int):
    obs = encode_obs(env_state, num_pieces)
    am = env_state["action_mask"]
    logits, value = net_apply_fn(params, obs, am)
    raw = jnp.where(am, logits, -jnp.float32(1e9))
    return raw, value


def make_root_fn(net_apply_fn, num_pieces: int):
    def root_fn(params, env_state):
        prior_logits, value = _net_apply(net_apply_fn, params, env_state, num_pieces)
        return mctx.RootFnOutput(
            prior_logits=prior_logits,
            value=value,
            embedding=env_state,
        )
    return root_fn


def make_recurrent_fn(net_apply_fn, env_consts: dict, num_pieces: int,
                      max_steps: int, discount: float, combo_coef: float = 0.0):
    pmf = env_consts["pmasks_flat"]
    vaf = env_consts["valid_anchor_flat"]
    pnc = env_consts["pieces_ncells"]
    rm = env_consts["row_masks"]
    cm = env_consts["col_masks"]
    lbt = env_consts["line_bonus_table"]
    ct = env_consts["cascade_table"]

    def recurrent_fn(params, rng, action, env_state):
        new_state, info = _step(
            env_state, action, pmf, vaf, pnc, rm, cm, lbt, ct,
            auto_reset=False, max_steps=max_steps, num_pieces=num_pieces,
        )
        prior_logits, value = _net_apply(net_apply_fn, params, new_state, num_pieces)
        terminated = info["terminated"] | info["truncated"]
        combo_mult = info["combo_mult"].astype(jnp.float32)
        alive_r = jnp.float32(1.0) + jnp.float32(combo_coef) * (combo_mult - jnp.float32(1.0))
        reward = jnp.where(terminated, jnp.float32(0.0), alive_r)
        disc = jnp.where(terminated, jnp.float32(0.0), jnp.float32(discount))
        return mctx.RecurrentFnOutput(
            reward=reward, discount=disc,
            prior_logits=prior_logits, value=value,
        ), new_state
    return recurrent_fn


def make_run_mcts(net_apply_fn, env_consts, num_pieces, max_steps, discount,
                  combo_coef: float = 0.0):
    root_fn = make_root_fn(net_apply_fn, num_pieces)
    recurrent_fn = make_recurrent_fn(net_apply_fn, env_consts, num_pieces, max_steps,
                                      discount, combo_coef=combo_coef)

    def run_mcts(params, env_state, key, num_simulations, max_depth,
                 dirichlet_fraction, dirichlet_alpha, pb_c_init, pb_c_base,
                 use_dirichlet: bool):
        root = root_fn(params, env_state)
        am = env_state["action_mask"]
        invalid = ~am
        out = mctx.gumbel_muzero_policy(
            params=params,
            rng_key=key,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=num_simulations,
            invalid_actions=invalid,
            max_depth=max_depth,
            qtransform=mctx.qtransform_completed_by_mix_value,
            gumbel_scale=1.0,
        )
        return out

    return run_mcts
