from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from block_blast_jax.env import _step
from block_blast_jax.rl.encode import encode_obs, encode_obs_2d


def make_eval_fn(net, env, num_eval_envs: int, max_step_cap: int, arch_type: str = "mlp"):
    pmf = env._pmasks_flat
    vaf = env._valid_anchor_flat
    pnc = env._pieces_ncells
    rm = env._row_masks
    cm = env._col_masks
    lbt = env._line_bonus_table
    ct = env._cascade_table
    P = env.num_pieces
    auto_reset = False

    @jax.jit
    def eval_fn(params, env_state):
        def body(carry, _):
            es = carry
            am = es["action_mask"]
            if arch_type == "cnn":
                b, t, c, s = encode_obs_2d(es, P)
                raw, _v = net.apply(params, b, t, c, s)
                logits = jnp.where(am, raw, jnp.float32(-1e9))
            else:
                obs = encode_obs(es, P)
                logits, _v = net.apply(params, obs, am)
            action = jnp.argmax(logits, axis=-1).astype(jnp.int32)
            any_valid = am.any(axis=-1)
            action = jnp.where(any_valid, action, jnp.int32(0))
            new_state, _info = _step(
                es, action, pmf, vaf, pnc, rm, cm, lbt, ct,
                auto_reset=auto_reset, max_steps=max_step_cap,
                num_pieces=P,
            )
            return new_state, None
        final, _ = jax.lax.scan(body, env_state, None, length=max_step_cap)
        return {
            "ep_len": final["ep_len"],
            "score": final["score"],
            "done": final["done"],
        }

    return eval_fn
