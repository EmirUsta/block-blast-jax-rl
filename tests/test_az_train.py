from __future__ import annotations

import sys, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

import block_blast_jax
import jax, jax.numpy as jnp
import numpy as np
from flax.serialization import msgpack_restore
from flax.training import train_state as ts
import optax

from block_blast_jax import BatchedBlockBlastEnv
from block_blast_jax.rl.network import ActorCritic
from block_blast_jax.rl.encode import feature_dim
from block_blast_jax.alphazero.config import AZConfig
from block_blast_jax.alphazero.mcts import make_run_mcts
from block_blast_jax.alphazero.replay_buffer import init_buffer, add as buf_add, sample as buf_sample
from block_blast_jax.alphazero.selfplay import self_play_iteration
from block_blast_jax.alphazero.train_az import (
    encode_batch, make_train_step, make_optimizer, reset_value_head, load_warmstart,
)


def test_warmstart_load():
    import os
    import pytest
    path = os.environ.get("BB_WARMSTART_CKPT")
    if not path or not os.path.exists(path):
        pytest.skip("set BB_WARMSTART_CKPT to a trained checkpoint")
    params, src_cfg = load_warmstart(path)
    print(f"  Loaded params from {path}, src_cfg keys: {list(src_cfg.keys())[:5]}...")
    params_reset = reset_value_head(params)
    net = ActorCritic(hidden_size=128, num_actions=192)
    B = 4
    P = 31
    F = feature_dim(P)
    obs = jnp.zeros((B, F), dtype=jnp.float32)
    am = jnp.ones((B, 192), dtype=jnp.bool_)
    logits, value = net.apply(params_reset, obs, am)
    print(f"  forward OK: logits {logits.shape}, value {value.shape}")
    return logits.shape == (B, 192) and value.shape == (B,)


def test_one_train_iter_loss_drop():
    cfg = AZConfig(B=4, num_simulations_train=20, max_moves_per_iter=20,
                   minibatch_size=64, epochs_per_iter=1, total_iters=2)
    env = BatchedBlockBlastEnv(num_envs=cfg.B, seed=0, auto_reset=True)
    P = env.num_pieces
    F = feature_dim(P)
    net = ActorCritic(hidden_size=128, num_actions=192)
    key = jax.random.PRNGKey(0)
    params = net.init(key, jnp.zeros((cfg.B, F)), jnp.ones((cfg.B, 192), dtype=jnp.bool_))
    tx = make_optimizer(cfg)
    state = ts.TrainState.create(apply_fn=net.apply, params=params, tx=tx)

    env_consts = dict(
        pmasks_flat=env._pmasks_flat, valid_anchor_flat=env._valid_anchor_flat,
        pieces_ncells=env._pieces_ncells, row_masks=env._row_masks,
        col_masks=env._col_masks, line_bonus_table=env._line_bonus_table,
        cascade_table=env._cascade_table,
    )
    run_mcts = make_run_mcts(net.apply, env_consts, P, max_steps=env.max_steps, discount=cfg.discount)
    run_mcts_jit = jax.jit(run_mcts, static_argnames=(
        "num_simulations","max_depth","dirichlet_fraction","dirichlet_alpha",
        "pb_c_init","pb_c_base","use_dirichlet"))
    add_jit = jax.jit(buf_add)
    sample_jit = jax.jit(buf_sample, static_argnums=2)
    train_step_fn = make_train_step(net, cfg, P)
    buf = init_buffer(2000)

    losses = []
    for it in range(1, 3):
        key, sub = jax.random.split(key)
        sp = self_play_iteration(run_mcts_jit, net.apply, state.params, env, sub,
                                 num_simulations=cfg.num_simulations_train,
                                 max_depth=cfg.max_depth,
                                 max_moves=cfg.max_moves_per_iter,
                                 discount=cfg.discount)
        T, B = sp["grids"].shape
        buf = add_jit(buf,
                      sp["grids"].reshape(T*B), sp["trays"].reshape(T*B,3),
                      sp["combos"].reshape(T*B), sp["streaks"].reshape(T*B),
                      sp["visits"].reshape(T*B,192), sp["values"].reshape(T*B))
        iter_loss = 0.0
        for s in range(4):
            key, sub = jax.random.split(key)
            batch = sample_jit(buf, sub, cfg.minibatch_size)
            obs, am = encode_batch(batch, P)
            state, m = train_step_fn(state, obs, am, batch["visit"], batch["value"], None)
            iter_loss += float(m["loss"])
        iter_loss /= 4
        losses.append(iter_loss)
        print(f"  iter {it} loss={iter_loss:.4f}")
    return losses[1] < losses[0] + 1e-3, f"losses={losses}"


if __name__ == "__main__":
    print("=== Smoke 6: warmstart load ===")
    try:
        ok = test_warmstart_load()
        print(f"  {'PASS' if ok else 'FAIL'}")
        s6 = ok
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
        s6 = False

    print("\n=== Smoke 5: 1 train iter loss ===")
    try:
        ok, msg = test_one_train_iter_loss_drop()
        print(f"  {'PASS' if ok else 'INFO/maybe FAIL'} ({msg})")
        s5 = ok
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        s5 = False

    sys.exit(0 if (s5 and s6) else 1)
