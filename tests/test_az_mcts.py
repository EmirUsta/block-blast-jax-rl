from __future__ import annotations

import sys, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

import block_blast_jax
import jax, jax.numpy as jnp
import numpy as np
from flax.serialization import msgpack_restore

from block_blast_jax import BatchedBlockBlastEnv
from block_blast_jax.rl.network import ActorCritic
from block_blast_jax.rl.encode import feature_dim
from block_blast_jax.alphazero.mcts import make_run_mcts


def make_apparatus(B: int, params=None, init_random_params=True):
    env = BatchedBlockBlastEnv(num_envs=B, seed=0, auto_reset=False)
    P = env.num_pieces
    F = feature_dim(P)
    net = ActorCritic(hidden_size=128, num_actions=192)
    if params is None and init_random_params:
        key = jax.random.PRNGKey(0)
        dummy_obs = jnp.zeros((B, F), dtype=jnp.float32)
        dummy_mask = jnp.ones((B, 192), dtype=jnp.bool_)
        params = net.init(key, dummy_obs, dummy_mask)
    env_consts = {
        "pmasks_flat": env._pmasks_flat,
        "valid_anchor_flat": env._valid_anchor_flat,
        "pieces_ncells": env._pieces_ncells,
        "row_masks": env._row_masks,
        "col_masks": env._col_masks,
        "line_bonus_table": env._line_bonus_table,
        "cascade_table": env._cascade_table,
    }
    run_mcts = make_run_mcts(net.apply, env_consts, P, max_steps=env.max_steps, discount=0.997)
    return env, net, params, run_mcts, P


def test_basic_shapes():
    B = 4
    env, net, params, run_mcts, P = make_apparatus(B)
    key = jax.random.PRNGKey(42)
    print("  compiling MCTS (B=4, sims=50) ...")
    t0 = time.perf_counter()
    out = jax.jit(run_mcts, static_argnames=(
        "num_simulations", "max_depth", "dirichlet_fraction", "dirichlet_alpha",
        "pb_c_init", "pb_c_base", "use_dirichlet"))(
        params, env.state, key,
        num_simulations=50, max_depth=64,
        dirichlet_fraction=0.0, dirichlet_alpha=0.3,
        pb_c_init=1.25, pb_c_base=19652.0, use_dirichlet=False,
    )
    jax.block_until_ready(out.action)
    print(f"  compile + 1 call: {time.perf_counter()-t0:.1f}s")
    aw = out.action_weights
    a = out.action
    print(f"  action_weights shape={aw.shape} dtype={aw.dtype}")
    print(f"  action shape={a.shape} dtype={a.dtype}, values={np.asarray(a)}")
    print(f"  action_weights sum per env: {np.asarray(aw.sum(axis=-1))}")
    if aw.shape != (B, 192):
        return False, f"action_weights shape {aw.shape}"
    if a.shape != (B,):
        return False, f"action shape {a.shape}"
    sums = np.asarray(aw.sum(axis=-1))
    if not np.allclose(sums, 1.0, atol=1e-4):
        return False, f"action_weights sum ≠ 1: {sums}"
    am = np.asarray(env.state["action_mask"])
    masked_aw = np.asarray(aw) * (~am).astype(np.float32)
    if masked_aw.sum() > 1e-4:
        return False, f"masked action_weights nonzero: {masked_aw.sum()}"
    return True, "OK"


def test_one_episode():
    B = 4
    env, net, params, run_mcts, P = make_apparatus(B)
    run_mcts_jit = jax.jit(run_mcts, static_argnames=(
        "num_simulations", "max_depth", "dirichlet_fraction", "dirichlet_alpha",
        "pb_c_init", "pb_c_base", "use_dirichlet"))
    key = jax.random.PRNGKey(42)
    print("  compiling (B=4, sims=100) ...")
    t0 = time.perf_counter()
    out = run_mcts_jit(
        params, env.state, key,
        num_simulations=100, max_depth=64,
        dirichlet_fraction=0.0, dirichlet_alpha=0.3,
        pb_c_init=1.25, pb_c_base=19652.0, use_dirichlet=False,
    )
    jax.block_until_ready(out.action)
    print(f"  compile + 1 call: {time.perf_counter()-t0:.1f}s")
    moves = 0
    max_len_per_env = np.zeros(B, dtype=int)
    for step_i in range(120):
        key, sub = jax.random.split(key)
        out = run_mcts_jit(
            params, env.state, sub,
            num_simulations=100, max_depth=64,
            dirichlet_fraction=0.0, dirichlet_alpha=0.3,
            pb_c_init=1.25, pb_c_base=19652.0, use_dirichlet=False,
        )
        actions = out.action
        new_state, reward, term, trunc, info = env.step(actions)
        moves += 1
        max_len_per_env = np.maximum(max_len_per_env, np.asarray(new_state["ep_len"]))
        all_done = bool(np.asarray(new_state["done"]).all())
        if all_done:
            break
    print(f"  played {moves} moves, ep_len per env: {max_len_per_env}")
    return moves >= 30 and max_len_per_env.max() >= 5, f"moves={moves}, max_len={max_len_per_env.max()}"


if __name__ == "__main__":
    print("=== Test 2: MCTS basic shapes ===")
    ok, msg = test_basic_shapes()
    print(f"  {'PASS' if ok else 'FAIL'}: {msg}")
    f1 = ok
    print()
    print("=== Test 3: One episode ===")
    ok, msg = test_one_episode()
    print(f"  {'PASS' if ok else 'FAIL'}: {msg}")
    f2 = ok
    sys.exit(0 if (f1 and f2) else 1)
