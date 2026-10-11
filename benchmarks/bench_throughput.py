from __future__ import annotations

import argparse
import json
import sys
import time

import sys as _sys
_sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

import block_blast_jax

import jax
import jax.numpy as jnp
import numpy as np

from block_blast_jax import BatchedBlockBlastEnv
from block_blast_jax.env import _step
from block_blast_jax.constants import (
    COL_MASKS_NP, LINES_TABLE_NP, NUM_ACTIONS, ROW_MASKS_NP,
)


def make_step_fn(env: BatchedBlockBlastEnv):
    pmf = env._pmasks_flat
    vaf = env._valid_anchor_flat
    pnc = env._pieces_ncells
    rm = env._row_masks
    cm = env._col_masks
    lbt = env._line_bonus_table
    ct = env._cascade_table

    @jax.jit
    def one_step(state, step_key):
        am = state["action_mask"]
        weights = am.astype(jnp.float32)
        gumbel = jax.random.gumbel(step_key, shape=weights.shape)
        scores = jnp.log(weights + 1e-9) + gumbel
        actions = jnp.argmax(scores, axis=-1).astype(jnp.int32)
        any_valid = am.any(axis=-1)
        actions = jnp.where(any_valid, actions, jnp.int32(0))

        new_state, info = _step(
            state, actions, pmf, vaf, pnc, rm, cm, lbt, ct,
            auto_reset=env.auto_reset,
            max_steps=env.max_steps,
            num_pieces=env.num_pieces,
        )
        return new_state, info["reward"]

    return one_step


def make_scan_fn(env: BatchedBlockBlastEnv, k_steps: int):
    one_step = make_step_fn(env)

    @jax.jit
    def scan_fn(state, master_key):
        keys = jax.random.split(master_key, k_steps)
        def body(s, k):
            ns, r = one_step(s, k)
            return ns, r
        final_state, _ = jax.lax.scan(body, state, keys)
        return final_state

    return scan_fn


def bench(num_envs: int, warmup_steps: int = 200, scan_k: int = 200, n_iters: int = 10, seed: int = 42):
    env = BatchedBlockBlastEnv(num_envs=num_envs, seed=seed, auto_reset=True)

    scan_fn = make_scan_fn(env, scan_k)
    master_key = jax.random.PRNGKey(seed)

    t0 = time.perf_counter()
    state = scan_fn(env.state, master_key)
    jax.block_until_ready(state["grid"])
    t1 = time.perf_counter()
    compile_time = t1 - t0
    env.state = state

    for _ in range(2):
        master_key, sub = jax.random.split(master_key)
        state = scan_fn(env.state, sub)
        env.state = state
    jax.block_until_ready(env.state["grid"])

    t0 = time.perf_counter()
    for _ in range(n_iters):
        master_key, sub = jax.random.split(master_key)
        state = scan_fn(env.state, sub)
        env.state = state
    jax.block_until_ready(env.state["grid"])
    t1 = time.perf_counter()

    wall = t1 - t0
    total_env_steps = n_iters * scan_k * num_envs
    fps = total_env_steps / wall
    batch_steps = n_iters * scan_k
    ms_per_batch_step = wall / batch_steps * 1000

    return {
        "num_envs": num_envs,
        "scan_k": scan_k,
        "n_iters": n_iters,
        "warmup_compile_time_s": compile_time,
        "wall_s": wall,
        "total_env_steps": total_env_steps,
        "fps": fps,
        "step_per_env_per_s": (n_iters * scan_k) / wall,
        "ms_per_batch_step": ms_per_batch_step,
    }


def vram_peak_mb():
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2,
        )
        return float(out.stdout.strip().split("\n")[0])
    except Exception:
        return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="1,64,256,1024,4096,16384")
    ap.add_argument("--scan_k", type=int, default=200)
    ap.add_argument("--n_iters", type=int, default=10)
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    results = []
    for N in sizes:
        try:
            r = bench(num_envs=N, scan_k=args.scan_k, n_iters=args.n_iters)
            r["vram_mb"] = vram_peak_mb()
            results.append(r)
            print(json.dumps(r))
        except Exception as e:
            print(json.dumps({"num_envs": N, "error": str(e)}))
