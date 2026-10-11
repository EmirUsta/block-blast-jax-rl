from __future__ import annotations

import argparse
import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import block_blast_jax

import jax
import jax.numpy as jnp

from block_blast_jax import BatchedBlockBlastEnv
from benchmarks.bench_throughput import make_step_fn, make_scan_fn


def hlo_summary(num_envs: int, scan_k: int = 200) -> str:
    env = BatchedBlockBlastEnv(num_envs=num_envs, seed=42, auto_reset=True)
    scan_fn = make_scan_fn(env, scan_k)
    key = jax.random.PRNGKey(0)
    lowered = jax.jit(scan_fn).lower(env.state, key)
    compiled = lowered.compile()
    return compiled.as_text()


def host_sync_check(num_envs: int, scan_k: int = 200) -> dict:
    env = BatchedBlockBlastEnv(num_envs=num_envs, seed=42, auto_reset=True)
    scan_fn = make_scan_fn(env, scan_k)
    key = jax.random.PRNGKey(0)
    state = scan_fn(env.state, key)
    jax.block_until_ready(state["grid"])
    env.state = state

    transfers = {"to_host": 0, "to_device": 0}
    try:
        with jax.transfer_guard("log_explicit"):
            for i in range(5):
                key, sub = jax.random.split(key)
                state = scan_fn(env.state, sub)
                env.state = state
            jax.block_until_ready(env.state["grid"])
        transfers["status"] = "no explicit transfers detected within transfer_guard"
    except Exception as e:
        transfers["status"] = f"transfer_guard error: {type(e).__name__}: {e}"
    return transfers


def time_per_op(num_envs: int, scan_k: int = 200, n_iters: int = 20):
    env = BatchedBlockBlastEnv(num_envs=num_envs, seed=42, auto_reset=True)
    one_step = make_step_fn(env)
    key = jax.random.PRNGKey(0)
    state = env.state
    for _ in range(50):
        key, sub = jax.random.split(key)
        state, _ = one_step(state, sub)
    jax.block_until_ready(state["grid"])
    t0 = time.perf_counter()
    for _ in range(n_iters * scan_k):
        key, sub = jax.random.split(key)
        state, _ = one_step(state, sub)
    jax.block_until_ready(state["grid"])
    t1 = time.perf_counter()
    fps = (n_iters * scan_k * num_envs) / (t1 - t0)
    return {"per_step_call_fps": fps, "wall_s": t1 - t0}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_envs", type=int, default=4096)
    ap.add_argument("--scan_k", type=int, default=200)
    args = ap.parse_args()

    print(f"=== HLO summary (head) — N={args.num_envs} ===")
    hlo = hlo_summary(args.num_envs, args.scan_k)
    for line in hlo.splitlines()[:80]:
        print(line)
    print(f"... (HLO total: {len(hlo.splitlines())} lines)")

    print(f"\n=== Per-op count summary — N={args.num_envs} ===")
    op_counts = {}
    for line in hlo.splitlines():
        line = line.strip()
        if "= " in line and not line.startswith("//"):
            try:
                rhs = line.split("= ", 1)[1]
                op = rhs.split("(")[0].split(" ")[-1].strip()
                op_counts[op] = op_counts.get(op, 0) + 1
            except Exception:
                pass
    for op, cnt in sorted(op_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"  {op:30s} {cnt}")

    print(f"\n=== Host-sync check — N={args.num_envs}, {args.scan_k} steps × 5 iters ===")
    chk = host_sync_check(args.num_envs, args.scan_k)
    for k, v in chk.items():
        print(f"  {k}: {v}")
