from __future__ import annotations

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

import block_blast_jax

import jax
import jax.numpy as jnp
import numpy as np

from block_blast_jax import BatchedBlockBlastEnv

NUM_STEPS = 2000
SEED = 42


def run_traj(env: BatchedBlockBlastEnv, num_envs: int, n_steps: int, action_rng: np.random.Generator):
    obs = env.observe()
    out = []
    for _ in range(n_steps):
        am0 = np.asarray(obs["action_mask"][0])
        valid = np.flatnonzero(am0)
        if valid.size > 0:
            a0 = int(action_rng.choice(valid))
        else:
            a0 = 0
        actions = np.zeros(num_envs, dtype=np.int32)
        actions[0] = a0
        obs, reward, term, trunc, info = env.step(jnp.asarray(actions))
        out.append((
            int(obs["grid"][0]),
            tuple(np.asarray(obs["tray_idx"][0]).tolist()),
            int(obs["score"][0]),
            bool(obs["done"][0]),
            int(reward[0]),
        ))
    return out


def main():
    env_solo = BatchedBlockBlastEnv(num_envs=1, seed=SEED, auto_reset=True)
    rng_solo = np.random.default_rng(0xC0FFEE)
    t_solo = run_traj(env_solo, 1, NUM_STEPS, rng_solo)

    env_batch = BatchedBlockBlastEnv(num_envs=1024, seed=SEED, auto_reset=True)
    rng_batch = np.random.default_rng(0xC0FFEE)
    t_batch = run_traj(env_batch, 1024, NUM_STEPS, rng_batch)

    fails = 0
    first_diff = None
    for i, (a, b) in enumerate(zip(t_solo, t_batch)):
        if a != b:
            fails += 1
            if first_diff is None:
                first_diff = (i, a, b)
    status = "PASS" if fails == 0 else f"FAIL {fails}/{NUM_STEPS}"
    print(f"Batch isolation N=1 vs N=1024 (env_id=0): {status}")
    if first_diff:
        print(f"  first diff at step {first_diff[0]}:")
        print(f"    solo : {first_diff[1]}")
        print(f"    batch: {first_diff[2]}")


if __name__ == "__main__":
    main()
