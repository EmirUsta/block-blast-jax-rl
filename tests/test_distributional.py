from __future__ import annotations

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

import block_blast_jax
import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats

from block_blast_jax import BatchedBlockBlastEnv
import pytest

pytest.importorskip("block_blast", reason="reference NumPy implementation not installed")
from block_blast import BlockBlastEnv


N_STEPS = 100_000


def collect_jax(seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    env = BatchedBlockBlastEnv(num_envs=1, seed=seed, auto_reset=True)
    obs = env.observe()
    rng = np.random.default_rng(seed + 1)
    rewards = np.empty(N_STEPS, dtype=np.int32)
    ep_lens = []
    term_scores = []
    last_score = 0
    last_ep = 0
    for i in range(N_STEPS):
        am = np.asarray(obs["action_mask"][0])
        valid = np.flatnonzero(am)
        a = int(rng.choice(valid)) if valid.size else 0
        obs, reward, term, trunc, info = env.step(jnp.array([a], dtype=jnp.int32))
        r = int(reward[0])
        rewards[i] = r
        if bool(info["terminated"][0]) or bool(info["truncated"][0]):
            ep_lens.append(last_ep + (0 if bool(info["invalid_action"][0]) else 1))
            term_scores.append(last_score + r)
            last_score = 0
            last_ep = 0
        else:
            last_score += r
            last_ep += (0 if bool(info["invalid_action"][0]) else 1)
    return rewards, np.array(ep_lens), np.array(term_scores)


def collect_np(seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    env = BlockBlastEnv(seed=seed)
    state = env.reset(seed=seed)
    rng = np.random.default_rng(seed + 1)
    rewards = np.empty(N_STEPS, dtype=np.int32)
    ep_lens = []
    term_scores = []
    for i in range(N_STEPS):
        if state.done:
            ep_lens.append(state.episode_length)
            term_scores.append(state.score)
            state = env.reset()
        am = state.action_mask
        valid = np.flatnonzero(am)
        a = int(rng.choice(valid)) if valid.size else 0
        state, info = env.step(a)
        r = int(info.get("score_delta", 0))
        rewards[i] = r
    return rewards, np.array(ep_lens), np.array(term_scores)


def main():
    print("Collecting JAX trajectory ...")
    jx_r, jx_ep, jx_sc = collect_jax(seed=42)
    print(f"  JAX rewards: n={len(jx_r)}, episodes={len(jx_ep)}")
    print("Collecting NumPy ref trajectory ...")
    np_r, np_ep, np_sc = collect_np(seed=42)
    print(f"  NumPy rewards: n={len(np_r)}, episodes={len(np_ep)}")

    ks_r = stats.ks_2samp(jx_r.astype(float), np_r.astype(float))
    ks_e = stats.ks_2samp(jx_ep.astype(float), np_ep.astype(float))

    print(f"\nKS reward             : stat={ks_r.statistic:.4f}  p={ks_r.pvalue:.4f}  ({'PASS' if ks_r.pvalue > 0.05 else 'FAIL'})")
    print(f"KS episode_length     : stat={ks_e.statistic:.4f}  p={ks_e.pvalue:.4f}  ({'PASS' if ks_e.pvalue > 0.05 else 'FAIL'})")

    print("\nScore stats (terminal):")
    for label, jx, npr in [("mean", jx_sc.mean(), np_sc.mean()),
                            ("std",  jx_sc.std(),  np_sc.std()),
                            ("p99",  np.percentile(jx_sc, 99), np.percentile(np_sc, 99))]:
        delta = (jx - npr) / max(abs(npr), 1e-9)
        ok = abs(delta) < 0.02
        print(f"  {label}: jx={jx:.3f}  np={npr:.3f}  Δ%={delta*100:+.2f}  ({'PASS' if ok else 'FAIL'})")

    print("\nReward stats:")
    print(f"  jx: mean={jx_r.mean():.3f} std={jx_r.std():.3f} max={jx_r.max()}")
    print(f"  np: mean={np_r.mean():.3f} std={np_r.std():.3f} max={np_r.max()}")
    print("Episode-length stats:")
    print(f"  jx: mean={jx_ep.mean():.2f} median={np.median(jx_ep):.0f} p99={np.percentile(jx_ep, 99):.0f}")
    print(f"  np: mean={np_ep.mean():.2f} median={np.median(np_ep):.0f} p99={np.percentile(np_ep, 99):.0f}")


if __name__ == "__main__":
    main()
