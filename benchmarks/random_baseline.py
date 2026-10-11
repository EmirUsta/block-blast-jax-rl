from __future__ import annotations

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

import block_blast_jax

import jax
import jax.numpy as jnp
import numpy as np

from block_blast_jax import BatchedBlockBlastEnv
from block_blast_jax.rl.network import ActorCritic
from block_blast_jax.rl.encode import encode_obs


def run(num_envs: int = 1024, n_episodes: int = 1000, seed: int = 0):
    env = BatchedBlockBlastEnv(num_envs=num_envs, seed=seed, auto_reset=True)
    P = env.num_pieces
    net = ActorCritic(hidden_size=128, num_actions=192)
    init_key = jax.random.PRNGKey(seed)
    F = encode_obs(env.observe(), P).shape[-1]
    dummy_obs = jnp.zeros((num_envs, F), dtype=jnp.float32)
    dummy_mask = jnp.ones((num_envs, 192), dtype=jnp.bool_)
    params = net.init(init_key, dummy_obs, dummy_mask)

    @jax.jit
    def act(state, key):
        obs = encode_obs(state, P)
        am = state["action_mask"]
        logits, _ = net.apply(params, obs, am)
        any_valid = am.any(axis=-1)
        action = jax.random.categorical(key, logits)
        action = jnp.where(any_valid, action, jnp.int32(0))
        return action

    key = jax.random.PRNGKey(seed + 1)
    ep_lens = []
    ep_scores = []
    while len(ep_lens) < n_episodes:
        key, sub = jax.random.split(key)
        prev_ep_len = np.asarray(env.state["ep_len"])
        prev_score = np.asarray(env.state["score"])
        action = act(env.state, sub)
        new_state, reward, term, trunc, info = env.step(action)
        done_mask = np.asarray(term | trunc)
        if done_mask.any():
            valid_int = np.asarray(info["cells_placed"]).astype(bool).astype(int)
            pre_reset_len = prev_ep_len + valid_int
            pre_reset_score = prev_score + np.asarray(info["score_delta"])
            for i in np.where(done_mask)[0]:
                ep_lens.append(int(pre_reset_len[i]))
                ep_scores.append(int(pre_reset_score[i]))
        if len(ep_lens) >= n_episodes:
            break
    ep_lens = np.array(ep_lens[:n_episodes])
    ep_scores = np.array(ep_scores[:n_episodes])
    return {
        "n_episodes": int(len(ep_lens)),
        "ep_length_mean": float(ep_lens.mean()),
        "ep_length_median": float(np.median(ep_lens)),
        "ep_length_p99": float(np.percentile(ep_lens, 99)),
        "ep_score_mean": float(ep_scores.mean()),
        "ep_score_p99": float(np.percentile(ep_scores, 99)),
    }


if __name__ == "__main__":
    r = run()
    print(r)
