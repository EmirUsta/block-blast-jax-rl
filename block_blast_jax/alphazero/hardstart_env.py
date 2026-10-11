from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax.serialization import msgpack_restore

from block_blast_jax import BatchedBlockBlastEnv
from block_blast_jax.env import compute_tray_action_mask


class HardStartBatchedEnv(BatchedBlockBlastEnv):
    def __init__(self, *args, pool_path: str | None = None,
                 hardstart_prob: float = 0.0, init_from_pool: bool = False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.hardstart_prob = float(hardstart_prob)
        self.pool = None
        self._last_hardstart_frac = 0.0
        if pool_path and self.hardstart_prob > 0.0:
            blob = msgpack_restore(Path(pool_path).read_bytes())
            self.pool = {
                "grids": jnp.asarray(blob["grids"], dtype=jnp.uint64),
                "tray_idxs": jnp.asarray(blob["tray_idxs"], dtype=jnp.int32),
                "combo_counts": jnp.asarray(blob["combo_counts"], dtype=jnp.int32),
                "no_clear_streaks": jnp.asarray(blob["no_clear_streaks"], dtype=jnp.int32),
            }
            self.pool_size = int(self.pool["grids"].shape[0])
            print(f"[hardstart] pool loaded: {self.pool_size} states from {pool_path}")
        else:
            self.pool_size = 0
        self._hardstart_key = jax.random.PRNGKey(int(kwargs.get("seed", 0)) + 99991)
        self._prev_done = None
        if init_from_pool and self.pool is not None:
            self._seed_initial_from_pool()

    def _seed_initial_from_pool(self):
        self._hardstart_key, k1, k2 = jax.random.split(self._hardstart_key, 3)
        coin = np.asarray(jax.random.uniform(k1, (self.num_envs,)))
        do_hard = coin < self.hardstart_prob
        n_hard = int(do_hard.sum())
        if n_hard == 0:
            return
        idx = np.asarray(jax.random.randint(k2, (self.num_envs,), 0, self.pool_size))
        pool_grid = np.asarray(self.pool["grids"])[idx]
        pool_tray = np.asarray(self.pool["tray_idxs"])[idx]
        pool_combo = np.asarray(self.pool["combo_counts"])[idx]
        pool_streak = np.asarray(self.pool["no_clear_streaks"])[idx]
        do_hard_jx = jnp.asarray(do_hard)
        new_grid = jnp.where(do_hard_jx, jnp.asarray(pool_grid).astype(jnp.uint64),
                             self.state["grid"])
        new_tray = jnp.where(do_hard_jx[:, None], jnp.asarray(pool_tray).astype(jnp.int32),
                             self.state["tray_idx"])
        new_combo = jnp.where(do_hard_jx, jnp.asarray(pool_combo).astype(jnp.int32),
                              self.state["combo_count"])
        new_streak = jnp.where(do_hard_jx, jnp.asarray(pool_streak).astype(jnp.int32),
                               self.state["no_clear_streak"])
        am = compute_tray_action_mask(new_grid, new_tray, self._pmasks_flat,
                                      self._valid_anchor_flat, self.state["done"])
        self.state = dict(self.state)
        self.state["grid"] = new_grid
        self.state["tray_idx"] = new_tray
        self.state["combo_count"] = new_combo
        self.state["no_clear_streak"] = new_streak
        self.state["action_mask"] = am
        print(f"[hardstart] init_from_pool: {n_hard}/{self.num_envs} envs seeded")

    def step(self, actions):
        prev_done = np.asarray(self.state["done"]).copy()
        new_state, reward, term, trunc, info = super().step(actions)
        if self.pool is None or self.hardstart_prob <= 0.0:
            self.state = new_state
            self._last_hardstart_frac = 0.0
            return new_state, reward, term, trunc, info

        terminated = np.asarray(term | trunc)
        freshly_done = terminated & (~prev_done)
        n_fresh = int(freshly_done.sum())
        if n_fresh == 0:
            self.state = new_state
            self._last_hardstart_frac = 0.0
            return new_state, reward, term, trunc, info

        self._hardstart_key, k1, k2 = jax.random.split(self._hardstart_key, 3)
        coin = np.asarray(jax.random.uniform(k1, (self.num_envs,)))
        do_hard = freshly_done & (coin < self.hardstart_prob)
        n_hard = int(do_hard.sum())
        if n_hard == 0:
            self.state = new_state
            self._last_hardstart_frac = 0.0
            return new_state, reward, term, trunc, info

        idx = np.asarray(jax.random.randint(k2, (self.num_envs,), 0, self.pool_size))
        pool_grid = np.asarray(self.pool["grids"])[idx]
        pool_tray = np.asarray(self.pool["tray_idxs"])[idx]
        pool_combo = np.asarray(self.pool["combo_counts"])[idx]
        pool_streak = np.asarray(self.pool["no_clear_streaks"])[idx]
        do_hard_jx = jnp.asarray(do_hard)
        new_grid = jnp.where(do_hard_jx, jnp.asarray(pool_grid).astype(jnp.uint64),
                             new_state["grid"])
        new_tray = jnp.where(do_hard_jx[:, None], jnp.asarray(pool_tray).astype(jnp.int32),
                             new_state["tray_idx"])
        new_combo = jnp.where(do_hard_jx, jnp.asarray(pool_combo).astype(jnp.int32),
                              new_state["combo_count"])
        new_streak = jnp.where(do_hard_jx, jnp.asarray(pool_streak).astype(jnp.int32),
                               new_state["no_clear_streak"])
        am = compute_tray_action_mask(new_grid, new_tray, self._pmasks_flat,
                                      self._valid_anchor_flat, new_state["done"])
        new_state = dict(new_state)
        new_state["grid"] = new_grid
        new_state["tray_idx"] = new_tray
        new_state["combo_count"] = new_combo
        new_state["no_clear_streak"] = new_streak
        new_state["action_mask"] = am
        self.state = new_state
        self._last_hardstart_frac = n_hard / self.num_envs
        return new_state, reward, term, trunc, info
