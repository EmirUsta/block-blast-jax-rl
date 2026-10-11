from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class BufferState(NamedTuple):
    states_grid:   jax.Array
    states_tray:   jax.Array
    states_combo:  jax.Array
    states_streak: jax.Array
    visit_dists:   jax.Array
    value_targets: jax.Array
    write_idx:     jax.Array
    size:          jax.Array
    capacity:      int


def init_buffer(capacity: int) -> BufferState:
    return BufferState(
        states_grid=jnp.zeros(capacity, dtype=jnp.uint64),
        states_tray=jnp.zeros((capacity, 3), dtype=jnp.int32),
        states_combo=jnp.zeros(capacity, dtype=jnp.int32),
        states_streak=jnp.zeros(capacity, dtype=jnp.int32),
        visit_dists=jnp.zeros((capacity, 192), dtype=jnp.float32),
        value_targets=jnp.zeros(capacity, dtype=jnp.float32),
        write_idx=jnp.int32(0),
        size=jnp.int32(0),
        capacity=int(capacity),
    )


def add(
    buf: BufferState,
    grid: jax.Array,
    tray_idx: jax.Array,
    combo: jax.Array,
    streak: jax.Array,
    visit_dist: jax.Array,
    value_target: jax.Array,
) -> BufferState:
    B = grid.shape[0]
    capacity = buf.capacity
    idx = (jnp.arange(B, dtype=jnp.int32) + buf.write_idx) % capacity
    new = BufferState(
        states_grid=buf.states_grid.at[idx].set(grid.astype(jnp.uint64)),
        states_tray=buf.states_tray.at[idx].set(tray_idx.astype(jnp.int32)),
        states_combo=buf.states_combo.at[idx].set(combo.astype(jnp.int32)),
        states_streak=buf.states_streak.at[idx].set(streak.astype(jnp.int32)),
        visit_dists=buf.visit_dists.at[idx].set(visit_dist.astype(jnp.float32)),
        value_targets=buf.value_targets.at[idx].set(value_target.astype(jnp.float32)),
        write_idx=(buf.write_idx + B) % capacity,
        size=jnp.minimum(buf.size + B, capacity).astype(jnp.int32),
        capacity=capacity,
    )
    return new


def sample(buf: BufferState, key: jax.Array, batch_size: int) -> dict:
    indices = jax.random.randint(key, (batch_size,), 0, buf.size, dtype=jnp.int32)
    return {
        "grid":   buf.states_grid[indices],
        "tray":   buf.states_tray[indices],
        "combo":  buf.states_combo[indices],
        "streak": buf.states_streak[indices],
        "visit":  buf.visit_dists[indices],
        "value":  buf.value_targets[indices],
        "indices": indices,
    }
