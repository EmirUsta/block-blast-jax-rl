from __future__ import annotations

import jax
import jax.numpy as jnp


def encode_obs(state: dict, num_pieces: int) -> jax.Array:
    grid = state["grid"]
    tray_idx = state["tray_idx"]
    combo = state["combo_count"]
    streak = state["no_clear_streak"]
    N = grid.shape[0]

    bit_idx = jnp.arange(64, dtype=jnp.uint64)
    grid_bits = ((grid[:, None] >> bit_idx[None, :]) & jnp.uint64(1)).astype(jnp.float32)

    safe_tray = jnp.maximum(tray_idx, 0)
    tray_oh = jax.nn.one_hot(safe_tray, num_pieces, dtype=jnp.float32)
    valid_slot = (tray_idx >= 0)[:, :, None].astype(jnp.float32)
    tray_oh = tray_oh * valid_slot
    tray_flat = tray_oh.reshape(N, 3 * num_pieces)

    combo_norm = (jnp.minimum(combo, 10).astype(jnp.float32) / 10.0)[:, None]
    streak_norm = (streak.astype(jnp.float32) / 3.0)[:, None]

    return jnp.concatenate([grid_bits, tray_flat, combo_norm, streak_norm], axis=-1)


def feature_dim(num_pieces: int) -> int:
    return 64 + 3 * num_pieces + 2


def encode_obs_2d(state: dict, num_pieces: int):
    grid = state["grid"]
    tray_idx = state["tray_idx"]
    combo = state["combo_count"]
    streak = state["no_clear_streak"]
    N = grid.shape[0]

    bits = ((grid[:, None] >> jnp.arange(64, dtype=jnp.uint64)) & jnp.uint64(1))
    board = bits.reshape(N, 8, 8, 1).astype(jnp.float32)

    consumed = tray_idx < 0
    safe = jnp.where(consumed, 0, tray_idx)
    oh = jax.nn.one_hot(safe, num_pieces, dtype=jnp.float32)
    oh = oh * (~consumed)[..., None].astype(jnp.float32)
    tray_oh = oh.reshape(N, 3 * num_pieces)

    combo_norm = (jnp.minimum(combo, 10).astype(jnp.float32) / 10.0)[:, None]
    streak_norm = (streak.astype(jnp.float32) / 3.0)[:, None]
    return board, tray_oh, combo_norm, streak_norm
