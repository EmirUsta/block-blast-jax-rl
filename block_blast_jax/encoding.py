from __future__ import annotations

import jax.numpy as jnp


def decode_action_batched(actions: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    actions = actions.astype(jnp.int32)
    slot = actions // 64
    bit_idx = actions % 64
    return slot, bit_idx
