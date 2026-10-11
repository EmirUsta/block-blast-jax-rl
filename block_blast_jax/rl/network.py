from __future__ import annotations

import numpy as np
import flax.linen as nn
import jax.numpy as jnp
from flax.linen.initializers import constant, orthogonal


class ActorCritic(nn.Module):
    hidden_size: int = 128
    num_actions: int = 192
    use_bf16: bool = False

    @nn.compact
    def __call__(self, obs: jnp.ndarray, action_mask: jnp.ndarray):
        compute_dtype = jnp.bfloat16 if self.use_bf16 else jnp.float32
        x = obs.astype(compute_dtype)
        x = nn.Dense(self.hidden_size, name="trunk1",
                     dtype=compute_dtype, param_dtype=jnp.float32)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_size, name="trunk2",
                     dtype=compute_dtype, param_dtype=jnp.float32)(x)
        x = nn.relu(x)
        logits = nn.Dense(self.num_actions, name="policy_head",
                           dtype=compute_dtype, param_dtype=jnp.float32)(x)
        logits = logits.astype(jnp.float32)
        masked = jnp.where(action_mask, logits, jnp.float32(-1e9))
        x_fp32 = x.astype(jnp.float32)
        value = nn.Dense(1, name="value_head",
                          dtype=jnp.float32, param_dtype=jnp.float32)(x_fp32).squeeze(-1)
        return masked, value


class CNNActorCritic(nn.Module):
    n_filters: int = 32
    n_conv: int = 2
    H: int = 128
    num_actions: int = 192

    @nn.compact
    def __call__(self, board, tray_oh, combo_norm, streak_norm):
        x = board
        for i in range(self.n_conv):
            x = nn.Conv(
                features=self.n_filters,
                kernel_size=(3, 3), padding="SAME",
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
                name=f"conv{i}",
            )(x)
            x = nn.relu(x)
        x = x.reshape(x.shape[0], -1)
        feat = jnp.concatenate([x, tray_oh, combo_norm, streak_norm], axis=-1)
        h = nn.Dense(self.H, kernel_init=orthogonal(np.sqrt(2)), name="trunk1")(feat)
        h = nn.relu(h)
        h = nn.Dense(self.H, kernel_init=orthogonal(np.sqrt(2)), name="trunk2")(h)
        h = nn.relu(h)
        policy = nn.Dense(self.num_actions, kernel_init=orthogonal(0.01), name="policy_head")(h)
        value = nn.Dense(1, kernel_init=orthogonal(1.0), name="value_head")(h).squeeze(-1)
        return policy, value
