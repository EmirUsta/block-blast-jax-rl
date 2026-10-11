from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

__all__ = ["BatchedBlockBlastEnv"]
__version__ = "0.1.0"


def __getattr__(name: str):
    if name == "BatchedBlockBlastEnv":
        from block_blast_jax.env import BatchedBlockBlastEnv as _Env
        return _Env
    raise AttributeError(name)
