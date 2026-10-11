from __future__ import annotations

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

import block_blast_jax
import jax, jax.numpy as jnp
import numpy as np

from block_blast_jax.alphazero.replay_buffer import init_buffer, add, sample


def make_batch(B, key):
    keys = jax.random.split(key, 6)
    return (
        jax.random.randint(keys[0], (B,), 0, 1<<32, dtype=jnp.uint32).astype(jnp.uint64),
        jax.random.randint(keys[1], (B, 3), 0, 31, dtype=jnp.int32),
        jax.random.randint(keys[2], (B,), 0, 10, dtype=jnp.int32),
        jax.random.randint(keys[3], (B,), 0, 3, dtype=jnp.int32),
        jax.random.uniform(keys[4], (B, 192), dtype=jnp.float32),
        jax.random.uniform(keys[5], (B,), dtype=jnp.float32),
    )


def test_basic_add():
    buf = init_buffer(capacity=2000)
    key = jax.random.PRNGKey(0)
    for i in range(10):
        key, sub = jax.random.split(key)
        b = make_batch(100, sub)
        buf = add(buf, *b)
    return int(buf.size) == 1000 and int(buf.write_idx) == 1000


def test_wraparound():
    buf = init_buffer(capacity=100_000)
    key = jax.random.PRNGKey(0)
    for i in range(20):
        key, sub = jax.random.split(key)
        buf = add(buf, *make_batch(5000, sub))
    assert int(buf.size) == 100_000 and int(buf.write_idx) == 0, f"size={buf.size}, write_idx={buf.write_idx}"
    for i in range(10):
        key, sub = jax.random.split(key)
        buf = add(buf, *make_batch(5000, sub))
    return int(buf.size) == 100_000 and int(buf.write_idx) == 50_000


def test_sample_shapes():
    buf = init_buffer(capacity=2000)
    key = jax.random.PRNGKey(0)
    for _ in range(10):
        key, sub = jax.random.split(key)
        buf = add(buf, *make_batch(100, sub))
    key, sub = jax.random.split(key)
    s = sample(buf, sub, batch_size=512)
    return (s["grid"].shape == (512,) and s["visit"].shape == (512, 192) and
            int(s["indices"].max()) < 1000)


def test_jit_compile():
    buf = init_buffer(capacity=10_000)
    add_jit = jax.jit(add)
    sample_jit = jax.jit(sample, static_argnums=2)
    key = jax.random.PRNGKey(0)
    b = make_batch(100, key)
    buf = add_jit(buf, *b)
    buf = add_jit(buf, *b)
    s = sample_jit(buf, key, 64)
    s2 = sample_jit(buf, key, 64)
    return s["grid"].shape == (64,) and s2["grid"].shape == (64,)


if __name__ == "__main__":
    tests = [
        ("test 1: 1000 add count", test_basic_add),
        ("test 4: wraparound 150K", test_wraparound),
        ("sample shapes", test_sample_shapes),
        ("jit smoke (2× cached)", test_jit_compile),
    ]
    failed = 0
    for name, fn in tests:
        try:
            ok = fn()
            print(f"  {'PASS' if ok else 'FAIL':5s} {name}")
            if not ok: failed += 1
        except Exception as e:
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
            failed += 1
    sys.exit(0 if failed == 0 else 1)
