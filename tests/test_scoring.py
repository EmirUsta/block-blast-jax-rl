from __future__ import annotations

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

import block_blast_jax

import jax
import jax.numpy as jnp
import numpy as np

from block_blast_jax import BatchedBlockBlastEnv
from block_blast_jax.constants import BOARD_CLEAR_BONUS


def _set_state(env: BatchedBlockBlastEnv, *, grid_2d: np.ndarray | None = None,
               tray: list[int] | None = None, combo_count: int = 0, no_clear_streak: int = 0,
               score: int = 0):
    if grid_2d is None:
        grid_2d = np.zeros((8, 8), dtype=bool)
    bits = np.uint64(0)
    for r in range(8):
        for c in range(8):
            if grid_2d[r, c]:
                bits |= np.uint64(1) << np.uint64(r * 8 + c)
    if tray is None:
        tray = [0, 0, 0]
    env.state = {
        "grid": jnp.array([bits], dtype=jnp.uint64),
        "tray_idx": jnp.asarray([tray], dtype=jnp.int32),
        "score": jnp.asarray([score], dtype=jnp.int32),
        "ep_len": jnp.zeros(1, dtype=jnp.int32),
        "done": jnp.zeros(1, dtype=jnp.bool_),
        "action_mask": jnp.zeros((1, 192), dtype=jnp.bool_),
        "key": jax.random.split(jax.random.PRNGKey(0), 1),
        "combo_count": jnp.asarray([combo_count], dtype=jnp.int32),
        "no_clear_streak": jnp.asarray([no_clear_streak], dtype=jnp.int32),
    }


def _act(slot: int, r: int, c: int) -> int:
    return slot * 64 + r * 8 + c


def _step1(env, action_id: int):
    obs, reward, term, trunc, info = env.step(jnp.asarray([action_id], dtype=jnp.int32))
    return obs, int(reward[0]), info



def test_basic_placement_no_clear():
    env = BatchedBlockBlastEnv(num_envs=1, seed=0, auto_reset=False)
    _set_state(env, tray=[9, 0, 1])
    obs, reward, info = _step1(env, _act(0, 0, 0))
    assert reward == 4, f"reward {reward} != 4"
    assert int(info["combo_count"][0]) == 0
    assert int(info["no_clear_streak"][0]) == 1
    assert int(info["lines_cleared"][0]) == 0
    return True


def test_single_row_clear_combo_start():
    env = BatchedBlockBlastEnv(num_envs=1, seed=0, auto_reset=False)
    g = np.zeros((8, 8), dtype=bool)
    g[0, 0:7] = True
    g[7, 0] = True
    _set_state(env, grid_2d=g, tray=[0, 1, 2])
    obs, reward, info = _step1(env, _act(0, 0, 7))
    expected = 1 + 10
    assert not bool(info["is_board_clear"][0])
    assert reward == expected, f"reward {reward} != {expected}"
    assert int(info["combo_count"][0]) == 1
    assert int(info["no_clear_streak"][0]) == 0
    assert int(info["rows_cleared"][0]) == 1
    assert int(info["cols_cleared"][0]) == 0
    return True


def test_combo_chain_2nd_clear():
    env = BatchedBlockBlastEnv(num_envs=1, seed=0, auto_reset=False)
    g = np.zeros((8, 8), dtype=bool)
    g[1, 0:7] = True
    g[7, 0] = True
    _set_state(env, grid_2d=g, tray=[0, 0, 0], combo_count=1, no_clear_streak=0)
    obs, reward, info = _step1(env, _act(0, 1, 7))
    expected = 1 + int(round(10 * 1.5 * 1.0))
    assert not bool(info["is_board_clear"][0])
    assert reward == expected, f"reward {reward} != {expected}"
    assert int(info["combo_count"][0]) == 2
    return True


def test_combo_break_3_no_clears():
    env = BatchedBlockBlastEnv(num_envs=1, seed=0, auto_reset=False)
    _set_state(env, tray=[0, 0, 0], combo_count=2, no_clear_streak=0)
    obs, _, info = _step1(env, _act(0, 0, 0))
    assert int(info["combo_count"][0]) == 2 and int(info["no_clear_streak"][0]) == 1
    env.state["tray_idx"] = jnp.asarray([[0, 0, 0]], dtype=jnp.int32)
    obs, _, info = _step1(env, _act(0, 0, 1))
    assert int(info["combo_count"][0]) == 2 and int(info["no_clear_streak"][0]) == 2
    env.state["tray_idx"] = jnp.asarray([[0, 0, 0]], dtype=jnp.int32)
    obs, _, info = _step1(env, _act(0, 0, 2))
    assert int(info["combo_count"][0]) == 0, f"combo {int(info['combo_count'][0])} != 0"
    assert int(info["no_clear_streak"][0]) == 0
    return True


def test_combo_survives_2_no_clears():
    env = BatchedBlockBlastEnv(num_envs=1, seed=0, auto_reset=False)
    _set_state(env, tray=[0, 0, 0], combo_count=1, no_clear_streak=0)
    obs, _, info = _step1(env, _act(0, 0, 0))
    assert int(info["combo_count"][0]) == 1 and int(info["no_clear_streak"][0]) == 1
    env.state["tray_idx"] = jnp.asarray([[0, 0, 0]], dtype=jnp.int32)
    obs, _, info = _step1(env, _act(0, 0, 1))
    assert int(info["combo_count"][0]) == 1 and int(info["no_clear_streak"][0]) == 2
    g = np.zeros((8, 8), dtype=bool)
    g[0, 0:7] = True
    bits = np.uint64(0)
    for r in range(8):
        for c in range(8):
            if g[r, c]:
                bits |= np.uint64(1) << np.uint64(r * 8 + c)
    g[7, 0] = True
    bits = np.uint64(0)
    for r in range(8):
        for c in range(8):
            if g[r, c]:
                bits |= np.uint64(1) << np.uint64(r * 8 + c)
    env.state["grid"] = jnp.asarray([bits], dtype=jnp.uint64)
    env.state["tray_idx"] = jnp.asarray([[0, 0, 0]], dtype=jnp.int32)
    obs, reward, info = _step1(env, _act(0, 0, 7))
    expected = 1 + int(round(10 * 1.5 * 1.0))
    assert not bool(info["is_board_clear"][0])
    assert reward == expected, f"reward {reward} != {expected}"
    assert int(info["combo_count"][0]) == 2
    assert int(info["no_clear_streak"][0]) == 0
    return True


def test_multi_line_bonus_4_lines():
    env = BatchedBlockBlastEnv(num_envs=1, seed=0, auto_reset=False)
    g = np.zeros((8, 8), dtype=bool)
    g[0, 0:6] = True
    g[1, 0:6] = True
    g[2:8, 7] = True
    g[7, 3] = True
    _set_state(env, grid_2d=g, tray=[9, 0, 1])
    obs, reward, info = _step1(env, _act(0, 0, 6))
    rows = int(info["rows_cleared"][0]); cols = int(info["cols_cleared"][0])
    k = int(info["lines_cleared"][0])
    expected_line = int(round(60 * 1.0 * 1.5))
    expected = 4 + expected_line
    assert not bool(info["is_board_clear"][0])
    assert rows == 2 and cols == 1 and k == 3, f"rows={rows} cols={cols} k={k}"
    assert reward == expected, f"reward {reward} != {expected}"
    return True


def test_cascade_multi_axis_1row_1col():
    env = BatchedBlockBlastEnv(num_envs=1, seed=0, auto_reset=False)
    g = np.zeros((8, 8), dtype=bool)
    g[0, 1:8] = True
    g[1:8, 0] = True
    g[7, 7] = True
    _set_state(env, grid_2d=g, tray=[0, 0, 0])
    obs, reward, info = _step1(env, _act(0, 0, 0))
    expected = 1 + int(round(20 * 1.0 * 1.5))
    assert not bool(info["is_board_clear"][0])
    assert reward == expected, f"reward {reward} != {expected}"
    assert int(info["rows_cleared"][0]) == 1
    assert int(info["cols_cleared"][0]) == 1
    return True


def test_cascade_2rows_2cols():
    env = BatchedBlockBlastEnv(num_envs=1, seed=0, auto_reset=False)
    g = np.zeros((8, 8), dtype=bool)
    g[0, 2:8] = True
    g[1, 2:8] = True
    g[2:8, 0] = True
    g[2:8, 1] = True
    g[7, 7] = True
    _set_state(env, grid_2d=g, tray=[9, 0, 1])
    obs, reward, info = _step1(env, _act(0, 0, 0))
    expected = 4 + int(round(120 * 1.0 * 2.5))
    rows = int(info["rows_cleared"][0]); cols = int(info["cols_cleared"][0])
    assert not bool(info["is_board_clear"][0])
    assert rows == 2 and cols == 2, f"rows={rows} cols={cols}"
    assert reward == expected, f"reward {reward} != {expected}"
    return True


def test_board_clear_bonus():
    env = BatchedBlockBlastEnv(num_envs=1, seed=0, auto_reset=False)
    g = np.ones((8, 8), dtype=bool)
    g[7, 7] = False
    _set_state(env, grid_2d=g, tray=[0, 0, 0])
    obs, reward, info = _step1(env, _act(0, 7, 7))
    expected = 1 + int(round(300 * 1.0 * 5.0)) + BOARD_CLEAR_BONUS
    assert reward == expected, f"reward {reward} != {expected}"
    assert bool(info["is_board_clear"][0])
    return True


def test_combo_persists_across_tray_refill():
    env = BatchedBlockBlastEnv(num_envs=1, seed=0, auto_reset=False)
    g = np.zeros((8, 8), dtype=bool)
    g[0, 0:7] = True
    _set_state(env, grid_2d=g, tray=[0, 0, 0])
    obs, _, info = _step1(env, _act(0, 0, 7))
    assert int(info["combo_count"][0]) == 1
    obs, _, info = _step1(env, _act(1, 1, 0))
    assert int(info["combo_count"][0]) == 1
    assert int(info["no_clear_streak"][0]) == 1
    obs, _, info = _step1(env, _act(2, 1, 1))
    assert int(info["combo_count"][0]) == 1, f"combo {int(info['combo_count'][0])} != 1"
    assert int(info["no_clear_streak"][0]) == 2
    new_tray = np.asarray(obs["tray_idx"][0])
    assert (new_tray >= 0).all(), f"tray after refill: {new_tray}"
    return True


def test_reset_clears_combo():
    env = BatchedBlockBlastEnv(num_envs=1, seed=0, auto_reset=False)
    _set_state(env, tray=[0, 0, 0], combo_count=5, no_clear_streak=2)
    env.reset(seed=42)
    obs = env.observe()
    assert int(obs["combo_count"][0]) == 0
    assert int(obs["no_clear_streak"][0]) == 0
    return True



TESTS = [
    ("test_basic_placement_no_clear", test_basic_placement_no_clear),
    ("test_single_row_clear_combo_start", test_single_row_clear_combo_start),
    ("test_combo_chain_2nd_clear", test_combo_chain_2nd_clear),
    ("test_combo_break_3_no_clears", test_combo_break_3_no_clears),
    ("test_combo_survives_2_no_clears", test_combo_survives_2_no_clears),
    ("test_multi_line_bonus_4_lines", test_multi_line_bonus_4_lines),
    ("test_cascade_multi_axis_1row_1col", test_cascade_multi_axis_1row_1col),
    ("test_cascade_2rows_2cols", test_cascade_2rows_2cols),
    ("test_board_clear_bonus", test_board_clear_bonus),
    ("test_combo_persists_across_tray_refill", test_combo_persists_across_tray_refill),
    ("test_reset_clears_combo", test_reset_clears_combo),
]


if __name__ == "__main__":
    passed = 0
    failed = []
    for name, fn in TESTS:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {name}: {e}")
            failed.append(name)
        except Exception as e:
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
            failed.append(name)
    print(f"\n{passed}/{len(TESTS)} passed")
    sys.exit(0 if not failed else 1)
