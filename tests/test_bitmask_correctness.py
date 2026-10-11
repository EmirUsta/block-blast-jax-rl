from __future__ import annotations

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

import block_blast_jax  # noqa: F401  (x64 enable)

import jax.numpy as jnp
import numpy as np

from block_blast_jax.board import (
    clear_full_lines_batch,
    compute_tray_action_mask,
)
from block_blast_jax.constants import (
    BOARD_SIZE,
    COL_MASKS_NP,
    NUM_ACTIONS,
    ROW_MASKS_NP,
)
from block_blast_jax.pieces import load_piece_set as load_jax_pieces

import pytest

pytest.importorskip("block_blast", reason="reference NumPy implementation not installed")
from block_blast.game.board import Board as BoardNP
from block_blast.game.pieces import load_piece_set as load_np_pieces


def grid_to_bitmask(grid_2d: np.ndarray) -> np.uint64:
    out = np.uint64(0)
    for r in range(8):
        for c in range(8):
            if grid_2d[r, c]:
                out |= np.uint64(1) << np.uint64(r * 8 + c)
    return out


def bitmask_to_grid(m: np.uint64) -> np.ndarray:
    g = np.zeros((8, 8), dtype=bool)
    for r in range(8):
        for c in range(8):
            if int(m) & (1 << (r * 8 + c)):
                g[r, c] = True
    return g


def test_clear_lines(n_trials: int = 200) -> tuple[int, int]:
    rng = np.random.default_rng(42)
    ok = 0
    for trial in range(n_trials):
        density = rng.uniform(0.0, 0.85)
        g2d = (rng.random((8, 8)) < density)
        if rng.random() < 0.4:
            g2d[int(rng.integers(0, 8)), :] = True
        if rng.random() < 0.4:
            g2d[:, int(rng.integers(0, 8))] = True
        b = BoardNP(8); b.grid = g2d.copy()
        ref_n = b.clear_full_lines()
        ref_was_cleared = b.is_empty() if ref_n > 0 else False
        ref_grid = b.grid.copy()
        gm = jnp.array([grid_to_bitmask(g2d)], dtype=jnp.uint64)
        ng, nc, bwc = clear_full_lines_batch(gm, jnp.asarray(ROW_MASKS_NP), jnp.asarray(COL_MASKS_NP))
        ours_n = int(nc[0])
        ours_was = bool(bwc[0])
        ours_grid = bitmask_to_grid(np.uint64(int(ng[0])))
        if ours_n != ref_n:
            print(f'  trial {trial}: n_cleared {ours_n} vs ref {ref_n}'); continue
        if ours_was != ref_was_cleared:
            print(f'  trial {trial}: board_was_cleared mismatch'); continue
        if not np.array_equal(ours_grid, ref_grid):
            print(f'  trial {trial}: grid mismatch'); continue
        ok += 1
    return ok, n_trials


def test_action_mask(n_trials: int = 100) -> tuple[int, int]:
    rng = np.random.default_rng(7)
    jx_ps = load_jax_pieces()
    np_pieces = load_np_pieces(str(__import__("pathlib").Path(__file__).resolve().parents[1] / "block_blast_jax" / "data" / "pieces.yaml"))
    pmf = jnp.asarray(jx_ps.masks_flat)
    vaf = jnp.asarray(jx_ps.valid_anchor_flat)
    P = jx_ps.num_pieces
    ok = 0
    for trial in range(n_trials):
        density = rng.uniform(0.0, 0.6)
        g2d = (rng.random((8, 8)) < density)
        tray = rng.choice(P, size=3, replace=False)
        ref = np.zeros(NUM_ACTIONS, dtype=bool)
        b = BoardNP(8); b.grid = g2d.copy()
        for slot in range(3):
            pidx = int(tray[slot])
            piece = np_pieces[pidx]
            for r in range(8):
                for c in range(8):
                    if r + piece.height <= 8 and c + piece.width <= 8:
                        if b.can_place(piece, c, r):
                            ref[slot * 64 + r * 8 + c] = True
        gm = jnp.array([grid_to_bitmask(g2d)], dtype=jnp.uint64)
        tray_t = jnp.asarray(tray.reshape(1, 3), dtype=jnp.int32)
        done = jnp.zeros(1, dtype=jnp.bool_)
        am = compute_tray_action_mask(gm, tray_t, pmf, vaf, done)
        ours = np.asarray(am[0])
        if not np.array_equal(ours, ref):
            diff = np.where(ours != ref)[0]
            print(f'  trial {trial}: tray={tray} mismatch on {len(diff)} actions'); continue
        ok += 1
    return ok, n_trials


if __name__ == "__main__":
    print("=== test_clear_lines ===")
    ok, n = test_clear_lines(200)
    print(f"  PASS: {ok}/{n}")

    print("=== test_action_mask ===")
    ok2, n2 = test_action_mask(100)
    print(f"  PASS: {ok2}/{n2}")

    overall = "ALL PASS" if (ok == 200 and ok2 == 100) else "FAIL"
    print(f"\n{overall}: clear={ok}/{200} action_mask={ok2}/{100}")
