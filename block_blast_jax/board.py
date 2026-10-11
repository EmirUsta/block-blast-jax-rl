from __future__ import annotations

import jax
import jax.numpy as jnp

from block_blast_jax.constants import (
    BOARD_SIZE, FULL_BOARD_MASK_NP, NUM_ACTIONS, TRAY_SIZE,
    BOTTOM_ROW_MASK_NP, LEFT_COL_MASK_NP, RIGHT_COL_MASK_NP, TOP_ROW_MASK_NP,
    ROW_MASK_NO_FIRST_COL_NP, ROW_MASK_NO_LAST_COL_NP,
    VALID_HORIZ_EDGE_MASK_NP, VALID_VERT_EDGE_MASK_NP,
)


def clear_full_lines_batch(
    grid: jnp.ndarray,
    row_masks: jnp.ndarray,
    col_masks: jnp.ndarray,
):
    new_grid, rows_cleared, cols_cleared, bwc = clear_full_lines_batch_axes(
        grid, row_masks, col_masks
    )
    n_cleared = rows_cleared + cols_cleared
    return new_grid, n_cleared, bwc


def clear_full_lines_batch_axes(
    grid: jnp.ndarray,
    row_masks: jnp.ndarray,
    col_masks: jnp.ndarray,
):
    g = grid[:, None]
    full_rows = (g & row_masks[None, :]) == row_masks[None, :]
    full_cols = (g & col_masks[None, :]) == col_masks[None, :]

    rows_cleared = full_rows.astype(jnp.int32).sum(axis=-1)
    cols_cleared = full_cols.astype(jnp.int32).sum(axis=-1)

    rmask = jnp.where(full_rows, row_masks[None, :], jnp.uint64(0))
    cmask = jnp.where(full_cols, col_masks[None, :], jnp.uint64(0))
    clear_mask = jnp.bitwise_or.reduce(rmask, axis=-1) | jnp.bitwise_or.reduce(cmask, axis=-1)
    new_grid = grid & ~clear_mask
    n_total = rows_cleared + cols_cleared
    bwc = (new_grid == jnp.uint64(0)) & (n_total > 0)
    return new_grid, rows_cleared, cols_cleared, bwc


def compute_board_metrics(grid: jnp.ndarray):
    full = jnp.uint64(int(FULL_BOARD_MASK_NP))
    pc_grid = jax.lax.population_count(grid).astype(jnp.int32)
    num_empty = jnp.int32(64) - pc_grid

    horiz_xor = grid ^ (grid >> jnp.uint64(1))
    horiz_edges = horiz_xor & jnp.uint64(int(VALID_HORIZ_EDGE_MASK_NP))
    vert_xor = grid ^ (grid >> jnp.uint64(8))
    vert_edges = vert_xor & jnp.uint64(int(VALID_VERT_EDGE_MASK_NP))
    boundary = (jax.lax.population_count(horiz_edges).astype(jnp.int32) +
                jax.lax.population_count(vert_edges).astype(jnp.int32))

    up_filled = (grid << jnp.uint64(8)) | jnp.uint64(int(TOP_ROW_MASK_NP))
    down_filled = (grid >> jnp.uint64(8)) | jnp.uint64(int(BOTTOM_ROW_MASK_NP))
    left_filled = ((grid << jnp.uint64(1)) & jnp.uint64(int(ROW_MASK_NO_FIRST_COL_NP))) | jnp.uint64(int(LEFT_COL_MASK_NP))
    right_filled = ((grid >> jnp.uint64(1)) & jnp.uint64(int(ROW_MASK_NO_LAST_COL_NP))) | jnp.uint64(int(RIGHT_COL_MASK_NP))
    all_neigh = up_filled & down_filled & left_filled & right_filled
    empty = (~grid) & full
    holes = empty & all_neigh
    num_holes = jax.lax.population_count(holes).astype(jnp.int32)

    return num_empty, num_holes, boundary


def compute_tray_action_mask(
    grid: jnp.ndarray,
    tray_idx: jnp.ndarray,
    piece_masks_flat: jnp.ndarray,
    valid_anchor_flat: jnp.ndarray,
    done: jnp.ndarray,
) -> jnp.ndarray:
    N = grid.shape[0]
    safe_tray = jnp.maximum(tray_idx, 0)
    pmasks_per_action = piece_masks_flat[safe_tray]
    valid_anchor = valid_anchor_flat[safe_tray]
    can = (grid[:, None, None] & pmasks_per_action) == jnp.uint64(0)
    valid_slot = (tray_idx >= 0)[:, :, None]
    not_done = (~done)[:, None, None]
    out = can & valid_anchor & valid_slot & not_done
    return out.reshape(N, NUM_ACTIONS)
