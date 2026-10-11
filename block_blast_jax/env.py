from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from block_blast_jax.board import (
    clear_full_lines_batch_axes,
    compute_board_metrics,
    compute_tray_action_mask,
)
from block_blast_jax.constants import (
    BOARD_CLEAR_BONUS,
    BOARD_SIZE,
    CASCADE_TABLE_NP,
    COL_MASKS_NP,
    COMBO_BREAK_STREAK,
    DEFAULT_MAX_STEPS,
    LINE_BONUS_TABLE_NP,
    NUM_ACTIONS,
    ROW_MASKS_NP,
    TRAY_SIZE,
)
from block_blast_jax.encoding import decode_action_batched
from block_blast_jax.pieces import PieceSet, load_piece_set


def _gumbel_top3(key: jax.Array, num_pieces: int) -> jax.Array:
    u = jax.random.uniform(key, (num_pieces,), minval=1e-9, maxval=1.0)
    g = -jnp.log(-jnp.log(u))
    _, top = jax.lax.top_k(g, TRAY_SIZE)
    return top.astype(jnp.int32)


_gumbel_top3_batched = jax.vmap(_gumbel_top3, in_axes=(0, None))


def _make_initial_keys(seed: int, num_envs: int) -> jax.Array:
    base = jax.random.PRNGKey(seed)
    keys = jax.vmap(lambda i: jax.random.fold_in(base, i))(jnp.arange(num_envs, dtype=jnp.int32))
    return keys


def _initial_state(num_envs: int, seed: int, pieces: PieceSet, max_steps: int) -> dict[str, jax.Array]:
    keys = _make_initial_keys(seed, num_envs)
    sk = jax.vmap(lambda k: jax.random.split(k, 4))(keys)
    next_keys = sk[:, 0]
    init_tray_keys = sk[:, 1]
    sampled_tray = _gumbel_top3_batched(init_tray_keys, pieces.num_pieces)

    grid = jnp.zeros(num_envs, dtype=jnp.uint64)
    tray_idx = sampled_tray
    score = jnp.zeros(num_envs, dtype=jnp.int32)
    ep_len = jnp.zeros(num_envs, dtype=jnp.int32)
    done = jnp.zeros(num_envs, dtype=jnp.bool_)
    combo_count = jnp.zeros(num_envs, dtype=jnp.int32)
    no_clear_streak = jnp.zeros(num_envs, dtype=jnp.int32)

    pmf = jnp.asarray(pieces.masks_flat)
    vaf = jnp.asarray(pieces.valid_anchor_flat)
    am = compute_tray_action_mask(grid, tray_idx, pmf, vaf, done)
    return {
        "grid": grid,
        "tray_idx": tray_idx,
        "score": score,
        "ep_len": ep_len,
        "done": done,
        "action_mask": am,
        "key": next_keys,
        "combo_count": combo_count,
        "no_clear_streak": no_clear_streak,
    }


@partial(jax.jit, static_argnames=("auto_reset", "max_steps", "num_pieces"))
def _step(
    state: dict[str, jax.Array],
    actions: jax.Array,
    pmasks_flat: jax.Array,
    valid_anchor_flat: jax.Array,
    pieces_ncells: jax.Array,
    row_masks: jax.Array,
    col_masks: jax.Array,
    line_bonus_table: jax.Array,
    cascade_table: jax.Array,
    auto_reset: bool,
    max_steps: int,
    num_pieces: int,
) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
    grid = state["grid"]
    tray_idx = state["tray_idx"]
    score = state["score"]
    ep_len = state["ep_len"]
    done = state["done"]
    key = state["key"]
    combo_count = state["combo_count"]
    no_clear_streak = state["no_clear_streak"]

    actions = actions.astype(jnp.int32)
    slot, bit_idx = decode_action_batched(actions)

    sk = jax.vmap(lambda k: jax.random.split(k, 4))(key)
    next_key = sk[:, 0]
    refill_key = sk[:, 1]
    reset_tray_key = sk[:, 2]

    safe_slot = jnp.clip(slot, 0, TRAY_SIZE - 1)
    piece_type = jnp.take_along_axis(tray_idx, safe_slot[:, None], axis=1).squeeze(1)
    safe_piece = jnp.maximum(piece_type, 0)

    pmask_action = pmasks_flat[safe_piece, bit_idx]
    valid_anchor = valid_anchor_flat[safe_piece, bit_idx]

    not_done = ~done
    can_place = ((grid & pmask_action) == jnp.uint64(0)) & valid_anchor & (piece_type >= 0)
    valid_action = can_place & not_done
    invalid_now = (~valid_action) & not_done

    new_grid = jnp.where(valid_action, grid | pmask_action, grid)
    arange3 = jnp.arange(TRAY_SIZE, dtype=jnp.int32)
    take_mask = (arange3[None, :] == safe_slot[:, None]) & valid_action[:, None]
    new_tray = jnp.where(take_mask, jnp.int32(-1), tray_idx)

    cells_placed = jnp.where(valid_action, pieces_ncells[safe_piece], jnp.int32(0))

    new_grid_after_clear, rows_cleared, cols_cleared, bwc_raw = clear_full_lines_batch_axes(
        new_grid, row_masks, col_masks
    )
    apply_clear = valid_action
    new_grid = jnp.where(apply_clear, new_grid_after_clear, new_grid)
    rows_cleared = jnp.where(apply_clear, rows_cleared, jnp.int32(0))
    cols_cleared = jnp.where(apply_clear, cols_cleared, jnp.int32(0))
    is_board_clear = bwc_raw & apply_clear
    k = rows_cleared + cols_cleared
    cleared_this = k >= 1

    incremented_combo = combo_count + jnp.int32(1)
    incremented_streak = no_clear_streak + jnp.int32(1)
    tentative_combo = jnp.where(cleared_this, incremented_combo, combo_count)
    tentative_streak = jnp.where(cleared_this, jnp.int32(0), incremented_streak)
    break_combo = tentative_streak >= jnp.int32(COMBO_BREAK_STREAK)
    new_combo = jnp.where(break_combo, jnp.int32(0), tentative_combo)
    new_streak = jnp.where(break_combo, jnp.int32(0), tentative_streak)
    new_combo = jnp.where(valid_action, new_combo, combo_count)
    new_streak = jnp.where(valid_action, new_streak, no_clear_streak)

    min_axis = jnp.minimum(rows_cleared, cols_cleared)
    min_axis_clipped = jnp.clip(min_axis, 0, jnp.int32(cascade_table.shape[0] - 1))
    cascade_mult = cascade_table[min_axis_clipped]

    combo_mult = jnp.where(
        cleared_this,
        jnp.float32(1.0) + jnp.float32(0.5) * (new_combo - jnp.int32(1)).astype(jnp.float32),
        jnp.float32(1.0),
    )

    k_clipped = jnp.clip(k, 0, jnp.int32(line_bonus_table.shape[0] - 1))
    line_bonus_raw = line_bonus_table[k_clipped]
    line_score_float = line_bonus_raw.astype(jnp.float32) * combo_mult * cascade_mult
    line_score = jnp.round(line_score_float).astype(jnp.int32)

    board_clear_bonus = jnp.where(
        is_board_clear, jnp.int32(BOARD_CLEAR_BONUS), jnp.int32(0)
    )

    placement_pts = cells_placed
    move_score = placement_pts + line_score + board_clear_bonus
    move_score = move_score * valid_action.astype(jnp.int32)
    new_score = score + move_score

    tray_empty = (new_tray == -1).all(axis=-1)
    refill_needed = tray_empty & valid_action
    sampled_tray = _gumbel_top3_batched(refill_key, num_pieces)
    new_tray = jnp.where(refill_needed[:, None], sampled_tray, new_tray)

    am_check = compute_tray_action_mask(
        new_grid, new_tray, pmasks_flat, valid_anchor_flat,
        jnp.zeros_like(done),
    )
    no_action = ~am_check.any(axis=-1)

    new_ep_len = ep_len + valid_action.astype(jnp.int32)
    truncated = new_ep_len >= jnp.int32(max_steps)
    terminated = invalid_now | (valid_action & no_action)
    new_done = done | terminated | truncated

    new_action_mask = am_check & (~new_done)[:, None]

    if auto_reset:
        reset_tray = _gumbel_top3_batched(reset_tray_key, num_pieces)
        reset_grid = jnp.zeros_like(grid)
        reset_score = jnp.zeros_like(score)
        reset_ep_len = jnp.zeros_like(ep_len)
        reset_done = jnp.zeros_like(done)
        reset_combo = jnp.zeros_like(combo_count)
        reset_streak = jnp.zeros_like(no_clear_streak)

        new_grid = jnp.where(new_done, reset_grid, new_grid)
        new_tray = jnp.where(new_done[:, None], reset_tray, new_tray)
        new_score = jnp.where(new_done, reset_score, new_score)
        new_ep_len = jnp.where(new_done, reset_ep_len, new_ep_len)
        new_combo = jnp.where(new_done, reset_combo, new_combo)
        new_streak = jnp.where(new_done, reset_streak, new_streak)

        am_post_reset = compute_tray_action_mask(
            new_grid, new_tray, pmasks_flat, valid_anchor_flat, reset_done,
        )
        new_action_mask = jnp.where(new_done[:, None], am_post_reset, new_action_mask)
        new_done_after_reset = jnp.where(new_done, reset_done, new_done)
    else:
        new_done_after_reset = new_done

    new_state = {
        "grid": new_grid,
        "tray_idx": new_tray,
        "score": new_score,
        "ep_len": new_ep_len,
        "done": new_done_after_reset,
        "action_mask": new_action_mask,
        "key": next_key,
        "combo_count": new_combo,
        "no_clear_streak": new_streak,
    }
    num_empty, num_holes, boundary_edges = compute_board_metrics(new_grid)

    info = {
        "placement_pts": placement_pts,
        "line_bonus": line_bonus_raw * valid_action.astype(jnp.int32),
        "combo_mult": combo_mult,
        "cascade_mult": cascade_mult,
        "board_clear_bonus": board_clear_bonus,
        "lines_cleared": k,
        "rows_cleared": rows_cleared,
        "cols_cleared": cols_cleared,
        "combo_count": new_combo,
        "no_clear_streak": new_streak,
        "is_board_clear": is_board_clear,
        "score_delta": move_score,
        "cells_placed": cells_placed,
        "invalid_action": invalid_now,
        "terminated": terminated,
        "truncated": truncated,
        "reward": move_score,
        "num_empty_cells": num_empty,
        "num_holes": num_holes,
        "boundary_edges": boundary_edges,
        "cleared_lines": k,
        "board_was_cleared": is_board_clear,
    }
    return new_state, info


class BatchedBlockBlastEnv:

    def __init__(
        self,
        num_envs: int,
        seed: int = 0,
        auto_reset: bool = True,
        max_steps: int = DEFAULT_MAX_STEPS,
        piece_yaml: str | None = None,
    ) -> None:
        self.num_envs = int(num_envs)
        self.auto_reset = bool(auto_reset)
        self.max_steps = int(max_steps)

        self.pieces: PieceSet = load_piece_set(piece_yaml)
        self.num_pieces = self.pieces.num_pieces

        self._pmasks_flat = jnp.asarray(self.pieces.masks_flat)
        self._valid_anchor_flat = jnp.asarray(self.pieces.valid_anchor_flat)
        self._pieces_ncells = jnp.asarray(self.pieces.n_cells)
        self._row_masks = jnp.asarray(ROW_MASKS_NP)
        self._col_masks = jnp.asarray(COL_MASKS_NP)
        self._line_bonus_table = jnp.asarray(LINE_BONUS_TABLE_NP)
        self._cascade_table = jnp.asarray(CASCADE_TABLE_NP)

        self._seed = int(seed)
        self.state = _initial_state(self.num_envs, self._seed, self.pieces, self.max_steps)

    def reset(self, seed: int | None = None) -> dict[str, jax.Array]:
        if seed is not None:
            self._seed = int(seed)
        self.state = _initial_state(self.num_envs, self._seed, self.pieces, self.max_steps)
        return self.state

    def step(
        self, actions: jax.Array
    ) -> tuple[dict[str, jax.Array], jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
        new_state, info = _step(
            self.state,
            actions,
            self._pmasks_flat,
            self._valid_anchor_flat,
            self._pieces_ncells,
            self._row_masks,
            self._col_masks,
            self._line_bonus_table,
            self._cascade_table,
            auto_reset=self.auto_reset,
            max_steps=self.max_steps,
            num_pieces=self.num_pieces,
        )
        self.state = new_state
        return new_state, info["reward"], info["terminated"], info["truncated"], info

    def observe(self) -> dict[str, jax.Array]:
        return self.state
