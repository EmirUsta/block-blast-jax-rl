from __future__ import annotations

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

import block_blast_jax

import jax
import jax.numpy as jnp
import numpy as np

from block_blast_jax import BatchedBlockBlastEnv
from tests.test_bitmask_correctness import bitmask_to_grid, grid_to_bitmask  # noqa: E402

import pytest

pytest.importorskip("block_blast", reason="reference NumPy implementation not installed")
from block_blast.game.game import Game
from block_blast.game.pieces import load_piece_set as load_np_pieces


PIECES_YAML = str(__import__("pathlib").Path(__file__).resolve().parents[1] / "block_blast_jax" / "data" / "pieces.yaml")


def make_np_game(np_pieces):
    cfg = {
        "board": {"size": 8},
        "tray": {"size": 3, "same_tray_unique": True},
        "pieces": {"config_file": PIECES_YAML},
        "episode": {"max_steps": 5000},
    }
    rng = np.random.default_rng(0)
    return Game(cfg, rng=rng)


def run_one_scenario(env: BatchedBlockBlastEnv, np_game: Game, np_pieces, rng) -> str | None:
    P = env.num_pieces
    g2d = (rng.random((8, 8)) < 0.25)
    tray = rng.choice(P, size=3, replace=False)
    np_game.board.grid = g2d.copy()
    np_game.tray.slots = [np_pieces[int(t)] for t in tray]
    np_game.score = 0
    np_game.steps = 0
    np_game.game_over = False
    env.state = {
        "grid": jnp.array([grid_to_bitmask(g2d)], dtype=jnp.uint64),
        "tray_idx": jnp.asarray(tray.reshape(1, 3), dtype=jnp.int32),
        "score": jnp.zeros(1, dtype=jnp.int32),
        "ep_len": jnp.zeros(1, dtype=jnp.int32),
        "done": jnp.zeros(1, dtype=jnp.bool_),
        "action_mask": jnp.zeros((1, 192), dtype=jnp.bool_),
        "key": jax.random.split(jax.random.PRNGKey(0), 1),
        "combo_count": jnp.zeros(1, dtype=jnp.int32),
        "no_clear_streak": jnp.zeros(1, dtype=jnp.int32),
    }

    np_mask = np.zeros(192, dtype=bool)
    for slot in range(3):
        piece = np_game.tray.get(slot)
        if piece is None:
            continue
        pm = np_game.board.placement_mask(piece)
        H, W = pm.shape
        for r in range(H):
            for c in range(W):
                if pm[r, c]:
                    np_mask[slot * 64 + r * 8 + c] = True
    valid = np.flatnonzero(np_mask)
    if valid.size == 0:
        return None

    a = int(rng.choice(valid))
    slot = a // 64
    bit_idx = a % 64
    r = bit_idx // 8
    c = bit_idx % 8
    piece_idx = int(tray[slot])

    res = np_game.step(slot, c, r)
    np_grid_after = np_game.board.grid.copy()
    np_score_delta = res.score_delta
    np_cleared = res.cleared_lines
    np_was_cleared = res.board_was_cleared
    np_cells_placed = res.cells_placed
    np_done = np_game.game_over

    actions = jnp.array([a], dtype=jnp.int32)
    obs, reward, term, trunc, info = env.step(actions)
    jx_grid_after = bitmask_to_grid(np.uint64(int(obs["grid"][0])))
    jx_score_delta = int(reward[0])
    jx_cleared = int(info["cleared_lines"][0])
    jx_was_cleared = bool(info["board_was_cleared"][0])
    jx_cells_placed = int(info["cells_placed"][0])
    jx_done = bool(obs["done"][0])

    if env.auto_reset and (np_done or jx_done):
        pass
    else:
        if not np.array_equal(jx_grid_after, np_grid_after):
            return f"grid mismatch: tray={tray} action={a} (slot={slot}, r={r}, c={c}) piece={piece_idx}"

    if jx_cleared != np_cleared:
        return f"cleared {jx_cleared} vs {np_cleared}"
    if jx_was_cleared != np_was_cleared:
        return f"board_was_cleared {jx_was_cleared} vs {np_was_cleared}"
    if jx_cells_placed != np_cells_placed:
        return f"cells_placed {jx_cells_placed} vs {np_cells_placed}"
    if jx_done != np_done:
        return f"done {jx_done} vs {np_done}"
    return None


def main(n: int = 1000):
    np_pieces = load_np_pieces(PIECES_YAML)
    env = BatchedBlockBlastEnv(num_envs=1, seed=0, auto_reset=False)
    np_game = make_np_game(np_pieces)
    rng = np.random.default_rng(0xDEADBEEF)
    fails = []
    skipped = 0
    for i in range(n):
        err = run_one_scenario(env, np_game, np_pieces, rng)
        if err is None:
            continue
        if err == "skipped":
            skipped += 1
        else:
            fails.append((i, err))
    passed = n - len(fails)
    print(f"Logic parity: {passed}/{n} PASS  ({len(fails)} fail, {skipped} skipped)")
    for i, e in fails[:5]:
        print(f"  - scenario {i}: {e}")
    return len(fails) == 0


if __name__ == "__main__":
    ok = main(1000)
    sys.exit(0 if ok else 1)
