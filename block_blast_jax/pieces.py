from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from block_blast_jax.constants import BOARD_SIZE, MAX_PIECE_DIM


@dataclass(frozen=True)
class PieceSet:
    cells: np.ndarray
    heights: np.ndarray
    widths: np.ndarray
    n_cells: np.ndarray
    masks: np.ndarray
    masks_flat: np.ndarray
    valid_anchor: np.ndarray
    valid_anchor_flat: np.ndarray
    ids: tuple[str, ...]

    @property
    def num_pieces(self) -> int:
        return self.cells.shape[0]


_DEFAULT_YAML = Path(__file__).parent / "data" / "pieces.yaml"


def _build_anchor_masks(piece_cells: np.ndarray, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    masks = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.uint64)
    valid = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=bool)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if r + h > BOARD_SIZE or c + w > BOARD_SIZE:
                continue
            valid[r, c] = True
            m = np.uint64(0)
            for i in range(h):
                for j in range(w):
                    if piece_cells[i, j]:
                        bit = (r + i) * BOARD_SIZE + (c + j)
                        m |= np.uint64(1) << np.uint64(bit)
            masks[r, c] = m
    return masks, valid


def load_piece_set(yaml_path: str | Path | None = None) -> PieceSet:
    yaml_path = Path(yaml_path) if yaml_path is not None else _DEFAULT_YAML
    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    raw = data["pieces"]
    P = len(raw)
    M = MAX_PIECE_DIM
    cells = np.zeros((P, M, M), dtype=bool)
    heights = np.zeros(P, dtype=np.int32)
    widths = np.zeros(P, dtype=np.int32)
    ncells = np.zeros(P, dtype=np.int32)
    ids: list[str] = []
    for idx, item in enumerate(raw):
        ids.append(item["id"])
        mat = np.array(item["cells"], dtype=bool)
        h, w = mat.shape
        cells[idx, :h, :w] = mat
        heights[idx] = h
        widths[idx] = w
        ncells[idx] = int(mat.sum())

    masks = np.zeros((P, BOARD_SIZE, BOARD_SIZE), dtype=np.uint64)
    valid = np.zeros((P, BOARD_SIZE, BOARD_SIZE), dtype=bool)
    for t in range(P):
        masks[t], valid[t] = _build_anchor_masks(
            cells[t], int(heights[t]), int(widths[t])
        )
    masks_flat = masks.reshape(P, BOARD_SIZE * BOARD_SIZE)
    valid_flat = valid.reshape(P, BOARD_SIZE * BOARD_SIZE)
    return PieceSet(
        cells=cells, heights=heights, widths=widths, n_cells=ncells,
        masks=masks, masks_flat=masks_flat,
        valid_anchor=valid, valid_anchor_flat=valid_flat,
        ids=tuple(ids),
    )
