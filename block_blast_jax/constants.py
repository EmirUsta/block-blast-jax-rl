from __future__ import annotations

import numpy as np

BOARD_SIZE: int = 8
TRAY_SIZE: int = 3
MAX_PIECE_DIM: int = 5
NUM_ACTIONS: int = TRAY_SIZE * BOARD_SIZE * BOARD_SIZE
DEFAULT_MAX_STEPS: int = 5000


ROW_MASKS_NP = np.array(
    [(np.uint64(0xFF) << np.uint64(8 * i)) for i in range(BOARD_SIZE)],
    dtype=np.uint64,
)
_COL_BASE = np.uint64(0x0101010101010101)
COL_MASKS_NP = np.array(
    [(_COL_BASE << np.uint64(j)) for j in range(BOARD_SIZE)],
    dtype=np.uint64,
)

FULL_BOARD_MASK_NP = np.uint64((1 << 64) - 1)

ROW_MASK_NO_LAST_COL_NP = np.uint64(0x7F7F7F7F7F7F7F7F)
ROW_MASK_NO_FIRST_COL_NP = np.uint64(0xFEFEFEFEFEFEFEFE)
TOP_ROW_MASK_NP = np.uint64(0xFF)
BOTTOM_ROW_MASK_NP = np.uint64(0xFF00000000000000)
LEFT_COL_MASK_NP = np.uint64(0x0101010101010101)
RIGHT_COL_MASK_NP = np.uint64(0x8080808080808080)
VALID_HORIZ_EDGE_MASK_NP = np.uint64(0x7F7F7F7F7F7F7F7F)
VALID_VERT_EDGE_MASK_NP = np.uint64(0x00FFFFFFFFFFFFFF)

LINES_TABLE_NP = np.array(
    [0, 10, 30, 60, 100, 150, 210, 280, 360, 450],
    dtype=np.int32,
)

LINE_BONUS_TABLE_NP = np.array(
    [0, 10, 20, 60, 120, 200] + [300] * 12,
    dtype=np.int32,
)

CASCADE_TABLE_NP = np.array(
    [1.0, 1.5, 2.5, 4.0, 5.0],
    dtype=np.float32,
)

BOARD_CLEAR_BONUS: int = 360

COMBO_BREAK_STREAK: int = 3
