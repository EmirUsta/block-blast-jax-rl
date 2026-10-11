from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pygame

import block_blast_jax
from block_blast_jax import BatchedBlockBlastEnv
from block_blast_jax.constants import BOARD_SIZE, NUM_ACTIONS, TRAY_SIZE


CELL = 50
GRID_PX = BOARD_SIZE * CELL
LEFT_PAD = 24
TOP_PAD = 24
HUD_W = 220
TRAY_H = 140
WIN_W = LEFT_PAD + GRID_PX + 24 + HUD_W + 24
WIN_H = TOP_PAD + GRID_PX + 24 + TRAY_H + 24
GRID_X = LEFT_PAD
GRID_Y = TOP_PAD
HUD_X = LEFT_PAD + GRID_PX + 24
HUD_Y = TOP_PAD
TRAY_X = LEFT_PAD
TRAY_Y = TOP_PAD + GRID_PX + 24

BG = (24, 24, 30)
GRID_BG = (40, 40, 50)
GRID_LINE = (60, 60, 72)
EMPTY_CELL = (52, 52, 65)
TEXT = (235, 235, 240)
DIM = (160, 160, 170)
WARN = (255, 80, 80)
GOOD = (100, 230, 130)

PIECE_COLORS = [
    (244, 67, 54),
    (255, 152, 0),
    (255, 193, 7),
    (255, 235, 59),
    (139, 195, 74),
    (76, 175, 80),
    (0, 188, 212),
    (33, 150, 243),
    (63, 81, 181),
    (156, 39, 176),
    (233, 30, 99),
    (244, 67, 54),
    (244, 100, 80),
    (240, 130, 100),
    (240, 160, 120),
    (255, 180, 100),
    (255, 200, 130),
    (255, 215, 160),
    (255, 235, 200),
    (200, 230, 255),
    (180, 215, 255),
    (160, 195, 255),
    (140, 175, 255),
    (220, 100, 255),
    (200, 80, 235),
    (180, 60, 215),
    (160, 40, 195),
    (50, 220, 200),
    (40, 200, 180),
    (30, 180, 160),
    (20, 160, 140),
]


def grid_bits_to_array(bits_uint64: int) -> np.ndarray:
    g = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=bool)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if bits_uint64 & (1 << (r * BOARD_SIZE + c)):
                g[r, c] = True
    return g


def piece_color(t: int) -> tuple[int, int, int]:
    if t < 0 or t >= len(PIECE_COLORS):
        return (180, 180, 180)
    return PIECE_COLORS[t]


@dataclass
class PieceShape:
    cells: np.ndarray
    h: int
    w: int


class SingleEnv:
    def __init__(self, seed: int = 0):
        self.env = BatchedBlockBlastEnv(num_envs=1, seed=seed, auto_reset=False, max_steps=10**9)
        self.piece_set = self.env.pieces
        cells = np.asarray(self.piece_set.cells)
        hs = np.asarray(self.piece_set.heights)
        ws = np.asarray(self.piece_set.widths)
        self.shapes = [
            PieceShape(cells[t, :hs[t], :ws[t]], int(hs[t]), int(ws[t]))
            for t in range(self.piece_set.num_pieces)
        ]

    def obs(self) -> dict:
        s = self.env.observe()
        out = {
            "grid_bits": int(np.asarray(s["grid"][0])),
            "tray_idx": [int(x) for x in np.asarray(s["tray_idx"][0])],
            "score": int(s["score"][0]),
            "ep_len": int(s["ep_len"][0]),
            "done": bool(s["done"][0]),
            "combo_count": int(s["combo_count"][0]),
            "no_clear_streak": int(s["no_clear_streak"][0]),
            "action_mask": np.asarray(s["action_mask"][0]),
        }
        return out

    def step(self, action_id: int) -> dict:
        actions = jnp.array([action_id], dtype=jnp.int32)
        new_state, reward, term, trunc, info = self.env.step(actions)
        return {
            "reward": int(reward[0]),
            "placement_pts": int(info["placement_pts"][0]),
            "line_bonus": int(info["line_bonus"][0]),
            "combo_mult": float(info["combo_mult"][0]),
            "cascade_mult": float(info["cascade_mult"][0]),
            "board_clear_bonus": int(info["board_clear_bonus"][0]),
            "lines_cleared": int(info["lines_cleared"][0]),
            "rows_cleared": int(info["rows_cleared"][0]),
            "cols_cleared": int(info["cols_cleared"][0]),
            "is_board_clear": bool(info["is_board_clear"][0]),
            "invalid_action": bool(info["invalid_action"][0]),
            "terminated": bool(info["terminated"][0]),
        }

    def reset(self, seed: int | None = None):
        self.env.reset(seed=seed)


class Renderer:
    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        try:
            self.font_big = pygame.font.SysFont("Arial", 36, bold=True)
            self.font_med = pygame.font.SysFont("Arial", 18)
            self.font_sm = pygame.font.SysFont("Arial", 14)
        except Exception:
            self.font_big = pygame.font.Font(None, 36)
            self.font_med = pygame.font.Font(None, 18)
            self.font_sm = pygame.font.Font(None, 14)

    def cell_rect(self, r: int, c: int) -> pygame.Rect:
        return pygame.Rect(GRID_X + c * CELL + 1, GRID_Y + r * CELL + 1, CELL - 2, CELL - 2)

    def draw_grid(self, grid: np.ndarray, cell_color: np.ndarray, flash_alpha: int = 0,
                  flash_mask: np.ndarray | None = None):
        pygame.draw.rect(self.screen, GRID_BG,
                         (GRID_X - 4, GRID_Y - 4, GRID_PX + 8, GRID_PX + 8), border_radius=8)
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                rect = self.cell_rect(r, c)
                if grid[r, c]:
                    col = tuple(int(x) for x in cell_color[r, c])
                    pygame.draw.rect(self.screen, col, rect, border_radius=4)
                else:
                    pygame.draw.rect(self.screen, EMPTY_CELL, rect, border_radius=4)
        if flash_alpha > 0 and flash_mask is not None:
            overlay = pygame.Surface((CELL, CELL), pygame.SRCALPHA)
            overlay.fill((255, 255, 255, flash_alpha))
            for r in range(BOARD_SIZE):
                for c in range(BOARD_SIZE):
                    if flash_mask[r, c]:
                        rect = self.cell_rect(r, c)
                        self.screen.blit(overlay, rect.topleft)

    def draw_preview(self, anchor_r: int, anchor_c: int, shape: PieceShape, valid: bool):
        col = (90, 230, 130, 130) if valid else (230, 90, 90, 130)
        overlay_size = (CELL, CELL)
        for i in range(shape.h):
            for j in range(shape.w):
                if not shape.cells[i, j]:
                    continue
                rr = anchor_r + i
                cc = anchor_c + j
                if 0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE:
                    rect = self.cell_rect(rr, cc)
                    surf = pygame.Surface(overlay_size, pygame.SRCALPHA)
                    surf.fill(col)
                    self.screen.blit(surf, rect.topleft)

    def draw_tray(self, tray_idx: list[int], shapes: list[PieceShape], selected_slot: int):
        slot_w = (LEFT_PAD + GRID_PX + 24 - LEFT_PAD) // 3
        for s in range(TRAY_SIZE):
            x0 = TRAY_X + s * slot_w
            slot_rect = pygame.Rect(x0 + 4, TRAY_Y, slot_w - 8, TRAY_H - 8)
            color = (60, 70, 90) if selected_slot == s else (45, 50, 65)
            pygame.draw.rect(self.screen, color, slot_rect, border_radius=8)
            pygame.draw.rect(self.screen, GRID_LINE, slot_rect, 2, border_radius=8)
            label = self.font_sm.render(f"{s+1}", True, DIM)
            self.screen.blit(label, (slot_rect.x + 6, slot_rect.y + 4))
            t = tray_idx[s]
            if t < 0:
                used = self.font_sm.render("used", True, DIM)
                self.screen.blit(used, (slot_rect.centerx - 18, slot_rect.centery - 8))
                continue
            shape = shapes[t]
            mini = 18
            piece_w = shape.w * mini
            piece_h = shape.h * mini
            ox = slot_rect.centerx - piece_w // 2
            oy = slot_rect.centery - piece_h // 2 + 4
            col = piece_color(t)
            for i in range(shape.h):
                for j in range(shape.w):
                    if shape.cells[i, j]:
                        rect = pygame.Rect(ox + j * mini + 1, oy + i * mini + 1, mini - 2, mini - 2)
                        pygame.draw.rect(self.screen, col, rect, border_radius=3)

    def draw_hud(self, score: int, combo: int, streak: int, mult_text: str,
                 last_move_breakdown: str | None,
                 board_clear_text: str | None,
                 done: bool, ep_len: int):
        x, y = HUD_X, HUD_Y
        pygame.draw.rect(self.screen, (38, 42, 55), (x - 8, y - 8, HUD_W + 16, GRID_PX + 16),
                         border_radius=10)
        title = self.font_sm.render("SCORE", True, DIM)
        self.screen.blit(title, (x, y))
        score_surf = self.font_big.render(str(score), True, TEXT)
        self.screen.blit(score_surf, (x, y + 16))
        ep_surf = self.font_sm.render(f"step  {ep_len}", True, DIM)
        self.screen.blit(ep_surf, (x, y + 60))

        ny = y + 96
        combo_label = self.font_med.render(f"Combo: {combo}", True, TEXT)
        self.screen.blit(combo_label, (x, ny))
        mult_label = self.font_med.render(f"Mult:  {mult_text}", True, GOOD if combo >= 2 else TEXT)
        self.screen.blit(mult_label, (x, ny + 28))
        streak_color = WARN if streak >= 2 else TEXT
        streak_label = self.font_med.render(f"Streak: {streak}/3", True, streak_color)
        self.screen.blit(streak_label, (x, ny + 56))

        if last_move_breakdown:
            bd_y = ny + 100
            for i, line in enumerate(last_move_breakdown.split("\n")):
                surf = self.font_sm.render(line, True, DIM)
                self.screen.blit(surf, (x, bd_y + i * 16))

        if board_clear_text:
            t = self.font_med.render(board_clear_text, True, GOOD)
            self.screen.blit(t, (x, ny + 200))

        if done:
            t = self.font_med.render("GAME OVER", True, WARN)
            self.screen.blit(t, (x, HUD_Y + GRID_PX - 60))
            t2 = self.font_sm.render("Press R to reset", True, DIM)
            self.screen.blit(t2, (x, HUD_Y + GRID_PX - 32))


def grid_cell_colors(grid: np.ndarray, color_grid: np.ndarray) -> np.ndarray:
    return color_grid


def empty_cell_color_grid() -> np.ndarray:
    return np.zeros((BOARD_SIZE, BOARD_SIZE, 3), dtype=np.uint8)


def make_action_id(slot: int, r: int, c: int) -> int:
    return slot * 64 + r * 8 + c


def _load_ai_agent(checkpoint_path: str, num_pieces: int, use_mcts: bool = False,
                   mcts_sims: int = 200, env_for_mcts=None,
                   pb_c_init: float = 1.25, max_depth: int = 64):
    import json
    from pathlib import Path
    from flax.serialization import msgpack_restore
    from block_blast_jax.rl.network import ActorCritic, CNNActorCritic
    from block_blast_jax.rl.encode import encode_obs, encode_obs_2d

    ckpt_path = Path(checkpoint_path)
    raw = ckpt_path.read_bytes()
    blob = msgpack_restore(raw)
    cfg = dict(blob.get("config", {}))
    cfg_json = ckpt_path.parent / "config.json"
    if cfg_json.exists():
        try:
            cfg.update(json.loads(cfg_json.read_text()))
        except Exception:
            pass
    arch = cfg.get("arch_type", "mlp")
    H = int(cfg.get("hidden_size", 128))
    params = blob["params"]

    if arch == "az":
        net = ActorCritic(hidden_size=H, num_actions=192)
        if use_mcts:
            from block_blast_jax.alphazero.mcts import make_run_mcts
            assert env_for_mcts is not None, "use_mcts=True requires env_for_mcts"
            env_consts = dict(
                pmasks_flat=env_for_mcts._pmasks_flat,
                valid_anchor_flat=env_for_mcts._valid_anchor_flat,
                pieces_ncells=env_for_mcts._pieces_ncells,
                row_masks=env_for_mcts._row_masks,
                col_masks=env_for_mcts._col_masks,
                line_bonus_table=env_for_mcts._line_bonus_table,
                cascade_table=env_for_mcts._cascade_table,
            )
            run_mcts = make_run_mcts(net.apply, env_consts, num_pieces,
                                      max_steps=env_for_mcts.max_steps,
                                      discount=float(cfg.get("discount", 0.997)))
            run_mcts_jit = jax.jit(run_mcts, static_argnames=(
                "num_simulations", "max_depth", "dirichlet_fraction", "dirichlet_alpha",
                "pb_c_init", "pb_c_base", "use_dirichlet"))

            def policy(state, key):
                am = state["action_mask"]
                any_valid = am.any(axis=-1)
                out = run_mcts_jit(
                    params, state, key,
                    num_simulations=mcts_sims, max_depth=max_depth,
                    dirichlet_fraction=0.0, dirichlet_alpha=0.3,
                    pb_c_init=pb_c_init, pb_c_base=19652.0, use_dirichlet=False,
                )
                action = out.action
                return jnp.where(any_valid, action, jnp.int32(0))

            return policy, cfg

        @jax.jit
        def policy(state, key):
            obs = encode_obs(state, num_pieces)
            am = state["action_mask"]
            logits, _ = net.apply(params, obs, am)
            any_valid = am.any(axis=-1)
            action = jax.random.categorical(key, logits)
            action = jnp.where(any_valid, action, jnp.int32(0))
            return action

        return policy, cfg

    if arch == "cnn":
        net = CNNActorCritic(
            n_filters=int(cfg.get("n_filters", 32)),
            n_conv=int(cfg.get("n_conv", 2)),
            H=H, num_actions=192,
        )

        @jax.jit
        def policy(state, key):
            am = state["action_mask"]
            b, t, c, s = encode_obs_2d(state, num_pieces)
            raw_l, _ = net.apply(params, b, t, c, s)
            logits = jnp.where(am, raw_l, jnp.float32(-1e9))
            any_valid = am.any(axis=-1)
            action = jax.random.categorical(key, logits)
            action = jnp.where(any_valid, action, jnp.int32(0))
            return action
    else:
        net = ActorCritic(hidden_size=H, num_actions=192)

        @jax.jit
        def policy(state, key):
            obs = encode_obs(state, num_pieces)
            am = state["action_mask"]
            logits, _ = net.apply(params, obs, am)
            any_valid = am.any(axis=-1)
            action = jax.random.categorical(key, logits)
            action = jnp.where(any_valid, action, jnp.int32(0))
            return action

    return policy, cfg


def main(headless: bool = False, screenshot_dir: str | None = None,
         smoke_steps: int = 0, seed: int = 7,
         checkpoint: str | None = None, ai_step_ms: int = 200,
         use_mcts: bool = False, mcts_sims: int = 200,
         trace_path: str | None = None,
         pb_c_init: float = 1.25, max_depth: int = 64):
    use_mcts_flag = use_mcts
    mcts_sims_n = mcts_sims
    import json as _json
    import time as _time
    from pathlib import Path
    trace_fp = None
    trace_episode = 0
    if checkpoint:
        if trace_path is None:
            ts = _time.strftime("%Y%m%d_%H%M%S")
            trace_dir = Path("runs") / "replays"
            trace_dir.mkdir(parents=True, exist_ok=True)
            trace_path = str(trace_dir / f"replay_{ts}.jsonl")
        trace_fp = open(trace_path, "w", buffering=1)
        meta = {
            "event": "meta",
            "checkpoint": checkpoint,
            "use_mcts": use_mcts_flag,
            "mcts_sims": mcts_sims_n if use_mcts_flag else None,
            "seed": seed,
            "ai_step_ms": ai_step_ms,
            "ts": ts if trace_path else _time.strftime("%Y%m%d_%H%M%S"),
        }
        trace_fp.write(_json.dumps(meta) + "\n")
        print(f"[trace] writing to {trace_path}", flush=True)
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Block Blast — JAX")

    env = SingleEnv(seed=seed)
    rend = Renderer(screen)
    color_grid = empty_cell_color_grid()

    ai_policy = None
    ai_cfg = {}
    ai_paused = False
    ai_next_step_at = 0.0
    ai_key = jax.random.PRNGKey(seed + 99)
    if checkpoint:
        try:
            ai_policy, ai_cfg = _load_ai_agent(
                checkpoint, env.piece_set.num_pieces,
                use_mcts=use_mcts_flag, mcts_sims=mcts_sims_n,
                env_for_mcts=env.env if use_mcts_flag else None,
                pb_c_init=pb_c_init, max_depth=max_depth,
            )
            mode = f"MCTS sims={mcts_sims_n}" if use_mcts_flag else "argmax"
            print(f"[ai] Loaded checkpoint ({mode}): {checkpoint} "
                  f"arch={ai_cfg.get('arch_type','mlp')}")
        except Exception as e:
            print(f"[ai] Failed to load checkpoint: {e}")
            import traceback; traceback.print_exc()
            ai_policy = None

    selected_slot: int = -1
    last_breakdown: str | None = None
    breakdown_until: float = 0.0
    flash_until: float = 0.0
    flash_mask: np.ndarray | None = None
    board_clear_text: str | None = None
    board_clear_until: float = 0.0

    clock = pygame.time.Clock()
    running = True
    smoke_step_counter = 0
    smoke_screenshot_idx = 0

    def screenshot(name: str):
        nonlocal smoke_screenshot_idx
        if screenshot_dir:
            os.makedirs(screenshot_dir, exist_ok=True)
            path = os.path.join(screenshot_dir, name)
            pygame.image.save(screen, path)
            return path
        return None

    while running:
        now = pygame.time.get_ticks() / 1000.0
        ob = env.obs()
        grid = grid_bits_to_array(ob["grid_bits"])
        combo = ob["combo_count"]
        mult_text = f"{1.0 + 0.5 * max(combo - 1, 0):.1f}×" if combo > 0 else "1.0×"
        if breakdown_until and now > breakdown_until:
            last_breakdown = None
        if board_clear_until and now > board_clear_until:
            board_clear_text = None

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    if selected_slot >= 0:
                        selected_slot = -1
                    else:
                        running = False
                elif ev.key == pygame.K_r:
                    env.reset(seed=seed)
                    color_grid[:] = 0
                    selected_slot = -1
                    last_breakdown = None
                    board_clear_text = None
                elif ev.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                    s = {pygame.K_1: 0, pygame.K_2: 1, pygame.K_3: 2}[ev.key]
                    if 0 <= s < TRAY_SIZE and ob["tray_idx"][s] >= 0:
                        selected_slot = s
                elif ev.key == pygame.K_p and ai_policy is not None:
                    ai_paused = not ai_paused
                elif ev.key == pygame.K_PLUS or ev.key == pygame.K_EQUALS:
                    ai_step_ms = max(50, ai_step_ms - 50)
                elif ev.key == pygame.K_MINUS:
                    ai_step_ms = min(2000, ai_step_ms + 50)
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                mx, my = ev.pos
                slot_w = (LEFT_PAD + GRID_PX + 24 - LEFT_PAD) // 3
                if TRAY_Y <= my <= TRAY_Y + TRAY_H:
                    s = (mx - TRAY_X) // slot_w
                    if 0 <= s < TRAY_SIZE and ob["tray_idx"][s] >= 0:
                        selected_slot = int(s)
                        continue
                if (GRID_X <= mx < GRID_X + GRID_PX and
                        GRID_Y <= my < GRID_Y + GRID_PX and
                        selected_slot >= 0 and ob["tray_idx"][selected_slot] >= 0 and
                        not ob["done"]):
                    c = min(int((mx - GRID_X) // CELL), BOARD_SIZE - 1)
                    r = min(int((my - GRID_Y) // CELL), BOARD_SIZE - 1)
                    aid = make_action_id(selected_slot, int(r), int(c))
                    if ob["action_mask"][aid]:
                        piece_t = ob["tray_idx"][selected_slot]
                        shape = env.shapes[piece_t]
                        col = piece_color(piece_t)
                        for i in range(shape.h):
                            for j in range(shape.w):
                                if shape.cells[i, j]:
                                    color_grid[r + i, c + j] = col
                        info = env.step(aid)
                        if info["lines_cleared"] > 0:
                            new_grid = grid_bits_to_array(env.obs()["grid_bits"])
                            cleared = grid_bits_to_array(ob["grid_bits"])
                            placed_grid_bits = ob["grid_bits"]
                            pre_plus = cleared.copy()
                            for i in range(shape.h):
                                for j in range(shape.w):
                                    if shape.cells[i, j]:
                                        pre_plus[r + i, c + j] = True
                            cleared_cells = pre_plus & ~new_grid
                            flash_mask = cleared_cells.copy()
                            flash_until = now + 0.12
                            for rr in range(BOARD_SIZE):
                                for cc in range(BOARD_SIZE):
                                    if cleared_cells[rr, cc]:
                                        color_grid[rr, cc] = 0
                        parts = [f"+{info['placement_pts']} place"]
                        if info["lines_cleared"] > 0:
                            line_total = int(round(info["line_bonus"] * info["combo_mult"] * info["cascade_mult"]))
                            parts.append(f"+{line_total} line ({info['line_bonus']}×{info['combo_mult']:.1f}×{info['cascade_mult']:.1f})")
                        if info["board_clear_bonus"] > 0:
                            parts.append(f"+{info['board_clear_bonus']} board clear")
                            board_clear_text = f"BOARD CLEAR! +{info['board_clear_bonus']}"
                            board_clear_until = now + 1.0
                        last_breakdown = " = +%d\n%s" % (info["reward"], " ".join(parts))
                        breakdown_until = now + 2.0
                        selected_slot = -1
                    else:
                        pass

        if ai_policy is not None and not ai_paused and not ob["done"] and now >= ai_next_step_at:
            ai_key, sub = jax.random.split(ai_key)
            action_jx = ai_policy(env.env.state, sub)
            aid = int(np.asarray(action_jx)[0])
            if ob["action_mask"][aid]:
                slot_pick = aid // 64
                r_pick = (aid % 64) // 8
                c_pick = (aid % 64) % 8
                piece_t = ob["tray_idx"][slot_pick]
                shape = env.shapes[piece_t]
                col = piece_color(piece_t)
                for i in range(shape.h):
                    for j in range(shape.w):
                        if shape.cells[i, j]:
                            color_grid[r_pick + i, c_pick + j] = col
                info = env.step(aid)
                if info["lines_cleared"] > 0:
                    new_grid = grid_bits_to_array(env.obs()["grid_bits"])
                    pre_plus = grid_bits_to_array(ob["grid_bits"])
                    for i in range(shape.h):
                        for j in range(shape.w):
                            if shape.cells[i, j]:
                                pre_plus[r_pick + i, c_pick + j] = True
                    cleared_cells = pre_plus & ~new_grid
                    flash_mask = cleared_cells.copy()
                    flash_until = now + 0.12
                    for rr in range(BOARD_SIZE):
                        for cc in range(BOARD_SIZE):
                            if cleared_cells[rr, cc]:
                                color_grid[rr, cc] = 0
                parts = [f"+{info['placement_pts']}"]
                if info["lines_cleared"] > 0:
                    line_total = int(round(info["line_bonus"] * info["combo_mult"] * info["cascade_mult"]))
                    parts.append(f"+{line_total}")
                if info["board_clear_bonus"] > 0:
                    parts.append(f"+{info['board_clear_bonus']} BC")
                    board_clear_text = f"BOARD CLEAR! +{info['board_clear_bonus']}"
                    board_clear_until = now + 1.0
                last_breakdown = f"AI: {' '.join(parts)} = +{info['reward']}"
                breakdown_until = now + 5.0
                if trace_fp is not None:
                    ob_after = env.obs()
                    rec = {
                        "event": "step",
                        "ep": trace_episode,
                        "t": int(ob_after["ep_len"]),
                        "action": int(aid),
                        "slot": int(slot_pick),
                        "r": int(r_pick),
                        "c": int(c_pick),
                        "piece_t": int(piece_t),
                        "placement_pts": int(info["placement_pts"]),
                        "lines_cleared": int(info["lines_cleared"]),
                        "line_bonus": int(info["line_bonus"]),
                        "combo_mult": float(info["combo_mult"]),
                        "cascade_mult": float(info["cascade_mult"]),
                        "board_clear_bonus": int(info["board_clear_bonus"]),
                        "reward": int(info["reward"]),
                        "score": int(ob_after["score"]),
                        "combo": int(ob_after["combo_count"]),
                        "no_clear_streak": int(ob_after["no_clear_streak"]),
                        "done": bool(ob_after["done"]),
                    }
                    trace_fp.write(_json.dumps(rec) + "\n")
                    if bool(ob_after["done"]):
                        summary = {
                            "event": "summary",
                            "ep": trace_episode,
                            "ep_len": int(ob_after["ep_len"]),
                            "score": int(ob_after["score"]),
                        }
                        trace_fp.write(_json.dumps(summary) + "\n")
                        print(f"[trace] ep{trace_episode} done: ep_len={int(ob_after['ep_len'])} score={int(ob_after['score'])}", flush=True)
                        trace_episode += 1
            ai_next_step_at = now + ai_step_ms / 1000.0

        if smoke_steps > 0 and smoke_step_counter < smoke_steps and not ob["done"]:
            am = ob["action_mask"]
            valid = np.flatnonzero(am)
            if valid.size > 0:
                rng = np.random.default_rng(0xBEEF + smoke_step_counter)
                aid = int(rng.choice(valid))
                slot_pick = aid // 64
                r_pick = (aid % 64) // 8
                c_pick = (aid % 64) % 8
                piece_t = ob["tray_idx"][slot_pick]
                shape = env.shapes[piece_t]
                col = piece_color(piece_t)
                for i in range(shape.h):
                    for j in range(shape.w):
                        if shape.cells[i, j]:
                            color_grid[r_pick + i, c_pick + j] = col
                info = env.step(aid)
                if info["lines_cleared"] > 0:
                    new_grid = grid_bits_to_array(env.obs()["grid_bits"])
                    pre_plus = grid_bits_to_array(ob["grid_bits"])
                    for i in range(shape.h):
                        for j in range(shape.w):
                            if shape.cells[i, j]:
                                pre_plus[r_pick + i, c_pick + j] = True
                    cleared_cells = pre_plus & ~new_grid
                    for rr in range(BOARD_SIZE):
                        for cc in range(BOARD_SIZE):
                            if cleared_cells[rr, cc]:
                                color_grid[rr, cc] = 0
                parts = [f"+{info['placement_pts']}"]
                if info["lines_cleared"] > 0:
                    line_total = int(round(info["line_bonus"] * info["combo_mult"] * info["cascade_mult"]))
                    parts.append(f"+{line_total}")
                if info["board_clear_bonus"] > 0:
                    parts.append(f"+{info['board_clear_bonus']} BC")
                last_breakdown = f"step {smoke_step_counter+1}: {' '.join(parts)} = +{info['reward']}"
                breakdown_until = now + 5.0
                smoke_step_counter += 1

        screen.fill(BG)
        rend.draw_grid(grid, color_grid,
                       flash_alpha=int(220 if now < flash_until else 0),
                       flash_mask=flash_mask if now < flash_until else None)

        if selected_slot >= 0 and ob["tray_idx"][selected_slot] >= 0 and not ob["done"]:
            mx, my = pygame.mouse.get_pos()
            if (GRID_X <= mx <= GRID_X + GRID_PX and
                    GRID_Y <= my <= GRID_Y + GRID_PX):
                c_hover = min(int((mx - GRID_X) // CELL), BOARD_SIZE - 1)
                r_hover = min(int((my - GRID_Y) // CELL), BOARD_SIZE - 1)
                aid = make_action_id(selected_slot, int(r_hover), int(c_hover))
                valid = bool(ob["action_mask"][aid])
                shape = env.shapes[ob["tray_idx"][selected_slot]]
                rend.draw_preview(int(r_hover), int(c_hover), shape, valid)

        rend.draw_tray(ob["tray_idx"], env.shapes, selected_slot)
        rend.draw_hud(
            ob["score"], ob["combo_count"], ob["no_clear_streak"],
            mult_text, last_breakdown, board_clear_text,
            ob["done"], ob["ep_len"],
        )

        pygame.display.flip()
        clock.tick(60)

        if checkpoint and screenshot_dir and ob["ep_len"] in (5, 12) and ai_next_step_at > 0:
            screenshot(f"ai_replay_ep{ob['ep_len']}.png")

        if smoke_steps > 0 and screenshot_dir:
            if smoke_step_counter == 1 and smoke_screenshot_idx == 0:
                screenshot(f"01_first_move.png"); smoke_screenshot_idx += 1
            if smoke_step_counter == 5 and smoke_screenshot_idx == 1:
                screenshot(f"02_after_5_moves.png"); smoke_screenshot_idx += 1
            if board_clear_text and smoke_screenshot_idx < 3:
                screenshot(f"03_board_clear.png"); smoke_screenshot_idx += 1
            if ob["done"] and smoke_screenshot_idx < 4:
                screenshot(f"04_game_over.png"); smoke_screenshot_idx += 1
                running = False
            if smoke_step_counter >= smoke_steps and smoke_screenshot_idx >= 2:
                screenshot(f"05_final.png")
                running = False

    pygame.quit()
    if trace_fp is not None:
        trace_fp.close()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--smoke_steps", type=int, default=0)
    ap.add_argument("--screenshot_dir", default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--checkpoint", default=None,
                    help="msgpack checkpoint path → AI replay modu (P pause, +/- speed)")
    ap.add_argument("--ai_step_ms", type=int, default=200)
    ap.add_argument("--use_mcts", action="store_true",
                    help="AZ checkpointlerinde inference için MCTS kullan")
    ap.add_argument("--mcts_sims", type=int, default=200)
    ap.add_argument("--trace_path", default=None,
                    help="JSONL trace dosyası (yoksa runs/replays/replay_<ts>.jsonl)")
    ap.add_argument("--pb_c_init", type=float, default=1.25,
                    help="PUCT keşif katsayısı (eğitimde 1.25)")
    ap.add_argument("--max_depth", type=int, default=64,
                    help="MCTS max ağaç derinliği (eğitimde 64)")
    args = ap.parse_args()
    main(headless=args.headless, smoke_steps=args.smoke_steps,
         screenshot_dir=args.screenshot_dir, seed=args.seed,
         checkpoint=args.checkpoint, ai_step_ms=args.ai_step_ms,
         use_mcts=args.use_mcts, mcts_sims=args.mcts_sims,
         trace_path=args.trace_path,
         pb_c_init=args.pb_c_init, max_depth=args.max_depth)
