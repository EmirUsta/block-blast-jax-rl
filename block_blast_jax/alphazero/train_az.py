from __future__ import annotations

import argparse
import json
import os
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.serialization import msgpack_restore, msgpack_serialize
from flax.training import train_state as ts
from tensorboardX import SummaryWriter

import block_blast_jax
from block_blast_jax import BatchedBlockBlastEnv
from block_blast_jax.rl.network import ActorCritic
from block_blast_jax.rl.encode import encode_obs, feature_dim
from block_blast_jax.alphazero.config import AZConfig
from block_blast_jax.alphazero.mcts import make_run_mcts
from block_blast_jax.alphazero.replay_buffer import init_buffer, add as buf_add, sample as buf_sample
from block_blast_jax.alphazero.selfplay import self_play_iteration



def make_run_dir(cfg: AZConfig) -> Path:
    p = Path(cfg.checkpoint_dir) / cfg.run_name
    p.mkdir(parents=True, exist_ok=True)
    (p / "checkpoints").mkdir(exist_ok=True)
    return p


def atomic_write_msgpack(path: Path, blob):
    raw = msgpack_serialize(jax.device_get(blob))
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(raw)
    os.replace(tmp, path)


def reset_value_head(params):
    new = dict(params)
    new["params"] = dict(new["params"])
    if "value_head" in new["params"]:
        key = jax.random.PRNGKey(123)
        h = new["params"]["value_head"]
        new_h = dict(h)
        kernel_shape = h["kernel"].shape
        new_h["kernel"] = jax.nn.initializers.orthogonal(1.0)(key, kernel_shape, jnp.float32)
        new_h["bias"] = jnp.zeros_like(h["bias"])
        new["params"]["value_head"] = new_h
    return new


def load_warmstart(path: str):
    raw = Path(path).read_bytes()
    blob = msgpack_restore(raw)
    return blob["params"], blob.get("config", {})


def make_optimizer(cfg: AZConfig, grad_clip: float | None = None):
    gc = grad_clip if grad_clip is not None else cfg.grad_clip
    main_tx = optax.chain(
        optax.clip_by_global_norm(gc),
        optax.adam(cfg.lr),
    )
    head_tx = optax.chain(
        optax.clip_by_global_norm(gc),
        optax.adam(cfg.lr * cfg.lr_value_head_mult),
    )

    def label_fn(params):
        return jax.tree_util.tree_map_with_path(
            lambda kp, _: "value_head" if any("value_head" in str(p) for p in kp) else "main",
            params,
        )

    return optax.multi_transform({"main": main_tx, "value_head": head_tx}, label_fn)


def make_train_step(net, cfg: AZConfig, num_pieces: int):
    vf_coef = float(cfg.vf_coef)
    kl_coef = float(cfg.kl_anchor_coef)
    use_kl = kl_coef > 0.0

    if not use_kl:
        @jax.jit
        def loss_fn(params, batch_obs, batch_action_mask, batch_visit, batch_value):
            logits, value = net.apply(params, batch_obs, batch_action_mask)
            log_probs = jax.nn.log_softmax(logits)
            policy_loss = -(batch_visit * log_probs).sum(axis=-1).mean()
            value_loss = ((value - batch_value) ** 2).mean()
            total = policy_loss + vf_coef * value_loss
            return total, (policy_loss, value_loss, jnp.float32(0.0))

        @jax.jit
        def step_fn(state, batch_obs, batch_action_mask, batch_visit, batch_value, ref_params):
            (loss, (pl, vl, kl)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                state.params, batch_obs, batch_action_mask, batch_visit, batch_value
            )
            state = state.apply_gradients(grads=grads)
            gn = optax.global_norm(grads)
            return state, {"loss": loss, "policy_loss": pl, "value_loss": vl,
                           "kl_anchor": kl, "grad_norm": gn}
    else:
        @jax.jit
        def loss_fn(params, ref_params, batch_obs, batch_action_mask, batch_visit, batch_value):
            logits, value = net.apply(params, batch_obs, batch_action_mask)
            log_probs = jax.nn.log_softmax(logits)
            policy_loss = -(batch_visit * log_probs).sum(axis=-1).mean()
            value_loss = ((value - batch_value) ** 2).mean()
            ref_logits, _ = net.apply(ref_params, batch_obs, batch_action_mask)
            ref_logp = jax.nn.log_softmax(ref_logits)
            cur_p = jnp.exp(log_probs)
            kl = (cur_p * (log_probs - ref_logp)).sum(axis=-1).mean()
            total = policy_loss + vf_coef * value_loss + kl_coef * kl
            return total, (policy_loss, value_loss, kl)

        @jax.jit
        def step_fn(state, batch_obs, batch_action_mask, batch_visit, batch_value, ref_params):
            (loss, (pl, vl, kl)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                state.params, ref_params, batch_obs, batch_action_mask, batch_visit, batch_value
            )
            state = state.apply_gradients(grads=grads)
            gn = optax.global_norm(grads)
            return state, {"loss": loss, "policy_loss": pl, "value_loss": vl,
                           "kl_anchor": kl, "grad_norm": gn}

    return step_fn


def encode_batch(buf_batch, num_pieces):
    grid = buf_batch["grid"]
    bit_idx = jnp.arange(64, dtype=jnp.uint64)
    grid_bits = ((grid[:, None] >> bit_idx[None, :]) & jnp.uint64(1)).astype(jnp.float32)
    safe_tray = jnp.maximum(buf_batch["tray"], 0)
    tray_oh = jax.nn.one_hot(safe_tray, num_pieces, dtype=jnp.float32)
    valid_slot = (buf_batch["tray"] >= 0)[:, :, None].astype(jnp.float32)
    tray_oh = tray_oh * valid_slot
    tray_flat = tray_oh.reshape(grid.shape[0], -1)
    combo = (jnp.minimum(buf_batch["combo"], 10).astype(jnp.float32) / 10.0)[:, None]
    streak = (buf_batch["streak"].astype(jnp.float32) / 3.0)[:, None]
    obs = jnp.concatenate([grid_bits, tray_flat, combo, streak], axis=-1)
    am = jnp.ones((grid.shape[0], 192), dtype=jnp.bool_)
    return obs, am



def main(cfg: AZConfig):
    print(f"[az] config: {cfg.to_dict()}")
    run_dir = make_run_dir(cfg)
    log_path = run_dir / "log.jsonl"
    eval_log_path = run_dir / "eval_log.jsonl"
    cfg_path = run_dir / "config.json"
    cfg_dict = cfg.to_dict()
    cfg_dict["arch_type"] = "az"
    cfg_path.write_text(json.dumps(cfg_dict, indent=2))

    if cfg.hardstart_pool_path and cfg.hardstart_prob > 0.0:
        from block_blast_jax.alphazero.hardstart_env import HardStartBatchedEnv
        env = HardStartBatchedEnv(num_envs=cfg.B, seed=cfg.seed, auto_reset=True,
                                  pool_path=cfg.hardstart_pool_path,
                                  hardstart_prob=cfg.hardstart_prob,
                                  init_from_pool=cfg.hardstart_init_from_pool)
    else:
        env = BatchedBlockBlastEnv(num_envs=cfg.B, seed=cfg.seed, auto_reset=True)
    P = env.num_pieces
    F = feature_dim(P)
    net = ActorCritic(hidden_size=cfg.H, num_actions=cfg.num_actions, use_bf16=cfg.use_bf16)

    init_key = jax.random.PRNGKey(cfg.seed)
    if cfg.warmstart:
        print(f"[az] warmstart from {cfg.warmstart}")
        params, _src_cfg = load_warmstart(cfg.warmstart)
        if cfg.reset_value_head:
            params = reset_value_head(params)
            print("[az] value head reset")
        else:
            print("[az] value head PRESERVED (no reset)")
    else:
        dummy_obs = jnp.zeros((cfg.B, F), dtype=jnp.float32)
        dummy_mask = jnp.ones((cfg.B, cfg.num_actions), dtype=jnp.bool_)
        params = net.init(init_key, dummy_obs, dummy_mask)
    ref_params = None
    if cfg.kl_anchor_coef > 0.0:
        ref_path = cfg.kl_anchor_ref or cfg.warmstart
        if ref_path is None:
            raise ValueError("kl_anchor_coef>0 requires kl_anchor_ref or warmstart")
        ref_params, _ = load_warmstart(ref_path)
        print(f"[az] KL anchor: λ={cfg.kl_anchor_coef} ref={ref_path}")
    n_params = sum(np.prod(p.shape) for p in jax.tree_util.tree_leaves(params))
    print(f"[az] params: {n_params:,}")

    tx = make_optimizer(cfg, grad_clip=cfg.grad_clip)
    state = ts.TrainState.create(apply_fn=net.apply, params=params, tx=tx)
    current_grad_clip = cfg.grad_clip

    env_consts = dict(
        pmasks_flat=env._pmasks_flat, valid_anchor_flat=env._valid_anchor_flat,
        pieces_ncells=env._pieces_ncells, row_masks=env._row_masks,
        col_masks=env._col_masks, line_bonus_table=env._line_bonus_table,
        cascade_table=env._cascade_table,
    )
    run_mcts = make_run_mcts(net.apply, env_consts, P, max_steps=env.max_steps,
                             discount=cfg.discount, combo_coef=cfg.reward_combo_coef)
    run_mcts_jit = jax.jit(run_mcts, static_argnames=(
        "num_simulations", "max_depth", "dirichlet_fraction", "dirichlet_alpha",
        "pb_c_init", "pb_c_base", "use_dirichlet"))

    buf = init_buffer(cfg.buffer_capacity)
    add_jit = jax.jit(buf_add)
    sample_jit = jax.jit(buf_sample, static_argnums=2)

    train_step_fn = make_train_step(net, cfg, P)

    eval_env = BatchedBlockBlastEnv(num_envs=cfg.eval_B, seed=cfg.seed + 10000, auto_reset=False, max_steps=cfg.eval_max_steps)
    eval_consts = dict(
        pmasks_flat=eval_env._pmasks_flat, valid_anchor_flat=eval_env._valid_anchor_flat,
        pieces_ncells=eval_env._pieces_ncells, row_masks=eval_env._row_masks,
        col_masks=eval_env._col_masks, line_bonus_table=eval_env._line_bonus_table,
        cascade_table=eval_env._cascade_table,
    )
    run_mcts_eval = make_run_mcts(net.apply, eval_consts, P, max_steps=eval_env.max_steps,
                                  discount=cfg.discount, combo_coef=cfg.reward_combo_coef)
    run_mcts_eval_jit = jax.jit(run_mcts_eval, static_argnames=(
        "num_simulations", "max_depth", "dirichlet_fraction", "dirichlet_alpha",
        "pb_c_init", "pb_c_base", "use_dirichlet"))

    tb = SummaryWriter(logdir=str(run_dir / "tb"))
    tb_iters = SummaryWriter(logdir=str(run_dir / "tb_iters"))
    tb.add_text("config", json.dumps(cfg_dict, indent=2), 0)
    log_f = open(log_path, "w", buffering=1)
    eval_log_f = open(eval_log_path, "w", buffering=1)

    key = jax.random.PRNGKey(cfg.seed + 1)
    eval_history = deque(maxlen=cfg.plateau_check_window)
    best_eval_ep_len = 0.0
    best_iter = 0
    early_stop_iter = None
    wall_start = time.perf_counter()

    print(f"[az] {cfg.total_iters} iters × B={cfg.B}, sims={cfg.num_simulations_train}, "
          f"grad_clip warmup: {cfg.grad_clip}→{cfg.grad_clip_floor} after iter {cfg.grad_clip_warmup_iters}")
    for it in range(1, cfg.total_iters + 1):
        target_clip = cfg.grad_clip if it <= cfg.grad_clip_warmup_iters else cfg.grad_clip_floor
        if abs(target_clip - current_grad_clip) > 1e-6:
            tx = make_optimizer(cfg, grad_clip=target_clip)
            state = ts.TrainState.create(apply_fn=net.apply, params=state.params, tx=tx)
            current_grad_clip = target_clip
            print(f"  [grad_clip] switched to {current_grad_clip} at iter {it}")
        tb.add_scalar("train/grad_clip", current_grad_clip, global_step=it)
        t_iter0 = time.perf_counter()
        key, sub = jax.random.split(key)
        sp = self_play_iteration(
            run_mcts_jit, net.apply, state.params, env, sub,
            num_simulations=cfg.num_simulations_train,
            max_depth=cfg.max_depth,
            max_moves=cfg.max_moves_per_iter,
            discount=cfg.discount,
            combo_coef=cfg.reward_combo_coef,
            visit_temperature=cfg.visit_temperature,
        )
        sp_wall = time.perf_counter() - t_iter0

        T, B = sp["grids"].shape
        flat_grid = sp["grids"].reshape(T * B)
        flat_tray = sp["trays"].reshape(T * B, 3)
        flat_combo = sp["combos"].reshape(T * B)
        flat_streak = sp["streaks"].reshape(T * B)
        flat_visit = sp["visits"].reshape(T * B, 192)
        flat_value = sp["values"].reshape(T * B)
        buf = add_jit(buf, flat_grid, flat_tray, flat_combo, flat_streak, flat_visit, flat_value)

        train_wall_t0 = time.perf_counter()
        n_samples = int(buf.size)
        steps_per_epoch = max(n_samples // cfg.minibatch_size, 1)
        total_steps = cfg.epochs_per_iter * steps_per_epoch
        train_metrics = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0,
                         "kl_anchor": 0.0, "grad_norm": 0.0}
        for s in range(total_steps):
            key, sub = jax.random.split(key)
            batch = sample_jit(buf, sub, cfg.minibatch_size)
            obs, am = encode_batch(batch, P)
            state, m = train_step_fn(state, obs, am, batch["visit"], batch["value"], ref_params)
            for k in train_metrics:
                train_metrics[k] += float(m[k])
        for k in train_metrics:
            train_metrics[k] /= max(total_steps, 1)
        train_wall = time.perf_counter() - train_wall_t0
        iter_wall = time.perf_counter() - t_iter0
        wall_total = time.perf_counter() - wall_start

        ep_lens = np.asarray(sp["ep_lens"]); ep_scores = np.asarray(sp["ep_scores"])
        sims_per_sec = sp["n_simulations_total"] / max(sp_wall, 1e-6)
        eps_per_sec = sp["n_episodes_done"] / max(iter_wall, 1e-6)
        line = {
            "iter": it,
            "wall_seconds": round(wall_total, 1),
            "iter_seconds": round(iter_wall, 1),
            "selfplay_seconds": round(sp_wall, 1),
            "train_seconds": round(train_wall, 1),
            "buffer_size": int(buf.size),
            "buffer_fill_ratio": float(buf.size) / cfg.buffer_capacity,
            "n_episodes_done": int(sp["n_episodes_done"]),
            "n_transitions": int(sp["n_transitions"]),
            "value_target_mean": float(sp["values"].mean()),
            "value_target_p99": float(jnp.percentile(sp["values"], 99)),
            "selfplay_ep_length_mean": float(ep_lens.mean()),
            "selfplay_ep_length_p50": float(np.percentile(ep_lens, 50)),
            "selfplay_ep_length_p99": float(np.percentile(ep_lens, 99)),
            "selfplay_ep_score_mean": float(ep_scores.mean()),
            "mcts_visit_entropy_mean": float(sp["visit_entropy_mean"]),
            "mcts_visits_max_mean": float(sp["visits_max_mean"]),
            "mcts_root_value_mean": float(sp["root_value_mean"]),
            "throughput_sims_per_sec": float(sims_per_sec),
            "throughput_eps_per_sec": float(eps_per_sec),
            **{k: float(v) for k, v in train_metrics.items()},
        }
        if hasattr(env, "_last_hardstart_frac"):
            line["hardstart_fraction"] = float(env._last_hardstart_frac)
            tb.add_scalar("selfplay/hardstart_fraction", line["hardstart_fraction"], global_step=it)
        log_f.write(json.dumps(line) + "\n")

        tb.add_scalar("loss/total", line["loss"], global_step=it)
        tb.add_scalar("loss/policy", line["policy_loss"], global_step=it)
        tb.add_scalar("loss/value", line["value_loss"], global_step=it)
        tb.add_scalar("loss/grad_norm", line["grad_norm"], global_step=it)
        tb.add_scalar("selfplay/ep_length_mean", line["selfplay_ep_length_mean"], global_step=it)
        tb.add_scalar("selfplay/ep_length_p50",  line["selfplay_ep_length_p50"],  global_step=it)
        tb.add_scalar("selfplay/ep_length_p99",  line["selfplay_ep_length_p99"],  global_step=it)
        tb.add_scalar("selfplay/ep_score_mean",  line["selfplay_ep_score_mean"],  global_step=it)
        tb.add_scalar("selfplay/episodes_completed", line["n_episodes_done"], global_step=it)
        tb.add_scalar("selfplay/transitions_collected", line["n_transitions"], global_step=it)
        tb.add_scalar("mcts/root_value_mean", line["mcts_root_value_mean"], global_step=it)
        tb.add_scalar("mcts/visit_entropy_mean", line["mcts_visit_entropy_mean"], global_step=it)
        tb.add_scalar("mcts/visits_max", line["mcts_visits_max_mean"], global_step=it)
        tb.add_scalar("agreement/policy_xe", line["policy_loss"], global_step=it)
        if cfg.kl_anchor_coef > 0.0:
            tb.add_scalar("loss/kl_anchor", line["kl_anchor"], global_step=it)
        tb.add_scalar("throughput/iter_seconds", line["iter_seconds"], global_step=it)
        tb.add_scalar("throughput/sims_per_sec", line["throughput_sims_per_sec"], global_step=it)
        tb.add_scalar("throughput/eps_per_sec",  line["throughput_eps_per_sec"], global_step=it)
        tb.add_scalar("buffer/size", line["buffer_size"], global_step=it)
        tb.add_scalar("buffer/fill_ratio", line["buffer_fill_ratio"], global_step=it)
        tb.add_scalar("train/value_target_mean", line["value_target_mean"], global_step=it)
        tb.add_scalar("train/value_target_p99", line["value_target_p99"], global_step=it)

        if it % 5 == 0 or it == 1:
            print(f"  [iter {it:4d}] wall={wall_total/60:.1f}m iter={iter_wall:.0f}s "
                  f"loss={line['loss']:.4f} pl={line['policy_loss']:.3f} vl={line['value_loss']:.3f} "
                  f"v_target_mean={line['value_target_mean']:.2f} eps_done={line['n_episodes_done']}")

        if it % cfg.eval_every == 0:
            t_eval0 = time.perf_counter()
            eval_env.reset(seed=cfg.seed + 10000 + it)
            ep_lens, ep_scores = [], []
            for step_i in range(eval_env.max_steps):
                key, sub = jax.random.split(key)
                out = run_mcts_eval_jit(
                    state.params, eval_env.state, sub,
                    num_simulations=cfg.num_simulations_eval, max_depth=cfg.max_depth,
                    dirichlet_fraction=0.0, dirichlet_alpha=0.3,
                    pb_c_init=cfg.pb_c_init, pb_c_base=cfg.pb_c_base,
                    use_dirichlet=False,
                )
                actions = out.action
                prev_done = np.asarray(eval_env.state["done"])
                prev_ep = np.asarray(eval_env.state["ep_len"])
                prev_sc = np.asarray(eval_env.state["score"])
                new_state, _, term, trunc, info = eval_env.step(actions)
                done_now = np.asarray(term | trunc)
                freshly_done = done_now & (~prev_done)
                if freshly_done.any():
                    valid = np.asarray(info["cells_placed"]).astype(bool).astype(int)
                    full_len = prev_ep + valid
                    full_sc = prev_sc + np.asarray(info["score_delta"])
                    for i in np.where(freshly_done)[0]:
                        ep_lens.append(int(full_len[i]))
                        ep_scores.append(int(full_sc[i]))
                if bool(np.asarray(new_state["done"]).all()):
                    break
            eval_wall = time.perf_counter() - t_eval0
            ep_lens = np.array(ep_lens) if ep_lens else np.array([0])
            ep_scores = np.array(ep_scores) if ep_scores else np.array([0])
            ev = {
                "iter": it,
                "frames": it * cfg.B * cfg.max_moves_per_iter,
                "ep_length_mean": float(ep_lens.mean()),
                "ep_length_p50": float(np.percentile(ep_lens, 50)),
                "ep_length_p99": float(np.percentile(ep_lens, 99)),
                "ep_score_mean": float(ep_scores.mean()),
                "ep_score_p99": float(np.percentile(ep_scores, 99)),
                "n_episodes": int(len(ep_lens)),
                "eval_seconds": round(eval_wall, 1),
            }
            total_moves = int(ep_lens.sum()) if hasattr(ep_lens, "sum") else 0
            move_latency_ms = (eval_wall / max(total_moves, 1)) * 1000.0
            ev["move_latency_ms"] = move_latency_ms
            eval_log_f.write(json.dumps(ev) + "\n")
            tb.add_scalar("eval/ep_length_mean", ev["ep_length_mean"], global_step=it)
            tb.add_scalar("eval/ep_length_p99",  ev["ep_length_p99"],  global_step=it)
            tb.add_scalar("eval/ep_score_mean",  ev["ep_score_mean"],  global_step=it)
            tb.add_scalar("eval/ep_score_p99",   ev["ep_score_p99"],   global_step=it)
            tb.add_scalar("eval/move_latency_ms", move_latency_ms, global_step=it)
            print(f"  [eval iter={it}] len_mean={ev['ep_length_mean']:.2f} "
                  f"score_mean={ev['ep_score_mean']:.1f} n={ev['n_episodes']} "
                  f"({eval_wall:.0f}s, move_lat={move_latency_ms:.0f}ms)")
            eval_history.append((it, ev["ep_length_mean"]))
            if ev["ep_length_mean"] > best_eval_ep_len:
                best_eval_ep_len = ev["ep_length_mean"]
                best_iter = it
                ckpt_blob = {
                    "params": state.params, "config": cfg_dict,
                    "frames": it * cfg.B * cfg.max_moves_per_iter, "update": it,
                }
                atomic_write_msgpack(run_dir / "checkpoint_best.msgpack", ckpt_blob)
            if (it >= cfg.plateau_min_iters and
                    len(eval_history) == cfg.plateau_check_window and
                    all(v > 1.0 for _, v in eval_history)):
                first = eval_history[0][1]
                last = eval_history[-1][1]
                rel = (last - first) / max(first, 1e-9)
                if rel < cfg.plateau_min_improvement:
                    print(f"[az] PLATEAU @ iter {it}: rel_imp={rel*100:.3f}% — early stop")
                    early_stop_iter = it
                    break

        if it % cfg.checkpoint_every == 0:
            ckpt_blob = {"params": state.params, "config": cfg_dict,
                         "frames": it * cfg.B * cfg.max_moves_per_iter, "update": it}
            atomic_write_msgpack(run_dir / "checkpoints" / f"ckpt_iter_{it}.msgpack", ckpt_blob)

    log_f.close()
    eval_log_f.close()
    final_blob = {"params": state.params, "config": cfg_dict,
                  "frames": (early_stop_iter or cfg.total_iters) * cfg.B * cfg.max_moves_per_iter,
                  "update": (early_stop_iter or cfg.total_iters)}
    atomic_write_msgpack(run_dir / "checkpoint_final.msgpack", final_blob)
    try:
        tb.add_hparams(
            {"H": cfg.H, "B": cfg.B,
             "num_simulations": cfg.num_simulations_train,
             "lr": cfg.lr, "discount": cfg.discount,
             "warmstart": str(cfg.warmstart) if cfg.warmstart else "none"},
            {"hparam/best_eval_ep_length": best_eval_ep_len,
             "hparam/iters_completed": float(early_stop_iter or cfg.total_iters),
             "hparam/wall_seconds": float(time.perf_counter() - wall_start)},
        )
    except Exception:
        pass
    tb.close()
    tb_iters.close()
    total_wall = time.perf_counter() - wall_start
    summary = run_dir / "training_summary.md"
    with summary.open("w") as f:
        f.write(f"# {cfg.run_name} — AZ training summary\n\n")
        f.write(f"Status: {'EARLY_STOPPED @ iter ' + str(early_stop_iter) if early_stop_iter else 'COMPLETED'}\n\n")
        f.write(f"Wall: {total_wall/3600:.2f} h ({total_wall/60:.1f} min)\n\n")
        f.write(f"Best eval ep_length: {best_eval_ep_len:.2f} @ iter {best_iter}\n")
        f.write(f"Warmstart: {cfg.warmstart}\n")
    print(f"[az] DONE. wall={total_wall/3600:.2f}h best_ep_len={best_eval_ep_len:.2f}@iter{best_iter}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", default="az_v1.0")
    ap.add_argument("--warmstart", default=None)
    ap.add_argument("--total_iters", type=int, default=400)
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--num_simulations", type=int, default=200)
    ap.add_argument("--num_simulations_eval", type=int, default=500)
    ap.add_argument("--max_moves_per_iter", type=int, default=100)
    ap.add_argument("--eval_every", type=int, default=10)
    ap.add_argument("--checkpoint_every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=5.0)
    ap.add_argument("--grad_clip_floor", type=float, default=0.5)
    ap.add_argument("--grad_clip_warmup_iters", type=int, default=50)
    ap.add_argument("--eval_B", type=int, default=32)
    ap.add_argument("--eval_max_steps", type=int, default=200)
    ap.add_argument("--vf_coef", type=float, default=0.5)
    ap.add_argument("--plateau_min_iters", type=int, default=50)
    ap.add_argument("--reset_value_head", action="store_true",
                    help="legacy: reset value head after warmstart (v3.0 default OFF)")
    ap.add_argument("--no_reset_value_head", dest="reset_value_head",
                    action="store_false")
    ap.set_defaults(reset_value_head=False)
    ap.add_argument("--reward_combo_coef", type=float, default=0.0)
    ap.add_argument("--visit_temperature", type=float, default=1.0)
    ap.add_argument("--kl_anchor_coef", type=float, default=0.0)
    ap.add_argument("--kl_anchor_ref", default=None)
    ap.add_argument("--hardstart_pool_path", default=None)
    ap.add_argument("--hardstart_prob", type=float, default=0.0)
    ap.add_argument("--hardstart_init_from_pool", action="store_true",
                    help="iter 1'den itibaren initial state'leri pool'dan örnekle")
    ap.add_argument("--lr_value_head_mult", type=float, default=3.0)
    ap.add_argument("--use_bf16", action="store_true",
                    help="bf16 mixed precision (trunk+policy bf16, value fp32)")
    ap.add_argument("--max_depth", type=int, default=64)
    ap.add_argument("--discount", type=float, default=0.997)
    args = ap.parse_args()
    cfg = AZConfig(
        run_name=args.run_name, warmstart=args.warmstart,
        total_iters=args.total_iters, B=args.B,
        num_simulations_train=args.num_simulations,
        num_simulations_eval=args.num_simulations_eval,
        max_moves_per_iter=args.max_moves_per_iter,
        eval_every=args.eval_every, checkpoint_every=args.checkpoint_every,
        seed=args.seed, lr=args.lr,
        lr_value_head_mult=args.lr_value_head_mult,
        grad_clip=args.grad_clip, grad_clip_floor=args.grad_clip_floor,
        grad_clip_warmup_iters=args.grad_clip_warmup_iters,
        eval_B=args.eval_B, eval_max_steps=args.eval_max_steps,
        vf_coef=args.vf_coef, plateau_min_iters=args.plateau_min_iters,
        reset_value_head=args.reset_value_head,
        reward_combo_coef=args.reward_combo_coef,
        visit_temperature=args.visit_temperature,
        kl_anchor_coef=args.kl_anchor_coef,
        kl_anchor_ref=args.kl_anchor_ref,
        hardstart_pool_path=args.hardstart_pool_path,
        hardstart_prob=args.hardstart_prob,
        hardstart_init_from_pool=args.hardstart_init_from_pool,
        use_bf16=args.use_bf16,
        max_depth=args.max_depth,
        discount=args.discount,
    )
    main(cfg)
