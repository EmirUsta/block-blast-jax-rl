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
from tensorboardX import SummaryWriter

import block_blast_jax
from block_blast_jax import BatchedBlockBlastEnv
from block_blast_jax.rl.config import PPOConfig
from block_blast_jax.rl.encode import encode_obs, feature_dim
from block_blast_jax.rl.eval import make_eval_fn
from block_blast_jax.rl.network import ActorCritic, CNNActorCritic
from block_blast_jax.rl.ppo import make_train_state, train_step



def make_run_dir(cfg: PPOConfig) -> Path:
    if cfg.run_name:
        name = cfg.run_name
    else:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        name = f"{ts}_H{cfg.hidden_size}_dp{int(cfg.death_penalty)}"
    p = Path(cfg.checkpoint_dir) / name
    p.mkdir(parents=True, exist_ok=True)
    (p / "checkpoints").mkdir(exist_ok=True)
    return p


def atomic_write_msgpack(path: Path, blob: dict) -> None:
    raw = msgpack_serialize(jax.device_get(blob))
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(raw)
    os.replace(tmp, path)


def save_ckpt(path: Path, params, cfg: PPOConfig, frames: int, update: int,
              ep_length_mean: float, ep_score_mean: float, wall_seconds: float):
    blob = {
        "params": params,
        "config": cfg.to_dict(),
        "frames": int(frames),
        "update": int(update),
    }
    atomic_write_msgpack(path, blob)
    meta = {
        "frames": int(frames),
        "update": int(update),
        "ep_length_mean": float(ep_length_mean),
        "ep_score_mean": float(ep_score_mean),
        "wall_seconds": float(wall_seconds),
    }
    meta_path = path.with_suffix(".json")
    tmp_meta = meta_path.with_suffix(".json.tmp")
    tmp_meta.write_text(json.dumps(meta, indent=2))
    os.replace(tmp_meta, meta_path)


def find_latest_ckpt(ckpt_dir: Path) -> tuple[int, Path] | None:
    cands = []
    for f in ckpt_dir.glob("ckpt_*.msgpack"):
        s = f.stem.removeprefix("ckpt_")
        if s.isdigit():
            cands.append((int(s), f))
    if not cands:
        return None
    return max(cands, key=lambda x: x[0])


def load_ckpt(path: Path) -> dict:
    return msgpack_restore(path.read_bytes())


def make_lr_schedule(cfg: PPOConfig) -> float | optax.Schedule:
    if cfg.lr_schedule == "linear_to_zero":
        n_updates = cfg.updates
        steps = n_updates * cfg.ppo_epochs * cfg.num_minibatches
        return optax.linear_schedule(cfg.lr, 0.0, max(steps, 1))
    return cfg.lr



def main(cfg: PPOConfig):
    print(f"[train] config: {cfg.to_dict()}")
    run_dir = make_run_dir(cfg)
    log_path = run_dir / "log.jsonl"
    eval_log_path = run_dir / "eval_log.jsonl"
    cfg_path = run_dir / "config.json"
    cfg_path.write_text(json.dumps(cfg.to_dict(), indent=2))

    env = BatchedBlockBlastEnv(num_envs=cfg.num_envs, seed=cfg.seed, auto_reset=True)
    P = env.num_pieces
    F = feature_dim(P)
    print(f"[train] num_envs={cfg.num_envs}, feature_dim={F}, num_pieces={P}")

    eval_env = BatchedBlockBlastEnv(
        num_envs=cfg.eval_num_envs, seed=cfg.seed + 1000,
        auto_reset=False, max_steps=cfg.eval_max_steps,
    )

    if cfg.arch_type == "cnn":
        net = CNNActorCritic(n_filters=cfg.n_filters, n_conv=cfg.n_conv,
                              H=cfg.hidden_size, num_actions=192)
        init_key = jax.random.PRNGKey(cfg.seed)
        dummy_b = jnp.zeros((cfg.num_envs, 8, 8, 1), dtype=jnp.float32)
        dummy_t = jnp.zeros((cfg.num_envs, 3 * P), dtype=jnp.float32)
        dummy_c = jnp.zeros((cfg.num_envs, 1), dtype=jnp.float32)
        dummy_s = jnp.zeros((cfg.num_envs, 1), dtype=jnp.float32)
        params = net.init(init_key, dummy_b, dummy_t, dummy_c, dummy_s)
    else:
        net = ActorCritic(hidden_size=cfg.hidden_size, num_actions=192)
        init_key = jax.random.PRNGKey(cfg.seed)
        dummy_obs = jnp.zeros((cfg.num_envs, F), dtype=jnp.float32)
        dummy_mask = jnp.ones((cfg.num_envs, 192), dtype=jnp.bool_)
        params = net.init(init_key, dummy_obs, dummy_mask)
    n_params = sum(np.prod(p.shape) for p in jax.tree_util.tree_leaves(params))
    print(f"[train] arch={cfg.arch_type} params count: {n_params:,}")

    ckpt_dir = run_dir / "checkpoints"
    start_update = 0
    start_frames = 0
    if cfg.resume:
        latest = find_latest_ckpt(ckpt_dir)
        if latest is not None:
            f_idx, ckpt_path = latest
            print(f"[train] RESUME from {ckpt_path.name} (frame={f_idx})")
            blob = load_ckpt(ckpt_path)
            params = blob["params"]
            start_update = int(blob["update"])
            start_frames = int(blob["frames"])

    lr = make_lr_schedule(cfg)
    state_train = make_train_state(net, params, lr, cfg.max_grad_norm)
    key = jax.random.PRNGKey(cfg.seed + 1)

    pmf = env._pmasks_flat; vaf = env._valid_anchor_flat
    pnc = env._pieces_ncells; rm = env._row_masks; cm = env._col_masks
    lbt = env._line_bonus_table; ct = env._cascade_table

    def gamma_at_frame(frames_seen: int) -> float:
        if cfg.gamma_schedule != "linear":
            return float(cfg.gamma)
        if cfg.gamma_warmup_frames <= 0:
            return float(cfg.gamma_end)
        progress = min(frames_seen / cfg.gamma_warmup_frames, 1.0)
        return float(cfg.gamma_start + (cfg.gamma_end - cfg.gamma_start) * progress)

    def do_update(ts_state, env_st, k, current_gamma: float):
        return train_step(
            ts_state, env_st, k,
            pmf, vaf, pnc, rm, cm, lbt, ct,
            auto_reset=env.auto_reset, max_steps=env.max_steps,
            num_pieces=env.num_pieces, num_actions=192,
            rollout_len=cfg.rollout_len, ppo_epochs=cfg.ppo_epochs,
            num_minibatches=cfg.num_minibatches,
            death_penalty=cfg.death_penalty,
            gamma=current_gamma, gae_lambda=cfg.gae_lambda,
            clip_eps=cfg.clip_eps, ent_coef=cfg.ent_coef, vf_coef=cfg.vf_coef,
            hole_penalty=cfg.hole_penalty,
            empty_board_reward=cfg.empty_board_reward,
            roughness_penalty=cfg.roughness_penalty,
            combo_preserve_bonus=cfg.combo_preserve_bonus,
            reward_mode=cfg.reward_mode,
            arch_type=cfg.arch_type,
        )

    eval_fn = make_eval_fn(net, eval_env, cfg.eval_num_envs, cfg.eval_max_steps,
                            arch_type=cfg.arch_type)

    print("[train] compiling train_step...")
    t0 = time.perf_counter()
    warmup_gamma = gamma_at_frame(start_frames)
    state_train, env.state, key, m = do_update(state_train, env.state, key, warmup_gamma)
    jax.block_until_ready(m["policy_loss"])
    print(f"[train] compile train_step: {time.perf_counter()-t0:.1f}s")

    per_update = cfg.num_envs * cfg.rollout_len
    n_updates = cfg.updates
    print(f"[train] {n_updates} updates × {per_update} frames/update")

    tb_update = SummaryWriter(logdir=str(run_dir / "tb"))
    tb_frames = SummaryWriter(logdir=str(run_dir / "tb_frames"))
    tb_update.add_text("config", json.dumps(cfg.to_dict(), indent=2), 0)

    SCALAR_TAGS = [
        ("episode/length_mean",         "ep_length_mean"),
        ("episode/length_p50",          "ep_length_p50"),
        ("episode/length_p99",          "ep_length_p99"),
        ("episode/score_mean",          "ep_score_mean"),
        ("episode/score_p50",           "ep_score_p50"),
        ("episode/score_p99",           "ep_score_p99"),
        ("episode/shaped_reward_mean",  "ep_shaped_reward_mean"),
        ("episode/pure_reward_mean",    "ep_pure_reward_mean"),
        ("throughput/step_per_sec",     "step_per_sec"),
        ("throughput/ep_per_sec",       "ep_per_sec"),
        ("loss/policy",                 "policy_loss"),
        ("loss/value",                  "value_loss"),
        ("loss/entropy",                "entropy"),
        ("loss/approx_kl",              "approx_kl"),
        ("loss/clip_frac",              "clip_frac"),
        ("shaping/death_pen_per_ep",        "shaping/death_pen_per_ep"),
        ("shaping/hole_pen_per_ep",         "shaping/hole_pen_per_ep"),
        ("shaping/empty_board_rew_per_ep",  "shaping/empty_board_rew_per_ep"),
        ("shaping/roughness_pen_per_ep",    "shaping/roughness_pen_per_ep"),
        ("shaping/combo_preserve_per_ep",   "shaping/combo_preserve_per_ep"),
    ]

    log_f = open(log_path, "a", buffering=1)
    eval_log_f = open(eval_log_path, "a", buffering=1)
    wall_start = time.perf_counter()

    next_eval_frames = (start_frames // cfg.eval_every_frames + 1) * cfg.eval_every_frames
    next_ckpt_frames = (start_frames // cfg.checkpoint_every_frames + 1) * cfg.checkpoint_every_frames
    eval_history: deque[tuple[int, float]] = deque(maxlen=cfg.plateau_window_evals)
    best_eval_ep_len = 0.0
    best_eval_frames = 0
    best_eval_score = 0.0
    early_stopped = False
    early_stop_frame = None
    last_metrics = None
    last_eval_metrics = None

    if start_update > 0:
        u_iter_start = start_update + 1
        frames = start_frames
    else:
        u_iter_start = 1
        frames = per_update

    try:
        for u in range(u_iter_start, n_updates + 1):
            t_u0 = time.perf_counter()
            current_gamma = gamma_at_frame((u - 1) * per_update)
            state_train, env.state, key, metrics = do_update(state_train, env.state, key, current_gamma)
            jax.block_until_ready(metrics["policy_loss"])
            t_u1 = time.perf_counter()
            wall = t_u1 - wall_start
            frames = u * per_update
            m = jax.device_get(metrics)
            ep_per_sec = float(m["n_episodes"]) / (t_u1 - t_u0) if (t_u1 - t_u0) > 0 else 0.0
            step_per_sec = per_update / (t_u1 - t_u0) if (t_u1 - t_u0) > 0 else 0.0
            line = {
                "timestamp": datetime.now().isoformat(),
                "update": u, "frames": int(frames),
                "wall_seconds": round(wall, 2),
                "n_episodes": int(m["n_episodes"]),
                "ep_per_sec": round(ep_per_sec, 1),
                "step_per_sec": round(step_per_sec, 1),
                "config": cfg.to_dict(),
            }
            for k, v in m.items():
                if k == "n_episodes":
                    continue
                try:
                    line[k] = float(v)
                except (TypeError, ValueError):
                    pass
            log_f.write(json.dumps(line) + "\n")
            for tag, mk in SCALAR_TAGS:
                v = float(line[mk])
                tb_update.add_scalar(tag, v, global_step=u)
                tb_frames.add_scalar(tag, v, global_step=int(frames))
            tb_update.add_scalar("train/current_gamma", float(current_gamma), global_step=u)
            tb_frames.add_scalar("train/current_gamma", float(current_gamma), global_step=int(frames))
            line["current_gamma"] = float(current_gamma)
            last_metrics = line

            if frames >= next_eval_frames:
                eval_env.reset(seed=cfg.seed + 1000 + u)
                ev = eval_fn(state_train.params, eval_env.state)
                ev = jax.device_get(ev)
                ep_lens = np.asarray(ev["ep_len"], dtype=np.float64)
                scores = np.asarray(ev["score"], dtype=np.float64)
                eval_metrics = {
                    "frames": int(frames),
                    "update": u,
                    "ep_length_mean": float(ep_lens.mean()),
                    "ep_length_p50": float(np.percentile(ep_lens, 50)),
                    "ep_length_p99": float(np.percentile(ep_lens, 99)),
                    "ep_score_mean": float(scores.mean()),
                    "ep_score_p99": float(np.percentile(scores, 99)),
                    "n_envs_done": int(np.asarray(ev["done"]).sum()),
                }
                eval_log_f.write(json.dumps(eval_metrics) + "\n")
                for k, v in [
                    ("eval/length_mean", eval_metrics["ep_length_mean"]),
                    ("eval/length_p50",  eval_metrics["ep_length_p50"]),
                    ("eval/length_p99",  eval_metrics["ep_length_p99"]),
                    ("eval/score_mean",  eval_metrics["ep_score_mean"]),
                    ("eval/score_p99",   eval_metrics["ep_score_p99"]),
                ]:
                    tb_update.add_scalar(k, v, global_step=u)
                    tb_frames.add_scalar(k, v, global_step=int(frames))
                last_eval_metrics = eval_metrics
                eval_history.append((int(frames), eval_metrics["ep_length_mean"]))

                if eval_metrics["ep_length_mean"] > best_eval_ep_len:
                    best_eval_ep_len = eval_metrics["ep_length_mean"]
                    best_eval_score = eval_metrics["ep_score_mean"]
                    best_eval_frames = int(frames)
                    save_ckpt(
                        run_dir / "checkpoint_best.msgpack",
                        state_train.params, cfg, frames, u,
                        eval_metrics["ep_length_mean"], eval_metrics["ep_score_mean"],
                        wall,
                    )

                if (frames >= cfg.plateau_min_frames and
                        len(eval_history) == cfg.plateau_window_evals):
                    first_v = eval_history[0][1]
                    last_v = eval_history[-1][1]
                    rel_imp = (last_v - first_v) / max(first_v, 1e-9)
                    if rel_imp < cfg.plateau_min_improvement:
                        print(f"[train] PLATEAU: improvement {rel_imp*100:.3f}% "
                              f"over last {cfg.plateau_window_evals} evals — early stop")
                        (run_dir / "early_stopped.txt").write_text(
                            f"frames={frames}\nrel_improvement={rel_imp:.6f}\n"
                            f"window={cfg.plateau_window_evals}\nmin_required={cfg.plateau_min_improvement}\n"
                        )
                        early_stopped = True
                        early_stop_frame = int(frames)

                print(f"  [eval u={u} f={frames/1e6:.0f}M] "
                      f"len_mean={eval_metrics['ep_length_mean']:.2f} "
                      f"score_mean={eval_metrics['ep_score_mean']:.1f} "
                      f"best_len={best_eval_ep_len:.2f}@{best_eval_frames/1e6:.0f}M")

                next_eval_frames += cfg.eval_every_frames

            if frames >= next_ckpt_frames:
                save_ckpt(
                    ckpt_dir / f"ckpt_{int(frames)}.msgpack",
                    state_train.params, cfg, frames, u,
                    line["ep_length_mean"], line["ep_score_mean"], wall,
                )
                next_ckpt_frames += cfg.checkpoint_every_frames

            if u % 50 == 0 or u == n_updates:
                shaped = line.get("ep_shaped_reward_mean", line.get("ep_score_mean", 0))
                print(f"  [u={u:5d}] frames={frames/1e6:.1f}M "
                      f"ep_len_mean={line['ep_length_mean']:.2f} "
                      f"ep_score={line['ep_score_mean']:.1f} "
                      f"shaped={shaped:.1f} "
                      f"fps={line['step_per_sec']:.0f} "
                      f"kl={line['approx_kl']:.4f} ent={line['entropy']:.2f}")

            if early_stopped:
                break

    finally:
        log_f.close()
        eval_log_f.close()
        final_ckpt = run_dir / "checkpoint_final.msgpack"
        save_ckpt(
            final_ckpt, state_train.params, cfg, frames,
            (early_stop_frame // per_update) if early_stop_frame else n_updates,
            (last_metrics or {}).get("ep_length_mean", 0.0),
            (last_metrics or {}).get("ep_score_mean", 0.0),
            time.perf_counter() - wall_start,
        )
        try:
            hparams = {k: (float(v) if isinstance(v, float) else int(v))
                       for k, v in {
                           "hidden_size": cfg.hidden_size,
                           "death_penalty": cfg.death_penalty,
                           "num_envs": cfg.num_envs,
                           "rollout_len": cfg.rollout_len,
                           "lr": cfg.lr,
                           "ppo_epochs": cfg.ppo_epochs,
                           "num_minibatches": cfg.num_minibatches,
                           "total_frames": cfg.total_frames,
                       }.items()}
            fm = {
                "hparam/ep_length_mean": float((last_metrics or {}).get("ep_length_mean", 0)),
                "hparam/ep_score_mean":  float((last_metrics or {}).get("ep_score_mean", 0)),
                "hparam/best_eval_ep_length": float(best_eval_ep_len),
                "hparam/step_per_sec":   float((last_metrics or {}).get("step_per_sec", 0)),
            }
            tb_update.add_hparams(hparams, fm)
        except Exception:
            pass
        tb_update.close()
        tb_frames.close()
        write_training_summary(
            run_dir, cfg, last_metrics, last_eval_metrics,
            best_eval_ep_len, best_eval_score, best_eval_frames,
            early_stop_frame, time.perf_counter() - wall_start,
        )
        print(f"[train] DONE. logs: {log_path}, ckpt_final: {final_ckpt}")
        if early_stop_frame:
            print(f"[train] EARLY STOPPED at frame {early_stop_frame}")


def write_training_summary(run_dir, cfg, last, last_eval, best_len, best_score, best_frames,
                            early_stop_frame, wall_s):
    p = run_dir / "training_summary.md"
    md = []
    md.append(f"# {cfg.run_name or run_dir.name} — training summary\n")
    md.append(f"Config: H={cfg.hidden_size}, dp={cfg.death_penalty}, "
              f"lr={cfg.lr}→{0.0 if cfg.lr_schedule=='linear_to_zero' else cfg.lr} "
              f"({cfg.lr_schedule}), {cfg.total_frames/1e6:.0f}M frames, seed={cfg.seed}\n")
    md.append("## OUTCOME\n")
    md.append(f"- Status: {'EARLY_STOPPED' if early_stop_frame else 'COMPLETED'}"
              f"{' @ ' + f'{early_stop_frame/1e6:.0f}M frames' if early_stop_frame else ''}")
    md.append(f"- Wall time: {wall_s/60:.2f} min ({wall_s:.0f} s)")
    md.append(f"- Mean train FPS: {(last or {}).get('step_per_sec', 0):.0f} step/s\n")
    md.append("## LEARNING CURVE (training, stochastic policy)\n")
    md.append(f"- Last ep_length_mean: {(last or {}).get('ep_length_mean', 0):.2f}")
    md.append(f"- Last ep_score_mean:  {(last or {}).get('ep_score_mean', 0):.1f}\n")
    md.append("## EVAL (deterministic argmax, 1024 envs)\n")
    md.append(f"- Best ep_length: {best_len:.2f} @ frame {best_frames/1e6:.0f}M")
    md.append(f"- Best ep_score:  {best_score:.1f}")
    if last_eval:
        md.append(f"- Final ep_length: {last_eval['ep_length_mean']:.2f}")
        md.append(f"- Final ep_score:  {last_eval['ep_score_mean']:.1f}")
        diff = best_len - last_eval["ep_length_mean"]
        md.append(f"- Best > Final: {'Yes' if diff > 0.05 else 'No'} (Δ={diff:+.2f})\n")
    has_shaping = (cfg.hole_penalty != 0 or cfg.empty_board_reward != 0
                    or cfg.roughness_penalty != 0 or cfg.combo_preserve_bonus != 0
                    or cfg.death_penalty != 0)
    if has_shaping and last:
        md.append("## SHAPING DECOMPOSITION   (mean per-episode contribution)\n")
        d = last.get("shaping/death_pen_per_ep", 0.0)
        h = last.get("shaping/hole_pen_per_ep", 0.0)
        e = last.get("shaping/empty_board_rew_per_ep", 0.0)
        r = last.get("shaping/roughness_pen_per_ep", 0.0)
        c = last.get("shaping/combo_preserve_per_ep", 0.0)
        pure = last.get("ep_pure_reward_mean", last.get("ep_score_mean", 0.0))
        shaped = last.get("ep_shaped_reward_mean", 0.0)
        sum_shaping = -d - h + e - r + c
        md.append(f"- death_pen_total       : -{d:.3f}")
        md.append(f"- hole_pen_total        : -{h:.3f}")
        md.append(f"- empty_board_rew_total : +{e:.3f}")
        md.append(f"- roughness_pen_total   : -{r:.3f}")
        md.append(f"- combo_preserve_total  : +{c:.3f}")
        md.append(f"- Sum of shaping        : {sum_shaping:+.3f}")
        md.append(f"- Pure move_score       : {pure:.3f}")
        md.append(f"- Shaped reward total   : {shaped:.3f}")
        denom = max(abs(pure) + abs(sum_shaping), 1e-9)
        share = abs(sum_shaping) / denom * 100
        md.append(f"- Shaping |share| of |reward| : {share:.1f}%\n")

    md.append("## STABILITY\n")
    md.append(f"- approx_kl final: {(last or {}).get('approx_kl', 0):.4f}")
    md.append(f"- entropy final:   {(last or {}).get('entropy', 0):.3f}")
    md.append(f"- policy_loss final: {(last or {}).get('policy_loss', 0):.4f}")
    md.append(f"- value_loss  final: {(last or {}).get('value_loss', 0):.1f}\n")
    md.append("## CHECKPOINTS\n")
    md.append("- runs/{name}/checkpoints/ckpt_<frame>.msgpack (her 100M)".format(name=cfg.run_name or run_dir.name))
    md.append("- runs/{name}/checkpoint_best.msgpack".format(name=cfg.run_name or run_dir.name))
    md.append("- runs/{name}/checkpoint_final.msgpack\n".format(name=cfg.run_name or run_dir.name))
    md.append("## REPLAY\n")
    md.append(f"```\npython -m block_blast_jax.ui.play --checkpoint runs/{cfg.run_name or run_dir.name}/checkpoint_best.msgpack --ai_step_ms 200\n```")
    p.write_text("\n".join(md))



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, default=128)
    ap.add_argument("--death_penalty", type=float, default=0.0)
    ap.add_argument("--num_envs", type=int, default=4096)
    ap.add_argument("--rollout_len", type=int, default=128)
    ap.add_argument("--total_frames", type=int, default=50_000_000)
    ap.add_argument("--ppo_epochs", type=int, default=4)
    ap.add_argument("--num_minibatches", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr_schedule", default="constant",
                    choices=["constant", "linear_to_zero"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--checkpoint_every_frames", type=int, default=100_000_000)
    ap.add_argument("--eval_every_frames", type=int, default=50_000_000)
    ap.add_argument("--eval_num_envs", type=int, default=1024)
    ap.add_argument("--eval_max_steps", type=int, default=500)
    ap.add_argument("--no_resume", action="store_true")
    ap.add_argument("--reward_mode", default="score",
                    choices=["score", "survival"])
    ap.add_argument("--ent_coef", type=float, default=0.01)
    ap.add_argument("--gamma_schedule", default="constant",
                    choices=["constant", "linear"])
    ap.add_argument("--gamma_start", type=float, default=0.99)
    ap.add_argument("--gamma_end", type=float, default=0.99)
    ap.add_argument("--gamma_warmup_frames", type=int, default=0)
    ap.add_argument("--plateau_min_frames", type=int, default=500_000_000)
    ap.add_argument("--arch_type", default="mlp", choices=["mlp", "cnn"])
    ap.add_argument("--n_conv", type=int, default=2)
    ap.add_argument("--n_filters", type=int, default=32)
    ap.add_argument("--shaping_config", default=None,
                    help='JSON dict, e.g. \'{"death_penalty":25,"hole_penalty":0.15,'
                         '"empty_board_reward":0.01,"roughness_penalty":0.01,'
                         '"combo_preserve_bonus":0.15}\'')
    args = ap.parse_args()

    shaping = {}
    if args.shaping_config:
        shaping = json.loads(args.shaping_config)

    cfg = PPOConfig(
        hidden_size=args.H,
        death_penalty=float(shaping.get("death_penalty", args.death_penalty)),
        num_envs=args.num_envs,
        rollout_len=args.rollout_len,
        total_frames=args.total_frames,
        ppo_epochs=args.ppo_epochs,
        num_minibatches=args.num_minibatches,
        lr=args.lr,
        lr_schedule=args.lr_schedule,
        seed=args.seed,
        run_name=args.run_name,
        checkpoint_every_frames=args.checkpoint_every_frames,
        eval_every_frames=args.eval_every_frames,
        eval_num_envs=args.eval_num_envs,
        eval_max_steps=args.eval_max_steps,
        resume=(not args.no_resume),
        hole_penalty=float(shaping.get("hole_penalty", 0.0)),
        empty_board_reward=float(shaping.get("empty_board_reward", 0.0)),
        roughness_penalty=float(shaping.get("roughness_penalty", 0.0)),
        combo_preserve_bonus=float(shaping.get("combo_preserve_bonus", 0.0)),
        reward_mode=args.reward_mode,
        ent_coef=args.ent_coef,
        gamma_schedule=args.gamma_schedule,
        gamma_start=args.gamma_start,
        gamma_end=args.gamma_end,
        gamma_warmup_frames=args.gamma_warmup_frames,
        plateau_min_frames=args.plateau_min_frames,
        arch_type=args.arch_type,
        n_conv=args.n_conv,
        n_filters=args.n_filters,
    )
    main(cfg)
