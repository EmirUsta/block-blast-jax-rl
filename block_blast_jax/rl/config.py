from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class PPOConfig:
    hidden_size: int = 128
    death_penalty: float = 0.0
    num_envs: int = 4096
    rollout_len: int = 128
    ppo_epochs: int = 4
    num_minibatches: int = 8
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    total_frames: int = 50_000_000
    seed: int = 0
    lr_schedule: str = "constant"
    log_every_updates: int = 1
    checkpoint_dir: str = "runs"
    run_name: str | None = None
    checkpoint_every_frames: int = 100_000_000
    eval_every_frames: int = 50_000_000
    eval_num_envs: int = 1024
    eval_max_steps: int = 500
    plateau_window_evals: int = 5
    plateau_min_improvement: float = 0.005
    plateau_min_frames: int = 500_000_000
    resume: bool = True
    hole_penalty: float = 0.0
    empty_board_reward: float = 0.0
    roughness_penalty: float = 0.0
    combo_preserve_bonus: float = 0.0
    reward_mode: str = "score"
    gamma_schedule: str = "constant"
    gamma_start: float = 0.99
    gamma_end: float = 0.99
    gamma_warmup_frames: int = 0
    arch_type: str = "mlp"
    n_conv: int = 2
    n_filters: int = 32

    @property
    def updates(self) -> int:
        per_update = self.num_envs * self.rollout_len
        return max(1, self.total_frames // per_update)

    def to_dict(self) -> dict:
        return asdict(self)
