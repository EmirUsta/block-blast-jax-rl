from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class AZConfig:
    H: int = 128
    num_actions: int = 192
    num_simulations_train: int = 100
    num_simulations_eval: int = 100
    max_depth: int = 64
    dirichlet_fraction: float = 0.25
    dirichlet_alpha: float = 0.3
    pb_c_init: float = 1.25
    pb_c_base: float = 19652.0
    B: int = 64
    max_moves_per_iter: int = 50
    eval_B: int = 32
    eval_max_steps: int = 200
    lr: float = 1e-4
    lr_value_head_mult: float = 3.0
    minibatch_size: int = 512
    epochs_per_iter: int = 4
    grad_clip: float = 5.0
    grad_clip_floor: float = 0.5
    grad_clip_warmup_iters: int = 50
    vf_coef: float = 0.5
    discount: float = 0.997
    buffer_capacity: int = 100_000
    total_iters: int = 400
    eval_every: int = 10
    checkpoint_every: int = 25
    plateau_check_window: int = 5
    plateau_min_iters: int = 50
    plateau_min_improvement: float = 0.005
    seed: int = 42
    run_name: str = "az_v1.0"
    warmstart: str | None = None
    checkpoint_dir: str = "runs"
    reset_value_head: bool = True
    reward_combo_coef: float = 0.0
    visit_temperature: float = 1.0
    kl_anchor_coef: float = 0.0
    kl_anchor_ref: str | None = None
    hardstart_pool_path: str | None = None
    hardstart_prob: float = 0.0
    hardstart_init_from_pool: bool = False
    use_bf16: bool = False

    def to_dict(self) -> dict:
        return asdict(self)
