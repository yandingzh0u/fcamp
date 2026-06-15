from __future__ import annotations

import argparse
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Train a G1 flow-matching policy with MixGRPO SDE exploration and group-relative policy optimization."
)
parser.add_argument("--num_envs", type=int, default=8192, help="Number of parallel IsaacLab environments.")
parser.add_argument("--action_dim", type=int, default=29, help="Single-step action dimension.")
parser.add_argument("--policy_obs_dim", type=int, default=0, help="Observation dim consumed by the policy. 0 uses the full env observation.")
parser.add_argument(
    "--actor_hidden_dims",
    type=int,
    nargs="+",
    default=[512, 256, 128],
    help="Actor hidden dimensions. Default matches Unitree RSL-RL PPO.",
)
parser.add_argument("--activation", type=str, default="elu", help="Actor activation. Default matches Unitree RSL-RL PPO.")
parser.add_argument("--flow_steps", type=int, default=4, help="Number of flow denoising steps.")
parser.add_argument(
    "--action_squash_scale",
    type=float,
    default=5.0,
    help="Tanh scale for mapping final flow latents to robot actions.",
)
parser.add_argument(
    "--init_noise_std",
    type=float,
    default=0.8,
    help="Initial latent std for flow matching. Non-zero opens up intra-group diversity beyond the SDE process noise.",
)
parser.add_argument(
    "--init_same_noise",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Use the same initial latent for all generations in a GRPO group. Default False so every branch gets an independent initial latent (broader group exploration).",
)
parser.add_argument(
    "--first_generation_zero_noise",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Make branch 0 in each GRPO group use zero initial/SDE noise. Off by default because an exact mean-path SDE sample has almost no first-order log-prob gradient.",
)
parser.add_argument(
    "--eval_initial_noise",
    choices=("random", "zero"),
    default="zero",
    help="Initial latent used during validation/playback. Zero gives the deterministic policy path.",
)
parser.add_argument("--sde_eta", type=float, default=0.7, help="SDE noise eta for the four flow steps.")
parser.add_argument(
    "--num_generations",
    type=int,
    default=4,
    help="Number of same-state SDE samples per GRPO group.",
)
parser.add_argument(
    "--chunks_per_rollout",
    type=int,
    default=24,
    help=(
        "Fallback number of SDE-explored policy chunks per GRPO update when "
        "--rollout_env_steps <= 0."
    ),
)
parser.add_argument(
    "--rollout_env_steps",
    type=int,
    default=120,
    help=(
        "Fixed environment frames per GRPO update. Effective chunks are "
        "rollout_env_steps // horizon and must divide exactly. "
        "Set <=0 to use --chunks_per_rollout directly. 120 = 10 chunks of horizon 12, "
        "covering the ~74-frame death region while every frame is trained (no open-loop tail)."
    ),
)
parser.add_argument(
    "--horizon",
    type=int,
    default=12,
    help="Number of env-frames each policy chunk advances; chunk reward = discounted sum across these frames.",
)
parser.add_argument(
    "--basis_count",
    type=int,
    default=4,
    help=(
        "Velocity-basis coefficient count for h>1 action chunks. The flow/log_prob run in a "
        "basis_count*action_dim coefficient latent of low-frequency velocity modes; these are "
        "integrated into a displacement trajectory (first frame 0) anchored to the previous "
        "residual so cross-chunk continuity is intrinsic. Clamped to [1, horizon-1]. 0 = horizon-1."
    ),
)
parser.add_argument("--discount_gamma", type=float, default=0.99, help="Chunk return-to-go discount for GRPO advantages.")
parser.add_argument(
    "--policy_epochs",
    type=int,
    default=5,
    help="Number of MixGRPO passes over each sampled batch. Default matches the PPO update budget used for robot control.",
)
parser.add_argument(
    "--clip_range",
    type=float,
    default=0.3,
    help="Clipping range for the MixGRPO step log-prob ratio. Default is widened for robot control.",
)
parser.add_argument("--adv_clip_max", type=float, default=5.0, help="Clamp absolute advantages in the MixGRPO policy loss.")
parser.add_argument("--desired_kl", type=float, default=0.06, help="Adaptive learning-rate KL target.")
parser.add_argument("--kl_penalty_coef", type=float, default=0.0, help="KL penalty coefficient added to the policy loss. 0 disables (standard PPO clip only).")
parser.add_argument(
    "--num_mini_batches",
    type=int,
    default=4,
    help="Number of mini-batches per policy epoch.",
)
parser.add_argument(
    "--mini_batch_size",
    type=int,
    default=0,
    help="Deprecated override for policy mini-batch size. 0 derives it from --num_mini_batches.",
)
parser.add_argument(
    "--micro_batch_size",
    type=int,
    default=8192,
    help="Max samples per log-prob forward/backward inside one logical mini-batch. Uses gradient accumulation.",
)
parser.add_argument("--policy_lr", type=float, default=1.0e-3, help="Flow policy learning rate for robot control.")
parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping threshold.")
parser.add_argument("--max_updates", type=int, default=3000, help="Total number of MixGRPO training iterations.")
parser.add_argument("--seed", type=int, default=0, help="Random seed.")
parser.add_argument("--sim_dt", type=float, default=0.02, help="Simulation timestep.")
parser.add_argument(
    "--action_scale_multiplier",
    type=float,
    default=1.0,
    help="Multiplier on the env residual action scale. Values like 0.25 make the policy stay closer to the reference pose.",
)
parser.add_argument("--action_rate_weight", type=float, default=1.0e-1, help="Reward penalty weight for action delta.")
parser.add_argument("--max_episode_steps", type=int, default=-1, help="Episode time-out in steps. <=0 follows the motion clip length (recommended), so finishing the whole motion is the success bar.")
parser.add_argument(
    "--startup_randomization",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Apply official startup randomization for friction, torso COM, and default joint positions.",
)
parser.add_argument(
    "--reset_noise",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Enable official reset pose/velocity/joint noise. Use --no-reset_noise for clean curriculum probes.",
)
parser.add_argument(
    "--interval_pushes",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Enable official interval push perturbations. Use --no-interval_pushes for clean curriculum probes.",
)
parser.add_argument(
    "--observation_noise",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Enable actor observation noise during training. Validation defaults to a clean observation path.",
)
parser.add_argument("--motion_start_phase", type=int, default=0, help="First reference phase sampled for training resets.")
parser.add_argument("--motion_end_phase", type=int, default=-1, help="Last reference phase sampled for training resets. Negative uses motion end.")
parser.add_argument(
    "--motion_start_phase_ratio",
    type=float,
    default=0.25,
    help=(
        "Per-update group fraction pinned exactly to phase 0; the remaining (1-ratio) is "
        "stratified-uniform over the whole clip. Default 0.25 so the real deployment entry "
        "(phase 0) is not starved to ~0.1% under pure uniform sampling, while still covering "
        "the full motion. Not adaptive."
    ),
)
parser.add_argument("--motion_file", type=str, default="", help="Optional override for the dance npz path.")
parser.add_argument("--run_name", type=str, default="", help="Optional run folder name under --run_root.")
parser.add_argument("--run_root", type=str, default="runs", help="Root directory for automatic logs and checkpoints.")
parser.add_argument("--checkpoint_dir", type=str, default="", help="Directory for saving checkpoints. Defaults to runs/<run_name>/checkpoints.")
parser.add_argument("--log_file", type=str, default="", help="Path for train stdout/stderr log. Defaults to runs/<run_name>/logs/train.log.")
parser.add_argument("--save_every", type=int, default=50, help="Save a checkpoint every N updates. 0 disables.")
parser.add_argument("--resume", type=str, default="", help="Optional checkpoint path to resume from.")
parser.add_argument(
    "--reset_optimizer_on_resume",
    action="store_true",
    default=False,
    help="Load policy weights from --resume but start a fresh optimizer state.",
)
parser.add_argument("--log_every", type=int, default=1, help="Print metrics every N updates.")
parser.add_argument("--validation_every", type=int, default=50, help="Run a validation rollout every N updates. 0 disables.")
parser.add_argument("--validation_max_steps", type=int, default=1500, help="Max simulation steps per validation rollout.")
parser.add_argument("--validation_start_phase", type=int, default=0, help="Reference motion phase for validation resets.")
parser.add_argument("--validation_done_frac_early_stop", type=float, default=0.98, help="Stop a validation rollout once this fraction of envs have terminated (trims the long survivor tail).")
parser.add_argument(
    "--validation_fixed_seed",
    type=int,
    default=-1,
    help="Optional second fixed-seed validation seed. Negative disables the extra validation pass.",
)
parser.add_argument(
    "--validation_preserve_state",
    action="store_true",
    default=True,
    help="Snapshot and restore the training simulator around validation.",
)
parser.add_argument(
    "--validation_observation_noise",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Enable observation noise during validation. Default off so deterministic eval/playback is actually deterministic.",
)
parser.add_argument(
    "--target_validation_steps",
    type=int,
    default=0,
    help="Stop early and save success checkpoint once validation min steps reach this value.",
)
parser.add_argument("--success_checkpoint_name", type=str, default="success_10s.pt", help="Filename for the early-stop success checkpoint.")
parser.add_argument("--validate_only", action="store_true", default=False, help="Load a checkpoint and run one validation rollout without training.")
parser.add_argument("--debug_probe", action="store_true", default=False, help="Print detailed tensor probes during training.")
parser.add_argument("--debug_probe_every", type=int, default=1, help="Print debug probes every N updates when --debug_probe is set.")
parser.add_argument("--fix_root_link", action="store_true", default=False, help="Lock the robot base in place.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.rollout_env_steps > 0:
    if args_cli.horizon <= 0:
        parser.error("--horizon must be positive when --rollout_env_steps is enabled.")
    if args_cli.rollout_env_steps % args_cli.horizon != 0:
        parser.error(
            "--rollout_env_steps must be divisible by --horizon; "
            f"got rollout_env_steps={args_cli.rollout_env_steps}, horizon={args_cli.horizon}."
        )

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from env.config import DEFAULT_MOTION_FILE
from engine.mixgrpo.config import MixGRPOConfig
from engine.mixgrpo.trainer import MixGRPOTrainer


class _TeeStream:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def _setup_run_io() -> tuple[str, str]:
    if args_cli.checkpoint_dir:
        checkpoint_dir = Path(args_cli.checkpoint_dir).expanduser()
        run_dir = checkpoint_dir.parent if checkpoint_dir.name == "checkpoints" else checkpoint_dir
    else:
        run_name = args_cli.run_name or f"g1_mixgrpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir = Path(args_cli.run_root).expanduser() / run_name
        checkpoint_dir = run_dir / "checkpoints"

    log_file = Path(args_cli.log_file).expanduser() if args_cli.log_file else run_dir / "logs" / "train.log"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    log_handle = open(log_file, "a", buffering=1)
    sys.stdout = _TeeStream(sys.__stdout__, log_handle)
    sys.stderr = _TeeStream(sys.__stderr__, log_handle)
    print(f"[INFO] run_dir={run_dir.resolve()}", flush=True)
    print(f"[INFO] log_file={log_file.resolve()}", flush=True)
    return str(checkpoint_dir), str(log_file)


def main() -> None:
    checkpoint_dir, _ = _setup_run_io()
    motion_file = args_cli.motion_file if args_cli.motion_file else str(DEFAULT_MOTION_FILE)
    cfg = MixGRPOConfig(
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        sim_dt=args_cli.sim_dt,
        render=not args_cli.headless,
        fix_root_link=args_cli.fix_root_link,
        action_scale_multiplier=args_cli.action_scale_multiplier,
        startup_randomization=args_cli.startup_randomization,
        motion_start_phase=args_cli.motion_start_phase,
        motion_end_phase=args_cli.motion_end_phase,
        motion_start_phase_ratio=args_cli.motion_start_phase_ratio,
        max_episode_steps=args_cli.max_episode_steps,
        motion_file=motion_file,
        reset_noise=args_cli.reset_noise,
        interval_pushes=args_cli.interval_pushes,
        observation_noise=args_cli.observation_noise,
        action_rate_weight=args_cli.action_rate_weight,
        action_dim=args_cli.action_dim,
        policy_obs_dim=args_cli.policy_obs_dim,
        horizon=args_cli.horizon,
        basis_count=args_cli.basis_count,
        actor_hidden_dims=tuple(args_cli.actor_hidden_dims),
        activation=args_cli.activation,
        flow_steps=args_cli.flow_steps,
        action_squash_scale=args_cli.action_squash_scale,
        init_noise_std=args_cli.init_noise_std,
        init_same_noise=args_cli.init_same_noise,
        first_generation_zero_noise=args_cli.first_generation_zero_noise,
        eval_initial_noise=args_cli.eval_initial_noise,
        sde_eta=args_cli.sde_eta,
        num_generations=args_cli.num_generations,
        rollout_env_steps=args_cli.rollout_env_steps,
        chunks_per_rollout=args_cli.chunks_per_rollout,
        discount_gamma=args_cli.discount_gamma,
        clip_range=args_cli.clip_range,
        adv_clip_max=args_cli.adv_clip_max,
        desired_kl=args_cli.desired_kl,
        kl_penalty_coef=args_cli.kl_penalty_coef,
        policy_epochs=args_cli.policy_epochs,
        num_mini_batches=args_cli.num_mini_batches,
        mini_batch_size=args_cli.mini_batch_size,
        micro_batch_size=args_cli.micro_batch_size,
        policy_lr=args_cli.policy_lr,
        max_grad_norm=args_cli.max_grad_norm,
        max_updates=args_cli.max_updates,
        seed=args_cli.seed,
        log_every=args_cli.log_every,
        save_every=args_cli.save_every,
        checkpoint_dir=checkpoint_dir,
        resume=args_cli.resume,
        reset_optimizer_on_resume=args_cli.reset_optimizer_on_resume,
        validation_every=args_cli.validation_every,
        validation_max_steps=args_cli.validation_max_steps,
        validation_start_phase=args_cli.validation_start_phase,
        validation_done_frac_early_stop=args_cli.validation_done_frac_early_stop,
        validation_fixed_seed=args_cli.validation_fixed_seed,
        validation_preserve_state=args_cli.validation_preserve_state,
        validation_observation_noise=args_cli.validation_observation_noise,
        target_validation_steps=args_cli.target_validation_steps,
        success_checkpoint_name=args_cli.success_checkpoint_name,
        debug_probe=args_cli.debug_probe,
        debug_probe_every=args_cli.debug_probe_every,
    )

    trainer = MixGRPOTrainer(simulation_app=simulation_app, cfg=cfg)
    try:
        if args_cli.validate_only:
            metrics = trainer.run_validation_rollout()
            trainer._log_validation_metrics(metrics)
        else:
            trainer.train()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
