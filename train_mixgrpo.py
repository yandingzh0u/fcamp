from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Train a G1 flow-matching policy with MixGRPO SDE exploration and group-relative policy optimization."
)
parser.add_argument("--num_envs", type=int, default=4096, help="Number of parallel IsaacLab environments.")
parser.add_argument("--horizon", type=int, default=1, help="Action chunk size (number of actions per policy sample).")
parser.add_argument("--action_dim", type=int, default=29, help="Single-step action dimension.")
parser.add_argument("--policy_obs_dim", type=int, default=0, help="Observation dim consumed by the policy. 0 uses the full env observation.")
parser.add_argument("--hidden_dim", type=int, default=512, help="Actor MLP hidden dimension.")
parser.add_argument("--time_embed_dim", type=int, default=64, help="Sinusoidal time embedding dimension.")
parser.add_argument("--depth", type=int, default=4, help="Number of residual MLP blocks in the velocity network.")
parser.add_argument("--flow_steps", type=int, default=4, help="Number of Euler integration steps for flow matching.")
parser.add_argument("--action_limit", type=float, default=1.0, help="Tanh-squashed policy action offset limit.")
parser.add_argument("--cps_eta", type=float, default=0.7, help="CPS exploration strength in [0, 1).")
parser.add_argument("--eta", dest="cps_eta", type=float, help=argparse.SUPPRESS)
parser.add_argument("--group_size", type=int, default=4, help="Number of groups per observation for GRPO.")
parser.add_argument("--chunks_per_rollout", type=int, default=24, help="Number of action chunks per rollout (total steps = chunks * horizon).")
parser.add_argument("--discount_gamma", type=float, default=0.99, help="Chunk return-to-go discount for GRPO advantages.")
parser.add_argument("--policy_epochs", type=int, default=1, help="Number of PPO gradient epochs per update.")
parser.add_argument("--clip_range", type=float, default=1e-2, help="MixGRPO clipping range.")
parser.add_argument("--adv_clip_max", type=float, default=5.0, help="Clamp advantage magnitude.")
parser.add_argument("--latent_reg_coeff", type=float, default=0.01, help="Soft penalty coefficient for excessive pre-tanh latent magnitude.")
parser.add_argument("--latent_soft_limit", type=float, default=2.0, help="Pre-tanh latent magnitude allowed before soft regularization.")
parser.add_argument("--action_saturation_coeff", type=float, default=0.05, help="Penalty coefficient for tanh-squashed actions near the action limit.")
parser.add_argument("--action_saturation_threshold", type=float, default=0.85, help="Absolute action value threshold before saturation penalty.")
parser.add_argument("--mini_batch_size", type=int, default=1024, help="Mini-batch size for policy updates.")
parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping threshold.")
parser.add_argument("--max_updates", type=int, default=1000, help="Total number of MixGRPO training iterations.")
parser.add_argument("--seed", type=int, default=0, help="Random seed.")
parser.add_argument("--sim_dt", type=float, default=0.02, help="Simulation timestep.")
parser.add_argument("--max_episode_steps", type=int, default=1500, help="Official 30s time-out threshold for mimic env.")
parser.add_argument("--motion_start_phase", type=int, default=0, help="First reference phase sampled for training resets.")
parser.add_argument("--motion_end_phase", type=int, default=-1, help="Last reference phase sampled for training resets. Negative uses motion end.")
parser.add_argument("--motion_file", type=str, default="", help="Optional override for the dance npz path.")
parser.add_argument("--checkpoint_dir", type=str, default="", help="Directory for saving checkpoints.")
parser.add_argument("--save_every", type=int, default=50, help="Save a checkpoint every N updates. 0 disables.")
parser.add_argument("--resume", type=str, default="", help="Optional checkpoint path to resume from.")
parser.add_argument("--log_every", type=int, default=1, help="Print metrics every N updates.")
parser.add_argument("--validation_every", type=int, default=25, help="Run a validation rollout every N updates. 0 disables.")
parser.add_argument("--validation_max_steps", type=int, default=500, help="Max simulation steps per validation rollout.")
parser.add_argument("--validation_start_phase", type=int, default=0, help="Reference motion phase for validation resets.")
parser.add_argument(
    "--target_validation_steps",
    type=int,
    default=0,
    help="Stop early and save success checkpoint once both random and fixed validation min steps reach this value.",
)
parser.add_argument("--success_checkpoint_name", type=str, default="success_10s.pt", help="Filename for the early-stop success checkpoint.")
parser.add_argument("--fix_root_link", action="store_true", default=False, help="Lock the robot base in place.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from env import DEFAULT_MOTION_FILE
from engine import MixGRPOConfig, MixGRPOTrainer


def main() -> None:
    motion_file = args_cli.motion_file if args_cli.motion_file else str(DEFAULT_MOTION_FILE)
    cfg = MixGRPOConfig(
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        sim_dt=args_cli.sim_dt,
        fix_root_link=args_cli.fix_root_link,
        motion_start_phase=args_cli.motion_start_phase,
        motion_end_phase=args_cli.motion_end_phase,
        max_episode_steps=args_cli.max_episode_steps,
        motion_file=motion_file,
        action_dim=args_cli.action_dim,
        policy_obs_dim=args_cli.policy_obs_dim,
        horizon=args_cli.horizon,
        hidden_dim=args_cli.hidden_dim,
        time_embed_dim=args_cli.time_embed_dim,
        depth=args_cli.depth,
        flow_steps=args_cli.flow_steps,
        action_limit=args_cli.action_limit,
        cps_eta=args_cli.cps_eta,
        group_size=args_cli.group_size,
        chunks_per_rollout=args_cli.chunks_per_rollout,
        discount_gamma=args_cli.discount_gamma,
        clip_range=args_cli.clip_range,
        adv_clip_max=args_cli.adv_clip_max,
        latent_reg_coeff=args_cli.latent_reg_coeff,
        latent_soft_limit=args_cli.latent_soft_limit,
        action_saturation_coeff=args_cli.action_saturation_coeff,
        action_saturation_threshold=args_cli.action_saturation_threshold,
        policy_epochs=args_cli.policy_epochs,
        mini_batch_size=args_cli.mini_batch_size,
        lr=args_cli.lr,
        max_grad_norm=args_cli.max_grad_norm,
        max_updates=args_cli.max_updates,
        seed=args_cli.seed,
        log_every=args_cli.log_every,
        save_every=args_cli.save_every,
        checkpoint_dir=args_cli.checkpoint_dir,
        resume=args_cli.resume,
        validation_every=args_cli.validation_every,
        validation_max_steps=args_cli.validation_max_steps,
        validation_start_phase=args_cli.validation_start_phase,
        target_validation_steps=args_cli.target_validation_steps,
        success_checkpoint_name=args_cli.success_checkpoint_name,
    )

    trainer = MixGRPOTrainer(simulation_app=simulation_app, cfg=cfg)
    try:
        trainer.train()
    finally:
        import os
        os._exit(0)


if __name__ == "__main__":
    main()
