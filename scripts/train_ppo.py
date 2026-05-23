from __future__ import annotations

import argparse
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Train the official Unitree-style PPO baseline on the local mimic env.")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--action_dim", type=int, default=29)
parser.add_argument("--policy_obs_dim", type=int, default=0)
parser.add_argument("--critic_obs_dim", type=int, default=0)
parser.add_argument("--actor_hidden_dims", type=int, nargs="+", default=[512, 256, 128])
parser.add_argument("--critic_hidden_dims", type=int, nargs="+", default=[512, 256, 128])
parser.add_argument("--activation", type=str, default="elu")
parser.add_argument("--init_noise_std", type=float, default=1.0)
parser.add_argument("--num_steps_per_env", type=int, default=24)
parser.add_argument("--policy_epochs", type=int, default=5)
parser.add_argument("--num_mini_batches", type=int, default=4)
parser.add_argument("--clip_range", type=float, default=0.2)
parser.add_argument("--discount_gamma", type=float, default=0.99)
parser.add_argument("--gae_lambda", type=float, default=0.95)
parser.add_argument("--value_loss_coef", type=float, default=1.0)
parser.add_argument("--entropy_coef", type=float, default=0.005)
parser.add_argument("--lr", type=float, default=1.0e-3)
parser.add_argument("--schedule", choices=("adaptive", "fixed"), default="adaptive")
parser.add_argument("--desired_kl", type=float, default=0.01)
parser.add_argument("--max_grad_norm", type=float, default=1.0)
parser.add_argument("--use_clipped_value_loss", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--max_updates", type=int, default=30000)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--sim_dt", type=float, default=0.02)
parser.add_argument("--max_episode_steps", type=int, default=1500)
parser.add_argument("--startup_randomization", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--reset_noise", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--interval_pushes", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--motion_start_phase", type=int, default=0)
parser.add_argument("--motion_end_phase", type=int, default=-1)
parser.add_argument("--motion_file", type=str, default="")
parser.add_argument("--run_name", type=str, default="")
parser.add_argument("--run_root", type=str, default="runs")
parser.add_argument("--checkpoint_dir", type=str, default="")
parser.add_argument("--log_file", type=str, default="")
parser.add_argument("--save_every", type=int, default=500)
parser.add_argument("--resume", type=str, default="")
parser.add_argument("--reset_optimizer_on_resume", action="store_true", default=False)
parser.add_argument("--log_every", type=int, default=1)
parser.add_argument("--validation_every", type=int, default=0)
parser.add_argument("--validation_max_steps", type=int, default=500)
parser.add_argument("--validation_start_phase", type=int, default=0)
parser.add_argument("--validation_fixed_seed", type=int, default=-1)
parser.add_argument("--validation_preserve_state", action="store_true", default=True)
parser.add_argument("--target_validation_steps", type=int, default=0)
parser.add_argument("--success_checkpoint_name", type=str, default="success_10s.pt")
parser.add_argument("--fix_root_link", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from env import DEFAULT_MOTION_FILE
from engine.ppo import OfficialPPOConfig, OfficialPPOTrainer


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
        run_name = args_cli.run_name or f"g1_official_ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
    cfg = OfficialPPOConfig(
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        sim_dt=args_cli.sim_dt,
        fix_root_link=args_cli.fix_root_link,
        startup_randomization=args_cli.startup_randomization,
        motion_start_phase=args_cli.motion_start_phase,
        motion_end_phase=args_cli.motion_end_phase,
        max_episode_steps=args_cli.max_episode_steps,
        motion_file=motion_file,
        reset_noise=args_cli.reset_noise,
        interval_pushes=args_cli.interval_pushes,
        action_dim=args_cli.action_dim,
        policy_obs_dim=args_cli.policy_obs_dim,
        critic_obs_dim=args_cli.critic_obs_dim,
        actor_hidden_dims=tuple(args_cli.actor_hidden_dims),
        critic_hidden_dims=tuple(args_cli.critic_hidden_dims),
        activation=args_cli.activation,
        init_noise_std=args_cli.init_noise_std,
        num_steps_per_env=args_cli.num_steps_per_env,
        policy_epochs=args_cli.policy_epochs,
        num_mini_batches=args_cli.num_mini_batches,
        clip_range=args_cli.clip_range,
        discount_gamma=args_cli.discount_gamma,
        gae_lambda=args_cli.gae_lambda,
        value_loss_coef=args_cli.value_loss_coef,
        entropy_coef=args_cli.entropy_coef,
        lr=args_cli.lr,
        schedule=args_cli.schedule,
        desired_kl=args_cli.desired_kl,
        max_grad_norm=args_cli.max_grad_norm,
        use_clipped_value_loss=args_cli.use_clipped_value_loss,
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
        validation_fixed_seed=args_cli.validation_fixed_seed,
        validation_preserve_state=args_cli.validation_preserve_state,
        target_validation_steps=args_cli.target_validation_steps,
        success_checkpoint_name=args_cli.success_checkpoint_name,
    )

    trainer = OfficialPPOTrainer(simulation_app=simulation_app, cfg=cfg)
    try:
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
