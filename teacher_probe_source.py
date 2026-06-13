"""Teacher probe in the SOURCE mimic project (dance_102) for comparison.

Drives PD targets to the reference pose (zero residual action) and reports how long the
robot survives. This is the baseline: if the source teacher survives long, the source
physics+action-semantics+termination contract works; comparing against the holosoma probe
isolates which part diverged.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=("zero", "reference"), default="reference")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--max_steps", type=int, default=960)
parser.add_argument("--motion_file", type=str, default="", help="Override motion npz. Empty uses DEFAULT_MOTION_FILE.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

from env.config import MimicEnvConfig, DEFAULT_MOTION_FILE  # noqa: E402
from env.mimic import G1MimicEnv  # noqa: E402


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    motion_file = args_cli.motion_file if args_cli.motion_file else str(DEFAULT_MOTION_FILE)
    cfg = MimicEnvConfig(
        device=device,
        num_envs=args_cli.num_envs,
        motion_file=motion_file,
        reset_noise=False,
        interval_pushes=False,
        observation_noise=False,
        startup_randomization=False,
        max_episode_steps=args_cli.max_steps + 10,
    )
    env = G1MimicEnv(cfg)
    env.reset_envs(torch.arange(env.num_envs, device=device), phase_indices=torch.zeros(env.num_envs, dtype=torch.long, device=device))

    num_envs = env.num_envs
    alive = torch.ones(num_envs, dtype=torch.bool, device=device)
    live_steps = torch.zeros(num_envs, device=device)
    joint_err_sum = torch.zeros(num_envs, device=device)
    counted = torch.zeros(num_envs, device=device)
    cause_counts: dict[str, int] = {}

    for step in range(int(args_cli.max_steps)):
        if args_cli.mode == "zero":
            action = torch.zeros(num_envs, env.action_dim, device=device)
        else:
            # Teacher: zero residual -> _apply_action_targets uses ref_joint_pos[t+1] + scale*0.
            action = torch.zeros(num_envs, env.action_dim, device=device)
        obs, reward, done, info = env.step(action, auto_reset=False)
        if step < 60 and bool(alive[0]):
            db = info["debug_terms"]
            print(
                f"STEP {step:3d} alive={int(alive.sum().item()):3d} "
                f"anchor_z_err={db['anchor_z_error'][0].item():.3f} "
                f"anchor_grav={db['anchor_gravity_z_error'][0].item():.3f} "
                f"anchor_tilt={db['robot_anchor_tilt'][0].item():.3f} "
                f"anchor_h={db['robot_anchor_height'][0].item():.3f} "
                f"ee_z_max={db['termination_z_error_max'][0].item():.3f}",
                flush=True,
            )
        cur = env.robot.data.joint_pos[:, env.action_joint_ids]
        next_phase = env.motion.clamp_time_steps(env.phase_steps)
        ref = env.motion.joint_pos.index_select(0, next_phase)
        err = torch.norm(cur - ref, dim=-1)
        joint_err_sum = joint_err_sum + err * alive.to(err.dtype)
        counted = counted + alive.to(err.dtype)
        live_steps = live_steps + alive.to(live_steps.dtype)
        newly_done = alive & done.to(torch.bool)
        if bool(newly_done.any()):
            dt = info["done_terms"]
            for key in ("anchor_pos_bad", "anchor_ori_bad", "ee_body_bad", "time_out"):
                if key in dt:
                    cause_counts[key] = cause_counts.get(key, 0) + int((dt[key] & newly_done).sum().item())
        alive = alive & ~done.to(torch.bool)
        if not bool(alive.any()):
            break

    mean_err = (joint_err_sum / counted.clamp(min=1)).mean().item()
    print("=" * 60, flush=True)
    print(f"[SOURCE TEACHER PROBE] mode={args_cli.mode} num_envs={num_envs} max_steps={args_cli.max_steps}", flush=True)
    print(
        f"  live_steps: mean={live_steps.mean().item():.1f} p50={live_steps.median().item():.0f} "
        f"min={live_steps.min().item():.0f} max={live_steps.max().item():.0f}",
        flush=True,
    )
    print(f"  mean_joint_tracking_error(rad)={mean_err:.4f}", flush=True)
    print(f"  death_causes={cause_counts}", flush=True)
    print(f"  motion_frames={env.motion.num_frames}", flush=True)
    print("=" * 60, flush=True)
    import os
    os._exit(0)


if __name__ == "__main__":
    main()
