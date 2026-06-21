from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Zero-residual PD teacher feasibility probe: roll the whole clip from phase 0 "
    "with zero action offsets (PD target = reference pose) and report survival. If even the "
    "teacher cannot finish the clip, the scene/termination contract is infeasible."
)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--start_phase", type=int, default=0)
parser.add_argument("--no_reset_noise", action="store_true", default=False)
parser.add_argument("--no_obs_noise", action="store_true", default=True)
parser.add_argument(
    "--feet_only_termination",
    action="store_true",
    default=False,
    help="Override termination body set to feet only (drop wrists). Isolates whether the "
    "teacher infeasibility is caused by the wrist z-gate or by something physical.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

from env.config import MimicEnvConfig
from env.mimic import G1MimicEnv


def main() -> None:
    cfg = MimicEnvConfig(
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        render=False,
        startup_randomization=False,
        reset_noise=not args_cli.no_reset_noise,
        interval_pushes=False,
        observation_noise=not args_cli.no_obs_noise,
        adaptive_motion_sampling=False,
        max_episode_steps=-1,
        motion_start_phase=args_cli.start_phase,
    )
    env = G1MimicEnv(cfg)
    if args_cli.feet_only_termination:
        feet = ["left_ankle_roll_link", "right_ankle_roll_link"]
        env.termination_body_indices = [env.track_body_names.index(n) for n in feet]
        print(f"[PROBE] OVERRIDE termination_body_indices -> feet only {feet}", flush=True)
    num_frames = int(env.motion.num_frames)
    device = env.device

    start = torch.full((env.num_envs,), int(args_cli.start_phase), dtype=torch.long, device=device)
    env.reset(phase_indices=start)

    zero_action = torch.zeros(env.num_envs, env.action_dim, device=device)
    alive = torch.ones(env.num_envs, dtype=torch.bool, device=device)
    survived_steps = torch.zeros(env.num_envs, dtype=torch.long, device=device)
    death_cause = {"anchor_pos_bad": 0, "anchor_ori_bad": 0, "ee_body_bad": 0, "time_out": 0}

    from env.config import EE_Z_TERMINATION_THRESHOLD
    max_steps = num_frames - int(args_cli.start_phase) - 1
    term_names = list(env.ee_body_names)  # ee_z_error_by_body is indexed over ee_body_indices
    print(f"[PROBE] num_frames={num_frames} start_phase={args_cli.start_phase} max_steps={max_steps} num_envs={env.num_envs}", flush=True)
    print(f"[PROBE] ee_body_order={term_names} term_threshold={EE_Z_TERMINATION_THRESHOLD}", flush=True)
    for step in range(max_steps):
        _, _, done, info = env.step(zero_action, auto_reset=False)
        dbg = info["debug_terms"]
        ee_by_body = dbg["ee_z_error_by_body"]  # (num_envs, num_ee_bodies)
        if step % 5 == 0 or bool((alive & done).any()):
            per_body = ee_by_body[alive].mean(dim=0) if bool(alive.any()) else ee_by_body.mean(dim=0)
            body_str = " ".join(f"{n}={float(v):.3f}" for n, v in zip(term_names, per_body.tolist()))
            print(f"[PROBE] step={step+1} alive={float(alive.float().mean()):.3f} ee_z[{body_str}]", flush=True)
        newly_dead = alive & done
        if bool(newly_dead.any()):
            terms = info["done_terms"]
            for key in death_cause:
                if key in terms:
                    death_cause[key] += int((newly_dead & terms[key].bool()).sum().item())
        survived_steps[alive] = step + 1
        alive = alive & ~done
        if not bool(alive.any()):
            break

    ss = survived_steps.float()
    print("[PROBE_RESULT] "
          f"alive_frac_end={float(alive.float().mean()):.3f} "
          f"survived_steps_mean={float(ss.mean()):.1f} "
          f"min={int(ss.min())} p50={int(ss.median())} max={int(ss.max())} "
          f"reached_phase_mean={int(args_cli.start_phase)+float(ss.mean()):.1f}", flush=True)
    print(f"[PROBE_DEATH_CAUSE] {death_cause}", flush=True)
    full = int((survived_steps >= max_steps).sum().item())
    print(f"[PROBE_FULL_CLIP] {full}/{env.num_envs} envs finished the whole clip "
          f"({100.0*full/env.num_envs:.1f}%)", flush=True)
    simulation_app.close()


if __name__ == "__main__":
    main()
