from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="PD teacher feasibility probe. Default: zero action (official semantics -> PD "
    "target = q_default). With --reference_action it drives the legal reference action so the "
    "PD target tracks q_ref(t). Reports survival; if even the reference action cannot finish the "
    "clip, the scene/termination contract is infeasible (asset/terrain/physics, not RL)."
)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--start_phase", type=int, default=0)
parser.add_argument(
    "--reference_action",
    action="store_true",
    default=False,
    help="Drive the legal reference action a_ref(t) = S^-1 * (q_ref(t) - q_default) each step "
    "instead of the zero action. With the official q_target = q_default + S*a contract this makes "
    "the PD target equal the reference pose q_ref(t). If even this cannot pass the 760-850 stand-up "
    "segment, the remaining problem is asset/terrain/physics, not RL.",
)
# Nominal teacher probe: reset noise, pushes, observation noise AND startup randomization are
# all OFF by default so friction / COM / default-joint bias are deterministic. Pass
# --startup_randomization to re-enable domain randomization for the follow-up robustness test.
parser.add_argument("--no_reset_noise", action="store_true", default=True)
parser.add_argument("--no_obs_noise", action="store_true", default=True)
parser.add_argument("--startup_randomization", action="store_true", default=False)
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
        startup_randomization=args_cli.startup_randomization,
        reset_noise=not args_cli.no_reset_noise,
        interval_pushes=False,
        observation_noise=not args_cli.no_obs_noise,
        adaptive_motion_sampling=False,
        max_episode_steps=-1,
        motion_start_phase=args_cli.start_phase,
    )
    env = G1MimicEnv(cfg)
    # Motion end is a clean stop (success), not a teleport roll-in, for the probe.
    env.terminate_on_motion_end = True
    if args_cli.feet_only_termination:
        feet = ["left_ankle_roll_link", "right_ankle_roll_link"]
        env.termination_body_indices = [env.track_body_names.index(n) for n in feet]
        print(f"[PROBE] OVERRIDE termination_body_indices -> feet only {feet}", flush=True)
    num_frames = int(env.motion.num_frames)
    device = env.device

    start = torch.full((env.num_envs,), int(args_cli.start_phase), dtype=torch.long, device=device)
    env.reset(phase_indices=start)

    zero_action = torch.zeros(env.num_envs, env.action_dim, device=device)

    def reference_action() -> torch.Tensor:
        # a_ref(t) = S^-1 * (q_ref(t) - q_default), evaluated at the CURRENT (pre-advance) phase
        # so the resulting PD target q_default + S*a_ref equals the reference pose q_ref(t) that
        # this step's reward is scored against. Motion frames are in the action-joint order.
        q_ref = env.motion.get_frame(env.phase_steps)["joint_pos"]
        return (q_ref - env.default_action_joint_pos) / env.action_scale

    alive = torch.ones(env.num_envs, dtype=torch.bool, device=device)
    survived_steps = torch.zeros(env.num_envs, dtype=torch.long, device=device)
    death_cause = {"anchor_pos_bad": 0, "anchor_ori_bad": 0, "ee_body_bad": 0, "time_out": 0, "motion_complete": 0}

    from env.config import EE_Z_TERMINATION_THRESHOLD
    # Roll every reference frame from start_phase through the final frame inclusive (diagnostic
    # only; does not affect training).
    max_steps = num_frames - int(args_cli.start_phase)
    term_names = list(env.ee_body_names)  # ee_z_error_by_body is indexed over ee_body_indices
    print(f"[PROBE] num_frames={num_frames} start_phase={args_cli.start_phase} max_steps={max_steps} num_envs={env.num_envs}", flush=True)
    print(f"[PROBE] ee_body_order={term_names} term_threshold={EE_Z_TERMINATION_THRESHOLD}", flush=True)
    mode = "reference_action a_ref(t)=S^-1*(q_ref(t)-q_default)" if args_cli.reference_action else "zero_action (PD target=q_default)"
    print(f"[PROBE] action_mode={mode}", flush=True)
    for step in range(max_steps):
        action = reference_action() if args_cli.reference_action else zero_action
        _, _, done, info = env.step(action, auto_reset=False)
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
