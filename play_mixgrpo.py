from __future__ import annotations

import argparse
from pathlib import Path
import time

import torch

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Play a trained MixGRPO checkpoint in IsaacLab.")
parser.add_argument("--checkpoint", type=str, default="", help="Path to a saved MixGRPO checkpoint (.pt).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to play.")
parser.add_argument("--max_steps", type=int, default=0, help="Optional hard stop. 0 means run until the app closes.")
parser.add_argument("--start_phase", type=int, default=-1, help="Motion phase index used for reset. Negative uses motion_start_phase.")
parser.add_argument("--flow_steps", type=int, default=-1, help="Override checkpoint flow_steps. -1 keeps checkpoint value.")
parser.add_argument(
    "--action_squash_scale",
    type=float,
    default=-1.0,
    help="Override checkpoint action squash scale. Negative keeps checkpoint value; old checkpoints without this field use a near-identity scale.",
)
parser.add_argument(
    "--eval_initial_noise",
    choices=("random", "zero"),
    default="",
    help="Initial flow latent for playback. Empty keeps checkpoint/default setting.",
)
parser.add_argument("--seed", type=int, default=0, help="Random seed.")
parser.add_argument("--motion_file", type=str, default="", help="Optional override for the motion npz path.")
parser.add_argument("--sim_dt", type=float, default=-1.0, help="Override checkpoint sim_dt. Negative keeps checkpoint value.")
parser.add_argument(
    "--action_scale_multiplier",
    type=float,
    default=-1.0,
    help="Override env residual action scale during playback. Negative keeps checkpoint value.",
)
parser.add_argument("--fix_root_link", action="store_true", default=False, help="Lock the robot base in place.")
parser.add_argument("--motion_start_phase", type=int, default=-1, help="Override first reference phase for resets. Negative keeps checkpoint/default.")
parser.add_argument("--motion_end_phase", type=int, default=-1, help="Override last reference phase for resets. Negative keeps checkpoint/default.")
parser.add_argument(
    "--startup_randomization",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Override startup randomization. Default keeps checkpoint setting.",
)
parser.add_argument(
    "--reset_noise",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Override reset pose/velocity noise. Default keeps checkpoint setting.",
)
parser.add_argument(
    "--interval_pushes",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Override interval push perturbations. Default keeps checkpoint setting.",
)
parser.add_argument(
    "--observation_noise",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Override actor observation noise. Default off for clean deterministic playback.",
)
parser.add_argument("--reset_on_done", action="store_true", default=False, help="Reset robot when a termination condition is hit.")
parser.add_argument("--loop_motion", action="store_true", default=False, help="Reset to start_phase when reference motion ends.")
parser.add_argument("--log_every", type=int, default=100, help="Print playback stats every N simulation steps.")
parser.add_argument("--real_time", action="store_true", default=False, help="Throttle playback to wall-clock time. GUI playback enables this automatically.")
parser.add_argument("--no_real_time", action="store_true", default=False, help="Disable automatic wall-clock throttling in GUI playback.")
parser.add_argument(
    "--render_every",
    type=int,
    default=1,
    help="Render every N control steps in GUI playback. 2 or 3 reduces Isaac GUI stutter on slower scenes.",
)
parser.add_argument(
    "--contact_debug_vis",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Show contact sensor debug visualization. Default off because it is expensive in GUI playback.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from env.config import DEFAULT_MOTION_FILE, MimicEnvConfig
from env.mimic import G1MimicEnv
from engine.mixgrpo.inference import deterministic_sde_ode_actions
from net.mixgrpo.flow_policy import FlowMatchingPolicy


def _load_checkpoint_payload(checkpoint_path: Path, device: torch.device) -> dict:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    load_device = device
    if device.type == "cuda" and not torch.cuda.is_available():
        load_device = torch.device("cpu")
    payload = torch.load(checkpoint_path, map_location=load_device)
    if "policy" not in payload or "config" not in payload:
        raise KeyError("Checkpoint must contain 'policy' and 'config' keys")
    return payload


def _make_reset_phases(num_envs: int, start_phase: int, device: torch.device) -> torch.Tensor:
    return torch.full((num_envs,), max(0, start_phase), dtype=torch.long, device=device)


def main() -> None:
    if not args_cli.checkpoint:
        parser.error("--checkpoint is required")
    checkpoint_path = Path(args_cli.checkpoint).expanduser().resolve()
    device = torch.device(args_cli.device)
    payload = _load_checkpoint_payload(checkpoint_path, device=device)
    train_cfg = payload["config"]

    sim_dt = train_cfg["sim_dt"] if args_cli.sim_dt < 0.0 else args_cli.sim_dt
    flow_steps = train_cfg["flow_steps"] if args_cli.flow_steps < 0 else args_cli.flow_steps
    motion_file = args_cli.motion_file if args_cli.motion_file else train_cfg.get("motion_file", str(DEFAULT_MOTION_FILE))
    motion_start_phase = (
        train_cfg.get("motion_start_phase", 0)
        if args_cli.motion_start_phase < 0
        else args_cli.motion_start_phase
    )
    motion_end_phase = (
        train_cfg.get("motion_end_phase", -1)
        if args_cli.motion_end_phase < 0
        else args_cli.motion_end_phase
    )
    if args_cli.action_squash_scale > 0.0:
        action_squash_scale = float(args_cli.action_squash_scale)
    else:
        action_squash_scale = float(train_cfg.get("action_squash_scale", 1.0e6))
    if args_cli.action_scale_multiplier >= 0.0:
        action_scale_multiplier = float(args_cli.action_scale_multiplier)
    else:
        action_scale_multiplier = float(train_cfg.get("action_scale_multiplier", 1.0))
    startup_randomization = (
        bool(train_cfg.get("startup_randomization", True))
        if args_cli.startup_randomization is None
        else bool(args_cli.startup_randomization)
    )
    reset_noise = (
        bool(train_cfg.get("reset_noise", True)) if args_cli.reset_noise is None else bool(args_cli.reset_noise)
    )
    interval_pushes = (
        bool(train_cfg.get("interval_pushes", True))
        if args_cli.interval_pushes is None
        else bool(args_cli.interval_pushes)
    )
    observation_noise = False if args_cli.observation_noise is None else bool(args_cli.observation_noise)

    torch.manual_seed(args_cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args_cli.seed)

    train_future_ref_steps = 0

    env = G1MimicEnv(
        MimicEnvConfig(
            device=args_cli.device,
            num_envs=args_cli.num_envs,
            sim_dt=sim_dt,
            render=not args_cli.headless,
            render_every=max(1, args_cli.render_every),
            contact_debug_vis=args_cli.contact_debug_vis,
            action_scale_multiplier=action_scale_multiplier,
            fix_root_link=args_cli.fix_root_link or train_cfg.get("fix_root_link", False),
            startup_randomization=startup_randomization,
            motion_start_phase=motion_start_phase,
            motion_end_phase=motion_end_phase,
            motion_file=motion_file,
            max_episode_steps=int(1.0e9 / sim_dt),
            reset_noise=reset_noise,
            interval_pushes=interval_pushes,
            observation_noise=observation_noise,
            future_ref_steps=train_future_ref_steps,
        )
    )

    policy = FlowMatchingPolicy(
        obs_dim=train_cfg.get("policy_obs_dim", 0) or env.observation_dim,
        action_dim=train_cfg["action_dim"],
        horizon=train_cfg["horizon"],
        hidden_dims=tuple(train_cfg.get("actor_hidden_dims", (512, 256, 128))),
        activation=train_cfg.get("activation", "elu"),
        action_squash_scale=action_squash_scale,
        basis_count=int(train_cfg.get("basis_count", 0)),
        chunk_stitch_frames=int(train_cfg.get("chunk_stitch_frames", 0)),
        chunk_stitch_mode=str(train_cfg.get("chunk_stitch_mode", "smoothstep")),
    ).to(env.device)
    policy.load_state_dict(payload["policy"])
    policy.eval()

    reset_start_phase = motion_start_phase if args_cli.start_phase < 0 else args_cli.start_phase
    reset_phases = _make_reset_phases(env.num_envs, reset_start_phase, env.device)
    current_obs = env.reset(phase_indices=reset_phases)
    cached_chunk: torch.Tensor | None = None
    chunk_index = train_cfg["horizon"]
    total_steps = 0

    use_real_time = (args_cli.real_time or not args_cli.headless) and not args_cli.no_real_time
    next_frame_time = time.perf_counter()

    print("[INFO] Playing trained MixGRPO checkpoint", flush=True)
    print(f"[INFO] checkpoint={checkpoint_path}", flush=True)
    print(f"[INFO] motion_file={motion_file}", flush=True)
    print(
        f"[INFO] horizon={train_cfg['horizon']} action_dim={train_cfg['action_dim']} "
        f"flow_steps={flow_steps} action_squash_scale={action_squash_scale} "
        f"chunk_stitch_frames={policy.chunk_stitch_frames} "
        f"chunk_stitch_mode={policy.chunk_stitch_mode}",
        flush=True,
    )
    print(
        f"[INFO] startup_randomization={startup_randomization} "
        f"reset_noise={reset_noise} interval_pushes={interval_pushes} "
        f"observation_noise={observation_noise} render_every={max(1, args_cli.render_every)} "
        f"contact_debug_vis={args_cli.contact_debug_vis} "
        f"action_scale_multiplier={action_scale_multiplier}",
        flush=True,
    )
    eval_initial_noise = args_cli.eval_initial_noise or train_cfg.get("eval_initial_noise", "random")
    print(f"[INFO] eval_initial_noise={eval_initial_noise}", flush=True)

    while simulation_app.is_running():
        if cached_chunk is None or chunk_index >= train_cfg["horizon"]:
            with torch.inference_mode():
                initial_noise = None
                if eval_initial_noise == "random":
                    initial_noise = torch.randn(
                        current_obs.shape[0],
                        policy.chunk_dim,
                        device=current_obs.device,
                        dtype=current_obs.dtype,
                    )
                cached_chunk = deterministic_sde_ode_actions(
                    policy,
                    current_obs,
                    steps=flow_steps,
                    sde_eta=train_cfg.get("sde_eta", 0.7),
                    initial_noise=initial_noise,
                )
            chunk_index = 0

        action = cached_chunk[:, chunk_index, :]
        chunk_index += 1
        current_obs, reward, done, info = env.step(action, auto_reset=False)

        total_steps += 1
        if use_real_time:
            next_frame_time += float(env.dt)
            sleep_s = next_frame_time - time.perf_counter()
            if sleep_s > 0.0:
                time.sleep(sleep_s)
            else:
                next_frame_time = time.perf_counter()
        if args_cli.log_every > 0 and total_steps % args_cli.log_every == 0:
            print(
                f"[PLAY] step={total_steps} "
                f"phase={int(env.phase_steps[0].item())} "
                f"action_abs={float(action.abs().mean().item()):.5f} "
                f"reward={float(reward.mean().item()):.5f} "
                f"done={float(done.float().mean().item()):.5f} "
                f"height={float(info['debug_terms']['robot_anchor_height'].mean().item()):.5f} "
                f"tilt={float(info['debug_terms']['robot_anchor_tilt'].mean().item()):.5f} "
                f"ee_z_max={float(info['debug_terms']['ee_z_error_max'].mean().item()):.5f} "
                f"term_z_max={float(info['debug_terms']['termination_z_error_max'].mean().item()):.5f} "
                f"anchor_z={float(info['debug_terms']['anchor_z_error'].mean().item()):.5f} "
                f"bad_contact={float(info['reward_terms']['undesired_contacts'].mean().item()):.5f} "
                f"anchor_pos_bad={float(info['done_terms']['anchor_pos_bad'].float().mean().item()):.5f} "
                f"anchor_ori_bad={float(info['done_terms']['anchor_ori_bad'].float().mean().item()):.5f} "
                f"ee_bad={float(info['done_terms']['ee_body_bad'].float().mean().item()):.5f}",
                flush=True,
            )

        if args_cli.max_steps > 0 and total_steps >= args_cli.max_steps:
            break

        need_reset = False
        reset_reason = ""
        if args_cli.reset_on_done and bool(done.any()):
            need_reset = True
            reset_reason = "termination"
        if args_cli.loop_motion and bool(torch.any(env.phase_steps >= env.motion.num_frames - 1)):
            need_reset = True
            reset_reason = "motion_end"

        if need_reset:
            print(
                f"[INFO] Resetting at step {total_steps} due to {reset_reason}. "
                f"phase={int(env.phase_steps[0].item())}",
                flush=True,
            )
            current_obs = env.reset(phase_indices=reset_phases)
            cached_chunk = None
            chunk_index = train_cfg["horizon"]
    print(f"[INFO] Playback finished after {total_steps} simulation steps.", flush=True)
if __name__ == "__main__":
    main()
    import os
    os._exit(0)
