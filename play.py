from __future__ import annotations

import argparse
import sys
import time
from dataclasses import fields, replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a trained checkpoint in IsaacLab.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to a saved checkpoint (.pt).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to play.")
parser.add_argument("--max_steps", type=int, default=0, help="Hard stop. 0 runs until the app closes.")
parser.add_argument("--start_phase", type=int, default=-1, help="Reset motion phase. Negative uses motion_start_phase.")
parser.add_argument("--task", choices=("largebox_plane", "crawl_slope"), default="")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--reset_on_done", action="store_true", default=False, help="Reset when a termination fires.")
parser.add_argument("--loop_motion", action="store_true", default=False, help="Reset to start_phase when the motion ends.")
parser.add_argument("--log_every", type=int, default=100, help="Print playback stats every N steps.")
parser.add_argument("--render_every", type=int, default=1, help="Render every N control steps in GUI.")
parser.add_argument("--real_time", action="store_true", default=False, help="Throttle to wall-clock. GUI enables this automatically.")
parser.add_argument("--no_real_time", action="store_true", default=False, help="Disable automatic wall-clock throttle in GUI.")
parser.add_argument("--fix_root_link", action="store_true", default=False, help="Lock the robot base in place.")
parser.add_argument("--observation_noise", action=argparse.BooleanOptionalAction, default=False, help="Actor observation noise. Default off for clean deterministic playback.")
parser.add_argument("--interval_pushes", action=argparse.BooleanOptionalAction, default=None, help="Override interval pushes. Default keeps checkpoint setting.")
parser.add_argument("--reset_noise", action=argparse.BooleanOptionalAction, default=None, help="Override reset pose/velocity noise. Default keeps checkpoint setting.")
parser.add_argument("--startup_randomization", action=argparse.BooleanOptionalAction, default=None, help="Override startup randomization. Default keeps checkpoint setting.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

from engine.config import (
    EnvironmentConfig,
    ExperimentConfig,
    TrainingConfig,
    config_from_dict,
    METHOD_CONFIGS,
)
from envs.g1_mimic import G1MimicEnv
from method import load_method_class


def _deployment_action_chunk(algo, obs: torch.Tensor) -> torch.Tensor:
    payload = algo.deployment_actions(obs)
    if payload.dim() == 2:
        return payload.unsqueeze(1)
    if payload.dim() == 3:
        return payload
    raise ValueError(f"deployment_actions must return [N,D] or [N,H,D], got shape={tuple(payload.shape)}")


def _split_reference_action(algo, env, payload: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
    if not bool(getattr(algo, "uses_reference_dt", False)):
        return payload, None
    expected_dim = int(env.action_dim) + 1
    if payload.shape[-1] != expected_dim:
        raise ValueError(
            f"{algo.__class__.__name__} uses reference_dt but returned action dim "
            f"{payload.shape[-1]}, expected {expected_dim}"
        )
    return payload[..., : env.action_dim], payload[..., -1]


def _select(cls, values: dict) -> dict:
    return {field.name: values[field.name] for field in fields(cls) if field.name in values}


def _rebuild_config(payload: dict) -> ExperimentConfig:
    raw = payload.get("config", {})
    if "method" in raw or "algorithm" in raw:
        return config_from_dict(raw)
    if not {"algo_name", "env", "algo", "train"}.issubset(raw):
        raise KeyError("Checkpoint has no supported configuration schema")
    method = raw["algo_name"]
    if method != "fcamp":
        raise ValueError(f"Unsupported legacy checkpoint method {method!r}; available: {sorted(METHOD_CONFIGS)}")
    legacy_env = raw["env"]
    environment = _select(EnvironmentConfig, legacy_env)
    environment["task"] = "largebox_plane" if legacy_env.get("terrain_type") == "plane" else "crawl_slope"
    environment["decimation"] = 4
    parameters = dict(raw["algo"])
    tree = {
        "method": method,
        "environment": environment,
        "parameters": parameters,
        "training": _select(TrainingConfig, raw["train"]),
    }
    return config_from_dict(tree)


def main() -> None:
    checkpoint_path = Path(args_cli.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    device = torch.device(args_cli.device)
    load_device = device if (device.type != "cuda" or torch.cuda.is_available()) else torch.device("cpu")
    payload = torch.load(checkpoint_path, map_location=load_device, weights_only=False)
    if "policy" not in payload:
        raise KeyError("Checkpoint must contain a 'policy' state dict.")
    cfg = _rebuild_config(payload)

    environment = replace(
        cfg.environment,
        task=args_cli.task or cfg.environment.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        fix_root_link=args_cli.fix_root_link or cfg.environment.fix_root_link,
        observation_noise=bool(args_cli.observation_noise),
        max_episode_steps=int(1.0e9),
        motion_start_phase=(args_cli.start_phase if args_cli.start_phase >= 0 else cfg.environment.motion_start_phase),
        interval_pushes=(cfg.environment.interval_pushes if args_cli.interval_pushes is None else args_cli.interval_pushes),
        reset_noise=(cfg.environment.reset_noise if args_cli.reset_noise is None else args_cli.reset_noise),
        startup_randomization=(
            cfg.environment.startup_randomization
            if args_cli.startup_randomization is None
            else args_cli.startup_randomization
        ),
    )
    torch.manual_seed(args_cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args_cli.seed)

    env = G1MimicEnv(
        environment,
        1,
        render=not args_cli.headless,
        render_every=args_cli.render_every,
    )
    algo = load_method_class(cfg.method)(cfg.parameters, env, simulation_app)
    algo.build()
    algo.policy.load_state_dict(payload["policy"])
    algo.policy.eval()

    horizon = algo.horizon
    reset_start_phase = environment.motion_start_phase
    reset_phases = torch.full((env.num_envs,), max(0, reset_start_phase), dtype=torch.long, device=env.device)
    current_obs = algo.evaluation_reset(reset_phases)
    cached_chunk: torch.Tensor | None = None
    chunk_index = horizon
    total_steps = 0

    use_real_time = (args_cli.real_time or not args_cli.headless) and not args_cli.no_real_time
    next_frame_time = time.perf_counter()

    print("[INFO] Playing trained checkpoint", flush=True)
    print(f"[INFO] checkpoint={checkpoint_path}", flush=True)
    print(
        f"[INFO] method={cfg.method} task={env.task.name} terrain={env.task.terrain} "
        f"motion_file={env.task.motion_file}",
        flush=True,
    )
    print(
        f"[INFO] horizon={horizon} action_dim={env.action_dim} "
        f"observation_noise={environment.observation_noise} interval_pushes={environment.interval_pushes} "
        f"reset_noise={environment.reset_noise}",
        flush=True,
    )

    while simulation_app.is_running():
        if cached_chunk is None or chunk_index >= cached_chunk.shape[1]:
            with torch.inference_mode():
                cached_chunk = _deployment_action_chunk(algo, current_obs)
            chunk_index = 0
        action_payload = cached_chunk[:, chunk_index, :]
        action, reference_dt = _split_reference_action(algo, env, action_payload)
        chunk_index += 1
        current_obs, reward, done, info = algo.evaluation_step(action, reference_dt)
        total_steps += 1

        if use_real_time:
            next_frame_time += float(env.dt)
            sleep_s = next_frame_time - time.perf_counter()
            if sleep_s > 0.0:
                time.sleep(sleep_s)
            else:
                next_frame_time = time.perf_counter()

        if args_cli.log_every > 0 and total_steps % args_cli.log_every == 0:
            frame_delta = info.get("reference_frame_delta")
            frame_delta_mean = float(frame_delta.mean().item()) if torch.is_tensor(frame_delta) else 1.0
            reference_dt_mean = float(reference_dt.mean().item()) if torch.is_tensor(reference_dt) else float(env.dt)
            print(
                f"[PLAY] step={total_steps} phase={float(env.phase_steps[0].item()):.2f} "
                f"reference_dt={reference_dt_mean:.5f} "
                f"frame_delta={frame_delta_mean:.3f} "
                f"action_abs={float(action.abs().mean().item()):.5f} "
                f"reward={float(reward.mean().item()):.5f} "
                f"done={float(done.float().mean().item()):.5f} "
                f"height={float(info['debug_terms']['robot_anchor_height'].mean().item()):.5f} "
                f"tilt={float(info['debug_terms']['robot_anchor_tilt'].mean().item()):.5f} "
                f"ee_z_max={float(info['debug_terms']['ee_z_error_max'].mean().item()):.5f} "
                f"anchor_z={float(info['debug_terms']['anchor_z_error'].mean().item()):.5f} "
                f"anchor_pos_bad={float(info['done_terms']['anchor_pos_bad'].float().mean().item()):.5f} "
                f"anchor_ori_bad={float(info['done_terms']['anchor_ori_bad'].float().mean().item()):.5f} "
                f"ee_bad={float(info['done_terms']['ee_body_bad'].float().mean().item()):.5f}",
                flush=True,
            )

        if args_cli.max_steps > 0 and total_steps >= args_cli.max_steps:
            break

        need_reset = False
        reason = ""
        if args_cli.reset_on_done and bool(done.any()):
            need_reset, reason = True, "termination"
        if args_cli.loop_motion and bool(torch.any(env.phase_steps >= env.motion.num_frames - 1)):
            need_reset, reason = True, "motion_end"
        if need_reset:
            print(f"[INFO] Reset at step {total_steps} ({reason}). phase={float(env.phase_steps[0].item()):.2f}", flush=True)
            current_obs = algo.evaluation_reset(reset_phases)
            cached_chunk = None
            chunk_index = horizon

    print(f"[INFO] Playback finished after {total_steps} steps.", flush=True)


if __name__ == "__main__":
    main()
    import os
    os._exit(0)
