"""Play / visualize a trained checkpoint in IsaacLab (any algorithm).

    python play.py --checkpoint runs/<run>/checkpoints/last.pt [--num_envs 4] [--start_phase 0] \
        [--loop_motion] [--reset_on_done]

Reuses the unified config + algorithm system: the checkpoint stores asdict(Config), so we
rebuild EnvCfg/AlgoCfg/TrainCfg, construct the env + the same Algorithm used for training, load
the policy weights, and roll the algorithm's greedy action chunk. Works for MixGRPO today and
any future algorithm (PPO/FPO) with no playback code changes.
"""
from __future__ import annotations

import argparse
import sys
import time
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
parser.add_argument("--motion_file", type=str, default="", help="Optional motion npz override.")
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

from core.config import Config, EnvCfg, AlgoCfg, TrainCfg
from core.env_factory_adapter import build_env
from algorithms import make_algorithm


def _rebuild_config(payload: dict) -> Config:
    """Rebuild the typed Config from the checkpoint's asdict(Config)."""
    raw = payload.get("config", {})
    if "env" in raw and "algo" in raw:  # new nested format
        env = EnvCfg(**raw["env"])
        algo = AlgoCfg(**{**raw["algo"], "actor_hidden_dims": tuple(raw["algo"]["actor_hidden_dims"])})
        train = TrainCfg(**raw["train"])
        return Config(algo_name=raw.get("algo_name", "mixgrpo"), env=env, algo=algo, train=train)
    raise KeyError("Checkpoint config is not in the expected nested {env, algo, train} format.")


def main() -> None:
    checkpoint_path = Path(args_cli.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    device = torch.device(args_cli.device)
    load_device = device if (device.type != "cuda" or torch.cuda.is_available()) else torch.device("cpu")
    payload = torch.load(checkpoint_path, map_location=load_device)
    if "policy" not in payload:
        raise KeyError("Checkpoint must contain a 'policy' state dict.")
    cfg = _rebuild_config(payload)

    # Playback overrides on the env section.
    cfg.env.device = args_cli.device
    cfg.env.num_envs = args_cli.num_envs
    cfg.env.render = not args_cli.headless
    cfg.env.render_every = max(1, args_cli.render_every)
    cfg.env.fix_root_link = args_cli.fix_root_link or cfg.env.fix_root_link
    cfg.env.observation_noise = bool(args_cli.observation_noise)
    # Play the whole clip: lift the episode cap.
    cfg.env.max_episode_steps = int(1.0e9)
    if args_cli.motion_file:
        cfg.env.motion_file = args_cli.motion_file
    if args_cli.start_phase >= 0:
        cfg.env.motion_start_phase = args_cli.start_phase
    if args_cli.interval_pushes is not None:
        cfg.env.interval_pushes = bool(args_cli.interval_pushes)
    if args_cli.reset_noise is not None:
        cfg.env.reset_noise = bool(args_cli.reset_noise)
    if args_cli.startup_randomization is not None:
        cfg.env.startup_randomization = bool(args_cli.startup_randomization)
    # Playback does not use GRPO groups; force single-branch so any num_envs is valid and the
    # observation-noise group sharing is disabled.
    cfg.algo.num_generations = 1
    cfg.env.num_generations = 1

    torch.manual_seed(args_cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args_cli.seed)

    env = build_env(cfg)
    algo = make_algorithm(cfg.algo_name)(cfg.algo, env, simulation_app)
    algo.build()
    algo.policy.load_state_dict(payload["policy"])
    algo.policy.eval()

    horizon = int(cfg.algo.horizon)
    reset_start_phase = cfg.env.motion_start_phase if args_cli.start_phase < 0 else args_cli.start_phase
    reset_phases = torch.full((env.num_envs,), max(0, reset_start_phase), dtype=torch.long, device=env.device)
    current_obs = env.reset(phase_indices=reset_phases)
    cached_chunk: torch.Tensor | None = None
    chunk_index = horizon
    total_steps = 0

    use_real_time = (args_cli.real_time or not args_cli.headless) and not args_cli.no_real_time
    next_frame_time = time.perf_counter()

    print("[INFO] Playing trained checkpoint", flush=True)
    print(f"[INFO] checkpoint={checkpoint_path}", flush=True)
    print(f"[INFO] algo={cfg.algo_name} motion_file={env.task_cfg.motion_file}", flush=True)
    print(
        f"[INFO] horizon={horizon} action_dim={cfg.algo.action_dim} flow_steps={cfg.algo.flow_steps} "
        f"observation_noise={cfg.env.observation_noise} interval_pushes={cfg.env.interval_pushes} "
        f"reset_noise={cfg.env.reset_noise}",
        flush=True,
    )

    while simulation_app.is_running():
        if cached_chunk is None or chunk_index >= horizon:
            with torch.inference_mode():
                cached_chunk = algo.deterministic_actions(current_obs)
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
                f"[PLAY] step={total_steps} phase={int(env.phase_steps[0].item())} "
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
            print(f"[INFO] Reset at step {total_steps} ({reason}). phase={int(env.phase_steps[0].item())}", flush=True)
            current_obs = env.reset(phase_indices=reset_phases)
            cached_chunk = None
            chunk_index = horizon

    print(f"[INFO] Playback finished after {total_steps} steps.", flush=True)


if __name__ == "__main__":
    main()
    import os
    os._exit(0)
