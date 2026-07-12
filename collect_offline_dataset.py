from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isaaclab.app import AppLauncher
from core.config import load_config

parser = argparse.ArgumentParser(description="Collect FQL transition data from a trained teacher.")
parser.add_argument("--config", required=True, help="Teacher algorithm config.")
parser.add_argument("--checkpoint", required=True, help="Teacher checkpoint.")
parser.add_argument("--output", required=True, help="Output .pt transition dataset.")
parser.add_argument("--num_frames", type=int, default=24, help="Physical vector-env steps to collect.")
parser.add_argument(
    "--environment_action_scale",
    "--action_limit",
    dest="environment_action_scale",
    type=float,
    default=1.0,
    help=(
        "Map normalized FQL actions u in [-1,1] to environment actions as "
        "a_env=scale*u; --action_limit is retained as a compatibility alias."
    ),
)
parser.add_argument(
    "--set",
    dest="overrides",
    action="append",
    default=[],
    help="Override a teacher config leaf, e.g. environment.num_envs=512.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

cfg = load_config(args_cli.config, args_cli.overrides)
args_cli.headless = True
args_cli.device = cfg.environment.device
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def main() -> None:
    import torch

    from algorithms import make_algorithm
    from core.offline_dataset import save_offline_transition_dataset
    from env.mimic import G1MimicEnv

    if args_cli.num_frames < 1:
        raise ValueError("--num_frames must be positive")
    if args_cli.environment_action_scale <= 0.0:
        raise ValueError("--environment_action_scale must be positive")

    checkpoint_path = Path(args_cli.checkpoint).expanduser().resolve()
    payload = torch.load(checkpoint_path, map_location=cfg.environment.device, weights_only=False)
    checkpoint_algorithm = payload.get("config", {}).get("algorithm")
    if checkpoint_algorithm != cfg.algorithm:
        raise ValueError(
            f"Checkpoint algorithm {checkpoint_algorithm!r} does not match config {cfg.algorithm!r}"
        )

    env = G1MimicEnv(cfg.environment, cfg.observation_group_size)
    algorithm = make_algorithm(cfg.algorithm)(cfg.parameters, env, simulation_app)
    algorithm.build()
    observation = algorithm.initial_reset()
    algorithm.policy.load_state_dict(payload["policy"], strict=True)
    algorithm.load_extra_checkpoint_state(payload.get("algo_state", {}), reset_optimizer=True)
    algorithm.policy.eval()

    observations: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []
    next_observations: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    clipped_elements = 0
    action_elements = 0
    done_count = 0
    timeout_count = 0

    print(
        f"[DATASET] teacher={cfg.algorithm} checkpoint={checkpoint_path} "
        f"num_envs={env.num_envs} frames={args_cli.num_frames} "
        f"teacher_horizon={algorithm.horizon} "
        f"environment_action_scale={args_cli.environment_action_scale}",
        flush=True,
    )
    with torch.no_grad():
        for frame in range(args_cli.num_frames):
            action_chunk = algorithm.deterministic_actions(observation)
            if action_chunk.ndim != 3 or action_chunk.shape[0] != env.num_envs:
                raise ValueError(f"Unexpected teacher action shape {tuple(action_chunk.shape)}")
            proposed_action = action_chunk[:, 0, :]
            clipped_elements += int(
                (proposed_action.abs() > args_cli.environment_action_scale).sum().item()
            )
            action_elements += proposed_action.numel()
            normalized_action = (
                proposed_action / args_cli.environment_action_scale
            ).clamp(-1.0, 1.0)
            action = normalized_action * args_cli.environment_action_scale

            next_observation, reward, done, info = env.step(
                action,
                auto_reset=True,
                reset_horizon=1,
            )
            timeout = info["done_terms"]["time_out"].bool()
            replay_next_observation = next_observation.clone()
            if bool(timeout.any()) and "final_observation" in info:
                replay_next_observation[timeout] = info["final_observation"][timeout]
            bootstrap_mask = ((~done.bool()) | timeout).float().unsqueeze(-1)

            observations.append(observation.cpu())
            actions.append(normalized_action.cpu())
            rewards.append(reward.unsqueeze(-1).cpu())
            next_observations.append(replay_next_observation.cpu())
            masks.append(bootstrap_mask.cpu())
            done_count += int(done.sum().item())
            timeout_count += int(timeout.sum().item())
            observation = next_observation
            print(
                f"[DATASET_PROGRESS] frame={frame + 1}/{args_cli.num_frames} "
                f"reward={reward.mean().item():.5f} done_frac={done.float().mean().item():.5f}",
                flush=True,
            )

    transition_count = args_cli.num_frames * env.num_envs
    clip_fraction = clipped_elements / max(action_elements, 1)
    output_path = save_offline_transition_dataset(
        args_cli.output,
        observations=torch.cat(observations, dim=0),
        actions=torch.cat(actions, dim=0),
        rewards=torch.cat(rewards, dim=0),
        next_observations=torch.cat(next_observations, dim=0),
        masks=torch.cat(masks, dim=0),
        metadata={
            "teacher_algorithm": cfg.algorithm,
            "teacher_checkpoint": str(checkpoint_path),
            "teacher_checkpoint_update": int(payload.get("update_idx", -1)),
            "teacher_horizon": int(algorithm.horizon),
            "collection_mode": "deterministic_receding_horizon_first_action",
            "task": env.task.name,
            "num_envs": int(env.num_envs),
            "physical_frames": int(args_cli.num_frames),
            "action_coordinate": "normalized_fql_action",
            "policy_action_limit": 1.0,
            "environment_action_scale": float(args_cli.environment_action_scale),
            "action_clip_element_fraction": float(clip_fraction),
            "done_count": int(done_count),
            "timeout_count": int(timeout_count),
        },
    )
    print(
        f"[DATASET_DONE] path={output_path} transitions={transition_count} "
        f"action_clip_element_fraction={clip_fraction:.6f} done={done_count} "
        f"timeouts={timeout_count}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
