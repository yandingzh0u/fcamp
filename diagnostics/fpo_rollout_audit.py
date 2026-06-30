"""Audit FPO credit assignment around a known failure phase.

This is a read-only diagnostic: it loads a checkpoint, resets every environment to one
requested motion phase, collects a stochastic FPO rollout, and reports whether terminal
failures receive negative normalized advantages far enough before death.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument("--start_phase", type=int, default=800)
parser.add_argument("--rollout_steps", type=int, default=64)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

from algorithms import make_algorithm
from core.config import AlgoCfg, Config, EnvCfg, TrainCfg
from core.env_factory_adapter import build_env


def rebuild_config(payload: dict) -> Config:
    raw = payload["config"]
    return Config(
        algo_name=raw["algo_name"],
        env=EnvCfg(**raw["env"]),
        algo=AlgoCfg(**{**raw["algo"], "actor_hidden_dims": tuple(raw["algo"]["actor_hidden_dims"])}),
        train=TrainCfg(**raw["train"]),
    )


def indexed_mean(x: torch.Tensor, step: torch.Tensor, mask: torch.Tensor) -> tuple[float, float, int]:
    valid = mask & (step >= 0) & (step < x.shape[0])
    if not bool(valid.any()):
        return float("nan"), float("nan"), 0
    env_ids = valid.nonzero(as_tuple=False).squeeze(-1)
    values = x[step[env_ids], env_ids, 0]
    return float(values.mean().item()), float((values < 0).float().mean().item()), int(values.numel())


def main() -> None:
    checkpoint = Path(args_cli.checkpoint).expanduser().resolve()
    payload = torch.load(checkpoint, map_location=args_cli.device, weights_only=False)
    cfg = rebuild_config(payload)
    cfg.env.device = args_cli.device
    cfg.env.num_envs = int(args_cli.num_envs)
    cfg.env.startup_randomization = False
    cfg.env.reset_noise = False
    cfg.env.interval_pushes = False
    cfg.env.observation_noise = False
    cfg.env.adaptive_motion_sampling = False
    cfg.env.max_episode_steps = int(1.0e9)

    env = build_env(cfg)
    algo = make_algorithm(cfg.algo_name)(cfg.algo, env, simulation_app)
    algo.build()
    algo.policy.load_state_dict(payload["policy"])
    algo.load_extra_checkpoint_state(payload.get("algo_state", {}), reset_optimizer=True)
    algo.num_steps_per_env = int(args_cli.rollout_steps)

    # Keep stochastic action sampling, but freeze the checkpoint's observation statistics.
    algo.actor.train()
    algo.critic.eval()
    algo.actor_obs_normalizer.eval()
    algo.critic_obs_normalizer.eval()
    env.record_motion_failures = False

    phase = torch.full(
        (env.num_envs,), int(args_cli.start_phase), dtype=torch.long, device=env.device
    )
    obs = env.reset(phase_indices=phase)
    algo._obs = obs
    algo._critic_obs = env.get_critic_observation()
    rollout = algo.collect(obs)

    first_done = rollout["first_done_step"]
    timeout = rollout["first_done_timeout"]
    failed = (first_done < algo.num_steps_per_env) & (~timeout)
    print(
        f"[AUDIT] checkpoint={checkpoint} start_phase={args_cli.start_phase} "
        f"steps={algo.num_steps_per_env} envs={env.num_envs} "
        f"failure_frac={float(failed.float().mean().item()):.6f} "
        f"death_step_mean={float(first_done[failed].float().mean().item()) if bool(failed.any()) else float('nan'):.3f}"
    )

    advantages = rollout["advantages"]
    rewards = rollout["rewards"]
    values = rollout["values"]
    returns = rollout["returns"]
    for lookback in (0, 1, 5, 10, 20, 30, 40):
        step = first_done - lookback
        adv_mean, adv_neg, count = indexed_mean(advantages, step, failed)
        rew_mean, _, _ = indexed_mean(rewards, step, failed)
        value_mean, _, _ = indexed_mean(values, step, failed)
        return_mean, _, _ = indexed_mean(returns, step, failed)
        print(
            f"[CREDIT] lookback={lookback:02d} count={count} advantage={adv_mean:.6f} "
            f"negative_frac={adv_neg:.6f} reward={rew_mean:.6f} "
            f"value={value_mean:.6f} return={return_mean:.6f}"
        )

    old_cfm = rollout["old_cfm"]
    sampled = rollout["actions"].abs()
    first_actions = rollout["actions"][0]
    first_conditional_std = first_actions.std(dim=0, unbiased=False)
    with torch.no_grad():
        first_greedy = algo.actor.act_inference(rollout["actor_obs"][0], eval_mode="zero")
    first_bias = (first_actions.mean(dim=0) - first_greedy.mean(dim=0)).abs()
    print(
        f"[CFM] mean={float(old_cfm.mean().item()):.6f} "
        f"clamp3_frac={float((old_cfm >= 3.0).float().mean().item()):.6f} "
        f"action_abs_mean={float(sampled.mean().item()):.6f} "
        f"action_abs_p99={float(torch.quantile(sampled, 0.99).item()):.6f} "
        f"action_abs_max={float(sampled.max().item()):.6f}"
    )
    print(
        f"[EXPLORATION] conditional_std_mean={float(first_conditional_std.mean().item()):.6f} "
        f"conditional_std_min={float(first_conditional_std.min().item()):.6f} "
        f"conditional_std_max={float(first_conditional_std.max().item()):.6f} "
        f"sample_mean_vs_greedy_mae={float(first_bias.mean().item()):.6f}"
    )
    simulation_app.close()
    # Isaac Sim may keep telemetry/background threads alive after close in headless diagnostics.
    # All output above is flushed line-by-line by the terminal; force a clean process exit.
    os._exit(0)


if __name__ == "__main__":
    main()
